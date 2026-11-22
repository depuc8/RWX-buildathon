"""Run candidate tests on the patched sqlglot, inside the scoring container.

The source tree on disk is read-only and unpatched. Each verification forks a
child from the selector process; the child serves the patched Python files
through an import hook, runs the tests with pytest and reports which failed.
Anything that goes wrong (a crash, a timeout) reports every test as failed,
so the caller only ever selects more.
"""

import concurrent.futures
import importlib.abc
import importlib.util
import json
import os
import pickle
import select
import signal
import sys
import time

SRC = None


def warm_up(src):
    """Import what every child needs once, in the parent, before forking."""
    global SRC
    SRC = str(src)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ["SKIP_INTEGRATION"] = "0"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    import _pytest.config  # noqa: F401
    import _pytest.main  # noqa: F401
    import _pytest.python  # noqa: F401
    import _pytest.unittest  # noqa: F401
    import pytest  # noqa: F401

    for heavy in ("numpy", "pandas", "duckdb"):
        try:
            importlib.import_module(heavy)
        except Exception:  # noqa: BLE001
            pass


def _module_name(path):
    name = path[:-3].replace("/", ".")
    return name[: -len(".__init__")] if name.endswith(".__init__") else name


class _Overlay(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serve patched source for the modules a patch changes, and hide the
    ones it deletes."""

    def __init__(self, files, deleted=()):
        self.by_module = {}
        for path, text in files.items():
            if text is None or not path.endswith(".py"):
                continue
            self.by_module[_module_name(path)] = (os.path.join(SRC, path), text, path.endswith("/__init__.py"))
        self.deleted = {_module_name(p) for p in deleted if p.endswith(".py")} - set(self.by_module)

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.deleted:
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        entry = self.by_module.get(fullname)
        if entry is None:
            return None
        origin, _, is_pkg = entry
        spec = importlib.util.spec_from_loader(fullname, self, origin=origin, is_package=is_pkg)
        spec.has_location = True
        if is_pkg:
            spec.submodule_search_locations = [os.path.dirname(origin)]
        return spec

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        origin, text, _ = self.by_module[module.__name__]
        module.__file__ = origin
        exec(compile(text, origin, "exec", dont_inherit=True), module.__dict__)


def _cross(obj):
    return pickle.loads(pickle.dumps(obj))


class SyncPool:
    """ProcessPoolExecutor stand-in: same pickling, no extra processes."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        if initializer:
            initializer(*_cross(initargs))

    def submit(self, fn, /, *args, **kwargs):
        future = concurrent.futures.Future()
        try:
            args, kwargs = _cross((args, kwargs))
            future.set_result(_cross(fn(*args, **kwargs)))
        except BaseException as e:  # noqa: BLE001
            future.set_exception(e)
        return future

    def map(self, fn, *iterables, **_):
        return [self.submit(fn, *a).result() for a in zip(*iterables)]

    def shutdown(self, wait=True, **_):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Collector:
    def __init__(self):
        self.failed = set()
        self.errors = set()
        self.ran = set()

    def pytest_collection_modifyitems(self, session, config, items):
        for mod in list(sys.modules.values()):
            if getattr(mod, "ProcessPoolExecutor", None) is not None and (
                getattr(mod, "__name__", "").startswith("tests.")
            ):
                mod.ProcessPoolExecutor = SyncPool

    def pytest_runtest_logreport(self, report):
        self.ran.add(report.nodeid)
        if report.failed:
            self.failed.add(report.nodeid)

    def pytest_collectreport(self, report):
        if report.failed:
            self.errors.add(report.nodeid)


RUNNER = os.environ.get("BR_RUNNER", "pytest")


def _run_unittest(nodeids):
    """Every sqlglot test is a unittest.TestCase method, so plain unittest
    gives pytest's outcome (subtest failures fail the test) for less start-up."""
    import unittest

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    failed = set()
    for nid in nodeids:
        path, cls, meth = nid.split("::")
        try:
            suite.addTest(loader.loadTestsFromName(f"{path[:-3].replace('/', '.')}.{cls}.{meth}"))
        except Exception:  # noqa: BLE001  (the test file no longer imports)
            failed.add(nid)
    for mod in list(sys.modules.values()):
        if getattr(mod, "ProcessPoolExecutor", None) is not None and (
            getattr(mod, "__name__", "").startswith("tests.")
        ):
            mod.ProcessPoolExecutor = SyncPool
    result = unittest.TestResult()
    suite.run(result)
    by_id = {}
    for nid in nodeids:
        path, cls, meth = nid.split("::")
        by_id[f"{path[:-3].replace('/', '.')}.{cls}.{meth}"] = nid
    for test, _ in result.failures + result.errors + [(t, None) for t in result.unexpectedSuccesses]:
        case = getattr(test, "test_case", test)  # a failing subtest fails its test
        nid = by_id.get(case.id()) if hasattr(case, "id") else None
        if nid is None:
            return set(nodeids)  # class or module setup failed: count them all
        failed.add(nid)
    return failed


def _child(files, nodeids, wfd):
    os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
    os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
    os.chdir(SRC)
    sys.path.insert(0, SRC)
    sys.meta_path.insert(0, _Overlay(files))
    if RUNNER == "unittest":
        os.write(wfd, json.dumps(sorted(_run_unittest(nodeids))).encode())
        os._exit(0)
    import pytest

    col = _Collector()
    pytest.main(
        ["-q", "-p", "no:cacheprovider", "--assert=plain", "-o", "addopts=", "--rootdir", SRC, *nodeids],
        plugins=[col],
    )
    failed = set(col.failed)
    for nid in nodeids:
        if any(nid == e or nid.startswith(e + "::") for e in col.errors):
            failed.add(nid)
        elif nid not in col.ran:
            failed.add(nid)  # never ran: treat as failing
    os.write(wfd, json.dumps(sorted(failed)).encode())
    os._exit(0)


class _Anything:
    """Whatever a stubbed package hands out at import time."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Anything()

    def __call__(self, *a, **k):
        return _Anything()


class _Stubs(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Stand-ins for third-party packages this image lacks (duckdb, pandas,
    numpy), so test modules that import them can still be imported."""

    def __init__(self, tops):
        self.tops = tops

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.tops:
            return importlib.util.spec_from_loader(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__path__ = []
        module.__getattr__ = lambda name: _Anything()


def _install_stubs():
    missing = {t for t in ("duckdb", "pandas", "numpy") if importlib.util.find_spec(t) is None}
    if missing:
        sys.meta_path.append(_Stubs(missing))


def _import_everything(test_files):
    """Import every sqlglot module, load every dialect and import every test
    module; return the names of those that fail."""
    import pkgutil

    failed = []
    try:
        import sqlglot
    except BaseException:  # noqa: BLE001
        return ["sqlglot"]
    for info in pkgutil.walk_packages(sqlglot.__path__, prefix="sqlglot.", onerror=failed.append):
        if info.name == "sqlglot.__main__":
            continue
        try:
            importlib.import_module(info.name)
        except BaseException:  # noqa: BLE001
            failed.append(info.name)
    try:
        from sqlglot.dialects.dialect import Dialect, Dialects

        for d in Dialects:
            try:
                Dialect.get_or_raise(d.value)
            except BaseException:  # noqa: BLE001
                failed.append(f"dialect:{d.value}")
    except BaseException:  # noqa: BLE001
        failed.append("sqlglot.dialects.dialect")
    for path in test_files:
        try:
            importlib.import_module(_module_name(path))
        except BaseException:  # noqa: BLE001
            failed.append(path)
    return sorted(set(failed))


def start_import_check(src, files, deleted, test_files):
    """Fork a child that imports everything on the patched code (see
    _import_everything). Returns a handle for poll_json."""
    global SRC
    SRC = str(src)
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
            os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
            os.chdir(SRC)
            sys.path.insert(0, SRC)
            sys.dont_write_bytecode = True
            os.environ["SKIP_INTEGRATION"] = "0"
            _install_stubs()
            sys.meta_path.insert(0, _Overlay(files, deleted))
            os.write(w, json.dumps(_import_everything(test_files)).encode())
        except BaseException:  # noqa: BLE001
            pass
        os._exit(0)
    os.close(w)
    return {"pid": pid, "fd": r, "buf": b"", "t0": time.monotonic()}


def start_tpcds(src, files, indices, baseline=None, deleted=(), attrs=None):
    """Fork a child that fingerprints these TPC-DS queries on the patched files
    (see tpcds_check). Returns a handle for poll_json."""
    global SRC
    SRC = str(src)
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
            os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
            os.chdir(SRC)
            sys.path.insert(0, SRC)
            sys.dont_write_bytecode = True
            sys.meta_path.insert(0, _Overlay(files, deleted))
            # In the suite run, earlier tests have imported every module by
            # the time this test runs: let patched ones have their import-time
            # effects here too.
            for path in sorted(files):
                if path.startswith("sqlglot/") and path.endswith(".py") and files[path] is not None:
                    name = path[:-3].replace("/", ".").removesuffix(".__init__")
                    try:
                        importlib.import_module(name)
                    except Exception:  # noqa: BLE001
                        pass
            import tpcds_check

            if attrs is not None:
                # Class attributes the data phase reads must be as they were.
                data_attrs, attr_baseline = attrs
                import sqlglot.executor.python  # noqa: F401  (loads what a worker loads)

                if tpcds_check.attr_fingerprints(data_attrs) != attr_baseline:
                    os.write(w, json.dumps({}).encode())  # nothing matches: differs
                    os._exit(0)
            fp = tpcds_check.fingerprints(indices, baseline)
            os.write(w, json.dumps({str(k): v for k, v in fp.items()}).encode())
        except BaseException:  # noqa: BLE001
            pass
        os._exit(0)
    os.close(w)
    return {"pid": pid, "fd": r, "buf": b"", "t0": time.monotonic()}


def poll_json(handle, timeout):
    """Non-blocking: ("done", object) once the child wrote its answer,
    ("failed", None) if it died, timed out or wrote nothing; None meanwhile."""
    fd = handle["fd"]
    while True:
        ready, _, _ = select.select([fd], [], [], 0)
        if not ready:
            break
        chunk = os.read(fd, 65536)
        if not chunk:
            os.close(fd)
            _, _, ru = os.wait4(handle["pid"], 0)
            handle["cpu"] = ru.ru_utime + ru.ru_stime
            try:
                return "done", json.loads(handle["buf"].decode())
            except ValueError:
                return "failed", None
        handle["buf"] += chunk
    if time.monotonic() - handle["t0"] > timeout:
        try:
            os.kill(handle["pid"], signal.SIGKILL)
        except ProcessLookupError:
            pass
        os.close(fd)
        _, _, ru = os.wait4(handle["pid"], 0)
        handle["cpu"] = ru.ru_utime + ru.ru_stime
        return "failed", None
    return None


def start(files, nodeids):
    """Fork a child that runs nodeids on the patched files. Returns a handle."""
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            _child(files, nodeids, w)
        except BaseException:  # noqa: BLE001
            pass
        os._exit(1)
    os.close(w)
    return {"pid": pid, "fd": r, "buf": b"", "nodeids": nodeids, "t0": time.monotonic()}


def poll(handle, timeout):
    """Non-blocking: the set of failing nodeids once done, else None."""
    fd = handle["fd"]
    while True:
        ready, _, _ = select.select([fd], [], [], 0)
        if not ready:
            break
        chunk = os.read(fd, 65536)
        if not chunk:
            os.close(fd)
            _, _, ru = os.wait4(handle["pid"], 0)
            handle["cpu"] = ru.ru_utime + ru.ru_stime
            try:
                return set(json.loads(handle["buf"].decode()))
            except ValueError:
                return set(handle["nodeids"])
        handle["buf"] += chunk
    if time.monotonic() - handle["t0"] > timeout:
        try:
            os.kill(handle["pid"], signal.SIGKILL)
        except ProcessLookupError:
            pass
        os.close(fd)
        _, _, ru = os.wait4(handle["pid"], 0)
        handle["cpu"] = ru.ru_utime + ru.ru_stime
        return set(handle["nodeids"])
    return None
