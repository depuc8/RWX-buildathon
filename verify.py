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


class _Overlay(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serve patched source for the modules a patch changes."""

    def __init__(self, files):
        self.by_module = {}
        for path, text in files.items():
            name = path[:-3].replace("/", ".")
            is_pkg = name.endswith(".__init__")
            if is_pkg:
                name = name[: -len(".__init__")]
            self.by_module[name] = (os.path.join(SRC, path), text, is_pkg)

    def find_spec(self, fullname, path=None, target=None):
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


def start_tpcds(src, files, indices, baseline=None):
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
            sys.meta_path.insert(0, _Overlay(files))
            import tpcds_check

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
