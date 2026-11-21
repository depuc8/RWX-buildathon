"""pytest plugin that records, for every test, what it depends on.

Run inside the labeling image from the sqlglot checkout:

    BRMAP_OUT=/out/map BRMAP_TESTS=/kit/tests.json BRMAP_JOBS=12 \
        python -m pytest -p brmap -q -p no:cacheprovider

The parent process collects the suite and imports every sqlglot module while
recording which lines run at import time (and which module was importing).
Then it forks one child per test, so caches filled by one test never hide a
dependency from another. Each child records:

  lines      every (file, line) it executed in sqlglot/ or tests/
  classes    every sqlglot/tests class (and its bases) a method ran on
  opens      files it opened, dirs it listed
  subprocess whether it spawned a process (its work is invisible to us)
"""

import concurrent.futures
import enum
import importlib
import json
import os
import pickle
import pkgutil
import signal
import sys
import time
from collections import defaultdict

ROOT = os.getcwd().rstrip("/") + "/"
OUT = os.environ.get("BRMAP_OUT", "/out/map")
JOBS = int(os.environ.get("BRMAP_JOBS", "8"))
TIMEOUT = int(os.environ.get("BRMAP_TIMEOUT", "1800"))
ONLY = os.environ.get("BRMAP_ONLY")  # comma-separated test numbers, for debugging
# BRMAP_CALIB=1: no tracing; record only the CPU each test costs when run cold
# in a fresh process with in-process pools, as the selector's verifier runs it.
CALIB = os.environ.get("BRMAP_CALIB") == "1"

M = sys.monitoring
TOOL = 2
DISABLE = M.DISABLE

_rel_cache = {}


def rel(path):
    """Repo-relative path for files we care about, else None."""
    r = _rel_cache.get(path, 0)
    if r == 0:
        r = None
        if isinstance(path, str) and path.startswith(ROOT):
            p = path[len(ROOT):]
            if p.startswith("sqlglot/") or p.startswith("tests/"):
                r = p
        _rel_cache[path] = r
    return r


def any_rel(path):
    """Repo-relative path for any file under the checkout, else None."""
    if isinstance(path, bytes):
        path = os.fsdecode(path)
    if not isinstance(path, str):
        return None
    path = os.path.normpath(os.path.join(os.getcwd(), path))
    if path.startswith(ROOT):
        return path[len(ROOT):]
    return None


def _cross(obj):
    """What arrives on the other side of a process boundary: a pickle round
    trip, so code run by (un)pickling, like sqlglot's serde, is traced too."""
    return pickle.loads(pickle.dumps(obj))


class SyncPool:
    """Stands in for ProcessPoolExecutor so pool work is traced in-process."""

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
        self.shutdown()
        return False


# ---------------------------------------------------------------- parent phase

import_lines = defaultdict(set)  # (file, line) -> importing module files
import_receivers = set()  # (class, importing module file): a method ran on it while importing
import_reads = set()  # (class, attr, key token or None, importing module): read while importing
IMPORTING = True  # parent phase: attribute reads are recorded per importing module
collect_opens = defaultdict(set)  # opened path -> test files collecting it
_parent_active = True


def _importing_module(frame):
    while frame is not None:
        if frame.f_code.co_name == "<module>":
            return rel(frame.f_code.co_filename) or "?"
        frame = frame.f_back
    return "?"


def _p_line(code, line):
    r = rel(code.co_filename)
    if r is None:
        return DISABLE
    import_lines[(r, line)].add(_importing_module(sys._getframe(1)))
    return DISABLE


def _p_module_edge(code, *_):
    # A module started or finished executing: let every line fire again so
    # shared helpers (metaclasses, builders) are attributed to each importer.
    if code.co_name == "<module>":
        M.restart_events()
        return None
    return DISABLE


def _p_module_return(code, *_):
    if code.co_name != "<module>":
        return DISABLE
    M.restart_events()
    g = sys._getframe(1).f_globals
    modname = g.get("__name__", "")
    if modname == "sqlglot" or modname.startswith("sqlglot."):
        # From here on, other modules' reads of this module's class
        # attributes are recorded against whichever module is importing.
        wrap_classes(g, modname)
    return None


_p_first_arg = {}


def _p_start(code, _off):
    if code.co_name == "<module>":
        M.restart_events()
        return None
    name = _p_first_arg.get(code, 0)
    if name == 0:
        name = None
        # Class creation hooks run on every class as it is defined; that is
        # not the module using the class.
        if (
            rel(code.co_filename)
            and code.co_argcount > 0
            and code.co_varnames[0] in ("self", "cls")
            and code.co_name not in ("__init_subclass__", "__set_name__", "__class_getitem__", "__new__")
        ):
            name = code.co_varnames[0]
        _p_first_arg[code] = name
    if name is None:
        return DISABLE
    frame = sys._getframe(1)
    mod = _importing_module(frame)
    if mod != "?":
        v = frame.f_locals.get(name)
        import_receivers.add((v if isinstance(v, type) else type(v), mod))
    return None


def _p_audit(event, args):
    if not _parent_active or event != "open":
        return
    path = any_rel(args[0])
    if path is None or path.endswith((".py", ".pyc")) or "__pycache__" in path:
        return
    frame = sys._getframe(1)
    while frame is not None:
        r = rel(frame.f_code.co_filename)
        if r and r.startswith("tests/") and os.path.basename(r).startswith("test_"):
            collect_opens[path].add(r)
            return
        frame = frame.f_back
    collect_opens[path].add("?")


def pytest_configure(config):
    if CALIB:
        return
    M.use_tool_id(TOOL, "brmap")
    M.register_callback(TOOL, M.events.LINE, _p_line)
    M.register_callback(TOOL, M.events.PY_START, _p_start)
    M.register_callback(TOOL, M.events.PY_RETURN, _p_module_return)
    M.set_events(TOOL, M.events.LINE | M.events.PY_START | M.events.PY_RETURN)
    sys.addaudithook(_p_audit)


def _import_everything():
    import sqlglot

    for info in pkgutil.walk_packages(sqlglot.__path__, prefix="sqlglot."):
        if info.name == "sqlglot.__main__":
            continue
        try:
            importlib.import_module(info.name)
        except Exception as e:  # noqa: BLE001
            print(f"brmap: cannot import {info.name}: {e}", file=sys.stderr)
    from sqlglot.dialects.dialect import Dialect, Dialects

    for d in Dialects:
        try:
            Dialect.get_or_raise(d.value)
        except Exception as e:  # noqa: BLE001
            print(f"brmap: cannot load dialect {d.value}: {e}", file=sys.stderr)


# ------------------------------------------------------- class attribute reads
#
# Every data attribute of a sqlglot class is replaced by a descriptor that
# records which class it was looked up through, and dict values by a dict that
# records which keys were looked up (or that the whole dict was used). A
# class-level change to one setting then reaches only the tests that read it.

WHOLE = "*"


def _ktok(key):
    if isinstance(key, str):
        return key
    if isinstance(key, type):
        return key.__name__
    if isinstance(key, enum.Enum):
        return key.name
    if isinstance(key, (int, float, bool)) or key is None:
        return repr(key)
    return "?"


class Tracked:
    __slots__ = ("value", "name")

    def __init__(self, value, name):
        self.value = value
        self.name = name

    def __get__(self, obj, owner=None):
        rec = REC
        if rec is not None:
            if not rec.paused:
                rec.attr_reads.add((owner if owner is not None else type(obj), self.name))
        elif IMPORTING:
            frame = sys._getframe(1)
            if not _in_class_body(frame):
                mod = _importing_module(frame)
                if mod != "?":
                    import_reads.add((owner if owner is not None else type(obj), self.name, None, mod))
        return self.value


_COMPREHENSIONS = ("<listcomp>", "<dictcomp>", "<setcomp>", "<genexpr>")


def _in_class_body(frame):
    """Is this read made by a class body (e.g. KEYWORDS = {**Base.KEYWORDS})?
    Those are copies or derivations the static analysis already accounts for."""
    if frame.f_code.co_name in _COMPREHENSIONS and frame.f_back is not None:
        frame = frame.f_back
    if frame.f_code.co_name == "<module>":
        return False
    loc = frame.f_locals
    return "__qualname__" in loc and "__module__" in loc


def _key_read(d, key):
    rec = REC
    owner = getattr(d, "_br_owner", None)
    if owner is None:
        return
    if rec is not None:
        if not rec.paused:
            rec.key_reads.add((owner, d._br_name, key))
    elif IMPORTING:
        frame = sys._getframe(2)
        if not _in_class_body(frame):
            mod = _importing_module(frame)
            if mod != "?":
                import_reads.add((owner, d._br_name, key, mod))


class TDict(dict):
    def __getitem__(self, k):
        _key_read(self, _ktok(k))
        return dict.__getitem__(self, k)

    def get(self, k, default=None):
        _key_read(self, _ktok(k))
        return dict.get(self, k, default)

    def __contains__(self, k):
        _key_read(self, _ktok(k))
        return dict.__contains__(self, k)

    def setdefault(self, k, default=None):
        _key_read(self, _ktok(k))
        return dict.setdefault(self, k, default)

    def pop(self, k, *default):
        _key_read(self, _ktok(k))
        return dict.pop(self, k, *default)


def _whole(name):
    base = getattr(dict, name)

    def method(self, *args, **kwargs):
        _key_read(self, WHOLE)
        return base(self, *args, **kwargs)

    method.__name__ = name
    return method


for _name in ("__iter__", "keys", "values", "items", "__len__", "copy", "__or__", "__ror__",
              "__eq__", "__ne__", "__reversed__", "popitem", "__reduce_ex__"):
    setattr(TDict, _name, _whole(_name))


def _tdict(value, owner, name):
    d = TDict(value)
    d._br_owner = owner
    d._br_name = name
    return d


_wrapped = set()


def wrap_classes(namespace, modname):
    """Replace the data attributes of the classes defined in a module."""

    def walk(cls):
        if not isinstance(cls, type) or cls in _wrapped or cls.__module__ != modname:
            return
        _wrapped.add(cls)
        if isinstance(cls, enum.EnumMeta):
            return
        for name, v in list(cls.__dict__.items()):
            if name.startswith("__") or name.startswith("_abc"):
                continue
            if isinstance(v, type):
                walk(v)
                continue
            if callable(v) or hasattr(type(v), "__get__") or hasattr(type(v), "__set__"):
                continue
            if type(v) is dict:
                v = _tdict(v, cls, name)
            type.__setattr__(cls, name, Tracked(v, name))

    for obj in list(namespace.values()):
        walk(obj)


def install_tracking():
    global IMPORTING
    import sqlglot.generator as generator_mod

    IMPORTING = False
    for modname, mod in list(sys.modules.items()):
        if modname == "sqlglot" or modname.startswith("sqlglot."):
            wrap_classes(vars(mod), modname)
    seen = _wrapped

    # The generator copies TRANSFORMS into a per-class dispatch table on first
    # use; keep tracking key lookups through that copy.
    orig = generator_mod._build_dispatch

    def build_dispatch(cls):
        rec = REC
        if rec is not None:
            rec.paused = True
        try:
            d = orig(cls)
        finally:
            if rec is not None:
                rec.paused = False
        return _tdict(d, cls, "TRANSFORMS")

    generator_mod._build_dispatch = build_dispatch
    generator_mod._DISPATCH_CACHE.clear()

    # Caches filled while importing would hide reads from the tests.
    import functools
    import gc

    for obj in gc.get_objects():
        if isinstance(obj, functools._lru_cache_wrapper):
            mod = getattr(obj, "__module__", "") or ""
            if mod.startswith("sqlglot"):
                obj.cache_clear()
    print(f"brmap: tracking attributes of {len(seen)} classes", file=sys.stderr)
    return sorted(k for k in (class_key(c) for c in seen) if k)


def class_key(k):
    mod = sys.modules.get(getattr(k, "__module__", None) or "")
    f = rel(getattr(mod, "__file__", None) or "")
    return f"{f}::{k.__qualname__}" if f else None


# ----------------------------------------------------------------- child phase

class Recorder:
    def __init__(self):
        self.lines = set()
        self.types = set()
        self.opens = set()
        self.dirs = set()
        self.subprocess = False
        self.failed = []
        self.cinfo = {}
        self.attr_reads = set()
        self.key_reads = set()
        self.paused = False

    def on_line(self, code, line):
        r = rel(code.co_filename)
        if r is not None:
            self.lines.add((r, line))
        return DISABLE

    def on_start(self, code, _off):
        name = self.cinfo.get(code, 0)
        if name == 0:
            name = None
            if rel(code.co_filename) and code.co_argcount > 0:
                first = code.co_varnames[0]
                if first in ("self", "cls"):
                    name = first
            self.cinfo[code] = name
        if name is None:
            return DISABLE
        try:
            v = sys._getframe(1).f_locals.get(name)
        except Exception:  # noqa: BLE001
            return None
        self.types.add(v if isinstance(v, type) else type(v))
        return None

    def audit(self, event, args):
        if event == "open":
            p = any_rel(args[0])
            if p is not None:
                self.opens.add(p)
        elif event in ("os.listdir", "os.scandir", "glob.glob"):
            p = any_rel(args[0] if args and args[0] is not None else ".")
            if p is not None:
                self.dirs.add(p)
        elif event in ("subprocess.Popen", "os.exec", "os.posix_spawn", "os.spawn"):
            self.subprocess = True

    def classes(self):
        out = set()
        for t in self.types:
            try:
                mro = t.__mro__
            except Exception:  # noqa: BLE001
                continue
            for k in mro:
                mod = sys.modules.get(getattr(k, "__module__", None) or "")
                f = rel(getattr(mod, "__file__", None) or "")
                if f:
                    out.add((f, k.__qualname__))
        return sorted(out)


REC = None


def pytest_runtest_logreport(report):
    if REC is not None and report.failed:
        REC.failed.append(f"{report.when}:{getattr(report, 'context', '')}")


def _run_calib(item, number):
    import resource
    from _pytest.runner import runtestprotocol

    r0 = resource.getrusage(resource.RUSAGE_SELF)
    reports = runtestprotocol(item, nextitem=None, log=False)
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    data = {"number": number, "id": item.nodeid, "cpu": round(cpu, 4),
            "failed": [r.when for r in reports if r.failed]}
    with open(f"{OUT}/tests/{number}.json", "w") as fh:
        json.dump(data, fh)


def _run_child(item, number):
    global REC
    from _pytest.runner import runtestprotocol

    REC = rec = Recorder()
    sys.addaudithook(rec.audit)
    M.use_tool_id(TOOL, "brmap-child")
    M.register_callback(TOOL, M.events.LINE, rec.on_line)
    M.register_callback(TOOL, M.events.PY_START, rec.on_start)
    M.set_events(TOOL, M.events.LINE | M.events.PY_START)
    t0 = time.time()
    reports = runtestprotocol(item, nextitem=None, log=False)
    M.set_events(TOOL, 0)
    for r in reports:
        if r.failed:
            rec.failed.append(r.when)
    lines = defaultdict(list)
    for f, ln in rec.lines:
        lines[f].append(ln)
    mro = {}

    def key_with_mro(cls):
        k = class_key(cls)
        if k is not None and k not in mro:
            mro[k] = [x for x in (class_key(b) for b in cls.__mro__) if x]
        return k

    attrs = sorted({(k, name) for cls, name in rec.attr_reads if (k := key_with_mro(cls))})
    keys = sorted({(k, name, tok) for cls, name, tok in rec.key_reads if (k := key_with_mro(cls))})
    data = {
        "number": number,
        "id": item.nodeid,
        "seconds": round(time.time() - t0, 3),
        "failed": rec.failed,
        "skipped": any(r.skipped for r in reports),
        "lines": {f: sorted(v) for f, v in sorted(lines.items())},
        "classes": rec.classes(),
        "opens": sorted(rec.opens),
        "dirs": sorted(rec.dirs),
        "subprocess": rec.subprocess,
        "attrs": attrs,
        "keys": keys,
        "mro": mro,
    }
    tmp = f"{OUT}/tests/{number}.json.tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh)
    os.rename(tmp, f"{OUT}/tests/{number}.json")


def pytest_runtestloop(session):
    global _parent_active
    if session.config.option.collectonly:
        return None

    _import_everything()
    _parent_active = False
    if CALIB:
        tracked = []
    else:
        M.set_events(TOOL, 0)
        M.free_tool_id(TOOL)
        tracked = install_tracking()

    os.makedirs(f"{OUT}/tests", exist_ok=True)
    with open(f"{OUT}/import.json", "w") as fh:
        json.dump(
            {
                "import_lines": sorted([f, ln, sorted(m)] for (f, ln), m in import_lines.items()),
                "collect_opens": {p: sorted(v) for p, v in sorted(collect_opens.items())},
                "tracked": tracked,
                "import_reads": sorted(
                    {(k, name, tok or "", mod) for cls, name, tok, mod in import_reads
                     for b in getattr(cls, "__mro__", ()) if (k := class_key(b))}
                ),
                "import_receivers": sorted(
                    {(k, mod) for cls, mod in import_receivers for b in getattr(cls, "__mro__", ())
                     if (k := class_key(b))}
                ),
            },
            fh,
        )

    for name in ("tests.test_optimizer", "tests.test_executor"):
        mod = sys.modules.get(name)
        if mod is not None and hasattr(mod, "ProcessPoolExecutor"):
            mod.ProcessPoolExecutor = SyncPool

    numbers = {}
    with open(os.environ["BRMAP_TESTS"]) as fh:
        for t in json.load(fh)["tests"]:
            numbers[t["id"]] = t["number"]
    only = {int(x) for x in ONLY.split(",")} if ONLY else None

    todo = []
    for item in session.items:
        n = numbers.get(item.nodeid)
        if n is None or (only is not None and n not in only):
            continue
        if os.path.exists(f"{OUT}/tests/{n}.json"):
            continue
        todo.append((item, n))
    print(f"brmap: {len(todo)} tests to map with {JOBS} jobs", file=sys.stderr)

    running = {}
    done = 0
    sys.stdout.flush()
    sys.stderr.flush()
    while todo or running:
        while todo and len(running) < JOBS:
            item, n = todo[0]
            try:
                pid = os.fork()
            except BlockingIOError:
                if not running:
                    raise
                break  # out of processes for now: wait for a child to finish
            todo.pop(0)
            if pid == 0:
                code = 0
                try:
                    (_run_calib if CALIB else _run_child)(item, n)
                except BaseException as e:  # noqa: BLE001
                    print(f"brmap: child {n} crashed: {e!r}", file=sys.stderr)
                    code = 1
                os._exit(code)
            running[pid] = (n, time.time())
        pid, status = os.waitpid(-1, os.WNOHANG)
        if pid == 0:
            now = time.time()
            for p, (n, t0) in list(running.items()):
                if now - t0 > TIMEOUT:
                    print(f"brmap: test {n} timed out, killing", file=sys.stderr)
                    os.kill(p, signal.SIGKILL)
            time.sleep(0.05)
            continue
        n, t0 = running.pop(pid)
        done += 1
        if status != 0:
            print(f"brmap: test {n} exited with {status}", file=sys.stderr)
        if done % 50 == 0:
            print(f"brmap: {done} done, {len(todo)} left", file=sys.stderr)
    return True
