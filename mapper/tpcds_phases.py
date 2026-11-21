"""Phase-tagged line coverage of test_execute_tpcds.

The test's outcome depends on (a) what sqlglot produces before execution
(parsed and optimized trees, plans, DuckDB SQL) and (b) the code that runs
the plans over data. This records which sqlglot lines run in (b), the data
phase: generating Python for plan steps, row loops, env functions, tables,
loading data. Writes {file: [lines]} to argv[1].
"""
import concurrent.futures, json, os, pickle, sys
sys.path.insert(0, os.getcwd())
M = sys.monitoring
TOOL = 4
ROOT = os.getcwd() + "/"
PHASE = ["data"]
data_lines = set()


def on_line(code, line):
    f = code.co_filename
    if not f.startswith(ROOT + "sqlglot/"):
        return M.DISABLE
    if PHASE[-1] == "data":
        data_lines.add((f[len(ROOT):], line))
        return M.DISABLE  # being in the data phase once is all we need to know
    return None


M.use_tool_id(TOOL, "phases")
M.register_callback(TOOL, M.events.LINE, on_line)


def in_phase(name, fn):
    def wrapped(*a, **k):
        PHASE.append(name)
        try:
            return fn(*a, **k)
        finally:
            PHASE.pop()
    return wrapped


def _cross(o):
    return pickle.loads(pickle.dumps(o))


class SyncPool:
    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        if initializer:
            initializer(*_cross(initargs))

    def submit(self, fn, *a, **k):
        f = concurrent.futures.Future()
        try:
            a, k = in_phase("plan", _cross)((a, k))  # expressions cross via serde
            f.set_result(_cross(fn(*a, **k)))
        except BaseException as e:
            f.set_exception(e)
        return f

    def __enter__(self):
        return self

    def __exit__(self, *e):
        return False


M.set_events(TOOL, M.events.LINE)
import sqlglot.executor as ex
import sqlglot.executor.python as py
import tests.test_executor as te

te.ProcessPoolExecutor = SyncPool
te.parse_one = in_phase("plan", te.parse_one)
te.transpile = in_phase("plan", te.transpile)
# Everything execute() does before running the plan is the plan phase...
ex.execute = in_phase("plan", ex.execute)
te.execute = ex.execute
# ...and running it (including generating Python for its steps) is data.
py.PythonExecutor.execute = in_phase("data", py.PythonExecutor.execute)

import unittest
te.TestExecutor.setUpClass()
res = unittest.TestResult()
te.TestExecutor("test_execute_tpcds").run(res)
M.set_events(TOOL, 0)
assert not res.failures and not res.errors, (res.failures, res.errors)
out = {}
for f, ln in data_lines:
    out.setdefault(f, []).append(ln)
json.dump({f: sorted(v) for f, v in sorted(out.items())}, open(sys.argv[1], "w"))
print("data-phase lines:", len(data_lines), "files:", len(out))
