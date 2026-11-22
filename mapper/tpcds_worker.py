"""What test_execute_tpcds's pool workers load and run.

The test sends each parsed query to a worker started with the "spawn"
context: a fresh interpreter that imports tests.test_executor, receives the
tables and schema, plans the query and runs the plan over the data. This
replays that in a fresh process and writes to argv[1]:

  worker_modules  every sqlglot/tests file the worker has loaded by the end.
                  A file outside this set never runs in a worker, so a change
                  to it can only reach the test through the main process
                  (the parsed tree, the DuckDB SQL), which tpcds_check
                  fingerprints.
  data_lines      lines run while PythonExecutor runs a plan (generating
                  Python for its steps, row loops, env functions), and
                  nothing else: not imports, not setUpClass.
  data_attrs      class data attributes (and dict keys) read while plans
                  run, as "file::Class" -> names, for every class in the
                  receiver's MRO (brmap's tracking).

Run from the sqlglot checkout with SKIP_INTEGRATION=0:
    python /repo/mapper/tpcds_worker.py /out/tpcds_worker.json
    TPW_ALL=1 python /repo/mapper/tpcds_worker.py /out/tpcds_attrs.json
and take worker_modules and data_lines from the first, data_attrs from the
second.
"""
import json, os, pickle, sys

sys.path.insert(0, os.getcwd())
ROOT = os.getcwd() + "/"
M = sys.monitoring
TOOL = 4
IN_DATA = [0]
data_lines = set()


def on_line(code, line):
    if not IN_DATA[0]:
        return None  # stay enabled: the same line may run later in the data phase
    f = code.co_filename
    if f.startswith(ROOT + "sqlglot/"):
        data_lines.add((f[len(ROOT):], line))
    return M.DISABLE


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brmap  # noqa: E402

import tests.test_executor as te  # noqa: E402  (what unpickling mp_execute imports)
import sqlglot.executor.python as py  # noqa: E402
from sqlglot import parse_one  # noqa: E402
from tests.helpers import TPCDS_SCHEMA  # noqa: E402

if os.environ.get("TPW_ALL"):
    # Wrap every class, even those a worker would load later (this run's
    # worker_modules then covers everything; take data_attrs from it).
    brmap._import_everything()
brmap.install_tracking()
brmap.REC = rec = brmap.Recorder()
rec.paused = True
te.TestExecutor.setUpClass()
tables = pickle.loads(pickle.dumps(te.TestExecutor.tpcds_tables))
te.initializer(*pickle.loads(pickle.dumps((TPCDS_SCHEMA, tables))))

orig = py.PythonExecutor.execute


def execute(self, *a, **k):
    IN_DATA[0] += 1
    rec.paused = False
    try:
        return orig(self, *a, **k)
    finally:
        IN_DATA[0] -= 1
        rec.paused = IN_DATA[0] == 0


py.PythonExecutor.execute = execute
M.use_tool_id(TOOL, "worker")
M.register_callback(TOOL, M.events.LINE, on_line)
M.set_events(TOOL, M.events.LINE)
ran = 0
for meta, sql, _ in te.TestExecutor.tpcds_sqls:
    expr = pickle.loads(pickle.dumps(parse_one(sql)))
    args = pickle.loads(pickle.dumps(meta))
    table = te.mp_execute(expr, args)
    ran += table is not None
    pickle.dumps(table)
M.set_events(TOOL, 0)

mods = set()
for mod in list(sys.modules.values()):
    f = getattr(mod, "__file__", None) or ""
    if f.startswith(ROOT) and f.endswith(".py"):
        r = f[len(ROOT):]
        if r.startswith(("sqlglot/", "tests/")):
            mods.add(r)
attrs = {}
for cls, name in rec.attr_reads:
    for b in getattr(cls, "__mro__", ()):
        k = brmap.class_key(b)
        if k:
            attrs.setdefault(k, set()).add(name)
for cls, name, tok in rec.key_reads:
    for b in getattr(cls, "__mro__", ()):
        k = brmap.class_key(b)
        if k:
            attrs.setdefault(k, set()).add(name)
out = {}
for f, ln in data_lines:
    out.setdefault(f, []).append(ln)
json.dump({"worker_modules": sorted(mods), "data_lines": {f: sorted(v) for f, v in sorted(out.items())},
           "data_attrs": {k: sorted(v) for k, v in sorted(attrs.items())}},
          open(sys.argv[1], "w"))
print(f"{ran} queries executed; {len(mods)} worker modules; {len(data_lines)} data-phase lines in {len(out)} files")
