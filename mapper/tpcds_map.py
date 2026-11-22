"""Build maps/tpcds.json.gz for tpcds_check: baseline fingerprints, the lines
each query's plan phase runs, and the lines the data phase runs.

    python mapper/tpcds_map.py /out/tpcds_worker.json /out/tpcds.json.gz /out/tpcds_attrs.json
(run from the sqlglot checkout, with the repo root on PYTHONPATH; the first
and last files come from mapper/tpcds_worker.py, the last run with TPW_ALL=1)
"""
import gzip, json, os, sys
sys.path.insert(0, os.getcwd())
import tpcds_check

ROOT = os.getcwd() + "/"
M = sys.monitoring
TOOL = 4

# Import everything the check touches first, so per-query coverage is runtime only.
import sqlglot, sqlglot.executor, sqlglot.executor.python, sqlglot.planner  # noqa: E401
from sqlglot.dialects.dialect import Dialect
Dialect.get_or_raise("duckdb")
base = tpcds_check.fingerprints()
n = len(base)

# Baseline variants across hash seeds.
import subprocess
variants = {i: {fp} for i, fp in base.items()}
for seed in range(1, 9):
    res = subprocess.run(
        [sys.executable, "-c", "import json, tpcds_check; print(json.dumps(tpcds_check.fingerprints()))"],
        env=dict(os.environ, PYTHONHASHSEED=str(seed)), capture_output=True, text=True, check=True,
    )
    for i, fp in json.loads(res.stdout.strip().splitlines()[-1]).items():
        variants[int(i)].add(fp)
unstable = [i for i, v in variants.items() if len(v) > 1]
print("queries with seed-dependent fingerprints:", unstable)

query_lines = {}
for i in range(n):
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        seen = set()

        def on_line(code, line):
            f = code.co_filename
            if f.startswith(ROOT + "sqlglot/"):
                seen.add((f[len(ROOT):], line))
            return M.DISABLE

        M.use_tool_id(TOOL, "q")
        M.register_callback(TOOL, M.events.LINE, on_line)
        M.set_events(TOOL, M.events.LINE)
        fp = tpcds_check.fingerprints([i])
        M.set_events(TOOL, 0)
        os.write(w, json.dumps({"lines": sorted(seen), "fp": fp[i]}).encode())
        os._exit(0)
    os.close(w)
    buf = b""
    while chunk := os.read(r, 1 << 20):
        buf += chunk
    os.waitpid(pid, 0)
    d = json.loads(buf)
    if d["fp"] not in variants[i]:
        variants[i].add(d["fp"])
        print(f"query {i}: new variant in a fresh process")
    for f, ln in d["lines"]:
        query_lines.setdefault(f, {}).setdefault(str(ln), 0)
        query_lines[f][str(ln)] |= 1 << i

# CPU per query (warm, untraced), to check the cheap ones first.
import resource


def _cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


cost = {}
for i in range(n):
    t0 = _cpu()
    tpcds_check.fingerprints([i])
    cost[i] = round(_cpu() - t0, 4)

worker = json.load(open(sys.argv[1]))
data_attrs = json.load(open(sys.argv[3]))["data_attrs"]
out = {
    "test": tpcds_check.TEST_ID,
    "baseline": {str(i): sorted(v, key=str) for i, v in variants.items()},
    "query_lines": {f: {ln: format(m, "x") for ln, m in v.items()} for f, v in query_lines.items()},
    "data_lines": worker["data_lines"],
    "worker_modules": worker["worker_modules"],
    # Class attributes the data phase reads, and their values at the pinned
    # commit: a class-level change must leave all of them as they were.
    "data_attrs": data_attrs,
    "attr_baseline": tpcds_check.attr_fingerprints(data_attrs),
    "cost": {str(i): c for i, c in cost.items()},
}
with gzip.open(sys.argv[2], "wb", compresslevel=9) as fh:
    fh.write(json.dumps(out, separators=(",", ":")).encode())
print(f"{n} queries, {sum(fp is not None for fp in base.values())} fingerprinted, "
      f"{sum(len(v) for v in query_lines.values())} plan-phase lines, {os.path.getsize(sys.argv[2])} bytes")
