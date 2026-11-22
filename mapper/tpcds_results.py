"""Baseline results of every executed TPC-DS query, for tpcds_check.run_queries.

When a patch changes some queries' plans but not their DuckDB SQL, the check
runs just those queries over the real data and compares the rows with these.
Writes {"results": {index: [hash, ...]}, "exec_cost": {index: seconds},
"tables": {index: [table, ...]}, "load_cost": {table: seconds}} to argv[1].

Run from the sqlglot checkout (with duckdb and pandas installed, since it
also checks that tpcds_check.load_table matches the test's setUpClass) and
SKIP_INTEGRATION=0, once per hash seed:
    PYTHONHASHSEED=0 python /repo/mapper/tpcds_results.py /out/tpcds_results0.json
"""
import json, os, pickle, resource, sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tpcds_check  # noqa: E402
from sqlglot import find_tables, parse_one  # noqa: E402
from tests.helpers import TPCDS_SCHEMA, load_sql_fixture_pairs  # noqa: E402


def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


tables, load_cost = {}, {}
for name in TPCDS_SCHEMA:
    t0 = cpu()
    tables[name] = tpcds_check.load_table(name)
    load_cost[name] = round(cpu() - t0, 4)

if os.environ.get("CHECK_SETUP"):
    import tests.test_executor as te

    te.TestExecutor.setUpClass()
    for name, t in te.TestExecutor.tpcds_tables.items():
        assert list(t.columns) == list(tables[name].columns) and t.rows == tables[name].rows, name
    print("load_table matches setUpClass for all", len(tables), "tables")

results, exec_cost, used = {}, {}, {}
for i, (meta, sql, _) in enumerate(load_sql_fixture_pairs(tpcds_check.FIXTURE)):
    if not meta.get("execute"):
        continue
    expr = pickle.loads(pickle.dumps(parse_one(sql)))
    used[i] = sorted({t.name for t in find_tables(expr)})
    t0 = cpu()
    results[i] = tpcds_check.run_query(expr, tables)
    exec_cost[i] = round(cpu() - t0, 4)
json.dump({"results": results, "exec_cost": exec_cost, "tables": used, "load_cost": load_cost},
          open(sys.argv[1], "w"))
print(len(results), "queries;", round(sum(exec_cost.values()), 2), "s to run;",
      round(sum(load_cost.values()), 2), "s to load")
