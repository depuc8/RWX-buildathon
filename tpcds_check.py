"""Exact check of test_execute_tpcds for patches that only touch plan-phase code.

The test parses each TPC-DS query, sends it to a worker (pickled), where
sqlglot optimizes and plans it and the Python executor runs the plan over the
data; the result is compared with DuckDB running the query transpiled to
DuckDB SQL. Its outcome is therefore fixed by

  (a) what sqlglot produces before running a plan, per query: the tree the
      worker receives, the plan with every step's fields, the tables handed
      to the executor, and the DuckDB SQL;
  (b) the code that runs plans over data (generating Python for each step,
      the row loops, env functions, tables), and the data itself.

For a patch that touches none of the code in (b), the test passes exactly
when (a) is unchanged for every query. This module computes (a) as one
fingerprint per query; the map holds the pinned commit's fingerprints (every
variant seen across hash seeds, since sqlglot's output for a few queries
depends on set ordering), the lines (b) runs, and which lines each query's
(a) runs.
"""

import hashlib
import json
import math
import pickle

TEST_ID = "tests/test_executor.py::TestExecutor::test_execute_tpcds"
FIXTURE = "optimizer/tpc-ds/tpc-ds.sql"


class _Stop(Exception):
    pass


def _jkey(v):
    return json.dumps(v, sort_keys=True, default=str)


def _canon(x):
    from sqlglot import exp, serde
    from sqlglot.executor.table import Table
    from sqlglot.planner import Step

    if isinstance(x, exp.Expr):
        return {"E": serde.dump(x)}
    if isinstance(x, Step):
        return {"S": x.name}
    if isinstance(x, Table):
        return {"Table": list(x.columns)}
    if isinstance(x, dict):
        return {"D": sorted(([_canon(k), _canon(v)] for k, v in x.items()), key=lambda kv: _jkey(kv[0]))}
    if isinstance(x, (list, tuple)):
        return {"L": [_canon(v) for v in x]}
    if isinstance(x, (set, frozenset)):
        return {"Set": sorted((_canon(v) for v in x), key=_jkey)}
    if isinstance(x, type):
        return {"T": f"{x.__module__}.{x.__qualname__}"}
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return {"F": repr(x)}
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    return {"?": repr(x)}  # unexpected: may not be stable, so it can only cause a mismatch


def _step(step):
    attrs = {k: v for k, v in vars(step).items() if k not in ("dependencies", "dependents")}
    return {
        "type": type(step).__name__,
        "attrs": _canon(attrs),
        "deps": sorted((_step(d) for d in step.dependencies), key=_jkey),
    }


def fingerprints(indices=None, baseline=None):
    """{query index: fingerprint of everything the test derives from it
    before running a plan}. A query whose fingerprinting raises gets None.
    With baseline ({index: allowed fingerprints}), stop at the first query
    that matches none of them: one difference already decides."""
    import sqlglot.executor as ex
    import sqlglot.executor.python as py
    from sqlglot import find_tables, parse_one, transpile
    from sqlglot.executor.table import Table
    from tests.helpers import TPCDS_SCHEMA, load_sql_fixture_pairs

    entries = list(load_sql_fixture_pairs(FIXTURE))
    # The real tables hold data, but nothing before the plan runs reads rows.
    tables = {name: Table(columns=columns, rows=[]) for name, columns in TPCDS_SCHEMA.items()}
    schema = pickle.loads(pickle.dumps(TPCDS_SCHEMA))  # the worker's copy
    captured = {}

    def stop(self, plan, outer_scope=None):
        captured["plan"], captured["tables"] = plan, self.tables
        raise _Stop

    orig = py.PythonExecutor.execute
    py.PythonExecutor.execute = stop
    out = {}
    try:
        for i in range(len(entries)) if indices is None else indices:
            meta, sql, _ = entries[i]
            try:
                expr = pickle.loads(pickle.dumps(parse_one(sql)))
                if not meta.get("execute"):
                    # The worker returns at once for these: only that the
                    # query parses and pickles matters, not the tree (whose
                    # shape can even vary with the hash seed).
                    out[i] = "parsed"
                    continue
                parts = [_canon(expr)]
                if meta.get("execute"):
                    captured.clear()
                    try:
                        ex.execute(expr, schema=schema, tables={t.name: tables[t.name] for t in find_tables(expr)})
                    except _Stop:
                        pass
                    plan = captured["plan"]
                    parts.append({
                        "expression": _canon(plan.expression),
                        "ctes": _canon(plan.ctes),
                        "root": _step(plan.root),
                        "tables": _canon(captured["tables"].mapping),
                    })
                    parts.append(transpile(sql, write="duckdb")[0])
                out[i] = hashlib.sha256(_jkey(parts).encode()).hexdigest()
            except Exception:  # noqa: BLE001  (the real test fails too, or differs)
                out[i] = None
            if baseline is not None and out[i] not in baseline.get(i, ()):
                break
    finally:
        py.PythonExecutor.execute = orig
    return out
