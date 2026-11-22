"""Exact check of test_execute_tpcds for patches that only touch plan-phase code.

The test parses each TPC-DS query, sends it to a worker (pickled), where
sqlglot optimizes and plans it and the Python executor runs the plan over the
data; the result is compared with DuckDB running the query transpiled to
DuckDB SQL. Its outcome is therefore fixed by

  (a) what sqlglot produces before running a plan, per query: the plan the
      worker builds from the tree it receives (with every step's fields),
      the tables handed to the executor, and the DuckDB SQL; and that the
      tree parses and pickles;
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


def _value_key(v, depth=0):
    """A stable description of a class attribute's value, or None when it has
    no stable form (then it counts as changed)."""
    import enum
    import types

    if depth > 8:
        return None
    if v is None or isinstance(v, (bool, int, float, str, bytes)):
        return [type(v).__name__, repr(v)]
    if isinstance(v, enum.Enum):
        return ["enum", type(v).__qualname__, v.name]
    if isinstance(v, type):
        return ["type", v.__module__, v.__qualname__]
    if isinstance(v, (types.FunctionType, types.MethodType)):
        f = getattr(v, "__func__", v)
        return ["func", f.__module__, f.__qualname__, _code_key(f.__code__),
                _value_key(f.__defaults__, depth + 1), _value_key(getattr(f, "__wrapped__", None), depth + 1)]
    if isinstance(v, (types.BuiltinFunctionType, types.BuiltinMethodType)):
        return ["builtin", getattr(v, "__module__", None), v.__qualname__]
    if isinstance(v, (list, tuple)):
        items = [_value_key(x, depth + 1) for x in v]
        return None if None in items else [type(v).__name__, items]
    if isinstance(v, (set, frozenset)):
        items = [_value_key(x, depth + 1) for x in v]
        return None if None in items else [type(v).__name__, sorted(items, key=_jkey)]
    if isinstance(v, dict):
        items = [[_value_key(k, depth + 1), _value_key(x, depth + 1)] for k, x in v.items()]
        return None if any(None in kv for kv in items) else [type(v).__name__, items]
    r = repr(v)
    return None if " at 0x" in r or " object at " in r else ["repr", type(v).__qualname__, r]


def _code_key(code):
    consts = [_code_key(c) if hasattr(c, "co_code") else repr(c) for c in code.co_consts]
    return hashlib.sha256(_jkey([code.co_code.hex(), consts, code.co_names, code.co_varnames]).encode()).hexdigest()


def attr_fingerprints(data_attrs):
    """{"file::Class": {name: hash or None}} of the class attributes the data
    phase reads (maps/tpcds.json.gz), as they are after importing."""
    import importlib

    out = {}
    for ckey, names in data_attrs.items():
        path, qual = ckey.split("::")
        mod = path[:-3].replace("/", ".").removesuffix(".__init__")
        try:
            obj = importlib.import_module(mod)
            for part in qual.split("."):
                obj = getattr(obj, part)
        except Exception:  # noqa: BLE001
            out[ckey] = None
            continue
        row = {}
        for name in names:
            try:
                key = _value_key(getattr(obj, name))
            except Exception:  # noqa: BLE001
                key = None
            row[name] = None if key is None else hashlib.sha256(_jkey(key).encode()).hexdigest()[:16]
        out[ckey] = row
    return out


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
                # The worker only plans the tree (find_tables included), so
                # the plan stands for it.
                parts = []
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
