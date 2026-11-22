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


def _open_file(file_name):
    """tests/test_executor.py's open_file."""
    with open(file_name, "rb") as f:
        gzipped = f.read(2) == b"\x1f\x8b"
    if gzipped:
        import gzip

        return gzip.open(file_name, "rt", newline="")
    return open(file_name, encoding="utf-8", newline="")


def load_table(name):
    """One TPC-DS table, built the way TestExecutor.setUpClass builds it."""
    import ast
    import csv

    from sqlglot.executor.table import Table
    from tests.helpers import FIXTURES_DIR, TPCDS_SCHEMA

    columns = TPCDS_SCHEMA[name]
    reader = csv.reader(_open_file(f"{FIXTURES_DIR}/optimizer/tpc-ds/{name}.csv.gz"), delimiter="|")
    rows = []
    ctypes = []
    next(reader)
    for row in reader:
        if not ctypes:
            for i, v in enumerate(row):
                try:
                    ctypes.append(type(ast.literal_eval(v)))
                except (ValueError, SyntaxError):
                    ctypes.append(str)
        rows.append(tuple(None if (t is not str and v == "") else t(v) for t, v in zip(ctypes, row)))
    return Table(columns=columns, rows=rows)


def run_query(expr, tables):
    """What a pool worker returns for this parsed query (mp_execute), as a
    hash of its columns and rows; None if it raises."""
    import sqlglot.executor as ex
    from sqlglot import find_tables
    from tests.helpers import TPCDS_SCHEMA

    try:
        schema = pickle.loads(pickle.dumps(TPCDS_SCHEMA))
        used = pickle.loads(pickle.dumps({t.name: tables[t.name] for t in find_tables(expr)}))
        table = pickle.loads(pickle.dumps(ex.execute(expr, schema=schema, tables=used)))
        return hashlib.sha256(repr((list(table.columns), table.rows)).encode()).hexdigest()
    except Exception:  # noqa: BLE001
        return None


def fingerprints(indices=None, baseline=None, ducks=None, duck_baseline=None, plan_only=None, affordable=None):
    """{query index: fingerprint of everything the test derives from it
    before running a plan}. A query whose fingerprinting raises gets None.
    ducks, if given, receives each executed query's DuckDB SQL hash.
    With baseline ({index: allowed fingerprints}), stop at the first query
    that matches none of them: one difference already decides; unless only
    its plan differs (its DuckDB SQL is in duck_baseline), which goes to
    plan_only for run_query to settle, as long as affordable(plan_only)."""
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
                    duck = transpile(sql, write="duckdb")[0]
                    parts.append(duck)
                    if ducks is not None:
                        ducks[i] = hashlib.sha256(duck.encode()).hexdigest()
                out[i] = hashlib.sha256(_jkey(parts).encode()).hexdigest()
            except Exception:  # noqa: BLE001  (the real test fails too, or differs)
                out[i] = None
            if baseline is not None and out[i] not in baseline.get(i, ()):
                if (plan_only is not None and out[i] is not None and meta.get("execute")
                        and ducks.get(i) in duck_baseline.get(i, ())):
                    plan_only.append(i)
                    if affordable is None or affordable(plan_only):
                        continue
                break
    finally:
        py.PythonExecutor.execute = orig
    return out
