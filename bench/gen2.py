"""A second synthetic set with kinds of change the sample does not have, to
catch misses on unseen kinds before the final does. Local evaluation only.

    python3 bench/gen2.py work/sqlglot work/bench2 160 [seed]
"""

import ast
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from gen import add_comment, diff, mutate_code, mutate_fixture  # noqa: E402

import glob  # noqa: E402
import gzip  # noqa: E402


def _stmts(tree):
    return [n for n in ast.walk(tree) if isinstance(n, ast.stmt)]


def _funcs(tree):
    return [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _valid(old, new):
    try:
        return ast.dump(ast.parse(new)) != ast.dump(ast.parse(old))
    except SyntaxError:
        return False


def mutate_string(rng, text):
    tree = ast.parse(text)
    lines = text.split("\n")
    cands = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
             and n.lineno == n.end_lineno and 1 <= len(n.value) <= 40 and "\\" not in lines[n.lineno - 1]
             and n.value.isprintable()]
    rng.shuffle(cands)
    for n in cands[:20]:
        line = lines[n.lineno - 1]
        seg = line[n.col_offset:n.end_col_offset]
        if not seg or seg[0] not in "'\"" or seg[-1] != seg[0]:
            continue
        v = n.value
        k = rng.randrange(len(v))
        nv = v[:k] + rng.choice("xyzQ_") + v[k + 1:] if rng.random() < 0.7 else v + "X"
        lines[n.lineno - 1] = line[:n.col_offset] + seg[0] + nv + seg[0] + line[n.end_col_offset:]
        out = "\n".join(lines)
        if _valid(text, out):
            return out
        lines[n.lineno - 1] = line
    return None


def delete_statement(rng, text):
    tree = ast.parse(text)
    lines = text.split("\n")
    cands = []
    for fn in _funcs(tree):
        for block in (fn.body,):
            for s in block:
                if (len(block) > 1 and s.lineno == s.end_lineno and not isinstance(s, (ast.Return, ast.Pass))
                        and not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))):
                    cands.append(s)
        for n in ast.walk(fn):
            for field in ("body", "orelse"):
                block = getattr(n, field, None)
                if isinstance(block, list) and len(block) > 1 and n is not fn:
                    cands.extend(s for s in block if s.lineno == s.end_lineno and isinstance(s, (ast.Assign, ast.Expr, ast.AugAssign)))
    rng.shuffle(cands)
    for s in cands[:20]:
        new = lines[: s.lineno - 1] + lines[s.lineno:]
        out = "\n".join(new)
        if _valid(text, out):
            return out
    return None


def insert_return(rng, text):
    tree = ast.parse(text)
    lines = text.split("\n")
    fns = [f for f in _funcs(tree) if len(f.body) > 1 and not any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(f))]
    rng.shuffle(fns)
    for fn in fns[:20]:
        s = rng.choice(fn.body[1:])
        indent = lines[s.lineno - 1][: len(lines[s.lineno - 1]) - len(lines[s.lineno - 1].lstrip())]
        new = lines[: s.lineno - 1] + [indent + "return None"] + lines[s.lineno - 1:]
        out = "\n".join(new)
        if _valid(text, out):
            return out
    return None


def swap_args(rng, text):
    tree = ast.parse(text)
    lines = text.split("\n")
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and len(n.args) >= 2 and n.lineno == n.end_lineno
             and all(a.lineno == a.end_lineno == n.lineno for a in n.args[:2])]
    rng.shuffle(calls)
    for c in calls[:20]:
        a, b = c.args[0], c.args[1]
        line = lines[c.lineno - 1]
        sa, sb = line[a.col_offset:a.end_col_offset], line[b.col_offset:b.end_col_offset]
        if sa == sb or not line.isascii():
            continue
        new_line = line[:a.col_offset] + sb + line[a.end_col_offset:b.col_offset] + sa + line[b.end_col_offset:]
        lines[c.lineno - 1] = new_line
        out = "\n".join(lines)
        if _valid(text, out):
            return out
        lines[c.lineno - 1] = line
    return None


def remove_import(rng, text):
    tree = ast.parse(text)
    lines = text.split("\n")
    imps = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)) and n.lineno == n.end_lineno
            and not (isinstance(n, ast.ImportFrom) and n.module == "__future__")]
    rng.shuffle(imps)
    for n in imps[:10]:
        out = "\n".join(lines[: n.lineno - 1] + lines[n.lineno:])
        if _valid(text, out):
            return out
    return None


def remove_decorator(rng, text):
    tree = ast.parse(text)
    lines = text.split("\n")
    decs = [d for f in _funcs(tree) for d in f.decorator_list if d.lineno == d.end_lineno]
    rng.shuffle(decs)
    for d in decs[:10]:
        out = "\n".join(lines[: d.lineno - 1] + lines[d.lineno:])
        if _valid(text, out):
            return out
    return None


def change_docstring(rng, text):
    tree = ast.parse(text)
    lines = text.split("\n")
    docs = []
    for n in [tree, *ast.walk(tree)]:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef, ast.Module)) and n.body:
            s = n.body[0]
            if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str):
                docs.append(s)
    rng.shuffle(docs)
    for s in docs[:10]:
        ln = s.lineno - 1
        line = lines[ln]
        if '"""' not in line and "'''" not in line:
            continue
        lines[ln] = line.replace('"""', '"""Note: ', 1) if '"""' in line else line.replace("'''", "'''Note: ", 1)
        out = "\n".join(lines)
        if _valid(text, out):
            return out
        lines[ln] = line
    return None


def edit_test(rng, text):
    """Change an expected string in a test assertion."""
    return mutate_string(rng, text)


KINDS = {
    "string": (mutate_string, "sqlglot"),
    "delete": (delete_statement, "sqlglot"),
    "insert": (insert_return, "sqlglot"),
    "swap": (swap_args, "sqlglot"),
    "import": (remove_import, "sqlglot"),
    "decorator": (remove_decorator, "sqlglot"),
    "docstring": (change_docstring, "sqlglot"),
    "test": (edit_test, "tests"),
}


def main():
    src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    rng = random.Random(seed)
    py = sorted(os.path.relpath(f, src) for f in glob.glob(f"{src}/sqlglot/**/*.py", recursive=True))
    tests = sorted(os.path.relpath(f, src) for f in glob.glob(f"{src}/tests/**/test_*.py", recursive=True))
    with gzip.open(os.path.join(os.path.dirname(__file__), "..", "maps", "brmap.json.gz")) as fh:
        opened = json.loads(fh.read())["opens"]
    fixtures = sorted(f for f in opened if f.startswith("tests/fixtures/") and f.endswith(".sql"))
    os.makedirs(f"{out}/commits", exist_ok=True)
    kinds = {}
    weights = {"string": 3, "delete": 3, "insert": 2, "swap": 2, "import": 1, "decorator": 1, "docstring": 1,
               "test": 2, "multi": 2}

    def one(kind):
        for _ in range(80):
            if kind == "fixture":
                f = rng.choice(fixtures)
                fn = mutate_fixture
            elif kind in ("code", "comment"):
                f = rng.choice(py)
                fn = mutate_code if kind == "code" else add_comment
            else:
                fn, where = KINDS[kind]
                f = rng.choice(tests if where == "tests" else py)
            old = open(f"{src}/{f}", encoding="utf-8").read()
            try:
                new = fn(rng, old)
            except (SyntaxError, ValueError, IndexError):
                new = None
            if new and new != old:
                return f, old, new
        raise RuntimeError(f"no {kind} mutation found")

    i = 0
    names = list(weights)
    while i < n:
        kind = rng.choices(names, [weights[k] for k in names])[0]
        if kind == "multi":
            parts = [one(k) for k in rng.sample(["string", "delete", "swap", "fixture", "code", "test"], 2)]
        else:
            parts = [one(kind)]
        if len({p[0] for p in parts}) != len(parts):
            continue
        cid = f"c{seed}_{i:03d}"
        with open(f"{out}/commits/{cid}.patch", "w") as fh:
            for f, old, new in sorted(parts):
                fh.write(diff(f, old, new))
        kinds[cid] = kind
        i += 1
    json.dump(kinds, open(f"{out}/kinds.json", "w"), indent=1)


if __name__ == "__main__":
    main()
