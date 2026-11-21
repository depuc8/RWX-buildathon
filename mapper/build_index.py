"""Turn brmap's per-test records into the compact map the selector ships with.

    python3 mapper/build_index.py work/out/map work/sqlglot sample/tests.json maps/brmap.json.gz

Every set of tests is stored as a bitset over test numbers (a hex string),
deduplicated into one table and referenced by index.
"""

import ast
import glob
import gzip
import json
import os
import subprocess
import sys
from collections import defaultdict


def module_name(path):
    p = path[:-3]
    if p.endswith("/__init__"):
        p = p[: -len("/__init__")]
    return p.replace("/", ".")


def static_imports(src_root):
    """file -> set of repo files it imports (including parent packages)."""
    files = [
        os.path.relpath(f, src_root)
        for pattern in ("sqlglot/**/*.py", "tests/**/*.py")
        for f in glob.glob(os.path.join(src_root, pattern), recursive=True)
    ]
    by_module = {module_name(f): f for f in files}

    def resolve(name):
        out = []
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            f = by_module.get(".".join(parts[:i]))
            if f:
                out.append(f)
        return out

    imports = defaultdict(set)
    for f in files:
        try:
            tree = ast.parse(open(os.path.join(src_root, f), encoding="utf-8").read())
        except SyntaxError:
            continue
        pkg = module_name(f) if f.endswith("__init__.py") else module_name(f).rpartition(".")[0]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    imports[f].update(resolve(a.name))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = pkg.split(".")
                    base = base[: len(base) - node.level + 1]
                    mod = ".".join(base + ([node.module] if node.module else []))
                else:
                    mod = node.module or ""
                imports[f].update(resolve(mod))
                for a in node.names:
                    sub = by_module.get(f"{mod}.{a.name}")
                    if sub:
                        imports[f].add(sub)
        imports[f].discard(f)
    return files, imports


def derived_names(src_root):
    """Attribute names read at import time to build *other* things: in
    __init_subclass__ or metaclass methods, or in class bodies and module code
    that copy one attribute into a differently named one (or from a class that
    is not a base). A change to such an attribute keeps the class-wide
    treatment, since tests use what was derived from it, not the attribute."""
    names = set()

    def is_meta(cls):
        return any(isinstance(b, ast.Name) and b.id == "type" for b in cls.bases)

    def loads(node, skip_lambdas=True):
        stack = [node]
        while stack:
            n = stack.pop()
            if skip_lambdas and isinstance(n, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            yield n
            stack.extend(ast.iter_child_nodes(n))

    def construction(fn):
        # X.pop(k) / X.remove(k) only take entries out: per-key tracking of the
        # (already pruned) dict stays exact, so these are not derivations.
        pruned = set()
        for n in loads(fn, skip_lambdas=False):
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr in ("pop", "remove", "discard")
            ):
                pruned.add(id(n.func.value))
        for n in loads(fn, skip_lambdas=False):
            if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load):
                if id(n) in pruned:
                    continue
                names.add(n.attr)
            elif isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.isidentifier():
                names.add(n.value)

    def class_level(stmt, cls):
        targets = set()
        if isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name):
                    targets.add(t.id)
        elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)) and isinstance(stmt.target, ast.Name):
            targets.add(stmt.target.id)
        bases = {ast.unparse(b) for b in cls.bases} if cls is not None else set()
        value = getattr(stmt, "value", None)
        if value is None:
            return
        for n in loads(value):
            if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load):
                # X = {**Base.X, ...} in a subclass of Base is a copy: keys
                # stay tracked through the subclass's own dict.
                if n.attr in targets and ast.unparse(n.value) in bases:
                    continue
                names.add(n.attr)
            elif cls is not None and isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                if n.id not in targets:
                    names.add(n.id)

    def walk(body, cls):
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if stmt.name in ("__init_subclass__", "__set_name__") or (cls is not None and is_meta(cls)):
                    construction(stmt)
            elif isinstance(stmt, ast.ClassDef):
                walk(stmt.body, stmt)
            elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                class_level(stmt, cls)
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                # e.g. X.update(Y.Z) in a class body
                for arg in stmt.value.args:
                    class_level(ast.Assign(targets=[], value=arg), cls)
            else:
                for field in ("body", "orelse", "finalbody"):
                    sub = getattr(stmt, field, None)
                    if isinstance(sub, list):
                        walk(sub, cls)

    for f in glob.glob(os.path.join(src_root, "sqlglot/**/*.py"), recursive=True):
        try:
            tree = ast.parse(open(f, encoding="utf-8").read())
        except SyntaxError:
            continue
        walk(tree.body, None)
    return names


def main():
    map_dir, src_root, tests_json, out_path = sys.argv[1:5]
    tests = json.load(open(tests_json))["tests"]
    n_tests = len(tests)

    bits_index = {}
    bits = []

    def ref(mask):
        h = format(mask, "x")
        i = bits_index.get(h)
        if i is None:
            i = bits_index[h] = len(bits)
            bits.append(h)
        return i

    line_users = defaultdict(lambda: defaultdict(int))
    class_users = defaultdict(lambda: defaultdict(int))
    opens = defaultdict(int)
    dirs = defaultdict(int)
    attr_users = defaultdict(lambda: defaultdict(int))  # class key -> attr -> mask
    key_users = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # class -> attr -> key
    subprocess_mask = 0
    seen = 0
    for t in tests:
        path = os.path.join(map_dir, "tests", f"{t['number']}.json")
        if not os.path.exists(path):
            print(f"missing map for test {t['number']} {t['id']}", file=sys.stderr)
            continue
        d = json.load(open(path))
        if d["failed"]:
            print(f"test {t['number']} failed while mapping: {d['failed']}", file=sys.stderr)
        seen += 1
        b = 1 << t["number"]
        for f, lines in d["lines"].items():
            lu = line_users[f]
            for ln in lines:
                lu[ln] |= b
        for f, q in d["classes"]:
            class_users[f][q] |= b
        for p in d["opens"]:
            opens[p] |= b
        for p in d["dirs"]:
            dirs[p] |= b
        if d["subprocess"]:
            subprocess_mask |= b
        mro = d.get("mro", {})
        keyed = set()
        for k, name, tok in d.get("keys", ()):
            keyed.add(name)
            for c in mro.get(k, [k]):
                key_users[c][name][tok] |= b
        for k, name in d.get("attrs", ()):
            for c in mro.get(k, [k]):
                attr_users[c][name] |= b
                if name not in keyed:
                    # Read but never looked up by key: count it as using all of it.
                    key_users[c][name]["*"] |= b
    print(f"{seen}/{n_tests} tests mapped", file=sys.stderr)

    imp = json.load(open(os.path.join(map_dir, "import.json")))
    import_lines = defaultdict(dict)
    for f, ln, mods in imp["import_lines"]:
        import_lines[f][str(ln)] = mods

    files, imports = static_imports(src_root)
    importers = defaultdict(set)
    for f, deps in imports.items():
        for d in deps:
            importers[d].add(f)

    # Transitive importers: a module-level change can reach anything that
    # imports it directly or through re-exports.
    closure = {}
    for f in files:
        seen_f, stack = set(), [f]
        while stack:
            for g in importers.get(stack.pop(), ()):
                if g not in seen_f:
                    seen_f.add(g)
                    stack.append(g)
        seen_f.discard(f)
        closure[f] = sorted(seen_f)

    # Modules a bare `import sqlglot` loads: only changes to these can break
    # the tests that import sqlglot in a subprocess.
    eager = subprocess.run(
        [sys.executable, "-I", "-c",
         "import sys; sys.path.insert(0, sys.argv[1]); import sqlglot, json; "
         "print(json.dumps(sorted(m.__file__ for n, m in list(sys.modules.items()) "
         "if (n == 'sqlglot' or n.startswith('sqlglot.')) and getattr(m, '__file__', None))))",
         os.path.abspath(src_root)],
        check=True, capture_output=True, text=True,
    ).stdout
    eager = sorted(os.path.relpath(p, os.path.abspath(src_root)) for p in json.loads(eager))
    print(f"{len(eager)} modules load eagerly with `import sqlglot`", file=sys.stderr)

    out = {
        "version": 1,
        "eager": eager,
        "n_tests": n_tests,
        "bits": bits,
        "lines": {
            f: {str(ln): ref(m) for ln, m in sorted(lu.items())} for f, lu in sorted(line_users.items())
        },
        "classes": {
            f: {q: ref(m) for q, m in sorted(cu.items())} for f, cu in sorted(class_users.items())
        },
        "opens": {p: ref(m) for p, m in sorted(opens.items())},
        "collect_opens": imp["collect_opens"],
        "dirs": {p: ref(m) for p, m in sorted(dirs.items())},
        "subprocess": ref(subprocess_mask),
        "import_lines": import_lines,
        "importers": closure,
        "tracked": imp.get("tracked", []),
        "attrs": {c: {n: ref(m) for n, m in sorted(v.items())} for c, v in sorted(attr_users.items())},
        "keys": {
            c: {n: {t: ref(m) for t, m in sorted(toks.items())} for n, toks in sorted(v.items())}
            for c, v in sorted(key_users.items())
        },
        "derived": sorted(derived_names(src_root)),
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    raw = json.dumps(out, separators=(",", ":")).encode()
    with gzip.open(out_path, "wb", compresslevel=9) as fh:
        fh.write(raw)
    print(
        f"wrote {out_path}: {len(raw) / 1e6:.1f} MB raw, {os.path.getsize(out_path) / 1e6:.2f} MB gz, "
        f"{len(bits)} distinct test sets",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
