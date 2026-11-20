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

    out = {
        "version": 1,
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
