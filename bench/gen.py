"""Generate synthetic commits shaped like the sample set, for local evaluation
only (never shipped in the image).

    python3 bench/gen.py work/sqlglot work/bench 300 [seed]

Kinds, as in the sample: "code" (one token-level mutation in sqlglot/),
"comment" (a comment line added to sqlglot/), "fixture" (an edit to a SQL
fixture), "multi" (two or three of those in different files).
"""

import ast
import difflib
import glob
import gzip
import io
import json
import os
import random
import string
import sys
import tokenize

SWAPS = {
    "==": ["!="],
    "!=": ["=="],
    "<": ["<=", ">"],
    "<=": ["<", ">="],
    ">": [">=", "<"],
    ">=": [">", "<="],
    "and": ["or"],
    "or": ["and"],
    "True": ["False"],
    "False": ["True"],
    "+": ["-"],
    "-": ["+"],
    "is": ["==", "is not"],
    "in": ["not in"],
}
COMMENTS = [
    "# Intentionally not cached",
    "# Cheap check first",
    "# Keep this in sync with the generator",
    "# NOTE: order matters here",
    "# TODO: revisit",
    "# Fast path",
    "# See the dialect docs",
    "# Normalize before comparing",
]


def mutation_sites(text):
    sites = []
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError):
        return sites
    for i, t in enumerate(toks):
        if t.type in (tokenize.OP, tokenize.NAME) and t.string in SWAPS:
            nxt = toks[i + 1].string if i + 1 < len(toks) else ""
            prv = toks[i - 1].string if i else ""
            if t.string == "is" and nxt == "not":
                continue
            if t.string == "in" and prv == "not":
                continue
            if t.string in ("+", "-") and prv in ("(", ",", "=", "[", "", ":", "return"):
                continue  # unary
            sites.append(t)
        elif t.type == tokenize.NUMBER and t.string.isdigit():
            sites.append(t)
    return sites


def mutate_code(rng, text):
    sites = mutation_sites(text)
    rng.shuffle(sites)
    lines = text.split("\n")
    for t in sites[:20]:
        (r, c0), (_, c1) = t.start, t.end
        line = lines[r - 1]
        if t.type == tokenize.NUMBER:
            v = int(t.string)
            new = str(v + rng.choice([1, -1]) if v > 0 else v + 1)
        else:
            new = rng.choice(SWAPS[t.string])
        lines[r - 1] = line[:c0] + new + line[c1:]
        out = "\n".join(lines)
        try:
            if ast.dump(ast.parse(out)) != ast.dump(ast.parse(text)):
                return out
        except SyntaxError:
            pass
        lines[r - 1] = line
    return None


def add_comment(rng, text):
    lines = text.split("\n")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    stmts = [n for n in ast.walk(tree) if isinstance(n, ast.stmt) and n.lineno == n.end_lineno]
    if not stmts:
        return None
    s = rng.choice(stmts)
    indent = lines[s.lineno - 1][: len(lines[s.lineno - 1]) - len(lines[s.lineno - 1].lstrip())]
    lines.insert(s.lineno - 1, indent + rng.choice(COMMENTS))
    return "\n".join(lines)


def mutate_fixture(rng, text):
    lines = text.split("\n")
    cands = [i for i, l in enumerate(lines) if l.strip() and not l.startswith(("#", "--"))]
    if not cands:
        return None
    for _ in range(10):
        i = rng.choice(cands)
        words = lines[i].split(" ")
        op = rng.random()
        if op < 0.4:
            j = rng.randrange(len(words) + 1)
            words.insert(j, rng.choice(["42", "1", "x", "AS", "NOT"]))
        elif op < 0.7:
            nums = [j for j, w in enumerate(words) if w.strip(",;()").isdigit()]
            if not nums:
                continue
            j = rng.choice(nums)
            w = words[j]
            d = w.strip(",;()")
            words[j] = w.replace(d, str(int(d) + 1), 1)
        else:
            j = rng.randrange(len(words))
            w = words[j]
            if len(w) < 2:
                continue
            k = rng.randrange(len(w))
            words[j] = w[:k] + rng.choice(string.ascii_lowercase) + w[k + 1 :]
        new = " ".join(words)
        if new != lines[i]:
            lines[i] = new
            return "\n".join(lines)
    return None


def diff(path, old, new):
    body = list(
        difflib.unified_diff(
            old.split("\n"), new.split("\n"), f"a/{path}", f"b/{path}", n=3, lineterm=""
        )
    )
    return f"diff --git a/{path} b/{path}\nindex 0000000..1111111 100644\n" + "\n".join(body) + "\n"


def main():
    src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    rng = random.Random(seed)
    py = sorted(os.path.relpath(f, src) for f in glob.glob(f"{src}/sqlglot/**/*.py", recursive=True))
    weights = [len(mutation_sites(open(f"{src}/{f}").read())) + 1 for f in py]
    with gzip.open(os.path.join(os.path.dirname(__file__), "..", "maps", "brmap.json.gz")) as fh:
        opened = json.loads(fh.read())["opens"]
    fixtures = sorted(f for f in opened if f.startswith("tests/fixtures/") and f.endswith(".sql"))
    os.makedirs(f"{out}/commits", exist_ok=True)
    kinds = {}

    def one(kind):
        for _ in range(50):
            if kind == "fixture":
                f = rng.choice(fixtures)
                old = open(f"{src}/{f}", encoding="utf-8").read()
                new = mutate_fixture(rng, old)
            else:
                f = rng.choices(py, weights)[0]
                old = open(f"{src}/{f}", encoding="utf-8").read()
                new = (mutate_code if kind == "code" else add_comment)(rng, old)
            if new and new != old:
                return f, old, new
        raise RuntimeError("no mutation found")

    i = 0
    while i < n:
        kind = rng.choices(["code", "comment", "fixture", "multi"], [0.55, 0.1, 0.25, 0.1])[0]
        parts = []
        if kind == "multi":
            for k in rng.sample(["code", "comment", "fixture", "code"], rng.choice([2, 3])):
                parts.append(one(k))
        else:
            parts.append(one(kind))
        if len({p[0] for p in parts}) != len(parts):
            continue
        cid = f"b{seed}_{i:03d}"
        with open(f"{out}/commits/{cid}.patch", "w") as fh:
            for f, old, new in sorted(parts):
                fh.write(diff(f, old, new))
        kinds[cid] = kind
        i += 1
    json.dump(kinds, open(f"{out}/kinds.json", "w"), indent=1)


if __name__ == "__main__":
    main()
