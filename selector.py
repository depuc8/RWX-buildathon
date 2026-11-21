"""Blast Radius selector: pick the tests a patch can break, from a map built at
the pinned sqlglot commit.

For every changed region of a Python file we work out *how* the change can
reach a test, from the old and new syntax trees:

  inside a function body  -> tests that executed those lines at runtime
                             (plus import-time effects if the lines ran
                             while a module was being imported)
  a def header / a method -> tests that ran that function, or every user of
  added or removed           the class when a new method may override one
  a class body statement  -> every test that ran a method on that class or a
                             subclass
  a module statement      -> every user of the module and of everything that
                             imports it, directly or through re-exports

Other files map to the tests that opened them (fixtures) or listed their
directory. Anything we cannot analyse selects every test: a miss costs 300 s,
running everything costs well under that.
"""

import ast
import gzip
import json
import os
import re
import sys
import time
from pathlib import Path

DATA = Path(os.environ.get("BR_DATA", "/data"))
HERE = Path(__file__).resolve().parent
SRC = Path(os.environ.get("BR_SRC", HERE / "src"))
MAP = Path(os.environ.get("BR_MAP", HERE / "maps" / "brmap.json.gz"))

MAX_WORKERS = 8
WORKER_MS = 2000
ALL_FILES = {"pyproject.toml", "setup.cfg", "pytest.ini", "tox.ini"}


def log(*args):
    print(*args, file=sys.stderr)


# ------------------------------------------------------------------- the map

class Map:
    def __init__(self, path, n_tests):
        with gzip.open(path, "rb") as fh:
            m = json.loads(fh.read())
        self.raw_bits = m["bits"]
        self._bits = {}
        self.lines = m["lines"]
        self.classes = m["classes"]
        self.opens = m["opens"]
        self.collect_opens = m["collect_opens"]
        self.dirs = m["dirs"]
        self.subprocess = self.bits(m["subprocess"])
        self.import_lines = m["import_lines"]
        self.importers = m["importers"]
        self.attrs = m.get("attrs", {})
        self.keys = m.get("keys", {})
        self.derived = set(m.get("derived", ()))
        self.tracked = set(m.get("tracked", ()))
        self.eager = set(m.get("eager", ()))
        self.n_tests = n_tests
        self.all = (1 << n_tests) - 1
        self._module_users = {}

    def bits(self, i):
        b = self._bits.get(i)
        if b is None:
            b = self._bits[i] = int(self.raw_bits[i], 16)
        return b

    def line_users(self, f, lines):
        lu = self.lines.get(f)
        if not lu:
            return 0
        mask = 0
        for ln in lines:
            i = lu.get(str(ln))
            if i is not None:
                mask |= self.bits(i)
        return mask

    def range_users(self, f, start, end):
        return self.line_users(f, range(start, end + 1))

    def class_users(self, f, qualname):
        i = self.classes.get(f, {}).get(qualname)
        return 0 if i is None else self.bits(i)

    def attr_users(self, ckey, name):
        """Tests that read class attribute `name` through this class or a subclass."""
        i = self.attrs.get(ckey, {}).get(name)
        return 0 if i is None else self.bits(i)

    def key_users(self, ckey, name, tok):
        i = self.keys.get(ckey, {}).get(name, {}).get(tok)
        return 0 if i is None else self.bits(i)

    def module_users(self, f):
        """Tests that touched module f at all."""
        m = self._module_users.get(f)
        if m is None:
            m = 0
            if f.startswith("tests/"):
                m |= TESTS_BY_FILE.get(f, 0)
            for i in self.lines.get(f, {}).values():
                m |= self.bits(i)
            for i in self.classes.get(f, {}).values():
                m |= self.bits(i)
            self._module_users[f] = m
        return m

    def module_impact(self, f):
        """A module-level change to f: f's users and its importers' users."""
        m = self.module_users(f) | self.eager_tests(f)
        for g in self.importers.get(f, ()):
            m |= self.module_users(g)
        return m

    def eager_tests(self, f):
        """Tests that run `import sqlglot` in a subprocess (we cannot see
        inside it): reachable by import-time changes to eagerly loaded modules."""
        return self.subprocess if f in self.eager else 0

    def import_impact(self, f, lines):
        imp = self.import_lines.get(f)
        if not imp:
            return 0
        m = 0
        for ln in lines:
            mods = imp.get(str(ln))
            if mods:
                for mod in mods:
                    m |= self.all if mod == "?" else self.module_impact(mod)
        if m:
            m |= self.eager_tests(f)
        return m


TESTS_BY_FILE = {}


# ----------------------------------------------------------------- the patch

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class FileDiff:
    def __init__(self):
        self.old = None  # path, or None for a new file
        self.new = None  # path, or None for a deleted file
        self.hunks = []  # (old_start, new_start, [lines])
        self.binary = False


def parse_patch(text):
    diffs = []
    cur = None
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("diff --git "):
            cur = FileDiff()
            diffs.append(cur)
            m = re.match(r"diff --git a/(.*) b/(.*)$", line)
            if m:
                cur.old, cur.new = m.group(1), m.group(2)
        elif cur is None:
            pass
        elif line.startswith("--- "):
            p = line[4:].split("\t")[0]
            cur.old = None if p == "/dev/null" else p[2:] if p.startswith("a/") else p
        elif line.startswith("+++ "):
            p = line[4:].split("\t")[0]
            cur.new = None if p == "/dev/null" else p[2:] if p.startswith("b/") else p
        elif line.startswith("new file mode"):
            cur.old = None
        elif line.startswith("deleted file mode"):
            cur.new = None
        elif line.startswith("rename from "):
            cur.old = line[len("rename from "):]
        elif line.startswith("rename to "):
            cur.new = line[len("rename to "):]
        elif line.startswith("Binary files") or line.startswith("GIT binary patch"):
            cur.binary = True
        elif line.startswith("@@"):
            m = HUNK.match(line)
            if not m:
                raise ValueError(f"bad hunk header {line!r}")
            old_len = int(m.group(2)) if m.group(2) is not None else 1
            new_len = int(m.group(4)) if m.group(4) is not None else 1
            body = []
            i += 1
            o = n = 0
            while i < len(lines) and (o < old_len or n < new_len or lines[i].startswith("\\")):
                hl = lines[i]
                if hl.startswith("\\"):
                    i += 1
                    continue
                tag = hl[:1] or " "
                if tag == " ":
                    o += 1
                    n += 1
                elif tag == "-":
                    o += 1
                elif tag == "+":
                    n += 1
                else:
                    raise ValueError(f"bad hunk line {hl!r}")
                body.append((tag, hl[1:]))
                i += 1
            cur.hunks.append((int(m.group(1)), int(m.group(3)), body))
            continue
        i += 1
    return diffs


def apply_hunks(old_lines, hunks):
    """New file lines, and the change groups: (old_start, old_end, new_start,
    new_end) as 1-based half-open line ranges of each run of -/+ lines."""
    new = []
    groups = []
    pos = 0  # 0-based index into old_lines
    for old_start, _new_start, body in hunks:
        start = old_start - 1 if old_start > 0 else 0
        if start < pos:
            raise ValueError("overlapping hunks")
        new.extend(old_lines[pos:start])
        pos = start
        k = 0
        while k < len(body):
            tag, text = body[k]
            if tag == " ":
                if pos >= len(old_lines) or old_lines[pos].rstrip("\r") != text.rstrip("\r"):
                    raise ValueError("context mismatch")
                new.append(old_lines[pos])
                pos += 1
                k += 1
                continue
            g_old, g_new = pos, len(new)
            while k < len(body) and body[k][0] != " ":
                tag, text = body[k]
                if tag == "-":
                    if pos >= len(old_lines) or old_lines[pos].rstrip("\r") != text.rstrip("\r"):
                        raise ValueError("removed line mismatch")
                    pos += 1
                else:
                    new.append(text)
                k += 1
            groups.append((g_old + 1, pos + 1, g_new + 1, len(new) + 1))
    new.extend(old_lines[pos:])
    return new, groups


# --------------------------------------------------------------- python trees

STMT_LISTS = ("body", "orelse", "finalbody", "handlers", "cases")
FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)


def has_pos(node):
    return hasattr(node, "lineno") and getattr(node, "end_lineno", None) is not None


def start(node):
    return (node.lineno, node.col_offset)


def end(node):
    return (node.end_lineno, node.end_col_offset)


def child_stmts(node):
    out = []
    for field in STMT_LISTS:
        v = getattr(node, field, None)
        if isinstance(v, list):
            out.extend(x for x in v if isinstance(x, (ast.stmt, ast.excepthandler, ast.match_case)))
    for x in out:
        if isinstance(x, ast.match_case) and not hasattr(x, "lineno"):
            x.lineno, x.col_offset = x.pattern.lineno, x.pattern.col_offset
            x.end_lineno, x.end_col_offset = x.body[-1].end_lineno, x.body[-1].end_col_offset
    return sorted(out, key=start)


def body_range(node):
    """(start, end) of the statements a compound node owns, or None."""
    kids = child_stmts(node)
    if not kids:
        return None
    return start(kids[0]), end(kids[-1])


def stmt_lines(node):
    """Lines that run when this statement runs: all of a simple statement, the
    header of a compound one."""
    lo = node.lineno
    if isinstance(node, FUNCS + (ast.ClassDef,)) and node.decorator_list:
        lo = min(d.lineno for d in node.decorator_list)
    kids = child_stmts(node)
    if not kids:
        return lo, node.end_lineno
    return lo, max(lo, kids[0].lineno - 1)


class Tree:
    """An old or new version of a Python file."""

    def __init__(self, text):
        self.tree = ast.parse(text)
        self._funcs = None
        self._toplevel = None

    def funcs(self):
        """qualname -> (first line incl. decorators, last line) of every def."""
        if self._funcs is None:
            self._funcs = {}

            def walk(node, prefix):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, FUNCS):
                        q = prefix + child.name
                        lo = min([child.lineno] + [d.lineno for d in child.decorator_list])
                        self._funcs[q] = (lo, child.end_lineno)
                        walk(child, q + ".<locals>.")
                    elif isinstance(child, ast.ClassDef):
                        walk(child, prefix + child.name + ".")
                    else:
                        walk(child, prefix)

            walk(self.tree, "")
        return self._funcs

    def toplevel_names(self):
        if self._toplevel is None:
            names = set()
            for node in self.tree.body:
                for sub in ast.walk(node):
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        names.add(sub.name)
                        break
                    if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                        names.add(sub.id)
                    elif isinstance(sub, ast.alias):
                        names.add((sub.asname or sub.name).split(".")[0])
            self._toplevel = names
        return self._toplevel


def contains(node, a, b):
    """Does node's span contain [a, b] (b may equal a for a point)?"""
    if not has_pos(node):
        return False
    s, e = start(node), end(node)
    if a == b:
        return s <= a < e
    return s <= a and b <= e


def path_to(tree, a, b):
    """Nodes from the module down to the deepest one containing [a, b]."""
    path = [tree]
    node = tree
    while True:
        nxt = None
        for child in ast.iter_child_nodes(node):
            if contains(child, a, b):
                nxt = child
                break
            if not has_pos(child):
                # arguments, comprehension, ...: look inside
                for g in ast.iter_child_nodes(child):
                    if contains(g, a, b):
                        nxt = g
                        break
                if nxt is not None:
                    path.append(child)
                    break
        if nxt is None:
            return path
        path.append(nxt)
        node = nxt


def in_body(node, a, b):
    if isinstance(node, ast.Lambda):
        return contains(node.body, a, b) or (a == b and start(node.body) <= a <= end(node.body))
    r = body_range(node)
    if r is None:
        return False
    return r[0] <= a and b <= r[1] if a != b else r[0] <= a <= r[1]


def qualname(path, upto):
    parts = []
    for node in path[: upto + 1]:
        if isinstance(node, ast.ClassDef):
            parts.append(node.name)
        elif isinstance(node, FUNCS):
            parts.append(node.name)
            parts.append("<locals>")
    if parts and parts[-1] == "<locals>":
        parts.pop()
    return ".".join(parts)


def classify(tree, a, b):
    """Where does the span [a, b] live?

    ("runtime", path, idx)      inside the body of path[idx] (def or lambda)
    ("def", qualname, name)     in a def header (decorators, signature, name)
    ("class", path, idx)        in the body of class path[idx], outside any def
    ("module", path)            at module level
    """
    path = path_to(tree.tree, a, b)
    runtime = None
    for i, node in enumerate(path):
        if isinstance(node, FUNCS + (ast.Lambda,)) and in_body(node, a, b):
            runtime = i
    if runtime is not None:
        return ("runtime", path, runtime)
    for i in range(len(path) - 1, -1, -1):
        node = path[i]
        if isinstance(node, FUNCS):
            return ("def", qualname(path, i), node.name, path, i)
        if isinstance(node, ast.ClassDef) and in_body(node, a, b):
            return ("class", path, i)
    return ("module", path)


def overlapping(stmts, a, b):
    """Statements overlapping [a, b]; for a point, the ones it touches."""
    out = []
    for s in stmts:
        if not has_pos(s):
            continue
        if a == b:
            if start(s) <= a <= end(s):
                out.append(s)
        elif start(s) < b and a < end(s):
            out.append(s)
    return out


# ------------------------------------------------------------------ analysis

class Selector:
    def __init__(self, tests, m):
        self.tests = tests
        self.m = m
        self.old_cache = {}

    def old_text(self, f):
        p = SRC / f
        return p.read_text(encoding="utf-8") if p.exists() else None

    def old_tree(self, f, text):
        t = self.old_cache.get(f)
        if t is None:
            t = self.old_cache[f] = Tree(text)
        return t

    # -- one commit
    def select(self, patch_text):
        """(must, maybe, files) for one patch.

        must   tests to select no matter what: readers of changed non-Python
               files, broken or deleted modules, anything we cannot analyse
        maybe  tests reached through changed Python code, which running the
               patched code can confirm or rule out
        files  path -> patched text of every changed Python file
        """
        self.must = 0
        self.files = {}
        try:
            diffs = parse_patch(patch_text)
        except Exception as e:  # noqa: BLE001
            log(f"  unparseable patch ({e}); selecting everything")
            return self.m.all, 0, {}
        if not diffs:
            log("  empty patch")
            return 0, 0, {}
        maybe = 0
        for d in diffs:
            try:
                maybe |= self.file_mask(d)
            except Exception as e:  # noqa: BLE001
                log(f"  {d.old or d.new}: analysis failed ({e!r}); selecting everything")
                self.must |= self.m.all
        return self.must, maybe & ~self.must, self.files

    def file_mask(self, d):
        m = self.m
        paths = [p for p in (d.old, d.new) if p]
        for p in paths:
            base = os.path.basename(p)
            if p in ALL_FILES or base == "conftest.py":
                log(f"  {p}: test configuration; selecting everything")
                self.must |= m.all
                return 0
        # Readers of either path, and listers of the directories it is in.
        for p in paths:
            i = m.opens.get(p)
            if i is not None:
                self.must |= m.bits(i)
            for tf in m.collect_opens.get(p, ()):
                self.must |= m.all if tf == "?" else TESTS_BY_FILE.get(tf, 0)
        if d.old != d.new or d.old is None or d.new is None:
            for p in paths:
                parent = os.path.dirname(p)
                while True:
                    i = m.dirs.get(parent or ".")
                    if i is not None:
                        self.must |= m.bits(i)
                    if not parent:
                        break
                    parent = os.path.dirname(parent)
        if d.binary:
            return 0

        py = [p for p in paths if p.endswith(".py") and (p.startswith("sqlglot/") or p.startswith("tests/"))]
        if not py:
            return 0

        if d.old is None:
            # A new module runs only if something imports it, and that
            # importer's change is analysed on its own.
            log(f"  {d.new}: new file")
            return 0
        if d.new is None or d.old != d.new:
            log(f"  {d.old}: deleted or renamed module")
            self.must |= m.module_impact(d.old)
            return 0

        f = d.old
        old_text = self.old_text(f)
        if old_text is None:
            raise ValueError("old file missing from source tree")
        old_lines = old_text.split("\n")
        new_lines, groups = apply_hunks(old_lines, d.hunks)
        new_text = "\n".join(new_lines)
        old = self.old_tree(f, old_text)
        try:
            new = Tree(new_text)
        except SyntaxError as e:
            log(f"  {f}: no longer parses ({e.msg}); module impact")
            self.must |= m.module_impact(f)
            return 0
        self.files[f] = new_text
        if ast.dump(old.tree) == ast.dump(new.tree):
            log(f"  {f}: same syntax tree (comments or formatting)")
            return 0
        mask = 0
        for g in groups:
            mask |= self.group_mask(f, old, new, old_lines, new_lines, g)
        return mask

    def group_mask(self, f, old, new, old_lines, new_lines, group):
        m = self.m
        o0, o1, n0, n1 = group
        old_chunk = old_lines[o0 - 1 : o1 - 1]
        new_chunk = new_lines[n0 - 1 : n1 - 1]
        if all(is_trivial(x) for x in old_chunk + new_chunk) and not in_string(old, o0) and not in_string(new, n0):
            return 0
        a_old, b_old, a_new, b_new = narrow(old_chunk, new_chunk, o0, n0)
        mask = 0
        mask |= self.side_mask(f, old, new, a_old, b_old, "old", (a_old, b_old))
        mask |= self.side_mask(f, old, new, a_new, b_new, "new", (a_old, b_old))
        return mask

    def side_mask(self, f, old, new, a, b, side, old_span):
        """Tests reachable through the span [a, b] of the old or new tree."""
        m = self.m
        tree = old if side == "old" else new
        if side == "new" and a == b:
            return 0  # pure deletion: the old side covers it
        c = classify(tree, a, b)
        kind = c[0]
        if kind == "runtime":
            _, path, idx = c
            scope = path[idx]
            q = qualname(path, idx - 1) if isinstance(scope, ast.Lambda) else qualname(path, idx)
            if side == "old":
                lines = runtime_lines(path, idx, a, b)
                return self.lines_mask(f, lines)
            # New side: find the same function in the old tree and use the old
            # statements around where the new code went in.
            return self.runtime_in_old(f, old, q, scope, old_span)
        if kind == "def":
            _, q, name, path, idx = c
            if side == "old":
                return self.func_mask(f, old, q)
            if q in old.funcs():
                return self.func_mask(f, old, q)
            return self.new_def_mask(f, old, path, idx, name)
        if kind == "class":
            _, path, idx = c
            cls = path[idx]
            cq = qualname(path, idx)
            stmts = overlapping(child_stmts(cls), a, b)
            mask = 0
            for s in stmts:
                if isinstance(s, FUNCS):
                    sq = f"{cq}.{s.name}" if cq else s.name
                    if side == "old" or sq in old.funcs():
                        mask |= self.func_mask(f, old, sq)
                    else:
                        mask |= m.class_users(f, cq)
                elif isinstance(s, ast.ClassDef):
                    mask |= m.class_users(f, f"{cq}.{s.name}") | m.class_users(f, cq)
                elif may_raise(path, s):
                    # Class bodies run at import: if this one now raises, the
                    # whole module fails to import.
                    mask |= m.module_impact(f)
                else:
                    mask |= self.attr_mask(f, cq, s, a, b)
            if not stmts and side == "new":
                mask |= self.class_mask(f, cq)
            return mask
        # module level
        stmts = overlapping(tree.tree.body, a, b)
        mask = 0
        for s in stmts:
            if isinstance(s, FUNCS):
                if side == "old" or s.name in old.funcs():
                    mask |= self.func_mask(f, old, s.name)
                elif s.name in old.toplevel_names():
                    mask |= m.module_impact(f)
            elif isinstance(s, ast.ClassDef):
                if side == "old" or s.name in old.toplevel_names():
                    mask |= m.class_users(f, s.name) | m.eager_tests(f)
            else:
                mask |= m.module_impact(f)
            if side == "old" and a <= start(s) and end(s) <= b:
                # A whole top-level def or class went away: importers of the
                # name break at import time.
                mask |= m.module_impact(f)
        if not stmts and side == "new":
            mask |= m.module_impact(f)
        return mask

    def lines_mask(self, f, lines):
        m = self.m
        return m.line_users(f, lines) | m.import_impact(f, lines)

    def func_mask(self, f, old, q):
        r = old.funcs().get(q)
        if r is None:
            return self.m.module_impact(f)
        lines = range(r[0], r[1] + 1)
        return self.lines_mask(f, lines)

    def class_mask(self, f, cq):
        m = self.m
        return m.class_users(f, cq) | m.eager_tests(f)

    def attr_mask(self, f, cq, s, a, b):
        """A class-body statement: if it just sets data attributes, only the
        tests that read them (or the dict keys that changed) can see it."""
        m = self.m
        ckey = f"{f}::{cq}"
        names = assigned_names(s)
        if not names or ckey not in m.tracked or names & m.derived:
            return self.class_mask(f, cq)
        toks = None
        if len(names) == 1 and isinstance(getattr(s, "value", None), ast.Dict):
            toks = dict_entry_tokens(s.value, a, b)
        mask = m.eager_tests(f)
        for n in names:
            if toks is None:
                mask |= m.attr_users(ckey, n)
            else:
                mask |= m.key_users(ckey, n, "*")
                for t in toks:
                    mask |= m.key_users(ckey, n, t)
        return mask

    def new_def_mask(self, f, old, path, idx, name):
        m = self.m
        # A def that did not exist: it may override a method for every user of
        # the enclosing class, or shadow a module-level name.
        for i in range(idx - 1, -1, -1):
            if isinstance(path[i], ast.ClassDef):
                return m.class_users(f, qualname(path, i))
            if isinstance(path[i], FUNCS):
                return self.func_mask(f, old, qualname(path, i))
        return m.module_impact(f) if name in old.toplevel_names() else 0

    def runtime_in_old(self, f, old, q, scope, old_span):
        a, b = old_span
        if isinstance(scope, ast.Lambda):
            # Lambdas have no name; use the old statement at the same spot.
            c = classify(old, a, b)
            if c[0] == "runtime":
                return self.lines_mask(f, runtime_lines(c[1], c[2], a, b))
            return self.side_mask(f, old, old, a, b, "old", old_span) or self.m.module_impact(f)
        r = old.funcs().get(q)
        if r is None:
            # A brand-new function; its callers are changed elsewhere in the
            # patch, or it is a method another module may now resolve to.
            return 0
        c = classify(old, a, b)
        if c[0] == "runtime" and qualname(c[1], c[2]) == q:
            return self.lines_mask(f, runtime_lines(c[1], c[2], a, b))
        # The insertion point fell just outside the old function (e.g. new
        # lines at the very end of it): use its last and first statements.
        node = find_func(old.tree, q)
        if node is None:
            return self.func_mask(f, old, q)
        kids = flat_stmts(node)
        near = [s for s in kids if end(s) <= a][-1:] + [s for s in kids if start(s) >= b][:1]
        if not near:
            return self.func_mask(f, old, q)
        lines = set()
        for s in near:
            lo, hi = stmt_lines(s)
            lines.update(range(lo, hi + 1))
        return self.lines_mask(f, lines)


def may_raise(path, s):
    """Does the change at the end of `path` (inside statement s) touch an
    operation that could now raise when the statement runs, like `+` turned
    into `-` on two lists? A change inside a literal cannot."""
    if not path or path[-1] is s or s not in path:
        return False
    node = path[-1]
    if isinstance(node, ast.UnaryOp):
        return not isinstance(node.operand, ast.Constant)
    return isinstance(node, (ast.BinOp, ast.BoolOp, ast.Compare, ast.Subscript))


def assigned_names(s):
    """Names a plain class-level assignment binds, or None for anything else."""
    if isinstance(s, ast.Assign):
        targets = s.targets
    elif isinstance(s, ast.AnnAssign):
        targets = [s.target]
    else:
        return None
    names = set()
    for t in targets:
        if not isinstance(t, ast.Name):
            return None
        names.add(t.id)
    return names


def static_key(node):
    """The token a dict key is recorded under at runtime (str, class or enum
    member name, number), or None when it cannot be known statically."""
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, str):
            return v
        if isinstance(v, (int, float, bool)) or v is None:
            return repr(v)
        return None
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def dict_entry_tokens(d, a, b):
    """Keys of the entries of dict display d that the span [a, b] touches, or
    None if it touches the dict's structure or an entry with a dynamic key."""
    if not (start(d) < a and b < end(d)):
        return None
    toks = set()
    for k, v in zip(d.keys, d.values):
        lo = start(k) if k is not None else start(v)
        hi = end(v)
        if (a == b and lo <= a <= hi) or (a != b and lo < b and a < hi):
            if k is None:
                return None
            tok = static_key(k)
            if tok is None:
                return None
            toks.add(tok)
    return toks


def find_func(tree, q):
    parts = q.replace(".<locals>", "").split(".")
    node = tree
    for name in parts:
        for child in ast.walk(node):
            if child is not node and isinstance(child, FUNCS + (ast.ClassDef,)) and child.name == name:
                node = child
                break
        else:
            return None
    return node if isinstance(node, FUNCS) else None


def flat_stmts(node):
    out = []
    for s in child_stmts(node):
        out.append(s)
        if not isinstance(s, FUNCS + (ast.ClassDef,)):
            out.extend(flat_stmts(s))
    return sorted(out, key=start)


def runtime_lines(path, idx, a, b):
    """Old lines whose execution decides whether a test sees the change at
    [a, b] inside the function or lambda path[idx]."""
    scope = path[idx]
    if isinstance(scope, ast.Lambda):
        return range(scope.body.lineno, scope.body.end_lineno + 1)
    deep = path[idx + 1 :]
    # Innermost statement containing the whole span.
    inner = None
    for i, node in enumerate(deep):
        if isinstance(node, (ast.stmt, ast.excepthandler, ast.match_case)):
            inner = (i, node)
    lines = set()
    if inner is not None:
        i, node = inner
        if not in_body(node, a, b):
            # The change is inside this statement (or its header).
            lo, hi = stmt_lines(node)
            lines.update(range(lo, hi + 1))
            return lines
        owner = node
    else:
        owner = scope
    # Between or across the statements of owner's body: the statements it
    # touches and the ones on either side of it.
    kids = child_stmts(owner)
    touched = overlapping(kids, a, b)
    before = [s for s in kids if end(s) <= a and s not in touched][-1:]
    after = [s for s in kids if start(s) >= b and s not in touched][:1]
    for s in touched + before + after:
        lo, hi = stmt_lines(s)
        lines.update(range(lo, hi + 1))
    if not lines:
        lo, hi = stmt_lines(owner)
        lines.update(range(lo, hi + 1))
    return lines


def is_trivial(line):
    s = line.strip()
    return not s or s.startswith("#")


def in_string(tree, line):
    """Is this line inside a multi-line string literal?"""
    for node in ast.walk(tree.tree):
        if (
            isinstance(node, (ast.Constant, ast.JoinedStr))
            and has_pos(node)
            and node.lineno < line <= node.end_lineno
            and (isinstance(node, ast.JoinedStr) or isinstance(node.value, (str, bytes)))
        ):
            return True
    return False


def narrow(old_chunk, new_chunk, o0, n0):
    """Trim the common prefix and suffix of a change group; return the changed
    spans as ((line, col), (line, col)) in old and new coordinates."""
    old = "\n".join(old_chunk)
    new = "\n".join(new_chunk)
    if not old_chunk:
        p = (o0, 0)
        q0 = (n0, 0)
        q1 = (n0 + len(new_chunk) - 1, len(new_chunk[-1])) if new_chunk else q0
        return p, p, q0, q1
    if not new_chunk:
        p0 = (o0, 0)
        p1 = (o0 + len(old_chunk) - 1, len(old_chunk[-1]))
        q = (n0, 0)
        return p0, p1, q, q
    pre = 0
    lim = min(len(old), len(new))
    while pre < lim and old[pre] == new[pre]:
        pre += 1
    suf = 0
    while suf < lim - pre and old[-1 - suf] == new[-1 - suf]:
        suf += 1

    def pos(text, base, offset):
        line = text.count("\n", 0, offset)
        col = offset - (text.rfind("\n", 0, offset) + 1)
        return (base + line, col)

    a_old, b_old = pos(old, o0, pre), pos(old, o0, len(old) - suf)
    a_new, b_new = pos(new, n0, pre), pos(new, n0, len(new) - suf)
    # Columns from the AST are UTF-8 byte offsets; fine for ASCII source,
    # and a mismatch only widens what we select.
    return a_old, b_old, a_new, b_new


# --------------------------------------------------------------- the workers

def split(numbers, ms):
    """Pick the worker count and assignment with the lowest cost."""
    if not numbers:
        return []
    order = sorted(numbers, key=lambda n: -ms[n])
    best = None
    for k in range(1, MAX_WORKERS + 1):
        if k > len(order):
            break
        loads = [0] * k
        workers = [[] for _ in range(k)]
        for n in order:
            j = loads.index(min(loads))
            workers[j].append(n)
            loads[j] += ms[n]
        cost = max(loads) + WORKER_MS * k
        if best is None or cost < best[0]:
            best = (cost, workers)
    return best[1]


# ------------------------------------------------------------ verification
#
# Listing a test costs its seconds on one of up to 8 parallel workers; running
# it here costs its seconds of CPU. So we only run the cheap candidates, and
# only when the best outcome (they all pass) beats listing everything.

HEAVY_MS = int(os.environ.get("BR_HEAVY_MS", 1500))  # never run these here
VERIFY_MAX_MS = int(os.environ.get("BR_VERIFY_MAX_MS", 4000))
OVERHEAD_MS = int(os.environ.get("BR_OVERHEAD_MS", 500))  # per verification
# Drop the heavy candidates when every cheap test reaching the same code passes.
BET = os.environ.get("BR_BET", "1") == "1"
VERIFY = os.environ.get("BR_VERIFY", "0") == "1"  # off until CPU and the bet are validated
SIM_ANSWERS = os.environ.get("BR_SIM_ANSWERS")  # tune offline: labels stand in for runs
PARALLEL = 2


def numbers_of(mask, n):
    return [i for i in range(n) if mask >> i & 1]


def cost_of(workers, ms):
    return (max(sum(ms[n] for n in w) for w in workers) + WORKER_MS * len(workers)) if workers else 0


def main():
    tests = json.loads((DATA / "tests.json").read_text())["tests"]
    n = len(tests)
    ms = {t["number"]: round(t["seconds"] * 1000) for t in tests}
    ids = {t["number"]: t["id"] for t in tests}
    for t in tests:
        TESTS_BY_FILE[t["file"]] = TESTS_BY_FILE.get(t["file"], 0) | (1 << t["number"])
    heavy = sum(1 << t["number"] for t in tests if ms[t["number"]] >= HEAVY_MS)
    m = Map(MAP, n)
    sel = Selector(tests, m)

    plans = []
    for patch in sorted((DATA / "commits").glob("*.patch")):
        log(f"{patch.stem}:")
        must, maybe, files = sel.select(patch.read_text(encoding="utf-8", errors="replace"))
        light = maybe & ~heavy
        hv = maybe & heavy
        plan = {"commit": patch.stem, "must": must, "light": light, "heavy": hv, "files": files,
                "verify": False}
        if VERIFY and light:
            sum_l = sum(ms[i] for i in numbers_of(light, n))
            listed = cost_of(split(numbers_of(must | maybe, n), ms), ms)
            keep = must if BET else must | hv
            best = OVERHEAD_MS + sum_l + cost_of(split(numbers_of(keep, n), ms), ms)
            if sum_l <= VERIFY_MAX_MS and best < listed:
                plan["verify"] = True
                plan["sum_l"] = sum_l
        plans.append(plan)

    todo = [p for p in plans if p["verify"]]
    est_cpu = 0
    if todo and SIM_ANSWERS:
        answers = json.load(open(SIM_ANSWERS))["commits"]
        for p in todo:
            failing = set(answers.get(p["commit"], {}).get("failing", ()))
            p["failed"] = sum(1 << i for i in numbers_of(p["light"], n) if i in failing)
            est_cpu += OVERHEAD_MS + p["sum_l"]
    elif todo:
        sys.path.insert(0, str(HERE))  # python -I leaves the script's folder off the path
        import verify

        verify.warm_up(SRC)
        queue = list(todo)
        running = []
        while queue or running:
            while queue and len(running) < PARALLEL:
                p = queue.pop(0)
                nodeids = [ids[i] for i in numbers_of(p["light"], n)]
                running.append((p, verify.start(p["files"], nodeids),
                                10 + 5 * p["sum_l"] / 1000))
            still = []
            for p, h, timeout in running:
                failed = verify.poll(h, timeout)
                if failed is None:
                    still.append((p, h, timeout))
                else:
                    num = {v: k for k, v in ids.items()}
                    p["failed"] = sum(1 << num[x] for x in failed if x in num)
            running = still
            if running:
                time.sleep(0.01)

    total = 0
    for p in plans:
        if p["verify"]:
            failed = p["failed"]
            keep_heavy = p["heavy"] if (failed or not BET) else 0
            mask = p["must"] | failed | keep_heavy
            log(f"{p['commit']}: verified {bin(p['light']).count('1')} cheap tests, "
                f"{bin(failed).count('1')} failed; heavy {'kept' if keep_heavy else 'dropped'}")
        else:
            mask = p["must"] | p["light"] | p["heavy"]
        workers = split(numbers_of(mask, n), ms)
        total += cost_of(workers, ms)
        print(json.dumps({"commit": p["commit"], "workers": workers}), flush=True)
    import resource

    ru = resource.getrusage(resource.RUSAGE_SELF)
    rc = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = ru.ru_utime + ru.ru_stime + rc.ru_utime + rc.ru_stime
    log(f"verified {len(todo)} commits; estimated simulated seconds (no misses): {total / 1000:.1f}")
    log(f"selector CPU: {cpu + est_cpu / 1000:.2f} s" + (" (simulated)" if SIM_ANSWERS else ""))


if __name__ == "__main__":
    main()
