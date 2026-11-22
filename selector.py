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
TPCDS_MAP = Path(os.environ.get("BR_TPCDS_MAP", HERE / "maps" / "tpcds.json.gz"))
TPCDS = os.environ.get("BR_TPCDS", "1") == "1"
# Share of plan-phase-only patches whose TPC-DS plans come out identical
# (about 0.4 on the local benches): what a check is expected to save.
TPCDS_HIT_RATE = float(os.environ.get("BR_TPCDS_HIT_RATE", 0.4))
# CPU seconds (measured at the pinned commit) worth spending on running
# queries whose plans changed, to keep test_execute_tpcds (21 s) off the list.
TPCDS_RUN_BUDGET = float(os.environ.get("BR_TPCDS_RUN_BUDGET", 5))

MAX_WORKERS = 8
START = time.monotonic()
# No check starts after this many seconds of wall time (the batch must finish
# in 3 minutes, 4.5 in the final); unchecked commits keep full selections.
CHECK_DEADLINE = float(os.environ.get("BR_CHECK_DEADLINE", 120))
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
        # Only the lines that function or lambda code ran while importing: a
        # change inside a body has an import-time effect only if those ran.
        self.import_func_lines = m.get("import_func_lines", self.import_lines)
        self.importers = m["importers"]
        self.direct_importers = m.get("direct_importers", {})
        # Light mode: assume every module still imports (an import check run
        # on the patched code confirms it), so a module-level change reaches
        # the users of that module and of its direct importers, not every
        # module that imports it transitively (which is nearly all of them).
        self.light = False
        self.light_used = False
        self.attrs = m.get("attrs", {})
        self.keys = m.get("keys", {})
        self.derived = set(m.get("derived", ()))
        self.tracked = set(m.get("tracked", ()))
        self.eager = set(m.get("eager", ()))
        self.import_users = m.get("import_users", {})
        self.import_reads = m.get("import_reads", {})
        self.conds = m.get("conds", {})
        self.cond_sites = {f: set(v) for f, v in m.get("cond_sites", {}).items()}
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
        if self.light:
            self.light_used = True
            m = self.module_users(f) | self.eager_tests(f)
            for g in self.direct_importers.get(f, ()):
                m |= self.module_users(g)
            return m
        return self.module_impact_full(f)

    def module_impact_full(self, f):
        """Every test that imports f, directly or not: f may fail to import."""
        m = self.module_users(f) | self.eager_tests(f)
        for g in self.importers.get(f, ()):
            m |= self.module_users(g)
        return m

    def import_read(self, ckey, name, toks=None):
        """Modules that read this class attribute (or these keys of it) while
        being imported: the change can make their import fail, or change what
        they built from it."""
        reads = self.import_reads.get(ckey, {}).get(name)
        if not reads:
            return 0
        mods = set()
        for t in ([""] if toks is None else ["*", *toks]):
            mods.update(reads.get(t, ()))
        if toks is None:
            for v in reads.values():
                mods.update(v)
        m = 0
        for mod in mods:
            m |= self.module_impact(mod)
        return m

    def import_read_any(self, ckey):
        m = 0
        for name in self.import_reads.get(ckey, {}):
            m |= self.import_read(ckey, name)
        return m

    def eager_tests(self, f):
        """Tests that run `import sqlglot` in a subprocess (we cannot see
        inside it): reachable by import-time changes to eagerly loaded modules."""
        return self.subprocess if f in self.eager else 0

    def import_impact(self, f, lines):
        """A change inside function bodies on these lines: the modules whose
        import ran them may now build something else, or fail."""
        imp = self.import_func_lines.get(f)
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

# TestExecutor's setUpClass parses these SQL fixtures; the tests that use
# the pairs, and which part of each pair (see tests/test_executor.py at the
# pinned commit). Every other TestExecutor test only needs them to parse.
EXECUTOR_FIXTURES = {
    "tests/fixtures/optimizer/tpc-ds/tpc-ds.sql": {
        "tests/test_executor.py::TestExecutor::test_execute_tpcds": "inputs",
    },
    "tests/fixtures/optimizer/tpc-h/tpc-h.sql": {
        "tests/test_executor.py::TestExecutor::test_execute_tpch": "inputs",
        "tests/test_executor.py::TestExecutor::test_optimized_tpch": "all",
    },
}
EXECUTOR = {}  # TestExecutor test id -> number


def sql_fixture_pairs(text):
    """tests/helpers.py's load_sql_fixture_pairs, on a text."""
    s = "\n".join([line for line in text.splitlines() if line and not line.startswith("--")])
    statements = s.split(";")
    for i in range(0, len(statements), 2):
        if i + 1 < len(statements):
            sql = statements[i].strip()
            meta = {}
            sql_lines = sql.split("\n")
            j = 0
            while sql_lines[j].startswith("#"):
                key, val = sql_lines[j].split(":", maxsplit=1)
                meta[key.lstrip("#").strip()] = val.strip()
                j += 1
            yield meta, "\n".join(sql_lines[j:]), statements[i + 1].strip()


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
    # A decorated def or class starts at its first decorator: Python puts its
    # lineno on the `def`/`class` line, which would leave decorator lines
    # outside every statement.
    decorators = getattr(node, "decorator_list", None)
    if decorators:
        first = min(decorators, key=lambda d: (d.lineno, d.col_offset))
        return min((node.lineno, node.col_offset), (first.lineno, first.col_offset - 1))
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
        self._dump = None
        self._strings = None

    def dump(self):
        if self._dump is None:
            self._dump = ast.dump(self.tree)
        return self._dump

    def funcs(self):
        """qualname -> (first line incl. decorators, last line) of every def."""
        if self._funcs is None:
            self._funcs = {}

            def walk(node, prefix):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, FUNCS):
                        q = prefix + child.name
                        lo = min([child.lineno] + [d.lineno for d in child.decorator_list])
                        self._funcs[q] = (lo, child.end_lineno, child.body[0].lineno)
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
        self.new_cache = {}

    def old_text(self, f):
        p = SRC / f
        return p.read_text(encoding="utf-8") if p.exists() else None

    def old_tree(self, f, text):
        t = self.old_cache.get(f)
        if t is None:
            t = self.old_cache[f] = Tree(text)
        return t

    # -- one commit
    def select(self, patch_text, light=False):
        """(must, maybe, files) for one patch.

        must   tests to select no matter what: readers of changed non-Python
               files, broken or deleted modules, anything we cannot analyse
        maybe  tests reached through changed Python code, which running the
               patched code can confirm or rule out
        files  path -> patched text of every changed Python file
        """
        self.m.light, self.m.light_used = light, False
        self.state_check = False  # the light selection needs the import check's state diff
        # A change that may make an import fail: only the import check can tell
        # which tests that breaks (dialects are loaded by name, which no static
        # import map sees), so it always runs.
        self.import_risk = False
        self.must = 0
        self.files = {}
        self.deleted = set()  # Python files the patch deletes or renames away
        self.touched = set()  # (file, line): old lines a change inside function bodies touches
        # Changes that are not just lines inside function bodies, by file:
        # "class" (class data attributes), "module" (module-level imports and
        # assignments), "def" (an existing def's header or decorators) and
        # "newdef" (defs or classes added, removed or reshaped).
        self.nonline = {}
        self.module_names = {}  # file -> names its module-level changes bind (None: unknown)
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
                self.must |= self.fixture_readers(d, p, m.bits(i))
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

        if d.old != d.new and d.old is not None:
            self.deleted.add(d.old)
        if d.new is not None and d.new != d.old and d.new.endswith(".py"):
            # Served to the checks that import the patched code.
            self.files[d.new] = "\n".join(apply_hunks([], d.hunks)[0]) if d.old is None else None
        if d.old is None:
            # A new module runs only if something imports it, and that
            # importer's change is analysed on its own.
            log(f"  {d.new}: new file")
            return 0
        if d.new is None or d.old != d.new:
            log(f"  {d.old}: deleted or renamed module")
            self.import_risk = True
            self.must |= m.module_impact_full(d.old)
            if d.new is not None:
                old_text = self.old_text(d.old)
                if old_text is None:
                    raise ValueError("old file missing from source tree")
                self.files[d.new] = "\n".join(apply_hunks(old_text.split("\n"), d.hunks)[0])
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
            # Both selection passes of a commit see the same patched text.
            new = self.new_cache.get((f, new_text))
            if new is None:
                new = Tree(new_text)
                self.new_cache = {(f, new_text): new, **{k: v for k, v in self.new_cache.items() if k[0] != f}}
        except SyntaxError as e:
            log(f"  {f}: no longer parses ({e.msg}); module impact")
            self.import_risk = True
            self.must |= m.module_impact_full(f)
            return 0
        self.files[f] = new_text
        self.trees = (old, new)
        if old.dump() == new.dump():
            log(f"  {f}: same syntax tree (comments or formatting)")
            return 0
        if strings_only(old, new, old_lines, new_lines, groups) and without_strings(old_text) == without_strings(new_text):
            # Only docstrings (or other bare string statements) changed: no
            # collected test reads __doc__ (pytest does not run the doctests).
            log(f"  {f}: only docstrings changed")
            return 0
        mask = 0
        for g in groups:
            mask |= self.group_mask(f, old, new, old_lines, new_lines, g)
        return mask

    def fixture_readers(self, d, p, readers):
        """Readers of a changed file, less the executor tests that cannot see
        this change (see EXECUTOR_FIXTURES)."""
        uses = EXECUTOR_FIXTURES.get(p)
        if uses is None or d.old != d.new or d.binary:
            return readers
        try:
            old = list(sql_fixture_pairs(self.old_text(p)))
            new = list(sql_fixture_pairs("\n".join(apply_hunks(self.old_text(p).split("\n"), d.hunks)[0])))
        except Exception:  # noqa: BLE001  (setUpClass would fail too)
            return readers
        same_inputs = [x[:2] for x in old] == [x[:2] for x in new]
        same_all = old == new
        for test, num in EXECUTOR.items():
            part = uses.get(test)
            if part is None or same_all or (same_inputs and part == "inputs"):
                readers &= ~(1 << num)
        log(f"  {p}: {'unchanged pairs' if same_all else 'unchanged inputs' if same_inputs else 'inputs changed'}")
        return readers

    def group_mask(self, f, old, new, old_lines, new_lines, group):
        m = self.m
        o0, o1, n0, n1 = group
        old_chunk = old_lines[o0 - 1 : o1 - 1]
        new_chunk = new_lines[n0 - 1 : n1 - 1]
        if all(is_trivial(x) for x in old_chunk + new_chunk) and not in_string(old, o0) and not in_string(new, n0):
            return 0
        a_old, b_old, a_new, b_new = narrow(old_chunk, new_chunk, o0, n0)
        if all(x.isascii() for x in old_chunk + new_chunk):
            refined = self.cond_change(f, old, new, a_old, b_old, a_new, b_new)
            if refined is not None:
                return refined
        mask = 0
        mask |= self.side_mask(f, old, new, a_old, b_old, "old", (a_old, b_old))
        mask |= self.side_mask(f, old, new, a_new, b_new, "new", (a_old, b_old))
        return mask

    def cond_change(self, f, old, new, a_old, b_old, a_new, b_new):
        """A change that only swaps the operator of one `and`/`or` or one
        comparison whose outcomes the map recorded: the tests whose recorded
        evaluations could come out differently. None: analyse it as usual."""
        m = self.m
        sites = m.cond_sites.get(f)
        if not sites:
            return None
        found_old = find_op_node(old.tree, a_old, b_old)
        found_new = find_op_node(new.tree, a_new, b_new)
        if found_old is None or found_new is None:
            return None
        onode, opath = found_old
        nnode, _ = found_new
        key = f"{onode.lineno}:{onode.col_offset}:{onode.end_lineno}:{onode.end_col_offset}"
        if key not in sites or type(onode) is not type(nnode):
            return None
        if isinstance(onode, ast.BoolOp):
            if len(onode.values) != len(nnode.values) or any(
                ast.dump(x) != ast.dump(y) for x, y in zip(onode.values, nnode.values)
            ):
                return None
            old_and, new_and = isinstance(onode.op, ast.And), isinstance(nnode.op, ast.And)
            if old_and == new_and:
                return None
            truth = truth_only(opath, onode)

            def differs(code):
                return boolop_differs(code, len(onode.values), old_and, new_and, truth)
        else:
            if len(onode.ops) != 1 or len(nnode.ops) != 1:
                return None
            if ast.dump(onode.left) != ast.dump(nnode.left) or ast.dump(onode.comparators[0]) != ast.dump(
                nnode.comparators[0]
            ):
                return None
            oop, nop = type(onode.ops[0]).__name__, type(nnode.ops[0]).__name__
            if oop == nop:
                return None

            def differs(code):
                return compare_differs(code, oop, nop)

        self.touched.update((f, ln) for ln in range(onode.lineno, onode.end_lineno + 1))
        mask = 0
        for code, i in m.conds.get(f, {}).get(key, {}).items():
            if differs(code):
                mask |= m.bits(i)
        # Evaluations while modules import are not recorded per test.
        mask |= m.import_impact(f, range(onode.lineno, onode.end_lineno + 1))
        log(f"  {f}:{onode.lineno}: operator change, {bin(mask).count('1')} tests see a different outcome")
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
                mask = self.lines_mask(f, lines)
                if q in ("__getattr__", "__dir__"):
                    # A module's lazy attribute hook (PEP 562): its results are
                    # cached in the module, so later importers never run it.
                    mask |= m.module_impact_full(f)
                return mask
            # New side: find the same function in the old tree and use the old
            # statements around where the new code went in.
            mask = self.runtime_in_old(f, old, q, scope, old_span)
            if q in ("__getattr__", "__dir__"):
                mask |= m.module_impact_full(f)
            return mask
        if kind == "def":
            _, q, name, path, idx = c
            # Decorators and defaults run at import.
            risk = self.risky(f) if may_raise(path) else 0
            if side == "old":
                return self.func_mask(f, old, q) | risk
            if q in old.funcs():
                return self.func_mask(f, old, q) | risk
            return self.new_def_mask(f, old, path, idx, name) | risk
        if kind == "class":
            _, path, idx = c
            cls = path[idx]
            cq = qualname(path, idx)
            stmts = overlapping(child_stmts(cls), a, b)
            mask = 0
            for s in stmts:
                if isinstance(s, FUNCS):
                    # A def added, removed or reshaped in the class body: its
                    # decorators and defaults run when the class is created.
                    sq = f"{cq}.{s.name}" if cq else s.name
                    mask |= self.risky(f)
                    if side == "old" or sq in old.funcs():
                        mask |= self.func_mask(f, old, sq)
                    else:
                        self.mark(f, "newdef")
                        mask |= m.class_users(f, cq)
                elif isinstance(s, ast.ClassDef):
                    self.mark(f, "newdef")
                    mask |= m.class_users(f, f"{cq}.{s.name}") | m.class_users(f, cq) | self.risky(f)
                elif may_raise(path, s):
                    # Class bodies run at import: if this one now raises, the
                    # whole module fails to import.
                    self.mark(f, "class")
                    mask |= self.risky(f) | self.attr_mask(f, cq, s, a, b)
                elif is_bare_string(s):
                    pass  # a docstring
                else:
                    self.mark(f, "class")
                    mask |= self.attr_mask(f, cq, s, a, b)
            if not stmts and side == "new":
                self.mark(f, "newdef")
                mask |= self.class_mask(f, cq)
            return mask
        # module level
        stmts = overlapping(tree.tree.body, a, b)
        mask = 0
        for s in stmts:
            if isinstance(s, FUNCS):
                if may_raise(path_to(tree.tree, a, b)):
                    mask |= self.risky(f)  # decorators and defaults run at import
                if side == "old" or s.name in old.funcs():
                    mask |= self.func_mask(f, old, s.name)
                elif s.name in old.toplevel_names():
                    self.mark(f, "newdef")
                    mask |= self.risky(f) | self.name_mask(f, {s.name}, side, s)
            elif isinstance(s, ast.ClassDef):
                self.mark(f, "newdef")
                mask |= self.risky(f)  # a class header (bases, decorators) runs at import
                if side == "old" or s.name in old.toplevel_names():
                    mask |= m.class_users(f, s.name) | m.eager_tests(f)
            elif is_bare_string(s):
                pass  # a docstring
            else:
                names = bound_names(s)
                self.mark(f, "module", names)
                if may_raise(path_to(tree.tree, a, b)):
                    self.import_risk = True  # module code runs at import
                if m.light and names is not None:
                    # An import or a plain assignment: what can see it is code
                    # that names what it binds (here or anywhere else), and
                    # whatever was built from those names while modules imported.
                    m.light_used = True
                    mask |= self.name_mask(f, names, side, s) | m.eager_tests(f)
                else:
                    mask |= m.module_impact_full(f)
            if side == "old" and a <= start(s) and end(s) <= b and not is_bare_string(s):
                if isinstance(s, FUNCS + (ast.ClassDef,)):
                    self.mark(f, "newdef")
                # A whole top-level statement went away: importers of what it
                # bound break at import time (which the import check rules
                # out in light mode), and code using it breaks when it runs.
                if not m.light:
                    mask |= m.module_impact(f)
                elif isinstance(s, FUNCS + (ast.ClassDef,)):
                    mask |= self.name_mask(f, {s.name}, side, s)
                elif bound_names(s) is None:
                    mask |= m.module_impact_full(f)
        if not stmts and side == "new":
            self.mark(f, "module", None)
            self.import_risk = True
            mask |= m.module_impact_full(f)
        return mask

    def risky(self, f):
        """Code that runs while f imports changed in a way that could raise:
        the light selection leaves that to the import check (which always
        runs then); the full one takes every importer."""
        self.import_risk = True
        if self.m.light:
            self.m.light_used = True
            return 0
        return self.m.module_impact_full(f)

    def mark(self, f, kind, names=()):
        """Record a change that is not just lines in function bodies (see
        select); for "module", the names it binds or None. Returns True."""
        self.nonline.setdefault(f, set()).add(kind)
        if kind == "module":
            if names is None or self.module_names.get(f, set()) is None:
                self.module_names[f] = None
            else:
                self.module_names.setdefault(f, set()).update(names)
        return True

    def name_mask(self, f, names, side, stmt):
        """Tests that run a line using any of `names` from module f: in f
        (other than the changed statement itself), or in another file through
        f's namespace (an import from f, `alias.name`, or the same through a
        module that re-exports f's names, like `exp`), plus the users of
        modules whose import ran such a line."""
        m = self.m
        m.light_used = True
        skip = set(range(stmt.lineno, stmt.end_lineno + 1)) if side == "old" else set()
        mask = 0
        for g, uses in name_references(f, names).items():
            lines = [ln for ln in uses if not (g == f and ln in skip)]
            mask |= m.line_users(g, lines)
            for ln in lines:
                # Used while g imports (module or class code): what g built
                # with it; used in a body: only if that body ran at import.
                imp = (m.import_lines if uses[ln] else m.import_func_lines).get(g, {})
                for mod in imp.get(str(ln), ()):
                    mask |= m.all if mod == "?" else m.module_impact(mod)
        return mask

    def lines_mask(self, f, lines, import_lines=None):
        m = self.m
        self.touched.update((f, ln) for ln in lines)
        return m.line_users(f, lines) | m.import_impact(f, lines if import_lines is None else import_lines)

    def func_mask(self, f, old, q):
        self.mark(f, "def")  # the def statement itself runs at import
        r = old.funcs().get(q)
        if r is None:
            return self.m.module_impact_full(f)
        lines = range(r[0], r[1] + 1)
        mask = self.lines_mask(f, lines, import_lines=range(r[2], r[1] + 1))
        if q in ("__getattr__", "__dir__"):
            # A module's lazy attribute hook (PEP 562): what it resolves is
            # cached in the module, so only the first importer ever ran it.
            mask |= self.m.module_impact_full(f)
        return mask

    def outer_users(self, f, cq):
        """Users of the classes enclosing cq (MySQL for MySQL.Tokenizer): a
        dialect's metaclass copies settings of its nested Tokenizer, Parser and
        Generator into the dialect when it is created."""
        m = self.m
        parts = cq.split(".")
        mask = 0
        for i in range(len(parts) - 1, 0, -1):
            outer = ".".join(parts[:i])
            if outer.endswith("<locals>"):
                break
            mask |= m.class_users(f, outer)
        return mask

    def class_mask(self, f, cq):
        m = self.m
        return (m.class_users(f, cq) | self.outer_users(f, cq) | m.eager_tests(f)
                | m.import_read_any(f"{f}::{cq}"))

    def attr_mask(self, f, cq, s, a, b):
        """A class-body statement: if it just sets data attributes, only the
        tests that read them (or the dict keys that changed) can see it."""
        m = self.m
        ckey = f"{f}::{cq}"
        names = assigned_names(s)
        if not names or ckey not in m.tracked:
            return self.class_mask(f, cq)
        reads = None  # dict key tokens whose readers can see the change
        if len(names) == 1 and isinstance(getattr(s, "value", None), ast.Dict):
            toks = dict_entry_tokens(s.value, a, b)
            if toks is not None:
                # A value changed: readers of those keys and of the values as a
                # whole. Keys came or went: also key iteration and size.
                reads = {"*", *toks}
                if not self.same_keys(cq, next(iter(names))):
                    reads |= {"#keys", "#len", "#ref"}
        if names & m.derived:
            if m.light:
                # What was built from it when classes were created: the import
                # check compares everything importing left behind.
                self.state_check = True
                m.light_used = True
                mask = m.eager_tests(f)
                for n in names:
                    mask |= m.attr_users(ckey, n) if reads is None else key_mask(m, ckey, n, reads)
                return mask
            # Built into something else when classes are created: every user
            # of the class, and modules that read it while importing.
            mask = m.class_users(f, cq) | self.outer_users(f, cq) | m.eager_tests(f)
            for n in names:
                mask |= m.import_read(ckey, n)
            return mask
        mask = m.eager_tests(f)
        for n in names:
            mask |= m.import_read(ckey, n, reads)
            mask |= m.attr_users(ckey, n) if reads is None else key_mask(m, ckey, n, reads)
        return mask

    def same_keys(self, cq, name):
        """Does the dict display assigned to `name` in class cq have the same
        static keys, in the same order, before and after?"""
        def keys(tree):
            node = tree.tree
            for part in cq.split("."):
                node = next((c for c in node.body if isinstance(c, ast.ClassDef) and c.name == part), None)
                if node is None:
                    return None
            for st in node.body:
                if assigned_names(st) == {name} and isinstance(st.value, ast.Dict):
                    ks = [static_key(k) if k is not None else None for k in st.value.keys]
                    return None if None in ks else ks
            return None

        old, new = self.trees
        k = keys(old)
        return k is not None and k == keys(new)

    def new_def_mask(self, f, old, path, idx, name):
        m = self.m
        self.mark(f, "newdef")
        # A def that did not exist: it may override a method for every user of
        # the enclosing class, or shadow a module-level name.
        for i in range(idx - 1, -1, -1):
            if isinstance(path[i], ast.ClassDef):
                return m.class_users(f, qualname(path, i))
            if isinstance(path[i], FUNCS):
                return self.func_mask(f, old, qualname(path, i))
        return self.risky(f) | self.name_mask(f, {name}, "new", None) if name in old.toplevel_names() else 0

    def runtime_in_old(self, f, old, q, scope, old_span):
        a, b = old_span
        if isinstance(scope, ast.Lambda):
            # Lambdas have no name; use the old statement at the same spot.
            c = classify(old, a, b)
            if c[0] == "runtime":
                return self.lines_mask(f, runtime_lines(c[1], c[2], a, b))
            return self.side_mask(f, old, old, a, b, "old", old_span) or self.m.module_impact_full(f)
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


def _inside(node, a, b):
    """Is the span [a, b] strictly inside node (a point at its very start or
    end is an insertion next to it, not inside it)?"""
    if not has_pos(node):
        return False
    if a == b:
        return start(node) < a < end(node)
    return start(node) <= a and b <= end(node)


def find_op_node(tree, a, b):
    """The innermost and/or or single comparison whose operator region holds
    the span [a, b] (not inside one of its operands), with its path."""
    path = path_to(tree, a, b)
    for i in range(len(path) - 1, -1, -1):
        node = path[i]
        if isinstance(node, ast.BoolOp):
            operands = node.values
        elif isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
        else:
            continue
        if any(_inside(x, a, b) for x in operands):
            return None
        return node, path[: i + 1]
    return None


def truth_only(path, node):
    """Is only the truth of this expression used (a condition), not its value?"""
    i = len(path) - 1
    while i > 0:
        parent = path[i - 1]
        if isinstance(parent, (ast.If, ast.While, ast.IfExp, ast.Assert)) and parent.test is node:
            return True
        if isinstance(parent, ast.comprehension) and any(x is node for x in parent.ifs):
            return True
        if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not):
            return True
        if isinstance(parent, ast.BoolOp):
            if parent.values[-1] is not node:
                return True  # only its truth decides where evaluation stops
            node, i = parent, i - 1  # last operand: its value is the parent's value
            continue
        return False
    return False


def boolop_differs(code, n, old_and, new_and, truth):
    """Recorded evaluation `code` (one character per operand that ran: T/F
    bools, 1/0 truth of other values, ? unknown, E raised) of an n-operand
    and/or: does it come out differently with the other operator?"""
    if "E" in code or "?" in code or not code:
        return True
    j = len(code) - 1  # operand whose value the old expression returned
    k = None
    for i in range(n):
        if i >= len(code):
            return True  # the new operator needs an operand that never ran
        t = code[i] in "T1"
        if i == n - 1 or t != new_and:
            k = i
            break
    if truth:
        return (code[j] in "T1") != (code[k] in "T1")
    if k == j:
        return False
    return not (code[j] in "TF" and code[j] == code[k])


def _flag(code, letter):
    i = code.find(letter)
    if i < 0 or i + 1 >= len(code):
        return None
    c = code[i + 1]
    return True if c == "1" else False if c == "0" else None


def compare_outcome(op, f):
    e, l, g, i, n = f["e"], f["l"], f["g"], f["i"], f["n"]

    def neg(x):
        return None if x is None else not x

    def either(x, y):
        if x is True or y is True:
            return True
        if x is False and y is False:
            return False
        return None

    return {
        "Eq": e, "NotEq": neg(e), "Lt": l, "Gt": g, "LtE": either(l, e), "GtE": either(g, e),
        "Is": i, "IsNot": neg(i), "In": n, "NotIn": neg(n),
    }.get(op)


def compare_differs(code, old_op, new_op):
    """Recorded comparison flags (r result, e equal, l less, g greater,
    i identical, n contained; ? unknown): does the new operator give a
    different result?"""
    if code == "E":
        return True
    f = {x: _flag(code, x) for x in "relgin"}
    r = f["r"]
    if r is None:
        return True
    old = compare_outcome(old_op, f)
    new = compare_outcome(new_op, f)
    if new is None or (old is not None and old != r):
        return True
    return new != r


def may_raise(path, s=None):
    """Could the change at the end of `path` make code that runs at import
    raise (a class body, a decorator, a default)? Only a change confined to
    one literal constant cannot: a different name, attribute, operator, call
    argument (t.cast(list[str], X) swapped), or an entry removed can."""
    return not (path and isinstance(path[-1], ast.Constant))


def is_bare_string(s):
    """A statement that is only a string literal (a docstring, or a no-op)."""
    return isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str)


def _string_lines(tree):
    lines = set()
    for node in ast.walk(tree):
        for field in STMT_LISTS:
            body = getattr(node, field, None)
            if isinstance(body, list):
                for x in body:
                    if is_bare_string(x):
                        lines.update(range(x.lineno, x.end_lineno + 1))
    return lines


def strings_only(old, new, old_lines, new_lines, groups):
    """Does every changed line sit in a bare string statement (or is it blank
    or a comment) on both sides? A cheap test before comparing whole trees."""
    if old._strings is None:
        old._strings = _string_lines(old.tree)
    for o0, o1, _n0, _n1 in groups:
        for ln in range(o0, o1):
            if ln not in old._strings and not is_trivial(old_lines[ln - 1]):
                return False
    news = _string_lines(new.tree)
    for _o0, _o1, n0, n1 in groups:
        for ln in range(n0, n1):
            if ln not in news and not is_trivial(new_lines[ln - 1]):
                return False
    return True


def without_strings(text):
    """The syntax tree of text with bare string statements left out."""
    tree = ast.parse(text)
    for node in ast.walk(tree):
        for field in STMT_LISTS:
            body = getattr(node, field, None)
            if isinstance(body, list):
                body[:] = [x for x in body if not is_bare_string(x)]
    return ast.dump(tree)


def bound_names(s):
    """Names a module-level import or plain assignment binds, or None for
    any other statement (whose effects could reach anything)."""
    if isinstance(s, (ast.Import, ast.ImportFrom)):
        names = set()
        for a in s.names:
            if a.name == "*":
                return None
            names.add((a.asname or a.name).split(".")[0])
        return names
    if isinstance(s, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        names = set()
        for t in s.targets if isinstance(s, ast.Assign) else [s.target]:
            for n in ast.walk(t):
                if isinstance(n, ast.Name):
                    names.add(n.id)
                elif not isinstance(n, (ast.Tuple, ast.List, ast.Starred, ast.expr_context)):
                    return None  # X[k] = v, obj.attr = v: changes another object
        return names or None
    return None


_SOURCES = None


def source_texts():
    """path -> text of every Python file under sqlglot/ and tests/ at the
    pinned commit."""
    global _SOURCES
    if _SOURCES is None:
        _SOURCES = {}
        for top in ("sqlglot", "tests"):
            for dirpath, _dirs, names in os.walk(SRC / top):
                for name in names:
                    if name.endswith(".py"):
                        p = Path(dirpath) / name
                        _SOURCES[str(p.relative_to(SRC))] = p.read_text(encoding="utf-8", errors="replace")
    return _SOURCES


def module_of(path):
    name = path[:-3].replace("/", ".")
    return name[: -len(".__init__")] if name.endswith(".__init__") else name


_TREES = {}


def source_tree(path):
    t = _TREES.get(path)
    if t is None:
        try:
            t = ast.parse(source_texts()[path])
        except SyntaxError:
            t = ast.Module(body=[], type_ignores=[])
        _TREES[path] = t
    return t


def _resolve_from(path, node):
    """The absolute module an ImportFrom in `path` imports from."""
    if not node.level:
        return node.module or ""
    pkg = module_of(path) if path.endswith("__init__.py") else module_of(path).rpartition(".")[0]
    parts = pkg.split(".")
    parts = parts[: len(parts) - node.level + 1]
    return ".".join(parts + ([node.module] if node.module else []))


# `from sqlglot import exp`: sqlglot/__init__ binds exp to the expressions package.
_MODULE_ALIASES = {"sqlglot.exp": "sqlglot.expressions"}


def _uses(tree, match):
    """{line: runs at import?} of the nodes `match` accepts: True when one on
    that line sits in module or class code rather than a function body."""
    out = {}

    def walk(node, in_func):
        for child in ast.iter_child_nodes(node):
            inner = in_func or isinstance(child, FUNCS + (ast.Lambda,))
            # A def's decorators and defaults run where the def is.
            if isinstance(child, FUNCS + (ast.Lambda,)):
                for x in [*getattr(child, "decorator_list", ()), *child.args.defaults,
                          *[d for d in child.args.kw_defaults if d is not None]]:
                    walk_one(x, in_func)
            if match(child):
                ln = child.lineno
                out[ln] = out.get(ln, False) or not in_func
            walk(child, inner)

    def walk_one(node, in_func):
        if match(node):
            out[node.lineno] = out.get(node.lineno, False) or not in_func
        walk(node, in_func)

    walk(tree, False)
    return out


def name_references(f, names):
    """{path: {line: runs at import?}} of the uses of `names` as defined by
    module f: in f itself, and elsewhere through f's namespace."""
    srcs = source_texts()
    modules = {module_of(p) for p in srcs}
    fm = module_of(f)
    # Modules whose namespace holds f's names: f, modules that star-import a
    # holder, and modules that import the name itself from a holder (at
    # module level); grown to a fixpoint.
    holders = {fm: set(names)}
    changed = True
    while changed:
        changed = False
        tails = re.compile(r"\b(?:" + "|".join(sorted({h.rpartition(".")[2] for h in holders} | {"exp"})) + r")\b")
        for g, text in srcs.items():
            if not tails.search(text):
                continue
            gm = module_of(g)
            for node in source_tree(g).body:
                if not isinstance(node, ast.ImportFrom):
                    continue
                src = _MODULE_ALIASES.get(_resolve_from(g, node), _resolve_from(g, node))
                held = holders.get(src)
                if not held:
                    continue
                for a in node.names:
                    if a.name == "*":
                        new = {n for n in held if not n.startswith("_")}
                    elif a.name in held:
                        new = {a.asname or a.name}
                    else:
                        continue
                    if not new <= holders.get(gm, set()):
                        holders.setdefault(gm, set()).update(new)
                        changed = True
    every = set().union(*holders.values())
    pat = re.compile(r"\b(?:" + "|".join(sorted(map(re.escape, every))) + r")\b")
    out = {}
    for g, text in srcs.items():
        if not pat.search(text):
            continue
        gm = module_of(g)
        tree = source_tree(g)
        if gm == fm:
            uses = _uses(tree, lambda n: isinstance(n, ast.Name) and n.id in names)
            if uses:
                out[g] = uses
            continue
        # Local names bound to a holder module, and names imported from one.
        mod_alias, name_alias, import_lines = {}, set(), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.asname:
                        mod_alias[a.asname] = a.name
                    else:
                        mod_alias.setdefault(a.name.split(".")[0], a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                src = _MODULE_ALIASES.get(_resolve_from(g, node), _resolve_from(g, node))
                held = holders.get(src, ())
                for a in node.names:
                    full = _MODULE_ALIASES.get(f"{src}.{a.name}", f"{src}.{a.name}")
                    if full in modules or full in holders:
                        mod_alias[a.asname or a.name] = full
                    if a.name == "*" and held:
                        name_alias.update(held)
                        import_lines.add(node.lineno)
                    elif a.name in held:
                        name_alias.add(a.asname or a.name)
                        import_lines.add(node.lineno)

        def dotted(e):
            if isinstance(e, ast.Name):
                return mod_alias.get(e.id)
            if isinstance(e, ast.Attribute):
                base = dotted(e.value)
                return None if base is None else _MODULE_ALIASES.get(f"{base}.{e.attr}", f"{base}.{e.attr}")
            return None

        def match(n):
            if isinstance(n, ast.Name):
                return n.id in name_alias
            if isinstance(n, ast.Attribute):
                base = dotted(n.value)
                return base is not None and n.attr in holders.get(base, ())
            return isinstance(n, ast.ImportFrom) and n.lineno in import_lines

        uses = _uses(tree, match)
        if uses:
            out[g] = uses
    return out


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


def key_mask(m, ckey, name, reads):
    mask = 0
    for t in reads:
        mask |= m.key_users(ckey, name, t)
    return mask


def diff_users(m, diff, patched):
    """Tests that can see what importing the patched code left behind
    differently (verify.state_diff), or None when that cannot be told."""
    mask = 0
    for kind, owner, name, vals, keys, func in diff:
        if func:
            # A function built or bound at import: its body's own changes are
            # covered line by line; anything else (closures) is not.
            if owner.split("::")[0] in patched:
                continue
            return None
        if kind == "c":
            if owner not in m.tracked:
                return None
            if vals is None:
                mask |= m.attr_users(owner, name)
            else:
                mask |= key_mask(m, owner, name, {"*", *vals} | ({"#keys", "#len", "#ref", *keys} if keys else set()))
        else:
            for g, uses in name_references(owner, {name}).items():
                mask |= m.line_users(g, uses)
    return mask


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

# Smallest saving worth an import check (about half a second of CPU, two
# when it also compares what importing left behind).
IMPORT_CHECK_MS = int(os.environ.get("BR_IMPORT_CHECK_MS", 1000))
STATE_CHECK_MS = int(os.environ.get("BR_STATE_CHECK_MS", 2500))
IMPORT_BASE = Path(os.environ.get("BR_IMPORT_BASE", HERE / "maps" / "import_base.json"))
HEAVY_MS = int(os.environ.get("BR_HEAVY_MS", 1500))  # never run these here
VERIFY_MAX_MS = int(os.environ.get("BR_VERIFY_MAX_MS", 4000))
# With no heavy candidate to drop, a verification can only save a worker or
# two: worth it only when the cheap tests are nearly free to run.
VERIFY_LIGHT_MAX_MS = int(os.environ.get("BR_VERIFY_LIGHT_MAX_MS", 400))
CPU_FACTOR = float(os.environ.get("BR_CPU_FACTOR", 2.5))  # CPU per recorded second, run here
OVERHEAD_MS = int(os.environ.get("BR_OVERHEAD_MS", 500))  # per verification
# Drop the heavy candidates when every cheap test reaching the same code passes.
BET = os.environ.get("BR_BET", "1") == "1"
VERIFY = os.environ.get("BR_VERIFY", "0") == "1"  # off until CPU and the bet are validated
SIM_ANSWERS = os.environ.get("BR_SIM_ANSWERS")  # tune offline: labels stand in for runs
PARALLEL = 2


def load_tpcds(tests):
    with gzip.open(TPCDS_MAP, "rb") as fh:
        t = json.loads(fh.read())
    num = {x["id"]: x["number"] for x in tests}.get(t["test"])
    if num is None:
        return None
    return {
        "num": num,
        "baseline": {int(i): set(v) for i, v in t["baseline"].items()},
        "data": {f: set(v) for f, v in t["data_lines"].items()},
        "queries": {f: {int(ln): int(h, 16) for ln, h in v.items()} for f, v in t["query_lines"].items()},
        "cost": {int(i): c for i, c in t.get("cost", {}).items()},
        "worker": set(t.get("worker_modules", ())),
        "data_attrs": t.get("data_attrs"),
        "attr_baseline": t.get("attr_baseline"),
        # Queries whose plan changed but not their DuckDB SQL are run over the
        # data, within a CPU budget, and their rows compared.
        "runs": {
            "duck_baseline": {int(i): set(v) for i, v in t["duck_baseline"].items()},
            "results": {int(i): set(v) for i, v in t["results"].items()},
            "exec_cost": {int(i): c for i, c in t["exec_cost"].items()},
            "tables": {int(i): v for i, v in t["tables"].items()},
            "load_cost": t["load_cost"],
            "budget": TPCDS_RUN_BUDGET,
        } if "results" in t else None,
    }


def tpcds_queries(tp, sel, m, must, maybe):
    """The TPC-DS queries whose plan-phase output must be recomputed to rule
    out test_execute_tpcds, and whether the class attributes the data phase
    reads must be compared too; None when the check cannot decide it."""
    if tp is None:
        return None
    bit = 1 << tp["num"]
    if not maybe & bit or must & bit or not (sel.touched or sel.nonline):
        return None
    worker = tp["worker"]
    data = tp["data"]
    everything = (1 << (max(tp["baseline"]) + 1)) - 1
    affected = 0
    attrs = False
    for f, kinds in sel.nonline.items():
        # A change that is not just lines in function bodies. In a file the
        # test's pool workers never load, it can only reach the test through
        # the main process (the parsed trees and the DuckDB SQL), which the
        # check recomputes for every query.
        if not f.startswith("sqlglot/"):
            return None
        affected = everything
        if f not in worker:
            continue
        # In a module the workers load, it may also change what the data
        # phase runs. Class data attributes: the check compares every one the
        # data phase reads with its value at the pinned commit (derived ones
        # included). A def's header or decorators: the lines below cover its
        # body. Module-level names: no line using them may run in the data
        # phase or while a worker module imports. Defs and classes added,
        # removed or reshaped can change methods the data phase calls.
        if "newdef" in kinds:
            return None
        attrs = attrs or bool(kinds & {"class", "def"})
        if "module" in kinds:
            names = sel.module_names.get(f)
            if names is None:
                return None
            for g, uses in name_references(f, names).items():
                for ln, at_import in uses.items():
                    if g.startswith("sqlglot/executor/") or ln in data.get(g, ()):
                        return None
                    importers = (m.import_lines if at_import else m.import_func_lines).get(g, {}).get(str(ln), ())
                    if "?" in importers or not worker.isdisjoint(importers):
                        return None
    for f, ln in sel.touched:
        if f.startswith("sqlglot/executor/") or ln in tp["data"].get(f, ()):
            return None  # code that runs while plans execute over data
        importers = m.import_func_lines.get(f, {}).get(str(ln))
        if importers:
            # Runs while modules import. Fine when none of them is loaded by a
            # worker (DuckDB's generator parses its templates at import):
            # only the main process sees the result, and it is recomputed.
            if "?" in importers or not worker.isdisjoint(importers):
                return None
            affected = everything
            continue
        q = tp["queries"].get(f, {}).get(ln, 0)
        if not q and m.line_users(f, [ln]) & bit:
            # The test runs this line, but not inside anything the check
            # recomputes (one-time setup such as dispatch caches, or code
            # after the plan ran): the check cannot vouch for it.
            return None
        affected |= q
    if not affected:
        return None
    # Cheapest first: one differing query decides, and is usually found early.
    return sorted(numbers_of(affected, max(tp["baseline"]) + 1), key=lambda i: tp["cost"].get(i, 0)), attrs


def test_module_files():
    """Every test module, as the import checks import them (the same list
    when the image is built and when it runs)."""
    return sorted(str(p.relative_to(SRC)) for p in (SRC / "tests").rglob("test_*.py"))


def import_baseline(out_path, seeds=6):
    """What the import check finds at the pinned commit, with every value
    importing leaves behind under several hash seeds: computed when the
    image is built."""
    import subprocess

    runs = []
    for seed in range(seeds):
        res = subprocess.run([sys.executable, "-I", __file__, "--import-baseline-one"], capture_output=True,
                             text=True, env=dict(os.environ, PYTHONHASHSEED=str(seed)), check=True)
        runs.append(json.loads(res.stdout))
    failed = sorted(set().union(*(r["failed"] for r in runs)))
    flat, keyed = {}, {}
    for r in runs:
        rflat, rkeyed, funcs = r["state"]
        for k, v in rflat:
            flat.setdefault(tuple(k), set()).add(v)
        for k, per in rkeyed:
            d = keyed.setdefault(tuple(k), {})
            for t, h in per.items():
                d.setdefault(t, set()).add(h)
    out = {"failed": failed,
           "state": ([[list(k), sorted(v, key=str)] for k, v in flat.items()],
                     [[list(k), {t: sorted(h, key=str) for t, h in per.items()}] for k, per in keyed.items()],
                     funcs)}
    Path(out_path).write_text(json.dumps(out))
    unstable = sum(len(v) > 1 for v in flat.values())
    log(f"import baseline: {len(failed)} modules fail at the pinned commit; {len(flat)} values, {unstable} seed-dependent")


def import_baseline_one():
    sys.path.insert(0, str(HERE))
    import verify

    h = verify.start_import_check(SRC, {}, (), test_module_files(), True)
    while (res := verify.poll_json(h, 600)) is None:
        time.sleep(0.05)
    status, out = res
    if status != "done":
        raise SystemExit(f"import baseline failed: {status}")
    print(json.dumps(out))


def failure_impact(m, broken, files):
    """Tests that fail when these modules, dialects and test files no longer
    import (verify._import_everything's names), plus everything that imports
    the patched modules."""
    mask = 0
    for f in files:
        if f.endswith(".py"):
            mask |= m.module_impact_full(f)
    for b in broken:
        if b.startswith("tests/"):
            mask |= TESTS_BY_FILE.get(b, 0)
            continue
        mod = "sqlglot.dialects." + b.split(":", 1)[1] if b.startswith("dialect:") else b
        path = mod.replace(".", "/")
        for f in (path + ".py", path + "/__init__.py"):
            if (SRC / f).exists():
                mask |= m.module_impact_full(f)
    return mask


def run_import_checks(plans, m):
    """For each plan with a light selection (one that assumes every module
    still imports), import everything on the patched code; any module or
    test file that imported at the pinned commit and now fails sends the
    plan back to its full selection."""
    todo = [p for p in plans if p["full"]]
    if not todo:
        return
    sys.path.insert(0, str(HERE))
    import verify

    test_files = test_module_files()
    state = any(p["state"] for p in todo)
    try:
        # Made when the image was built (selector.py --import-baseline).
        status, out = "done", json.loads(IMPORT_BASE.read_text())
    except (OSError, ValueError):
        h = verify.start_import_check(SRC, {}, (), test_files, state)
        while (res := verify.poll_json(h, 60)) is None:
            time.sleep(0.01)
        status, out = res
    if status != "done" or not isinstance(out, dict):
        log(f"import check baseline failed ({status}); using full selections")
        for p in todo:
            p["must"], p["maybe"] = p["full"]
            if p["risk"]:
                p["must"] = m.all
        return
    baseline = set(out["failed"])
    if state and "state" not in out:
        log("import check baseline has no state; using full selections")
        for p in todo:
            if p["state"]:
                p["must"], p["maybe"] = p["full"]
        todo = [p for p in todo if not p["state"]]
    elif state:
        flat, keyed, funcs = out["state"]
        verify.BASE_STATE = ({tuple(k): set(v) for k, v in flat},
                             {tuple(k): {t: set(h) for t, h in per.items()} for k, per in keyed},
                             {tuple(k) for k in funcs})
    queue, running = list(todo), []
    while queue or running:
        while queue and len(running) < PARALLEL:
            p = queue.pop(0)
            if time.monotonic() - START > CHECK_DEADLINE:
                log(f"{p['commit']}: out of time for an import check; full selection")
                p["must"], p["maybe"] = p["full"]
                if p["risk"]:
                    p["must"] = m.all  # an import may fail, and nothing tells which tests that breaks
                continue
            running.append((p, verify.start_import_check(SRC, p["files"], p["deleted"], test_files, p["state"])))
        still = []
        for p, h in running:
            res = verify.poll_json(h, 30)
            if res is None:
                still.append((p, h))
                continue
            status, out = res
            broken = sorted(set(out["failed"]) - baseline) if status == "done" and isinstance(out, dict) else None
            extra = 0
            if broken == [] and p["state"]:
                # Readers of whatever else importing left behind differently.
                extra = diff_users(m, out.get("diff", ()), set(p["files"]) | p["deleted"])
                if extra is None:
                    broken = ["(import-time state changed in a way that cannot be traced)"]
            fmust, fmaybe = p["full"]
            if broken == []:
                p["maybe"] |= extra & (fmust | fmaybe) & ~p["must"]
                log(f"{p['commit']}: everything still imports ({h.get('cpu', 0):.2f} s CPU); light selection")
            else:
                # Something no longer imports (or the check hung): every test
                # that imports it fails, beyond what the full selection holds.
                log(f"{p['commit']}: import check {'failed: ' + ', '.join(broken[:5]) if broken else status}; "
                    f"full selection")
                p["must"], p["maybe"] = fmust | failure_impact(m, broken or [], p["files"]), fmaybe
        running = still
        if running:
            time.sleep(0.01)


def run_tpcds_checks(tp, plans):
    todo = [p for p in plans if p.get("tpcds")]
    if not todo:
        return
    sys.path.insert(0, str(HERE))
    import verify

    queue, running = list(todo), []
    while queue or running:
        while queue and len(running) < PARALLEL:
            p = queue.pop(0)
            if time.monotonic() - START > CHECK_DEADLINE:
                log(f"{p['commit']}: out of time for the TPC-DS check")
                continue
            attrs = (tp["data_attrs"], tp["attr_baseline"]) if p["tpcds_attrs"] else None
            running.append((p, verify.start_tpcds(SRC, p["files"], p["tpcds"],
                                                  {i: tp["baseline"][i] for i in p["tpcds"]}, p["deleted"], attrs,
                                                  tp["runs"])))
        still = []
        for p, h in running:
            # A check runs about as long as its queries did at the pinned
            # commit; far longer means the patch made sqlglot slow or loop,
            # and then the real test would time out too.
            res = verify.poll_json(h, 1.5 + 3 * sum(tp["cost"].get(i, 0.1) for i in p["tpcds"]) + 3 * TPCDS_RUN_BUDGET)
            if res is None:
                still.append((p, h))
                continue
            status, out = res
            same = status == "done" and out.get("same") is True
            if same:
                p["heavy"] &= ~(1 << tp["num"])
                p["light"] &= ~(1 << tp["num"])
            ran = f", {out.get('ran')} of {out.get('plan_only')} changed plans run" if status == "done" and out.get("plan_only") else ""
            log(f"{p['commit']}: TPC-DS check over {len(p['tpcds'])} queries{ran}"
                f"{' and data-phase attributes' if p['tpcds_attrs'] else ''}: "
                f"{'identical, test_execute_tpcds dropped' if same else 'differs, kept'} "
                f"({h.get('cpu', 0):.2f} s CPU)")
        running = still
        if running:
            time.sleep(0.01)


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
        if t["id"].startswith("tests/test_executor.py::TestExecutor::"):
            EXECUTOR[t["id"]] = t["number"]
    heavy = sum(1 << t["number"] for t in tests if ms[t["number"]] >= HEAVY_MS)
    m = Map(MAP, n)
    sel = Selector(tests, m)
    tp = load_tpcds(tests) if TPCDS and TPCDS_MAP.exists() else None

    plans = []
    for patch in sorted((DATA / "commits").glob("*.patch")):
        log(f"{patch.stem}:")
        text = patch.read_text(encoding="utf-8", errors="replace")
        must, maybe, files = sel.select(text, light=True)
        plan = {"commit": patch.stem, "must": must, "maybe": maybe, "files": files, "deleted": set(sel.deleted),
                "verify": False, "full": None, "state": sel.state_check, "risk": sel.import_risk}
        if m.light_used or sel.import_risk:
            # Both are sound when the patched code still imports everywhere,
            # and only the full one otherwise: an import check decides,
            # when it can save more than it costs.
            fmust, fmaybe, _ = sel.select(text, light=False)
            both = fmust | fmaybe
            must, maybe = must & both, maybe & both
            saving = cost_of(split(numbers_of(both, n), ms), ms) - cost_of(split(numbers_of(must | maybe, n), ms), ms)
            if plan["risk"] or saving > (STATE_CHECK_MS if plan["state"] else IMPORT_CHECK_MS):
                plan["must"], plan["maybe"] = must, maybe
                plan["full"] = (fmust, fmaybe)
            else:
                plan["must"], plan["maybe"] = fmust, fmaybe
            must, maybe = fmust, fmaybe
        tq = tpcds_queries(tp, sel, m, must, maybe)
        if tq and tq[1] and not tp["data_attrs"]:
            tq = None  # no attribute baselines to compare with
        plan["tpcds"], plan["tpcds_attrs"] = tq or (None, False)
        plans.append(plan)

    run_import_checks(plans, m)

    for plan in plans:
        must, maybe = plan["must"], plan["maybe"]
        plan["light"], plan["heavy"] = maybe & ~heavy, maybe & heavy
        if plan["tpcds"] and not (must | maybe) >> tp["num"] & 1:
            plan["tpcds"] = None
        if plan["tpcds"]:
            # Worth checking only if dropping the test would really cut this
            # commit's cost (another heavy test may set the pace anyway).
            everything = numbers_of(must | maybe, n)
            without = [i for i in everything if i != tp["num"]]
            saving = cost_of(split(everything, ms), ms) - cost_of(split(without, ms), ms)
            check_ms = 300 + 1050 * sum(tp["cost"].get(i, 0.1) for i in plan["tpcds"])
            if TPCDS_HIT_RATE * saving <= check_ms:
                log(f"{plan['commit']}: TPC-DS check skipped: saves {saving / 1000:.1f} s at best, "
                    f"costs ~{check_ms / 1000:.1f} s")
                plan["tpcds"] = None
        light, hv = plan["light"], plan["heavy"]
        if VERIFY and light:
            sum_l = sum(ms[i] for i in numbers_of(light, n))
            listed = cost_of(split(numbers_of(must | maybe, n), ms), ms)
            keep = must if BET else must | hv
            best = OVERHEAD_MS + CPU_FACTOR * sum_l + cost_of(split(numbers_of(keep, n), ms), ms)
            limit = VERIFY_MAX_MS if (hv and BET) else VERIFY_LIGHT_MAX_MS
            if sum_l <= limit and best < listed:
                plan["verify"] = True
                plan["sum_l"] = sum_l

    if os.environ.get("BR_DUMP"):
        with open(os.environ["BR_DUMP"], "w") as fh:
            for p in plans:
                fh.write(json.dumps({k: numbers_of(p[k], n) for k in ("must", "light", "heavy")}
                                    | {"commit": p["commit"]}) + "\n")

    run_tpcds_checks(tp, plans)

    todo = [p for p in plans if p["verify"]]
    est_cpu = 0
    if todo and SIM_ANSWERS:
        answers = json.load(open(SIM_ANSWERS))["commits"]
        for p in todo:
            failing = set(answers.get(p["commit"], {}).get("failing", ()))
            p["failed"] = sum(1 << i for i in numbers_of(p["light"], n) if i in failing)
            est_cpu += OVERHEAD_MS + CPU_FACTOR * p["sum_l"]
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
                    p["cpu"] = h.get("cpu", 0)
            running = still
            if running:
                time.sleep(0.01)

    total = 0
    for p in plans:
        if p["verify"]:
            failed = p["failed"]
            keep_heavy = p["heavy"] if (failed or not BET) else 0
            mask = p["must"] | failed | keep_heavy
            log(f"{p['commit']}: verified {bin(p['light']).count('1')} cheap tests "
                f"({p['sum_l'] / 1000:.1f} s recorded, {p.get('cpu', 0):.2f} s CPU), "
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
    if sys.argv[1:2] == ["--import-baseline"]:
        import_baseline(sys.argv[2])
    elif sys.argv[1:2] == ["--import-baseline-one"]:
        import_baseline_one()
    else:
        main()
