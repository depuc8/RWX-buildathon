"""Score the selector on a labeled commit set and explain where the cost goes.

    python3 bench/eval.py COMMITS_DIR ANSWERS_JSON [--map maps/brmap.json.gz] [-v]

Prints total simulated seconds, misses (with the tests that were missed),
the oracle (only the failing tests, best split), and a breakdown by kind.
"""

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import selector  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("commits")
    ap.add_argument("answers")
    ap.add_argument("--map", default=os.path.join(ROOT, "maps", "brmap.json.gz"))
    ap.add_argument("--tests", default=os.path.join(ROOT, "sample", "tests.json"))
    ap.add_argument("--src", default=os.path.join(ROOT, "work", "sqlglot"))
    ap.add_argument("-v", action="store_true")
    args = ap.parse_args()

    data = os.path.join(ROOT, "work", "evaldata")
    os.makedirs(data, exist_ok=True)
    for name, target in (("commits", os.path.abspath(args.commits)), ("tests.json", os.path.abspath(args.tests))):
        link = os.path.join(data, name)
        if os.path.islink(link):
            os.remove(link)
        os.symlink(target, link)

    env = dict(os.environ, BR_DATA=data, BR_SRC=args.src, BR_MAP=args.map)
    run = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "selector.py")],
                         env=env, capture_output=True, text=True)
    if run.returncode:
        print(run.stderr[-3000:])
        sys.exit("selector crashed")
    cpu = [l for l in run.stderr.splitlines() if l.startswith("selector CPU")]

    tests = json.load(open(args.tests))["tests"]
    ms = {t["number"]: round(t["seconds"] * 1000) for t in tests}
    ans = json.load(open(args.answers))["commits"]

    def cost(workers):
        return (max(sum(ms[n] for n in w) for w in workers) + 2000 * len(workers)) if workers else 0

    by_kind = defaultdict(lambda: [0, 0, 0, 0])  # n, ours, oracle, misses
    total = oracle = misses = 0
    rows = []
    for line in run.stdout.splitlines():
        d = json.loads(line)
        c = d["commit"]
        if c not in ans:
            continue
        sel = {n for w in d["workers"] for n in w}
        fail = set(ans[c]["failing"])
        missed = fail - sel
        o = cost(selector.split(sorted(fail), ms))
        s = cost(d["workers"])
        k = ans[c].get("kind", "?")
        by_kind[k][0] += 1
        by_kind[k][1] += s
        by_kind[k][2] += o
        total += s
        oracle += o
        if missed:
            misses += 1
            by_kind[k][3] += 1
            rows.append(f"MISS {c} ({k}): missed {[tests[n]['id'] for n in sorted(missed)][:5]}")
        if args.v:
            print(f"{c} {k:8} ours {s / 1000:6.1f} oracle {o / 1000:6.1f} sel {len(sel):4} fail {len(fail):3}"
                  + (" MISS" if missed else ""))
    for r in rows:
        print(r)
    cpu_s = float(cpu[0].split(":")[1].split()[0]) if cpu else 0.0
    print(f"{sum(v[0] for v in by_kind.values())} commits: SCORE {(total + 300000 * misses) / 1000 + cpu_s:.1f} = "
          f"simulated {total / 1000:.1f} + {misses} misses x 300 + CPU {cpu_s:.1f}; oracle {oracle / 1000:.1f} s")
    for k, (n, s, o, mi) in sorted(by_kind.items()):
        print(f"  {k:8} n={n:3}  ours {s / 1000:7.1f}  oracle {o / 1000:7.1f}  misses {mi}")


if __name__ == "__main__":
    main()
