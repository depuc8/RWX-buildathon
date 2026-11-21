"""Label synthetic commits by running sqlglot's full suite on each, the way the
organisers labeled theirs. Runs inside the dev image:

    python bench/label.py /src /bench/commits /kit/tests.json /bench/answers.json JOBS

Each job gets its own copy of the tree and two CPUs (so the suite's process
pools stay small). Results are cached per commit in <answers dir>/labels/.
"""

import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from selector import apply_hunks, parse_patch  # noqa: E402


def build_tree(src, patch_path, tree):
    shutil.rmtree(tree, ignore_errors=True)
    # The whole checkout: some tests read files outside sqlglot/ and tests/
    # (test_sqlglotc builds sqlglotc/).
    shutil.copytree(src, tree, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.so"))
    for d in parse_patch(open(patch_path, encoding="utf-8").read()):
        if d.old:
            old = open(os.path.join(src, d.old), encoding="utf-8").read().split("\n")
        else:
            old = []
        new, _ = apply_hunks(old, d.hunks)
        if d.old and d.old != d.new:
            os.remove(os.path.join(tree, d.old))
        if d.new:
            os.makedirs(os.path.dirname(os.path.join(tree, d.new)), exist_ok=True)
            with open(os.path.join(tree, d.new), "w", encoding="utf-8") as fh:
                fh.write("\n".join(new))


def main():
    src, commits, tests_json, answers, jobs = sys.argv[1:6]
    jobs = int(jobs)
    out_dir = os.path.join(os.path.dirname(answers), "labels")
    trees = os.path.join(os.path.dirname(answers), "trees")
    os.makedirs(out_dir, exist_ok=True)
    tests = json.load(open(tests_json))["tests"]
    number = {t["id"]: t["number"] for t in tests}

    todo = sorted(
        c[:-6] for c in os.listdir(commits)
        if c.endswith(".patch") and not os.path.exists(os.path.join(out_dir, c[:-6] + ".json"))
    )
    ncpu = os.cpu_count()
    slots = [list(range(2 * i, 2 * i + 2)) for i in range(min(jobs, ncpu // 2))]
    free = list(range(len(slots)))
    running = {}
    t0 = time.time()
    done = 0
    while todo or running:
        while todo and free:
            cid = todo.pop(0)
            slot = free.pop()
            tree = os.path.join(trees, cid)
            try:
                build_tree(src, os.path.join(commits, cid + ".patch"), tree)
            except Exception as e:  # noqa: BLE001
                print(f"{cid}: cannot apply patch: {e}", file=sys.stderr)
                free.append(slot)
                continue
            env = dict(os.environ, SKIP_INTEGRATION="0", PYTHONPATH=f"{tree}:{HERE}",
                       LABEL_OUT=os.path.join(out_dir, cid + ".raw"), PYTHONDONTWRITEBYTECODE="1")
            cpus = slots[slot]
            p = subprocess.Popen(
                [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "labelplug",
                 "--timeout", "600"],
                cwd=tree, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                preexec_fn=lambda cpus=cpus: os.sched_setaffinity(0, cpus),
            )
            running[p] = (cid, slot, tree)
        time.sleep(0.5)
        for p in [p for p in running if p.poll() is not None]:
            cid, slot, tree = running.pop(p)
            free.append(slot)
            shutil.rmtree(tree, ignore_errors=True)
            raw_path = os.path.join(out_dir, cid + ".raw")
            if not os.path.exists(raw_path):
                print(f"{cid}: pytest died ({p.returncode})", file=sys.stderr)
                continue
            raw = json.load(open(raw_path))
            failing = set()
            for nid in raw["failed"]:
                if nid in number:
                    failing.add(number[nid])
            for nid in raw["errors"]:
                for t in tests:
                    if t["id"] == nid or t["id"].startswith(nid + "::"):
                        failing.add(t["number"])
            json.dump({"failing": sorted(failing), "errors": raw["errors"]},
                      open(os.path.join(out_dir, cid + ".json"), "w"))
            os.remove(raw_path)
            done += 1
            print(f"{cid}: {len(failing)} failing ({done} done, {len(todo)} left, "
                  f"{time.time() - t0:.0f}s)", file=sys.stderr, flush=True)

    kinds_path = os.path.join(os.path.dirname(commits), "kinds.json")
    kinds = json.load(open(kinds_path)) if os.path.exists(kinds_path) else {}
    result = {"commits": {}}
    for f in sorted(os.listdir(out_dir)):
        if f.endswith(".json"):
            cid = f[:-5]
            result["commits"][cid] = {
                "failing": json.load(open(os.path.join(out_dir, f)))["failing"],
                "kind": kinds.get(cid, "?"),
            }
    json.dump(result, open(answers, "w"), indent=1)


if __name__ == "__main__":
    main()
