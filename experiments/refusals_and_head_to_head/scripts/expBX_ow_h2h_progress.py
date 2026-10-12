#!/usr/bin/env python3
"""Progress of the Qwen3.8 head-to-head arm on the GPU host (stdlib; reads status/result files only).
Usage: python3 expBX_ow_h2h_progress.py [TASKS_NAME] [--fail]   (default TASKS_NAME = ow_qwen38_h2h)
Per system: finished/total, running, timeouts, missing result.json, harness/driver errors, runs with labels
(chatspatial: labels.csv; agents: out/result.h5ad or another .h5ad under out/), mean wall time; replicates complete on
all 12 sections for all three systems; with --fail, one line per problem run."""
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

G = os.path.expanduser("~/expBX/mirror")
X = Path(G + os.environ["CHATSPATIAL_WORK_DIR"] + "/expBX")  # CPU-cluster work directory, mirrored under G
sys.path.insert(0, str(X / "scripts"))
import expBX_common as C  # noqa: E402

name = next((a for a in sys.argv[1:] if not a.startswith("--")), "ow_qwen38_h2h")
tasks = [json.loads(l) for l in open(X / "tasks" / f"{name}.jsonl") if l.strip()]
agg = defaultdict(lambda: defaultdict(float))
fails, done_by = [], defaultdict(set)
for t in tasks:
    rd = Path(G + str(C.run_dir(t)))
    s = t["system"]
    a = agg[s]
    a["n"] += 1
    try:
        st = json.loads((rd / "status.json").read_text())
    except Exception:
        continue
    if not st.get("finished"):
        a["running"] += 1
        continue
    a["done"] += 1
    a["wall"] += st.get("wall_time") or 0
    if st.get("timed_out"):
        a["timeout"] += 1
        fails.append((t["task_id"], "timeout (scored on saved output)"))
    try:
        r = json.loads((rd / "result.json").read_text())
    except Exception:
        a["no_result"] += 1
        fails.append((t["task_id"], f"no result.json rc={st.get('returncode')}"))
        r = {}
    if r.get("harness_error"):
        a["harness_err"] += 1
        fails.append((t["task_id"], "harness: " + str(r["harness_error"])[:150]))
    if r.get("error"):
        a["driver_err"] += 1
        fails.append((t["task_id"], "driver: " + str(r["error"])[:150]))
    if s in ("chatspatial", "chatspatial_single"):
        lab = (rd / "labels.csv").exists()
    else:
        lab = (rd / "out" / "result.h5ad").exists() or any((rd / "out").rglob("*.h5ad")) if (rd / "out").exists() else False
    a["labels"] += lab
    if not lab:
        fails.append((t["task_id"], "no labels/output h5ad"))
    done_by[(t["unit"], t["rep"])].add(s)
print(time.strftime("%Y-%m-%d %H:%M:%S"), name)
for s, v in sorted(agg.items()):
    d = v["done"] or 1
    print(f"{s:13s} done {int(v['done']):3d}/{int(v['n'])} running {int(v['running'])} timeout {int(v['timeout'])} "
          f"noResult {int(v['no_result'])} harnessErr {int(v['harness_err'])} driverErr {int(v['driver_err'])} "
          f"withLabels {int(v['labels'])} wallMean {v['wall'] / d / 60:.1f}min")
systems = sorted(agg)
reps = sorted({t["rep"] for t in tasks})
units = sorted({t["unit"] for t in tasks})
full = [r for r in reps if all(set(systems) <= done_by[(u, r)] for u in units)]
print(f"TOTAL done {int(sum(v['done'] for v in agg.values()))}/{len(tasks)}; replicates complete on all {len(units)} "
      f"sections x {len(systems)} systems: {full}")
if "--fail" in sys.argv:
    for tid, why in fails:
        print("FAIL", tid, why)
else:
    print(f"problem runs: {len(fails)} (add --fail to list)")
