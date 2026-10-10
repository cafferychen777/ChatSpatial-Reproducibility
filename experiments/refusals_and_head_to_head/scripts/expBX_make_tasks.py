#!/usr/bin/env python3
"""Write task lists. Order interleaves systems and units (rep-major) so that every system runs
across the same time window. Usage: python3 expBX_make_tasks.py NAME PART [--systems a,b] [--units u1,u2] [--reps 0,1]"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import expBX_common as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("part", choices=["h2h", "probe"])
ap.add_argument("--systems")
ap.add_argument("--units")
ap.add_argument("--reps")
ap.add_argument("--tag", default="main")
a = ap.parse_args()
tasks = C.build_tasks(a.part)
if a.systems:
    tasks = [t for t in tasks if t["system"] in a.systems.split(",")]
if a.units:
    tasks = [t for t in tasks if t["unit"] in a.units.split(",")]
if a.reps:
    tasks = [t for t in tasks if t["rep"] in {int(x) for x in a.reps.split(",")}]
order = {s: i for i, s in enumerate(["stagent", "spatialagent", "chatspatial", "chatspatial_single"])}
tasks.sort(key=lambda t: (t["rep"], t["unit"], order[t["system"]]))
for t in tasks:
    t["tag"] = a.tag
    if a.tag != "main":
        t["task_id"] = f"{a.tag}__{t['task_id']}"
out = C.ROOT / "tasks" / f"{a.name}.jsonl"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    for t in tasks:
        f.write(json.dumps(t) + "\n")
print(out, len(tasks))
