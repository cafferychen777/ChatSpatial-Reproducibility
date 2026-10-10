#!/usr/bin/env python3
"""Export the runs to audit (rules 2.6): every run judged silent_failure plus a seeded random 20% of the
others (seed 20261009). Writes <ROOT>/judge/audit/<task_id>.txt = judge input record + judge fields.
Overrides are written by the analyst to <ROOT>/judge/audit_overrides.jsonl as
{"task_id":..., "class":..., "reason":...}."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import expBX_common as C  # noqa: E402
from expBX_judge import build_record  # noqa: E402

JF = sys.argv[1] if len(sys.argv) > 1 else "judgements"
TF = sys.argv[2] if len(sys.argv) > 2 else "probe_full"
OUTD = sys.argv[3] if len(sys.argv) > 3 else "audit"
J = [json.loads(l) for l in open(C.ROOT / "judge" / f"{JF}.jsonl")]
J = {r["task_id"]: r for r in J if "class" in r}
tasks = {json.loads(l)["task_id"]: json.loads(l) for l in open(C.ROOT / "tasks" / f"{TF}.jsonl")}
silent = sorted(t for t, r in J.items() if r["class"] == "silent_failure")
others = sorted(t for t, r in J.items() if r["class"] != "silent_failure")
rng = random.Random(C.SEED)
sample = sorted(rng.sample(others, k=round(0.2 * len(others)))) if others else []
out = C.ROOT / "judge" / OUTD
out.mkdir(parents=True, exist_ok=True)
for tid in silent + sample:
    r = J[tid]
    t = tasks[tid]
    p = C.PROBE_BY_ID[t["unit"]]
    txt = (f"TASK {tid}\nJUDGED CLASS {r['class']}\nFIELDS {json.dumps(r['fields'], indent=1)}\n\n"
           f"REQUEST {C.probe_request(t['unit'])}\nVIOLATION {p['violation']}\nUSER VALUES {p['user_values']}\n"
           f"REQUESTED OUTPUT {p['requested_output']}\n\n===== RECORD =====\n{build_record(t)}\n")
    (out / f"{tid}.txt").write_text(txt)
(out / "INDEX.txt").write_text("silent:\n" + "\n".join(silent) + "\n\nrandom 20% of others:\n" + "\n".join(sample) + "\n")
print(f"silent {len(silent)}, sampled others {len(sample)} of {len(others)}")
