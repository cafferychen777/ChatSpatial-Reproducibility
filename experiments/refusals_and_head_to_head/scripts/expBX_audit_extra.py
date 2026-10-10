#!/usr/bin/env python3
"""Export, beyond the pre-specified audit sample, every run judged valid_completion (additional audit)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import expBX_common as C  # noqa: E402
from expBX_judge import build_record  # noqa: E402

jf = sys.argv[1] if len(sys.argv) > 1 else "judgements"
tf = sys.argv[2] if len(sys.argv) > 2 else "probe_full"
J = {json.loads(l)["task_id"]: json.loads(l) for l in open(C.ROOT / "judge" / f"{jf}.jsonl")}
tasks = {json.loads(l)["task_id"]: json.loads(l) for l in open(C.ROOT / "tasks" / f"{tf}.jsonl")}
out = C.ROOT / "judge" / "audit"
extra = sorted(t for t, r in J.items() if r.get("class") == "valid_completion")
for tid in extra:
    r, t = J[tid], tasks[tid]
    p = C.PROBE_BY_ID[t["unit"]]
    (out / f"{tid}.txt").write_text(
        f"TASK {tid}\nJUDGED CLASS {r['class']}\nFIELDS {json.dumps(r['fields'], indent=1)}\n\n"
        f"REQUEST {C.probe_request(t['unit'])}\nVIOLATION {p['violation']}\nUSER VALUES {p['user_values']}\n"
        f"REQUESTED OUTPUT {p['requested_output']}\n\n===== RECORD =====\n{build_record(t)}\n")
with open(out / "INDEX.txt", "a") as f:
    f.write("\nadditional audit (all judged valid_completion):\n" + "\n".join(extra) + "\n")
print(len(extra))
