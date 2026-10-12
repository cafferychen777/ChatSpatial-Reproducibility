#!/usr/bin/env python3
"""Per-run scoring of the Qwen head-to-head runs on the GPU host with the unchanged expBX_analysis.h2h code (rule 3.3/3.4).
Run with the ChatSpatial env python on a compute allocation:
  EXPBX_ROOT=<mirror>$CHATSPATIAL_WORK_DIR/expBX python expBX_ow_h2h_score.py TASKS_NAME GT_DIR
Writes <ROOT>/analysis/<TASKS_NAME>/h2h_per_run.csv (+ cross-rep / summary when enough runs exist)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import expBX_analysis as A  # noqa: E402
import expBX_common as C  # noqa: E402

name, gt_dir = sys.argv[1], Path(sys.argv[2])
A.GT_DIR = gt_dir
A.OUT = C.ROOT / "analysis" / name
A.OUT.mkdir(parents=True, exist_ok=True)
tasks = [json.loads(l) for l in open(C.ROOT / "tasks" / f"{name}.jsonl") if l.strip()]
tasks = [t for t in tasks if (C.run_dir(t) / "status.json").exists()]
try:
    A.h2h(tasks)
except Exception as e:  # summary statistics need several sections; per-run table is written first
    print("summary step skipped:", type(e).__name__, e)
import pandas as pd  # noqa: E402
df = pd.read_csv(A.OUT / "h2h_per_run.csv")
cols = [c for c in ["task_id", "system", "section", "rep", "timed_out", "wall_time", "success", "verified", "method",
                    "methods_in_code", "column", "n_labels", "coverage", "ari", "nmi"] if c in df]
with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 500):
    print(df[cols].to_string(index=False))
    print(df.groupby("system")[["ari", "nmi"]].agg(["count", "mean"]).round(3))
