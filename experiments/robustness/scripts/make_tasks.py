#!/usr/bin/env python3
"""Write task JSON files for an experiment-A stage.

  make_tasks.py pilot   DLPFC 12 sections, clean data: full grid (+2 stability
                        subsamples) for 6 methods, expert settings, and the
                        array-coordinate SpaGCN default (continuity with May 2026).
  make_tasks.py platforms  same grid for STARmap and ABC MERFISH (after the gate).
  make_tasks.py noise --from-stage pilot
                        noise axis: default, expert, rule-selected and clean-oracle
                        configurations on 5 degraded versions of every section.
  make_tasks.py test    end-to-end test subset.
Prints the number of array tasks; the sbatch array range is 0..N-1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import grid as G  # noqa: E402

CHUNK = {"spagcn": 12, "leiden": 75, "louvain": 75, "stagate": 28, "graphst": 42, "banksy": 25}
DEEP = ("stagate", "graphst")  # run with more CPUs
REPS = ["full"] + C.STABILITY_REPS


def grid_calls(method):
    g = G.method_grid(method)
    calls = []
    for i, cfg in enumerate(g["configs"]):
        cid = C.short_id(G.config_id(method, cfg))
        calls.append({"call_id": cid, "config": cfg, "strategy": "default" if cfg == g["defaults"] else "grid"})
    return calls


def write(stage, tasks, tdir: Path):
    tdir.mkdir(parents=True, exist_ok=True)
    tasks = sorted(tasks, key=lambda t: t["method"] in DEEP)  # light methods first
    for old in tdir.glob("task_*.json"):
        old.unlink(missing_ok=True)
    for i, t in enumerate(tasks):
        t["stage"] = stage
        t["index"] = i
        (tdir / f"task_{i:05d}.json").write_text(json.dumps(t, default=str))
    n_calls = sum(len(t["calls"]) * len(t["reps"]) for t in tasks)
    n_light = sum(t["method"] not in DEEP for t in tasks)
    print(f"{stage}: {len(tasks)} tasks, {n_calls} tool calls -> {tdir}")
    print(f"light {n_light}")
    return len(tasks)


def grid_tasks(dataset, sections, methods, reps=REPS):
    tasks = []
    for sid in sections:
        for m in methods:
            calls = grid_calls(m)
            k = CHUNK[m]
            for j in range(0, len(calls), k):
                tasks.append({"dataset": dataset, "section": sid, "condition": "clean", "method": m,
                              "calls": calls[j:j + k], "reps": reps})
            tasks.append({"dataset": dataset, "section": sid, "condition": "clean", "method": m,
                          "calls": [{"call_id": f"expert_{m}", "strategy": "expert", "config": None}],
                          "reps": ["full"]})
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pilot", "dlpfc", "platforms", "noise", "test"])
    ap.add_argument("--from-stage", default="pilot")
    ap.add_argument("--name", default=None,
                    help="result/task tag (default: the stage name), e.g. dlpfc155")
    args = ap.parse_args()
    name = args.name or args.stage
    tdir = C.ROOT / "tasks" / name
    if args.stage in ("pilot", "dlpfc"):
        tasks = grid_tasks("dlpfc", C.DLPFC_SECTIONS, G.METHODS)
        sp = [c for c in grid_calls("spagcn") if c["strategy"] == "default"]
        for sid in C.DLPFC_SECTIONS:
            tasks.append({"dataset": "dlpfc_array", "section": sid, "condition": "clean",
                          "method": "spagcn", "calls": sp, "reps": ["full"]})
        write(name, tasks, tdir)
    elif args.stage == "platforms":
        tasks = grid_tasks("starmap", C.DATASETS["starmap"]["sections"], G.METHODS)
        tasks += grid_tasks("abc_merfish", C.DATASETS["abc_merfish"]["sections"], G.METHODS)
        write(name, tasks, tdir)
    elif args.stage == "noise":
        import select_rule as S
        sel = S.strategy_table(args.from_stage)  # default / expert / rule / oracle per section x method
        tasks = []
        for (ds, sid, m), g in sel.groupby(["dataset", "section", "method"]):
            if ds == "dlpfc_array":
                continue
            calls = []
            for _, r in g.iterrows():
                if r.strategy == "expert":
                    calls.append({"call_id": f"expert_{m}", "strategy": "expert", "config": None})
                else:
                    spec = getattr(r, "config_spec", None)
                    cfg = json.loads(spec if isinstance(spec, str) and spec else r.config)
                    post = {"refine_domains": cfg.pop("refine_domains"),
                            "refinement_threshold": cfg.pop("refinement_threshold")}
                    calls.append({"call_id": f"{r.strategy}_{r.call_id}", "strategy": r.strategy,
                                  "config": cfg, "post": [post], "resolved": True})
            for cond in C.NOISE_CONDITIONS:
                tasks.append({"dataset": ds, "section": sid, "condition": cond, "method": m,
                              "calls": calls, "reps": ["full"]})
        write(name, tasks, tdir)
    elif args.stage == "test":
        tasks = []
        sp_calls = grid_calls("spagcn")
        ld_calls = grid_calls("leiden")
        pick = lambda calls: [calls[0]] + calls[1:3]  # default + 2 grid points
        tasks.append({"dataset": "dlpfc", "section": "151673", "condition": "clean", "method": "spagcn",
                      "calls": pick(sp_calls), "reps": REPS})
        tasks.append({"dataset": "dlpfc", "section": "151673", "condition": "clean", "method": "leiden",
                      "calls": pick(ld_calls), "reps": REPS})
        tasks.append({"dataset": "dlpfc", "section": "151673", "condition": "umi0.25", "method": "spagcn",
                      "calls": [sp_calls[0]], "reps": ["full"]})
        tasks.append({"dataset": "dlpfc", "section": "151673", "condition": "drop0.3", "method": "leiden",
                      "calls": [ld_calls[0]], "reps": ["full"]})
        tasks.append({"dataset": "dlpfc_array", "section": "151673", "condition": "clean", "method": "spagcn",
                      "calls": [sp_calls[0]], "reps": ["full"]})
        for m in ("stagate", "graphst", "banksy", "louvain"):
            c = grid_calls(m)
            tasks.append({"dataset": "dlpfc", "section": "151673", "condition": "clean", "method": m,
                          "calls": [c[0], c[1]], "reps": ["full"]})
        tasks.append({"dataset": "dlpfc", "section": "151673", "condition": "clean", "method": "leiden",
                      "calls": [{"call_id": "expert_leiden", "strategy": "expert", "config": None}],
                      "reps": ["full"]})
        write(name, tasks, tdir)


if __name__ == "__main__":
    main()
