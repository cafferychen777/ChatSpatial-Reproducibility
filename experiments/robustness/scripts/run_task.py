#!/usr/bin/env python3
"""Experiment A array task: run one chunk of (section x condition x method x configs).

Usage: run_task.py TASK_JSON            (or --tasks-dir DIR, index from SLURM_ARRAY_TASK_ID)

Checkpointing: every finished (call, replicate) appends its rows to the task CSV;
on restart, finished units are skipped. Unrefined labels are saved as .npz so
any later analysis can be recomputed without rerunning the backend.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import grid as G  # noqa: E402

COLS = ["stage", "dataset", "section", "condition", "method", "call_id", "strategy",
        "config", "config_spec", "rep", "refine_domains", "refinement_threshold", "n_obs", "n_domains_found",
        "kappa", "stability", "gt_ari", "gt_nmi", "gt_n", "runtime_s", "status", "error",
        "refine_matches_tool", "n_warnings"]


def bisect_resolution(pre, method, base_cfg, key, target, lo=0.01, hi=3.0, iters=12):
    """Expert rule for resolution-driven methods: tune resolution to the annotated domain count."""
    best = None
    for _ in range(iters):
        mid = round((lo + hi) / 2, 5)
        cfg = dict(base_cfg, **{key: mid})
        r = C.call_tool(pre, method, cfg)
        n = len(set(r["labels"]))
        cand = (abs(n - target), mid, cfg, r)
        if best is None or cand[:2] < best[:2]:
            best = cand
        if n == target:
            break
        if n > target:
            hi = mid
        else:
            lo = mid
    return best[2], best[3]


def expert_config(dataset, method, pre, n_annot):
    platform = C.DATASETS[dataset]["platform"]
    coords = pre.obsm["spatial"]
    if method == "spagcn":
        return {"n_domains": n_annot}, {"refine_domains": True, "refinement_threshold": 0.5}
    if method == "stagate":
        if os.environ.get("EXPA_MANUAL_STAGATE_RADIUS") == "1":  # pilot only (pre-release default was 50)
            return ({"n_domains": n_annot,
                     "stagate_rad_cutoff": round(C.radius_for_mean_neighbours(coords, 6.0), 4)},
                    {"refine_domains": False, "refinement_threshold": 0.5})
        # 1.5.5 default radius links ~6 neighbours per spot, as in STAGATE's tutorials
        return {"n_domains": n_annot}, {"refine_domains": False, "refinement_threshold": 0.5}
    if method == "graphst":
        return ({"n_domains": n_annot, "graphst_clustering_method": "mclust",
                 "graphst_refinement": True, "graphst_radius": 50},
                {"refine_domains": False, "refinement_threshold": 0.5})
    if method == "banksy":
        visium = platform == "Visium"
        return ({"banksy_lambda": 0.2 if visium else 0.8,
                 "banksy_num_neighbours": 18 if visium else 15,
                 "banksy_max_m": 1, "banksy_pca_dims": 20, "__tune__": "banksy_cluster_resolution"},
                {"refine_domains": False, "refinement_threshold": 0.5})
    if method in ("leiden", "louvain"):
        return ({"__tune__": "resolution"}, {"refine_domains": False, "refinement_threshold": 0.5})
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?")
    ap.add_argument("--tasks-dir")
    args = ap.parse_args()
    if args.task:
        tpath = Path(args.task)
    else:
        idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
        tpath = Path(args.tasks_dir) / f"task_{idx:05d}.json"
    T = json.loads(tpath.read_text())
    ds, sid, cond, method, stage = T["dataset"], T["section"], T["condition"], T["method"], T["stage"]
    outdir = C.ROOT / "results" / stage / ds / sid / cond / method
    labdir = outdir / "labels"
    labdir.mkdir(parents=True, exist_ok=True)
    labfile = labdir / f"{tpath.stem}_labels.jsonl.gz"

    def load_full_labels(cid):
        """Full-data labels of a call: per-task gzip JSON lines, or a per-unit .npz (older runs)."""
        found = None
        if labfile.exists():
            try:
                with gzip.open(labfile, "rt") as fh:
                    for line in fh:
                        try:
                            d = json.loads(line)
                        except json.JSONDecodeError:
                            continue  # truncated line from an interrupted write
                        if d["call_id"] == cid and d["rep"] == "full":
                            found = (np.asarray(d["labels"]), np.asarray(d["obs_names"]))
            except (EOFError, OSError):
                pass  # truncated gzip member at the end of the file
            if found is not None:  # the last record wins (re-run units append newer labels)
                return found
        f = np.load(labdir / f"{cid}_full.npz", allow_pickle=True)  # our own files
        return f["labels"], f["obs_names"]
    out_csv = outdir / f"{tpath.stem}.csv"
    done = set()
    timed_out = set()  # (rep, training-input key) whose training hit the runtime cap

    def train_key(call):
        # STAGATE training depends only on the graph radius (and seed); GraphST only on its seed
        cfg = call.get("config") or {}
        if method == "stagate":
            return str(cfg.get("stagate_rad_cutoff")) + "|" + str(cfg.get("stagate_random_seed"))
        if method == "graphst":
            return str(cfg.get("graphst_random_seed"))
        return None

    if out_csv.exists():
        prev = pd.read_csv(out_csv, dtype=str, keep_default_na=False)
        done = set(zip(prev.call_id, prev.rep))
        calls_by_id = {c["call_id"]: c for c in T["calls"]}
        for cid_, rep_, err_ in zip(prev.call_id, prev.rep, prev.error):
            if "timeout" in err_.lower() and cid_ in calls_by_id and train_key(calls_by_id[cid_]):
                timed_out.add((rep_, train_key(calls_by_id[cid_])))
    print(f"[{time.strftime('%H:%M:%S')}] {tpath.name}: {ds}/{sid}/{cond}/{method}, "
          f"{len(T['calls'])} calls x reps {T['reps']}, {len(done)} units already done", flush=True)

    gt = C.load_gt(ds, sid)
    raw = C.load_raw(ds, sid)
    pre_cache = {}

    def get_pre(rep):
        if rep not in pre_cache:
            cond_eff = cond if rep == "full" else rep
            if rep != "full" and cond != "clean":
                raise ValueError("stability replicates are defined on clean data only")
            pre_cache[rep] = C.preprocess(C.degrade(raw, cond_eff, ds, sid))
        return pre_cache[rep]

    n_annot = int(gt.dropna().nunique())
    post_all = G.postproc_variants()
    full_variant_labels = {}  # call_id -> {variant_key: labels Series (full rep)}

    for call in T["calls"]:
        cid, strategy = call["call_id"], call.get("strategy", "grid")
        for rep in T["reps"]:
            if (cid, rep) in done:
                continue
            pre = get_pre(rep)
            coords = pre.obsm["spatial"]
            rows = []
            base = dict(stage=stage, dataset=ds, section=sid, condition=cond, method=method,
                        call_id=cid, strategy=strategy, rep=rep, n_obs=pre.n_obs)
            try:
                if strategy == "expert":
                    cfg, post = expert_config(ds, method, pre, n_annot)
                    if "__tune__" in cfg:
                        key = cfg.pop("__tune__")
                        cfg, r = bisect_resolution(pre, method, cfg, key, n_annot)
                    else:
                        r = C.call_tool(pre, method, cfg)
                    posts = [post]
                else:
                    cfg = G.resolve(call["config"], coords)
                    tk = train_key(call)
                    if tk and (rep, tk) in timed_out:
                        raise TimeoutError("training timeout (identical training input already "
                                           "exceeded the runtime cap in this task; not rerun)")
                    try:
                        r = C.call_tool(pre, method, cfg)
                    except Exception as e:
                        if tk and "timeout" in str(e).lower():
                            timed_out.add((rep, tk))
                        raise
                    posts = call.get("post") or post_all
                labels = r["labels"]
                # one gzip-appended JSON-lines file per task (few files on scratch)
                with gzip.open(labfile, "at") as fh:
                    fh.write(json.dumps({"call_id": cid, "rep": rep, "config": C.jdump(cfg),
                                         "labels": [str(x) for x in labels],
                                         "obs_names": [str(x) for x in pre.obs_names]}) + "\n")
                for pv in posts:
                    lab = C.refine(labels, coords, pv["refinement_threshold"]) if pv["refine_domains"] else labels
                    vkey = f"{pv['refine_domains']}_{pv['refinement_threshold']}"
                    s_lab = pd.Series(lab, index=pre.obs_names)
                    stab = np.nan
                    if rep == "full":
                        full_variant_labels.setdefault(cid, {})[vkey] = s_lab
                    else:
                        ref = full_variant_labels.get(cid, {}).get(vkey)
                        if ref is None:  # resumed run: rebuild full-rep labels from disk
                            fl, fobs = load_full_labels(cid)
                            fpre = get_pre("full")
                            if pv["refine_domains"]:
                                fl = C.refine(fl, fpre.obsm["spatial"], pv["refinement_threshold"])
                            ref = pd.Series(fl, index=fobs)
                        common_idx = s_lab.index.intersection(ref.index)
                        stab = C.ari(ref[common_idx].values, s_lab[common_idx].values)
                    a, n, gn = C.gt_scores(pre.obs_names, lab, gt)
                    match = ""
                    if pv["refine_domains"] and pv["refinement_threshold"] == 0.5 and r["refined_tool"] is not None:
                        match = bool((lab == r["refined_tool"]).all())
                    spec = cfg if strategy == "expert" else call["config"]
                    rows.append(dict(base, config=C.jdump({**cfg, **pv}), config_spec=C.jdump({**spec, **pv}),
                                     refine_domains=pv["refine_domains"],
                                     refinement_threshold=pv["refinement_threshold"],
                                     n_domains_found=len(set(lab)), kappa=C.spatial_kappa(lab, coords),
                                     stability=stab, gt_ari=a, gt_nmi=n, gt_n=gn,
                                     runtime_s=round(r["runtime"], 2), status="ok", error="",
                                     refine_matches_tool=match, n_warnings=len(r["warnings"])))
            except Exception as e:  # recorded, not retried automatically
                rows.append(dict(base, config=C.jdump(call.get("config")), config_spec=C.jdump(call.get("config")), refine_domains="",
                                 refinement_threshold="", n_domains_found="", kappa="", stability="",
                                 gt_ari="", gt_nmi="", gt_n="", runtime_s="", status="error",
                                 error=f"{type(e).__name__}: {str(e)[:300]}", refine_matches_tool="",
                                 n_warnings=""))
                traceback.print_exc()
            df = pd.DataFrame(rows, columns=COLS)
            df.to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
            ok = rows[0]["status"]
            print(f"[{time.strftime('%H:%M:%S')}] {cid} {rep} {ok} "
                  f"{rows[0].get('runtime_s')}s", flush=True)
    print("TASK DONE", flush=True)


if __name__ == "__main__":
    main()
