#!/usr/bin/env python3
"""Experiment B: execute LLM-issued calls with the real ChatSpatial tools (rules: ../expB_rules_2026-10-08.txt).

Stage 1 (default): read invocation JSONL files, validate each call with the current ChatSpatial Pydantic models
(manipulation check), canonicalise valid calls, execute every unique (prompt, canonical call) once in an isolated
subprocess on the prompt's fixture, verify the output, and write
  $EXPB_DIR/<EXPB_EXEC_DIR>/exec_cache.jsonl     one line per unique executed call
  $EXPB_DIR/<EXPB_EXEC_DIR>/trial_outcomes.csv   one line per trial (all outcomes of rules section 2)
Stage 2 (--worker SPEC OUT): execute one call (used internally).

Usage: python expB_execute.py [--files glob] [--jobs 3] [--timeout 2700] [--dry-run]
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
SCRIPTS = Path(__file__).resolve().parent
def _expb_dir() -> Path:
    """Experiment-B working directory: $EXPB_DIR, else $CHATSPATIAL_WORK_DIR/expB."""
    if os.environ.get("EXPB_DIR"):
        return Path(os.environ["EXPB_DIR"])
    if os.environ.get("CHATSPATIAL_WORK_DIR"):
        return Path(os.environ["CHATSPATIAL_WORK_DIR"]) / "expB"
    raise SystemExit("Set EXPB_DIR or CHATSPATIAL_WORK_DIR (cluster work directory holding expB/).")

EXPB = _expb_dir()
FIX = Path(os.environ.get("EXPB_FIXTURES", EXPB / "fixtures"))
EXEC = EXPB / os.environ.get("EXPB_EXEC_DIR", "exec")
EXEC.mkdir(parents=True, exist_ok=True)
CACHE = EXEC / "exec_cache.jsonl"
COMPAT = Path(os.environ.get("EXPB_COMPAT", SCRIPTS / "compat"))  # SpaGCN scipy alias shim
# The local env carries torch_scatter/torch_sparse/... compiled for torch 2.8 while torch is 2.13; importing them
# segfaults (kills STAGATE). They are optional for torch_geometric, so they are masked with ImportError shims.
PYG_SHIMS = Path(os.environ.get("EXPB_PYG_SHIMS", SCRIPTS / "pyg_shims"))

FIXTURES = {
    "spatial_domains": dict(target=FIX / "dlpfc_151673.h5ad", type="visium"),
    "deconvolution_rctd": dict(target=FIX / "oscc_s1_celltype.h5ad", type="visium", ref=FIX / "sc_ref_puram.h5ad"),
    "svg_detection": dict(target=FIX / "merfish_slice.h5ad", type="merfish"),
    "cellchat": dict(target=FIX / "oscc_s1_celltype.h5ad", type="visium"),
    "moran_i": dict(target=FIX / "oscc_s1_celltype.h5ad", type="visium"),
    "spatial_plot": dict(target=FIX / "oscc_s1_celltype.h5ad", type="visium"),
    "trajectory": dict(target=FIX / "pancreas_velocity.h5ad", type="other"),
    "cnv": dict(target=FIX / "oscc_s1_celltype.h5ad", type="visium"),
}
SERVER_FN = {
    "identify_spatial_domains": "identify_spatial_domains",
    "deconvolve_data": "deconvolve_data",
    "find_spatial_genes": "find_spatial_genes",
    "analyze_cell_communication": "analyze_cell_communication",
    "analyze_spatial_statistics": "analyze_spatial_statistics",
    "visualize_data": "visualize_data",
    "analyze_trajectory_data": "analyze_trajectory_data",
    "analyze_cnv": "analyze_cnv",
}


def param_models():
    from chatspatial.models.data import (CellCommunicationParameters, CNVParameters, DeconvolutionParameters,
                                         SpatialDomainParameters, SpatialStatisticsParameters,
                                         SpatialVariableGenesParameters, TrajectoryParameters,
                                         VisualizationParameters)
    return {
        "identify_spatial_domains": SpatialDomainParameters,
        "deconvolve_data": DeconvolutionParameters,
        "find_spatial_genes": SpatialVariableGenesParameters,
        "analyze_cell_communication": CellCommunicationParameters,
        "analyze_spatial_statistics": SpatialStatisticsParameters,
        "visualize_data": VisualizationParameters,
        "analyze_trajectory_data": TrajectoryParameters,
        "analyze_cnv": CNVParameters,
    }


# ---------------------------------------------------------------------------
# Worker: execute one call
# ---------------------------------------------------------------------------

def _finite_frac(x):
    import numpy as np
    x = np.asarray(x, dtype=float)
    return float(np.isfinite(x).mean()) if x.size else 0.0


def verify(tool: str, adata, result, n_obs: int) -> tuple[bool, dict]:
    """Rules section 2.6. Returns (verified, details incl. small artifacts)."""
    import numpy as np
    import pandas as pd
    d: dict = {}
    if tool == "identify_spatial_domains":
        key = result.domain_key
        if key not in adata.obs:
            return False, {"reason": f"missing obs[{key}]"}
        lab = adata.obs[key].astype(str).values
        d["labels"] = lab.tolist()
        d["n_labels"] = int(len(set(lab)))
        return (len(lab) == n_obs and d["n_labels"] >= 2), d
    if tool == "deconvolve_data":
        key = result.proportions_key
        if key not in adata.obsm:
            return False, {"reason": f"missing obsm[{key}]"}
        P = adata.obsm[key]
        P = P.values if isinstance(P, pd.DataFrame) else np.asarray(P)
        rs = P.sum(1)
        d.update(shape=list(P.shape), rowsum_min=float(rs.min()), rowsum_max=float(rs.max()),
                 proportions=np.round(P, 6).tolist())
        ok = P.shape[0] == n_obs and P.shape[1] >= 2 and np.isfinite(P).all() and np.all(np.abs(rs - 1) <= 0.05)
        return bool(ok), d
    if tool == "find_spatial_genes":
        genes = list(result.spatial_genes or [])
        d.update(n_genes=len(genes), genes=genes[:200])
        return (len(genes) > 0 and all(g in adata.var_names for g in genes)), d
    if tool == "analyze_cell_communication":
        key = result.results_key
        stored = key is not None and (key in adata.uns or key in adata.obsm or key in adata.obs)
        n = result.n_lr_pairs or 0
        d.update(n_lr_pairs=n, n_significant=result.n_significant_pairs, results_key=key, stored=bool(stored),
                 top=list(result.top_lr_pairs or [])[:20])
        return bool(n >= 1 and stored), d
    if tool == "analyze_spatial_statistics":
        key = result.results_key
        d.update(results_key=key, n_features=result.n_features_analyzed, top=list(result.top_features or [])[:20])
        obj = None
        for store in (adata.uns, adata.obsm, adata.obs, adata.var):
            if key is not None and key in store:
                obj = store[key]
                break
        if obj is None:
            return False, dict(d, reason="results_key not stored")
        if isinstance(obj, pd.DataFrame):
            num = obj.select_dtypes("number")
            d["table_shape"] = list(obj.shape)
            return bool(len(obj) > 0 and num.size > 0 and _finite_frac(num.values) > 0), d
        if isinstance(obj, (pd.Series, np.ndarray)):
            return bool(np.size(obj) > 0 and _finite_frac(np.asarray(obj)) > 0), d
        return bool(obj is not None and len(obj) > 0), d
    if tool == "visualize_data":
        from PIL import Image
        import re as _re
        m = _re.search(r"saved:\s*(\S+)", str(result)) if result is not None else None
        path = m.group(1) if m else str(result).strip()
        d["image_path"] = path
        if not path or not os.path.exists(path):
            return False, dict(d, reason="no image file")
        with Image.open(path) as im:
            d["size"] = list(im.size)
            im.load()
        return bool(im.size[0] > 0 and im.size[1] > 0), d
    if tool == "analyze_trajectory_data":
        key = result.pseudotime_key
        if key is None or key not in adata.obs:
            return False, {"reason": f"missing obs[{key}]"}
        ff = _finite_frac(adata.obs[key].values)
        d.update(finite_frac=ff, pseudotime=[float(x) for x in adata.obs[key].values])
        return ff >= 0.5, d
    if tool == "analyze_cnv":
        keys = [k for k in adata.obsm.keys() if "cnv" in k.lower()]
        d["obsm_keys"] = keys
        if not keys:
            return False, dict(d, reason="no CNV obsm")
        M = adata.obsm[keys[0]]
        M = M.toarray() if hasattr(M, "toarray") else np.asarray(M)
        d["shape"] = list(M.shape)
        return bool(M.shape[0] == n_obs and M.shape[1] >= 1 and np.isfinite(M).all()), d
    return False, {"reason": "unknown tool"}


async def _worker(spec: dict) -> dict:
    import anndata as ad
    from chatspatial import server
    PM = param_models()
    fx = FIXTURES[spec["prompt_id"]]
    adata = ad.read_h5ad(fx["target"])
    n_obs = adata.n_obs
    dm = server.data_manager
    await dm.create_dataset(adata, data_id="target", metadata={"type": fx["type"]})
    if "ref" in fx:
        await dm.create_dataset(ad.read_h5ad(fx["ref"]), data_id="sc_ref", metadata={"type": "other"})
    params = PM[spec["tool"]](**spec["params"])
    fn = getattr(server, SERVER_FN[spec["tool"]])
    t0 = time.time()
    try:
        result = await fn("target", params=params)
    except Exception as e:  # explicit failure (user sees an error)
        return dict(exec_success=False, error_type=type(e).__name__, error=str(e)[:1500],
                    seconds=round(time.time() - t0, 1))
    secs = round(time.time() - t0, 1)
    adata = (await dm.get_dataset("target"))["adata"]
    try:
        ok, details = verify(spec["tool"], adata, result, n_obs)
    except Exception as e:
        ok, details = False, {"reason": f"verify error {type(e).__name__}: {e}"}
    res_dump = None
    try:
        res_dump = result.model_dump(mode="json") if hasattr(result, "model_dump") else str(result)
    except Exception:
        res_dump = str(result)[:2000]
    return dict(exec_success=True, error_type=None, error=None, seconds=secs, output_verified=bool(ok),
                details=details, result=res_dump, warnings=list(getattr(result, "warnings", []) or []))


def worker_main(spec_path: str, out_path: str):
    spec = json.loads(Path(spec_path).read_text())
    out = asyncio.run(_worker(spec))
    Path(out_path).write_text(json.dumps(out, default=str))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def canonical(tool: str, params_obj) -> str:
    return json.dumps({"tool": tool, "params": params_obj.model_dump(mode="json")}, sort_keys=True, default=str)


def load_trials(patterns: list[str]) -> list[dict]:
    recs = {}
    for pat in patterns:
        for f in sorted(glob.glob(str(pat))):
            if "pilot" in f:
                continue
            for line in open(f):
                r = json.loads(line)
                if r.get("error") is not None:
                    continue
                recs[(r["label"], r["trial_key"])] = r
    return list(recs.values())


def prepare(trials: list[dict]):
    """Pydantic manipulation check + canonical call per trial."""
    PM = param_models()
    for r in trials:
        r["pydantic_valid"] = False
        r["pydantic_error"] = None
        r["canonical_call"] = None
        r["effective_params"] = None
        tool, params = r.get("canonical_tool"), r.get("params")
        if r.get("parse_status") != "ok" or tool not in PM or not isinstance(params, dict):
            continue
        try:
            obj = PM[tool](**params)
        except Exception as e:
            r["pydantic_error"] = str(e)[:500]
            continue
        r["pydantic_valid"] = True
        r["canonical_call"] = canonical(tool, obj)
        r["effective_params"] = obj.model_dump(mode="json")
    return trials


def call_id(prompt_id: str, ccall: str) -> str:
    return hashlib.sha1(f"{prompt_id}|{ccall}".encode()).hexdigest()[:16]


def run_unique(trials, jobs: int, timeout: int, dry: bool):
    done = {}
    if CACHE.exists():
        for line in open(CACHE):
            c = json.loads(line)
            done[c["call_id"]] = c
    todo = {}
    for r in trials:
        if r.get("experiment") != "main" or not r["pydantic_valid"] or r["canonical_tool"] != r["expected_tool"]:
            continue
        cid = call_id(r["prompt_id"], r["canonical_call"])
        if cid not in done:
            todo[cid] = dict(call_id=cid, prompt_id=r["prompt_id"], tool=r["canonical_tool"], params=r["params"],
                             canonical_call=r["canonical_call"])
    print(f"unique calls: {len(done)} cached, {len(todo)} to execute", flush=True)
    if dry:
        from collections import Counter
        print(Counter(s["prompt_id"] for s in todo.values()))
        return done
    tmp = EXEC / "tmp"
    tmp.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(PYG_SHIMS), str(COMPAT), str(SCRIPTS)] +
                                                      [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]))
    queue = list(todo.values())
    running = {}
    while queue or running:
        while queue and len(running) < jobs:
            s = queue.pop(0)
            sp, op = tmp / f"{s['call_id']}.spec.json", tmp / f"{s['call_id']}.out.json"
            sp.write_text(json.dumps(s))
            log = open(tmp / f"{s['call_id']}.log", "w")
            p = subprocess.Popen([sys.executable, "-W", "ignore", __file__, "--worker", str(sp), str(op)],
                                 stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(tmp))
            running[s["call_id"]] = (p, s, time.time(), op)
        time.sleep(2)
        for cid, (p, s, t0, op) in list(running.items()):
            rc = p.poll()
            elapsed = time.time() - t0
            if rc is None and elapsed < timeout:
                continue
            if rc is None:
                p.kill()
                out = dict(exec_success=False, error_type="Timeout", error=f"exceeded {timeout}s", seconds=round(elapsed, 1))
            elif op.exists():
                out = json.loads(op.read_text())
            else:
                tail = (tmp / f"{cid}.log").read_text()[-1500:]
                out = dict(exec_success=False, error_type="WorkerCrash", error=tail, seconds=round(elapsed, 1))
            rec = dict(s, **out, executed_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
            with open(CACHE, "a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
            done[cid] = rec
            del running[cid]
            print(f"  [{len(done)}] {s['prompt_id']} {s['tool']} {json.dumps(s['params'])[:80]} -> "
                  f"exec={out['exec_success']} verified={out.get('output_verified')} {out.get('error_type') or ''} "
                  f"{out['seconds']}s", flush=True)
    return done


# Rules 2.7 (prompt-derived; executed values = validated object with defaults)
DL_METHODS = {"spagcn", "stagate", "graphst"}


def silent_violations(prompt_id: str, ep: dict) -> list[str]:
    v = []
    if prompt_id == "spatial_domains" and ep.get("method") not in DL_METHODS:
        v.append(f"method={ep.get('method')} not deep learning")
    if prompt_id == "deconvolution_rctd":
        if ep.get("method") != "rctd":
            v.append(f"method={ep.get('method')}!=rctd")
        if ep.get("reference_data_id") != "sc_ref":
            v.append("reference_data_id!=sc_ref")
        if ep.get("cell_type_key") != "celltype":
            v.append("cell_type_key!=celltype")
    if prompt_id == "cellchat":
        if ep.get("method") != "cellchat_r":
            v.append(f"method={ep.get('method')}!=cellchat_r")
        if ep.get("species") != "human":
            v.append("species!=human")
        if ep.get("cell_type_key") != "cell_type":
            v.append("cell_type_key!=cell_type")
    if prompt_id == "moran_i" and ep.get("analysis_type") != "moran":
        v.append(f"analysis_type={ep.get('analysis_type')}!=moran")
    if prompt_id == "trajectory" and ep.get("method") != "cellrank":
        v.append(f"method={ep.get('method')}!=cellrank")
    if prompt_id == "spatial_plot":
        feat = ep.get("feature")
        feats = feat if isinstance(feat, list) else [feat]
        if "cell_type" not in feats and ep.get("cluster_key") != "cell_type":
            v.append("not plotting cell_type")
        if ep.get("basis") not in ("spatial", None):
            v.append(f"basis={ep.get('basis')}")
    if prompt_id == "cnv":
        if ep.get("reference_key") != "cell_type":
            v.append("reference_key!=cell_type")
        if set(ep.get("reference_categories") or []) != {"T cells", "B cells"}:
            v.append("reference_categories!={T cells,B cells}")
    return v


def outcomes(trials, cache):
    import pandas as pd
    rows = []
    for r in trials:
        row = {k: r.get(k) for k in ("label", "model", "backend", "response_model", "experiment", "arm", "mechanism",
                                     "info", "prompt_id", "rep", "expected_tool", "canonical_tool", "raw_tool_name",
                                     "parse_status", "pydantic_valid", "temperature", "latency_s")}
        row["params"] = json.dumps(r.get("params"), sort_keys=True) if r.get("params") is not None else None
        row["tool_correct"] = r.get("canonical_tool") == r["expected_tool"]
        fmt_fail = r.get("parse_status") != "ok" or r.get("canonical_tool") is None
        row["error_class"] = "correct" if row["tool_correct"] else ("format_failure" if fmt_fail else "mis_selection")
        row["call"] = json.dumps([r.get("canonical_tool"), r.get("params")], sort_keys=True, default=str)
        row["canonical_call"] = r.get("canonical_call")
        ep = r.get("effective_params") or {}
        row["effective_params"] = json.dumps(ep, sort_keys=True) if ep else None
        row["executed"] = row["exec_success"] = row["completed"] = row["silent_failure"] = False
        row["explicit_failure"] = False
        row["violations"] = None
        row["exec_error_type"] = None
        if r.get("experiment") == "main" and r.get("pydantic_valid") and row["tool_correct"]:
            c = cache.get(call_id(r["prompt_id"], r["canonical_call"]))
            if c is not None:
                row["executed"] = True
                row["exec_success"] = bool(c["exec_success"])
                row["exec_error_type"] = c.get("error_type")
                if c["exec_success"]:
                    verified = bool(c.get("output_verified"))
                    row["completed"] = verified
                    viol = silent_violations(r["prompt_id"], ep)
                    if not verified:
                        viol = ["output not verified"] + viol
                    row["violations"] = "; ".join(viol) if viol else None
                    row["silent_failure"] = bool(viol)
                else:
                    row["explicit_failure"] = True
        elif r.get("experiment") == "main" and row["tool_correct"] and not r.get("pydantic_valid"):
            row["explicit_failure"] = True  # rejected by validation with an error message
            row["exec_error_type"] = "ValidationError"
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(EXEC / "trial_outcomes.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=2)
    ap.add_argument("--files", nargs="*", default=[str(EXPB / "invocation" / "*.jsonl")])
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=2700)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-exec", action="store_true", help="only (re)build trial_outcomes.csv from the cache")
    a = ap.parse_args()
    if a.worker:
        worker_main(*a.worker)
        return
    trials = prepare(load_trials(a.files))
    print(f"{len(trials)} trials loaded")
    if a.no_exec:
        cache = {}
        if CACHE.exists():
            for line in open(CACHE):
                c = json.loads(line)
                cache[c["call_id"]] = c
    else:
        cache = run_unique(trials, a.jobs, a.timeout, a.dry_run)
    if not a.dry_run:
        df = outcomes(trials, cache)
        print(df[df.experiment == "main"].groupby(["label", "mechanism", "info"])[["tool_correct", "pydantic_valid", "completed", "silent_failure"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
