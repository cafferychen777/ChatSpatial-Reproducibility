#!/usr/bin/env python3
"""Experiment B-X analysis (rules: expBX_rules_2026-10-09.txt, sections 2.7 and 3.3-3.6).

  python expBX_analysis.py h2h   [--tasks tasks/h2h_full.jsonl]
  python expBX_analysis.py probe [--tasks tasks/probe_full.jsonl]
  python expBX_analysis.py cost  [--tasks ...]
Outputs under <ROOT>/analysis/. Runs in the chatspatial_v155 env.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import expBX_common as C  # noqa: E402

OUT = C.ROOT / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
NB = 10000
GT_DIR = C.EXPA_DLPFC / "ground_truth"
INPUT_OBS = {"sample_id", "array_row", "array_col", "cell_count", "expr_chrM", "expr_chrM_ratio",
             "pxl_row_in_fullres", "pxl_col_in_fullres"}
NAME_PRI = [("domain",), ("spatial",), ("cluster", "leiden", "louvain", "kmeans", "mclust", "spagcn", "stagate",
                                         "graphst", "banksy", "bayes", "gmm")]
METHOD_KW = {"spagcn": r"spagcn", "stagate": r"stagate", "graphst": r"graphst", "banksy": r"banksy",
             "bayesspace": r"bayesspace", "mclust": r"mclust", "leiden": r"leiden", "louvain": r"louvain",
             "kmeans": r"kmeans|k-means|KMeans", "agglomerative": r"agglomerative|hierarchical|linkage\(",
             "gmm": r"gaussianmixture|gaussian mixture", "squidpy_nhood": r"spatial_neighbors"}


def load_tasks(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def jread(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


# ----------------------------------------------------------------------------- labels
def agent_labels(rd: Path, n_input: int):
    """Rule 3.3 for agents -> (labels Series or None, info dict)."""
    import anndata as ad
    out = rd / "out"
    req = out / "result.h5ad"
    info = {"requested_file": req.exists(), "file_used": None, "column": None, "candidates": []}
    files = [req] if req.exists() else sorted(out.rglob("*.h5ad"), key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files:
        try:
            a = ad.read_h5ad(f, backed="r")
            obs = a.obs.copy()
            a.file.close()
        except Exception as e:
            info.setdefault("read_errors", []).append(f"{f.name}: {type(e).__name__}")
            continue
        cands = []
        for c in obs.columns:
            if c in INPUT_OBS:
                continue
            s = obs[c]
            nun = s.dropna().nunique()
            ok_type = (isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object or pd.api.types.is_integer_dtype(s)
                       or pd.api.types.is_bool_dtype(s))
            if 2 <= nun <= 50 and ok_type:
                cands.append(c)
        chosen = None
        for group in NAME_PRI:
            hits = [c for c in cands if any(k in c.lower() for k in group)]
            if hits:
                chosen = hits[-1]
                break
        if chosen is None and cands:
            chosen = None  # qualifying but unnamed columns are not used by rule 3.3
        if chosen is None:
            legacy = [c for c in obs.columns if any(k in c.lower() for k in ("domain", "leiden", "cluster", "louvain"))]
            chosen = legacy[0] if legacy else None
        info.update(file_used=str(f.relative_to(rd)), candidates=cands, column=chosen, n_obs_file=len(obs))
        if chosen is not None:
            lab = obs[chosen].astype(str)
            lab = lab[obs[chosen].notna().values]
            return lab, {**info, "cand_series": {c: obs[c].astype(str)[obs[c].notna().values] for c in cands}}
        return None, info
    return None, info


def gt(sid):
    return pd.read_csv(GT_DIR / f"{sid}_labels.csv", index_col=0)["ground_truth"]


def score(lab: pd.Series, g: pd.Series):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    common = lab.index.intersection(g.index[g.notna()])
    if len(common) < 10:
        return np.nan, np.nan, len(common)
    return (adjusted_rand_score(g[common].astype(str), lab[common]),
            normalized_mutual_info_score(g[common].astype(str), lab[common]), len(common))


def code_text(rd: Path) -> str:
    """Executed code only: STAgent tool-call arguments; SpatialAgent <act> blocks (rule 3.4)."""
    res = jread(rd / "result.json")
    parts = [json.dumps(tc.get("args"), default=str) for tc in res.get("tool_calls") or []]
    for t in res.get("assistant_texts") or []:
        parts += re.findall(r"<act>(.*?)(?:</act>|$)", t or "", flags=re.S)
    return "\n".join(parts)


def h2h(tasks):
    import anndata as ad
    n_input = {}
    rows, label_store = [], {}
    for t in tasks:
        rd = C.run_dir(t)
        st, res = jread(rd / "status.json"), jread(rd / "result.json")
        sid, sysn = t["unit"], t["system"]
        if sid not in n_input:
            a = ad.read_h5ad(C.dlpfc_input(sid), backed="r")
            n_input[sid] = (a.n_obs, list(a.obs_names))
            a.file.close()
        n_obs, names = n_input[sid]
        row = dict(task_id=t["task_id"], system=sysn, section=sid, rep=t["rep"], finished=st.get("finished", False),
                   timed_out=st.get("timed_out"), wall_time=st.get("wall_time"), cost_usd=res.get("cost_usd"))
        lab, lab_alt = None, None
        if sysn in ("chatspatial", "chatspatial_single"):
            f = rd / "labels.csv"
            if sysn == "chatspatial":
                row["success"] = (not res.get("harness_error")) and res.get("stop_reason") == "end_turn" and not st.get("timed_out")
                row["method"] = res.get("method")
                row["domain_params"] = json.dumps(res.get("domain_params"))
                row["n_domain_calls"] = len(res.get("domain_calls") or [])
                row["tool_sequence"] = ">".join(res.get("tool_sequence") or [])
            else:
                row["success"] = bool(res.get("success"))
                row["method"] = res.get("method")
                row["domain_params"] = json.dumps(res.get("validated_params"))
            if f.exists():
                df = pd.read_csv(f, index_col=0, dtype=str)
                lab = df["label"]
                alt_col = "label_unrefined" if sysn == "chatspatial" else "label_refined"
                lab_alt = df[alt_col] if alt_col in df else None
                row["verified"] = bool(lab.notna().all() and lab.nunique() >= 2 and len(lab) == n_obs)
            else:
                row["verified"] = False
        else:
            row["success"] = bool(res.get("success"))
            lab, info = agent_labels(rd, n_obs)
            row.update(requested_file=info.get("requested_file"), file_used=info.get("file_used"),
                       column=info.get("column"), candidates="|".join(info.get("candidates", [])))
            row["verified"] = bool(info.get("requested_file") and info.get("file_used") == "out/result.h5ad"
                                   and lab is not None and lab.nunique() >= 2
                                   and len(lab.index.intersection(names)) >= 0.95 * n_obs)
            txt = code_text(rd)
            row["methods_in_code"] = "|".join(k for k, pat in METHOD_KW.items() if re.search(pat, txt, flags=re.I))
            if lab is not None:
                g = gt(sid)
                cs = info.get("cand_series", {})
                aris = [score(s, g)[0] for s in cs.values()] if cs else []
                row["ari_cand_min"] = np.nanmin(aris) if aris else np.nan
                row["ari_cand_max"] = np.nanmax(aris) if aris else np.nan
        if lab is not None:
            ari, nmi, n = score(lab, gt(sid))
            row.update(ari=ari, nmi=nmi, n_scored=n, n_labels=lab.nunique(), coverage=len(lab.index.intersection(names)) / n_obs)
            label_store[t["task_id"]] = lab
        if lab_alt is not None:
            row["ari_alt_labels"] = score(lab_alt, gt(sid))[0]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "h2h_per_run.csv", index=False)

    # cross-replicate ARI per section
    from sklearn.metrics import adjusted_rand_score
    xr = []
    for (sysn, sid), g in df.groupby(["system", "section"]):
        labs = [label_store[t] for t in g["task_id"] if t in label_store]
        vals = []
        for a, b in itertools.combinations(labs, 2):
            common = a.index.intersection(b.index)
            if len(common) >= 10:
                vals.append(adjusted_rand_score(a[common], b[common]))
        xr.append(dict(system=sysn, section=sid, xrep_ari=np.mean(vals) if vals else np.nan, n_pairs=len(vals)))
    xr = pd.DataFrame(xr)
    xr.to_csv(OUT / "h2h_cross_rep.csv", index=False)

    rng = np.random.default_rng(C.SEED)
    systems = [s for s in C.SYSTEMS_H2H if s in set(df.system)]
    secs = sorted(df.section.unique())
    summ = {"n_runs": int(len(df)), "systems": {}}
    per_sec = {}
    for s in systems:
        d = df[df.system == s]
        ps = d.groupby("section")["ari"].mean()
        ps_itt = d.assign(ari=d["ari"].fillna(0)).groupby("section")["ari"].mean()
        psn = d.groupby("section")["nmi"].mean()
        per_sec[s] = {"ari": ps, "itt": ps_itt, "nmi": psn,
                      "xrep": xr[xr.system == s].set_index("section")["xrep_ari"]}
        summ["systems"][s] = dict(
            n=int(len(d)), success=f"{int(d.success.sum())}/{len(d)}", verified=f"{int(d.verified.sum())}/{len(d)}",
            scorable=f"{int(d.ari.notna().sum())}/{len(d)}", timeouts=int(d.timed_out.fillna(False).sum()),
            ari_mean=float(ps.mean()), ari_itt_mean=float(ps_itt.mean()), nmi_mean=float(psn.mean()),
            xrep_ari_mean=float(per_sec[s]["xrep"].mean()),
            ari_per_section={k: round(float(v), 4) for k, v in ps.items()},
            method_counts=(d["method"].fillna("none").value_counts().to_dict() if "method" in d and s.startswith("chatspatial")
                           else d.get("methods_in_code", pd.Series(dtype=str)).fillna("").value_counts().to_dict()),
            ari_ci=boot_mean(d, "ari", rng))
    # paired differences chatspatial(_single) - agent
    comps = {}
    for a_ in ("chatspatial", "chatspatial_single"):
        for b_ in ("stagent", "spatialagent"):
            if a_ in systems and b_ in systems:
                for metric in ("ari", "itt", "nmi", "xrep"):
                    comps[f"{a_}-{b_}:{metric}"] = paired(df, xr, a_, b_, metric, per_sec, rng)
    summ["paired"] = comps
    (OUT / "h2h_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))


def boot_mean(d, col, rng):
    secs = sorted(d.section.unique())
    by = {s: d[d.section == s][col].dropna().values for s in secs}
    stats = []
    for _ in range(NB):
        pick = rng.choice(secs, size=len(secs), replace=True)
        m = []
        for s in pick:
            v = by[s]
            if len(v):
                m.append(rng.choice(v, size=len(v), replace=True).mean())
        stats.append(np.mean(m) if m else np.nan)
    return [float(np.nanpercentile(stats, 2.5)), float(np.nanpercentile(stats, 97.5))]


def paired(df, xr, a, b, metric, per_sec, rng):
    secs = sorted(set(per_sec[a][metric].dropna().index) & set(per_sec[b][metric].dropna().index))
    diffs = np.array([per_sec[a][metric][s] - per_sec[b][metric][s] for s in secs])
    out = {"n_sections": len(secs), "mean_diff": float(diffs.mean()) if len(diffs) else None,
           "n_sections_positive": int((diffs > 0).sum())}
    if metric == "xrep":
        boots = [rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(NB)]
    else:
        col = {"ari": "ari", "itt": "ari", "nmi": "nmi"}[metric]
        da, db = df[df.system == a], df[df.system == b]
        if metric == "itt":
            da, db = da.assign(ari=da.ari.fillna(0)), db.assign(ari=db.ari.fillna(0))
        va = {s: da[da.section == s][col].dropna().values for s in secs}
        vb = {s: db[db.section == s][col].dropna().values for s in secs}
        boots = []
        for _ in range(NB):
            pick = rng.choice(secs, size=len(secs), replace=True)
            boots.append(np.mean([rng.choice(va[s], len(va[s])).mean() - rng.choice(vb[s], len(vb[s])).mean() for s in pick]))
    out["ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    # exact sign-flip permutation over sections
    if len(diffs):
        obs = abs(diffs.mean())
        cnt = 0
        tot = 0
        for signs in itertools.product([1, -1], repeat=len(diffs)):
            tot += 1
            if abs((diffs * np.array(signs)).mean()) >= obs - 1e-12:
                cnt += 1
        out["p_signflip"] = cnt / tot
    return out


# ----------------------------------------------------------------------------- probes
def probe(tasks):
    J = {}
    jf = C.ROOT / "judge" / ("judgements.jsonl" if STEM == "probe_full" else f"judgements_{STEM}.jsonl")
    for l in open(jf):
        r = json.loads(l)
        if "class" in r:
            J[r["task_id"]] = r
    audit = {}
    af = C.ROOT / "judge" / "audit_overrides.jsonl"
    if af.exists():
        for l in open(af):
            r = json.loads(l)
            audit[r["task_id"]] = r
    rows = []
    for t in tasks:
        j = J.get(t["task_id"])
        st = jread(C.run_dir(t) / "status.json")
        cls_j = j["class"] if j else "not_judged"
        cls = audit[t["task_id"]]["class"] if t["task_id"] in audit else cls_j
        rows.append(dict(task_id=t["task_id"], system=t["system"], probe=t["unit"], rep=t["rep"],
                         judged_class=cls_j, audited_class=cls, overridden=t["task_id"] in audit,
                         timed_out=st.get("timed_out"), **({f"f_{k}": v for k, v in j["fields"].items() if k != "evidence"} if j else {})))
    df = pd.DataFrame(rows)
    df["silent"] = (df.audited_class == "silent_failure").astype(int)
    df["silent_judged"] = (df.judged_class == "silent_failure").astype(int)
    df.to_csv(OUT / "probe_per_run.csv", index=False)
    rng = np.random.default_rng(C.SEED)
    probes = [p["id"] for p in C.PROBES]
    arms = {"chatspatial": ["chatspatial"], "stagent": ["stagent"], "spatialagent": ["spatialagent"],
            "agents_pooled": ["stagent", "spatialagent"]}
    summ = {"arms": {}, "diff": {}, "class_table": df.groupby(["system", "audited_class"]).size().unstack(fill_value=0).to_dict()}
    rate = {}
    for arm, systems in arms.items():
        d = df[df.system.isin(systems)]
        by = {p: d[d.probe == p]["silent"].values for p in probes}
        rate[arm] = {p: by[p].mean() for p in probes}
        boots = []
        for _ in range(NB):
            pick = rng.choice(probes, size=len(probes), replace=True)
            boots.append(np.mean(np.concatenate([rng.choice(by[p], len(by[p])) for p in pick])))
        summ["arms"][arm] = dict(silent=f"{int(d.silent.sum())} of {len(d)}", rate=float(d.silent.mean()),
                                 ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                 silent_judged_unaudited=f"{int(d.silent_judged.sum())} of {len(d)}",
                                 per_probe={p: f"{int(by[p].sum())}/{len(by[p])}" for p in probes},
                                 correct_behaviour=f"{int(d.audited_class.isin(['explicit_refusal','disclosed_deviation','valid_completion']).sum())} of {len(d)}")
    for a_, b in [("chatspatial", x) for x in ("stagent", "spatialagent", "agents_pooled")]:
        dc = df[df.system == a_]
        if dc.empty:
            continue
        db = df[df.system.isin(arms[b])]
        va = {p: dc[dc.probe == p]["silent"].values for p in probes}
        vb = {p: db[db.probe == p]["silent"].values for p in probes}
        diffs = np.array([va[p].mean() - vb[p].mean() for p in probes])
        boots = []
        for _ in range(NB):
            pick = rng.choice(probes, size=len(probes), replace=True)
            boots.append(np.mean(np.concatenate([rng.choice(va[p], len(va[p])) for p in pick]))
                         - np.mean(np.concatenate([rng.choice(vb[p], len(vb[p])) for p in pick])))
        obs = abs(diffs.mean())
        cnt = sum(abs((diffs * np.array(s)).mean()) >= obs - 1e-12 for s in itertools.product([1, -1], repeat=len(diffs)))
        summ["diff"][f"{a_}-{b}"] = dict(diff=float(dc.silent.mean() - db.silent.mean()),
                                                ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                                p_signflip=cnt / 2 ** len(diffs))
    (OUT / "probe_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))


def cost(tasks):
    rows = []
    for t in tasks:
        rd = C.run_dir(t)
        res = jread(rd / "result.json")
        c = res.get("cost_usd")
        if c is None:  # killed runs: sum the incremental logs
            c = 0.0
            for name in ("api_log.jsonl", "transcript.jsonl"):
                p = rd / name
                if p.exists():
                    for l in open(p):
                        e = json.loads(l)
                        if e.get("event") == "api_response":
                            c += C.usage_cost(e.get("usage") or {})
        rows.append(dict(task_id=t["task_id"], part=t["part"], system=t["system"], unit=t["unit"], cost_usd=c,
                         wall=jread(rd / "status.json").get("wall_time")))
    df = pd.DataFrame(rows)
    print(df.groupby(["part", "system"]).agg(n=("cost_usd", "size"), total=("cost_usd", "sum"), mean=("cost_usd", "mean"),
                                             wall_mean=("wall", "mean")).round(3).to_string())
    print("TOTAL", round(df.cost_usd.sum(), 2))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["h2h", "probe", "cost"])
    ap.add_argument("--tasks")
    a = ap.parse_args()
    default = {"h2h": "h2h_full", "probe": "probe_full", "cost": "h2h_full"}[a.what]
    tpath = Path(a.tasks or (C.ROOT / "tasks" / f"{default}.jsonl"))
    STEM = tpath.stem
    if STEM not in ("h2h_full", "probe_full"):
        OUT = C.ROOT / "analysis" / STEM
        OUT.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(tpath)
    {"h2h": h2h, "probe": probe, "cost": cost}[a.what](tasks)
