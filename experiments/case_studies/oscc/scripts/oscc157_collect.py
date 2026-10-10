#!/usr/bin/env python
"""Collect the tables of the OSCC ChatSpatial 1.5.7 run from the exported h5ad files (cluster compute node only).

Reads only what the tools wrote (export_data files) and the call record; computes no new analysis except the
case-study summaries named below.
Usage: oscc157_collect.py WORKDIR OUTDIR
Outputs (OUTDIR):
  qc_statistics.csv                     spots/genes before and after preprocess_data, medians
  tc_le_domain_scores.csv               per section and Leiden domain: TC/LE scores and assignment
  spots/{sid}.csv                       barcode, Leiden domain, tc_le_region, FN1 (log-normalized X)
  proportions/{sid}.csv                 FlashDeconv proportions + tc_le_region (input of the Fig. 2d,e statistics)
  deg_tc_vs_le_top500.csv               find_markers results (top 500 genes per direction, scanpy statistics)
  consensus_le_upregulated_genes.csv    genes with q < 0.05 and log2FC > 0.5 in >= 9 sections (case-study rule)
  consensus_tc_upregulated_genes.csv
  moranI_results_all_samples.csv        analyze_spatial_statistics(moran) table per section, with rank by I
  cellchat_{label}_lr_pairs.csv         CellChat significant interactions (pval < 0.05, prob > 0), as
                                        CellChat::subsetCommunication returns them
  cellchat_{label}_caf_cancer_interactions.csv   fibroblast <-> cancer subset (case-study regex)
"""
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

W, OUT = Path(sys.argv[1]), Path(sys.argv[2])
import os
RUN = W / os.environ.get("OSCC_RUN_DIR", "runs")
SAMPLES = os.environ.get("OSCC_SAMPLES", " ".join(f"s{i}" for i in range(1, 13))).split()
(OUT / "spots").mkdir(parents=True, exist_ok=True)
(OUT / "proportions").mkdir(parents=True, exist_ok=True)

calls = [json.loads(line) for line in open(RUN / "oscc157_tool_calls.jsonl")]
loaded = {c["sample"]: c["result"] for c in calls if c.get("record") == "tool_call" and c["step"] == "step1_load"}
prep = {c["sample"]: c["result"] for c in calls if c.get("record") == "tool_call" and c["step"] == "step1_preprocess"}
dom = {c["sample"]: c["result"] for c in calls if c.get("record") == "tool_call" and c["step"] == "step1_leiden_domains"}
tcle = {c["sample"]: c["summary"] for c in calls if c.get("record") == "post_processing"
        and c["step"] == "step1_tc_le_assignment"}

qc_rows, score_rows, deg_rows, moran_parts = [], [], [], []
for sid in SAMPLES:
    a = ad.read_h5ad(RUN / "final_h5ad" / f"{sid}.h5ad")
    obs = a.obs
    pr = prep[sid]
    pr = pr.get("result", pr)
    qc_rows.append({
        "sample": sid, "spots_before": loaded[sid]["n_cells"], "genes_before": loaded[sid]["n_genes"],
        "spots_after": a.n_obs, "genes_after": a.n_vars, "n_hvgs": pr.get("n_hvgs"),
        "median_genes": float(np.median(obs["n_genes_by_counts"])) if "n_genes_by_counts" in obs else np.nan,
        "median_counts": float(np.median(obs["total_counts"])) if "total_counts" in obs else np.nan,
        "median_mito_pct": float(np.median(obs["pct_counts_mt"])) if "pct_counts_mt" in obs else np.nan,
        "n_leiden_domains": dom[sid]["n_domains"],
        "n_TC": int((obs["tc_le_region"] == "TC").sum()), "n_LE": int((obs["tc_le_region"] == "LE").sum()),
        "n_Other": int((obs["tc_le_region"] == "Other").sum()),
        "qc_metrics": json.dumps(pr.get("qc_metrics", {})),
    })
    for r in tcle[sid]["domain_scores"]:
        score_rows.append({"sample": sid, **r})

    dkey = dom[sid]["domain_key"]
    fn1 = a.X[:, a.var_names.get_loc("FN1")] if "FN1" in a.var_names else None
    fn1 = (fn1.toarray().ravel() if sparse.issparse(fn1) else np.asarray(fn1).ravel()) if fn1 is not None else np.nan
    pd.DataFrame({"leiden_domain": obs[dkey].astype(str).values, "tc_le_region": obs["tc_le_region"].astype(str).values,
                  "FN1_lognorm": fn1}, index=pd.Index(a.obs_names, name="barcode")).to_csv(OUT / "spots" / f"{sid}.csv")

    key = "deconvolution_flashdeconv"
    cts = [str(c) for c in a.uns[f"{key}_cell_types"]]
    p = pd.DataFrame(np.asarray(a.obsm[key], dtype=float), index=pd.Index(a.obs_names, name="barcode"), columns=cts)
    p["tc_le_region"] = obs["tc_le_region"].astype(str).values
    p.to_csv(OUT / "proportions" / f"{sid}.csv")

    for k in [k for k in a.uns if k.startswith("rank_genes_groups_")]:
        rg = a.uns[k]
        g1 = rg["names"].dtype.names[0]
        n = len(rg["names"][g1])
        for i in range(n):
            deg_rows.append({"sample": sid, "group": g1, "reference": rg["params"].get("reference"),
                             "gene": rg["names"][g1][i], "score": float(rg["scores"][g1][i]),
                             "log2fc": float(rg["logfoldchanges"][g1][i]), "pvalue": float(rg["pvals"][g1][i]),
                             "pvalue_adj": float(rg["pvals_adj"][g1][i])})

    m = a.uns["moranI"].copy()
    m = m.sort_values("I", ascending=False)
    m["rank"] = np.arange(1, len(m) + 1)
    m["n_genes_tested"] = len(m)
    m["gene"] = m.index.astype(str)
    m["sample"] = sid
    moran_parts.append(m.reset_index(drop=True))
    print(sid, a.n_obs, a.n_vars, "TC", qc_rows[-1]["n_TC"], "LE", qc_rows[-1]["n_LE"], flush=True)

pd.DataFrame(qc_rows).to_csv(OUT / "qc_statistics.csv", index=False)
pd.DataFrame(score_rows).to_csv(OUT / "tc_le_domain_scores.csv", index=False)
deg = pd.DataFrame(deg_rows)
deg.to_csv(OUT / "deg_tc_vs_le_top500.csv", index=False)
n_deg_samples = deg["sample"].nunique()
for g in ("LE", "TC"):
    sig = deg[(deg["group"] == g) & (deg["pvalue_adj"] < 0.05) & (deg["log2fc"] > 0.5)]
    cnt = sig.groupby("gene")["sample"].nunique()
    cnt[cnt >= 9].sort_values(ascending=False).rename("sample").to_csv(
        OUT / f"consensus_{g.lower()}_upregulated_genes.csv", index_label="gene")
print("DEG sections:", n_deg_samples)
pd.concat(moran_parts, ignore_index=True).to_csv(OUT / "moranI_results_all_samples.csv", index=False)

for label in ("tc_le", "le_only"):
    path = RUN / "cellchat" / f"cellchat_{label}.h5ad"
    if not path.exists():
        continue
    c = ad.read_h5ad(path, backed="r")
    ccc = c.uns["ccc"]
    md_ = ccc["method_data"]
    prob, pval = np.asarray(md_["prob_matrix"]), np.asarray(md_["pval_matrix"])
    lr_names = [str(x) for x in md_["lr_names"]]
    cts = [str(x) for x in md_["cell_type_names"]]
    lrsig = pd.DataFrame(ccc["results"]).copy()
    lrsig.index = lrsig["interaction_name"].astype(str) if "interaction_name" in lrsig else lrsig.index.astype(str)
    rows = []
    for i, s in enumerate(cts):
        for j, t in enumerate(cts):
            for k, lr in enumerate(lr_names):
                if prob[i, j, k] > 0 and pval[i, j, k] < 0.05:
                    info = lrsig.loc[lr] if lr in lrsig.index else {}
                    rows.append({"source": s, "target": t, "ligand": info.get("ligand"), "receptor": info.get("receptor"),
                                 "prob": float(prob[i, j, k]), "pval": float(pval[i, j, k]), "interaction_name": lr,
                                 "interaction_name_2": info.get("interaction_name_2"),
                                 "pathway_name": info.get("pathway_name"), "annotation": info.get("annotation"),
                                 "evidence": info.get("evidence")})
    lr_df = pd.DataFrame(rows).sort_values(["source", "target", "prob"], ascending=[True, True, False])
    lr_df.to_csv(OUT / f"cellchat_{label}_lr_pairs.csv", index=False)
    fib = lr_df["source"].str.contains("Fibroblast|fibroblast|CAF") & lr_df["target"].str.contains("Cancer|cancer|Epithelial")
    fib |= lr_df["target"].str.contains("Fibroblast|fibroblast|CAF") & lr_df["source"].str.contains("Cancer|cancer|Epithelial")
    caf = lr_df[fib].sort_values("prob", ascending=False)
    caf.to_csv(OUT / f"cellchat_{label}_caf_cancer_interactions.csv", index=False)
    print(label, "significant interactions", len(lr_df), "fibroblast-cancer", len(caf),
          "cell types", cts, "stats", {k: v for k, v in dict(ccc["statistics"]).items()}, flush=True)
print("COLLECT_DONE")
