#!/usr/bin/env python
"""Collect the tables of the HGSOC ChatSpatial 1.5.7 run from the exported h5ad files (cluster compute node only).

Reads only what the tools wrote (export_data files) and the call record; computes no new analysis.
Usage: hgsoc157_collect.py RUN_DIR OUTDIR
Outputs (OUTDIR):
  qc_statistics.csv           spots/genes before and after preprocess_data, medians, analyze_cnv statistics
  spots/{pid}.csv             barcode, full-resolution pixel coordinates, RCTD proportions (deconvolve_data),
                              rctd_tumor_class (post-processing label), cnv_score and cnv_clusters (analyze_cnv)
  cnv_cluster_markers_{pid}.csv   find_markers on cnv_clusters (top 20 genes per cluster, scanpy statistics)
  moranI_{pid}.csv            analyze_spatial_statistics(moran) table, with rank by I
"""
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hgsoc157_post import deconv_table  # noqa: E402

RUN, OUT = Path(sys.argv[1]), Path(sys.argv[2])
(OUT / "spots").mkdir(parents=True, exist_ok=True)
calls = [json.loads(line) for line in open(RUN / "hgsoc157_tool_calls.jsonl")]
tc = [c for c in calls if c.get("record") == "tool_call"]
by = lambda step: {c["sample"]: c["result"] for c in tc if c["step"] == step}  # noqa: E731
loaded, prep, cnv = by("step1_load"), by("step1_preprocess"), by("step2_cnv")
samples = [s for s in loaded if s]

qc_rows = []
for pid in samples:
    a = ad.read_h5ad(RUN / "final_h5ad" / f"{pid}.h5ad")
    obs = a.obs
    pr = prep[pid]
    pr = pr.get("result", pr)
    cr = cnv[pid]
    cr = cr.get("result", cr)
    props = deconv_table(a)
    xy = np.asarray(a.obsm["spatial"], dtype=float)
    spots = pd.DataFrame({"spatial_x": xy[:, 0], "spatial_y": xy[:, 1]}, index=pd.Index(a.obs_names, name="barcode"))
    spots = spots.join(props.add_prefix("RCTD_"))
    spots["rctd_tumor_class"] = obs["rctd_tumor_class"].astype(str).values
    spots["cnv_score"] = obs["cnv_score"].astype(float).values
    spots["cnv_clusters"] = obs["cnv_clusters"].astype(str).values
    spots.to_csv(OUT / "spots" / f"{pid}.csv")
    qc_rows.append({
        "patient": pid, "spots_before": loaded[pid]["n_cells"], "genes_before": loaded[pid]["n_genes"],
        "spots_after": a.n_obs, "genes_after": a.n_vars, "n_hvgs": pr.get("n_hvgs"),
        "median_genes": float(np.median(obs["n_genes_by_counts"])) if "n_genes_by_counts" in obs else np.nan,
        "median_counts": float(np.median(obs["total_counts"])) if "total_counts" in obs else np.nan,
        "median_mito_pct": float(np.median(obs["pct_counts_mt"])) if "pct_counts_mt" in obs else np.nan,
        "n_low_tumor": int((spots["rctd_tumor_class"] == "low_tumor").sum()),
        "n_tumor_enriched": int((spots["rctd_tumor_class"] == "tumor_enriched").sum()),
        "n_cnv_clusters": int(obs["cnv_clusters"].nunique()),
        "cnv_tool_statistics": json.dumps(cr.get("statistics", {})),
        "qc_metrics": json.dumps(pr.get("qc_metrics", {})),
    })
    for k in [k for k in a.uns if k.startswith("rank_genes_groups_")]:
        rg = a.uns[k]
        rows = []
        for g in rg["names"].dtype.names:
            for i in range(len(rg["names"][g])):
                rows.append({"cluster": g, "rank": i + 1, "gene": rg["names"][g][i], "score": float(rg["scores"][g][i]),
                             "log2fc": float(rg["logfoldchanges"][g][i]), "pvalue": float(rg["pvals"][g][i]),
                             "pvalue_adj": float(rg["pvals_adj"][g][i])})
        pd.DataFrame(rows).to_csv(OUT / f"cnv_cluster_markers_{pid}.csv", index=False)
    if "moranI" in a.uns:
        m = a.uns["moranI"].copy().sort_values("I", ascending=False)
        m["rank"] = np.arange(1, len(m) + 1)
        m.index.name = "gene"
        m.to_csv(OUT / f"moranI_{pid}.csv")
    print(pid, a.n_obs, a.n_vars, "tumor-enriched", qc_rows[-1]["n_tumor_enriched"], "cnv clusters",
          qc_rows[-1]["n_cnv_clusters"], flush=True)
pd.DataFrame(qc_rows).to_csv(OUT / "qc_statistics.csv", index=False)
print("COLLECT_DONE")
