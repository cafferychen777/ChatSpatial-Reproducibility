#!/usr/bin/env python
"""Statistics of the HGSOC case study (Figure 6c,d,f and text), computed on the tables of the ChatSpatial runs.

The two-sample statistics are those of ../../compute_case_study_stats.py (cliffs_delta,
bootstrap_ci with seed 42 and 10,000 resamples, two-sided Mann-Whitney U), copied here unchanged.
Usage: hgsoc157_stats.py TABLE_DIR OUT_DIR [FOLLOWUP_PATIENT]
Inputs (TABLE_DIR, written by hgsoc157_collect.py): spots/{pid}.csv, moranI_{pid}.csv, cnv_cluster_markers_{pid}.csv
Outputs (OUT_DIR):
  cnv_score_per_patient_stats.csv        mean, SD (ddof=1), median, IQR of the per-spot CNV score (Fig. 6d)
  hgsoc_patient_summary.csv              spots, tumor-enriched spots (%), CNV clusters, mean RCTD proportions
  cliffs_delta_cnv_vs_background.csv     Moran's I of CNV-cluster marker genes versus all other tested genes
  celltype_marker_moran_ranks.csv        Moran's I rank of CD14, KRT7, CDH5
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEED, N_BOOT = 42, 10_000
CT_MARKERS = ["CD14", "KRT7", "CDH5"]
PATIENTS = [f"P{i}" for i in range(1, 9)]


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta effect size for two independent groups."""
    n1, n2 = len(x), len(y)
    more = np.sum(x[:, None] > y[None, :])
    less = np.sum(x[:, None] < y[None, :])
    return float((more - less) / (n1 * n2))


def bootstrap_ci(x, y, stat_func, n_boot=N_BOOT, seed=SEED):
    """Bootstrap 95% CI (percentile method) for a two-sample statistic."""
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        bx = rng.choice(x, len(x), replace=True)
        by = rng.choice(y, len(y), replace=True)
        boots[i] = stat_func(bx, by)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def cnv_stats(scores_by_patient: dict) -> pd.DataFrame:
    rows = []
    for p, s in scores_by_patient.items():
        s = np.asarray(pd.Series(s).dropna(), dtype=float)
        rows.append({"patient": p, "n_spots": len(s), "mean_cnv_score": round(s.mean(), 6),
                     "sd_cnv_score": round(s.std(ddof=1), 6), "median_cnv_score": round(np.median(s), 6),
                     "iqr_lower": round(np.percentile(s, 25), 6), "iqr_upper": round(np.percentile(s, 75), 6)})
    return pd.DataFrame(rows)


def marker_vs_background(moran_I: pd.Series, markers, exclude=()) -> dict:
    """Moran's I of marker genes versus the other tested genes (background excludes markers and `exclude`)."""
    moran_I = moran_I.dropna()
    genes = set(moran_I.index)
    mk = [g for g in dict.fromkeys(markers) if g in genes]
    bg = [g for g in moran_I.index if g not in set(mk) and g not in set(exclude)]
    x, y = moran_I.loc[mk].to_numpy(float), moran_I.loc[bg].to_numpy(float)
    d = cliffs_delta(x, y)
    lo, hi = bootstrap_ci(x, y, cliffs_delta)
    u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    return {"n_cnv_markers": len(mk), "n_background": len(bg), "cnv_mean_I": round(x.mean(), 4),
            "bg_mean_I": round(y.mean(), 4), "cliffs_delta": round(d, 4), "ci_lower": round(lo, 4),
            "ci_upper": round(hi, 4), "mannwhitney_U": round(u, 1), "mannwhitney_p": p,
            "cnv_median_I": float(np.median(x)), "bg_median_I": float(np.median(y))}


def main(tables: Path, out: Path, followup: str = "P3") -> None:
    out.mkdir(parents=True, exist_ok=True)
    spots = {p: pd.read_csv(tables / "spots" / f"{p}.csv", index_col=0) for p in PATIENTS
             if (tables / "spots" / f"{p}.csv").exists()}
    cs = cnv_stats({p: d["cnv_score"] for p, d in spots.items()})
    cs.to_csv(out / "cnv_score_per_patient_stats.csv", index=False)
    rows = []
    for p, d in spots.items():
        r = {"patient": p, "n_spots": len(d), "n_tumor_enriched": int((d["rctd_tumor_class"] == "tumor_enriched").sum()),
             "n_low_tumor": int((d["rctd_tumor_class"] == "low_tumor").sum()),
             "n_cnv_clusters": int(d["cnv_clusters"].nunique())}
        r["pct_tumor_enriched"] = round(100 * r["n_tumor_enriched"] / r["n_spots"], 2)
        for c in [c for c in d.columns if c.startswith("RCTD_")]:
            r[f"mean_{c}"] = float(d[c].mean())
        rows.append(r)
    summ = pd.DataFrame(rows).merge(cs, on=["patient", "n_spots"])
    summ.to_csv(out / "hgsoc_patient_summary.csv", index=False)
    print(summ[["patient", "n_spots", "n_tumor_enriched", "pct_tumor_enriched", "n_cnv_clusters", "mean_cnv_score",
                "sd_cnv_score"]].to_string(index=False))

    moran = pd.read_csv(tables / f"moranI_{followup}.csv", index_col=0)
    mk = pd.read_csv(tables / f"cnv_cluster_markers_{followup}.csv")
    markers = mk.loc[mk["rank"] <= 20, "gene"].astype(str).tolist()
    res = marker_vs_background(moran["I"], markers)
    res_fig = marker_vs_background(moran["I"], markers, exclude=CT_MARKERS)
    pd.DataFrame([res]).to_csv(out / "cliffs_delta_cnv_vs_background.csv", index=False)
    pd.DataFrame([{**res_fig, "background": "excluding CD14, KRT7, CDH5 (Fig. 6f boxplot)"}]).to_csv(
        out / "cnv_vs_background_fig3f.csv", index=False)
    print("CNV markers vs background:", res)
    print("Figure 6f version:", res_fig)
    ranked = moran["I"].dropna().sort_values(ascending=False)
    n = len(ranked)
    ct = [{"gene": g, "I": float(ranked[g]), "rank": int(ranked.index.get_loc(g)) + 1, "n_genes": n,
           "top_percent": round(100 * (ranked.index.get_loc(g) + 1) / n, 2)} for g in CT_MARKERS if g in ranked.index]
    pd.DataFrame(ct).to_csv(out / "celltype_marker_moran_ranks.csv", index=False)
    print(pd.DataFrame(ct).to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "P3")
