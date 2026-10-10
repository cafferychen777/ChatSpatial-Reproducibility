#!/usr/bin/env python3
"""
Compute the statistics of the case-study figures.

OSCC (Figure 2, Supplementary Table 1): inputs are the tables of the ChatSpatial 1.5.7 run, in which every
analysis step was a ChatSpatial MCP tool call (oscc/results/tables/, written by
oscc/scripts/oscc157_collect.py from the tools' exports):
  proportions/{sid}.csv                FlashDeconv proportions (deconvolve_data) + TC/LE label
  moranI_results_all_samples.csv       Moran's I per section (analyze_spatial_statistics); this table is
                                       committed as source_data/Fig2h_morans.csv and read from there
                                       when it is not present in OSCC_RUN_DIR
  deg_tc_vs_le_top500.csv              TC vs LE marker genes (find_markers)
  cellchat_tc_le_lr_pairs.csv          CellChat interactions (analyze_cell_communication)
Set OSCC_RUN_DIR / OSCC_OUT_DIR to point the OSCC part at other inputs or outputs.

Outputs (OSCC_OUT_DIR, default oscc/results/stats/):
  - per_sample_tc_le_enrichment.csv, consensus_tc_le_patterns.csv   (Fig. 2d,e)
  - effect_sizes_enrichment.csv, consensus_binomial_ci.csv
  - sample_level_tests.csv, effect_size_summary.csv                (Supplementary Table 1)
  - moranI_cross_sample_stats.csv, svg_target_genes_summary.csv    (Fig. 2h, Moran's I text)
  - consensus_le_upregulated_genes.csv, consensus_tc_upregulated_genes.csv
  - cellchat_summary.csv                                           (Fig. 2f text)
HGSOC (inputs in hgsoc/results/tables/, outputs in hgsoc/results/stats/; the same statistics are also written
by hgsoc/scripts/hgsoc157_stats.py, which the figure reads):
  - cliffs_delta_cnv_vs_background.csv                              (Fig. 3e,f)
  - cnv_score_per_patient_stats.csv                                 (Fig. 3d)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ── Paths ────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parents[1]

OSCC_RUN_DIR = Path(os.environ.get(
    "OSCC_RUN_DIR", SCRIPT_DIR / "oscc" / "results" / "tables"))
OSCC_OUT_DIR = Path(os.environ.get("OSCC_OUT_DIR", SCRIPT_DIR / "oscc" / "results" / "stats"))
OSCC_MORAN_FALLBACK = REPO_DIR / "source_data" / "Fig2h_morans.csv"

# HGSOC inputs are the outputs of the recorded ChatSpatial tool calls
# (hgsoc/results/tables/).
HGSOC_RUN_DIR = Path(os.environ.get(
    "HGSOC_RUN_DIR", SCRIPT_DIR / "hgsoc" / "results" / "tables"))
HGSOC_OUT_DIR = Path(os.environ.get("HGSOC_OUT_DIR", SCRIPT_DIR / "hgsoc" / "results" / "stats"))

SEED = 42
N_BOOT = 10_000
PSEUDOCOUNT = 1e-6
SAMPLE_IDS = [f"s{i}" for i in range(1, 13)]
MORAN_TARGETS = {"LE": ["FN1", "LAMC2", "ITGA5", "COL1A1"], "TC": ["SPRR1B", "CLDN4", "SPRR2E", "KRT6C"]}
MORAN_SAMPLE = "s1"


# ── Utilities ────────────────────────────────────────────────────

def rank_biserial_r(x: np.ndarray, y: np.ndarray) -> float:
    """Rank-biserial correlation from Mann-Whitney U."""
    u_stat, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    n1, n2 = len(x), len(y)
    return 1 - 2 * u_stat / (n1 * n2)


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    stat_func,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> tuple[float, float]:
    """Bootstrap 95% CI (percentile method) for a two-sample statistic."""
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        bx = rng.choice(x, len(x), replace=True)
        by = rng.choice(y, len(y), replace=True)
        boots[i] = stat_func(bx, by)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial (Clopper-Pearson) confidence interval."""
    if k == 0:
        lo = 0.0
    else:
        lo = stats.beta.ppf(alpha / 2, k, n - k + 1)
    if k == n:
        hi = 1.0
    else:
        hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def bootstrap_median_ci(
    x: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> tuple[float, float]:
    """Bootstrap 95% CI for the median (one-sample, percentile method)."""
    rng = np.random.default_rng(seed)
    medians = np.empty(n_boot)
    for i in range(n_boot):
        medians[i] = np.median(rng.choice(x, len(x), replace=True))
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta effect size for two independent groups."""
    n1, n2 = len(x), len(y)
    # Vectorized: count concordant and discordant pairs
    more = np.sum(x[:, None] > y[None, :])
    less = np.sum(x[:, None] < y[None, :])
    return float((more - less) / (n1 * n2))




# ═══════════════════════════════════════════════════════════════
# 0. OSCC: per-sample TC vs LE enrichment of deconvolved proportions (Fig. 2d,e)
# ═══════════════════════════════════════════════════════════════

def load_oscc_proportions() -> dict[str, pd.DataFrame]:
    props = {}
    for sid in SAMPLE_IDS:
        path = OSCC_RUN_DIR / "proportions" / f"{sid}.csv"
        if path.exists():
            props[sid] = pd.read_csv(path, index_col=0)
    return props


def compute_oscc_enrichment(props: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Within each sample: two-sided Mann-Whitney U of LE vs TC spot proportions, log2FC of the means
    (pseudocount 1e-6), BH-FDR over all sample x cell-type tests; enriched if q < 0.05 and |log2FC| > 0.25.
    Same procedure as the original case-study pipeline. A sample without TC or LE spots gets NaN."""
    print("Computing OSCC per-sample TC/LE enrichment...")
    rows = []
    for sid, pdf in props.items():
        f = pdf[pdf["tc_le_region"].isin(["TC", "LE"])]
        if len(f) < 10:
            continue
        prop_df = f.drop(columns="tc_le_region")
        regions = f["tc_le_region"].values
        tc_mean = prop_df[regions == "TC"].mean()
        le_mean = prop_df[regions == "LE"].mean()
        for ct in tc_mean.index:
            tc_vals = prop_df.loc[regions == "TC", ct].values
            le_vals = prop_df.loc[regions == "LE", ct].values
            if len(le_vals) == 0 or len(tc_vals) == 0 or (le_vals.sum() == 0 and tc_vals.sum() == 0):
                pval = np.nan
            else:
                pval = stats.mannwhitneyu(le_vals, tc_vals, alternative="two-sided").pvalue
            log2fc = np.log2((le_mean[ct] + PSEUDOCOUNT) / (tc_mean[ct] + PSEUDOCOUNT))
            rows.append({"sample": sid, "cell_type": ct, "tc_mean": tc_mean[ct], "le_mean": le_mean[ct],
                         "log2fc": log2fc, "pvalue": pval, "enriched_in": "Neither"})
    df = pd.DataFrame(rows)
    valid = df["pvalue"].notna()
    df["pvalue_fdr"] = np.nan
    df.loc[valid, "pvalue_fdr"] = multipletests(df.loc[valid, "pvalue"].values, method="fdr_bh")[1]
    sig = (df["pvalue_fdr"] < 0.05) & (df["log2fc"].abs() > 0.25)
    df.loc[sig & (df["log2fc"] > 0), "enriched_in"] = "LE"
    df.loc[sig & (df["log2fc"] < 0), "enriched_in"] = "TC"
    df.to_csv(OSCC_OUT_DIR / "per_sample_tc_le_enrichment.csv", index=False)

    consensus = {}
    for ct in df["cell_type"].unique():
        r = df[df["cell_type"] == ct]
        n = r["sample"].nunique()
        consensus[ct] = {
            "le_enriched_count": int((r["enriched_in"] == "LE").sum()),
            "tc_enriched_count": int((r["enriched_in"] == "TC").sum()),
            "total_samples": n,
            "le_percentage": 100 * (r["enriched_in"] == "LE").sum() / n,
            "tc_percentage": 100 * (r["enriched_in"] == "TC").sum() / n,
        }
    cdf = pd.DataFrame(consensus).T.sort_values("le_percentage", ascending=False)
    cdf.to_csv(OSCC_OUT_DIR / "consensus_tc_le_patterns.csv")
    print(cdf.to_string())
    return df


# ═══════════════════════════════════════════════════════════════
# 1. OSCC: Rank-biserial r + CIs for TC vs LE enrichment
# ═══════════════════════════════════════════════════════════════

def compute_oscc_effect_sizes(props: dict[str, pd.DataFrame], enrich_df: pd.DataFrame):
    print("Computing OSCC enrichment effect sizes...")
    rows = []
    for sample_id in sorted(enrich_df["sample"].unique()):
        obs = props[sample_id]
        tc_mask = obs["tc_le_region"] == "TC"
        le_mask = obs["tc_le_region"] == "LE"
        n_tc, n_le = int(tc_mask.sum()), int(le_mask.sum())
        if n_tc < 5 or n_le < 5:
            continue
        for col in [c for c in obs.columns if c != "tc_le_region"]:
            tc_vals = obs.loc[tc_mask, col].values.astype(float)
            le_vals = obs.loc[le_mask, col].values.astype(float)
            if tc_vals.sum() == 0 and le_vals.sum() == 0:
                rows.append({"sample": sample_id, "cell_type": col, "n_TC": n_tc, "n_LE": n_le,
                             "rank_biserial_r": np.nan, "ci_lower": np.nan, "ci_upper": np.nan})
                continue
            r = rank_biserial_r(tc_vals, le_vals)
            ci_lo, ci_hi = bootstrap_ci(tc_vals, le_vals, rank_biserial_r)
            rows.append({"sample": sample_id, "cell_type": col, "n_TC": n_tc, "n_LE": n_le,
                         "rank_biserial_r": round(r, 4), "ci_lower": round(ci_lo, 4), "ci_upper": round(ci_hi, 4)})
        print(f"  {sample_id}: {n_tc} TC, {n_le} LE spots")
    merged = pd.DataFrame(rows).merge(enrich_df[["sample", "cell_type", "log2fc", "pvalue_fdr"]],
                                      on=["sample", "cell_type"], how="left")
    merged.to_csv(OSCC_OUT_DIR / "effect_sizes_enrichment.csv", index=False)
    return merged


# ═══════════════════════════════════════════════════════════════
# 2. OSCC: Clopper-Pearson CIs for consensus proportions
# ═══════════════════════════════════════════════════════════════

def compute_oscc_consensus_ci():
    print("Computing OSCC consensus binomial CIs...")
    consensus = pd.read_csv(OSCC_OUT_DIR / "consensus_tc_le_patterns.csv", index_col=0)
    rows = []
    for cell_type in consensus.index:
        n = int(consensus.loc[cell_type, "total_samples"])
        for direction in ["le", "tc"]:
            k = int(consensus.loc[cell_type, f"{direction}_enriched_count"])
            lo, hi = clopper_pearson_ci(k, n)
            rows.append({"cell_type": cell_type, "direction": direction.upper(), "k": k, "n": n,
                         "proportion": round(k / n, 4), "ci_lower": round(lo, 4), "ci_upper": round(hi, 4)})
    ci_df = pd.DataFrame(rows)
    ci_df.to_csv(OSCC_OUT_DIR / "consensus_binomial_ci.csv", index=False)
    return ci_df


# ═══════════════════════════════════════════════════════════════
# 3. OSCC: Sample-level Wilcoxon signed-rank tests
# ═══════════════════════════════════════════════════════════════

def compute_oscc_sample_level_tests(enrich_df: pd.DataFrame):
    """Wilcoxon signed-rank on the per-patient log2FC (samples without TC or LE spots excluded: their log2FC
    is undefined), sign test, median log2FC with bootstrap 95% CI, BH-FDR across cell types."""
    print("Computing OSCC sample-level tests (Wilcoxon signed-rank)...")
    rows = []
    for ct in sorted(enrich_df["cell_type"].unique()):
        log2fcs = enrich_df.loc[enrich_df["cell_type"] == ct, "log2fc"].dropna().values.astype(float)
        n_samples = len(log2fcs)
        nonzero = log2fcs[log2fcs != 0.0]
        n_nonzero = len(nonzero)
        w_stat, w_p = stats.wilcoxon(nonzero, alternative="two-sided") if n_nonzero >= 3 else (np.nan, np.nan)
        n_pos, n_neg = int(np.sum(nonzero > 0)), int(np.sum(nonzero < 0))
        sign_p = stats.binomtest(n_pos, n_nonzero, 0.5).pvalue if n_nonzero >= 1 else np.nan
        if n_samples >= 2:
            med = float(np.median(log2fcs))
            ci_lo, ci_hi = bootstrap_median_ci(log2fcs)
        elif n_samples == 1:
            med, ci_lo, ci_hi = float(log2fcs[0]), np.nan, np.nan
        else:
            med, ci_lo, ci_hi = np.nan, np.nan, np.nan
        direction = "--" if np.isnan(med) else ("LE" if med > 0 else ("TC" if med < 0 else "--"))
        rows.append({
            "cell_type": ct, "n_samples": n_samples, "n_nonzero": n_nonzero,
            "median_log2fc": round(med, 4) if not np.isnan(med) else np.nan,
            "median_ci_lower": round(ci_lo, 4) if not np.isnan(ci_lo) else np.nan,
            "median_ci_upper": round(ci_hi, 4) if not np.isnan(ci_hi) else np.nan,
            "wilcoxon_stat": round(w_stat, 1) if not np.isnan(w_stat) else np.nan,
            "wilcoxon_p": w_p, "sign_n_pos": n_pos, "sign_n_neg": n_neg, "sign_p": sign_p, "direction": direction,
        })
    result_df = pd.DataFrame(rows)
    valid = result_df["wilcoxon_p"].notna()
    result_df["wilcoxon_p_fdr"] = np.nan
    if valid.sum() > 0:
        result_df.loc[valid, "wilcoxon_p_fdr"] = multipletests(result_df.loc[valid, "wilcoxon_p"].values,
                                                               method="fdr_bh")[1]
    result_df.to_csv(OSCC_OUT_DIR / "sample_level_tests.csv", index=False)
    print(result_df.to_string(index=False))
    return result_df


# ═══════════════════════════════════════════════════════════════
# 4. OSCC: Effect size summary (all samples, not just significant)
# ═══════════════════════════════════════════════════════════════

def compute_oscc_effect_size_summary(effect_df: pd.DataFrame):
    print("Computing OSCC effect size summary...")
    consensus = pd.read_csv(OSCC_OUT_DIR / "consensus_tc_le_patterns.csv", index_col=0)
    rows = []
    for ct in sorted(effect_df["cell_type"].unique()):
        ct_data = effect_df[effect_df["cell_type"] == ct]
        valid = ct_data["rank_biserial_r"].dropna()
        sig = ct_data[ct_data["pvalue_fdr"] < 0.05]["rank_biserial_r"].dropna()
        le_pct, tc_pct = consensus.loc[ct, "le_percentage"], consensus.loc[ct, "tc_percentage"]
        if le_pct > tc_pct:
            direction, consensus_pct = "LE", le_pct
        elif tc_pct > le_pct:
            direction, consensus_pct = "TC", tc_pct
        else:
            direction, consensus_pct = "--", 0
        rows.append({
            "cell_type": ct, "direction": direction, "n_all": len(valid),
            "mean_abs_r_all": round(float(valid.abs().mean()), 4) if len(valid) else np.nan,
            "n_significant": len(sig),
            "mean_abs_r_sig": round(float(sig.abs().mean()), 4) if len(sig) else np.nan,
            "consensus_pct": round(consensus_pct, 1),
        })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OSCC_OUT_DIR / "effect_size_summary.csv", index=False)
    print(summary_df.to_string(index=False))
    return summary_df


# ═══════════════════════════════════════════════════════════════
# 5. OSCC: Moran's I — Sample 1 ranks and cross-sample statistics (Fig. 2h)
# ═══════════════════════════════════════════════════════════════

def compute_oscc_moranI_stats(moran_path: Path | None = None, sample_path: Path | None = None):
    """Cross-sample mean Moran's I (t-based 95% CI) over the sections in which the gene was tested, and the
    rank of each target gene among all genes tested in Sample 1 (by I, descending)."""
    print("Computing OSCC Moran's I statistics...")
    if moran_path is None:
        moran_path = OSCC_RUN_DIR / "moranI_results_all_samples.csv"
        if not moran_path.exists():
            moran_path = OSCC_MORAN_FALLBACK
    moranI = pd.read_csv(moran_path)
    rows = []
    for gene in ["FN1", "COL1A1", "SPRR1B", "KRT6C", "SPRR2E", "LAMC2", "ITGA5", "CLDN4"]:
        vals = moranI.loc[moranI["gene"] == gene, "I"].values
        n = len(vals)
        if n < 2:
            rows.append({"gene": gene, "n_samples": n, "mean_I": round(vals[0], 4) if n == 1 else np.nan,
                         "sd_I": np.nan, "ci_lower": np.nan, "ci_upper": np.nan})
            continue
        mean_I, sd_I = vals.mean(), vals.std(ddof=1)
        half = stats.t.ppf(0.975, df=n - 1) * sd_I / np.sqrt(n)
        rows.append({"gene": gene, "n_samples": n, "mean_I": round(mean_I, 4), "sd_I": round(sd_I, 4),
                     "ci_lower": round(mean_I - half, 4), "ci_upper": round(mean_I + half, 4)})
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(OSCC_OUT_DIR / "moranI_cross_sample_stats.csv", index=False)
    print(stats_df.to_string(index=False))

    if sample_path is not None:  # single-section table (validation of the old Sample 1 values)
        s1 = pd.read_csv(sample_path)
    else:
        s1 = moranI[moranI["sample"] == MORAN_SAMPLE]
    s1 = s1.dropna(subset=["I"]).sort_values("I", ascending=False).reset_index(drop=True)
    total = len(s1)
    qcol = next(c for c in ("pval_norm_fdr", "pval_norm_fdr_bh") if c in s1.columns)
    out = []
    for cat, genes in MORAN_TARGETS.items():
        for gene in genes:
            hit = s1.index[s1["gene"] == gene]
            if len(hit) == 0:
                out.append({"gene": gene, "category": cat, "moranI": np.nan, "pval_norm": np.nan,
                            "qval_fdr_bh": np.nan, "rank": np.nan, "total_genes": total, "percentile_top": np.nan})
                continue
            i = hit[0]
            out.append({"gene": gene, "category": cat, "moranI": s1.loc[i, "I"], "pval_norm": s1.loc[i, "pval_norm"],
                        "qval_fdr_bh": s1.loc[i, qcol], "rank": i + 1, "total_genes": total,
                        "percentile_top": round(100 * (i + 1) / total, 2)})
    svg = pd.DataFrame(out)
    svg.to_csv(OSCC_OUT_DIR / "svg_target_genes_summary.csv", index=False)
    print(svg.to_string(index=False))
    return stats_df, svg


# ═══════════════════════════════════════════════════════════════
# 6. OSCC: consensus TC/LE marker genes and CellChat summary
# ═══════════════════════════════════════════════════════════════

def compute_oscc_consensus_degs(deg_path: Path | None = None):
    """Genes with BH q < 0.05 and log2FC > 0.5 in at least 9 of the sections (case-study rule)."""
    deg = pd.read_csv(deg_path or OSCC_RUN_DIR / "deg_tc_vs_le_top500.csv")
    out = {}
    for g in ("LE", "TC"):
        sig = deg[(deg["group"] == g) & (deg["pvalue_adj"] < 0.05) & (deg["log2fc"] > 0.5)]
        cnt = sig.groupby("gene")["sample"].nunique()
        cons = cnt[cnt >= 9].sort_values(ascending=False).rename("sample")
        cons.to_csv(OSCC_OUT_DIR / f"consensus_{g.lower()}_upregulated_genes.csv", index_label="gene")
        out[g] = cons
        print(f"  consensus {g}-up genes (>= 9 sections): {len(cons)}")
    return out


def summarize_oscc_cellchat(lr_path: Path | None = None):
    lr = pd.read_csv(lr_path or OSCC_RUN_DIR / "cellchat_tc_le_lr_pairs.csv")
    fib = lr["source"].str.contains("Fibroblast") & lr["target"].str.contains("Cancer")
    rows = []
    for label, m in [("all", np.ones(len(lr), bool)),
                     ("Fibroblast->Cancer", (lr["source"] == "Fibroblast") & (lr["target"] == "Cancer")),
                     ("Fibroblast|-Fibroblast->Cancer", fib),
                     ("fibroblast<->cancer (case-study regex)",
                      fib | (lr["target"].str.contains("Fibroblast") & lr["source"].str.contains("Cancer")))]:
        sub = lr[m]
        rows.append({"subset": label, "n_interactions": len(sub), "n_pathways": sub["pathway_name"].nunique(),
                     "pathways": ";".join(sub["pathway_name"].value_counts().index.astype(str))})
    s = pd.DataFrame(rows)
    s.to_csv(OSCC_OUT_DIR / "cellchat_summary.csv", index=False)
    print(s.to_string(index=False))
    return s


# ═══════════════════════════════════════════════════════════════
# 4. HGSOC: Cliff's delta for CNV markers vs background
# ═══════════════════════════════════════════════════════════════

def compute_hgsoc_cliffs_delta():
    print("Computing HGSOC Cliff's delta (CNV markers vs background)...")

    moran = pd.read_csv(HGSOC_RUN_DIR / "moranI_P3.csv").set_index("gene")
    markers = pd.read_csv(HGSOC_RUN_DIR / "cnv_cluster_markers_P3.csv")
    # Top 20 marker genes of each CNV cluster, as returned by find_markers.
    top = markers[markers["rank"] <= 20]
    cnv_marker_genes = sorted(set(top["gene"].astype(str)))

    cnv_in_svg = [g for g in cnv_marker_genes if g in moran.index]
    background = [g for g in moran.index if g not in set(cnv_in_svg)]
    print(f"  CNV markers with Moran's I: {len(cnv_in_svg)}")
    print(f"  Background genes: {len(background)}")

    cnv_vals = moran.loc[cnv_in_svg, "I"].to_numpy(dtype=float)
    bg_vals = moran.loc[background, "I"].to_numpy(dtype=float)

    delta = cliffs_delta(cnv_vals, bg_vals)
    ci_lo, ci_hi = bootstrap_ci(cnv_vals, bg_vals, cliffs_delta)
    u_stat, p_val = stats.mannwhitneyu(cnv_vals, bg_vals, alternative="two-sided")

    result = {
        "n_cnv_markers": len(cnv_in_svg),
        "n_background": len(background),
        "cnv_mean_I": round(cnv_vals.mean(), 4),
        "bg_mean_I": round(bg_vals.mean(), 4),
        "cliffs_delta": round(delta, 4),
        "ci_lower": round(ci_lo, 4),
        "ci_upper": round(ci_hi, 4),
        "mannwhitney_U": round(u_stat, 1),
        "mannwhitney_p": p_val,
        "cnv_median_I": float(np.median(cnv_vals)),
        "bg_median_I": float(np.median(bg_vals)),
    }
    out_path = HGSOC_OUT_DIR / "cliffs_delta_cnv_vs_background.csv"
    pd.DataFrame([result]).to_csv(out_path, index=False)
    print(f"  Wrote {out_path}")
    print(f"  Cliff's delta = {delta:.4f} [{ci_lo:.4f}, {ci_hi:.4f}], p = {p_val:.4g}")
    return result


# ═══════════════════════════════════════════════════════════════
# 5. HGSOC: Per-patient CNV score mean + SD
# ═══════════════════════════════════════════════════════════════

def compute_hgsoc_cnv_stats():
    print("Computing HGSOC per-patient CNV score statistics...")

    rows = []
    for p in range(1, 9):
        spots_path = HGSOC_RUN_DIR / "spots" / f"P{p}.csv"
        if not spots_path.exists():
            print(f"  WARNING: {spots_path} missing")
            continue
        scores = pd.read_csv(spots_path)["cnv_score"].dropna().to_numpy(dtype=float)
        rows.append({
            "patient": f"P{p}",
            "n_spots": len(scores),
            "mean_cnv_score": round(scores.mean(), 6),
            "sd_cnv_score": round(scores.std(ddof=1), 6),
            "median_cnv_score": round(np.median(scores), 6),
            "iqr_lower": round(np.percentile(scores, 25), 6),
            "iqr_upper": round(np.percentile(scores, 75), 6),
        })
        print(f"  P{p}: {len(scores)} spots, mean={scores.mean():.4f}, SD={scores.std(ddof=1):.4f}")

    cnv_df = pd.DataFrame(rows)
    out_path = HGSOC_OUT_DIR / "cnv_score_per_patient_stats.csv"
    cnv_df.to_csv(out_path, index=False)
    print(f"  Wrote {out_path}")
    return cnv_df


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def run_oscc_stats():
    OSCC_OUT_DIR.mkdir(parents=True, exist_ok=True)
    props = load_oscc_proportions()
    enrich_df = compute_oscc_enrichment(props)
    print()
    merged = compute_oscc_effect_sizes(props, enrich_df)
    print()
    compute_oscc_consensus_ci()
    print()
    compute_oscc_sample_level_tests(enrich_df)
    print()
    compute_oscc_effect_size_summary(merged)
    print()
    compute_oscc_moranI_stats()
    print()
    compute_oscc_consensus_degs()
    print()
    summarize_oscc_cellchat()


if __name__ == "__main__":
    import sys

    run_oscc_stats()
    if "--oscc-only" not in sys.argv[1:]:
        HGSOC_OUT_DIR.mkdir(parents=True, exist_ok=True)
        print()
        compute_hgsoc_cliffs_delta()
        print()
        compute_hgsoc_cnv_stats()
    print("\nDone.")
