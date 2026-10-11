"""
Figure 6: Spatial transcriptomics reveals tumor heterogeneity in HGSOC
Nature Communications style, double-column (183 mm)

Panels:
  Row 1 (a): Tumor cell fraction spatial maps (8 patients)
  Row 2 (b): CNV subclone spatial maps (8 patients)
  Row 3 (c,d): Cell type composition + tumor burden / CNV score (side by side)
  Row 4 (e,f): Moran's I ranked plot (P3) + CNV vs background comparison
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib import cm
from scipy.stats import mannwhitneyu

from figure_style import (
    apply_style,
    add_panel_label,
    FULL_WIDTH,
    OI,
    SUBCLONE_COLORS,
    HGSOC_CELLTYPE_COLORS as CELLTYPE_COLORS,
)

apply_style()

FIGURE_DIR = Path(__file__).resolve().parent
REPO_DIR = FIGURE_DIR.parent
# Every panel reads the ChatSpatial 1.5.7 run, in which each analysis step was an MCP tool call
# (hgsoc/results/hgsoc157_tool_calls.jsonl): per-spot RCTD proportions, low-tumor labels, CNV scores and CNV
# clusters, the P3 CNV-cluster marker genes and the P3 Moran's I table, written by
# hgsoc/scripts/hgsoc157_collect.py; statistics written by hgsoc/scripts/hgsoc157_stats.py.
HGSOC_DIR = REPO_DIR / "experiments" / "case_studies" / "hgsoc" / "results"
DATA_DIR = HGSOC_DIR / "tables"
SPOTS_DIR = DATA_DIR / "spots"
STATS_DIR = HGSOC_DIR / "stats"

PATIENTS = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]

# Cell type marker genes highlighted in Moran's I analysis
CELLTYPE_MARKER_GENES = {
    "CD14": "Monocyte/Macrophage",
    "KRT7": "Epithelial/Tumor",
    "CDH5": "Vascular endothelial",
}


def load_patient_data(patient: str) -> dict:
    """Spot table of the ChatSpatial run: coordinates, RCTD proportions, CNV score and CNV cluster."""
    obs = pd.read_csv(SPOTS_DIR / f"{patient}.csv", index_col=0)

    data = {
        "spatial": obs[["spatial_x", "spatial_y"]].to_numpy(dtype=float),
        "tumour_weight": obs["RCTD_Tumour_cells"].to_numpy(dtype=float),
        "cnv_score": obs["cnv_score"].to_numpy(dtype=float),
        "cnv_leiden": obs["cnv_clusters"].astype(int).to_numpy(),
        "n_spots": len(obs),
    }

    ct_cols = [c for c in obs.columns if c.startswith("RCTD_")]
    for col in ct_cols:
        ct_name = col.replace("RCTD_", "")
        data[ct_name] = np.nanmean(obs[col].values)

    return data


def center_spatial(all_data: dict) -> dict:
    for p in PATIENTS:
        sp = all_data[p]["spatial"]
        cx = (sp[:, 0].min() + sp[:, 0].max()) / 2
        cy = (sp[:, 1].min() + sp[:, 1].max()) / 2
        all_data[p]["spatial_centered"] = sp - np.array([cx, cy])
    return all_data


def compute_spot_size(n_spots: int, ax_width_inches: float) -> float:
    spots_per_row = np.sqrt(n_spots)
    ax_width_pts = ax_width_inches * 72
    spot_diameter_pts = max(ax_width_pts / spots_per_row * 0.9, 2.0)
    return max(spot_diameter_pts ** 2, 4.0)


def plot_spatial(
    ax, spatial, values, title, cmap="YlOrRd", vmin=None, vmax=None,
    is_categorical=False, categories=None, cat_colors=None, spot_size=4.0,
    global_lim=None,
):
    x = spatial[:, 0]
    y = spatial[:, 1]

    ax.set_facecolor("white")

    if is_categorical:
        unique_cats = np.unique(categories)
        for cat in unique_cats:
            mask = categories == cat
            color = cat_colors[cat % len(cat_colors)]
            ax.scatter(x[mask], y[mask], c=color, s=spot_size,
                       edgecolors="none", rasterized=True, label=f"C{cat}")
    else:
        ax.scatter(x, y, c=values, cmap=cmap, s=spot_size,
                   vmin=vmin, vmax=vmax, edgecolors="none", rasterized=True)

    if global_lim is not None:
        ax.set_xlim(-global_lim, global_lim)
        ax.set_ylim(global_lim, -global_lim)
    else:
        x_pad = np.ptp(x) * 0.05
        y_pad = np.ptp(y) * 0.05
        ax.set_xlim(x.min() - x_pad, x.max() + x_pad)
        ax.set_ylim(y.max() + y_pad, y.min() - y_pad)

    ax.set_aspect("equal")
    ax.set_title(title, fontsize=7, fontweight="bold", pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def compute_cnv_markers_p3() -> set:
    """Top 20 genes per CNV cluster of P3 (find_markers on cnv_clusters, Wilcoxon, one cluster vs rest)."""
    mk = pd.read_csv(DATA_DIR / "cnv_cluster_markers_P3.csv")
    mk = mk[mk["rank"] <= 20]
    markers = set(mk["gene"].astype(str))
    print(f"    {mk['cluster'].nunique()} clusters, {len(markers)} unique marker genes")
    return markers


def load_svg_data() -> pd.DataFrame:
    """Moran's I of every gene of P3 (analyze_spatial_statistics, moran), ranked by I."""
    return pd.read_csv(DATA_DIR / "moranI_P3.csv", index_col=0).dropna(subset=["I"])


def plot_morans_ranked(ax, svg_df, cnv_markers):
    svg_sorted = svg_df.sort_values("rank")
    ranks = svg_sorted["rank"].values
    morans_i = svg_sorted["I"].values
    genes = svg_sorted.index.values

    is_ct_marker = np.array([g in CELLTYPE_MARKER_GENES for g in genes])
    is_cnv_marker = np.array([g in cnv_markers for g in genes])
    cnv_in_svg = is_cnv_marker & ~is_ct_marker
    is_background = ~is_ct_marker & ~is_cnv_marker

    ax.scatter(ranks[is_background], morans_i[is_background],
               c="#DDDDDD", s=3, edgecolors="none", rasterized=True,
               zorder=1, label=f"Background (n={is_background.sum():,})")

    n_cnv_in_svg = cnv_in_svg.sum()
    ax.scatter(ranks[cnv_in_svg], morans_i[cnv_in_svg],
               c=OI["blue"], s=10, edgecolors="none", rasterized=True,
               zorder=2, alpha=0.7, label=f"CNV markers (n={n_cnv_in_svg})")

    for gene in CELLTYPE_MARKER_GENES:
        if gene in svg_sorted.index:
            row = svg_sorted.loc[gene]
            ax.scatter(row["rank"], row["I"], c=OI["vermillion"], s=24,
                       edgecolors="white", linewidths=0.3, zorder=3)
            ax.annotate(f"  {gene}", xy=(row["rank"], row["I"]),
                        fontsize=5.5, fontstyle="italic",
                        color=OI["vermillion"], va="center", ha="left", zorder=4)

    ax.scatter([], [], c=OI["vermillion"], s=14, edgecolors="white",
               linewidths=0.3, label="Cell type markers")

    top2_rank = int(len(svg_sorted) * 0.02)
    ax.axvline(x=top2_rank, color="#AAAAAA", linewidth=0.5,
               linestyle="--", zorder=0)
    ax.text(top2_rank + 20, morans_i.max() * 0.80,
            "Top 2%", fontsize=5.5, color="#888888", va="top")

    ax.set_xlabel(f"Gene rank (P3, n={len(svg_sorted):,})", fontsize=7)
    ax.set_ylabel("Moran's I", fontsize=7)
    ax.set_xlim(0, len(svg_sorted) + 50)

    ax.legend(loc="upper right", frameon=False, fontsize=5.5,
              handletextpad=0.3, labelspacing=0.35, markerscale=0.8)

    add_panel_label(ax, "e")


def plot_morans_comparison(ax, svg_df, cnv_markers):
    genes_in_svg = set(svg_df.index)
    cnv_in_svg = cnv_markers & genes_in_svg
    ct_genes = set(CELLTYPE_MARKER_GENES.keys()) & genes_in_svg

    cnv_morans = svg_df.loc[svg_df.index.isin(cnv_in_svg), "I"].values
    bg_genes = genes_in_svg - cnv_in_svg - ct_genes
    bg_morans = svg_df.loc[svg_df.index.isin(bg_genes), "I"].values

    stat, pval = mannwhitneyu(cnv_morans, bg_morans, alternative="two-sided")
    print(f"    Mann-Whitney U: stat={stat:.0f}, p={pval:.4f}")
    print(f"    CNV markers: n={len(cnv_morans)}, median={np.median(cnv_morans):.4f}")
    print(f"    Background: n={len(bg_morans)}, median={np.median(bg_morans):.4f}")

    bp = ax.boxplot(
        [cnv_morans, bg_morans], positions=[0, 1], widths=0.5,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=0.8),
        whiskerprops=dict(linewidth=0.5),
        capprops=dict(linewidth=0.5),
        boxprops=dict(linewidth=0.5),
    )
    bp["boxes"][0].set_facecolor(OI["blue"])
    bp["boxes"][0].set_alpha(0.4)
    bp["boxes"][1].set_facecolor("#CCCCCC")
    bp["boxes"][1].set_alpha(0.4)

    jitter_cnv = np.random.default_rng(42).normal(0, 0.06, len(cnv_morans))
    ax.scatter(jitter_cnv, cnv_morans, c=OI["blue"], s=6, edgecolors="none",
               alpha=0.6, rasterized=True, zorder=2)

    rng = np.random.default_rng(42)
    n_bg_show = min(200, len(bg_morans))
    bg_sample = rng.choice(bg_morans, n_bg_show, replace=False)
    jitter_bg = rng.normal(1, 0.06, n_bg_show)
    ax.scatter(jitter_bg, bg_sample, c="#999999", s=2, edgecolors="none",
               alpha=0.4, rasterized=True, zorder=2)

    for gene in CELLTYPE_MARKER_GENES:
        if gene in svg_df.index:
            mi = svg_df.loc[gene, "I"]
            ax.scatter(1, mi, c=OI["vermillion"], s=18, edgecolors="white",
                       linewidths=0.3, zorder=3)
            ax.annotate(f" {gene}", xy=(1, mi), fontsize=5.5, fontstyle="italic",
                        color=OI["vermillion"], va="center", ha="left", zorder=4)

    q75_bg = np.percentile(bg_morans, 97.5)
    q75_cnv = np.percentile(cnv_morans, 97.5) if len(cnv_morans) > 0 else 0
    y_bracket = max(q75_bg, q75_cnv) * 1.1
    ax.plot([0, 0, 1, 1],
            [y_bracket * 0.92, y_bracket, y_bracket, y_bracket * 0.92],
            color="black", linewidth=0.5)
    p_text = f"p = {pval:.2f}" if pval > 0.01 else f"p = {pval:.2e}"

    delta_path = STATS_DIR / "cliffs_delta_cnv_vs_background.csv"
    delta_label = ""
    if delta_path.exists():
        delta_df = pd.read_csv(delta_path)
        d = delta_df["cliffs_delta"].values[0]
        ci_lo = delta_df["ci_lower"].values[0]
        ci_hi = delta_df["ci_upper"].values[0]
        delta_label = f"\n$\\delta$ = {d:.2f} [{ci_lo:.2f}, {ci_hi:.2f}]"

    ax.text(0.5, y_bracket * 1.03, f"M-W {p_text}{delta_label}",
            fontsize=5.5, ha="center", va="bottom", linespacing=1.1)
    ax.set_ylim(None, y_bracket * 1.20)

    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"CNV markers\n(n={len(cnv_morans)})",
                        f"Background\n(n={len(bg_morans)})"], fontsize=6)
    ax.set_ylabel("Moran's I", fontsize=7)
    ax.set_xlim(-0.6, 1.8)

    add_panel_label(ax, "f")


def main():
    print("Loading patient data...")
    all_data = {}
    for p in PATIENTS:
        print(f"  {p}...")
        all_data[p] = load_patient_data(p)
    print("Data loaded.\n")

    print("Loading SVG data and computing CNV markers...")
    svg_df = load_svg_data()
    cnv_markers = compute_cnv_markers_p3()
    print(f"  SVG genes: {len(svg_df)}, CNV markers in SVG: "
          f"{len(cnv_markers & set(svg_df.index))}\n")

    all_data = center_spatial(all_data)

    global_lim = 0
    for p in PATIENTS:
        sp = all_data[p]["spatial_centered"]
        lim = np.abs(sp).max() * 1.08
        global_lim = max(global_lim, lim)

    # ── Content-driven sizing ─────────────────────────────────────
    fig_w = FULL_WIDTH  # 7.2 in = 183 mm
    n_patients = len(PATIENTS)

    usable_w = fig_w * (0.97 - 0.05)
    panel_w_inches = usable_w / (n_patients + 0.12)
    spatial_row_h = panel_w_inches + 0.25

    summary_row_h = 1.40
    morans_row_h = 1.45

    margin_h = 0.40
    gap_h = 0.50
    fig_h = 2 * spatial_row_h + summary_row_h + morans_row_h + margin_h + gap_h
    fig_h = min(fig_h, 7.2)

    print(f"Figure size: {fig_w:.2f} x {fig_h:.2f} in "
          f"({fig_w * 25.4:.0f} x {fig_h * 25.4:.0f} mm)")
    print(f"  Spatial panel width: {panel_w_inches:.2f} in")

    fig = plt.figure(figsize=(fig_w, fig_h))

    gs_main = gridspec.GridSpec(
        4, 1, figure=fig,
        height_ratios=[spatial_row_h, spatial_row_h, summary_row_h, morans_row_h],
        hspace=0.25,
        left=0.05, right=0.97, top=0.96, bottom=0.07,
    )

    # ── Row 0: Tumor fraction spatial maps (a) ─────────────────────
    gs_row0 = gs_main[0].subgridspec(
        1, n_patients + 1, wspace=0.06,
        width_ratios=[1] * n_patients + [0.12],
    )

    axes_tumour = []
    for i, p in enumerate(PATIENTS):
        ax = fig.add_subplot(gs_row0[0, i])
        axes_tumour.append(ax)
        ss = compute_spot_size(all_data[p]["n_spots"], panel_w_inches)
        plot_spatial(ax, all_data[p]["spatial_centered"],
                     all_data[p]["tumour_weight"], p,
                     cmap="YlOrRd", vmin=0, vmax=1,
                     spot_size=ss, global_lim=global_lim)

    cbar_ax = fig.add_subplot(gs_row0[0, n_patients])
    norm = Normalize(vmin=0, vmax=1)
    sm = cm.ScalarMappable(cmap="YlOrRd", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Tumor\nfraction", fontsize=6, labelpad=2)
    cbar.ax.tick_params(labelsize=5, width=0.4, length=2)
    cbar.outline.set_linewidth(0.4)

    add_panel_label(axes_tumour[0], "a")

    # ── Row 1: CNV subclone spatial maps (b) ───────────────────────
    gs_row1 = gs_main[1].subgridspec(
        1, n_patients + 1, wspace=0.06,
        width_ratios=[1] * n_patients + [0.12],
    )

    axes_cnv = []
    max_n_clusters = max(
        len(np.unique(all_data[p]["cnv_leiden"])) for p in PATIENTS
    )

    for i, p in enumerate(PATIENTS):
        ax = fig.add_subplot(gs_row1[0, i])
        axes_cnv.append(ax)
        ss = compute_spot_size(all_data[p]["n_spots"], panel_w_inches)
        plot_spatial(ax, all_data[p]["spatial_centered"], None, p,
                     is_categorical=True,
                     categories=all_data[p]["cnv_leiden"],
                     cat_colors=SUBCLONE_COLORS,
                     spot_size=ss, global_lim=global_lim)

    legend_ax = fig.add_subplot(gs_row1[0, n_patients])
    legend_ax.axis("off")
    for i in range(max_n_clusters):
        legend_ax.scatter([], [], c=SUBCLONE_COLORS[i % len(SUBCLONE_COLORS)],
                          s=10, label=f"C{i}", rasterized=True)
    legend_ax.legend(loc="center left", frameon=False, fontsize=5.5,
                     handletextpad=0.2, labelspacing=0.35,
                     title="Subclone", title_fontsize=6, markerscale=0.8)

    add_panel_label(axes_cnv[0], "b")

    # ── Row 2: Summary statistics (c, d_left, d_right) ─────────────
    gs_row2 = gs_main[2].subgridspec(
        1, 4, wspace=0.42,
        width_ratios=[1.0, 0.28, 0.50, 0.50],
    )

    # Panel c: Cell type composition stacked bar
    ax_comp = fig.add_subplot(gs_row2[0, 0])

    ct_names = list(CELLTYPE_COLORS.keys())
    ct_fractions = np.zeros((len(PATIENTS), len(ct_names)))
    for i, p in enumerate(PATIENTS):
        for j, ct in enumerate(ct_names):
            ct_fractions[i, j] = all_data[p].get(ct, 0.0)

    row_sums = ct_fractions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    ct_fractions_norm = ct_fractions / row_sums

    x_pos = np.arange(len(PATIENTS))
    bottoms = np.zeros(len(PATIENTS))

    ct_labels_display = {
        "Tumour_cells": "Tumor", "Macrophages": "Macrophage",
        "Fibroblasts": "Fibroblast", "T_cells": "T cell",
        "B_plasma_cells": "B/Plasma", "Endothelial_cells": "Endothelial",
        "Mesothelial_cells": "Mesothelial", "Myofibroblasts": "Myofibroblast",
    }

    for j, ct in enumerate(ct_names):
        color = CELLTYPE_COLORS[ct]
        ax_comp.bar(x_pos, ct_fractions_norm[:, j], bottom=bottoms,
                    width=0.7, color=color,
                    label=ct_labels_display.get(ct, ct),
                    edgecolor="white", linewidth=0.3)
        bottoms += ct_fractions_norm[:, j]

    ax_comp.set_xticks(x_pos)
    ax_comp.set_xticklabels(PATIENTS, fontsize=6, rotation=0)
    ax_comp.set_ylabel("Cell type fraction", fontsize=7)
    ax_comp.set_ylim(0, 1.02)
    ax_comp.set_xlim(-0.5, len(PATIENTS) - 0.5)

    add_panel_label(ax_comp, "c")

    # Cell type legend
    ax_ct_legend = fig.add_subplot(gs_row2[0, 1])
    ax_ct_legend.axis("off")
    handles, labels = ax_comp.get_legend_handles_labels()
    ax_ct_legend.legend(handles, labels, loc="center left", frameon=False,
                        fontsize=5.5, handletextpad=0.3, labelspacing=0.4,
                        handlelength=1.0)

    # Panel d-left: Tumor spots (%)
    ax_d_left = fig.add_subplot(gs_row2[0, 2])

    cnv_df = pd.read_csv(STATS_DIR / "hgsoc_patient_summary.csv")
    cnv_df = cnv_df.set_index("patient").loc[PATIENTS]

    tumour_pct = cnv_df["n_tumor_enriched"].values / cnv_df["n_spots"].values * 100

    ax_d_left.bar(x_pos, tumour_pct, width=0.6, color=OI["vermillion"],
                  edgecolor="white", linewidth=0.3)
    ax_d_left.set_ylabel("Tumor spots (%)", fontsize=7, color=OI["vermillion"])
    ax_d_left.tick_params(axis="y", colors=OI["vermillion"])
    ax_d_left.set_xticks(x_pos)
    ax_d_left.set_xticklabels(PATIENTS, fontsize=6)
    ax_d_left.set_ylim(0, 110)
    ax_d_left.set_xlim(-0.5, len(PATIENTS) - 0.5)

    add_panel_label(ax_d_left, "d")

    # Panel d-right: Mean CNV score with SD error bars
    ax_d_right = fig.add_subplot(gs_row2[0, 3])

    mean_cnv = cnv_df["mean_cnv_score"].values
    ax_d_right.bar(x_pos, mean_cnv, width=0.6, color=OI["blue"],
                   edgecolor="white", linewidth=0.3)

    cnv_stats_path = STATS_DIR / "cnv_score_per_patient_stats.csv"
    if cnv_stats_path.exists():
        cnv_stats = pd.read_csv(cnv_stats_path)
        cnv_stats = cnv_stats.set_index("patient").loc[PATIENTS]
        sd_cnv = cnv_stats["sd_cnv_score"].values
        ax_d_right.errorbar(x_pos, mean_cnv, yerr=sd_cnv, fmt="none",
                            ecolor=OI["blue"], elinewidth=0.5,
                            capsize=2, capthick=0.5, zorder=5)

    ax_d_right.set_ylabel("Mean CNV score", fontsize=7, color=OI["blue"])
    ax_d_right.tick_params(axis="y", colors=OI["blue"])
    ax_d_right.set_xticks(x_pos)
    ax_d_right.set_xticklabels(PATIENTS, fontsize=6)
    ax_d_right.set_xlim(-0.5, len(PATIENTS) - 0.5)

    # ── Row 3: Moran's I analysis (e, f) ──────────────────────────
    gs_row3 = gs_main[3].subgridspec(1, 3, wspace=0.12,
                                      width_ratios=[1.0, 0.06, 0.9])

    ax_e = fig.add_subplot(gs_row3[0, 0])
    plot_morans_ranked(ax_e, svg_df, cnv_markers)

    ax_spacer = fig.add_subplot(gs_row3[0, 1])
    ax_spacer.axis("off")

    ax_f = fig.add_subplot(gs_row3[0, 2])
    plot_morans_comparison(ax_f, svg_df, cnv_markers)

    # ── Save ──────────────────────────────────────────────────────
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ["pdf", "png"]:
        out_path = FIGURE_DIR / f"figure3.{fmt}"
        fig.savefig(str(out_path), dpi=300, facecolor="white")
        print(f"Saved: {out_path}")

    plt.close(fig)
    print("\nDone!")


if __name__ == "__main__":
    main()
