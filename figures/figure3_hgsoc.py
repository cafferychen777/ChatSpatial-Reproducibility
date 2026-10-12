"""
Fig. 6 of the manuscript (file figure3): ChatSpatial reanalysis of the HGSOC subclone study.
Direction A ("Ink & Cobalt") style, authored at 180 mm width.

Panels:
  a  Tumor-cell proportion maps (RCTD) of the 8 patients, one physical scale (um) and a scale bar
  b  CNV clusters of each sample (infercnvpy Leiden clusters, numbered by size within each sample)
  c  Cell type composition per patient (mean RCTD proportion)
  d  Tumor burden (top) and CNV score, mean +/- s.d. (bottom), per patient
  e  Moran's I of all 19,180 genes of P3 against gene rank (log scale)
  f  Moran's I of the 95 CNV-cluster marker genes versus the other 19,085 genes of P3
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu

from figure_style import (
    apply_style,
    CELLTYPE_COLORS,
    CMAP_SEQUENTIAL,
    COBALT,
    COBALT_LIGHT,
    DOUBLE_COL,
    HAIRLINE,
    INK,
    INK_SOFT,
    RUST,
)

apply_style()
plt.rcParams["savefig.bbox"] = "standard"  # keep the authored 180 mm page size

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
SOURCE_DIR = REPO_DIR / "source_data"  # source data of panel f (Fig6f_cnv_vs_background.csv)

PATIENTS = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]
VISIUM_SPOT_PITCH_UM = 100.0  # centre-to-centre distance of neighbouring Visium spots

# Cell type marker genes highlighted in the Moran's I analysis
CELLTYPE_MARKER_GENES = ["KRT7", "CD14", "CDH5"]

# RCTD cell types of the deposited reference -> shared Direction A cell type keys (stack order, bottom first)
CELLTYPES = [
    ("Tumour_cells", "Tumour", "Tumor"),
    ("Macrophages", "Macrophage", "Macrophage"),
    ("Fibroblasts", "Fibroblast", "Fibroblast"),
    ("T_cells", "T cell", "T cell"),
    ("B_plasma_cells", "B/plasma", "B/plasma"),
    ("Endothelial_cells", "Endothelial", "Endothelial"),
    ("Mesothelial_cells", "Mesothelial", "Mesothelial"),
    ("Myofibroblasts", "Myofibroblast", "Myofibroblast"),
]

# CNV clusters are defined within each sample, so they get their own muted set (greens, violets, rose,
# greys) that shares no tone with CELLTYPE_COLORS or the cobalt accent. Defined locally (not in
# figure_style.py). Index = cluster rank by size within the sample (Leiden labels are size-ordered).
CNV_CLUSTER_COLORS = [
    "#3E6B48",  # forest green
    "#D48BA0",  # rose
    "#A6A15A",  # olive
    "#6A5F9E",  # violet
    "#A8A8A8",  # mid grey
    "#93BA86",  # sage
    "#3D3D3D",  # charcoal
    "#BDB2DD",  # lavender
    "#EBC6D1",  # pale pink
    "#D3E5C6",  # pale mint
]

BG_GREY = "#C9C9C9"      # points of the other genes (e, f)
TOP_GREY = "#3D3D3D"     # highlighted patients in d
REST_GREY = "#ABABAB"    # other patients in d

# ── figure geometry (mm) ─────────────────────────────────────────
W_MM = DOUBLE_COL * 25.4  # 180 mm
H_MM = 160.0
MM = 1 / 25.4


def ax_mm(fig, x, y, w, h):
    """Axes at x, y (top-left, mm from the top-left corner of the figure), size w x h mm."""
    return fig.add_axes([x / W_MM, 1 - (y + h) / H_MM, w / W_MM, h / H_MM])


def fig_text(fig, x, y, s, **kw):
    return fig.text(x / W_MM, 1 - y / H_MM, s, **kw)


def panel_letter(fig, x, y, letter):
    fig_text(fig, x, y, letter, fontsize=8, fontweight="bold", color=INK, va="baseline", ha="left")


def sci_parts(v, digits=1):
    """5.486e-63 -> ('5.5', '\u221263'): mantissa and exponent with a true minus sign."""
    m, e = f"{v:.{digits}e}".split("e")
    return m, str(int(e)).replace("-", "\u2212")


def p_value_text(ax, x, y, pval, fontsize=6.5, exp_size=5.5):
    """'P = 5.5 × 10^-63' with an italic P and an exponent no smaller than 5.5 pt (mathtext would shrink it
    to 70%, below the 5 pt minimum), drawn as a separate raised text."""
    m, e = sci_parts(pval)
    t = ax.text(x, y, f"$P$ = {m} × 10", fontsize=fontsize, ha="center", va="bottom")
    ax.annotate(e, xy=(1, 0.62), xycoords=t, xytext=(0.3, 0), textcoords="offset points",
                fontsize=exp_size, ha="left", va="bottom")
    return t


# ── data ─────────────────────────────────────────────────────────
def load_patient_data(patient: str) -> dict:
    """Spot table of the ChatSpatial run: coordinates, RCTD proportions, CNV score and CNV cluster."""
    obs = pd.read_csv(SPOTS_DIR / f"{patient}.csv", index_col=0)
    xy = obs[["spatial_x", "spatial_y"]].to_numpy(dtype=float)
    # Full-resolution pixel coordinates differ in scale between sections (129 vs 207 px per spot pitch);
    # convert to micrometres with the Visium spot pitch so that all maps share one physical scale.
    nn = cKDTree(xy).query(xy, k=2)[0][:, 1]
    um = xy * (VISIUM_SPOT_PITCH_UM / np.median(nn))
    um -= (um.min(axis=0) + um.max(axis=0)) / 2
    data = {
        "xy_um": um,
        "tumour_weight": obs["RCTD_Tumour_cells"].to_numpy(dtype=float),
        "cnv_cluster": obs["cnv_clusters"].astype(int).to_numpy(),
        "n_spots": len(obs),
    }
    for col, _, _ in CELLTYPES:
        data[col] = np.nanmean(obs[f"RCTD_{col}"].values)
    return data


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


# ── panels a, b ──────────────────────────────────────────────────
def style_map(ax, lim):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(lim, -lim)  # image coordinates: y downwards
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def spot_size_pts2(ax_w_mm, lim):
    """Marker area (pt^2) so that spots of 100 um pitch tile the map without gaps."""
    pts_per_um = ax_w_mm * MM * 72 / (2 * lim)
    return (VISIUM_SPOT_PITCH_UM * pts_per_um * 1.05) ** 2


def draw_scale_bar(ax, lim, length_um=1000):
    x0 = -lim * 0.97
    y0 = lim * 0.96
    ax.plot([x0, x0 + length_um], [y0, y0], color=INK, lw=0.8, solid_capstyle="butt", clip_on=False)
    ax.text(x0 + length_um / 2, y0 - lim * 0.05, "1 mm", fontsize=5.5, color=INK,
            ha="center", va="bottom")


# ── panel e ──────────────────────────────────────────────────────
def plot_morans_ranked(ax, svg_df, cnv_markers):
    s = svg_df.sort_values("rank")
    ranks = s["rank"].values
    mi = s["I"].values
    genes = s.index.values
    is_cnv = np.array([g in cnv_markers for g in genes])
    # Other genes: every gene other than the CNV-cluster markers (19,085 genes of P3), as in the
    # statistics of cliffs_delta_cnv_vs_background.csv; CD14, KRT7 and CDH5 belong to it and are highlighted.
    is_bg = ~is_cnv
    n_genes = len(s)

    top2_rank = int(n_genes * 0.02)
    ax.axvline(top2_rank, color="#8A8A8A", lw=0.5, ls=(0, (3, 2)), zorder=0)
    ax.text(top2_rank * 1.12, 0.84, "top 2%", fontsize=5.5, color=INK_SOFT, ha="left", va="center")

    ax.scatter(ranks[is_bg], mi[is_bg], s=2.0, c=BG_GREY, edgecolors="none", rasterized=True, zorder=1)
    ax.scatter(ranks[is_cnv], mi[is_cnv], s=7, c=COBALT, edgecolors="white", linewidths=0.25,
               rasterized=True, zorder=3)

    in_top = int((is_cnv & (ranks <= top2_rank)).sum())
    print(f"    CNV markers in top 2% (rank <= {top2_rank}): {in_top} of {is_cnv.sum()}")

    ax.text(1.4, 0.36, f"CNV-cluster markers\n($n$ = {is_cnv.sum()})", fontsize=6, color=COBALT,
            ha="left", va="top", linespacing=1.15)
    ax.text(3500, 0.2, f"Other genes\n($n$ = {is_bg.sum():,})", fontsize=6, color=INK_SOFT,
            ha="left", va="bottom", linespacing=1.15)

    label_pos = {"KRT7": (60, 0.19), "CD14": (1408 * 0.55, 0.30), "CDH5": (6062 * 0.6, 0.11)}
    for g in CELLTYPE_MARKER_GENES:
        r, v = s.loc[g, "rank"], s.loc[g, "I"]
        ax.scatter(r, v, s=16, c=RUST, edgecolors="white", linewidths=0.4, zorder=5)
        tx, ty = label_pos[g]
        ax.annotate(g, xy=(r, v), xytext=(tx, ty), fontsize=6, fontstyle="italic", color=RUST,
                    ha="center", va="center", zorder=6,
                    arrowprops=dict(arrowstyle="-", color=RUST, lw=0.4, shrinkA=1.5, shrinkB=2.2))

    ax.set_xscale("log")
    ax.set_xlim(0.8, n_genes * 1.3)
    ax.set_xticks([1, 10, 100, 1000, 10000])
    ax.set_xticklabels(["1", "10", "100", "1,000", "10,000"])
    ax.minorticks_off()
    ax.set_ylim(-0.1, 0.9)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel(f"Gene rank by Moran's $I$ (P3, $n$ = {n_genes:,} genes)")
    ax.set_ylabel("Moran's $I$")
    ax.tick_params(pad=1.5)


# ── panel f ──────────────────────────────────────────────────────
def plot_morans_comparison(ax, svg_df, cnv_markers):
    genes = set(svg_df.index)
    cnv_in = cnv_markers & genes
    cnv_I = svg_df.loc[svg_df.index.isin(cnv_in), "I"].values
    # Other genes include CD14, KRT7 and CDH5, which are marked on them (19,085 genes of P3)
    bg_I = svg_df.loc[svg_df.index.isin(genes - cnv_in), "I"].values

    stat, pval = mannwhitneyu(cnv_I, bg_I, alternative="two-sided")
    print(f"    Mann-Whitney U: stat={stat:.0f}, p={pval:.4g}")
    print(f"    CNV markers: n={len(cnv_I)}, median={np.median(cnv_I):.4f}")
    print(f"    Other genes: n={len(bg_I)}, median={np.median(bg_I):.4f}")

    bp = ax.boxplot([cnv_I, bg_I], positions=[0, 1], widths=0.42, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=0.9), whiskerprops=dict(lw=0.5, color=INK),
                    capprops=dict(lw=0.5, color=INK), boxprops=dict(lw=0.5, color=INK), zorder=2)
    bp["boxes"][0].set_facecolor(COBALT_LIGHT)
    bp["boxes"][1].set_facecolor("#ECECEC")

    rng = np.random.default_rng(42)
    ax.scatter(rng.normal(0, 0.07, len(cnv_I)), cnv_I, s=5, c=COBALT, edgecolors="white",
               linewidths=0.2, alpha=0.9, rasterized=True, zorder=3)
    rng = np.random.default_rng(42)
    n_show = min(200, len(bg_I))
    bg_sample = rng.choice(bg_I, n_show, replace=False)
    ax.scatter(rng.normal(1, 0.07, n_show), bg_sample, s=2, c="#9A9A9A", edgecolors="none",
               alpha=0.6, rasterized=True, zorder=3)

    label_y = {"KRT7": 0.30, "CD14": 0.17, "CDH5": 0.06}
    for g in CELLTYPE_MARKER_GENES:
        v = svg_df.loc[g, "I"]
        ax.scatter(1, v, s=14, c=RUST, edgecolors="white", linewidths=0.4, zorder=5)
        ax.annotate(g, xy=(1, v), xytext=(1.42, label_y[g]), fontsize=6, fontstyle="italic", color=RUST,
                    ha="left", va="center", zorder=6,
                    arrowprops=dict(arrowstyle="-", color=RUST, lw=0.4, shrinkA=0.5, shrinkB=2.2))

    delta_df = pd.read_csv(STATS_DIR / "cliffs_delta_cnv_vs_background.csv")
    d, lo, hi = delta_df[["cliffs_delta", "ci_lower", "ci_upper"]].values[0]
    yb = 0.88
    ax.plot([0, 0, 1, 1], [yb - 0.025, yb, yb, yb - 0.025], color=INK, lw=0.5)
    p_value_text(ax, 0.47, yb + 0.025, pval)
    ax.text(0.5, yb + 0.15, f"Cliff's $\\delta$ = {d:.2f} [{lo:.2f}, {hi:.2f}]", fontsize=6.5,
            ha="center", va="bottom")

    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"CNV-cluster\nmarkers\n$n$ = {len(cnv_I)}",
                        f"Other\ngenes\n$n$ = {len(bg_I):,}"], fontsize=6, linespacing=1.15)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.tick_params(axis="y", pad=1.5)
    ax.set_xlim(-0.5, 1.95)
    ax.set_ylim(-0.1, 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.spines["left"].set_bounds(-0.1, 0.8)
    ax.set_ylabel("Moran's $I$")

    # Source data of panel f: every gene of the two groups with its Moran's I, the cell type markers
    # marked on the other genes, and the random subset of 200 other genes drawn as points.
    cnv_genes = svg_df.index[svg_df.index.isin(cnv_in)]
    bg_genes = svg_df.index[svg_df.index.isin(genes - cnv_in)]
    idx = np.random.default_rng(42).choice(len(bg_I), n_show, replace=False)
    assert np.array_equal(bg_I[idx], bg_sample)
    shown = np.zeros(len(bg_I), bool)
    shown[idx] = True
    sd = pd.concat([
        pd.DataFrame({"gene": cnv_genes, "group": "CNV-cluster marker", "moran_I": cnv_I,
                      "cell_type_marker": False, "drawn_as_point": True}),
        pd.DataFrame({"gene": bg_genes, "group": "Other gene", "moran_I": bg_I,
                      "cell_type_marker": bg_genes.isin(CELLTYPE_MARKER_GENES), "drawn_as_point": shown}),
    ], ignore_index=True)
    SOURCE_DIR.mkdir(exist_ok=True)
    sd.to_csv(SOURCE_DIR / "Fig6f_cnv_vs_background.csv", index=False)
    print(f"    source data: {len(cnv_I)} CNV-cluster markers, {len(bg_I):,} other genes")
    return pval


def main():
    print("Loading patient data...")
    data = {p: load_patient_data(p) for p in PATIENTS}
    svg_df = load_svg_data()
    cnv_markers = compute_cnv_markers_p3()
    summ = pd.read_csv(STATS_DIR / "hgsoc_patient_summary.csv").set_index("patient").loc[PATIENTS]
    cnv_stats = pd.read_csv(STATS_DIR / "cnv_score_per_patient_stats.csv").set_index("patient").loc[PATIENTS]

    lim = max(np.abs(data[p]["xy_um"]).max() for p in PATIENTS) * 1.04
    print(f"  map half-width: {lim:.0f} um")

    fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))

    # ── rows a, b: maps ──────────────────────────────────────────
    x0, map_w, gap = 7.0, 19.0, 1.3
    xs = [x0 + i * (map_w + gap) for i in range(len(PATIENTS))]
    ya, yb = 7.0, 31.5
    norm = Normalize(0, 1)
    ss = spot_size_pts2(map_w, lim)

    panel_letter(fig, 0.5, 3.6, "a")
    for i, p in enumerate(PATIENTS):
        d = data[p]
        ax = ax_mm(fig, xs[i], ya, map_w, map_w)
        ax.scatter(d["xy_um"][:, 0], d["xy_um"][:, 1], c=d["tumour_weight"], cmap=CMAP_SEQUENTIAL,
                   norm=norm, s=ss, marker="h", edgecolors="none", rasterized=True)
        style_map(ax, lim)
        fig_text(fig, xs[i] + map_w / 2, 3.6, p, fontsize=6.5, fontweight="bold", color=INK,
                 ha="center", va="baseline")
        if i == 0:
            draw_scale_bar(ax, lim)

    cax = ax_mm(fig, xs[-1] + map_w + 3.2, ya + 3.0, 1.6, map_w - 6.0)
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=CMAP_SEQUENTIAL), cax=cax, ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(["0", "0.5", "1"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.5, width=0.4, length=1.5, pad=1)
    fig_text(fig, xs[-1] + map_w + 1.4, ya + 1.6, "Tumor-cell\nproportion", fontsize=5.5,
             color=INK, ha="left", va="baseline", linespacing=1.1)

    panel_letter(fig, 0.5, yb - 3.4, "b")
    max_k = 0
    for i, p in enumerate(PATIENTS):
        d = data[p]
        ax = ax_mm(fig, xs[i], yb, map_w, map_w)
        k = len(np.unique(d["cnv_cluster"]))
        max_k = max(max_k, k)
        ax.scatter(d["xy_um"][:, 0], d["xy_um"][:, 1],
                   c=[CNV_CLUSTER_COLORS[c] for c in d["cnv_cluster"]],
                   s=ss, marker="h", edgecolors="none", rasterized=True)
        style_map(ax, lim)
        fig_text(fig, xs[i] + map_w / 2, yb + map_w + 2.6, f"{k} clusters", fontsize=5.5, color=INK_SOFT,
                 ha="center", va="baseline")

    # swatch legend for b, below the maps
    ly = yb + map_w + 6.6
    fig_text(fig, x0, ly, "CNV cluster within each sample, numbered by size (not matched across samples):",
             fontsize=5.5, color=INK, ha="left", va="center")
    sx = x0 + 89.5
    for j in range(max_k):
        fig.patches.append(plt.Rectangle(((sx + j * 7.2) / W_MM, 1 - (ly + 1.1) / H_MM),
                                         2.2 / W_MM, 2.2 / H_MM, transform=fig.transFigure,
                                         facecolor=CNV_CLUSTER_COLORS[j], edgecolor="none"))
        fig_text(fig, sx + j * 7.2 + 2.9, ly, str(j + 1), fontsize=5.5, color=INK, ha="left", va="center")

    # ── row c, d ─────────────────────────────────────────────────
    yr3 = 68.0
    h3 = 34.0
    panel_letter(fig, 0.5, yr3 - 3.4, "c")
    ax_c = ax_mm(fig, 12.0, yr3, 62.0, h3)
    xpos = np.arange(len(PATIENTS))
    frac = np.array([[data[p][col] for col, _, _ in CELLTYPES] for p in PATIENTS])
    frac = frac / frac.sum(axis=1, keepdims=True)
    bottom = np.zeros(len(PATIENTS))
    for j, (_, key, _) in enumerate(CELLTYPES):
        ax_c.bar(xpos, frac[:, j], bottom=bottom, width=0.72, color=CELLTYPE_COLORS[key],
                 edgecolor="white", linewidth=0.3)
        bottom += frac[:, j]
    ax_c.set_xticks(xpos)
    ax_c.set_xticklabels(PATIENTS)
    ax_c.tick_params(axis="x", length=0, pad=2)
    ax_c.set_xlim(-0.55, len(PATIENTS) - 0.45)
    ax_c.set_ylim(0, 1)
    ax_c.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax_c.set_yticklabels(["0", "0.25", "0.50", "0.75", "1"])
    ax_c.set_ylabel("Mean cell type proportion")
    ax_c.spines["bottom"].set_visible(False)
    # legend in stack order (top of the stack first)
    lx, ly0, dy = 76.5, yr3 + 1.6, h3 / len(CELLTYPES) * 0.98
    for j, (_, key, disp) in enumerate(reversed(CELLTYPES)):
        yy = ly0 + j * dy
        fig.patches.append(plt.Rectangle((lx / W_MM, 1 - (yy + 1.1) / H_MM), 2.2 / W_MM, 2.2 / H_MM,
                                         transform=fig.transFigure, facecolor=CELLTYPE_COLORS[key],
                                         edgecolor="#BDBDBD" if key == "Mesothelial" else "none",
                                         linewidth=0.3))
        fig_text(fig, lx + 3.2, yy, disp, fontsize=6, color=INK, ha="left", va="center")

    # d: tumor burden (top) and CNV score (bottom), sharing the patient axis
    tumour_pct = summ["n_tumor_enriched"].values / summ["n_spots"].values * 100
    top3 = set(np.argsort(-tumour_pct)[:3])
    cols = [TOP_GREY if i in top3 else REST_GREY for i in range(len(PATIENTS))]
    print("  top-3 tumor burden:", [PATIENTS[i] for i in sorted(top3)])

    xd, wd = 113.0, 64.0
    hd = 14.5
    panel_letter(fig, 100.0, yr3 - 3.4, "d")
    ax_d1 = ax_mm(fig, xd, yr3, wd, hd)
    ax_d1.bar(xpos, tumour_pct, width=0.62, color=cols, edgecolor="none")
    for i, v in enumerate(tumour_pct):
        ax_d1.text(i, v + 2.5, f"{v:.0f}", fontsize=5.5, color=INK_SOFT, ha="center", va="bottom")
    ax_d1.set_ylim(0, 100)
    ax_d1.set_yticks([0, 50, 100])
    ax_d1.set_xlim(-0.55, len(PATIENTS) - 0.45)
    ax_d1.set_xticks(xpos)
    ax_d1.set_xticklabels([])
    ax_d1.tick_params(axis="x", length=0)
    ax_d1.spines["bottom"].set_visible(False)
    ax_d1.set_ylabel("Tumor-enriched\nspots (%)", linespacing=1.1)

    ax_d2 = ax_mm(fig, xd, yr3 + hd + 5.0, wd, hd)
    mean_cnv = cnv_stats["mean_cnv_score"].values
    sd_cnv = cnv_stats["sd_cnv_score"].values
    assert np.allclose(mean_cnv, summ["mean_cnv_score"].values)
    for i in range(len(PATIENTS)):
        ax_d2.errorbar(i, mean_cnv[i], yerr=sd_cnv[i], fmt="o", ms=3.8, mfc=cols[i], mec="white",
                       mew=0.4, ecolor="black", elinewidth=0.6, capsize=1.6, capthick=0.6, zorder=3)
    ax_d2.set_ylim(0, 0.02)
    ax_d2.set_yticks([0, 0.01, 0.02])
    ax_d2.set_yticklabels(["0", "0.01", "0.02"])
    ax_d2.set_xlim(-0.55, len(PATIENTS) - 0.45)
    ax_d2.set_xticks(xpos)
    ax_d2.set_xticklabels(PATIENTS)
    ax_d2.tick_params(axis="x", length=0, pad=2)
    ax_d2.spines["bottom"].set_visible(False)
    ax_d2.set_ylabel("CNV score\n(mean ± s.d.)", linespacing=1.1)
    for a in (ax_d1, ax_d2):
        a.tick_params(axis="y", pad=1.5)

    # ── row e, f ─────────────────────────────────────────────────
    yr4 = 117.0
    h4 = 33.0
    panel_letter(fig, 0.5, yr4 - 3.4, "e")
    ax_e = ax_mm(fig, 12.0, yr4, 86.0, h4)
    plot_morans_ranked(ax_e, svg_df, cnv_markers)

    panel_letter(fig, 109.0, yr4 - 3.4, "f")
    ax_f = ax_mm(fig, 122.0, yr4, 55.0, h4)
    plot_morans_comparison(ax_f, svg_df, cnv_markers)

    # align y labels of the left column and of the right column
    fig.align_ylabels([ax_c, ax_e])
    fig.align_ylabels([ax_d1, ax_d2])

    for fmt in ["pdf", "png"]:
        out = FIGURE_DIR / f"figure3.{fmt}"
        fig.savefig(str(out), dpi=300, facecolor="white", bbox_inches=None)
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
