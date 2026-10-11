"""
Figure 5: OSCC spatial transcriptomics case study
Nature Communications style, double-column (183 mm)

Layout: 3x3 grid (a-i), 183 mm x ~160 mm

Row 1 - Spatial Discovery:
  [a] H&E + TC/LE overlay (s1)
  [b] H&E + TC/LE overlay (s9)
  [c] Fibroblast spatial map (s1)

Row 2 - Cellular Composition:
  [d] Per-sample enrichment heatmap (10 x 12)
  [e] Consensus bar chart
  [f] CellChat L-R dot plot

Row 3 - Molecular Mechanism:
  [g] FN1 spatial expression (s1)
  [h] Moran's I lollipop chart
  [i] TC/LE marker gene summary
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from PIL import Image as PILImage

from matplotlib.colors import LinearSegmentedColormap

from figure_style import (
    FULL_WIDTH,
    OI,
    add_panel_label,
    apply_style,
)

# Custom diverging colormap: TC-blue ← white → LE-vermillion
# Semantically matches the TC/LE overlay colors in panels a/b
_TC_LE_CMAP = LinearSegmentedColormap.from_list(
    "tc_le_diverging",
    [OI["blue"], "#FFFFFF", OI["vermillion"]],
    N=256,
)

# Sequential colormap for LE-associated continuous data (fibroblast, FN1)
# Starts from a light warm tint (not pure white) so colorbar is visible on white bg
_VERM_SEQ = LinearSegmentedColormap.from_list(
    "vermillion_seq",
    ["#FDF0E8", "#E8985A", OI["vermillion"]],
    N=256,
)

# ── Apply global style ────────────────────────────────────────────
apply_style()

# ── Paths ─────────────────────────────────────────────────────────
FIGURE_DIR = Path(__file__).resolve().parent
REPO_DIR = FIGURE_DIR.parent
# Raw GEO files of GSE208253 (tissue_hires_image.png, tissue_positions_list.csv, scalefactors_json.json,
# named {GSM}_{sample}_*), needed only for the H&E backgrounds and spot positions of panels a-c and g.
GEO_DIR = Path(os.environ.get("OSCC_GEO_DIR", REPO_DIR / "raw_data" / "GSE208253"))
CASE_DIR = REPO_DIR / "experiments" / "case_studies" / "oscc" / "results"
# Every panel reads the ChatSpatial 1.5.7 run, in which each analysis step was an MCP tool call
# (oscc/results/oscc157_tool_calls.jsonl): spot labels, FN1 expression, proportions, Moran's I and CellChat
# tables written by oscc/scripts/oscc157_collect.py; Fig. 5d,e statistics and the Sample 1 Moran's I
# summary written by experiments/case_studies/compute_case_study_stats.py.
RUN_DIR = CASE_DIR / "tables"
ANALYSIS_DIR = CASE_DIR / "stats"
SPOTS_DIR = RUN_DIR / "spots"

# ── Constants ─────────────────────────────────────────────────────
SAMPLE_GEO = {f"s{i}": f"GSM{6339630 + i}" for i in range(1, 13)}
SAMPLES = [f"s{i}" for i in range(1, 13)]

CELLTYPE_ORDER = [
    "B cell",
    "T cell",
    "Fibroblast",
    "myocyte",
    "-Fibroblast",
    "Endothelial",
    "Dendritic",
    "Mast",
    "Macrophage",
    "Cancer",
]

CELLTYPE_DISPLAY: Dict[str, str] = {
    "B cell": "B cell",
    "T cell": "T cell",
    "Fibroblast": "Fibroblast",
    "myocyte": "Myocyte",
    "-Fibroblast": "Myofibroblast",
    "Endothelial": "Endothelial",
    "Dendritic": "Dendritic",
    "Mast": "Mast cell",
    "Macrophage": "Macrophage",
    "Cancer": "Cancer",
}

# Spot annotation colors
COLOR_TC = OI["blue"]       # #0072B2
COLOR_LE = OI["vermillion"]  # #D55E00
COLOR_OTHER = "#CCCCCC"


# ── Helpers ───────────────────────────────────────────────────────

def _load_he_image(sid: str) -> Optional[np.ndarray]:
    gsm = SAMPLE_GEO[sid]
    path = GEO_DIR / f"{gsm}_{sid}_tissue_hires_image.png"
    if not path.exists():
        return None
    return np.array(PILImage.open(path))


def _load_tissue_positions(sid: str) -> Optional[pd.DataFrame]:
    import json
    gsm = SAMPLE_GEO[sid]
    path = GEO_DIR / f"{gsm}_{sid}_tissue_positions_list.csv"
    if not path.exists():
        return None
    df = pd.read_csv(
        path, header=None,
        names=["barcode", "in_tissue", "row", "col", "pixel_row", "pixel_col"],
    )
    df = df[df["in_tissue"] == 1].copy()
    sf_path = GEO_DIR / f"{gsm}_{sid}_scalefactors_json.json"
    if sf_path.exists():
        with open(sf_path) as f:
            sf = json.load(f)
        scale = sf.get("tissue_hires_scalef", 1.0)
        df["pixel_row"] = df["pixel_row"] * scale
        df["pixel_col"] = df["pixel_col"] * scale
    return df


def _load_h5ad(sid: str) -> Any:
    """Spot table of the ChatSpatial run: Leiden domain, TC/LE label and FN1 (log-normalized)."""
    path = SPOTS_DIR / f"{sid}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


SpatialData = Tuple[Optional[np.ndarray], Optional[pd.DataFrame], Optional[Any]]


def _load_spatial_data(sid: str) -> SpatialData:
    return _load_he_image(sid), _load_tissue_positions(sid), _load_h5ad(sid)


def _autocrop_limits(img: np.ndarray, pad: int = 40) -> Tuple[float, float, float, float]:
    gray = np.mean(img[:, :, :3], axis=2)
    mask = gray < 230
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return (
            max(0, cmin - pad), min(img.shape[1], cmax + pad),
            min(img.shape[0], rmax + pad), max(0, rmin - pad),
        )
    return (0, img.shape[1], img.shape[0], 0)


def _strip_axes(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _fallback_text(ax: plt.Axes, msg: str) -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=6,
            transform=ax.transAxes, color="#999999", style="italic")


# ── Shared spatial overlay helpers ────────────────────────────────

TC_LE_ANNOT_COLS = ["tc_le_region", "spatial_domain", "region", "annotation", "cluster"]


def _prepare_he_faded(img: np.ndarray, tissue_alpha: float = 0.30) -> np.ndarray:
    """Fade H&E tissue toward white and whiten the grey slide background.

    Returns an RGB uint8 array where:
    - Slide background (uniform grey) → pure white
    - Tissue → blended toward white by (1 - tissue_alpha)
    This lets overlay spots pop against a clean, muted background.
    """
    rgb = img[:, :, :3].astype(np.float64) / 255.0
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    lum = rgb.mean(axis=2)
    # Slide background: low-saturation mid-grey
    bg = (sat < 0.06) & (lum > 0.60) & (lum < 0.96)
    # Dark slide edges / debris
    bg |= (lum < 0.12)
    rgb[bg] = 1.0
    # Blend remaining tissue toward white
    rgb = tissue_alpha * rgb + (1.0 - tissue_alpha)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def _draw_tc_le_overlay(ax: plt.Axes, sid: str, label: str, title: str) -> None:
    img, pos, adata = _load_spatial_data(sid)

    if img is not None:
        faded = _prepare_he_faded(img, tissue_alpha=0.30)
        ax.imshow(faded, interpolation="lanczos")
        xlim = _autocrop_limits(img)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(xlim[2], xlim[3])
        ax.set_facecolor("white")

    annotated = False
    if adata is not None and pos is not None:
        obs = adata
        annot_col = next((c for c in TC_LE_ANNOT_COLS if c in obs.columns), None)
        if annot_col is not None:
            merged = pos.merge(obs[[annot_col]], left_on="barcode",
                               right_index=True, how="left")
            region_str = merged[annot_col].astype(str).fillna("")
            tc_mask = region_str.apply(
                lambda v: any(k in v.upper() for k in ("TC", "TUMOR", "CORE")))
            le_mask = region_str.apply(
                lambda v: any(k in v.upper() for k in ("LE", "LEADING", "EDGE")))
            other_mask = ~tc_mask & ~le_mask

            # Other spots: subtle grey to show tissue shape
            if other_mask.any():
                ax.scatter(merged.loc[other_mask, "pixel_col"],
                           merged.loc[other_mask, "pixel_row"],
                           c="#E0E0E0", s=8, alpha=0.5, linewidths=0,
                           rasterized=False, zorder=1)
            # TC spots
            ax.scatter(merged.loc[tc_mask, "pixel_col"],
                       merged.loc[tc_mask, "pixel_row"],
                       c=COLOR_TC, s=16, alpha=0.80, linewidths=0,
                       rasterized=False, zorder=2)
            # LE spots
            ax.scatter(merged.loc[le_mask, "pixel_col"],
                       merged.loc[le_mask, "pixel_row"],
                       c=COLOR_LE, s=16, alpha=0.80, linewidths=0,
                       rasterized=False, zorder=2)

            handles = [
                mpatches.Patch(color=COLOR_TC, label="TC"),
                mpatches.Patch(color=COLOR_LE, label="LE"),
            ]
            ax.legend(handles=handles, loc="lower right", frameon=False,
                      fontsize=5, handlelength=0.6, handletextpad=0.3,
                      borderpad=0.2, labelspacing=0.2)
            annotated = True

    if not annotated:
        if pos is not None and img is not None:
            ax.scatter(pos["pixel_col"], pos["pixel_row"],
                       c=COLOR_OTHER, s=6, alpha=0.6, linewidths=0, rasterized=False)
            _fallback_text(ax, "Annotations pending")
        elif img is None:
            _fallback_text(ax, "H&E not found")

    _strip_axes(ax)
    ax.set_title(title, fontsize=7, pad=2)
    add_panel_label(ax, label)


# ── Panel a: H&E + TC/LE (s1) ────────────────────────────────────

def draw_panel_a(ax: plt.Axes) -> None:
    _draw_tc_le_overlay(ax, "s1", "a", "Sample 1")


# ── Panel b: H&E + TC/LE (s9) ────────────────────────────────────

def draw_panel_b(ax: plt.Axes) -> None:
    _draw_tc_le_overlay(ax, "s9", "b", "Sample 9")


# ── Panel c: Fibroblast deconvolution spatial map (s1) ────────────

def draw_panel_c(ax: plt.Axes) -> None:
    img, pos, adata = _load_spatial_data("s1")

    if img is not None:
        faded = _prepare_he_faded(img, tissue_alpha=0.25)
        ax.imshow(faded, interpolation="lanczos")
        xlim = _autocrop_limits(img)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(xlim[2], xlim[3])
        ax.set_facecolor("white")

    fib_col = "Fibroblast"
    prop_path = RUN_DIR / "proportions" / "s1.csv"
    props = pd.read_csv(prop_path, index_col=0) if prop_path.exists() else None
    if props is not None and pos is not None and fib_col in props.columns:
        merged = pos.merge(props[[fib_col]], left_on="barcode",
                           right_index=True, how="left")
        vals = merged[fib_col].fillna(0).values.astype(float)
        sc = ax.scatter(merged["pixel_col"], merged["pixel_row"],
                        c=vals, cmap=_VERM_SEQ, s=6, alpha=0.8, linewidths=0,
                        rasterized=False, vmin=0,
                        vmax=np.nanmax(vals) if np.nanmax(vals) > 0 else 1)
        cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, shrink=0.6)
        cbar.set_label("Proportion", fontsize=5, labelpad=2)
        cbar.ax.tick_params(labelsize=4.5, width=0.3, length=1.5)
        cbar.outline.set_linewidth(0.5)
        cbar.outline.set_edgecolor("#666666")
    elif pos is not None and img is not None:
        ax.scatter(pos["pixel_col"], pos["pixel_row"],
                   c=COLOR_OTHER, s=6, alpha=0.6, linewidths=0, rasterized=False)
        _fallback_text(ax, "Deconvolution\npending")
    elif img is None:
        _fallback_text(ax, "H&E not found")

    _strip_axes(ax)
    ax.set_title("Fibroblast", fontsize=7, pad=2)
    add_panel_label(ax, "c")


# ── Panel d: Per-sample enrichment heatmap ────────────────────────

def draw_panel_d(ax: plt.Axes) -> None:
    path = ANALYSIS_DIR / "per_sample_tc_le_enrichment.csv"
    df = pd.read_csv(path)

    pivot_fc = df.pivot(index="cell_type", columns="sample", values="log2fc"
                        ).reindex(index=CELLTYPE_ORDER, columns=SAMPLES)

    mat = pivot_fc.values.astype(float)
    vmax = 5.0
    mat_clipped = np.clip(mat, -vmax, vmax)

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(mat_clipped, aspect="auto", cmap=_TC_LE_CMAP, norm=norm,
                   interpolation="nearest")
    # Lock y-limits to match panel e's barh y = arange(n)
    ax.set_ylim(len(CELLTYPE_ORDER) - 0.5, -0.5)

    # Separator between LE-enriched and TC-enriched
    ax.axhline(y=8.5, color="black", linewidth=0.8, linestyle="--", zorder=5)

    display_names = [CELLTYPE_DISPLAY.get(ct, ct) for ct in CELLTYPE_ORDER]
    ax.set_yticks(range(len(CELLTYPE_ORDER)))
    ax.set_yticklabels(display_names, fontsize=5.5)
    ax.set_xticks(range(len(SAMPLES)))
    ax.set_xticklabels([str(i + 1) for i in range(len(SAMPLES))], fontsize=5.5)
    ax.set_xlabel("Sample", fontsize=6)
    ax.tick_params(length=2, width=0.4)

    ax.set_title(r"Cell type enrichment (log$_2$ FC)", fontsize=7, pad=3)

    # Restore spines for heatmap
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.3)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, shrink=0.8)
    cbar.set_label(r"log$_2$ FC", fontsize=5)
    cbar.ax.tick_params(labelsize=5, width=0.3, length=1.5)
    cbar.outline.set_linewidth(0.3)

    add_panel_label(ax, "d")


# ── Panel e: Consensus bar chart ──────────────────────────────────

def draw_panel_e(ax: plt.Axes) -> None:
    path = ANALYSIS_DIR / "consensus_tc_le_patterns.csv"
    consensus = pd.read_csv(path, index_col=0)
    consensus = consensus.reindex(CELLTYPE_ORDER).copy()

    le_counts = consensus["le_enriched_count"].values
    tc_counts = consensus["tc_enriched_count"].values

    y = np.arange(len(CELLTYPE_ORDER))
    bar_h = 0.55

    ax.barh(y, le_counts, height=bar_h, color=OI["vermillion"],
            edgecolor="white", linewidth=0.3, label="LE enriched", zorder=3)
    ax.barh(y, -tc_counts, height=bar_h, color=OI["blue"],
            edgecolor="white", linewidth=0.3, label="TC enriched", zorder=3)

    # Clopper-Pearson 95% CI error bars
    ci_path = ANALYSIS_DIR / "consensus_binomial_ci.csv"
    if ci_path.exists():
        ci_df = pd.read_csv(ci_path)
        for i, ct in enumerate(CELLTYPE_ORDER):
            le_row = ci_df[(ci_df["cell_type"] == ct) & (ci_df["direction"] == "LE")]
            if len(le_row) == 1:
                lo = le_row["ci_lower"].values[0] * 12
                hi = le_row["ci_upper"].values[0] * 12
                k = le_row["k"].values[0]
                if k > 0:
                    ax.plot([lo, hi], [y[i], y[i]], color="#333333",
                            linewidth=0.5, zorder=4, solid_capstyle="round")
            tc_row = ci_df[(ci_df["cell_type"] == ct) & (ci_df["direction"] == "TC")]
            if len(tc_row) == 1:
                lo = tc_row["ci_lower"].values[0] * 12
                hi = tc_row["ci_upper"].values[0] * 12
                k = tc_row["k"].values[0]
                if k > 0:
                    ax.plot([-hi, -lo], [y[i], y[i]], color="#333333",
                            linewidth=0.5, zorder=4, solid_capstyle="round")

    # 75% threshold line
    ax.axvline(x=9, color="black", linewidth=0.5, linestyle="--", alpha=0.6)
    ax.text(9.2, len(CELLTYPE_ORDER) - 0.5, "75%", fontsize=4,
            va="top", ha="left", style="italic", color="#555555")

    ax.set_yticks(y)
    ax.set_yticklabels([])  # shared with panel d
    ax.set_xlabel("# samples", fontsize=6)
    ax.set_xlim(min(-7, -(np.nanmax(tc_counts) + 1)), 12)
    ax.set_xticks([-5, 0, 5, 10])
    ax.set_ylim(len(CELLTYPE_ORDER) - 0.5, -0.5)  # match panel d
    ax.axvline(x=0, color="black", linewidth=0.4, zorder=1)

    ax.legend(loc="upper right", frameon=False, fontsize=4.5,
              handletextpad=0.2, handlelength=0.5, borderpad=0.1, labelspacing=0.1)
    ax.set_title("Consensus (n = 12)", fontsize=7, pad=3)
    add_panel_label(ax, "e")


# ── Panel f: CellChat L-R dot plot ────────────────────────────────

def draw_panel_f(ax: plt.Axes) -> None:
    caf_path = RUN_DIR / "cellchat_tc_le_caf_cancer_interactions.csv"
    lr_path = RUN_DIR / "cellchat_tc_le_lr_pairs.csv"

    path = caf_path if caf_path.exists() else lr_path
    if path.exists():
        _draw_cellchat_dotplot(ax, path, caf_only=caf_path.exists())
    else:
        _fallback_text(ax, "CellChat data\nnot found")

    add_panel_label(ax, "f")


def _draw_cellchat_dotplot(ax: plt.Axes, path: Path, caf_only: bool = False) -> None:
    df = pd.read_csv(path)
    if caf_only and {"source", "target"} <= set(df.columns):
        # The panel shows the Fibroblast -> Cancer direction named in its title.
        df = df[(df["source"] == "Fibroblast") & (df["target"] == "Cancer")]

    prob_col = next((c for c in ["prob", "probability"] if c in df.columns), None)
    name_col = next((c for c in ["interaction_name_2", "interaction_name"]
                     if c in df.columns), None)
    pw_col = next((c for c in ["pathway_name"] if c in df.columns), None)

    if prob_col is None or name_col is None:
        _fallback_text(ax, "CellChat data\nnot found")
        return

    if pw_col and caf_only:
        top_per_pw = (
            df.sort_values(prob_col, ascending=False)
            .groupby(pw_col).first().reset_index()
            .nlargest(10, prob_col)
        )
        df_sorted = top_per_pw.sort_values(prob_col, ascending=True)
    else:
        df_sorted = df.nlargest(10, prob_col).sort_values(prob_col, ascending=True)

    if "ligand" in df_sorted.columns and "receptor" in df_sorted.columns:
        names = [f"{l}\u2192{r}" for l, r in
                 zip(df_sorted["ligand"].values, df_sorted["receptor"].values)]
    else:
        names = df_sorted[name_col].values
    probs = df_sorted[prob_col].values.astype(float)

    pw_colors = {
        "COLLAGEN": OI["blue"], "FN1": OI["vermillion"],
        "TENASCIN": OI["skyblue"], "THBS": OI["orange"],
        "ADGRG": OI["redpurple"], "LAMININ": OI["green"],
    }
    dot_colors = []
    if pw_col:
        for pw in df_sorted[pw_col].values:
            dot_colors.append(pw_colors.get(str(pw), OI["grey"]))
    else:
        dot_colors = [OI["orange"]] * len(probs)

    y_pos = np.arange(len(names))
    ax.scatter(probs, y_pos, c=dot_colors, s=35, edgecolors="grey",
               linewidths=0.3, zorder=3)

    ax.set_yticks(y_pos)
    # Format L-R pair labels: italicize gene names, keep arrow plain
    def _format_lr_label(name: str) -> str:
        name = name.replace("_", "+")  # complex subunits (ITGAV_ITGB5) read as ITGAV+ITGB5, not a subscript
        if "\u2192" in name:
            ligand, receptor = name.split("\u2192", 1)
            return f"$\\it{{{ligand}}}$\u2192$\\it{{{receptor}}}$"
        return f"$\\it{{{name}}}$"
    ax.set_yticklabels([_format_lr_label(n) for n in names], fontsize=5)
    for i, n in enumerate(names):
        if "FN1" in str(n).upper() or "COL1A1" in str(n).upper():
            ax.get_yticklabels()[i].set_fontweight("bold")

    ax.set_xlabel("Comm. probability", fontsize=6)
    title = "Fibroblast \u2192 Cancer" if caf_only else "CellChat L-R pairs"
    ax.set_title(title, fontsize=7, pad=3)
    ax.tick_params(length=2, width=0.4)

    if pw_col and caf_only:
        used_pws = set(str(p) for p in df_sorted[pw_col].values)
        handles = [mpatches.Patch(color=c, label=pw)
                   for pw, c in pw_colors.items() if pw in used_pws]
        if handles:
            ax.legend(handles=handles, loc="lower right", frameon=False,
                      fontsize=4.5, handlelength=0.6, handletextpad=0.3,
                      borderpad=0.2, labelspacing=0.15)


# ── Panel g: FN1 spatial expression (s1) ──────────────────────────

def draw_panel_g(ax: plt.Axes) -> None:
    img, pos, adata = _load_spatial_data("s1")

    if img is not None:
        faded = _prepare_he_faded(img, tissue_alpha=0.25)
        ax.imshow(faded, interpolation="lanczos")
        xlim = _autocrop_limits(img)
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(xlim[2], xlim[3])
        ax.set_facecolor("white")

    has_fn1 = False
    if adata is not None and pos is not None:
        if "FN1_lognorm" in adata.columns:
            expr_df = pd.DataFrame({"expression": adata["FN1_lognorm"].values}, index=adata.index)
            merged = pos.merge(expr_df, left_on="barcode", right_index=True, how="left")
            vals = merged["expression"].fillna(0).values
            sc = ax.scatter(merged["pixel_col"], merged["pixel_row"],
                            c=vals, cmap=_VERM_SEQ, s=6, alpha=0.85,
                            linewidths=0, rasterized=False, vmin=0)
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            cax = inset_axes(ax, width="4%", height="30%", loc="lower left",
                             borderpad=0.5)
            cbar = plt.colorbar(sc, cax=cax)
            cbar.set_label("Expr.", fontsize=4, labelpad=1)
            cbar.ax.tick_params(labelsize=3.5, width=0.3, length=1)
            cbar.outline.set_linewidth(0.5)
            cbar.outline.set_edgecolor("#666666")
            has_fn1 = True

    if not has_fn1:
        if pos is not None and img is not None:
            ax.scatter(pos["pixel_col"], pos["pixel_row"],
                       c=COLOR_OTHER, s=6, alpha=0.6, linewidths=0, rasterized=False)
        _fallback_text(ax, "FN1 expression\npending")

    _strip_axes(ax)
    ax.set_title(r"$\it{FN1}$", fontsize=7, pad=2)
    add_panel_label(ax, "g")


# ── Panel h: Moran's I lollipop chart ─────────────────────────────

def draw_panel_h(ax: plt.Axes) -> None:
    # Sample 1 Moran's I of the target genes (scripts/compute_case_study_stats.py)
    single_path = ANALYSIS_DIR / "svg_target_genes_summary.csv"

    if single_path.exists():
        df = _load_single_moran(single_path)
    else:
        _fallback_text(ax, "Moran's I data\nnot found")
        _strip_axes(ax)
        add_panel_label(ax, "h")
        return

    df = df.dropna(subset=["moranI"]).sort_values("moranI", ascending=True).copy()

    y_pos = np.arange(len(df))
    colors = []
    for cat in df["category"].fillna("Unknown"):
        cat_upper = str(cat).upper()
        if "LE" in cat_upper:
            colors.append(OI["vermillion"])
        elif "TC" in cat_upper:
            colors.append(OI["blue"])
        else:
            colors.append("#999999")

    # Lollipop: horizontal lines + dots
    for i, (yp, val, col) in enumerate(zip(y_pos, df["moranI"].values, colors)):
        ax.plot([0, val], [yp, yp], color=col, linewidth=1.0, zorder=2)
        ax.plot(val, yp, "o", color=col, markersize=4, zorder=3,
                markeredgecolor="white", markeredgewidth=0.3)

    gene_labels = [f"$\\it{{{g}}}$" for g in df["gene"].values]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(gene_labels, fontsize=5.5)
    ax.set_xlabel("Moran's I", fontsize=6)
    ax.set_title("Spatial autocorrelation", fontsize=7, pad=3)
    ax.tick_params(length=2, width=0.4)

    handles = [
        mpatches.Patch(color=OI["vermillion"], label="LE marker"),
        mpatches.Patch(color=OI["blue"], label="TC marker"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False,
              fontsize=5, handlelength=0.6, handletextpad=0.3)

    add_panel_label(ax, "h")


def _load_single_moran(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[["gene", "category", "moranI"]].copy()


# ── Panel i: TC/LE marker gene summary ────────────────────────────

def draw_panel_i(ax: plt.Axes) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    _strip_axes(ax)

    # TC markers (top)
    tc_box = mpatches.FancyBboxPatch(
        (0.3, 5.5), 9.4, 4.0,
        boxstyle="round,pad=0.15", facecolor="#EEF2F8",
        edgecolor=OI["blue"], linewidth=0.6)
    ax.add_patch(tc_box)
    ax.text(5, 9.0, "Tumor Core", ha="center", va="center",
            fontsize=7, fontweight="bold", color=OI["blue"])
    ax.text(5, 7.5, r"$\it{SPRR1B}$, $\it{CLDN4}$,"
            "\n" r"$\it{SPRR2E}$, $\it{KRT6C}$",
            ha="center", va="center", fontsize=6, color="#333333", linespacing=1.4)
    ax.text(5, 6.1, "Keratinization /\ndifferentiation",
            ha="center", va="center", fontsize=5.5, color="#888888",
            style="italic", linespacing=1.3)

    # LE markers (bottom)
    le_box = mpatches.FancyBboxPatch(
        (0.3, 0.5), 9.4, 4.0,
        boxstyle="round,pad=0.15", facecolor="#FDF3E8",
        edgecolor=OI["vermillion"], linewidth=0.6)
    ax.add_patch(le_box)
    ax.text(5, 4.0, "Leading Edge", ha="center", va="center",
            fontsize=7, fontweight="bold", color=OI["vermillion"])
    ax.text(5, 2.5, r"$\it{FN1}$, $\it{LAMC2}$,"
            "\n" r"$\it{ITGA5}$, $\it{COL1A1}$",
            ha="center", va="center", fontsize=6, color="#333333", linespacing=1.4)
    ax.text(5, 1.1, "Partial-EMT /\ninvasion front",
            ha="center", va="center", fontsize=5.5, color="#888888",
            style="italic", linespacing=1.3)

    add_panel_label(ax, "i")


# ── Main ──────────────────────────────────────────────────────────

def main() -> None:
    print("Building Figure 5: OSCC case study")
    print(f"  Data dir: {GEO_DIR}")
    print(f"  ChatSpatial run: {RUN_DIR}")
    print()

    fig_w = FULL_WIDTH  # 7.2 in = 183 mm
    fig_h = 6.3         # ~160 mm

    fig = plt.figure(figsize=(fig_w, fig_h))

    # ── 3x3 GridSpec ─────────────────────────────────────────────
    gs = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[1.0, 1.0, 1.0],
        hspace=0.42,
        left=0.08, right=0.96, top=0.96, bottom=0.04,
    )

    # Row 1: equal 3 spatial maps
    gs_row1 = gs[0].subgridspec(1, 3, wspace=0.12)
    ax_a = fig.add_subplot(gs_row1[0])
    ax_b = fig.add_subplot(gs_row1[1])
    ax_c = fig.add_subplot(gs_row1[2])

    # Row 2: [d + e] tight (shared y-axis) | gap | [f] separate
    gs_row2 = gs[1].subgridspec(1, 2, width_ratios=[0.70, 0.30], wspace=0.40)
    gs_de = gs_row2[0].subgridspec(1, 2, width_ratios=[0.68, 0.32], wspace=0.08)
    ax_d = fig.add_subplot(gs_de[0])
    ax_e = fig.add_subplot(gs_de[1])
    ax_f = fig.add_subplot(gs_row2[1])

    # Row 3: FN1 (33%) + Moran's I (33%) + marker summary (34%)
    gs_row3 = gs[2].subgridspec(1, 3, width_ratios=[0.33, 0.33, 0.34], wspace=0.35)
    ax_g = fig.add_subplot(gs_row3[0])
    ax_h = fig.add_subplot(gs_row3[1])
    ax_i = fig.add_subplot(gs_row3[2])

    # ── Draw all panels ───────────────────────────────────────────
    print("Drawing Row 1: Spatial discovery...")
    draw_panel_a(ax_a)
    draw_panel_b(ax_b)
    draw_panel_c(ax_c)

    print("Drawing Row 2: Cellular composition...")
    draw_panel_d(ax_d)
    draw_panel_e(ax_e)
    draw_panel_f(ax_f)

    print("Drawing Row 3: Molecular mechanism...")
    draw_panel_g(ax_g)
    draw_panel_h(ax_h)
    draw_panel_i(ax_i)

    # ── Save ──────────────────────────────────────────────────────
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ["pdf", "png"]:
        out_path = FIGURE_DIR / f"figure2.{fmt}"
        fig.savefig(str(out_path), dpi=300, facecolor="white")
        print(f"  Saved: {out_path}")

    plt.close(fig)
    print("\nDone!")


if __name__ == "__main__":
    main()
