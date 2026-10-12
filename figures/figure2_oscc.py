"""
Figure 5 (file name figure2): OSCC spatial transcriptomics case study.
Direction A ("Ink & Cobalt") style, authored at 180 mm width.

Row 1 - Spatial discovery:
  [a] H&E + TC/LE spots (Sample 1)
  [b] H&E + TC/LE spots (Sample 9)
  [c] Fibroblast deconvolution proportion (Sample 1)

Row 2 - Cellular composition:
  [d] Per-sample LE/TC enrichment heatmap (10 cell types x 12 samples)
  [e] Consensus bar chart (rows shared with d)
  [f] CellChat Fibroblast -> Cancer interactions, ECM-syndecan axis highlighted

Row 3 - Molecular mechanism:
  [g] FN1 spatial expression (Sample 1)
  [h] Moran's I of the TC and LE target genes, grouped by programme
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from PIL import Image as PILImage  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from figure_style import (  # noqa: E402
    CMAP_DIVERGING,
    CMAP_SEQUENTIAL,
    DOUBLE_COL,
    HAIRLINE,
    INK,
    INK_SOFT,
    NA_COLOR,
    PANEL_LABEL_FONT,
    REGION_COLORS,
    apply_style,
)

apply_style()
mpl.rcParams["hatch.linewidth"] = 0.4
mpl.rcParams["hatch.color"] = "#9A9A9A"
mpl.rcParams["savefig.bbox"] = "standard"  # keep the 180 mm canvas exactly

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

# Rows of d/e: LE-enriched cell types ordered by consensus count, then the TC-enriched cancer cells
CELLTYPE_ORDER = ["-Fibroblast", "Fibroblast", "Macrophage", "Endothelial", "myocyte",
                  "B cell", "T cell", "Dendritic", "Mast", "Cancer"]
CELLTYPE_DISPLAY: Dict[str, str] = {
    "-Fibroblast": "CAF", "myocyte": "Myocyte", "Mast": "Mast cell",
}

COLOR_TC = REGION_COLORS["TC"]
COLOR_LE = REGION_COLORS["LE"]
REGION_NAME = {"TC": "Tumor core", "LE": "Leading edge"}

# Panel f: ECM-syndecan axis named in the text, highlighted in ink; everything else grey
ECM_SDC1_AXIS = {("COL1A1", "SDC1"), ("COL1A2", "SDC1"), ("FN1", "SDC1")}
F_TOP_N = 12
GREY_DOT = "#B8B8B8"

# Font sizes (pt at final 180 mm width; minimum 5 pt)
FS_TICK = 6.0
FS_LABEL = 6.5
FS_SMALL = 5.5

# Fraction of the 100-um Visium spot pitch used as marker diameter
SPOT_FRAC = 0.78
SPOT_FRAC_CONT = 0.92  # continuous maps: larger spots so the colour field reads over the H&E
# H&E opacity on white
HE_ALPHA = 0.6


# ── Loaders ───────────────────────────────────────────────────────

def _load_he_image(sid: str) -> Optional[np.ndarray]:
    path = GEO_DIR / f"{SAMPLE_GEO[sid]}_{sid}_tissue_hires_image.png"
    return np.array(PILImage.open(path).convert("RGB")) if path.exists() else None


def _load_tissue_positions(sid: str) -> Optional[pd.DataFrame]:
    gsm = SAMPLE_GEO[sid]
    path = GEO_DIR / f"{gsm}_{sid}_tissue_positions_list.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, header=None,
                     names=["barcode", "in_tissue", "row", "col", "pixel_row", "pixel_col"])
    df = df[df["in_tissue"] == 1].copy()
    with open(GEO_DIR / f"{gsm}_{sid}_scalefactors_json.json") as fh:
        scale = json.load(fh).get("tissue_hires_scalef", 1.0)
    df["pixel_row"] *= scale
    df["pixel_col"] *= scale
    return df


def _load_spots(sid: str) -> Optional[pd.DataFrame]:
    """Spot table of the ChatSpatial run: Leiden domain, TC/LE label and FN1 (log-normalized)."""
    path = SPOTS_DIR / f"{sid}.csv"
    return pd.read_csv(path, index_col=0) if path.exists() else None


SpatialData = Tuple[Optional[np.ndarray], Optional[pd.DataFrame], Optional[pd.DataFrame]]
_CACHE: Dict[str, SpatialData] = {}


def _load_spatial_data(sid: str) -> SpatialData:
    if sid not in _CACHE:
        _CACHE[sid] = (_load_he_image(sid), _load_tissue_positions(sid), _load_spots(sid))
    return _CACHE[sid]


def _fallback_text(ax: plt.Axes, msg: str) -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=6,
            transform=ax.transAxes, color="#999999", style="italic")


# ── Spatial helpers ───────────────────────────────────────────────

_SPOT_SCATTERS: List[Tuple[plt.Axes, Any, float]] = []  # (ax, scatter, marker diameter in data units)


def _spot_pitch(pos: pd.DataFrame) -> float:
    """Median nearest-neighbour distance between spots (= 100 um Visium pitch), in image pixels."""
    xy = pos[["pixel_col", "pixel_row"]].to_numpy(float)
    d, _ = cKDTree(xy).query(xy, k=2)
    return float(np.median(d[:, 1]))


def _tissue_extent(sid: str) -> Tuple[float, float, float, float, float]:
    """(x0, x1, y0, y1, pitch) of the analysed spots plus one spot pitch of margin."""
    _, pos, spots = _load_spatial_data(sid)
    if spots is not None:
        pos = pos[pos["barcode"].isin(spots.index)]
    pitch = _spot_pitch(pos)
    pad = 1.0 * pitch
    return (pos["pixel_col"].min() - pad, pos["pixel_col"].max() + pad,
            pos["pixel_row"].min() - pad, pos["pixel_row"].max() + pad, pitch)


def _prepare_he(img: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Whiten the slide background (and everything outside the spot-covered tissue mask);
    show the tissue at HE_ALPHA opacity on white."""
    rgb = img[:, :, :3].astype(np.float64) / 255.0
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    lum = rgb.mean(axis=2)
    bg = ((sat < 0.07) & (lum > 0.50)) | (lum < 0.12)
    rgb[bg] = 1.0
    if mask is not None:
        rgb = mask[..., None] * rgb + (1.0 - mask[..., None])
    rgb = HE_ALPHA * rgb + (1.0 - HE_ALPHA)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def _tissue_mask(pos: pd.DataFrame, c0: int, c1: int, r0: int, r1: int, pitch: float) -> np.ndarray:
    """Soft mask of the area covered by in-tissue spots (hides the fiducial frame and empty slide)."""
    from scipy.ndimage import gaussian_filter
    tree = cKDTree(pos[["pixel_col", "pixel_row"]].to_numpy(float))
    cc, rr = np.meshgrid(np.arange(c0, c1) + 0.5, np.arange(r0, r1) + 0.5)
    d, _ = tree.query(np.column_stack([cc.ravel(), rr.ravel()]), k=1)
    m = (d.reshape(cc.shape) < 0.62 * pitch).astype(float)
    return np.clip(gaussian_filter(m, sigma=0.12 * pitch), 0, 1)


def _draw_he(ax: plt.Axes, sid: str) -> float:
    """Show the H&E cropped to the analysed tissue (no fiducial frame); return the spot pitch."""
    img, pos, _ = _load_spatial_data(sid)
    x0, x1, y0, y1, pitch = _tissue_extent(sid)
    if img is not None:
        c0, c1 = int(max(np.floor(x0), 0)), int(min(np.ceil(x1), img.shape[1]))
        r0, r1 = int(max(np.floor(y0), 0)), int(min(np.ceil(y1), img.shape[0]))
        if sid not in _MASKS:
            _MASKS[sid] = _tissue_mask(pos, c0, c1, r0, r1, pitch)
        ax.imshow(_prepare_he(img[r0:r1, c0:c1], _MASKS[sid]), extent=(c0, c1, r1, r0),
                  interpolation="lanczos", zorder=1)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal")
    ax.axis("off")
    return pitch


_MASKS: Dict[str, np.ndarray] = {}


def _scatter_spots(ax: plt.Axes, x, y, pitch: float, frac: float = SPOT_FRAC, **kw) -> Any:
    sc = ax.scatter(x, y, s=4, linewidths=0, zorder=2, **kw)
    _SPOT_SCATTERS.append((ax, sc, pitch * frac))
    return sc


def _finalize_spot_sizes(fig: plt.Figure) -> None:
    """Marker diameter (a fraction of the spot pitch) converted to points from the final axes scale."""
    fig.canvas.draw()
    for ax, sc, pitch in _SPOT_SCATTERS:
        p0 = ax.transData.transform((0, 0))
        p1 = ax.transData.transform((pitch, 0))
        pitch_pt = abs(p1[0] - p0[0]) * 72.0 / fig.dpi
        sc.set_sizes([pitch_pt ** 2])
        print(f"    spot marker: diameter {pitch_pt:.2f} pt, s = {pitch_pt ** 2:.1f} pt^2")


def _identifier(ax: plt.Axes, text: str) -> None:
    """Short sample identifier under the lower-left corner of a map."""
    ax.text(0.0, -0.035, text, transform=ax.transAxes, fontsize=FS_TICK,
            color=INK_SOFT, ha="left", va="top")


def _map_colorbar(fig: plt.Figure, ax: plt.Axes, sc, label: str, ticks) -> None:
    """Short horizontal colour bar under the lower-right corner of a map; label to its left."""
    pos = ax.get_position()
    W, H = fig.get_size_inches()
    w_in, h_in = 0.50, 0.05
    cax = fig.add_axes([pos.x1 - w_in / W, pos.y0 - 0.075 / H - h_in / H, w_in / W, h_in / H])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal", ticks=ticks)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=FS_SMALL, length=1.5, width=0.4, pad=1)
    if isinstance(label, tuple):  # (italic gene symbol, plain remainder)
        t = cax.text(-0.06, 0.5, label[1], transform=cax.transAxes, fontsize=FS_SMALL,
                     ha="right", va="center", color=INK)
        cax.annotate(label[0], xy=(0, 0.5), xycoords=t, fontsize=FS_SMALL, fontstyle="italic",
                     ha="right", va="center", color=INK)
    else:
        cax.text(-0.06, 0.5, label, transform=cax.transAxes, fontsize=FS_SMALL,
                 ha="right", va="center", color=INK)


# ── Panels a, b: H&E + TC/LE ─────────────────────────────────────

def _draw_tc_le(ax: plt.Axes, sid: str) -> None:
    pitch = _draw_he(ax, sid)
    _, pos, spots = _load_spatial_data(sid)
    if spots is None or "tc_le_region" not in spots.columns:
        _fallback_text(ax, "Annotations pending")
        return
    merged = pos.merge(spots[["tc_le_region"]], left_on="barcode", right_index=True, how="inner")
    # Unlabelled spots are left unpainted; the H&E shows the tissue.
    for reg, col in (("TC", COLOR_TC), ("LE", COLOR_LE)):
        sub = merged[merged["tc_le_region"] == reg]
        _scatter_spots(ax, sub["pixel_col"], sub["pixel_row"], pitch, color=col)


def _region_key(ax: plt.Axes) -> None:
    """Direct key for the TC/LE colours under the lower-right corner of a map."""
    x = 1.0
    for reg in ("LE", "TC"):
        t = ax.text(x, -0.035, REGION_NAME[reg], transform=ax.transAxes, fontsize=FS_TICK,
                    color=REGION_COLORS[reg], ha="right", va="top")
        ax.figure.canvas.draw()
        bb = t.get_window_extent().transformed(ax.transAxes.inverted())
        x = bb.x0 - 0.02
        ax.scatter([x - 0.012], [-0.035 - 0.5 * bb.height], transform=ax.transAxes, s=9,
                   color=REGION_COLORS[reg], linewidths=0, clip_on=False)
        x -= 0.07


# ── Panel c: Fibroblast deconvolution map (s1) ────────────────────

def draw_panel_c(fig: plt.Figure, ax: plt.Axes) -> None:
    pitch = _draw_he(ax, "s1")
    _, pos, _ = _load_spatial_data("s1")
    props = pd.read_csv(RUN_DIR / "proportions" / "s1.csv", index_col=0)
    # Only spots analysed by the run (QC-passing) carry a proportion
    merged = pos.merge(props[["Fibroblast"]], left_on="barcode", right_index=True, how="inner")
    merged = merged.sort_values("Fibroblast")
    vals = merged["Fibroblast"].to_numpy(float)
    sc = _scatter_spots(ax, merged["pixel_col"], merged["pixel_row"], pitch, frac=SPOT_FRAC_CONT, c=vals,
                        cmap=CMAP_SEQUENTIAL, norm=Normalize(0, float(np.nanmax(vals))))
    _identifier(ax, "Sample 1")
    _map_colorbar(fig, ax, sc, "Fibroblast proportion", [0, 0.4, 0.8])


# ── Panel d: per-sample enrichment heatmap ────────────────────────

VLIM = 5.0


def draw_panel_d(fig: plt.Figure, ax: plt.Axes, key_y_in: float) -> None:
    df = pd.read_csv(ANALYSIS_DIR / "per_sample_tc_le_enrichment.csv")
    mat = (df.pivot(index="cell_type", columns="sample", values="log2fc")
           .reindex(index=CELLTYPE_ORDER, columns=SAMPLES).to_numpy(float))
    n_r, n_c = mat.shape
    cmap = CMAP_DIVERGING.copy()
    cmap.set_bad(NA_COLOR)
    mesh = ax.pcolormesh(np.ma.masked_invalid(np.clip(mat, -VLIM, VLIM)), cmap=cmap,
                         vmin=-VLIM, vmax=VLIM, edgecolors="white", linewidth=0.5)

    # Samples without any value (Sample 6: no LE spots) are hatched as not evaluable
    na_cols = [j for j in range(n_c) if np.all(np.isnan(mat[:, j]))]
    for j in na_cols:
        ax.add_patch(mpatches.Rectangle((j + 0.04, 0.03), 0.92, n_r - 0.06, facecolor=NA_COLOR,
                                        edgecolor="#9A9A9A", hatch="//////", linewidth=0, zorder=3))

    # Hairline separating the TC-enriched cancer row from the LE-enriched rows
    ax.plot([-0.25, n_c], [n_r - 1, n_r - 1], color=INK, lw=0.5, clip_on=False, zorder=4)

    ax.set_xlim(0, n_c)
    ax.set_ylim(n_r, 0)
    ax.set_yticks(np.arange(n_r) + 0.5)
    ax.set_yticklabels([CELLTYPE_DISPLAY.get(c, c) for c in CELLTYPE_ORDER], fontsize=FS_TICK)
    ax.set_xticks(np.arange(n_c) + 0.5)
    ax.set_xticklabels([str(i + 1) for i in range(n_c)], fontsize=FS_TICK)
    ax.set_xlabel("OSCC sample", fontsize=FS_LABEL, labelpad=2)
    ax.tick_params(length=0, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)

    # Key band above the heatmap: colour bar (left) and NA chip (right)
    W, H = fig.get_size_inches()
    pos = ax.get_position()
    cb_w, cb_h = 0.62, 0.055
    x_lab = pos.x0
    lab = fig.text(x_lab, 1 - key_y_in / H, "log$_2$FC (LE/TC)", fontsize=FS_LABEL,
                   ha="left", va="center", color=INK)
    fig.canvas.draw()
    lab_w = lab.get_window_extent().width / fig.dpi
    cax = fig.add_axes([pos.x0 + (lab_w + 0.08) / W, 1 - (key_y_in + cb_h / 2) / H,
                        cb_w / W, cb_h / H])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="both",
                      extendfrac=0.10, ticks=[-5, 0, 5])
    cb.outline.set_visible(False)
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.set_xticklabels(["−5", "0", "5"])
    cb.ax.tick_params(labelsize=FS_SMALL, length=1.5, width=0.4, pad=1)
    if na_cols:
        chip_w, chip_h = 0.075, 0.075
        cx = pos.x1 - (chip_w + 0.55) / W
        fig.add_artist(mpatches.Rectangle(
            (cx, 1 - (key_y_in + chip_h / 2) / H), chip_w / W, chip_h / H, transform=fig.transFigure,
            facecolor=NA_COLOR, edgecolor="#9A9A9A", hatch="//////", linewidth=0.4))
        fig.text(cx + (chip_w + 0.04) / W, 1 - key_y_in / H, "No LE spots", fontsize=FS_SMALL,
                 ha="left", va="center", color=INK)


# ── Panel e: consensus bars ───────────────────────────────────────

def draw_panel_e(fig: plt.Figure, ax: plt.Axes, key_y_in: float) -> None:
    consensus = pd.read_csv(ANALYSIS_DIR / "consensus_tc_le_patterns.csv", index_col=0).reindex(CELLTYPE_ORDER)
    le = consensus["le_enriched_count"].to_numpy(float)
    tc = consensus["tc_enriched_count"].to_numpy(float)
    n_total = 12
    y = np.arange(len(CELLTYPE_ORDER)) + 0.5
    bar_h = 0.62

    ax.barh(y, le, height=bar_h, color=COLOR_LE, edgecolor="none", zorder=3)
    ax.barh(y, -tc, height=bar_h, color=COLOR_TC, edgecolor="none", zorder=3)

    # Clopper-Pearson 95% CI
    ci = pd.read_csv(ANALYSIS_DIR / "consensus_binomial_ci.csv")
    for yi, ct in zip(y, CELLTYPE_ORDER):
        for direction, sign in (("LE", 1), ("TC", -1)):
            row = ci[(ci["cell_type"] == ct) & (ci["direction"] == direction)]
            if len(row) == 1 and row["k"].iloc[0] > 0:
                lo, hi = row["ci_lower"].iloc[0] * n_total, row["ci_upper"].iloc[0] * n_total
                ax.plot([sign * lo, sign * hi], [yi, yi], color=INK, lw=0.6, zorder=4,
                        solid_capstyle="butt")

    # 75% threshold (>= 9 of 12 samples) in both directions
    n_r = len(CELLTYPE_ORDER)
    for xv in (-9, 9):
        ax.plot([xv, xv], [0, n_r], color=INK_SOFT, lw=0.5, ls=(0, (2, 1.6)), zorder=2)
    ax.text(9, -0.12, "75%", fontsize=FS_SMALL, ha="center", va="bottom", color=INK_SOFT)
    ax.axvline(0, color=INK, lw=0.5, zorder=5)
    ax.plot([-12, 12], [n_r - 1, n_r - 1], color=INK, lw=0.5, zorder=4)

    ax.set_ylim(n_r, 0)
    ax.set_xlim(-12, 12)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_xticklabels(["10", "5", "0", "5", "10"])
    ax.tick_params(axis="x", labelsize=FS_TICK, pad=1.5)
    ax.set_xlabel("Samples enriched\n(← TC | LE →)", fontsize=FS_LABEL, labelpad=2,
                  linespacing=1.1)

    # Direct TC / LE labels in the key band above each half
    H = fig.get_size_inches()[1]
    tr = mpl.transforms.blended_transform_factory(ax.transData, fig.transFigure)
    ax.text(-6, 1 - key_y_in / H, "TC", transform=tr, fontsize=FS_TICK, color=COLOR_TC,
            ha="center", va="center", fontweight="bold")
    ax.text(6, 1 - key_y_in / H, "LE", transform=tr, fontsize=FS_TICK, color=COLOR_LE,
            ha="center", va="center", fontweight="bold")


# ── Panel f: CellChat Fibroblast -> Cancer ────────────────────────

def _gene_pair(lig: str, rec: str) -> str:
    """Label 'LIG -> REC' (complex subunits joined by '+'); drawn in italic."""
    return f"{lig.replace('_', '+')} \u2192 {rec.replace('_', '+')}"


def draw_panel_f(ax: plt.Axes, key_y_in: float) -> None:
    df = pd.read_csv(RUN_DIR / "cellchat_tc_le_caf_cancer_interactions.csv")
    df = df[(df["source"] == "Fibroblast") & (df["target"] == "Cancer")]
    top = df.nlargest(F_TOP_N, "prob").reset_index(drop=True)
    y = np.arange(len(top))
    on_axis = np.array([(l, r) in ECM_SDC1_AXIS for l, r in zip(top["ligand"], top["receptor"])])
    probs = top["prob"].to_numpy(float)

    for yi, p, a in zip(y, probs, on_axis):
        ax.plot([0.2, p], [yi, yi], color=INK if a else "#D6D6D6", lw=0.6, zorder=2,
                solid_capstyle="butt")
    ax.scatter(probs[~on_axis], y[~on_axis], s=14, color=GREY_DOT, linewidths=0, zorder=3)
    ax.scatter(probs[on_axis], y[on_axis], s=16, color=INK, linewidths=0, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([_gene_pair(l, r) for l, r in zip(top["ligand"], top["receptor"])],
                       fontsize=FS_TICK, fontstyle="italic")
    for lab, a in zip(ax.get_yticklabels(), on_axis):
        lab.set_color(INK if a else INK_SOFT)
    ax.set_ylim(len(top) - 0.4, -0.6)
    ax.set_xlim(0.2, 0.35)
    ax.set_xticks([0.2, 0.25, 0.3, 0.35])
    ax.set_xticklabels(["0.20", "0.25", "0.30", "0.35"])
    ax.set_xlabel("Communication probability", fontsize=FS_LABEL, labelpad=2)
    ax.tick_params(axis="x", labelsize=FS_TICK, pad=1.5)
    ax.tick_params(axis="y", length=0, pad=2)

    # Bracket naming the highlighted axis
    idx = np.where(on_axis)[0]
    if len(idx):
        y0, y1 = idx.min() - 0.3, idx.max() + 0.3
        xb = 0.343
        ax.plot([xb, xb + 0.002, xb + 0.002, xb], [y0, y0, y1, y1], color=INK, lw=0.5,
                clip_on=False, solid_capstyle="butt")
        ax.text(xb + 0.004, 0.5 * (y0 + y1), "ECM–syndecan\naxis", fontsize=FS_SMALL,
                color=INK, ha="left", va="center", linespacing=1.1, clip_on=False)
    # Direction identifier in the row's key band
    H = ax.figure.get_size_inches()[1]
    tr = mpl.transforms.blended_transform_factory(ax.transAxes, ax.figure.transFigure)
    ax.text(0.0, 1 - key_y_in / H, "Fibroblast \u2192 Cancer", transform=tr, fontsize=FS_TICK,
            color=INK, ha="left", va="center")


# ── Panel g: FN1 spatial expression (s1) ──────────────────────────

def draw_panel_g(fig: plt.Figure, ax: plt.Axes) -> None:
    pitch = _draw_he(ax, "s1")
    _, pos, spots = _load_spatial_data("s1")
    expr = spots[["FN1_lognorm"]]
    merged = pos.merge(expr, left_on="barcode", right_index=True, how="inner").sort_values("FN1_lognorm")
    vals = merged["FN1_lognorm"].to_numpy(float)
    sc = _scatter_spots(ax, merged["pixel_col"], merged["pixel_row"], pitch, frac=SPOT_FRAC_CONT, c=vals,
                        cmap=CMAP_SEQUENTIAL, norm=Normalize(0, float(np.nanmax(vals))))
    _identifier(ax, "Sample 1")
    _map_colorbar(fig, ax, sc, ("FN1", " (log-normalized)"), [0, 2, 4])


# ── Panel h: Moran's I grouped by programme ───────────────────────

PROGRAMMES = [
    ("TC", "Tumor core", "Keratinization /\ndifferentiation"),
    ("LE", "Leading edge", "Partial EMT /\ninvasion front"),
]


def draw_panel_h(ax: plt.Axes) -> None:
    df = pd.read_csv(ANALYSIS_DIR / "svg_target_genes_summary.csv")[["gene", "category", "moranI"]]
    df = df.dropna(subset=["moranI"])
    df["category"] = df["category"].astype(str).str.upper()

    rows, groups = [], []
    for key, name, desc in PROGRAMMES:
        sub = df[df["category"] == key].sort_values("moranI", ascending=False)
        start = len(rows)
        rows.extend((g, v, REGION_COLORS[key]) for g, v in zip(sub["gene"], sub["moranI"]))
        groups.append((start, len(rows) - 1, name, desc, REGION_COLORS[key]))

    gap = 0.8
    y_pos, yv = [], 0.0
    for s, e, *_ in groups:
        for _ in range(s, e + 1):
            y_pos.append(yv)
            yv += 1.0
        yv += gap
    y_pos = np.array(y_pos)

    for yp, (g, val, col) in zip(y_pos, rows):
        ax.plot([0, val], [yp, yp], color=col, lw=0.9, zorder=2, solid_capstyle="butt")
        ax.scatter([val], [yp], s=14, color=col, linewidths=0, zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=FS_TICK, fontstyle="italic")
    ax.set_ylim(y_pos.max() + 0.6, y_pos.min() - 0.6)
    ax.set_xlim(0, 0.6)
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.set_xlabel("Moran's $I$ (Sample 1)", fontsize=FS_LABEL, labelpad=2)
    ax.tick_params(axis="x", labelsize=FS_TICK, pad=1.5)
    ax.tick_params(axis="y", length=0, pad=2)

    trans = ax.get_yaxis_transform()
    for s, e, name, desc, col in groups:
        y0, y1 = y_pos[s] - 0.3, y_pos[e] + 0.3
        xb = 1.03
        ax.plot([xb, xb + 0.015, xb + 0.015, xb], [y0, y0, y1, y1], transform=trans,
                color=col, lw=0.6, clip_on=False, solid_capstyle="butt")
        ym = 0.5 * (y_pos[s] + y_pos[e])
        ax.text(xb + 0.05, ym - 0.15, name, transform=trans, fontsize=FS_LABEL,
                fontweight="bold", color=col, ha="left", va="bottom")
        ax.text(xb + 0.05, ym + 0.05, desc, transform=trans, fontsize=FS_SMALL,
                color=INK_SOFT, ha="left", va="top", linespacing=1.15)


# ── Panel letters ─────────────────────────────────────────────────

def _row_letters(fig: plt.Figure, items, top_in: float, dx_in: float = 0.04) -> None:
    """8-pt bold letters on one top line per row, at the left edge of each panel's decorations."""
    renderer = fig.canvas.get_renderer()
    W, H = fig.get_size_inches()
    for ax, letter in items:
        x_in = ax.get_tightbbox(renderer).x0 / fig.dpi
        fig.text(max(x_in - dx_in, 0.0) / W, 1 - top_in / H, letter,
                 **PANEL_LABEL_FONT, ha="left", va="top")


# ── Main ──────────────────────────────────────────────────────────

def main() -> None:
    print("Building Figure 2 (manuscript Fig. 5): OSCC case study")
    W = DOUBLE_COL  # 180 mm

    # Map widths follow the tissue aspect ratio at a common map height
    asp = {}
    for sid in ("s1", "s9"):
        x0, x1, y0, y1, _ = _tissue_extent(sid)
        asp[sid] = (x1 - x0) / (y1 - y0)
    gap = 0.22
    left, right = 0.12, 0.04
    hmap = (W - left - right - 2 * gap) / (2 * asp["s1"] + asp["s9"])
    wa, wb = asp["s1"] * hmap, asp["s9"] * hmap

    r1 = 0.17                     # top of row-1 maps
    r2 = r1 + hmap + 0.70         # top of row-2 plot area (key band above it)
    h2 = 1.55
    r3 = r2 + h2 + 0.62           # top of row-3 axes
    h3 = 1.50 if hmap > 1.5 else hmap
    g_h = hmap * 0.98
    H = r3 + max(g_h, h3) + 0.30
    fig = plt.figure(figsize=(W, H))

    def axin(x, y_top, w, h):
        return fig.add_axes([x / W, 1 - (y_top + h) / H, w / W, h / H])

    ax_a = axin(left, r1, wa, hmap)
    ax_b = axin(left + wa + gap, r1, wb, hmap)
    ax_c = axin(left + wa + wb + 2 * gap, r1, wa, hmap)

    ax_d = axin(0.62, r2, 2.55, h2)
    ax_e = axin(3.27, r2, 1.02, h2)
    ax_f = axin(5.30, r2, 1.20, h2)

    ax_g = axin(left, r3, asp["s1"] * g_h, g_h)
    h_top = r3 + 0.08
    ax_h = axin(3.30, h_top, 2.45, g_h - 0.40)

    print("Drawing panels...")
    _draw_tc_le(ax_a, "s1")
    _identifier(ax_a, "Sample 1")
    _draw_tc_le(ax_b, "s9")
    _identifier(ax_b, "Sample 9")
    draw_panel_c(fig, ax_c)
    key2 = r2 - 0.17
    draw_panel_d(fig, ax_d, key2)
    draw_panel_e(fig, ax_e, key2)
    draw_panel_f(ax_f, key2)
    draw_panel_g(fig, ax_g)
    draw_panel_h(ax_h)

    _finalize_spot_sizes(fig)
    _region_key(ax_a)

    _row_letters(fig, [(ax_a, "a"), (ax_b, "b"), (ax_c, "c")], top_in=0.02, dx_in=0.10)
    _row_letters(fig, [(ax_d, "d"), (ax_e, "e"), (ax_f, "f")], top_in=r2 - 0.40, dx_in=0.06)
    _row_letters(fig, [(ax_g, "g"), (ax_h, "h")], top_in=r3 - 0.22, dx_in=0.10)

    for fmt in ("pdf", "png"):
        out = FIGURE_DIR / f"figure2.{fmt}"
        fig.savefig(str(out), dpi=300, facecolor="white", bbox_inches=None)
        print(f"  Saved: {out}")
    plt.close(fig)
    print(f"Done! ({W * 25.4:.0f} x {H * 25.4:.0f} mm)")


if __name__ == "__main__":
    main()
