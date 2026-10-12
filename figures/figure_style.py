"""
Shared figure style for ChatSpatial paper.
Nature Communications format: 183 mm (7.2 in) double-column.

Palette and conventions unified with visium_hd_challenge:
- Extended Okabe-Ito palette (12 colors: 8 base + wine, teal, grey, sand)
- sans-serif (Arial) 7pt base, 8pt bold lowercase panel labels; 5 pt minimum
- 300 DPI, PDF Type 42 font embedding
- Outward ticks, no top/right spines
"""

from __future__ import annotations

from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── Page dimensions ───────────────────────────────────────────────
MM_PER_INCH = 25.4
FULL_WIDTH = 183 / MM_PER_INCH          # 7.2 in
SINGLE_COL = 88 / MM_PER_INCH           # 3.46 in
TEXT_WIDTH_180 = 180 / MM_PER_INCH      # 7.09 in; author at this width to place at \\textwidth
MIN_FONT_PT = 5.0                       # minimum font size at final print width
NC_COLUMN_WIDTH_IN = FULL_WIDTH          # backward-compatible alias

# ── Nature Communications rcParams ────────────────────────────────
NC_RCPARAMS: Dict[str, object] = {
    # Font
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    # Math text in Arial (no DejaVu fallback for $r$, $I$, etc.)
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.sf": "Arial",
    "mathtext.cal": "Arial:italic",
    "mathtext.tt": "Arial",
    "mathtext.default": "it",
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,

    # Lines and markers
    "axes.linewidth": 0.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.5,

    # Layout
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,

    # PDF font embedding
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    # Remove top/right spines
    "axes.spines.top": False,
    "axes.spines.right": False,

    # Ticks
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",

    # Background
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}

# ── Extended Okabe-Ito colorblind-safe palette ────────────────────
# Base 8 Okabe-Ito + 4 extensions from visium_hd_challenge
OI: Dict[str, str] = {
    # Base Okabe-Ito
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "redpurple": "#CC79A7",
    "black": "#000000",
    # Extended (Paul Tol / visium_hd_challenge)
    "wine": "#882255",
    "teal": "#44AA99",
    "grey": "#999999",
    "sand": "#DDCC77",
    "navy": "#003F5C",
}

# ── Shared entity palettes (one colour per entity across all figures) ──
# Systems (Okabe-Ito). Import these rather than hard-coding hex values.
SYSTEM_COLORS: Dict[str, str] = {
    "ChatSpatial agent": "#0072B2",        # blue
    "ChatSpatial single call": "#56B4E9",  # sky blue
    "STAgent": "#D55E00",                  # vermillion
    "SpatialAgent": "#009E73",             # bluish green
}
SYSTEM_COLORS["ChatSpatial"] = SYSTEM_COLORS["ChatSpatial agent"]

# Schema levels: neutral grey series (never a system colour)
SCHEMA_COLORS: Dict[str, str] = {
    "Full": "#333333",
    "Bare": "#8C8C8C",
    "No schema": "#D0D0D0",
}
SCHEMA_COLORS["None"] = SCHEMA_COLORS["No schema"]
SCHEMA_COLORS["Minimal"] = SCHEMA_COLORS["No schema"]

# LLM models: ordered palette that does NOT reuse the system colours
MODEL_PALETTE: List[str] = [
    "#E69F00",  # orange
    "#CC79A7",  # reddish purple
    "#000000",  # black
    "#666666",  # dark grey
    "#F0E442",  # yellow
    "#882255",  # wine
]

# ── Ordered color sequences for categorical data ─────────────────
# Primary 8 — maximally distinct, use first
PALETTE_8: List[str] = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#E69F00",  # orange
    "#CC79A7",  # red-purple
    "#56B4E9",  # sky blue
    "#882255",  # wine
    "#44AA99",  # teal
]

# Blue gradient (light → dark) for ordinal grouping
BLUE_GRADIENT: List[str] = ["#56B4E9", "#0072B2", "#003F5C"]

# Warm-to-cool sequence for 4-level ordinal
WARM_TO_COOL: List[str] = ["#D55E00", "#E69F00", "#56B4E9", "#0072B2"]

# Generic 3-color: grey baseline, vermillion, blue
GREY_VERM_BLUE: List[str] = ["#999999", "#D55E00", "#0072B2"]

# ── CNV subclone colors (consistent across all figures) ───────────
SUBCLONE_COLORS: List[str] = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#E69F00",  # orange
    "#CC79A7",  # red-purple
    "#56B4E9",  # sky blue
    "#882255",  # wine
    "#44AA99",  # teal
    "#DDCC77",  # sand
    "#332288",  # indigo
]

# ── HGSOC cell type colors ────────────────────────────────────────
HGSOC_CELLTYPE_COLORS: Dict[str, str] = {
    "Tumour_cells": "#D55E00",     # vermillion
    "Macrophages": "#0072B2",      # blue
    "Fibroblasts": "#009E73",      # green
    "T_cells": "#CC79A7",          # red-purple
    "B_plasma_cells": "#E69F00",   # orange
    "Endothelial_cells": "#56B4E9",  # sky blue
    "Mesothelial_cells": "#999999",  # grey
    "Myofibroblasts": "#882255",   # wine
}

# ── Panel label styling ───────────────────────────────────────────
# Nature style: 8 pt bold lowercase Arial, black
PANEL_LABEL_FONT = {
    "fontsize": 8,
    "fontweight": "bold",
    "fontfamily": "sans-serif",
    "color": "black",
}


def apply_style() -> None:
    """Apply Nature Communications figure style globally."""
    mpl.rcParams.update(NC_RCPARAMS)
    if "mathtext.bfit" in mpl.rcParams:  # matplotlib >= 3.8
        mpl.rcParams["mathtext.bfit"] = "Arial:italic:bold"


def add_panel_label(
    ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.08
) -> None:
    """Add bold panel label (8 pt, lowercase, axes coordinates)."""
    ax.text(
        x, y, label.lower(),
        transform=ax.transAxes,
        **PANEL_LABEL_FONT,
        va="top",
        ha="left",
    )


def add_row_panel_labels(
    fig: plt.Figure,
    axes_labels: "List[tuple]",
    top: "float | None" = None,
    dx_in: float = 0.30,
    dy_in: float = 0.06,
) -> None:
    """Place panel letters for one row of axes on a common top line.

    ``axes_labels`` is a list of ``(ax, letter)``. Each letter is placed
    ``dx_in`` inches left of its axes' left edge; all letters share one
    baseline: ``top`` (figure fraction) if given, else ``dy_in`` inches
    above the highest axes top in the row. Call after layout is final.
    """
    w, h = fig.get_size_inches()
    if top is None:
        top = max(ax.get_position().y1 for ax, _ in axes_labels) + dy_in / h
    for ax, letter in axes_labels:
        x = ax.get_position().x0 - dx_in / w
        fig.text(x, top, letter.lower(), **PANEL_LABEL_FONT,
                 va="bottom", ha="left")


# ══════════════════════════════════════════════════════════════════
# Paper-wide visual system: Direction A, "Ink & Cobalt" (chosen 2026-10-12).
# One accent (cobalt) marks ChatSpatial and everything that belongs to it; one
# counter-accent (rust) marks the contrast condition. Everything else is grey.
# Thin 0.5 pt open L-axes, no gridlines, direct labels instead of legend boxes.
# Every figure takes its colours from the dictionaries below.
# ══════════════════════════════════════════════════════════════════
NAME = "Direction A — Ink & Cobalt"
TAGLINE = ("Restrained Nature style: neutral greys, one cobalt accent for ChatSpatial, "
           "one rust counter-accent; thin open axes, no grid, direct labels.")

# ── page ──
MM = 1 / 25.4
DOUBLE_COL = 180 * MM
SINGLE_COL = 88 * MM

# ── core tones ──
INK = "#1A1A1A"          # all text
INK_SOFT = "#5C5C5C"     # tick labels, secondary annotation
AXIS = "#333333"         # spines and ticks
GRID = None              # no gridlines
HAIRLINE = "#C8C8C8"     # reference lines, separators
NEUTRAL_LIGHT = "#E4E4E4"
BG = "#FFFFFF"

COBALT = "#1F5A96"
COBALT_MID = "#6E97C4"
COBALT_LIGHT = "#C9DAEC"
RUST = "#B4513A"

# ── systems (Fig. 2) ──
SYSTEM_COLORS = {
    "ChatSpatial agent": COBALT,
    "ChatSpatial single call": COBALT,   # same hue, drawn hollow
    "STAgent": RUST,
    "SpatialAgent": "#6B6B6B",
}
SYSTEM_FILL = {  # face colour of markers / bars
    "ChatSpatial agent": COBALT,
    "ChatSpatial single call": "#FFFFFF",
    "STAgent": RUST,
    "SpatialAgent": "#6B6B6B",
}
SYSTEM_MARKERS = {"ChatSpatial agent": "o", "ChatSpatial single call": "o",
                  "STAgent": "s", "SpatialAgent": "^"}

# ── schema levels: single-hue cobalt ramp (full schema = ChatSpatial's own setting) ──
SCHEMA_COLORS = {"Full": COBALT, "Bare": COBALT_MID, "None": COBALT_LIGHT}

# ── LLMs in the ablation ──
MODEL_COLORS = {"Claude Haiku 4.5": "#1A1A1A", "Gemini 2.5 Flash": "#7A7A7A", "GPT-5.4": "#B5B5B5"}
MODEL_MARKERS = {"Claude Haiku 4.5": "s", "Gemini 2.5 Flash": "o", "GPT-5.4": "D"}

# ── backend access (Fig. 3) ──
ACCESS_COLORS = {"Commercial": "#2B2B2B", "Open-weight": RUST}

# ── execution outcomes (Fig. 3) ──
OUTCOME_COLORS = {"Completed": COBALT, "Explicit error": "#BDBDBD",
                  "Not executed": "#E8E8E8", "Silent failure": RUST}

# ── parameter strategies (Fig. 4) ──
STRATEGY_COLORS = {"Publication settings": "#9E9E9E", "Schema defaults": COBALT,
                   "Label-free rule": RUST, "Best configuration": "#1A1A1A"}

# ── spatial-domain methods: spatial methods tinted, graph clustering in greys ──
METHOD_COLORS = {"SpaGCN": "#2E4A6B", "STAGATE": "#5E9AAE", "GraphST": "#C2913A",
                 "BANKSY": RUST, "Leiden": "#C4C4C4", "Louvain": "#5A5A5A"}

# ── tissue regions (Fig. 5) ──
REGION_COLORS = {"TC": "#2E4A6B", "LE": "#C0583C", "Other": "#E6E6E6"}

# ── cell types (Figs 5-6), muted earth set ──
CELLTYPE_COLORS = {
    "Tumour": "#2E4A6B", "Macrophage": "#C0583C", "Fibroblast": "#D9A84E",
    "T cell": "#5E9AAE", "B/plasma": "#8B5A8C", "Endothelial": "#A9C4DE",
    "Mesothelial": "#E2DED6", "Myofibroblast": "#4D3B26",
    # OSCC-only extras
    "Myocyte": "#9E9E9E", "Mast": "#D8B4C8", "Dendritic": "#5A5A5A",
}

# ── continuous maps ──
CMAP_DIVERGING = LinearSegmentedColormap.from_list(
    "A_div", ["#14406E", COBALT, "#8FB0D3", "#F4F4F4", "#E0A898", RUST, "#7E2F1E"])
CMAP_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "A_seq", ["#EEF1F5", "#C9DAEC", "#6E97C4", COBALT, "#0F2E4F"])
NA_COLOR = "#FFFFFF"

# ── schematic (Fig. 1) ──
SCHEMATIC = {
    "box_fill": "#FFFFFF", "box_edge": "#4D4D4D", "box_lw": 0.6,
    "accent_fill": "#EAF1F8", "accent_edge": COBALT, "accent_text": COBALT,
    "arrow": "#4D4D4D", "label": INK_SOFT, "radius": 1.2,  # mm
    "text_weight": "normal",
}

# ── layout principles consumed by the prototype renderer ──
LAYOUT = {
    "grid": False, "left_spine": True, "panel_titles": False, "title_rule": False,
    "marker_size": 3.2, "line_width": 0.8, "ci_lw": 0.7, "capsize": 0,
    "bar_edge": "none", "bar_width": 0.72, "point_edge": "#FFFFFF", "point_edge_w": 0.3,
    "legend": "direct",
}

A_RCPARAMS = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial", "mathtext.default": "it",
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": AXIS,
    "xtick.color": AXIS, "ytick.color": AXIS,
    "xtick.labelcolor": INK_SOFT, "ytick.labelcolor": INK_SOFT,
    "axes.linewidth": 0.5, "lines.linewidth": 0.8, "lines.markersize": 3.2,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "legend.frameon": False,
    "figure.facecolor": BG, "axes.facecolor": BG,
    "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
}


def apply_style() -> None:  # noqa: F811  (Direction A overrides the earlier definition)
    """Apply the paper-wide style (Nature Communications sizes, Direction A colours)."""
    mpl.rcParams.update(NC_RCPARAMS)
    mpl.rcParams.update(A_RCPARAMS)
    if "mathtext.bfit" in mpl.rcParams:  # matplotlib >= 3.8
        mpl.rcParams["mathtext.bfit"] = "Arial:italic:bold"
