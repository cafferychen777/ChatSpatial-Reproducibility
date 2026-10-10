"""
Shared figure style for ChatSpatial paper.
Nature Communications format: 183 mm (7.2 in) double-column.

Palette and conventions unified with visium_hd_challenge:
- Extended Okabe-Ito palette (12 colors: 8 base + wine, teal, grey, sand)
- sans-serif (Arial) 7pt base, 10pt bold panel labels
- 300 DPI, PDF Type 42 font embedding
- Outward ticks, no top/right spines
"""

from __future__ import annotations

from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Page dimensions ───────────────────────────────────────────────
MM_PER_INCH = 25.4
FULL_WIDTH = 183 / MM_PER_INCH          # 7.2 in
SINGLE_COL = 88 / MM_PER_INCH           # 3.46 in
NC_COLUMN_WIDTH_IN = FULL_WIDTH          # backward-compatible alias

# ── Nature Communications rcParams ────────────────────────────────
NC_RCPARAMS: Dict[str, object] = {
    # Font
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
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
PANEL_LABEL_FONT = {
    "fontsize": 10,
    "fontweight": "bold",
    "fontfamily": "sans-serif",
}


def apply_style() -> None:
    """Apply Nature Communications figure style globally."""
    mpl.rcParams.update(NC_RCPARAMS)


def add_panel_label(
    ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.08
) -> None:
    """Add bold panel label (10pt, positioned in axes coordinates)."""
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        **PANEL_LABEL_FONT,
        va="top",
        ha="left",
    )
