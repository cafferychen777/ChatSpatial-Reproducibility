"""
Figure 4: schema-derived parameter spaces make robustness analysis and
label-free parameter selection systematic (experiment A).

Inputs are small tables exported from the final experiment-A
stages (dlpfc155, noise155, platforms155) by experiments/robustness/scripts/export_figA.py:
    experiments/robustness/results/strategy_scores_all.csv   ARI of each strategy per sample and method
    experiments/robustness/results/sensitivity_dlpfc.csv     min/median/max ARI over schema-valid configurations
    experiments/robustness/results/noise_retention.csv       paired ARI retention under data degradation
    experiments/robustness/results/platform_method_means.csv mean ARI per platform, strategy and method

Paper-wide style: Direction A ("Ink & Cobalt", figure_style.py). Methods take
METHOD_COLORS (spatial methods tinted, graph clustering in greys), strategies
STRATEGY_COLORS (schema defaults cobalt, label-free rule rust), and the
platform heatmap CMAP_SEQUENTIAL. Series are labelled directly on the data.

Output: figures/figureA.pdf/.png and source data in source_data/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd

from figure_style import (
    CMAP_SEQUENTIAL,
    DOUBLE_COL,
    HAIRLINE,
    INK,
    INK_SOFT,
    METHOD_COLORS,
    PANEL_LABEL_FONT,
    STRATEGY_COLORS,
    apply_style,
)

ROOT = Path(__file__).resolve().parents[1]
EXPA = ROOT / "experiments" / "robustness" / "results"
FIG_DIR = ROOT / "figures"
SOURCE_DIR = ROOT / "source_data"

# Source-data order (unchanged) and plotting order (spatial methods, then graph clustering).
METHODS = [
    ("spagcn", "SpaGCN"),
    ("leiden", "Leiden"),
    ("louvain", "Louvain"),
    ("stagate", "STAGATE"),
    ("graphst", "GraphST"),
    ("banksy", "BANKSY"),
]
PLOT_METHODS = [
    ("spagcn", "SpaGCN"),
    ("stagate", "STAGATE"),
    ("graphst", "GraphST"),
    ("banksy", "BANKSY"),
    ("leiden", "Leiden"),
    ("louvain", "Louvain"),
]
METHOD_GROUPS = [(0, 3, "Spatial methods"), (4, 5, "Graph clustering")]
# key, display name, marker, filled, marker size
STRATEGIES = [
    ("expert", "Publication settings", "s", True, 3.6),
    ("default", "Schema defaults", "o", True, 4.0),
    ("rule", "Label-free rule", "D", True, 3.5),
    ("oracle", "Best configuration", "^", False, 4.3),
]
# condition, tick label, group
NOISE = [
    ("umi0.5", "50%", "thin"),
    ("umi0.25", "25%", "thin"),
    ("umi0.1", "10%", "thin"),
    ("drop0.1", "−10%", "drop"),
    ("drop0.3", "−30%", "drop"),
]
NOISE_GROUPS = [("thin", "Count thinning\n(counts kept)"), ("drop", "Spot removal")]
PLATFORMS = ["Visium", "STARmap", "MERFISH"]
SAMPLE_NAME = {"starmap": "STARmap", "abc_merfish": "MERFISH"}

# Figure geometry (inches); authored at 180 mm so it places at \textwidth.
W = DOUBLE_COL
H = 4.52


def add_axes(fig: plt.Figure, x0: float, y0: float, w: float, h: float) -> plt.Axes:
    """Axes placed in inches from the lower-left corner of the figure."""
    return fig.add_axes([x0 / W, y0 / H, w / W, h / H])


def legible(color: str) -> str:
    """Entity colour for text unless it is too light to read on white."""
    r, g, b = to_rgb(color)
    return INK_SOFT if (0.2126 * r + 0.7152 * g + 0.0722 * b) > 0.68 else color


def _luminance(rgb) -> float:
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def text_colour_for(rgb) -> str:
    """Ink or white, whichever has the higher WCAG contrast with the cell colour."""
    lum = _luminance(rgb)
    return "white" if (1.05 / (lum + 0.05)) > ((lum + 0.05) / 0.05) else INK


def strategy_marker(key: str) -> dict:
    name, marker, filled, ms = next((n, m, f, s) for k, n, m, f, s in STRATEGIES if k == key)
    color = STRATEGY_COLORS[name]
    if filled:
        return dict(marker=marker, ms=ms, mfc=color, mec="white", mew=0.35)
    return dict(marker=marker, ms=ms, mfc="none", mec=color, mew=0.7)


def group_brackets(ax, fig, spans, y_pt: float, text_pt: float, inset: float = 0.32) -> None:
    """Thin rule under each group of x ticks with the group name below it."""
    rule = mtransforms.offset_copy(ax.get_xaxis_transform(), fig=fig, y=y_pt, units="points")
    for x0, x1, name in spans:
        ax.plot([x0 - inset, x1 + inset], [0, 0], color=INK_SOFT, lw=0.5, transform=rule, clip_on=False,
                solid_capstyle="butt")
        ax.annotate(name, xy=((x0 + x1) / 2, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, text_pt), textcoords="offset points", ha="center", va="top",
                    fontsize=6, linespacing=1.0)


def main() -> None:
    apply_style()
    # Fixed 180 mm canvas: no tight-bbox cropping, so fonts print at their nominal size.
    mpl.rcParams["savefig.bbox"] = "standard"
    SOURCE_DIR.mkdir(exist_ok=True)

    scores = pd.read_csv(EXPA / "strategy_scores_all.csv", dtype={"section": str})
    scores = scores[scores.dataset != "dlpfc_array"]
    sens = pd.read_csv(EXPA / "sensitivity_dlpfc.csv", dtype={"section": str})
    sens = sens[sens.dataset == "dlpfc"].assign(range=lambda d: d["max"] - d["min"])
    noise = pd.read_csv(EXPA / "noise_retention.csv")
    plat = pd.read_csv(EXPA / "platform_method_means.csv")

    fig = plt.figure(figsize=(W, H))
    top_y0, top_h = 2.70, 1.40
    bot_y0, bot_h = 0.50, 1.36
    left_x0, left_w = 0.56, 2.45
    right_x0 = 3.70
    letters = []

    # ── a: ARI range across schema-valid configurations, per method and DLPFC section ──
    ax_a = ax = add_axes(fig, left_x0, top_y0, left_w, top_h)
    rows_a = []
    for key, name in METHODS:   # source data in the original order
        for _, r in sens[sens.method == key].iterrows():
            rows_a.append({"method": name, "section": r.section, "ari_min": r["min"],
                           "ari_median": r["median"], "ari_max": r["max"],
                           "ari_range": r["range"], "n_configurations": int(r["count"])})
    rng = np.random.default_rng(0)
    for i, (key, name) in enumerate(PLOT_METHODS):
        v = sens[sens.method == key]["range"].to_numpy()
        x = i + rng.uniform(-0.17, 0.17, len(v))
        ax.scatter(x, v, s=10, color=METHOD_COLORS[name], edgecolor="white", lw=0.3, zorder=3)
        med = float(np.median(v))
        ax.plot([i - 0.25, i + 0.25], [med, med], color=INK, lw=1.0, zorder=4, solid_capstyle="butt")
        ax.text(i + 0.29, med, f"{med:.2f}", ha="left", va="center", fontsize=5.5, color=INK_SOFT)
    ax.set_xticks(range(len(PLOT_METHODS)))
    ax.set_xticklabels([n for _, n in PLOT_METHODS], fontsize=6)
    for t, (_, n) in zip(ax.get_xticklabels(), PLOT_METHODS):
        t.set_color(legible(METHOD_COLORS[n]))
    ax.tick_params(axis="x", length=0, pad=2.5)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(-0.5, len(PLOT_METHODS) - 0.35)
    group_brackets(ax, fig, METHOD_GROUPS, y_pt=-12.5, text_pt=-14.0)
    ax.set_ylabel("ARI range across\nschema-valid settings")
    ax.set_ylim(0, 0.8)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    letters.append((ax, "a", 1))
    pd.DataFrame(rows_a).to_csv(SOURCE_DIR / "Fig4a_sensitivity.csv", index=False)

    # ── b: mean ARI over methods per sample for each strategy ──
    ax_b = ax = add_axes(fig, right_x0, top_y0, W - right_x0 - 0.06, top_h)
    piv = scores.pivot_table(index=["dataset", "section", "method"], columns="strategy", values="ari")
    by = piv.groupby(["dataset", "section"]).mean().reset_index()
    by["label"] = [SAMPLE_NAME.get(d, s) for d, s in zip(by.dataset, by.section)]
    order = [s for s in by[by.dataset == "dlpfc"].sort_values("section").label] + ["STARmap", "MERFISH"]
    by = by.set_index("label").loc[order].reset_index()
    xb = np.r_[np.arange(12), 12.7 + np.arange(len(by) - 12)]
    dodge = np.linspace(-0.27, 0.27, len(STRATEGIES))
    for (key, _, _, _, _), dx in zip(STRATEGIES, dodge):
        ax.plot(xb + dx, by[key], linestyle="none", zorder=3, clip_on=False, **strategy_marker(key))
    ax.axvline(12.35, color=HAIRLINE, lw=0.6, ls=(0, (1, 1.6)))
    ax.set_xticks(xb)
    ax.set_xticklabels(by.label, rotation=45, ha="right", rotation_mode="anchor", fontsize=5.8)
    ax.tick_params(axis="x", length=2, pad=1.5)
    ax.set_xlim(-0.6, xb[-1] + 0.5)
    ax.spines["bottom"].set_bounds(-0.4, xb[-1] + 0.4)
    ax.set_ylabel("Mean ARI over six methods")
    ax.set_ylim(0.2, 0.72)
    ax.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    # Marker key in one line above the axes (strategies overlap too much for in-data labels).
    renderer = fig.canvas.get_renderer()
    kx = ax.get_tightbbox(renderer).x0 / fig.dpi
    ky = top_y0 + top_h + 0.17
    for key, name, _, _, _ in STRATEGIES:
        fig.add_artist(Line2D([(kx + 0.04) / W], [ky / H], transform=fig.transFigure, ls="none",
                              **strategy_marker(key)))
        t = fig.text((kx + 0.11) / W, ky / H, name, fontsize=6, va="center", ha="left",
                     color=legible(STRATEGY_COLORS[name]) if key != "expert" else INK_SOFT)
        kx += 0.11 + t.get_window_extent(renderer).width / fig.dpi + 0.17
    letters.append((ax, "b", 1))
    by[["dataset", "section", "label", "expert", "default", "rule", "oracle"]].to_csv(
        SOURCE_DIR / "Fig4b_strategies.csv", index=False)

    # ── c: retention of ARI under data degradation (paired with the clean data) ──
    ax_c = ax = add_axes(fig, left_x0, bot_y0, left_w, bot_h)
    xs = np.array([0.0, 1.0, 2.0, 3.75, 4.75])
    rows_c = []
    series = {}
    for key, name in (("default", "Schema default"), ("rule", "Label-free rule")):
        vals = []
        for cond, _, _ in NOISE:
            r = noise[(noise.condition == cond) & (noise.strategy == key)].iloc[0]
            vals.append(r.retention)
            rows_c.append({"condition": cond, "strategy": name, "retention": r.retention,
                           "noisy_mean_ari": r.noisy_mean, "clean_mean_ari": r.clean_mean,
                           "n_method_section_cells": int(r.n_cells)})
        series[key] = np.array(vals)
    ax.axhline(1.0, color=HAIRLINE, lw=0.6, zorder=0)
    for key in ("rule", "default"):
        sname = next(n for k, n, *_ in STRATEGIES if k == key)
        color = STRATEGY_COLORS[sname]
        v = series[key]
        for idx in ([0, 1, 2], [3, 4]):
            ax.plot(xs[idx], v[idx], color=color, lw=1.0, zorder=2)
        ax.plot(xs, v, linestyle="none", zorder=3, **{**strategy_marker(key), "ms": 4.0 if key == "default" else 3.6})
        # Direct label at the end of the count-thinning series.
        ax.text(xs[2] + 0.17, v[2], sname, ha="left", va="center", fontsize=6, color=color)
    ax.set_xticks(xs)
    ax.set_xticklabels([n for _, n, _ in NOISE], fontsize=6)
    ax.tick_params(axis="x", length=0, pad=2.5)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim(xs[0] - 0.45, xs[-1] + 0.45)
    spans = []
    for g, gname in NOISE_GROUPS:
        idx = [k for k, (_, _, grp) in enumerate(NOISE) if grp == g]
        spans.append((xs[idx].min(), xs[idx].max(), gname.split("\n")[0]))
    group_brackets(ax, fig, spans, y_pt=-12.5, text_pt=-14.0, inset=0.3)
    ax.set_ylabel("ARI retained\n(degraded / original)")
    ax.set_ylim(0.5, 1.04)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.spines["left"].set_bounds(0.5, 1.0)
    letters.append((ax, "c", 2))
    pd.DataFrame(rows_c).to_csv(SOURCE_DIR / "Fig4c_noise.csv", index=False)

    # ── d: mean ARI of each method on each platform with the label-free rule; ranks annotated ──
    hx0, hw = right_x0 + 0.46, 2.42
    ax_d = ax = add_axes(fig, hx0, bot_y0 + 0.08, hw, bot_h - 0.08)
    cax = add_axes(fig, hx0 + hw + 0.12, bot_y0 + 0.08, 0.06, bot_h - 0.08)
    sub = plat[plat.strategy == "rule"].pivot(index="platform", columns="method", values="ari")
    sub = sub.loc[PLATFORMS, [k for k, _ in PLOT_METHODS]]
    cmap, norm = CMAP_SEQUENTIAL, Normalize(vmin=0.2, vmax=0.6)
    im = ax.pcolormesh(np.arange(sub.shape[1] + 1) - 0.5, np.arange(sub.shape[0] + 1) - 0.5, sub.values,
                       cmap=cmap, norm=norm, edgecolors="white", linewidth=1.0)
    ax.set_ylim(sub.shape[0] - 0.5, -0.5)
    ranks = sub.rank(axis=1, ascending=False).astype(int)
    for i in range(sub.shape[0]):
        for k in range(sub.shape[1]):
            v = sub.values[i, k]
            tc = text_colour_for(to_rgb(cmap(norm(v))))
            ax.text(k, i - 0.1, f"{v:.2f}", ha="center", va="center", fontsize=6, color=tc)
            ax.text(k, i + 0.22, f"#{ranks.values[i, k]}", ha="center", va="center", fontsize=5.2,
                    color=tc, alpha=0.85)
            if ranks.values[i, k] == 1:   # outline the top-ranked method on each platform
                ax.add_patch(Rectangle((k - 0.47, i - 0.47), 0.94, 0.94, fill=False, ec=INK, lw=0.9, zorder=3))
    ax.set_xticks(range(len(PLOT_METHODS)))
    ax.set_xticklabels([n for _, n in PLOT_METHODS], fontsize=5.8)
    for t, (_, n) in zip(ax.get_xticklabels(), PLOT_METHODS):
        t.set_color(legible(METHOD_COLORS[n]))
    ax.set_yticks(range(len(PLATFORMS)))
    ax.set_yticklabels(PLATFORMS, fontsize=6)
    ax.tick_params(length=0, pad=2.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.solids.set_rasterized(False)   # keep the colour bar vector, like the heatmap
    cb.set_ticks([0.2, 0.4, 0.6])
    cb.ax.tick_params(length=2, width=0.5, pad=1.5, labelsize=5.8)
    cb.set_label("Mean ARI", labelpad=3, fontsize=6)
    letters.append((ax, "d", 2))
    plat.rename(columns={"ari": "mean_ari"}).to_csv(SOURCE_DIR / "Fig4d_platforms.csv", index=False)

    # ── panel letters: one top line per row, left of each panel's left-most element ──
    renderer = fig.canvas.get_renderer()
    row_line = {1: ky + 0.15, 2: bot_y0 + bot_h + 0.22}
    for ax_, letter, row in letters:
        x_in = max(ax_.get_tightbbox(renderer).x0 / fig.dpi - 0.12, 0.0)
        fig.text(x_in / W, row_line[row] / H, letter, va="top", ha="left", **PANEL_LABEL_FONT)

    fig.savefig(FIG_DIR / "figureA.pdf")
    fig.savefig(FIG_DIR / "figureA.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
