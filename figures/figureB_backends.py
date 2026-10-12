"""
Figure 3: dependence of ChatSpatial on the language-model backend (experiment B).

All panels are computed from the experiment-B tables produced on the cluster
(ChatSpatial 1.5.5 execution, tag exec_v155):
    experiments/backends/results/analysis/*.csv
    experiments/backends/results/analysis/expB_summary.txt   (GLMM odds ratios)
    experiments/backends/results/trial_outcomes.csv.gz (per-trial outcomes)

Confidence intervals come from the analysis tables: cluster bootstraps over
the eight prompts, as pre-specified in experiments/backends/expB_rules_2026-10-08.txt.

Paper-wide style: Direction A ("Ink & Cobalt", figure_style.py). Commercial
backends are ink, open-weight backends rust; outcomes, schema levels and
method-choice classes use the cobalt ramp and neutral greys. Classes are
labelled directly on the data instead of in legend boxes.

Output: figures/figureB.pdf/.png and source data in source_data/.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullFormatter, NullLocator
import pandas as pd

from figure_style import (
    ACCESS_COLORS,
    COBALT,
    COBALT_LIGHT,
    DOUBLE_COL,
    HAIRLINE,
    INK,
    INK_SOFT,
    OUTCOME_COLORS,
    PANEL_LABEL_FONT,
    SCHEMA_COLORS,
    apply_style,
)

ROOT = Path(__file__).resolve().parents[1]
EXPB = ROOT / "experiments" / "backends" / "results"
ANALYSIS = EXPB / "analysis"
OUTCOMES = EXPB / "trial_outcomes.csv.gz"
FIG_DIR = ROOT / "figures"
SOURCE_DIR = ROOT / "source_data"

BACKENDS = [
    ("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
    ("claude-haiku-5-5", "Claude Haiku 5.5"),
    ("claude-sonnet-5-5", "Claude Sonnet 5.5"),
    ("claude-opus-5-5", "Claude Opus 5.5"),
    ("gpt-6-sol", "GPT-6 Sol"),
    ("gpt-6.1-sol", "GPT-6.1 Sol"),
    ("ow:google/gemma-4-12B-it", "Gemma 4 12B"),
    ("ow:Qwen/Qwen3.8-27B-FP8", "Qwen3.8 27B"),
]
OPEN_WEIGHT = {"ow:google/gemma-4-12B-it", "ow:Qwen/Qwen3.8-27B-FP8"}
# Row positions (top to bottom): a gap separates the open-weight models.
ROW_Y = [0, 1, 2, 3, 4, 5, 6.75, 7.75]
INFO_ORDER = ["full", "bare", "minimal"]
INFO_NAME = {"full": "Full", "bare": "Bare", "minimal": "Minimal"}
SCHEMA_KEY = {"full": "Full", "bare": "Bare", "minimal": "None"}
MECH_NAME = {"json": "JSON", "native": "Native"}

# Method-choice classes (panel f). Not in figure_style: defined here from the
# same tones (cobalt ramp for the schema default, neutral greys otherwise).
CHOICE_COLORS = {
    "Left to default": COBALT,
    "Default named": "#9DB9D9",       # between COBALT_MID and COBALT_LIGHT
    "Other method": "#D4D4D4",
    "No valid call": "#6B6B6B",
}
EXPLICIT_TEXT = "#8C8C8C"             # legible text tone for the explicit-error grey

# Figure geometry (inches); authored at 180 mm so it places at \textwidth.
W = DOUBLE_COL
H = 4.84
MS = 3.6          # marker size for estimates
CI_LW = 0.9       # interval line width


def access(key: str) -> str:
    return "Open-weight" if key in OPEN_WEIGHT else "Commercial"


def access_color(key: str) -> str:
    return ACCESS_COLORS[access(key)]


def parse_completion_or(summary: str) -> pd.DataFrame:
    """Native-vs-JSON odds ratios for output-verified completion (GLMM)."""
    block = summary.split("### outcome: completed")[1].split("### outcome:")[0]
    rows = []
    m = re.search(r"additive: OR\(native vs json\) = ([\d.]+) \[([\d.]+), ([\d.]+)\]", block)
    rows.append(("All levels", *map(float, m.groups())))
    for info in INFO_ORDER:
        m = re.search(
            rf"interaction: OR\(native vs json \| info={info}\) = ([\d.]+) \[([\d.]+), ([\d.]+)\]",
            block,
        )
        rows.append((INFO_NAME[info], *map(float, m.groups())))
    return pd.DataFrame(rows, columns=["level", "odds_ratio", "ci_low", "ci_high"])


def add_axes(fig: plt.Figure, x0: float, y0: float, w: float, h: float) -> plt.Axes:
    """Axes placed in inches from the lower-left corner of the figure."""
    return fig.add_axes([x0 / W, y0 / H, w / W, h / H])


def text_width_in(fig: plt.Figure, s: str, fs: float) -> float:
    t = fig.text(0, 0, s, fontsize=fs)
    w = t.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return w


def marker_key(fig: plt.Figure, x_in: float, y_in: float, items, fs: float = 6.0, gap: float = 0.14) -> None:
    """One line of marker + label pairs at figure inches (a key, not a legend box)."""
    for kw, label in items:
        fig.add_artist(mpl.lines.Line2D([(x_in + 0.04) / W], [y_in / H], transform=fig.transFigure,
                                        ls="none", **kw))
        base, _, sup = label.partition("^")   # manual superscript keeps it at >= 5 pt
        fig.text((x_in + 0.11) / W, y_in / H, base, fontsize=fs, va="center", ha="left", color=INK)
        x_in += 0.11 + text_width_in(fig, base, fs)
        if sup:
            fig.text((x_in + 0.002) / W, (y_in + 0.028) / H, sup, fontsize=5.0, va="center", ha="left", color=INK)
            x_in += 0.002 + text_width_in(fig, sup, 5.0)
        x_in += gap


def backend_axis(ax: plt.Axes, labels: bool = True) -> None:
    """Shared y axis for the per-backend panels: access-coloured names, grouped rows."""
    ax.set_yticks(ROW_Y)
    if labels:
        ax.set_yticklabels([n for _, n in BACKENDS])
        for t, (key, _) in zip(ax.get_yticklabels(), BACKENDS):
            t.set_color(access_color(key))
    else:
        ax.set_yticklabels([])
    ax.set_ylim(ROW_Y[-1] + 1.15, -0.6)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines["left"].set_visible(False)


def group_headers(ax: plt.Axes) -> None:
    """'Commercial' / 'Open-weight' set as row-group headers in the tick-label column."""
    trans = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for y, name in ((-0.95, "Commercial"), (5.85, "Open-weight")):
        ax.text(-0.04, y, name, transform=trans, ha="right", va="center", fontsize=5.6,
                color=ACCESS_COLORS[name], fontstyle="italic", clip_on=False)


def main() -> None:
    apply_style()
    # Fixed 180 mm canvas: no tight-bbox cropping, so fonts print at their nominal size.
    mpl.rcParams["savefig.bbox"] = "standard"
    SOURCE_DIR.mkdir(exist_ok=True)

    rates = pd.read_csv(ANALYSIS / "rates_by_backend_condition.csv")
    trials = pd.read_csv(OUTCOMES)
    main_trials = trials[trials["experiment"] == "main"].copy()
    for col in ("completed", "silent_failure", "explicit_failure", "tool_correct"):
        main_trials[col] = main_trials[col].astype(str).str.lower().eq("true").astype(int)
    bdesc = pd.read_csv(ANALYSIS / "bdesc_by_backend.csv")
    binv = pd.read_csv(ANALYSIS / "binv_per_backend.csv")
    bias = pd.read_csv(ANALYSIS / "bbias_method_counts.csv")
    summary = (ANALYSIS / "expB_summary.txt").read_text()

    fig = plt.figure(figsize=(W, H))
    top_y0, top_h = 3.06, 1.36     # row 1 axes
    bot_y0, bot_h = 0.47, 1.62     # row 2 axes
    letters = []                   # (x of left-most element in inches, row, letter)

    # ── a: output-verified completion and correct tool per backend (native + full) ──
    ax_a = ax = add_axes(fig, 0.95, top_y0, 1.30, top_h)
    sub = rates[(rates.mechanism == "native") & (rates["info"] == "full")].set_index("label")
    rows_a = []
    for (key, name), y in zip(BACKENDS, ROW_Y):
        r = sub.loc[key]
        c = access_color(key)
        ax.plot([r.completed_lo, r.completed_hi], [y, y], color=c, lw=CI_LW, solid_capstyle="round")
        ax.plot(r.completed, y, "o", color=c, ms=MS, mec="white", mew=0.4, zorder=3)
        ax.plot(r.tool_correct, y, "D", mfc="none", mec=INK, ms=5.6, mew=0.6, zorder=4, clip_on=False)
        rows_a.append(
            {
                "backend": name,
                "tool_selection_accuracy": r.tool_correct,
                "completion": r.completed,
                "completion_ci_low": r.completed_lo,
                "completion_ci_high": r.completed_hi,
                "n_trials": int(r.n),
            }
        )
    backend_axis(ax)
    group_headers(ax)
    ax.set_xlim(0.5, 1.04)
    ax.set_xticks([0.6, 0.8, 1.0])
    ax.spines["bottom"].set_bounds(0.5, 1.0)
    ax.set_xlabel("Rate (native tool calling, full schema)")
    pd.DataFrame(rows_a).to_csv(SOURCE_DIR / "Fig3a_backends.csv", index=False)

    # ── b: outcome composition by mechanism x information, pooled over backends ──
    ax_b = ax = add_axes(fig, 2.92, top_y0, 1.62, top_h)
    rows_b = []
    bw = 0.74
    xs, xticks, groups, counts = [], [], [], []
    xpos = 0.0
    for info in INFO_ORDER:
        first = xpos
        for mech in ("json", "native"):
            cell = main_trials[(main_trials["info"] == info) & (main_trials.mechanism == mech)]
            # Mutually exclusive classes: a silent failure can also pass the
            # output check (e.g. a substituted method), so it is counted once.
            sil_mask = cell.silent_failure.astype(bool)
            sil = sil_mask.mean()
            comp = (cell.completed.astype(bool) & ~sil_mask).mean()
            exp = (cell.explicit_failure.astype(bool) & ~sil_mask).mean()
            other = max(0.0, 1.0 - comp - sil - exp)
            bottom = 0.0
            for share, cls in ((comp, "Completed"), (exp, "Explicit error"), (other, "Not executed"),
                               (sil, "Silent failure")):
                ax.bar(xpos, share, bottom=bottom, color=OUTCOME_COLORS[cls], width=bw, lw=0)
                bottom += share
            n_other = int(round(other * len(cell)))
            n_sil = int(cell.silent_failure.sum())
            counts.append((xpos, n_other, n_sil))
            xs.append(xpos)
            xticks.append(MECH_NAME[mech])
            rows_b.append(
                {
                    "information": info,
                    "mechanism": mech,
                    "n_trials": len(cell),
                    "completed_without_silent_failure": comp,
                    "explicit_failure": exp,
                    "not_executed": other,
                    "silent_failure": sil,
                    "n_silent": n_sil,
                }
            )
            last = (comp, exp)
            xpos += 1.0
        groups.append((first, xpos - 1.0, info))
        xpos += 0.5
    x_last = xs[-1]
    ax.set_xticks(xs)
    ax.set_xticklabels(xticks, fontsize=5.6)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_xlim(xs[0] - 0.55, x_last + 0.55)
    ax.spines["bottom"].set_visible(False)
    # Schema-level groups: a bar in the schema colour under each pair, named below it.
    off = mpl.transforms.offset_copy(ax.get_xaxis_transform(), fig=fig, y=-11.5, units="points")
    for x0, x1, info in groups:
        ax.add_patch(Rectangle((x0 - bw / 2, 0), x1 - x0 + bw, 0.022, transform=off,
                               color=SCHEMA_COLORS[SCHEMA_KEY[info]], clip_on=False, lw=0))
        ax.annotate(INFO_NAME[info], xy=((x0 + x1) / 2, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -14.5), textcoords="offset points", ha="center", va="top", fontsize=6.2)
    ax.set_ylabel("Share of trials", labelpad=2)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 0.5, 1.0])
    ax.spines["left"].set_bounds(0, 1)
    # Classes too small to see as segments: counts of trials in two rows above the bars.
    count_rows = (("Not executed", 1.065, INK_SOFT), ("Silent failure", 1.155, OUTCOME_COLORS["Silent failure"]))
    for x, n_other, n_sil in counts:
        for (cls, yy, tc), n in zip(count_rows, (n_other, n_sil)):
            if n:
                ax.text(x, yy, str(n), ha="center", va="center", fontsize=5.6, color=tc,
                        fontweight="bold" if cls == "Silent failure" else "normal")
    # Direct class labels to the right of the last bar.
    lx = x_last + bw / 2 + 0.14
    comp, exp = last
    for cls, yy, tc in (("Completed", comp / 2, COBALT), ("Explicit error", comp + exp / 2, EXPLICIT_TEXT),
                        *[(c, y, t) for c, y, t in count_rows]):
        ax.text(lx, yy, cls, ha="left", va="center", fontsize=5.8, color=tc, clip_on=False)
    pd.DataFrame(rows_b).to_csv(SOURCE_DIR / "Fig3b_outcomes.csv", index=False)

    # ── c: native vs JSON odds ratio for completion (GLMM) ──
    ax_c = ax = add_axes(fig, 5.63, top_y0, 1.40, top_h)
    ors = parse_completion_or(summary)
    ys_c = {"All levels": 0, "Full": 1.25, "Bare": 2.25, "Minimal": 3.25}
    for _, r in ors.iterrows():
        y = ys_c[r.level]
        face = INK if r.level == "All levels" else SCHEMA_COLORS[SCHEMA_KEY[r.level.lower()]]
        ax.plot([r.ci_low, r.ci_high], [y, y], color=INK, lw=CI_LW, solid_capstyle="round")
        ax.plot(r.odds_ratio, y, "s", mfc=face, mec=INK, mew=0.6, ms=MS + 0.6, zorder=3)
    ax.axvline(1.0, color=HAIRLINE, lw=0.6, zorder=0)
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 4])
    ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.16, 6.5)
    ax.set_yticks(list(ys_c.values()))
    ax.set_yticklabels(list(ys_c.keys()))
    for t in ax.get_yticklabels():
        t.set_color(INK if t.get_text() == "All levels" else INK_SOFT)
    ax.set_ylim(3.8, -1.15)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines["left"].set_visible(False)
    for x, ha, s in ((0.9, "right", "← JSON higher"), (1.11, "left", "Native higher →")):
        ax.text(x, -0.95, s, ha=ha, va="center", fontsize=5.4, color=INK_SOFT)
    ax.set_xlabel("Odds ratio of completion,\nnative vs JSON")
    ors.to_csv(SOURCE_DIR / "Fig3c_native_vs_json.csv", index=False)

    # ── d: description ablation (labelled dot plot) ──
    ax_d = ax = add_axes(fig, 1.12, bot_y0, 1.13, 1.52)
    arms = [("desc_swap", "Descriptions\nswapped"), ("neutral_swap", "Neutral names,\ndescriptions\nswapped")]
    arm_y = {"desc_swap": 0.0, "neutral_swap": 1.45}
    rows_d = []
    for group, acc, dy in (("closed", "Commercial", -0.2), ("open", "Open-weight", 0.2)):
        color = ACCESS_COLORS[acc]
        for arm, _ in arms:
            sel = bdesc[bdesc.arm == arm]
            mask = sel.label.isin(OPEN_WEIGHT) if group == "open" else ~sel.label.isin(OPEN_WEIGHT)
            s = sel[mask]
            share = float((s.followed_desc * s.n_ok).sum() / s.n_ok.sum())
            y = arm_y[arm] + dy
            ax.plot([0, share], [y, y], color=color, lw=0.8, solid_capstyle="butt")
            ax.plot(share, y, "o", color=color, ms=MS, mec="white", mew=0.4, clip_on=False, zorder=3)
            t = ax.text(share + 0.016, y, f"{share:.2f}", ha="left", va="center", fontsize=5.8, color=color)
            if arm == "desc_swap":   # name the two model groups once, on the first arm
                ax.annotate(acc.lower(), xy=(1, 0.5), xycoords=t, xytext=(3, 0), textcoords="offset points",
                            ha="left", va="center", fontsize=5.8, color=color, fontstyle="italic")
            rows_d.append({"arm": arm, "models": group, "followed_description": share, "n": int(s.n_ok.sum())})
    ax.set_yticks([arm_y[a] for a, _ in arms])
    ax.set_yticklabels([lbl for _, lbl in arms], linespacing=1.05)
    ax.set_ylim(arm_y["neutral_swap"] + 0.75, -0.75)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines["left"].set_bounds(arm_y["neutral_swap"] + 0.45, -0.45)
    ax.set_xlim(-0.004, 0.3)
    ax.set_xticks([0, 0.1, 0.2, 0.3])
    ax.spines["bottom"].set_bounds(0, 0.3)
    ax.set_xlabel("Share of selections following\nthe moved description")
    pd.DataFrame(rows_d).to_csv(SOURCE_DIR / "Fig3d_description_ablation.csv", index=False)

    # ── e: invocation determinism per backend (native + full) ──
    ax_e = ax = add_axes(fig, 3.30, bot_y0, 1.50, bot_h)
    inv = binv.set_index("label")
    rows_e = []
    for (key, name), y in zip(BACKENDS, ROW_Y):
        r = inv.loc[key]
        c = access_color(key)
        ax.plot([r.lo, r.hi], [y, y], color=c, lw=CI_LW, solid_capstyle="round")
        ax.plot(r.modal_share, y, "o", color=c, ms=MS, mec="white", mew=0.4, zorder=3)
        ax.plot(r.pass_k, y, "x", color=INK, ms=4.6, mew=0.8, zorder=4)
        rows_e.append(
            {
                "backend": name,
                "modal_call_share": r.modal_share,
                "ci_low": r.lo,
                "ci_high": r.hi,
                "identical_call_cells": int(r.identical),
                "n_cells": int(r.n_cells),
                "pass_at_10": r.pass_k,
            }
        )
    backend_axis(ax)
    ax.set_xlim(0.4, 1.04)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0])
    ax.spines["bottom"].set_bounds(0.4, 1.0)
    ax.set_xlabel("Share of 10 repeated requests")
    pd.DataFrame(rows_e).to_csv(SOURCE_DIR / "Fig3e_invocation_determinism.csv", index=False)

    # ── f: method choice on open-ended requests (shares the backend rows of e) ──
    ax_f = ax = add_axes(fig, 4.98, bot_y0, 2.04, bot_h)
    # Default method stated in each tool's schema text for the open-ended prompts.
    schema_default = {
        "bias_autocorr": "neighborhood",
        "bias_ccc": "fastccc",
        "bias_cnv": "infercnvpy",
        "bias_deconv": "flashdeconv",
        "bias_domains": "spagcn",
        "bias_svg": "sparkx",
        "bias_traj": "cellrank",
    }
    rows_f = []
    seg = {}
    for (key, name), y in zip(BACKENDS, ROW_Y):
        sub_b = bias[bias.label == key]
        method_cols = [c for c in sub_b.columns if c not in ("prompt_id", "label")]
        total = sub_b[method_cols].to_numpy().sum()
        omitted = sub_b["(omitted)"].sum() / total
        invalid = sub_b["NO_VALID_CALL"].sum() / total if "NO_VALID_CALL" in sub_b else 0.0
        default_named = (
            sum(
                sub_b.loc[sub_b.prompt_id == pid, schema_default[pid]].sum()
                for pid in schema_default
                if schema_default[pid] in sub_b.columns
            )
            / total
        )
        other = 1.0 - omitted - invalid - default_named
        left = 0.0
        for share, cls in ((omitted, "Left to default"), (default_named, "Default named"),
                           (other, "Other method"), (invalid, "No valid call")):
            ax.barh(y, share, left=left, color=CHOICE_COLORS[cls], height=0.72, lw=0)
            seg[(name, cls)] = (left, share)
            left += share
        rows_f.append(
            {
                "backend": name,
                "n_calls": int(total),
                "default_left_to_schema": omitted,
                "documented_default_named": default_named,
                "other_method_named": other,
                "no_valid_call": invalid,
            }
        )
    backend_axis(ax, labels=False)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel("Share of open-ended requests")
    # Direct labels inside the widest segment of each class.
    for name, cls, tc in (("Claude Sonnet 5.5", "Left to default", "white"),
                          ("GPT-6 Sol", "Default named", INK),
                          ("Gemma 4 12B", "Other method", INK)):
        x0, w = seg[(name, cls)]
        y = ROW_Y[[n for _, n in BACKENDS].index(name)]
        ax.text(x0 + w / 2, y, cls, ha="center", va="center", fontsize=5.8, color=tc)
    x0, w = seg[("Qwen3.8 27B", "No valid call")]
    ax.annotate("No valid call", xy=(x0 + w / 2, ROW_Y[-1] + 0.36), xytext=(x0 + w / 2 - 0.035, ROW_Y[-1] + 0.78),
                ha="right", va="center", fontsize=5.6, color=CHOICE_COLORS["No valid call"],
                arrowprops=dict(arrowstyle="-", lw=0.5, color=CHOICE_COLORS["No valid call"],
                                shrinkA=0, shrinkB=0, connectionstyle="angle,angleA=0,angleB=90,rad=0"))
    pd.DataFrame(rows_f).to_csv(SOURCE_DIR / "Fig3f_method_choice.csv", index=False)

    # ── keys for marker meanings (a, e), placed above the axes ──
    renderer = fig.canvas.get_renderer()

    def left_in(ax_):
        return ax_.get_tightbbox(renderer).x0 / fig.dpi

    key_a_y = top_y0 + top_h + 0.25
    marker_key(fig, left_in(ax_a), key_a_y, [
        (dict(marker="o", ms=MS, color=INK_SOFT, mec="white", mew=0.4), "Output-verified completion"),
        (dict(marker="D", ms=5.0, mfc="none", mec=INK, mew=0.6), "Correct tool"),
    ])
    key_e_y = bot_y0 + bot_h + 0.12
    marker_key(fig, left_in(ax_e), key_e_y, [
        (dict(marker="o", ms=MS, color=INK_SOFT, mec="white", mew=0.4), "Most frequent call"),
        (dict(marker="x", ms=4.6, color=INK, mew=0.8), "pass^10"),
    ])

    # ── panel letters: one top line per row, left of each panel's left-most element ──
    row1 = [(ax_a, "a"), (ax_b, "b"), (ax_c, "c")]
    row2 = [(ax_d, "d"), (ax_e, "e"), (ax_f, "f")]
    for row, y_top in ((row1, key_a_y + 0.15), (row2, key_e_y + 0.23)):
        for ax_, letter in row:
            x = left_in(ax_) - 0.13
            if ax_ is ax_f:
                x = ax_f.get_position().x0 * W - 0.13
            fig.text(max(x, 0.0) / W, y_top / H, letter, va="top", ha="left", **PANEL_LABEL_FONT)

    fig.savefig(FIG_DIR / "figureB.pdf")
    fig.savefig(FIG_DIR / "figureB.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
