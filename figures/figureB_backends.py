"""
Figure B: dependence of ChatSpatial on the language-model backend (experiment B).

All panels are computed from the experiment-B tables produced on the cluster
(ChatSpatial 1.5.5 execution, tag exec_v155):
    experiments/backends/results/analysis/*.csv
    experiments/backends/results/analysis/expB_summary.txt   (GLMM odds ratios)
    experiments/backends/results/trial_outcomes.csv.gz (per-trial outcomes)

Confidence intervals come from the analysis tables: cluster bootstraps over
the eight prompts, as pre-specified in experiments/backends/expB_rules_2026-10-08.txt.

Output: figures/figureB.pdf/.png and source data in source_data/.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, NullLocator
import numpy as np
import pandas as pd

from figure_style import FULL_WIDTH, OI, add_panel_label, apply_style

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
LABEL = dict(BACKENDS)
OPEN_WEIGHT = {"ow:google/gemma-4-12B-it", "ow:Qwen/Qwen3.8-27B-FP8"}
INFO_ORDER = ["full", "bare", "minimal"]
INFO_NAME = {"full": "Full", "bare": "Bare", "minimal": "Minimal"}
MECH_NAME = {"json": "JSON", "native": "Native"}

def backend_color(label: str) -> str:
    return OI["vermillion"] if label in OPEN_WEIGHT else OI["blue"]


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


def main() -> None:
    apply_style()
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

    fig = plt.figure(figsize=(FULL_WIDTH, FULL_WIDTH * 0.62))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.8, hspace=0.75)

    # a: tool selection and output-verified completion per backend (native + full)
    ax = fig.add_subplot(gs[0, 0])
    sub = rates[(rates.mechanism == "native") & (rates["info"] == "full")].set_index("label")
    rows_a = []
    for i, (key, name) in enumerate(BACKENDS):
        r = sub.loc[key]
        y = len(BACKENDS) - 1 - i
        ax.plot([r.completed_lo, r.completed_hi], [y, y], color=backend_color(key), lw=1.0)
        ax.plot(r.completed, y, "o", color=backend_color(key), ms=4)
        ax.plot(r.tool_correct, y, "|", color="black", ms=7, mew=1.0)
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
    ax.set_yticks(range(len(BACKENDS)))
    ax.set_yticklabels([n for _, n in BACKENDS][::-1])
    ax.set_xlim(0.45, 1.03)
    ax.set_xlabel("Rate (native tool calling, full schema)")
    ax.plot([], [], "o", color="grey", ms=4, label="Output-verified completion")
    ax.plot([], [], "|", color="black", ms=7, mew=1.0, label="Correct tool")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=1, handletextpad=0.3)
    add_panel_label(ax, "a", x=-0.62, y=1.22)
    pd.DataFrame(rows_a).to_csv(SOURCE_DIR / "FigB_a_backends.csv", index=False)

    # b: outcome composition by mechanism x information, pooled over backends
    ax = fig.add_subplot(gs[0, 1])
    rows_b = []
    xpos, xticks = 0.0, []
    for info in INFO_ORDER:
        for mech in ("json", "native"):
            cell = main_trials[(main_trials["info"] == info) & (main_trials.mechanism == mech)]
            # Mutually exclusive classes: a silent failure can also pass the
            # output check (e.g. a substituted method), so it is counted once.
            sil_mask = cell.silent_failure.astype(bool)
            sil = sil_mask.mean()
            comp = (cell.completed.astype(bool) & ~sil_mask).mean()
            exp = (cell.explicit_failure.astype(bool) & ~sil_mask).mean()
            other = max(0.0, 1.0 - comp - sil - exp)
            ax.bar(xpos, comp, color=OI["green"], width=0.8)
            ax.bar(xpos, exp, bottom=comp, color=OI["grey"], width=0.8)
            ax.bar(xpos, other, bottom=comp + exp, color=OI["sand"], width=0.8)
            ax.bar(xpos, sil, bottom=comp + exp + other, color=OI["vermillion"], width=0.8)
            xticks.append((xpos, MECH_NAME[mech]))
            rows_b.append(
                {
                    "information": info,
                    "mechanism": mech,
                    "n_trials": len(cell),
                    "completed_without_silent_failure": comp,
                    "explicit_failure": exp,
                    "not_executed": other,
                    "silent_failure": sil,
                    "n_silent": int(cell.silent_failure.sum()),
                }
            )
            xpos += 1.0
        xpos += 0.6
    ax.set_xticks([x for x, _ in xticks])
    ax.set_xticklabels([t for _, t in xticks], rotation=90)
    for j, info in enumerate(INFO_ORDER):
        ax.text(0.5 + j * 2.6, -0.42, INFO_NAME[info], ha="center", transform=ax.get_xaxis_transform())
    ax.set_ylabel("Share of trials")
    ax.set_ylim(0, 1)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=OI["green"]),
        plt.Rectangle((0, 0), 1, 1, color=OI["grey"]),
        plt.Rectangle((0, 0), 1, 1, color=OI["sand"]),
        plt.Rectangle((0, 0), 1, 1, color=OI["vermillion"]),
    ]
    ax.legend(
        handles,
        ["Completed", "Explicit error", "Not executed", "Silent failure"],
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(-0.05, 1.0),
        ncol=2,
        handlelength=0.8,
        columnspacing=0.6,
    )
    add_panel_label(ax, "b", x=-0.3, y=1.22)
    pd.DataFrame(rows_b).to_csv(SOURCE_DIR / "FigB_b_outcomes.csv", index=False)

    # c: native vs JSON odds ratio for completion (GLMM)
    ax = fig.add_subplot(gs[0, 2])
    ors = parse_completion_or(summary)
    for i, r in ors.iloc[::-1].reset_index(drop=True).iterrows():
        ax.plot([r.ci_low, r.ci_high], [i, i], color="black", lw=1.0)
        ax.plot(r.odds_ratio, i, "s", color=OI["blue"], ms=4)
    ax.axvline(1.0, color=OI["grey"], lw=0.6, ls="--")
    ax.set_xscale("log")
    ax.set_xticks([0.25, 0.5, 1, 2, 4])
    ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.15, 6)
    ax.set_yticks(range(len(ors)))
    ax.set_yticklabels(ors.level.iloc[::-1])
    ax.set_xlabel("Odds ratio, native vs JSON\n(output-verified completion)")
    add_panel_label(ax, "c", x=-0.45, y=1.22)
    ors.to_csv(SOURCE_DIR / "FigB_c_native_vs_json.csv", index=False)

    # d: description ablation
    ax = fig.add_subplot(gs[1, 0])
    arms = [("desc_swap", "Descriptions\nswapped"), ("neutral_swap", "Neutral names,\ndescriptions swapped")]
    rows_d = []
    width = 0.36
    for k, (group, color, name) in enumerate(
        [("closed", OI["blue"], "Closed models"), ("open", OI["vermillion"], "Open-weight models")]
    ):
        for j, (arm, _) in enumerate(arms):
            sel = bdesc[bdesc.arm == arm]
            mask = sel.label.isin(OPEN_WEIGHT) if group == "open" else ~sel.label.isin(OPEN_WEIGHT)
            s = sel[mask]
            share = float((s.followed_desc * s.n_ok).sum() / s.n_ok.sum())
            x = j + (k - 0.5) * width
            ax.bar(x, share, width=width, color=color, label=name if j == 0 else None)
            ax.text(x, share + 0.01, f"{share:.0%}", ha="center", va="bottom", fontsize=6)
            rows_d.append({"arm": arm, "models": group, "followed_description": share, "n": int(s.n_ok.sum())})
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a for _, a in arms])
    ax.set_ylim(0, 0.5)
    ax.set_ylabel("Selections following\nthe moved description")
    ax.legend(frameon=False, loc="upper left", handlelength=0.8)
    add_panel_label(ax, "d", x=-0.32, y=1.22)
    pd.DataFrame(rows_d).to_csv(SOURCE_DIR / "FigB_d_description_ablation.csv", index=False)

    # e: invocation determinism per backend (native + full)
    ax = fig.add_subplot(gs[1, 1])
    inv = binv.set_index("label")
    rows_e = []
    for i, (key, name) in enumerate(BACKENDS):
        r = inv.loc[key]
        y = len(BACKENDS) - 1 - i
        ax.plot([r.lo, r.hi], [y, y], color=backend_color(key), lw=1.0)
        ax.plot(r.modal_share, y, "o", color=backend_color(key), ms=4)
        ax.plot(r.pass_k, y, "x", color="black", ms=4, mew=0.8)
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
    ax.set_yticks(range(len(BACKENDS)))
    ax.set_yticklabels([n for _, n in BACKENDS][::-1])
    ax.set_xlim(0.4, 1.04)
    ax.set_xlabel("Share of 10 repeated requests")
    ax.plot([], [], "o", color="grey", ms=4, label="Most frequent call")
    ax.plot([], [], "x", color="black", ms=4, mew=0.8, label="pass$^{10}$")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.0), handletextpad=0.3)
    add_panel_label(ax, "e", x=-0.62, y=1.22)
    pd.DataFrame(rows_e).to_csv(SOURCE_DIR / "FigB_e_invocation_determinism.csv", index=False)

    # f: method choice on open-ended requests
    ax = fig.add_subplot(gs[1, 2])
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
    for i, (key, name) in enumerate(BACKENDS):
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
        y = len(BACKENDS) - 1 - i
        left = 0.0
        for share, color in (
            (omitted, OI["green"]),
            (default_named, OI["skyblue"]),
            (other, OI["orange"]),
            (invalid, OI["grey"]),
        ):
            ax.barh(y, share, left=left, color=color, height=0.7)
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
    ax.set_yticks(range(len(BACKENDS)))
    ax.set_yticklabels([n for _, n in BACKENDS][::-1])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of open-ended requests")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=OI["green"]),
        plt.Rectangle((0, 0), 1, 1, color=OI["skyblue"]),
        plt.Rectangle((0, 0), 1, 1, color=OI["orange"]),
        plt.Rectangle((0, 0), 1, 1, color=OI["grey"]),
    ]
    ax.legend(
        handles,
        ["Left to default", "Default named", "Other method", "No valid call"],
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(-0.02, 1.0),
        ncol=2,
        handlelength=0.8,
        columnspacing=0.6,
    )
    add_panel_label(ax, "f", x=-0.62, y=1.22)
    pd.DataFrame(rows_f).to_csv(SOURCE_DIR / "FigB_f_method_choice.csv", index=False)

    fig.savefig(FIG_DIR / "figureB.pdf")
    fig.savefig(FIG_DIR / "figureB.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
