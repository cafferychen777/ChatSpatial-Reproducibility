"""
Figure 2 (file name figure4): DLPFC head-to-head, ablation, sensitivity, and
system comparison. Nature Communications style, double-column (183 mm)

Row 1 — DLPFC ground-truth benchmark:
  [a] ARI against expert-annotated cortical layers (strip plot)
  [b] Accuracy–consistency tradeoff (scatter)

Row 2 — Schema-enforcement ablation:
  [c] Pydantic validation rate (3 models x 3 conditions)
  [d] E2E execution success (3 tasks x 3 conditions)
  [e] Output concordance (3 tasks x 3 conditions, with survivor-bias hatch)

Row 3 — Prompt sensitivity + cross-system (tool selection):
  [f] Cross-paraphrase tool accuracy (5 groups x 3 models)
  [g] Within vs cross-paraphrase parameter concordance
  [h] Cross-system tool selection accuracy (5 conditions x 3 models)

Row 4 — Cross-system (consistency) + E2E benchmark:
  [i] Cross-system tool consistency (5 conditions x 3 models)
  [j] E2E execution success (3 systems x 2 tasks)
  [k] E2E pairwise concordance (strip plot)
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from figure_style import FULL_WIDTH, OI, add_panel_label, apply_style

apply_style()

# ── Paths ────────────────────────────────────────────────────────
FIGURE_DIR = Path(__file__).resolve().parent
REPO_DIR = FIGURE_DIR.parent
# Rows 2-4: round-1 experiments; row 1: DLPFC head-to-head (ChatSpatial 1.5.9)
DATA_DIR = REPO_DIR / "experiments" / "round1" / "results"

# Ablation
ABL_DATA_DIR = DATA_DIR / "ablation"

# Cross-system
CROSS_METRICS_CSV = DATA_DIR / "cross_system" / "analysis" / "cross_system_metrics.csv"

# E2E benchmark
E2E_RAW_PATH = DATA_DIR / "e2e_benchmark" / "e2e_benchmark_raw.jsonl"
# Pairwise replicate concordance, written by experiments/round1/scripts/e2e_pairwise_concordance.py
# from the benchmark outputs (not included: 5.7 GB)
E2E_PAIRWISE_CSV = DATA_DIR / "e2e_benchmark" / "e2e_pairwise_concordance.csv"

# Sensitivity
SENSITIVITY_DIR = DATA_DIR / "sensitivity"
ABLATION_INV_DIR = DATA_DIR / "ablation" / "invocation"

# ── Color constants ──────────────────────────────────────────────

# Ablation condition colors
ABL_COND_COLORS = {
    "full_schema": OI["blue"],
    "bare_schema": OI["orange"],
    "no_schema":   OI["vermillion"],
}
ABL_COND_LABELS = {
    "full_schema": "Full schema",
    "bare_schema": "Bare schema",
    "no_schema":   "No schema",
}
ABL_CONDITIONS = ["full_schema", "bare_schema", "no_schema"]

# Model colors (for sensitivity / cross-system panels)
MODEL_COLORS = {
    "gemini-2.5-flash": OI["orange"],
    "claude-haiku-4-5-20251001": OI["blue"],
    "gpt-5.4": OI["green"],
}
MODEL_LABELS = {
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "gpt-5.4": "GPT-5.4",
}

# System colors (for E2E benchmark)
SYSTEM_COLORS = {
    "chatspatial": OI["blue"],
    "stagent": OI["vermillion"],
    "spatialagent": OI["green"],
}
SYSTEM_LABELS = {
    "chatspatial": "ChatSpatial",
    "stagent": "STAgent",
    "spatialagent": "SpatialAgent",
}

# Model display names for ablation
ABL_MODEL_DISPLAY = {
    "gemini-2.5-flash":          "Gemini 2.5\nFlash",
    "claude-haiku-4-5-20251001": "Claude\nHaiku 4.5",
    "gpt-5.4":                   "GPT-5.4",
}
ABL_MODELS = list(ABL_MODEL_DISPLAY.keys())

# Task display names for ablation
ABL_TASK_DISPLAY = {
    "spatial_domains": "Spatial\ndomains",
    "svg_detection":   "SVG\ndetection",
    "deconvolution":   "Deconv.",
}
ABL_TASKS = list(ABL_TASK_DISPLAY.keys())

ABL_CONC_METRIC_LABEL = {
    "spatial_domains": "ARI",
    "svg_detection":   "Jaccard@100",
    "deconvolution":   "Pearson r",
}

# Cross-system conditions
CS_CONDITIONS = [
    "full_schema", "bare_schema", "no_schema",
    "stagent_context", "spatialagent_context",
]
CS_CONDITION_LABELS = {
    "full_schema": "Full\nschema",
    "bare_schema": "Bare\nschema",
    "no_schema": "No\nschema",
    "stagent_context": "STAgent",
    "spatialagent_context": "Spatial\nAgent",
}

# E2E benchmark
E2E_SYSTEMS = ["chatspatial", "stagent", "spatialagent"]
E2E_TASKS = ["spatial_domains", "svg_detection", "cell_communication"]
E2E_TASK_LABELS = {
    "spatial_domains": "Spatial\ndomains",
    "svg_detection": "SVG\ndetection",
    "cell_communication": "Cell\ncomm.",
}

# DLPFC ground-truth benchmark sample markers
DLPFC_SAMPLE_IDS = [151673, 151507, 151669]
DLPFC_SAMPLE_MARKERS = {151673: "o", 151507: "s", 151669: "^"}

# Sensitivity groups
SENS_GROUPS = ["spatial_domains", "deconvolution", "svg_detection",
               "cell_communication", "moran_i"]
SENS_GROUP_DISPLAY = {
    "spatial_domains": "Spatial\ndomains", "deconvolution": "Deconv.",
    "svg_detection": "SVG\ndetection", "cell_communication": "Cell\ncomm.",
    "moran_i": "Moran's I",
}
SENS_GROUP_CONSTRAINED = {
    "spatial_domains": ["method"], "deconvolution": ["method"],
    "svg_detection": ["method"], "cell_communication": ["method", "species"],
    "moran_i": ["analysis_type"],
}
ABLATION_TO_GROUP = {
    "spatial_domains": "spatial_domains", "deconvolution_rctd": "deconvolution",
    "svg_detection": "svg_detection", "cellchat": "cell_communication",
    "moran_i": "moran_i",
}


# ── Utilities ────────────────────────────────────────────────────

def bootstrap_ci(values, n_boot=10_000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    arr = np.array(values, dtype=float)
    boots = np.array([arr[rng.choice(n, size=n, replace=True)].mean()
                      for _ in range(n_boot)])
    return (arr.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5))


def load_jsonl(path):
    records = []
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


# ═════════════════════════════════════════════════════════════════
# Load all data
# ═════════════════════════════════════════════════════════════════

# --- Ablation data ---
inv_df = pd.read_csv(ABL_DATA_DIR / "invocation" / "ablation_invocation_results.csv")
e2e_abl_df = pd.read_csv(ABL_DATA_DIR / "e2e" / "ablation_e2e_results.csv")
conc_df = pd.read_csv(ABL_DATA_DIR / "analysis" / "ablation_metrics_bootstrap.csv")

# Panel (c): per-model Pydantic validation rate
pydantic_data = {}
for cond in ABL_CONDITIONS:
    for model in ABL_MODELS:
        sub = inv_df[(inv_df["condition"] == cond) & (inv_df["model"] == model)]
        if len(sub) > 0:
            rate = (sub["pydantic_valid_rate"] * sub["n_trials"]).sum() / sub["n_trials"].sum()
            pydantic_data[(cond, model)] = rate * 100
        else:
            pydantic_data[(cond, model)] = 0

pydantic_means = {}
for cond in ABL_CONDITIONS:
    pydantic_means[cond] = np.mean([pydantic_data[(cond, m)] for m in ABL_MODELS])

# Panel (d): per-task E2E execution success
abl_exec_data = {}
abl_exec_n = {}
for cond in ABL_CONDITIONS:
    for task in ABL_TASKS:
        sub = e2e_abl_df[(e2e_abl_df["condition"] == cond) & (e2e_abl_df["task_id"] == task)]
        total = sub["n_trials"].sum()
        successes = (sub["exec_success_rate"] * sub["n_trials"]).sum()
        abl_exec_data[(cond, task)] = (successes / total * 100) if total > 0 else 0
        abl_exec_n[(cond, task)] = (int(round(successes)), int(total))

# Panel (e): cross-model concordance
abl_conc_data = {}
for cond in ABL_CONDITIONS:
    for task in ABL_TASKS:
        row = conc_df[
            (conc_df["condition"] == cond)
            & (conc_df["model"] == "all_models")
            & (conc_df["task"] == task)
        ]
        if len(row) == 0:
            abl_conc_data[(cond, task)] = (np.nan, np.nan, np.nan, 0)
            continue
        row = row.iloc[0]
        n = int(row["n_runs"])
        if task == "spatial_domains":
            abl_conc_data[(cond, task)] = (row["ari_mean"], row["ari_ci_lo"], row["ari_ci_hi"], n)
        elif task == "svg_detection":
            abl_conc_data[(cond, task)] = (row["jaccard_mean"], row["jaccard_ci_lo"], row["jaccard_ci_hi"], n)
        elif task == "deconvolution":
            abl_conc_data[(cond, task)] = (row["pearson_mean"], row["pearson_ci_lo"], row["pearson_ci_hi"], n)

# --- Cross-system metrics ---
cs_metrics = []
if CROSS_METRICS_CSV.exists():
    with open(CROSS_METRICS_CSV) as f:
        for row in csv.DictReader(f):
            row["tool_correct_rate"] = float(row["tool_correct_rate"])
            row["tool_consistency"] = float(row["tool_consistency"])
            row["n_trials"] = int(row["n_trials"])
            cs_metrics.append(row)

models_present = sorted(set(r["model"] for r in cs_metrics)) if cs_metrics else []
MODELS = [m for m in MODEL_COLORS if m in models_present]
if not MODELS:
    MODELS = list(MODEL_COLORS.keys())

cs_accuracy = {}
cs_consistency = {}
for cond in CS_CONDITIONS:
    for model in MODELS:
        rows = [r for r in cs_metrics
                if r["condition"] == cond and r["model"] == model]
        if rows:
            acc = [r["tool_correct_rate"] for r in rows]
            con = [r["tool_consistency"] for r in rows]
            cs_accuracy[(cond, model)] = tuple(v * 100 for v in bootstrap_ci(acc))
            cs_consistency[(cond, model)] = tuple(v * 100 for v in bootstrap_ci(con))
        else:
            cs_accuracy[(cond, model)] = (0, 0, 0)
            cs_consistency[(cond, model)] = (0, 0, 0)

# --- E2E benchmark ---
e2e_records = load_jsonl(E2E_RAW_PATH)

e2e_counts = {}
for sys in E2E_SYSTEMS:
    for task in E2E_TASKS:
        trials = [r for r in e2e_records if r["system"] == sys and r["task_id"] == task]
        n_ok = sum(1 for t in trials if t["success"])
        e2e_counts[(sys, task)] = {"ok": n_ok, "fail": len(trials) - n_ok,
                                   "total": len(trials)}


_pairs = pd.read_csv(E2E_PAIRWISE_CSV)
e2e_concordance = {}
for sys in E2E_SYSTEMS:
    for task in E2E_TASKS:
        sel = _pairs[(_pairs["system"] == sys) & (_pairs["task"] == task)]
        e2e_concordance[(sys, task)] = sel["value"].astype(float).tolist()

# --- Sensitivity ---
sens_records = load_jsonl(SENSITIVITY_DIR / "prompt_sensitivity_raw.jsonl")
abl_records = load_jsonl(ABLATION_INV_DIR / "ablation_invocation_raw.jsonl")

sens_tool_accuracy = {}
for gid in SENS_GROUPS:
    for model in MODELS:
        trials = [r for r in sens_records if r["group_id"] == gid and r["model"] == model]
        correct = [1.0 if t["tool_correct"] else 0.0 for t in trials]
        if correct:
            m, lo, hi = bootstrap_ci(correct)
            sens_tool_accuracy[(gid, model)] = (m * 100, lo * 100, hi * 100)
        else:
            sens_tool_accuracy[(gid, model)] = (0, 0, 0)


def param_consistency_within(records, constrained_params):
    indexed = defaultdict(list)
    for r in records:
        indexed[(r.get("paraphrase_idx", 0), r["model"])].append(r)
    consistencies = []
    for (pidx, model), trials in indexed.items():
        for pname in constrained_params:
            vals = [str(t["params"].get(pname)) for t in trials
                    if t.get("tool_correct") and t.get("parse_ok")]
            if len(vals) >= 2:
                mode_count = Counter(vals).most_common(1)[0][1]
                consistencies.append(mode_count / len(vals))
    return consistencies


def param_consistency_cross(records, constrained_params):
    indexed = defaultdict(lambda: defaultdict(list))
    for r in records:
        indexed[r["model"]][r["paraphrase_idx"]].append(r)
    consistencies = []
    for model, para_map in indexed.items():
        for pname in constrained_params:
            para_modes = []
            for pidx in sorted(para_map.keys()):
                vals = [str(t["params"].get(pname)) for t in para_map[pidx]
                        if t.get("tool_correct") and t.get("parse_ok")]
                if vals:
                    para_modes.append(Counter(vals).most_common(1)[0][0])
            if len(para_modes) >= 2:
                overall_mode = Counter(para_modes).most_common(1)[0][0]
                consistencies.append(
                    sum(1 for m in para_modes if m == overall_mode) / len(para_modes))
    return consistencies


sens_concordance = {}
for gid in SENS_GROUPS:
    cp = SENS_GROUP_CONSTRAINED[gid]
    sens_group = [r for r in sens_records if r["group_id"] == gid]
    within_vals = param_consistency_within(sens_group, cp)
    abl_group = [r for r in abl_records
                 if r.get("condition") == "full_schema"
                 and ABLATION_TO_GROUP.get(r.get("prompt_id")) == gid]
    abl_within = []
    abl_indexed = defaultdict(list)
    for r in abl_group:
        abl_indexed[r["model"]].append(r)
    for model, trials in abl_indexed.items():
        for pname in cp:
            vals = [str(t["params"].get(pname)) for t in trials
                    if t.get("tool_correct") and t.get("parse_ok")]
            if len(vals) >= 2:
                mode_count = Counter(vals).most_common(1)[0][1]
                abl_within.append(mode_count / len(vals))
    cross_vals = param_consistency_cross(sens_group, cp)
    all_within = abl_within + within_vals
    within_boot = bootstrap_ci(all_within) if all_within else (np.nan, np.nan, np.nan)
    cross_boot = bootstrap_ci(cross_vals) if cross_vals else (np.nan, np.nan, np.nan)
    sens_concordance[gid] = {
        "within": (within_boot[0] * 100, within_boot[1] * 100, within_boot[2] * 100),
        "cross": (cross_boot[0] * 100, cross_boot[1] * 100, cross_boot[2] * 100),
    }



# ═════════════════════════════════════════════════════════════════
# Figure: 4-row grid (ground-truth row + 3×3)
# ═════════════════════════════════════════════════════════════════

FIG_W = FULL_WIDTH
FIG_H = FULL_WIDTH * 1.10  # ~7.9 in for 4 rows

fig = plt.figure(figsize=(FIG_W, FIG_H))
# Outer grid: DLPFC head-to-head row on top, then the 3x3 block. The extra gap
# between them holds the shared ablation legend.
gs_outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.26,
                             height_ratios=[0.80, 4.0],
                             left=0.07, right=0.96, top=0.975, bottom=0.06)
gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[0],
                                          wspace=0.45)
gs = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=gs_outer[1],
                                      hspace=0.50, wspace=0.45)

# ── Bar geometry ─────────────────────────────────────────────────
BAR_W_3 = 0.22
GAP_3 = 0.03
OFFSETS_3 = np.array([-(BAR_W_3 + GAP_3), 0, BAR_W_3 + GAP_3])

# ═══════════════════════════════════════════════════════════════
# Row 2: Schema-enforcement ablation
# ═══════════════════════════════════════════════════════════════

# Panel (c): Pydantic validation rate per model
ax_c = fig.add_subplot(gs[0, 0])
x_c = np.arange(len(ABL_MODELS))
for i, cond in enumerate(ABL_CONDITIONS):
    vals = [pydantic_data[(cond, m)] for m in ABL_MODELS]
    bars = ax_c.bar(x_c + OFFSETS_3[i], vals, BAR_W_3,
                    color=ABL_COND_COLORS[cond], edgecolor="white", linewidth=0.3,
                    label=ABL_COND_LABELS[cond], zorder=3)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax_c.text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                      f"{v:.0f}", ha="center", va="bottom", fontsize=4)

ax_c.text(0.97, 0.50,
          f"Mean\n"
          f"Full: {pydantic_means['full_schema']:.0f}%\n"
          f"Bare: {pydantic_means['bare_schema']:.0f}%\n"
          f"None: {pydantic_means['no_schema']:.0f}%",
          transform=ax_c.transAxes, fontsize=4.5, va="top", ha="right",
          linespacing=1.3,
          bbox=dict(boxstyle="round,pad=0.3", fc="#f7f7f7", ec="#cccccc", lw=0.4))

ax_c.set_xticks(x_c)
ax_c.set_xticklabels([ABL_MODEL_DISPLAY[m] for m in ABL_MODELS])
ax_c.set_ylabel("Pydantic validation (%)")
ax_c.set_ylim(0, 118)
ax_c.set_yticks([0, 25, 50, 75, 100])
ax_c.axhline(y=100, color="#dddddd", linewidth=0.3, zorder=1)
add_panel_label(ax_c, "c")

# Panel (d): Ablation E2E execution success
ax_d = fig.add_subplot(gs[0, 1])
x_d = np.arange(len(ABL_TASKS))
for i, cond in enumerate(ABL_CONDITIONS):
    vals = [abl_exec_data[(cond, t)] for t in ABL_TASKS]
    bars = ax_d.bar(x_d + OFFSETS_3[i], vals, BAR_W_3,
                    color=ABL_COND_COLORS[cond], edgecolor="white", linewidth=0.3,
                    zorder=3)
    for bar, v, t in zip(bars, vals, ABL_TASKS):
        n_s, n_t = abl_exec_n[(cond, t)]
        ax_d.text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                  f"{n_s}/{n_t}", ha="center", va="bottom",
                  fontsize=3.5, color="#444444")

ax_d.set_xticks(x_d)
ax_d.set_xticklabels([ABL_TASK_DISPLAY[t] for t in ABL_TASKS])
ax_d.set_ylabel("Execution success (%)")
ax_d.set_ylim(0, 118)
ax_d.set_yticks([0, 25, 50, 75, 100])
ax_d.axhline(y=100, color="#dddddd", linewidth=0.3, zorder=1)
add_panel_label(ax_d, "d")

# Panel (e): Ablation output concordance
ax_e = fig.add_subplot(gs[0, 2])
x_e_abl = np.arange(len(ABL_TASKS))
for i, cond in enumerate(ABL_CONDITIONS):
    means = []
    ci_lo_err = []
    ci_hi_err = []
    hatches = []
    for t in ABL_TASKS:
        m, lo, hi, n = abl_conc_data[(cond, t)]
        means.append(m if not np.isnan(m) else 0)
        ci_lo_err.append(m - lo if not np.isnan(lo) else 0)
        ci_hi_err.append(hi - m if not np.isnan(hi) else 0)
        n_s, n_t = abl_exec_n.get((cond, t), (0, 30))
        hatches.append(n_s / n_t < 0.50 if n_t > 0 else False)

    means_arr = np.array(means)
    errs = np.array([ci_lo_err, ci_hi_err])

    for j, t in enumerate(ABL_TASKS):
        hatch = "///" if hatches[j] else None
        ax_e.bar(x_e_abl[j] + OFFSETS_3[i], means_arr[j], BAR_W_3,
                 color=ABL_COND_COLORS[cond],
                 edgecolor="#666666" if hatch else "white",
                 linewidth=0.5 if hatch else 0.3,
                 hatch=hatch, zorder=3)

    ax_e.errorbar(x_e_abl + OFFSETS_3[i], means_arr, yerr=errs,
                  fmt="none", ecolor="#333333", elinewidth=0.5,
                  capsize=1.5, capthick=0.4, zorder=4)

    STAGGER = [0.02, 0.07, 0.12]
    for j, t in enumerate(ABL_TASKS):
        m_val, lo_val, hi_val, n = abl_conc_data[(cond, t)]
        if n > 0 and not np.isnan(m_val):
            y_pos = (hi_val if not np.isnan(hi_val) else m_val) + STAGGER[i]
            label = f"n={n}"
            if hatches[j]:
                label += "*"
            ax_e.text(x_e_abl[j] + OFFSETS_3[i], y_pos, label,
                      ha="center", va="bottom", fontsize=3.5, color="#555555")

task_labels_e = [f"{ABL_TASK_DISPLAY[t]}\n({ABL_CONC_METRIC_LABEL[t]})" for t in ABL_TASKS]
ax_e.set_xticks(x_e_abl)
ax_e.set_xticklabels(task_labels_e, fontsize=5)
ax_e.set_ylabel("Output concordance")
ax_e.set_ylim(0, 1.22)
ax_e.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
ax_e.axhline(y=1.0, color="#dddddd", linewidth=0.3, zorder=1)
ax_e.text(0.98, -0.02, "* exec. < 50%; survivor bias",
          transform=ax_e.transAxes, fontsize=3.8, color="#888888",
          ha="right", va="top", style="italic")
add_panel_label(ax_e, "e")

# Shared ablation legend above row 2 (ablation)
handles_abl = [plt.Rectangle((0, 0), 1, 1, fc=ABL_COND_COLORS[c], ec="white", lw=0.3)
               for c in ABL_CONDITIONS]
fig.legend(handles_abl, [ABL_COND_LABELS[c] for c in ABL_CONDITIONS],
           loc="lower center",
           bbox_to_anchor=(0.5, ax_d.get_position().y1 + 0.022),
           ncol=3, frameon=False, fontsize=6,
           handlelength=1.2, handletextpad=0.4, columnspacing=1.5)

# ═══════════════════════════════════════════════════════════════
# Row 3: Prompt sensitivity + cross-system tool accuracy
# ═══════════════════════════════════════════════════════════════

# Panel (f): Cross-paraphrase tool accuracy
ax_f = fig.add_subplot(gs[1, 0])
x_f = np.arange(len(SENS_GROUPS))
for i, model in enumerate(MODELS):
    means = [sens_tool_accuracy[(g, model)][0] for g in SENS_GROUPS]
    ci_lo = [sens_tool_accuracy[(g, model)][0] - sens_tool_accuracy[(g, model)][1]
             for g in SENS_GROUPS]
    ci_hi = [sens_tool_accuracy[(g, model)][2] - sens_tool_accuracy[(g, model)][0]
             for g in SENS_GROUPS]
    ax_f.bar(x_f + OFFSETS_3[i], means, BAR_W_3,
             color=MODEL_COLORS[model], edgecolor="white", linewidth=0.3,
             label=MODEL_LABELS[model], zorder=3)
    ax_f.errorbar(x_f + OFFSETS_3[i], means, yerr=[ci_lo, ci_hi],
                  fmt="none", ecolor="#333333", elinewidth=0.5,
                  capsize=1.2, capthick=0.4, zorder=4)

ax_f.set_xticks(x_f)
ax_f.set_xticklabels([SENS_GROUP_DISPLAY[g] for g in SENS_GROUPS], fontsize=5)
ax_f.set_ylabel("Tool accuracy (%)")
ax_f.set_ylim(0, 115)
ax_f.set_yticks([0, 25, 50, 75, 100])
ax_f.axhline(y=100, color="#dddddd", linewidth=0.3, zorder=1)
ax_f.legend(loc="lower right", fontsize=4, frameon=False, ncol=1)
add_panel_label(ax_f, "f")

# Panel (g): Within vs cross-paraphrase concordance
ax_g = fig.add_subplot(gs[1, 1])
x_g = np.arange(len(SENS_GROUPS))
BAR_W_G = 0.30
PAIR_OFFSET = np.array([-BAR_W_G / 2 - 0.02, BAR_W_G / 2 + 0.02])

for gidx, gid in enumerate(SENS_GROUPS):
    comp = sens_concordance[gid]
    w_mean, w_lo, w_hi = comp["within"]
    ax_g.bar(x_g[gidx] + PAIR_OFFSET[0], w_mean, BAR_W_G,
             color=OI["blue"], edgecolor="white", linewidth=0.3, zorder=3,
             label="Within-paraphrase" if gidx == 0 else "")
    if not np.isnan(w_lo):
        ax_g.errorbar(x_g[gidx] + PAIR_OFFSET[0], w_mean,
                      yerr=[[w_mean - w_lo], [w_hi - w_mean]],
                      fmt="none", ecolor="#333333", elinewidth=0.5,
                      capsize=1.5, capthick=0.4, zorder=4)
    c_mean, c_lo, c_hi = comp["cross"]
    ax_g.bar(x_g[gidx] + PAIR_OFFSET[1], c_mean, BAR_W_G,
             color=OI["orange"], edgecolor="white", linewidth=0.3, zorder=3,
             label="Cross-paraphrase" if gidx == 0 else "")
    if not np.isnan(c_lo):
        ax_g.errorbar(x_g[gidx] + PAIR_OFFSET[1], c_mean,
                      yerr=[[c_mean - c_lo], [c_hi - c_mean]],
                      fmt="none", ecolor="#333333", elinewidth=0.5,
                      capsize=1.5, capthick=0.4, zorder=4)

ax_g.set_xticks(x_g)
ax_g.set_xticklabels([SENS_GROUP_DISPLAY[g] for g in SENS_GROUPS], fontsize=5)
ax_g.set_ylabel("Param. concordance (%)")
ax_g.set_ylim(0, 115)
ax_g.set_yticks([0, 25, 50, 75, 100])
ax_g.axhline(y=100, color="#dddddd", linewidth=0.3, zorder=1)
ax_g.legend(loc="lower left", fontsize=4, frameon=False)
add_panel_label(ax_g, "g")

# Panel (h): Cross-system tool selection accuracy
ax_h = fig.add_subplot(gs[1, 2])
x_h = np.arange(len(CS_CONDITIONS))
for i, model in enumerate(MODELS):
    means = [cs_accuracy[(c, model)][0] for c in CS_CONDITIONS]
    ci_lo = [cs_accuracy[(c, model)][0] - cs_accuracy[(c, model)][1] for c in CS_CONDITIONS]
    ci_hi = [cs_accuracy[(c, model)][2] - cs_accuracy[(c, model)][0] for c in CS_CONDITIONS]
    ax_h.bar(x_h + OFFSETS_3[i], means, BAR_W_3,
             color=MODEL_COLORS[model], edgecolor="white", linewidth=0.3, zorder=3)
    ax_h.errorbar(x_h + OFFSETS_3[i], means, yerr=[ci_lo, ci_hi],
                  fmt="none", ecolor="#333333", elinewidth=0.5,
                  capsize=1.2, capthick=0.4, zorder=4)

ax_h.set_xticks(x_h)
ax_h.set_xticklabels([CS_CONDITION_LABELS[c] for c in CS_CONDITIONS], fontsize=5)
ax_h.set_ylabel("Tool accuracy (%)")
ax_h.set_ylim(0, 115)
ax_h.set_yticks([0, 25, 50, 75, 100])
ax_h.axhline(y=100, color="#dddddd", linewidth=0.3, zorder=1)
ax_h.axvline(x=2.5, color="#cccccc", linewidth=0.4, linestyle="--", zorder=1)
ax_h.text(1.0, -0.28, "ChatSpatial", ha="center", va="top",
          fontsize=4.5, transform=ax_h.get_xaxis_transform(), color="#555555")
ax_h.text(3.5, -0.28, "Competitors", ha="center", va="top",
          fontsize=4.5, transform=ax_h.get_xaxis_transform(), color="#555555")
add_panel_label(ax_h, "h")

# ═══════════════════════════════════════════════════════════════
# Row 4: Cross-system consistency + E2E benchmark
# ═══════════════════════════════════════════════════════════════

# Panel (i): Cross-system tool consistency
ax_i = fig.add_subplot(gs[2, 0])
for i, model in enumerate(MODELS):
    means = [cs_consistency[(c, model)][0] for c in CS_CONDITIONS]
    ci_lo = [cs_consistency[(c, model)][0] - cs_consistency[(c, model)][1] for c in CS_CONDITIONS]
    ci_hi = [cs_consistency[(c, model)][2] - cs_consistency[(c, model)][0] for c in CS_CONDITIONS]
    ax_i.bar(x_h + OFFSETS_3[i], means, BAR_W_3,
             color=MODEL_COLORS[model], edgecolor="white", linewidth=0.3, zorder=3)
    ax_i.errorbar(x_h + OFFSETS_3[i], means, yerr=[ci_lo, ci_hi],
                  fmt="none", ecolor="#333333", elinewidth=0.5,
                  capsize=1.2, capthick=0.4, zorder=4)

ax_i.set_xticks(x_h)
ax_i.set_xticklabels([CS_CONDITION_LABELS[c] for c in CS_CONDITIONS], fontsize=5)
ax_i.set_ylabel("Consistency (%)")
ax_i.set_ylim(0, 115)
ax_i.set_yticks([0, 25, 50, 75, 100])
ax_i.axhline(y=100, color="#dddddd", linewidth=0.3, zorder=1)
ax_i.axvline(x=2.5, color="#cccccc", linewidth=0.4, linestyle="--", zorder=1)
ax_i.text(1.0, -0.28, "ChatSpatial", ha="center", va="top",
          fontsize=4.5, transform=ax_i.get_xaxis_transform(), color="#555555")
ax_i.text(3.5, -0.28, "Competitors", ha="center", va="top",
          fontsize=4.5, transform=ax_i.get_xaxis_transform(), color="#555555")
add_panel_label(ax_i, "i")

# Panel (j): E2E execution success
ax_j = fig.add_subplot(gs[2, 1])
x_j = np.arange(len(E2E_TASKS))
for i, sys in enumerate(E2E_SYSTEMS):
    offset = (i - 1) * (BAR_W_3 + 0.04)
    ok_vals = [e2e_counts[(sys, t)]["ok"] / e2e_counts[(sys, t)]["total"] * 100
               for t in E2E_TASKS]
    ax_j.bar(x_j + offset, ok_vals, BAR_W_3,
             color=SYSTEM_COLORS[sys], edgecolor="white", linewidth=0.3,
             label=SYSTEM_LABELS[sys], zorder=3)
    for j, t in enumerate(E2E_TASKS):
        c = e2e_counts[(sys, t)]
        ax_j.text(x_j[j] + offset, ok_vals[j] + 2,
                  f"{c['ok']}/{c['total']}", ha="center", va="bottom",
                  fontsize=4, color="#333333")

ax_j.set_xticks(x_j)
ax_j.set_xticklabels([E2E_TASK_LABELS[t] for t in E2E_TASKS])
ax_j.set_ylabel("Execution success (%)")
ax_j.set_ylim(0, 118)
ax_j.set_yticks([0, 25, 50, 75, 100])
ax_j.axhline(y=100, color="#dddddd", linewidth=0.3, zorder=1)
ax_j.legend(fontsize=4, frameon=False, loc="lower right")
add_panel_label(ax_j, "j")

# Panel (k): E2E pairwise concordance (strip plot)
ax_k = fig.add_subplot(gs[2, 2])
JITTER_W = 0.18
for j, task in enumerate(E2E_TASKS):
    for i, sys in enumerate(E2E_SYSTEMS):
        offset = (i - 1) * (JITTER_W * 2 + 0.04)
        pairs = e2e_concordance.get((sys, task), [])

        if len(pairs) == 0:
            ax_k.text(j + offset, 0.05, "n = 1", ha="center", va="bottom",
                      fontsize=4.5, color="#999999", style="italic")
            continue

        rng = np.random.default_rng(42 + i * 10 + j)
        jitter = rng.uniform(-JITTER_W * 0.5, JITTER_W * 0.5, len(pairs))
        xs = np.full(len(pairs), j + offset) + jitter

        ax_k.scatter(xs, pairs, s=12, c=SYSTEM_COLORS[sys],
                     alpha=0.6, edgecolors="none", zorder=4)

        mean_val = np.mean(pairs)
        ax_k.plot([j + offset - JITTER_W * 0.7, j + offset + JITTER_W * 0.7],
                  [mean_val, mean_val], color=SYSTEM_COLORS[sys],
                  linewidth=1.2, zorder=5)

        if mean_val < 0.99:
            if sys == "chatspatial" and task == "svg_detection":
                high = [p for p in pairs if p > 0.5]
                low = [p for p in pairs if p <= 0.5]
                if high and low:
                    ax_k.text(j + offset + JITTER_W * 0.8, 0.95, "within\nmethod",
                              ha="left", va="top", fontsize=3.2,
                              color=SYSTEM_COLORS[sys], linespacing=0.85)
                    ax_k.text(j + offset + JITTER_W * 0.8, 0.17, "across\nmethod",
                              ha="left", va="bottom", fontsize=3.2,
                              color=SYSTEM_COLORS[sys], linespacing=0.85)
                    continue
            ax_k.text(j + offset, mean_val + 0.04, f"{mean_val:.2f}",
                      ha="center", va="bottom", fontsize=3.8,
                      color=SYSTEM_COLORS[sys])

ax_k.text(0, 1.08, "ARI", ha="center", fontsize=5, color="#666666")
ax_k.text(1, 1.08, "Jaccard@100", ha="center", fontsize=5, color="#666666")
ax_k.text(2, 1.08, "Jaccard@50", ha="center", fontsize=5, color="#666666")
ax_k.set_xticks(range(len(E2E_TASKS)))
ax_k.set_xticklabels([E2E_TASK_LABELS[t] for t in E2E_TASKS])
ax_k.set_ylabel("Pairwise concordance")
ax_k.set_ylim(-0.05, 1.15)
ax_k.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
ax_k.axhline(y=1.0, color="#dddddd", linewidth=0.3, zorder=1)
add_panel_label(ax_k, "k")

# ═══════════════════════════════════════════════════════════════
# Row 1: DLPFC ground-truth benchmark (12 sections, Claude Sonnet 5.5,
# ChatSpatial 1.5.9; experiments/refusals_and_head_to_head/results/h2h_v159)
# ═══════════════════════════════════════════════════════════════
H2H_DIR = REPO_DIR / "experiments" / "refusals_and_head_to_head" / "results" / "h2h_v159"
h2h_runs = pd.read_csv(H2H_DIR / "h2h_per_run.csv", dtype={"section": str})
h2h_xrep = pd.read_csv(H2H_DIR / "h2h_cross_rep.csv", dtype={"section": str})
H2H_SYSTEMS = ["chatspatial", "chatspatial_single", "stagent", "spatialagent"]
H2H_COLORS = {**SYSTEM_COLORS, "chatspatial_single": OI["skyblue"]}
H2H_LABELS = {
    "chatspatial": "ChatSpatial\n(agent)",
    "chatspatial_single": "ChatSpatial\n(single call)",
    "stagent": "STAgent",
    "spatialagent": "SpatialAgent",
}
section_means = (h2h_runs.groupby(["system", "section"]).ari.mean().reset_index())

# Panel (a): per-section mean ARI for each system (spans 2 columns)
ax_a = fig.add_subplot(gs_top[0, 0:2])
JITTER_W_A = 0.28
for i, sys in enumerate(H2H_SYSTEMS):
    vals = section_means[section_means["system"] == sys]["ari"].values
    rng = np.random.default_rng(42 + i)
    xs = i + rng.uniform(-JITTER_W_A * 0.5, JITTER_W_A * 0.5, len(vals))
    ax_a.scatter(xs, vals, s=14, c=H2H_COLORS[sys], alpha=0.7,
                 edgecolors="none", zorder=4)
    mean_val = vals.mean()
    ax_a.plot([i - JITTER_W_A * 0.85, i + JITTER_W_A * 0.85],
              [mean_val, mean_val], color=H2H_COLORS[sys],
              linewidth=1.5, zorder=5)
    ax_a.text(i + JITTER_W_A * 0.9, mean_val, f"{mean_val:.3f}", ha="left",
              va="center", fontsize=4.5, color=H2H_COLORS[sys],
              fontweight="bold")
ax_a.set_xticks(range(len(H2H_SYSTEMS)))
ax_a.set_xticklabels([H2H_LABELS[s] for s in H2H_SYSTEMS])
ax_a.set_ylabel("ARI (ground truth)")
ax_a.set_ylim(0.2, 0.65)
ax_a.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6])
# Align label "a" with the labels of the single-column panels below it
add_panel_label(ax_a, "a",
                x=-0.12 * ax_c.get_position().width / ax_a.get_position().width)

# Panel (b): accuracy against reproducibility (bootstrap over sections)
ax_b = fig.add_subplot(gs_top[0, 2])
rng_b = np.random.default_rng(42)
b_rows = []
for sys in H2H_SYSTEMS:
    acc = section_means[section_means["system"] == sys].set_index("section")["ari"]
    rep = h2h_xrep[h2h_xrep["system"] == sys].set_index("section")["xrep_ari"]
    secs = acc.index.values
    boots_a, boots_r = [], []
    for _ in range(10_000):
        r = rng_b.choice(secs, len(secs), replace=True)
        boots_a.append(acc.loc[r].mean())
        boots_r.append(rep.loc[r].mean())
    a, c = acc.mean(), rep.mean()
    a_lo, a_hi = np.percentile(boots_a, [2.5, 97.5])
    c_lo, c_hi = np.percentile(boots_r, [2.5, 97.5])
    b_rows.append({"system": sys, "n_sections": len(secs),
                   "mean_ari": a, "mean_ari_ci_lo": a_lo, "mean_ari_ci_hi": a_hi,
                   "xrep_ari": c, "xrep_ari_ci_lo": c_lo, "xrep_ari_ci_hi": c_hi})
    ax_b.errorbar(a, c, xerr=[[a - a_lo], [a_hi - a]],
                  yerr=[[c - c_lo], [c_hi - c]], fmt="o", ms=6,
                  color=H2H_COLORS[sys], ecolor=H2H_COLORS[sys],
                  elinewidth=0.8, capsize=2.5, capthick=0.6, zorder=4)
    ax_b.text(a + 0.006, c + 0.025, H2H_LABELS[sys].replace("\n", " "),
              fontsize=4.5, color=H2H_COLORS[sys], ha="left", va="bottom")
ax_b.axhline(y=1.0, color="#dddddd", linewidth=0.3, zorder=1)
ax_b.set_xlabel("Mean ARI (ground truth)")
ax_b.set_ylabel("Cross-replicate ARI")
ax_b.set_xlim(0.32, 0.52)
ax_b.set_ylim(0.4, 1.12)
ax_b.set_xticks([0.35, 0.40, 0.45, 0.50])
ax_b.set_yticks([0.4, 0.6, 0.8, 1.0])
add_panel_label(ax_b, "b")

# Source data for panels a and b, written from the values plotted above
h2h_runs[h2h_runs["system"].isin(H2H_SYSTEMS)][
    ["system", "section", "rep", "finished", "timed_out", "verified",
     "method", "ari", "nmi", "n_scored"]
].sort_values(["system", "section", "rep"]).to_csv(
    H2H_DIR / "fig2ab_trials.csv", index=False)
(section_means
 .merge(h2h_runs.groupby(["system", "section"]).ari.count()
        .rename("n_scored_runs").reset_index(), on=["system", "section"])
 .merge(h2h_xrep, on=["system", "section"], how="left")
 .rename(columns={"ari": "mean_ari", "n_pairs": "xrep_n_pairs"})
 .query("system in @H2H_SYSTEMS")
 .sort_values(["system", "section"])
 .to_csv(H2H_DIR / "fig2ab_sections.csv", index=False))
pd.DataFrame(b_rows).to_csv(H2H_DIR / "fig2b_summary.csv", index=False)

# ── Save ─────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    fig.savefig(FIGURE_DIR / f"figure4.{ext}",
                dpi=300, facecolor="white")
    print(f"Saved: figure4.{ext}")

plt.close(fig)
