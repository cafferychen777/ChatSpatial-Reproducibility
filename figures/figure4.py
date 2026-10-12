"""
Figure 2 (file name figure4): DLPFC head-to-head, schema ablation, tool
selection and end-to-end system comparison. Direction A ("Ink & Cobalt"),
authored at 180 mm (place at \\textwidth); minimum font size 5 pt.

Row 1 — DLPFC ground-truth benchmark (Claude Sonnet 5.5):
  [a] Per-section ARI and system mean with bootstrap 95% CI
  [b] Mean ground-truth ARI versus cross-replicate ARI
Row 2 — Schema-enforcement ablation (Full / Bare / No schema, cobalt ramp):
  [c] Pydantic validation rate per schema level (mean bar, model points)
  [d] End-to-end execution success (3 tasks x 3 schema levels)
  [e] Cross-model output concordance (survivor-bias condition hatched)
Row 3 — Tool selection and end-to-end comparison:
  [f] Tool selection accuracy across paraphrases and system contexts
  [g] End-to-end runs completed, human lymph node (3 systems x 3 tasks)
  [h] Pairwise output concordance among completed replicates

Also writes Supplementary Figure (supp_fig_prompt_consistency.pdf/png):
  [a] Constrained parameter concordance, within vs across paraphrases
  [b] Cross-replicate tool consistency across system contexts

Source data: panels a and b in experiments/refusals_and_head_to_head/results/h2h_v159/ (fig2ab_trials.csv,
fig2ab_sections.csv, fig2b_summary.csv); panels f-h and the Supplementary
Figure in source_data/ (Fig2f_tool_selection.csv, Fig2g_runs_completed.csv,
Fig2h_pairwise_concordance.csv, SuppFig1a_param_concordance.csv,
SuppFig1b_tool_consistency.csv).
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from figure_style import (
    COBALT,
    COBALT_LIGHT,
    COBALT_MID,
    DOUBLE_COL,
    HAIRLINE,
    INK,
    INK_SOFT,
    MODEL_COLORS as STYLE_MODEL_COLORS,
    MODEL_MARKERS as STYLE_MODEL_MARKERS,
    PANEL_LABEL_FONT,
    SCHEMA_COLORS,
    SYSTEM_COLORS as STYLE_SYSTEM_COLORS,
    SYSTEM_FILL as STYLE_SYSTEM_FILL,
    SYSTEM_MARKERS as STYLE_SYSTEM_MARKERS,
    apply_style,
)

apply_style()
plt.rcParams["savefig.bbox"] = "standard"  # keep the exact 180 mm canvas

FS_TICK = 6.0
FS_LABEL = 6.5
FS_ANNOT = 5.5
FS_SMALL = 5.0

# ── Paths ────────────────────────────────────────────────────────
FIGURE_DIR = Path(__file__).resolve().parent
REPO_DIR = FIGURE_DIR.parent
# Rows 2 and 3 and the Supplementary Figure: round-1 experiments; row 1: DLPFC head-to-head (ChatSpatial 1.5.9)
DATA_DIR = REPO_DIR / "experiments" / "round1" / "results"
ABL_DATA_DIR = DATA_DIR / "ablation"
CROSS_METRICS_CSV = DATA_DIR / "cross_system" / "analysis" / "cross_system_metrics.csv"
E2E_RAW_PATH = DATA_DIR / "e2e_benchmark" / "e2e_benchmark_raw.jsonl"
E2E_PAIRWISE_CSV = DATA_DIR / "e2e_benchmark" / "e2e_pairwise_concordance.csv"
SENSITIVITY_DIR = DATA_DIR / "sensitivity"
ABLATION_INV_DIR = DATA_DIR / "ablation" / "invocation"
DLPFC_DATA_DIR = DATA_DIR / "dlpfc_benchmark"
DLPFC_PER_TRIAL = DLPFC_DATA_DIR / "dlpfc_benchmark_per_trial.csv"
DLPFC_RESULTS = DLPFC_DATA_DIR / "dlpfc_benchmark_results.csv"

# ── Entities and colours (all from figure_style, Direction A) ────
ABL_CONDITIONS = ["full_schema", "bare_schema", "no_schema"]
ABL_COND_COLORS = {"full_schema": SCHEMA_COLORS["Full"],
                   "bare_schema": SCHEMA_COLORS["Bare"],
                   "no_schema": SCHEMA_COLORS["None"]}
ABL_COND_LABELS = {"full_schema": "Full schema", "bare_schema": "Bare schema",
                   "no_schema": "No schema"}

MODEL_LABELS = {"gemini-2.5-flash": "Gemini 2.5 Flash",
                "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
                "gpt-5.4": "GPT-5.4"}
MODEL_COLORS = {m: STYLE_MODEL_COLORS[l] for m, l in MODEL_LABELS.items()}
MODEL_MARKERS = {m: STYLE_MODEL_MARKERS[l] for m, l in MODEL_LABELS.items()}
# Light-grey markers get a slightly darker rim so they stay visible on white
MODEL_EDGES = {m: ("#8F8F8F" if MODEL_COLORS[m].upper() == "#B5B5B5" else MODEL_COLORS[m])
               for m in MODEL_LABELS}
ABL_MODELS = list(MODEL_LABELS)

SYS_NAME = {"chatspatial": "ChatSpatial agent", "chatspatial_single": "ChatSpatial single call",
            "stagent": "STAgent", "spatialagent": "SpatialAgent"}
SYSTEM_COLORS = {k: STYLE_SYSTEM_COLORS[v] for k, v in SYS_NAME.items()}
SYSTEM_FILL = {k: STYLE_SYSTEM_FILL[v] for k, v in SYS_NAME.items()}
SYSTEM_MARKERS = {k: STYLE_SYSTEM_MARKERS[v] for k, v in SYS_NAME.items()}
SYSTEM_LABELS = {"chatspatial": "ChatSpatial", "stagent": "STAgent",
                 "spatialagent": "SpatialAgent"}

ABL_TASK_DISPLAY = {"spatial_domains": "Spatial\ndomains",
                    "svg_detection": "SVG\ndetection",
                    "deconvolution": "Deconvo-\nlution"}
ABL_TASKS = list(ABL_TASK_DISPLAY)
ABL_CONC_METRIC_LABEL = {"spatial_domains": "ARI", "svg_detection": "Jaccard@100",
                         "deconvolution": "Pearson $r$"}

CS_CONDITIONS = ["full_schema", "bare_schema", "no_schema",
                 "stagent_context", "spatialagent_context"]
CS_CONDITION_LABELS = {"full_schema": "ChatSpatial, full schema",
                       "bare_schema": "ChatSpatial, bare schema",
                       "no_schema": "ChatSpatial, no schema",
                       "stagent_context": "STAgent context",
                       "spatialagent_context": "SpatialAgent context"}
CS_CONDITION_COLORS = {"full_schema": COBALT, "bare_schema": COBALT,
                       "no_schema": COBALT, "stagent_context": SYSTEM_COLORS["stagent"],
                       "spatialagent_context": SYSTEM_COLORS["spatialagent"]}

E2E_SYSTEMS = ["chatspatial", "stagent", "spatialagent"]
E2E_TASKS = ["spatial_domains", "svg_detection", "cell_communication"]
E2E_TASK_LABELS = {"spatial_domains": "Spatial domains",
                   "svg_detection": "SVG detection",
                   "cell_communication": "Cell communication"}
E2E_CONC_METRIC_LABEL = {"spatial_domains": "ARI", "svg_detection": "Jaccard@100",
                         "cell_communication": "Jaccard@50"}

DLPFC_SAMPLE_IDS = [151673, 151507, 151669]

SENS_GROUPS = ["spatial_domains", "deconvolution", "svg_detection",
               "cell_communication", "moran_i"]
SENS_GROUP_DISPLAY = {"spatial_domains": "Spatial domains", "deconvolution": "Deconvolution",
                      "svg_detection": "SVG detection",
                      "cell_communication": "Cell communication",
                      "moran_i": "Moran's $I$"}
SENS_GROUP_CONSTRAINED = {"spatial_domains": ["method"], "deconvolution": ["method"],
                          "svg_detection": ["method"],
                          "cell_communication": ["method", "species"],
                          "moran_i": ["analysis_type"]}
ABLATION_TO_GROUP = {"spatial_domains": "spatial_domains",
                     "deconvolution_rctd": "deconvolution",
                     "svg_detection": "svg_detection", "cellchat": "cell_communication",
                     "moran_i": "moran_i"}


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


# Pairwise replicate concordance (panel h), written by experiments/round1/scripts/e2e_pairwise_concordance.py
# from the benchmark outputs (not included: 5.7 GB); one row per pair of completed replicates.
METRIC_OF_TASK = {"spatial_domains": "ARI", "svg_detection": "Jaccard@100",
                  "cell_communication": "Jaccard@50"}
# Readable outputs where no pair exists: STAgent completed one spatially variable gene run and returned no
# ligand-receptor list that could be scored (the same script, run on the outputs).
E2E_OUTPUTS_WITHOUT_PAIRS = {("stagent", "svg_detection"): 1, ("stagent", "cell_communication"): 0}
_pairs = pd.read_csv(E2E_PAIRWISE_CSV)
e2e_concordance = {}
e2e_n_outputs = {}  # replicates with a readable output (for n labels)
e2e_pair_rows = []  # source data of panel h: one row per replicate pair
for sys in E2E_SYSTEMS:
    for task in E2E_TASKS:
        sel = _pairs[(_pairs["system"] == sys) & (_pairs["task"] == task)]
        e2e_concordance[(sys, task)] = sel["value"].astype(float).tolist()
        reps = set(sel["rep_a"]) | set(sel["rep_b"])
        e2e_n_outputs[(sys, task)] = len(reps) if reps else E2E_OUTPUTS_WITHOUT_PAIRS.get((sys, task), 0)
        e2e_pair_rows += [(sys, task, METRIC_OF_TASK[task], int(r.rep_a), int(r.rep_b), float(r.value))
                          for r in sel.itertuples()]

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


# --- DLPFC ground-truth benchmark ---
dlpfc_trials = pd.read_csv(DLPFC_PER_TRIAL)
dlpfc_results = pd.read_csv(DLPFC_RESULTS)


# ═════════════════════════════════════════════════════════════════
# Layout helpers (millimetres, measured from the top-left corner)
# ═════════════════════════════════════════════════════════════════
W_MM = DOUBLE_COL * 25.4  # 180 mm


def ax_mm(fig, x, y, w, h):
    """Axes whose top-left corner sits x mm from the left, y mm from the top."""
    fw, fh = fig.get_size_inches() * 25.4
    return fig.add_axes([x / fw, 1 - (y + h) / fh, w / fw, h / fh])


def text_mm(fig, x, y, s, **kw):
    fw, fh = fig.get_size_inches() * 25.4
    return fig.text(x / fw, 1 - y / fh, s, **kw)


def finish_axis(ax):
    ax.tick_params(axis="both", labelsize=FS_TICK, pad=1.5)
    ax.xaxis.label.set_size(FS_LABEL)
    ax.yaxis.label.set_size(FS_LABEL)


def panel_letters(fig, items, y_mm):
    """8 pt bold letters on one baseline, at the left edge of each panel's
    tight bounding box (tick labels and axis labels included)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fw_px = fig.bbox.width
    for ax, letter, dx in items:
        x0 = ax.get_tightbbox(r).x0 / fw_px
        fw, fh = fig.get_size_inches() * 25.4
        fig.text(max(x0 + dx / fw, 0.0), 1 - y_mm / fh, letter,
                 **{**PANEL_LABEL_FONT, "color": INK}, va="baseline", ha="left")


def marker_legend(ax, items, x, y, dx_list, fs=FS_ANNOT, ms=3.2):
    """One row of marker keys in axes coordinates; items = (label, kw)."""
    for (label, kw), dx in zip(items, dx_list):
        ax.plot([x + dx], [y], ls="none", transform=ax.transAxes, clip_on=False,
                ms=ms, **kw)
        ax.text(x + dx + 0.035, y, label, transform=ax.transAxes, fontsize=fs,
                va="center", ha="left", color=INK, clip_on=False)


def row_dots(ax, blocks, series, get_ci, colors, markers, edges=None,
             offset_step=0.24, ms=3.0, open_series=(), header_gap=0.35):
    """Horizontal dot plot with 95% CI whiskers.

    blocks: list of (header or None, [(key, row_label, label_colour), ...]).
    Returns the y positions of the header rows.
    """
    y = 0.0
    yt, yl, ycol, headers = [], [], [], []
    n = len(series)
    offs = (np.arange(n) - (n - 1) / 2) * offset_step
    for b, (header, rows) in enumerate(blocks):
        if b > 0:
            y -= header_gap
        if header is not None:
            headers.append(y)
            yt.append(y)
            yl.append(header)
            ycol.append(None)
            y -= 1.0
        for key, label, lcol in rows:
            for s, off in zip(series, offs):
                m, lo, hi = get_ci(key, s)
                yy = y - off
                if not np.isnan(lo):
                    ax.plot([lo, hi], [yy, yy], color=colors[s], lw=0.7,
                            solid_capstyle="butt", zorder=3)
                filled = s not in open_series
                ec = (edges or colors)[s]
                ax.plot([m], [yy], marker=markers[s], ms=ms, ls="none",
                        mfc=colors[s] if filled else "white", mec=ec,
                        mew=0.5 if filled else 0.7, zorder=4)
            yt.append(y)
            yl.append(label)
            ycol.append(lcol)
            y -= 1.0
    ax.set_yticks(yt)
    ax.set_yticklabels(yl)
    for t, c, yy in zip(ax.get_yticklabels(), ycol, yt):
        if c is None:
            t.set_color(INK)
            t.set_fontweight("bold")
        else:
            t.set_color(c)
    ax.set_ylim(y + 0.4, 0.6)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=2)
    return headers


# ═════════════════════════════════════════════════════════════════
# Figure 2
# ═════════════════════════════════════════════════════════════════
H_MM = 206.0
fig = plt.figure(figsize=(DOUBLE_COL, H_MM / 25.4))

# Row tops / heights (mm from the top)
R1_Y, R1_H = 8.0, 54.0
R2_Y, R2_H = 84.0, 36.0
R3_Y, R3_H = 143.0, 51.0

# ── Head-to-head data and bootstrap summary (needed by a and b) ──
H2H_DIR = REPO_DIR / "experiments" / "refusals_and_head_to_head" / "results" / "h2h_v159"
h2h_runs = pd.read_csv(H2H_DIR / "h2h_per_run.csv", dtype={"section": str})
h2h_xrep = pd.read_csv(H2H_DIR / "h2h_cross_rep.csv", dtype={"section": str})
H2H_SYSTEMS = ["chatspatial", "chatspatial_single", "stagent", "spatialagent"]
section_means = (h2h_runs.groupby(["system", "section"]).ari.mean().reset_index())

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
b_summary = pd.DataFrame(b_rows).set_index("system")

# ── Panel a: per-section ARI, mean ± 95% CI ─────────────────────
ax_a = ax_mm(fig, 14, R1_Y, 92, R1_H)
A_TICK = {"chatspatial": "ChatSpatial\nagent", "chatspatial_single": "ChatSpatial\nsingle call",
          "stagent": "STAgent", "spatialagent": "SpatialAgent"}
PT_X = -0.13      # centre of the section points, relative to the category
MEAN_X = 0.20     # position of the mean and CI
sec_x = {}
for i, sys in enumerate(H2H_SYSTEMS):
    d = section_means[section_means["system"] == sys].sort_values("ari")
    v = d["ari"].values
    # deterministic side-by-side offsets so close values do not overlap
    off = ((np.arange(len(v)) % 4) - 1.5) * 0.055
    xs = i + PT_X + off
    for s, xx in zip(d["section"].values, xs):
        sec_x[(sys, s)] = xx
    ax_a.scatter(xs, v, s=13, marker=SYSTEM_MARKERS[sys], facecolor=SYSTEM_FILL[sys],
                 edgecolor=SYSTEM_COLORS[sys],
                 linewidth=0.6 if SYSTEM_FILL[sys] == "#FFFFFF" else 0.3,
                 alpha=0.9, zorder=3)
    m, lo, hi = b_summary.loc[sys, ["mean_ari", "mean_ari_ci_lo", "mean_ari_ci_hi"]]
    xm = i + MEAN_X
    ax_a.plot([xm, xm], [lo, hi], color=SYSTEM_COLORS[sys], lw=1.0,
              solid_capstyle="butt", zorder=4)
    ax_a.plot([xm], [m], marker=SYSTEM_MARKERS[sys], ms=5.6, mfc=SYSTEM_FILL[sys],
              mec=SYSTEM_COLORS[sys], mew=1.0, zorder=5)
    ax_a.text(xm + 0.075, m, f"{m:.3f}", fontsize=FS_ANNOT + 0.5, va="center",
              ha="left", color=SYSTEM_COLORS[sys] if sys != "chatspatial_single" else COBALT,
              fontweight="bold" if sys == "chatspatial" else "normal")
ax_a.set_xlim(-0.5, len(H2H_SYSTEMS) - 0.38)
ax_a.set_ylim(0.25, 0.60)
ax_a.set_yticks([0.3, 0.4, 0.5, 0.6])
ax_a.set_xticks(range(len(H2H_SYSTEMS)))
ax_a.set_xticklabels([A_TICK[s] for s in H2H_SYSTEMS])
for t, s in zip(ax_a.get_xticklabels(), H2H_SYSTEMS):
    t.set_color(SYSTEM_COLORS[s])
    t.set_fontsize(6.5)
    if s == "chatspatial":
        t.set_fontweight("bold")
ax_a.tick_params(axis="x", length=0, pad=3)
ax_a.set_ylabel("ARI against expert layers (per section)")
finish_axis(ax_a)
ax_a.tick_params(axis="x", labelsize=6.5)

# ── Panel b: accuracy versus reproducibility ─────────────────────
ax_b = ax_mm(fig, 125, R1_Y, 53, R1_H)
B_LABEL = {  # (text, dx, dy, ha, va) in data units relative to the point
    "chatspatial": ("ChatSpatial agent", 0.004, 0.072, "center", "bottom"),
    "chatspatial_single": ("ChatSpatial single call", 0.0, -0.050, "center", "top"),
    "stagent": ("STAgent", 0.0, -0.045, "center", "top"),
    "spatialagent": ("SpatialAgent", 0.0, 0.050, "center", "bottom"),
}
for sys in H2H_SYSTEMS:
    r = b_summary.loc[sys]
    col = SYSTEM_COLORS[sys]
    ax_b.plot([r.mean_ari_ci_lo, r.mean_ari_ci_hi], [r.xrep_ari] * 2, color=col,
              lw=0.8, solid_capstyle="butt", zorder=2)
    ax_b.plot([r.mean_ari] * 2, [r.xrep_ari_ci_lo, r.xrep_ari_ci_hi], color=col,
              lw=0.8, solid_capstyle="butt", zorder=2)
    ax_b.plot(r.mean_ari, r.xrep_ari, marker=SYSTEM_MARKERS[sys], ms=5.6,
              mfc=SYSTEM_FILL[sys], mec=col, mew=1.0, zorder=3)
    txt, dx, dy, ha, va = B_LABEL[sys]
    anchor = r.xrep_ari_ci_hi if dy > 0 else r.xrep_ari_ci_lo
    if sys == "chatspatial_single":
        anchor = r.xrep_ari
    ax_b.text(r.mean_ari + dx, anchor + dy * 0.45, txt, fontsize=6.0, color=col,
              ha=ha, va=va, fontweight="bold" if sys == "chatspatial" else "normal",
              zorder=6)
    # cross-replicate value, directly right of the marker
    ax_b.text(r.mean_ari + 0.0065, r.xrep_ari + 0.008, f"{r.xrep_ari:.3f}",
              fontsize=FS_SMALL + 0.5, color=INK_SOFT, ha="left", va="bottom", zorder=6)
ax_b.axhline(1.0, color=HAIRLINE, lw=0.4, ls=(0, (2, 2)), zorder=1)
ax_b.set_xlim(0.31, 0.52)
ax_b.set_ylim(0.45, 1.06)
ax_b.set_xticks([0.35, 0.40, 0.45, 0.50])
ax_b.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax_b.set_xlabel("Mean ARI against expert layers")
ax_b.set_ylabel("Cross-replicate ARI")
finish_axis(ax_b)

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

# ── Row 2: schema-enforcement ablation ──────────────────────────
R2_W = 45.0
R2_X = [14.0, 74.0, 133.0]
BW = 0.25
OFF3 = np.array([-(BW + 0.02), 0.0, BW + 0.02])


def schema_bars(ax, x, heights, cond, hatches=None):
    for xx, hh, ht in zip(np.atleast_1d(x), np.atleast_1d(heights),
                          hatches or [False] * len(np.atleast_1d(x))):
        ax.bar(xx, hh, BW, color=ABL_COND_COLORS[cond],
               edgecolor=COBALT_MID if ht else "none", linewidth=0.4 if ht else 0,
               hatch="//////" if ht else None, zorder=3)


# Panel c: validation rate per schema level; bar = mean over models,
# markers = individual models
ax_c = ax_mm(fig, R2_X[0], R2_Y, R2_W, R2_H)
m_off = np.array([-0.17, 0.0, 0.17])
for i, cond in enumerate(ABL_CONDITIONS):
    ax_c.bar(i, pydantic_means[cond], 0.62, color=ABL_COND_COLORS[cond],
             edgecolor="none", zorder=2)
    vals = [pydantic_data[(cond, m)] for m in ABL_MODELS]
    for v, m, dx in zip(vals, ABL_MODELS, m_off):
        ax_c.plot([i + dx], [v], marker=MODEL_MARKERS[m], ms=3.0, ls="none",
                  mfc=MODEL_COLORS[m], mec="white" if cond != "no_schema" else MODEL_EDGES[m],
                  mew=0.35, zorder=4)
    top = max(max(vals), pydantic_means[cond])
    ax_c.text(i, top + 3.0, f"{pydantic_means[cond]:.1f}", ha="center", va="bottom",
              fontsize=FS_ANNOT, color=INK)
ax_c.set_xticks(range(3))
ax_c.set_xticklabels(["Full\nschema", "Bare\nschema", "No\nschema"])
ax_c.set_xlim(-0.55, 2.55)
ax_c.set_ylim(0, 112)
ax_c.set_yticks([0, 25, 50, 75, 100])
ax_c.set_ylabel("Calls passing validation (%)")
finish_axis(ax_c)
ax_c.tick_params(axis="x", length=0)
# model key in the empty upper right of c
for k, m in enumerate(ABL_MODELS):
    yy = 0.86 - k * 0.105
    ax_c.plot([0.735], [yy], marker=MODEL_MARKERS[m], ms=3.0, ls="none",
              mfc=MODEL_COLORS[m], mec=MODEL_EDGES[m], mew=0.35,
              transform=ax_c.transAxes, clip_on=False)
    ax_c.text(0.775, yy, MODEL_LABELS[m].replace(" ", " "), fontsize=FS_SMALL,
              color=INK, va="center", ha="left", transform=ax_c.transAxes)
ax_c.text(0.715, 0.97, "Model", fontsize=FS_SMALL, color=INK_SOFT, va="center",
          ha="left", transform=ax_c.transAxes)

# Panel d: execution success (numbers = successful trials of 30)
ax_d = ax_mm(fig, R2_X[1], R2_Y, R2_W, R2_H)
x3 = np.arange(len(ABL_TASKS))
for i, cond in enumerate(ABL_CONDITIONS):
    vals = [abl_exec_data[(cond, t)] for t in ABL_TASKS]
    schema_bars(ax_d, x3 + OFF3[i], vals, cond)
    for j, t in enumerate(ABL_TASKS):
        n_s, n_t = abl_exec_n[(cond, t)]
        ax_d.text(x3[j] + OFF3[i], vals[j] + 2.0, f"{n_s}", ha="center", va="bottom",
                  fontsize=FS_SMALL, color=INK_SOFT)
ax_d.set_xticks(x3)
ax_d.set_xticklabels([ABL_TASK_DISPLAY[t] for t in ABL_TASKS])
ax_d.set_xlim(-0.55, len(ABL_TASKS) - 0.45)
ax_d.set_ylim(0, 112)
ax_d.set_yticks([0, 25, 50, 75, 100])
ax_d.set_ylabel("Execution success (%)")
finish_axis(ax_d)
ax_d.tick_params(axis="x", length=0)

# Panel e: cross-model output concordance (hatched: execution < 50%)
ax_e = ax_mm(fig, R2_X[2], R2_Y, R2_W, R2_H)
for i, cond in enumerate(ABL_CONDITIONS):
    means, lo_e, hi_e, hatches = [], [], [], []
    for t in ABL_TASKS:
        m, lo, hi, n = abl_conc_data[(cond, t)]
        means.append(m if not np.isnan(m) else 0)
        lo_e.append(m - lo if not np.isnan(lo) else 0)
        hi_e.append(hi - m if not np.isnan(hi) else 0)
        n_s, n_t = abl_exec_n.get((cond, t), (0, 30))
        hatches.append(n_s / n_t < 0.50 if n_t > 0 else False)
    schema_bars(ax_e, x3 + OFF3[i], means, cond, hatches)
    ax_e.errorbar(x3 + OFF3[i], means, yerr=[lo_e, hi_e], fmt="none", ecolor=INK,
                  elinewidth=0.6, capsize=0, zorder=4)
ax_e.set_xticks(x3)
ax_e.set_xticklabels([f"{ABL_TASK_DISPLAY[t]}\n({ABL_CONC_METRIC_LABEL[t]})"
                      for t in ABL_TASKS])
ax_e.set_xlim(-0.55, len(ABL_TASKS) - 0.45)
ax_e.set_ylim(0, 1.12)
ax_e.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax_e.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
ax_e.set_ylabel("Cross-model concordance")
finish_axis(ax_e)
ax_e.tick_params(axis="x", length=0)

# Schema key above d and e (c carries its levels on the x axis)
key_y = R2_Y - 4.2
kx = R2_X[1] - 2.0
for cond in ABL_CONDITIONS:
    fig.add_artist(Rectangle(((kx) / W_MM, 1 - (key_y + 1.1) / H_MM), 2.6 / W_MM,
                             2.2 / H_MM, transform=fig.transFigure,
                             fc=ABL_COND_COLORS[cond], ec="none"))
    t = text_mm(fig, kx + 3.4, key_y, ABL_COND_LABELS[cond], fontsize=6.0,
                color=INK, va="center", ha="left")
    fig.canvas.draw()
    kx += 3.4 + t.get_window_extent().width / fig.dpi * 25.4 + 4.0
fig.add_artist(Rectangle((kx / W_MM, 1 - (key_y + 1.1) / H_MM), 2.6 / W_MM, 2.2 / H_MM,
                         transform=fig.transFigure, fc=COBALT_LIGHT, ec=COBALT_MID,
                         lw=0.4, hatch="//////"))
text_mm(fig, kx + 3.4, key_y, "Execution < 50% (survivor bias)",
        fontsize=6.0, color=INK, va="center", ha="left")

# ── Row 3 ────────────────────────────────────────────────────────
# Panel f: tool selection accuracy across paraphrases and system contexts
ax_f = ax_mm(fig, 41, R3_Y, 41, R3_H)
f_blocks = [
    ("Paraphrases (full schema)",
     [(("sens", g), SENS_GROUP_DISPLAY[g], INK_SOFT) for g in SENS_GROUPS]),
    ("System context",
     [(("cs", c), CS_CONDITION_LABELS[c], CS_CONDITION_COLORS[c]) for c in CS_CONDITIONS]),
]


def f_get(key, model):
    kind, k = key
    src = sens_tool_accuracy if kind == "sens" else cs_accuracy
    return src[(k, model)]


row_dots(ax_f, f_blocks, MODELS, f_get, MODEL_COLORS, MODEL_MARKERS, MODEL_EDGES,
         offset_step=0.27, ms=2.7)
ax_f.axvline(100, color=HAIRLINE, lw=0.4, zorder=1)
ax_f.set_xlim(38, 103)
ax_f.set_xticks([40, 60, 80, 100])
ax_f.set_xlabel("Tool selection accuracy (%)")
finish_axis(ax_f)
ax_f.tick_params(axis="y", length=0, labelsize=FS_TICK)

# model key above f
fx = 0.0
for m in MODELS:
    ax_f.plot([fx], [1.045], marker=MODEL_MARKERS[m], ms=3.0, ls="none",
              mfc=MODEL_COLORS[m], mec=MODEL_EDGES[m], mew=0.35,
              transform=ax_f.transAxes, clip_on=False)
    t = ax_f.text(fx + 0.04, 1.045, MODEL_LABELS[m], fontsize=FS_SMALL + 0.5, color=INK,
                  va="center", ha="left", transform=ax_f.transAxes)
    fig.canvas.draw()
    fx += 0.04 + t.get_window_extent().width / ax_f.bbox.width + 0.07

# Panels g and h share one row per system and task
g_rows = []  # (y, kind, task, sys)
y = 0.0
for b, task in enumerate(E2E_TASKS):
    if b > 0:
        y -= 0.45
    g_rows.append((y, "header", task, None))
    y -= 1.0
    for sys in E2E_SYSTEMS:
        g_rows.append((y, "sys", task, sys))
        y -= 1.0
Y_MIN = y + 0.45

G_X, G_W = 109.0, 19.0
H_X, H_W = 133.0, 37.5
ax_g = ax_mm(fig, G_X, R3_Y, G_W, R3_H)
ax_h = ax_mm(fig, H_X, R3_Y, H_W, R3_H)

for yy, kind, task, sys in g_rows:
    if kind == "header":
        continue
    cnt = e2e_counts[(sys, task)]
    for k in range(cnt["total"]):
        ok = k < cnt["ok"]
        ax_g.plot([k], [yy], marker="o", ms=3.4, ls="none",
                  mfc=SYSTEM_COLORS[sys] if ok else "white",
                  mec=SYSTEM_COLORS[sys] if ok else "#B0B0B0", mew=0.5, zorder=3)
    ax_g.text(cnt["total"] - 0.2, yy, f"{cnt['ok']}/{cnt['total']}", fontsize=FS_SMALL + 0.5,
              color=INK_SOFT, va="center", ha="left")
ax_g.set_yticks([r[0] for r in g_rows])
ax_g.set_yticklabels([E2E_TASK_LABELS[r[2]] if r[1] == "header" else SYSTEM_LABELS[r[3]]
                      for r in g_rows])
for t, r in zip(ax_g.get_yticklabels(), g_rows):
    if r[1] == "header":
        t.set_color(INK)
        t.set_fontweight("bold")
    else:
        t.set_color(SYSTEM_COLORS[r[3]])
ax_g.set_ylim(Y_MIN, 0.6)
ax_g.set_xlim(-0.6, 6.4)
ax_g.set_xticks([])
for sp in ("left", "bottom"):
    ax_g.spines[sp].set_visible(False)
ax_g.tick_params(axis="y", length=0, pad=2, labelsize=FS_TICK)
ax_g.set_xlabel("Runs completed", labelpad=14)
ax_g.xaxis.label.set_size(FS_LABEL)

# Panel h: pairwise concordance, one strip per system and task
for yy, kind, task, sys in g_rows:
    if kind == "header":
        ax_h.text(-0.01, yy, E2E_CONC_METRIC_LABEL[task], fontsize=FS_SMALL + 0.5,
                  color=INK_SOFT, va="center", ha="left")
        continue
    pairs = np.asarray(e2e_concordance.get((sys, task), []), float)
    if len(pairs) == 0:
        k = e2e_n_outputs.get((sys, task), 0)
        msg = "1 output, no pairs" if k == 1 else (
            "no scorable output" if k == 0 else f"{k} outputs")
        ax_h.text(0.5, yy, msg, fontsize=FS_SMALL, color=INK_SOFT, va="center",
                  ha="center", style="italic")
        continue
    rng = np.random.default_rng(7 + E2E_TASKS.index(task) * 10 + E2E_SYSTEMS.index(sys))
    jit = rng.uniform(-0.24, 0.24, len(pairs))
    ax_h.scatter(pairs, yy + jit, s=6, color=SYSTEM_COLORS[sys], alpha=0.75,
                 edgecolors="none", zorder=3)
    mv = pairs.mean()
    ax_h.plot([mv, mv], [yy - 0.36, yy + 0.36], color=INK, lw=1.0,
              solid_capstyle="butt", zorder=2)
    ax_h.text(1.07, yy, f"{mv:.2f}", fontsize=FS_SMALL + 0.5, color=INK, va="center",
              ha="left", clip_on=False)
ax_h.text(1.07, 0.6, "Mean", fontsize=FS_SMALL, color=INK_SOFT, va="bottom", ha="left",
          clip_on=False)
ax_h.set_ylim(Y_MIN, 0.6)
ax_h.set_xlim(-0.04, 1.04)
ax_h.set_xticks([0, 0.5, 1.0])
ax_h.set_xticklabels(["0", "0.5", "1"])
ax_h.set_yticks([])
ax_h.spines["left"].set_visible(False)
ax_h.axvline(1.0, color=HAIRLINE, lw=0.4, zorder=1)
ax_h.set_xlabel("Pairwise concordance between replicates")
finish_axis(ax_h)

# separators between task blocks, spanning g and h
sep_ys = [r[0] + 0.72 for r in g_rows if r[1] == "header"][1:]
for sy in sep_ys:
    for ax in (ax_g, ax_h):
        ax.axhline(sy, color=HAIRLINE, lw=0.4, zorder=1)

# ── Panel letters ───────────────────────────────────────────────
panel_letters(fig, [(ax_a, "a", 0), (ax_b, "b", 0)], 4.0)
panel_letters(fig, [(ax_c, "c", 0), (ax_d, "d", 0), (ax_e, "e", 0)], R2_Y - 7.5)
panel_letters(fig, [(ax_f, "f", 0), (ax_g, "g", 0), (ax_h, "h", -2.0)], R3_Y - 7.0)


def report_min_font(f, name):
    sizes = [t.get_fontsize() for t in f.findobj(plt.Text)
             if t.get_text().strip() and t.get_visible()]
    print(f"{name}: min font {min(sizes):.2f} pt at "
          f"{f.get_size_inches()[0] * 25.4:.0f} mm")


report_min_font(fig, "figure4")
for ext in ("pdf", "png"):
    fig.savefig(FIGURE_DIR / f"figure4.{ext}", dpi=300, facecolor="white")
    print(f"Saved: figure4.{ext}")
plt.close(fig)


# ═════════════════════════════════════════════════════════════════
# Supplementary Figure: prompt and replicate consistency
# ═════════════════════════════════════════════════════════════════
S_H_MM = 62.0
sfig = plt.figure(figsize=(DOUBLE_COL, S_H_MM / 25.4))
S_Y, S_H = 13.0, 40.0

PARA_SERIES = ["within", "cross"]
PARA_COLORS = {"within": INK, "cross": INK}
PARA_MARKERS = {"within": "o", "cross": "o"}
PARA_LABELS = {"within": "Within a paraphrase (replicates)",
               "cross": "Across paraphrases"}

ax_sa = ax_mm(sfig, 34, S_Y, 52, S_H)
row_dots(ax_sa, [(None, [(g, SENS_GROUP_DISPLAY[g], INK_SOFT) for g in SENS_GROUPS])],
         PARA_SERIES, lambda g, s: sens_concordance[g][s], PARA_COLORS, PARA_MARKERS,
         offset_step=0.30, ms=3.2, open_series=("cross",))
ax_sa.axvline(100, color=HAIRLINE, lw=0.4, zorder=1)
ax_sa.set_xlim(58, 102)
ax_sa.set_xticks([60, 70, 80, 90, 100])
ax_sa.set_xlabel("Constrained parameter concordance (%)")
finish_axis(ax_sa)
ax_sa.tick_params(axis="y", length=0)
sx = 0.0
for s in PARA_SERIES:
    ax_sa.plot([sx], [1.07], marker="o", ms=3.2, ls="none",
               mfc=INK if s == "within" else "white", mec=INK, mew=0.6,
               transform=ax_sa.transAxes, clip_on=False)
    t = ax_sa.text(sx + 0.03, 1.07, PARA_LABELS[s], fontsize=FS_SMALL + 0.5, color=INK,
                   va="center", ha="left", transform=ax_sa.transAxes)
    sfig.canvas.draw()
    sx += 0.03 + t.get_window_extent().width / ax_sa.bbox.width + 0.07

ax_sb = ax_mm(sfig, 128, S_Y, 50, S_H)
row_dots(ax_sb, [(None, [(c, CS_CONDITION_LABELS[c], CS_CONDITION_COLORS[c])
                         for c in CS_CONDITIONS])],
         MODELS, lambda c, m: cs_consistency[(c, m)], MODEL_COLORS, MODEL_MARKERS,
         MODEL_EDGES, offset_step=0.24, ms=3.0)
ax_sb.axvline(100, color=HAIRLINE, lw=0.4, zorder=1)
ax_sb.set_xlim(58, 102)
ax_sb.set_xticks([60, 70, 80, 90, 100])
ax_sb.set_xlabel("Cross-replicate tool consistency (%)")
finish_axis(ax_sb)
ax_sb.tick_params(axis="y", length=0)
sx = -0.62
for m in MODELS:
    ax_sb.plot([sx], [1.07], marker=MODEL_MARKERS[m], ms=3.0, ls="none",
               mfc=MODEL_COLORS[m], mec=MODEL_EDGES[m], mew=0.35,
               transform=ax_sb.transAxes, clip_on=False)
    t = ax_sb.text(sx + 0.035, 1.07, MODEL_LABELS[m], fontsize=FS_SMALL + 0.5, color=INK,
                   va="center", ha="left", transform=ax_sb.transAxes)
    sfig.canvas.draw()
    sx += 0.035 + t.get_window_extent().width / ax_sb.bbox.width + 0.07

panel_letters(sfig, [(ax_sa, "a", 0), (ax_sb, "b", 0)], 5.0)
report_min_font(sfig, "supp_fig_prompt_consistency")
for ext in ("pdf", "png"):
    sfig.savefig(FIGURE_DIR / f"supp_fig_prompt_consistency.{ext}", dpi=300,
                 facecolor="white")
    print(f"Saved: supp_fig_prompt_consistency.{ext}")
plt.close(sfig)


# ═════════════════════════════════════════════════════════════════
# Source data of panels f-h and of the Supplementary Figure (values plotted above)
# ═════════════════════════════════════════════════════════════════
SOURCE_DIR = REPO_DIR / "source_data"
SOURCE_DIR.mkdir(exist_ok=True)

rows_f = []
for gid in SENS_GROUPS:
    for model in MODELS:
        m, lo, hi = sens_tool_accuracy[(gid, model)]
        n = sum(1 for r in sens_records if r["group_id"] == gid and r["model"] == model)
        rows_f.append({"block": "Paraphrases (full schema)", "row": SENS_GROUP_DISPLAY[gid],
                       "key": gid, "model": model, "accuracy_pct": m, "ci_lo_pct": lo,
                       "ci_hi_pct": hi, "n_trials": n, "bootstrap_unit": "trial"})
for cond in CS_CONDITIONS:
    for model in MODELS:
        m, lo, hi = cs_accuracy[(cond, model)]
        rows = [r for r in cs_metrics if r["condition"] == cond and r["model"] == model]
        rows_f.append({"block": "System context", "row": CS_CONDITION_LABELS[cond],
                       "key": cond, "model": model, "accuracy_pct": m, "ci_lo_pct": lo,
                       "ci_hi_pct": hi, "n_trials": sum(r["n_trials"] for r in rows),
                       "bootstrap_unit": "prompt"})
pd.DataFrame(rows_f).to_csv(SOURCE_DIR / "Fig2f_tool_selection.csv", index=False)

pd.DataFrame([{"system": sys, "task": task, "n_runs": e2e_counts[(sys, task)]["total"],
               "n_completed": e2e_counts[(sys, task)]["ok"]}
              for task in E2E_TASKS for sys in E2E_SYSTEMS]).to_csv(
    SOURCE_DIR / "Fig2g_runs_completed.csv", index=False)

pd.DataFrame(e2e_pair_rows, columns=["system", "task", "metric", "rep_a", "rep_b", "concordance"]).to_csv(
    SOURCE_DIR / "Fig2h_pairwise_concordance.csv", index=False)

pd.DataFrame([{"prompt_group": gid, "series": s, "concordance_pct": sens_concordance[gid][s][0],
               "ci_lo_pct": sens_concordance[gid][s][1], "ci_hi_pct": sens_concordance[gid][s][2]}
              for gid in SENS_GROUPS for s in PARA_SERIES]).to_csv(
    SOURCE_DIR / "SuppFig1a_param_concordance.csv", index=False)

pd.DataFrame([{"condition": cond, "model": model, "consistency_pct": cs_consistency[(cond, model)][0],
               "ci_lo_pct": cs_consistency[(cond, model)][1], "ci_hi_pct": cs_consistency[(cond, model)][2]}
              for cond in CS_CONDITIONS for model in MODELS]).to_csv(
    SOURCE_DIR / "SuppFig1b_tool_consistency.csv", index=False)
print("Saved: source data of panels f-h and Supplementary Figure 1")
