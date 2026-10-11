"""
Figure 4: schema-derived parameter spaces make robustness analysis and
label-free parameter selection systematic (experiment A).

Inputs are small tables exported from the final experiment-A
stages (dlpfc155, noise155, platforms155) by experiments/robustness/scripts/export_figA.py:
    experiments/robustness/results/strategy_scores_all.csv   ARI of each strategy per sample and method
    experiments/robustness/results/sensitivity_dlpfc.csv     min/median/max ARI over schema-valid configurations
    experiments/robustness/results/noise_retention.csv       paired ARI retention under data degradation
    experiments/robustness/results/platform_method_means.csv mean ARI per platform, strategy and method

Output: figures/figureA.pdf/.png and source data in source_data/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_style import FULL_WIDTH, OI, add_panel_label, apply_style

ROOT = Path(__file__).resolve().parents[1]
EXPA = ROOT / "experiments" / "robustness" / "results"
FIG_DIR = ROOT / "figures"
SOURCE_DIR = ROOT / "source_data"

METHODS = [
    ("spagcn", "SpaGCN"),
    ("leiden", "Leiden"),
    ("louvain", "Louvain"),
    ("stagate", "STAGATE"),
    ("graphst", "GraphST"),
    ("banksy", "BANKSY"),
]
STRATEGIES = [
    ("expert", "Published settings", OI["grey"], "s"),
    ("default", "Schema default", OI["blue"], "o"),
    ("rule", "Label-free rule", OI["vermillion"], "D"),
    ("oracle", "Best configuration", "black", "|"),
]
NOISE = [
    ("umi0.5", "UMI 50%"),
    ("umi0.25", "UMI 25%"),
    ("umi0.1", "UMI 10%"),
    ("drop0.1", "Spots −10%"),
    ("drop0.3", "Spots −30%"),
]
PLATFORMS = ["Visium", "STARmap", "MERFISH"]
SAMPLE_NAME = {"starmap": "STARmap", "abc_merfish": "MERFISH"}


def main() -> None:
    apply_style()
    SOURCE_DIR.mkdir(exist_ok=True)

    scores = pd.read_csv(EXPA / "strategy_scores_all.csv", dtype={"section": str})
    scores = scores[scores.dataset != "dlpfc_array"]
    sens = pd.read_csv(EXPA / "sensitivity_dlpfc.csv", dtype={"section": str})
    sens = sens[sens.dataset == "dlpfc"].assign(range=lambda d: d["max"] - d["min"])
    noise = pd.read_csv(EXPA / "noise_retention.csv")
    plat = pd.read_csv(EXPA / "platform_method_means.csv")

    fig = plt.figure(figsize=(FULL_WIDTH, FULL_WIDTH * 0.66))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.42, hspace=0.62, width_ratios=[1, 1.25])

    # a: ARI range across schema-valid configurations, per method and DLPFC section
    ax = fig.add_subplot(gs[0, 0])
    rng = np.random.default_rng(0)
    rows_a = []
    for i, (key, name) in enumerate(METHODS):
        v = sens[sens.method == key]
        x = i + rng.uniform(-0.18, 0.18, len(v))
        ax.scatter(x, v["range"], s=9, color=OI["blue"], alpha=0.75, lw=0)
        med = v["range"].median()
        ax.plot([i - 0.3, i + 0.3], [med, med], color="black", lw=1.2)
        for _, r in v.iterrows():
            rows_a.append({"method": name, "section": r.section, "ari_min": r["min"],
                           "ari_median": r["median"], "ari_max": r["max"],
                           "ari_range": r["range"], "n_configurations": int(r["count"])})
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([n for _, n in METHODS], rotation=35, ha="right")
    ax.set_ylabel("ARI range across\nschema-valid settings")
    ax.set_ylim(0, 0.8)
    add_panel_label(ax, "a", x=-0.28, y=1.08)
    pd.DataFrame(rows_a).to_csv(SOURCE_DIR / "Fig4a_sensitivity.csv", index=False)

    # b: mean ARI over methods per sample for each strategy
    ax = fig.add_subplot(gs[0, 1])
    piv = scores.pivot_table(index=["dataset", "section", "method"], columns="strategy", values="ari")
    by = piv.groupby(["dataset", "section"]).mean().reset_index()
    by["label"] = [SAMPLE_NAME.get(d, s) for d, s in zip(by.dataset, by.section)]
    order = [s for s in by[by.dataset == "dlpfc"].sort_values("section").label] + ["STARmap", "MERFISH"]
    by = by.set_index("label").loc[order].reset_index()
    for key, name, color, marker in STRATEGIES:
        kw = {"ms": 8, "mew": 1.2} if marker == "|" else {"ms": 3.5}
        ax.plot(range(len(by)), by[key], marker, color=color, label=name, linestyle="none", **kw)
    ax.axvline(11.5, color=OI["grey"], lw=0.6, ls=":")
    ax.set_xticks(range(len(by)))
    ax.set_xticklabels(by.label, rotation=60, ha="right")
    ax.set_ylabel("Mean ARI over six methods")
    ax.set_ylim(0.1, 0.65)
    ax.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0), handletextpad=0.2,
              columnspacing=0.8)
    add_panel_label(ax, "b", x=-0.16, y=1.24)
    by[["dataset", "section", "label", "expert", "default", "rule", "oracle"]].to_csv(
        SOURCE_DIR / "Fig4b_strategies.csv", index=False)

    # c: retention of ARI under data degradation (paired with the clean data)
    ax = fig.add_subplot(gs[1, 0])
    width = 0.36
    rows_c = []
    for j, (key, name, color, _) in enumerate(STRATEGIES[1:3]):
        vals = []
        for cond, _ in NOISE:
            r = noise[(noise.condition == cond) & (noise.strategy == key)].iloc[0]
            vals.append(r.retention)
            rows_c.append({"condition": cond, "strategy": name, "retention": r.retention,
                           "noisy_mean_ari": r.noisy_mean, "clean_mean_ari": r.clean_mean,
                           "n_method_section_cells": int(r.n_cells)})
        ax.bar(np.arange(len(NOISE)) + (j - 0.5) * width, vals, width, color=color, label=name)
    ax.axhline(1.0, color=OI["grey"], lw=0.6, ls=":")
    ax.set_xticks(range(len(NOISE)))
    ax.set_xticklabels([n for _, n in NOISE], rotation=35, ha="right")
    ax.set_ylabel("ARI retained\n(degraded / original)")
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handletextpad=0.3)
    add_panel_label(ax, "c", x=-0.28, y=1.18)
    pd.DataFrame(rows_c).to_csv(SOURCE_DIR / "Fig4c_noise.csv", index=False)

    # d: mean ARI of each method on each platform with the label-free rule; ranks annotated
    ax = fig.add_subplot(gs[1, 1])
    sub = plat[plat.strategy == "rule"].pivot(index="platform", columns="method", values="ari")
    sub = sub.loc[PLATFORMS, [k for k, _ in METHODS]]
    im = ax.imshow(sub.values, cmap="Blues", vmin=0, vmax=0.65, aspect="auto")
    ranks = sub.rank(axis=1, ascending=False).astype(int)
    for i in range(sub.shape[0]):
        for k in range(sub.shape[1]):
            v = sub.values[i, k]
            ax.text(k, i, f"{v:.2f}\n#{ranks.values[i, k]}", ha="center", va="center", fontsize=5.5,
                    color="white" if v > 0.42 else "black")
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([n for _, n in METHODS], rotation=35, ha="right")
    ax.set_yticks(range(len(PLATFORMS)))
    ax.set_yticklabels(PLATFORMS)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Mean ARI (label-free rule)")
    add_panel_label(ax, "d", x=-0.16, y=1.18)
    plat.rename(columns={"ari": "mean_ari"}).to_csv(SOURCE_DIR / "Fig4d_platforms.csv", index=False)

    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"figureA.{ext}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
