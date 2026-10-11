#!/usr/bin/env python3
"""Export the small tables of Figure 4 (results/*.csv) from the final stages dlpfc155, platforms155, noise155."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C, select_rule as S
import evaluate as E

import os
OUT = Path(os.environ.get("EXPA_EXPORT_DIR", C.REPO_RESULTS)); OUT.mkdir(parents=True, exist_ok=True)
df = S.load_stage("dlpfc155"); df["section"] = df.section.astype(str)
sel = S.strategy_table("dlpfc155"); sel["section"] = sel.section.astype(str)
sc = E.strategy_scores(df, sel)
dfp = S.load_stage("platforms155"); dfp["section"] = dfp.section.astype(str)
selp = S.strategy_table("platforms155"); selp["section"] = selp.section.astype(str)
scp = E.strategy_scores(dfp, selp)
sc_all = pd.concat([sc, scp]); sc_all.to_csv(OUT / "strategy_scores_all.csv", index=False)

g = df[(df.status == "ok") & (df.rep == "full") & (df.strategy.isin(["grid", "default"]))]
sens = g.groupby(["dataset", "section", "method"]).gt_ari.agg(["min", "median", "max", "count"]).reset_index()
sens.to_csv(OUT / "sensitivity_dlpfc.csv", index=False)
gp = dfp[(dfp.status == "ok") & (dfp.rep == "full") & (dfp.strategy.isin(["grid", "default"]))]
gp.groupby(["dataset", "section", "method"]).gt_ari.agg(["min", "median", "max", "count"]).reset_index().to_csv(OUT / "sensitivity_platforms.csv", index=False)

noise = S.load_stage("noise155", conditions=tuple(C.NOISE_CONDITIONS))
noise = noise[(noise.status == "ok") & (noise.rep == "full")].copy()
noise["section"] = noise.section.astype(str); noise["strategy"] = noise.strategy.astype(str)
cl = sc[sc.dataset == "dlpfc"].set_index(["section", "method", "strategy"]).ari
rows = []
for cond, gg in noise.groupby("condition"):
    gi = gg.set_index(["section", "method", "strategy"]).gt_ari
    gi = gi[~gi.index.duplicated()]
    common = gi.index.intersection(cl.index)
    for strat in ["default", "rule", "expert", "oracle"]:
        idx = [i for i in common if i[2] == strat]
        if not idx: continue
        rows.append({"condition": cond, "strategy": strat, "noisy_mean": gi[idx].mean(), "clean_mean": cl[idx].mean(),
                     "retention": gi[idx].mean() / cl[idx].mean(), "n_cells": len(idx)})
pd.DataFrame(rows).to_csv(OUT / "noise_retention.csv", index=False)

plat = sc_all.assign(platform=sc_all.dataset.map(lambda d: C.DATASETS[d]["platform"]))
plat.groupby(["platform", "strategy", "method"]).ari.mean().reset_index().to_csv(OUT / "platform_method_means.csv", index=False)
print("exported to", OUT)
