#!/usr/bin/env python3
"""Pooled 14-sample comparison (DLPFC 12 + STARmap + MERFISH). Usage: pooled.py MAIN PLATFORMS"""
import sys
import pandas as pd
import evaluate as E
import select_rule as S
import common as C

main, plat = sys.argv[1], sys.argv[2]
sc = pd.read_csv(C.ROOT / "results" / main / "strategy_scores.csv", dtype={"section": str})
sc = sc[sc.dataset == "dlpfc"]
dfp = S.load_stage(plat); dfp["section"] = dfp.section.astype(str)
selp = S.strategy_table(plat); selp["section"] = selp.section.astype(str)
sp = E.strategy_scores(dfp, selp)
print(sp.pivot_table(index=["dataset", "method"], columns="strategy", values="ari").round(3).to_string())
p = sc.pivot_table(index="method", columns="strategy", values="ari").round(3)
p["rule-default"] = (p.rule - p.default).round(3)
print("DLPFC per method:\n" + p.to_string())
allsc = pd.concat([sc, sp])
bys = allsc.groupby(["dataset", "section", "strategy"]).ari.mean().unstack()
for s in ["rule", "default", "expert", "oracle"]:
    print(s, E.fmt(E.boot_mean(bys[s])))
d = bys.rule - bys.default
print("14 samples: rule-default", E.fmt(E.boot_mean(d)), "matched", int((bys.rule >= bys.default - 0.01).sum()),
      "of", len(bys), "gap", round((bys.rule.mean() - bys.default.mean()) / (bys.oracle.mean() - bys.default.mean()), 3))
piv = allsc.pivot_table(index=["dataset", "section", "method"], columns="strategy", values="ari")
print("A6 14 samples:", int((piv.default >= piv.oracle - 0.05).sum()), "of", len(piv))
