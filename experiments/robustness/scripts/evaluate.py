#!/usr/bin/env python3
"""Experiment A evaluation: the reported numbers (A4-A14, A-GAP, N-*) and the pilot gate.

Usage: evaluate.py [MAIN NOISE PLATFORMS]   default: pilot noise platforms
       e.g. evaluate.py dlpfc155 noise155 platforms155
Reads results/{stage}/**/task_*.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import grid as G  # noqa: E402
import select_rule as S  # noqa: E402

DELTA = 0.05
MATCH_TOL = 0.01
B = 10000
rng = np.random.default_rng(20261008)


def boot_mean(by_sample: pd.Series):
    v = by_sample.values
    idx = rng.integers(0, len(v), size=(B, len(v)))
    m = v[idx].mean(axis=1)
    return v.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def fmt(t):
    return f"{t[0]:.3f} (95% CI {t[1]:.3f}-{t[2]:.3f})"


def strategy_scores(df, sel):
    """ARI of each strategy per (dataset, section, method) on clean data."""
    ok = df[(df.status == "ok") & (df.rep == "full")]
    out = []
    for _, r in sel.iterrows():
        m = ok[(ok.dataset == r.dataset) & (ok.section == r.section) & (ok.method == r.method)]
        if r.strategy == "expert":
            row = m[m.strategy == "expert"]
        else:
            row = m[(m.call_id == r.call_id) & (m.config == r.config)] if r.strategy != "rule" else \
                m[(m.call_id == r.call_id) & (m.refine_domains.astype(str) == "True") &
                  (m.refinement_threshold.astype(float) == 0.5)]
        if len(row):
            out.append({"dataset": r.dataset, "section": str(r.section), "method": r.method,
                        "strategy": r.strategy, "ari": row.gt_ari.iloc[0], "nmi": row.gt_nmi.iloc[0],
                        "n_domains": row.n_domains_found.iloc[0]})
    return pd.DataFrame(out)


def kendall_w(ranks: pd.DataFrame) -> float:
    """ranks: rows = platforms (judges), columns = methods."""
    m, n = ranks.shape
    R = ranks.sum(axis=0)
    S_ = ((R - R.mean()) ** 2).sum()
    return 12 * S_ / (m ** 2 * (n ** 3 - n))


def concordance(sc_all: pd.DataFrame, strategy: str):
    plat = sc_all.assign(platform=sc_all.dataset.map(lambda d: C.DATASETS[d]["platform"]))
    mean = plat[plat.strategy == strategy].groupby(["platform", "method"]).ari.mean().unstack()
    ranks = mean.rank(axis=1, ascending=False)
    w = kendall_w(ranks)
    perm = [kendall_w(ranks.apply(lambda r: pd.Series(rng.permutation(r.values), index=r.index), axis=1))
            for _ in range(B)]
    return w, (np.sum(np.array(perm) >= w) + 1) / (B + 1), mean.round(3)


def main():
    MAIN, NOISE, PLAT = (sys.argv[1:4] + ["pilot", "noise", "platforms"][len(sys.argv[1:4]):])
    print(f"stages: main={MAIN} noise={NOISE} platforms={PLAT}")
    df = S.load_stage(MAIN)
    df["section"] = df.section.astype(str)
    sel = S.strategy_table(MAIN)
    sel["section"] = sel.section.astype(str)
    sc = strategy_scores(df, sel)
    piv = sc.pivot_table(index=["dataset", "section", "method"], columns="strategy", values="ari")
    d = piv.loc["dlpfc"]

    # A4 and A5
    grid_rows = df[(df.dataset == "dlpfc") & (df.status == "ok") & (df.rep == "full") &
                   (df.strategy.isin(["grid", "default"]))]
    print("A4 configurations per section:", grid_rows.groupby("section").size().median())
    rng_ = grid_rows.groupby(["method", "section"]).gt_ari.agg(lambda x: x.max() - x.min())
    print("A5 median ARI range per method:", rng_.groupby("method").median().round(3).to_dict())
    print("A6 default within delta of oracle:",
          f"{(d.default >= d.oracle - DELTA).mean():.0%} ({(d.default >= d.oracle - DELTA).sum()} of {len(d)})")

    by_s = d.groupby("section").mean()
    for strat in ("rule", "default", "expert", "oracle"):
        print(f"{strat:8s} mean ARI over sections:", fmt(boot_mean(by_s[strat])))
    diff = by_s.rule - by_s.default
    print("rule - default:", fmt(boot_mean(diff)))
    matched = (by_s.rule >= by_s.default - MATCH_TOL).sum()
    gap = (by_s.rule.mean() - by_s.default.mean()) / (by_s.oracle.mean() - by_s.default.mean())
    print(f"A10 matched-or-exceeded: {matched} of {len(by_s)}; A-GAP {gap:.0%}")
    sp = d.xs("spagcn", level="method")
    print("A12 SpaGCN default:", fmt(boot_mean(sp.default)), " A13 SpaGCN rule:", fmt(boot_mean(sp.rule)))
    cont = df[(df.dataset == "dlpfc_array") & (df.rep == "full") & (df.refine_domains.astype(str) == "False")]
    if len(cont):
        print("SpaGCN default, array coords, unrefined (May-2026 scoring):",
              cont.set_index("section").gt_ari.round(4).to_dict(), "mean", round(cont.gt_ari.mean(), 3))
    worse = d[d.rule < d.default - DELTA]
    print("A11 candidates (rule > delta below default):")
    print(worse.round(3).to_string())

    lo = boot_mean(diff)
    go = matched >= 7 and gap >= 0.25 and lo[1] > 0
    pivot = lo[2] < 0 or matched <= 4
    print("GATE:", "GO" if go else ("PIVOT" if pivot else "ADJUST"))

    noise = None
    try:
        noise = S.load_stage(NOISE, conditions=tuple(C.NOISE_CONDITIONS))
    except ValueError:
        pass
    if noise is not None and len(noise):
        noise_all = noise[noise.rep == "full"]
        noise = noise[(noise.status == "ok")]
        noise["section"] = noise.section.astype(str)
        noise["strategy"] = noise.strategy.astype(str)
        clean = sc[sc.dataset == "dlpfc"].groupby("strategy").ari.mean()
        for cond, g in noise.groupby("condition"):
            m = g.groupby("strategy").gt_ari.mean()
            ret = (m / clean.reindex(m.index)).round(3).to_dict()
            # paired retention: clean mean over the same (section, method) cells that ran under noise
            cl = sc[sc.dataset == "dlpfc"].set_index(["section", "method", "strategy"]).ari
            gi = g.set_index(["section", "method", "strategy"]).gt_ari
            gi = gi[~gi.index.duplicated()]
            common = gi.index.intersection(cl.index)
            ret_p = (gi[common].groupby(level="strategy").mean() /
                     cl[common].groupby(level="strategy").mean()).round(3).to_dict()
            n_fail = int(((noise_all.condition == cond) & (noise_all.status != "ok")).sum())
            print(f"noise {cond}: paired retention {ret_p}; failed calls {n_fail}")
            ps = g.pivot_table(index=["section", "method"], columns="strategy", values="gt_ari")
            dd = (ps["rule"] - ps["default"]).groupby("section").mean()
            print(f"noise {cond}: retention {ret}; rule-default {fmt(boot_mean(dd))}")
    try:
        dfp = S.load_stage(PLAT)
        dfp["section"] = dfp.section.astype(str)
        selp = S.strategy_table(PLAT)
        selp["section"] = selp.section.astype(str)
        sc_all = pd.concat([sc[sc.dataset == "dlpfc"], strategy_scores(dfp, selp)])
        for strat in ("rule", "default"):
            w, p, tab = concordance(sc_all, strat)
            print(f"A14 Kendall's W ({strat}) = {w:.2f}, permutation P = {p:.4f}\n{tab}")
    except ValueError:
        print("platforms stage not run yet: A14 pending")
    out = C.ROOT / "results" / MAIN / "strategy_scores.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    sc.to_csv(out, index=False)
    print("->", out)


if __name__ == "__main__":
    main()
