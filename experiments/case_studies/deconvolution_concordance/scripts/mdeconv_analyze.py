#!/usr/bin/env python
"""M-DECONV step 3: apply the pre-specified rules (mdeconv_rules_2026-10-09.txt) to the runs.

Usage: mdeconv_analyze.py RUN_DIR OUT_DIR
Writes per_sample_enrichment.csv, patient_level_tests.csv, consensus_counts.csv,
cross_method_agreement.csv, decision.txt.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

run_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
SAMPLES = [f"s{i}" for i in range(1, 13)]
RUNS = ["card", "rctd", "flashdeconv", "flashdeconv_logref", "manuscript"]
PRIMARY = ["Macrophage", "Fibroblast"]
PSEUDO = 1e-6


def load(run):
    parts = {}
    for s in SAMPLES:
        f = run_dir / f"{s}__{run}.csv"
        if not f.exists():
            f = run_dir / f"{s}__{run}.csv.gz"  # committed tables are gzipped
        if f.exists():
            parts[s] = pd.read_csv(f, index_col=0)
    return parts


def per_sample(parts):
    rows = []
    for s, df in parts.items():
        reg = df["tc_le_region"].astype(str).values
        P = df.drop(columns="tc_le_region")
        for ct in P.columns:
            tc, le = P.loc[reg == "TC", ct].values, P.loc[reg == "LE", ct].values
            if len(le) == 0 or len(tc) == 0:
                rows.append(dict(sample=s, cell_type=ct, n_TC=len(tc), n_LE=len(le), tc_mean=np.nan,
                                 le_mean=np.nan, log2fc=np.nan, pvalue=np.nan))
                continue
            p = np.nan if (le.sum() == 0 and tc.sum() == 0) else stats.mannwhitneyu(le, tc, alternative="two-sided").pvalue
            rows.append(dict(sample=s, cell_type=ct, n_TC=len(tc), n_LE=len(le), tc_mean=tc.mean(), le_mean=le.mean(),
                             log2fc=np.log2((le.mean() + PSEUDO) / (tc.mean() + PSEUDO)), pvalue=p))
    d = pd.DataFrame(rows)
    d["q"] = np.nan
    v = d["pvalue"].notna()
    if v.any():
        d.loc[v, "q"] = multipletests(d.loc[v, "pvalue"], method="fdr_bh")[1]
    d["enriched_in"] = "Neither"
    sig = (d["q"] < 0.05) & (d["log2fc"].abs() > 0.25)
    d.loc[sig & (d["log2fc"] > 0), "enriched_in"] = "LE"
    d.loc[sig & (d["log2fc"] < 0), "enriched_in"] = "TC"
    return d


def patient_level(d):
    rows = []
    dd = d[d["n_LE"] > 0]
    for ct in sorted(dd["cell_type"].unique()):
        x = dd.loc[dd["cell_type"] == ct, "log2fc"].dropna().values
        nz = x[x != 0]
        p = stats.wilcoxon(nz, alternative="two-sided").pvalue if len(nz) >= 3 else np.nan
        med = float(np.median(x)) if len(x) else np.nan
        rows.append(dict(cell_type=ct, n_samples=len(x), n_nonzero=len(nz), median_log2fc=med,
                         n_pos=int((nz > 0).sum()), n_neg=int((nz < 0).sum()), wilcoxon_p=p))
    t = pd.DataFrame(rows)
    t["q"] = np.nan
    v = t["wilcoxon_p"].notna()
    if v.any():
        t.loc[v, "q"] = multipletests(t.loc[v, "wilcoxon_p"], method="fdr_bh")[1]
    t["direction"] = np.where(t["median_log2fc"] > 0, "LE", np.where(t["median_log2fc"] < 0, "TC", "--"))
    return t


all_ps, all_pt, cons, loaded = [], [], [], {}
for run in RUNS:
    parts = load(run)
    loaded[run] = parts
    if not parts:
        continue
    d = per_sample(parts)
    d.insert(0, "run", run)
    t = patient_level(d)
    t.insert(0, "run", run)
    all_ps.append(d)
    all_pt.append(t)
    for ct, g in d.groupby("cell_type"):
        cons.append(dict(run=run, cell_type=ct, samples_run=len(parts), le_enriched=int((g["enriched_in"] == "LE").sum()),
                         tc_enriched=int((g["enriched_in"] == "TC").sum())))
ps, pt, cs = pd.concat(all_ps), pd.concat(all_pt), pd.DataFrame(cons)
ps.to_csv(out / "per_sample_enrichment.csv", index=False)
pt.to_csv(out / "patient_level_tests.csv", index=False)
cs.to_csv(out / "consensus_counts.csv", index=False)

# Cross-method agreement: per-sample log2FC sign agreement and spot-level Pearson r
agree = []
main3 = ["card", "rctd", "flashdeconv"]
for i, a in enumerate(main3 + ["manuscript"]):
    for b in (main3 + ["manuscript", "flashdeconv_logref"])[i + 1:]:
        if a == b or not loaded.get(a) or not loaded.get(b):
            continue
        shared = sorted(set(ps.loc[ps.run == a, "cell_type"]) & set(ps.loc[ps.run == b, "cell_type"]))
        for ct in shared:
            x = ps[(ps.run == a) & (ps.cell_type == ct)].set_index("sample")["log2fc"]
            y = ps[(ps.run == b) & (ps.cell_type == ct)].set_index("sample")["log2fc"]
            j = x.dropna().index.intersection(y.dropna().index)
            same = int((np.sign(x[j]) == np.sign(y[j])).sum())
            rs = []
            for s in set(loaded[a]) & set(loaded[b]):
                pa, pb = loaded[a][s][ct], loaded[b][s][ct].reindex(loaded[a][s].index)
                if pa.std() > 0 and pb.std() > 0:
                    rs.append(np.corrcoef(pa.values, pb.values)[0, 1])
            agree.append(dict(run_a=a, run_b=b, cell_type=ct, samples_sign_agree=same, samples_compared=len(j),
                              median_spot_pearson_r=float(np.median(rs)) if rs else np.nan))
pd.DataFrame(agree).to_csv(out / "cross_method_agreement.csv", index=False)

# Decision (primary rule)
lines, verdict = [], {}
for run in main3:
    for ct in PRIMARY:
        r = pt[(pt.run == run) & (pt.cell_type == ct)]
        if r.empty:
            verdict[(run, ct)] = "missing"
            continue
        r = r.iloc[0]
        if r.q < 0.05 and r.median_log2fc > 0:
            verdict[(run, ct)] = "reproduced"
        elif r.q < 0.05 and r.median_log2fc < 0:
            verdict[(run, ct)] = "CONTRADICTED (significant TC direction)"
        else:
            verdict[(run, ct)] = "not reproduced"
        lines.append(f"{run:12s} {ct:11s} median log2FC {r.median_log2fc:+.2f}  q {r.q:.4f}  "
                     f"(+{r.n_pos}/-{r.n_neg})  -> {verdict[(run, ct)]}")
ok = {k: v == "reproduced" for k, v in verdict.items()}
all_ok = all(ok.get((m, c), False) for m in main3 for c in PRIMARY)
ct_all3 = [c for c in PRIMARY if all(ok.get((m, c), False) for m in main3)]
meth_both = [m for m in main3 if all(ok.get((m, c), False) for c in PRIMARY)]
if all_ok:
    decision = "CONSISTENT"
elif ct_all3 or len(meth_both) >= 2:
    decision = "PARTLY CONSISTENT"
else:
    decision = "INCONSISTENT"
if any("CONTRADICTED" in v for v in verdict.values()):
    decision += " (with a contradiction)"
runlogs = []
for f in sorted(run_dir.glob("*__runlog.json")):
    runlogs += json.load(open(f))
fails = [r for r in runlogs if r["status"] != "ok"]
with open(out / "decision.txt", "w") as f:
    f.write("Primary rule (mdeconv_rules_2026-10-09.txt, section 4)\n")
    f.write("\n".join(lines) + f"\n\nDECISION: {decision}\n")
    f.write(f"samples per run: { {r: len(p) for r, p in loaded.items()} }\n")
    f.write(f"failed runs: {len(fails)}\n")
    for r in fails:
        f.write(f"  {r['sample']} {r['run']}: {r['error'][:300]}\n")
print(open(out / "decision.txt").read())
with pd.option_context("display.width", 200, "display.max_rows", 200):
    print(pt.to_string(index=False))
    print(cs.pivot(index="cell_type", columns="run", values="le_enriched").to_string())
    print(cs.pivot(index="cell_type", columns="run", values="tc_enriched").to_string())
    print(pd.DataFrame(agree).to_string(index=False))
