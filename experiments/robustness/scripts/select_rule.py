#!/usr/bin/env python3
"""Pre-specified label-free selection rule (PREREGISTRATION.md, section 3).

select_rule() never sees ground truth: gt_* columns are dropped before it runs.
strategy_table() adds the reference strategies (schema default, expert, post hoc
oracle); only the oracle reads gt_ari.

Usage: select_rule.py [stage]   -> writes results/{stage}/selection.csv
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

KEYS = ["dataset", "section", "method"]
MIN_DOMAINS = 3


def load_stage(stage: str, conditions=("clean",)) -> pd.DataFrame:
    files = sorted((C.ROOT / "results" / stage).rglob("task_*.csv"))
    if files:
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    else:
        # No cluster run directory: use the combined table committed in results/
        df = pd.read_csv(C.REPO_RESULTS / f"configurations_{stage}.csv.gz")
    df = df[df.condition.isin(conditions)]
    return df


def _deviations(method, config_json) -> int:
    cfg = json.loads(config_json)
    d = G.method_grid(method)["defaults"]
    d = {**d, **G.default_postproc()}
    return sum(1 for k, v in d.items() if k in cfg and str(cfg[k]) != str(v))


def select_rule(df: pd.DataFrame) -> pd.DataFrame:
    """Return one selected grid configuration per (dataset, section, method)."""
    df = df.drop(columns=[c for c in df.columns if c.startswith("gt_")])
    post = G.default_postproc()
    df = df[(df.status == "ok") & (df.strategy.isin(["grid", "default"]))]
    df = df[(df.refine_domains.astype(str) == str(post["refine_domains"])) &
            (df.refinement_threshold.astype(float) == float(post["refinement_threshold"]))]
    full = df[df.rep == "full"].set_index(KEYS + ["call_id"])
    stab = (df[df.rep != "full"].groupby(KEYS + ["call_id"]).stability.mean().rename("S"))
    cand = full.join(stab, how="inner")
    cand = cand[cand.n_domains_found >= MIN_DOMAINS].reset_index()
    out = []
    for key, g in cand.groupby(KEYS):
        g = g.copy()
        g["rank_S"] = g.S.rank(pct=True, method="average")
        g["rank_C"] = g.kappa.rank(pct=True, method="average")
        g["score"] = g.rank_S + g.rank_C
        g["dev"] = [_deviations(key[2], c) for c in g.config]
        g = g.sort_values(["score", "S", "dev", "call_id"], ascending=[False, False, True, True])
        best = g.iloc[0]
        out.append({**dict(zip(KEYS, key)), "strategy": "rule", "call_id": best.call_id,
                    "config": best.config, "config_spec": best.get("config_spec", best.config), "S": best.S, "kappa": best.kappa, "score": best.score,
                    "n_candidates": len(g)})
    return pd.DataFrame(out)


def strategy_table(stage: str) -> pd.DataFrame:
    df = load_stage(stage)
    rule = select_rule(df)
    ok = df[(df.status == "ok") & (df.rep == "full")]
    post = G.default_postproc()
    is_def_post = ((ok.refine_domains.astype(str) == str(post["refine_domains"])) &
                   (ok.refinement_threshold.astype(float) == float(post["refinement_threshold"])))
    rows = [rule]
    d = ok[(ok.strategy == "default") & is_def_post]
    cols = KEYS + ["call_id", "config"] + (["config_spec"] if "config_spec" in ok.columns else [])
    rows.append(d[cols].assign(strategy="default"))
    e = ok[ok.strategy == "expert"]
    rows.append(e[cols].assign(strategy="expert"))
    # post hoc oracle: best ARI over every schema-valid configuration in the grid (uses labels)
    g = ok[ok.strategy.isin(["grid", "default"])]
    o = g.loc[g.groupby(KEYS).gt_ari.idxmax()]
    rows.append(o[cols].assign(strategy="oracle"))
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    t = strategy_table(stage)
    path = C.ROOT / "results" / stage / "selection.csv"
    t.to_csv(path, index=False)
    print(t.groupby(["dataset", "method", "strategy"]).size().unstack(fill_value=0))
    print(f"-> {path}")
