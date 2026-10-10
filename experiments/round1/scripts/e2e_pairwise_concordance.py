#!/usr/bin/env python3
"""Pairwise cross-replicate concordance of the end-to-end benchmark (Figure 4i).

Computes, for each system and task of the lymph node end-to-end benchmark,
the concordance between every pair of successful replicates:

  spatial_domains     ARI of the domain labels on the spots shared by both
                      replicates (first obs column whose name contains
                      "domain", "leiden", "cluster" or "louvain")
  svg_detection       Jaccard index of the top 100 genes of svg_results.csv
  cell_communication  Jaccard index of the top 50 ligand-receptor pairs

This is the computation of figures/figure4.py panel i, moved here so that the
figure reads a small table instead of the 5.7 GB of benchmark outputs. Only
the obs table of each h5ad file is read.

Inputs
  results/e2e_benchmark/e2e_benchmark_raw.jsonl   trial records (success flags)
  E2E_OUTPUT_DIR (default results/e2e_benchmark/outputs), written by
  e2e_benchmark.py: <system>/<task>/rep_<k>/ with the files each system saved
Output
  results/e2e_benchmark/e2e_pairwise_concordance.csv
  (columns system, task, rep_a, rep_b, value)
"""

from __future__ import annotations

import json
import os
from itertools import combinations
from pathlib import Path

import h5py
import pandas as pd
from sklearn.metrics import adjusted_rand_score

try:  # anndata >= 0.11
    from anndata.io import read_elem
except ImportError:  # anndata 0.9-0.10
    from anndata.experimental import read_elem

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "results" / "e2e_benchmark"
RAW_PATH = DATA_DIR / "e2e_benchmark_raw.jsonl"
OUTPUT_DIR = Path(os.environ.get("E2E_OUTPUT_DIR", DATA_DIR / "outputs"))
OUT_CSV = DATA_DIR / "e2e_pairwise_concordance.csv"

SYSTEMS = ["chatspatial", "stagent", "spatialagent"]


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_obs(path: Path) -> pd.DataFrame:
    with h5py.File(path, "r") as handle:
        return read_elem(handle["obs"])


def load_domain_labels(system: str, task: str, rep) -> dict | None:
    trial_dir = OUTPUT_DIR / system / task / f"rep_{rep}"
    for f in sorted(trial_dir.glob("*.h5ad")):
        obs = read_obs(f)
        cols = [
            c
            for c in obs.columns
            if any(kw in c.lower() for kw in ["domain", "leiden", "cluster", "louvain"])
        ]
        if cols:
            return dict(zip(obs.index, obs[cols[0]].astype(str)))
    return None


def load_svg_list(system: str, task: str, rep) -> list | None:
    csv_path = OUTPUT_DIR / system / task / f"rep_{rep}" / "svg_results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if len(df) > 0:
            return df.iloc[:, 0].tolist()[:100]
    return None


def load_lr_pairs(system: str, task: str, rep) -> list | None:
    trial_dir = OUTPUT_DIR / system / task / f"rep_{rep}"
    for f in sorted(trial_dir.glob("*.csv")):
        df = pd.read_csv(f)
        cols_lower = {c.lower(): c for c in df.columns}
        if "ligand" in cols_lower and "receptor" in cols_lower:
            return [
                f"{r[cols_lower['ligand']]}^{r[cols_lower['receptor']]}".upper()
                for _, r in df.head(50).iterrows()
            ]
        if "interacting_pair" in cols_lower:
            return [p.upper() for p in df[cols_lower["interacting_pair"]].head(50).tolist()]
    return None


def main() -> None:
    records = load_jsonl(RAW_PATH)
    rows = []
    for system in SYSTEMS:
        ok = {
            task: [r["rep"] for r in records
                   if r["system"] == system and r["task_id"] == task and r["success"]]
            for task in ["spatial_domains", "svg_detection", "cell_communication"]
        }

        labels = {}
        for rep in ok["spatial_domains"]:
            lab = load_domain_labels(system, "spatial_domains", rep)
            if lab is not None:
                labels[rep] = lab
        for a, b in combinations(sorted(labels), 2):
            common = sorted(set(labels[a]) & set(labels[b]))
            if len(common) >= 10:
                value = adjusted_rand_score([labels[a][k] for k in common],
                                            [labels[b][k] for k in common])
                rows.append((system, "spatial_domains", a, b, value))

        genes = {}
        for rep in ok["svg_detection"]:
            gl = load_svg_list(system, "svg_detection", rep)
            if gl is not None:
                genes[rep] = gl
        for a, b in combinations(sorted(genes), 2):
            sa, sb = set(genes[a][:100]), set(genes[b][:100])
            if sa or sb:
                rows.append((system, "svg_detection", a, b, len(sa & sb) / len(sa | sb)))

        lr = {}
        for rep in ok["cell_communication"]:
            lp = load_lr_pairs(system, "cell_communication", rep)
            if lp is not None:
                lr[rep] = lp
        for a, b in combinations(sorted(lr), 2):
            sa, sb = set(lr[a][:50]), set(lr[b][:50])
            if sa or sb:
                rows.append((system, "cell_communication", a, b, len(sa & sb) / len(sa | sb)))

    out = pd.DataFrame(rows, columns=["system", "task", "rep_a", "rep_b", "value"])
    out.to_csv(OUT_CSV, index=False)
    print(out.groupby(["system", "task"])["value"].agg(["mean", "count"]))
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
