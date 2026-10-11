#!/usr/bin/env python3
"""Write the backend-study tables of the Supplementary Information
(supplementary_tables/Supplementary_Table_expB_*.csv) from the analysis tables of
experiments/backends/results/analysis/. The values are unchanged; model identifiers
are replaced by the display names used in Figure 3 and in Supplementary Table 8.
"""
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ANALYSIS = REPO / "experiments" / "backends" / "results" / "analysis"
OUT = REPO / "supplementary_tables"

LABEL = {
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "claude-haiku-5-5": "Claude Haiku 5.5",
    "claude-sonnet-5-5": "Claude Sonnet 5.5",
    "claude-opus-5-5": "Claude Opus 5.5",
    "gpt-6-sol": "GPT-6 Sol",
    "gpt-6.1-sol": "GPT-6.1 Sol",
    "ow:google/gemma-4-12B-it": "Gemma 4 12B",
    "ow:Qwen/Qwen3.8-27B-FP8": "Qwen3.8 27B (FP8)",
}
VERSION_PAIRS = [("claude-haiku-4-5-20251001", "claude-haiku-5-5"), ("gpt-6-sol", "gpt-6.1-sol")]


def main() -> None:
    rates = pd.read_csv(ANALYSIS / "rates_by_backend_condition.csv")
    rates = rates.rename(columns={"label": "backend"}).assign(backend=lambda d: d.backend.map(LABEL))
    rates.to_csv(OUT / "Supplementary_Table_expB_rates_by_backend_condition.csv", index=False)

    desc = pd.read_csv(ANALYSIS / "bdesc_by_backend.csv")
    desc = desc.rename(columns={"label": "backend"}).assign(backend=lambda d: d.backend.map(LABEL))
    desc.to_csv(OUT / "Supplementary_Table_expB_description_ablation.csv", index=False)

    pd.read_csv(ANALYSIS / "bmis_decomposition.csv").to_csv(
        OUT / "Supplementary_Table_expB_error_decomposition.csv", index=False)

    det = pd.read_csv(ANALYSIS / "binv_per_backend.csv")
    det = det.rename(columns={"label": "backend", "lo": "ci_low", "hi": "ci_high", "pass_k": "pass_at_10"})
    det["backend"] = det["backend"].map(LABEL)
    det.to_csv(OUT / "Supplementary_Table_expB_invocation_determinism.csv", index=False)

    bias = pd.read_csv(ANALYSIS / "bbias_method_counts.csv")
    bias = bias.rename(columns={"label": "backend", "(omitted)": "schema_default"})
    bias["backend"] = bias["backend"].map(LABEL)
    bias.to_csv(OUT / "Supplementary_Table_expB_method_choice.csv", index=False)

    agr = pd.read_csv(ANALYSIS / "cross_backend_agreement.csv")
    keep = pd.concat([agr[(agr.a == a) & (agr.b == b)] for a, b in VERSION_PAIRS])
    keep = keep.assign(a=keep.a.map(LABEL), b=keep.b.map(LABEL))
    keep.to_csv(OUT / "Supplementary_Table_expB_version_pair_agreement.csv", index=False)
    print("written to", OUT)


if __name__ == "__main__":
    main()
