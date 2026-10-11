# ChatSpatial reproducibility repository

Code and result tables for the manuscript

> **ChatSpatial: Schema-Enforced Agentic Orchestration for Reproducible and Cross-Platform Spatial Transcriptomics**
>
> Chen Yang, Xianyang Zhang, Jun Chen

This repository holds, for every figure, table and number of the manuscript, the script that produced it and the
small result table it was computed from. Raw data are public and are not stored here (see "Raw data").
ChatSpatial itself is developed at https://github.com/cafferychen777/ChatSpatial; release 1.5.9 is archived on
Zenodo (https://doi.org/10.5281/zenodo.23273054).

## Layout

```
experiments/
  robustness/                 Schema-derived parameter robustness (Figure 4)
  backends/                   Tool selection and execution across eight LLM backends (Figure 3, Supplementary Table 8)
  refusals_and_head_to_head/  Assumption-violation probes; 12-section DLPFC head-to-head (Figure 2a,b)
  case_studies/               OSCC (Figure 5) and HGSOC (Figure 6) case studies; deconvolution concordance (Supplementary Table 7)
  round1/                     Schema ablation, prompt sensitivity, cross-system and lymph-node benchmarks (Figure 2c-k),
                              CARD concordance, and the LLM-free reproduction (round1/reproduction/)
figures/                      Figure scripts and the Figure 1 schematic (figure1.svg)
source_data/                  Source Data of every figure panel and table (one CSV per sheet; Index.csv lists them)
supplementary_tables/         Supplementary Tables 1-7 and the CSV files of Supplementary Table 8
scripts/                      Builders of the supplementary tables and of the Source Data workbook
```

Each directory under `experiments/` has a README with its scripts, run order, cluster resources and result tables.
Rules that were fixed before the runs are kept next to the code (`robustness/PREREGISTRATION.md`,
`backends/expB_rules_2026-10-08.txt`, `refusals_and_head_to_head/expBX_rules_2026-10-09.txt`,
`case_studies/deconvolution_concordance/mdeconv_rules_2026-10-09.txt`).

## Software versions

| Analysis | ChatSpatial |
|---|---|
| Schema ablation, prompt sensitivity, cross-system comparison, lymph-node benchmark, CARD concordance (round 1) | 1.2.10 |
| Backend study (Figure 3); deconvolution method concordance (Supplementary Table 7) | 1.5.5 |
| Assumption-violation probes | 1.5.6 |
| OSCC and HGSOC case studies (Figures 5 and 6) | 1.5.7 |
| Parameter robustness (Figure 4) | 1.5.5 environment; units that use the Gaussian-mixture substitute for mclust re-run with the 1.5.9 code (see `experiments/robustness/README.md`) |
| DLPFC head-to-head (Figure 2a,b); reproduction image | 1.5.9 |

Install a release with `pip install "chatspatial[full]==<version>"` (or `uv pip install`). STAgent (git 054cc64)
and SpatialAgent (git e51ec0f) were run with their original configuration; the client changes needed for the
language model are listed in `experiments/refusals_and_head_to_head/README.md`.

The figure and statistics scripts need only Python 3.11 or later with numpy, pandas, scipy, statsmodels,
matplotlib and Pillow. They read the tables in this repository and run in seconds on a laptop. The experiments
themselves ran on Slurm clusters; their scripts read the cluster work directory from the environment variable
`CHATSPATIAL_WORK_DIR`. API keys are read from `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` (endpoint `OPENAI_BASE_URL`,
default `https://api.openai.com/v1`) and `GEMINI_API_KEY`.

## Figures and tables

Paths are relative to the repository root. `E/` stands for `experiments/`.
The figure scripts keep the file names of earlier drafts: `figures/figure4.py` draws Figure 2, `figureB_backends.py`
Figure 3, `figureA_robustness.py` Figure 4, `figure2_oscc.py` Figure 5 and `figure3_hgsoc.py` Figure 6.

| Manuscript item | Script | Input tables |
|---|---|---|
| Figure 1 (schematic) | none (`figures/figure1.svg`) | none |
| Table 1 (platform compatibility) | `scripts/build_submission_package.py` (writes `source_data/Table1_platform.csv`) | curated content in the script |
| Figure 2a,b | `figures/figure4.py`; `E/refusals_and_head_to_head/scripts/expBX_analysis.py` | `E/refusals_and_head_to_head/results/h2h_v159/h2h_per_run.csv`, `h2h_cross_rep.csv`, `h2h_summary.json` |
| Figure 2c | `figures/figure4.py`; `E/round1/scripts/ablation_invocation.py` | `E/round1/results/ablation/invocation/ablation_invocation_results.csv`, `ablation_invocation_raw.jsonl` |
| Figure 2d,e | `figures/figure4.py`; `E/round1/scripts/ablation_e2e.py`, `ablation_analysis.py` | `E/round1/results/ablation/e2e/ablation_e2e_results.csv`; `ablation/analysis/ablation_metrics_bootstrap.csv` |
| Figure 2f,g | `figures/figure4.py`; `E/round1/scripts/prompt_sensitivity.py`, `prompt_sensitivity_analysis.py` | `E/round1/results/sensitivity/prompt_sensitivity_raw.jsonl`, `prompt_sensitivity_results.csv` |
| Figure 2h,i | `figures/figure4.py`; `E/round1/scripts/cross_system_comparison.py`, `cross_system_analysis.py` | `E/round1/results/cross_system/analysis/cross_system_metrics.csv` |
| Figure 2j,k | `figures/figure4.py`; `E/round1/scripts/e2e_benchmark.py`, `e2e_benchmark_analysis.py`, `e2e_pairwise_concordance.py` | `E/round1/results/e2e_benchmark/e2e_benchmark_raw.jsonl`, `e2e_benchmark_results.csv`, `e2e_pairwise_concordance.csv` |
| Figure 3a-f | `figures/figureB_backends.py`; `E/backends/scripts/expB_analysis.py` | `E/backends/results/analysis/*.csv`, `expB_summary.txt`; `E/backends/results/trial_outcomes.csv.gz` |
| Figure 4a-d | `figures/figureA_robustness.py`; tables from `E/robustness/scripts/export_figA.py`; numbers from `E/robustness/scripts/evaluate.py` and `pooled.py` | `E/robustness/results/strategy_scores_all.csv`, `sensitivity_dlpfc.csv`, `noise_retention.csv`, `platform_method_means.csv`; per-configuration tables `configurations_*.csv.gz`; `evaluation_summary.txt` |
| Figure 5a-c,g | `figures/figure2_oscc.py` | `E/case_studies/oscc/results/tables/spots/`, `proportions/s1.csv`; H&E images and spot positions from GEO GSE208253 (`OSCC_GEO_DIR`) |
| Figure 5d,e | `figures/figure2_oscc.py`; statistics `E/case_studies/compute_case_study_stats.py` | `E/case_studies/oscc/results/stats/per_sample_tc_le_enrichment.csv`, `consensus_tc_le_patterns.csv`, `consensus_binomial_ci.csv` |
| Figure 5f | `figures/figure2_oscc.py` | `E/case_studies/oscc/results/tables/cellchat_tc_le_caf_cancer_interactions.csv`, `cellchat_tc_le_lr_pairs.csv` |
| Figure 5h,i | `figures/figure2_oscc.py`; `E/case_studies/compute_case_study_stats.py` | `E/case_studies/oscc/results/stats/svg_target_genes_summary.csv`, `moranI_cross_sample_stats.csv`; `source_data/Fig5h_morans.csv` |
| Figure 6a-d | `figures/figure3_hgsoc.py`; statistics `E/case_studies/hgsoc/scripts/hgsoc157_stats.py` | `E/case_studies/hgsoc/results/tables/spots/`; `E/case_studies/hgsoc/results/stats/hgsoc_patient_summary.csv`, `cnv_score_per_patient_stats.csv` |
| Figure 6e,f | `figures/figure3_hgsoc.py`; `hgsoc157_stats.py` | `E/case_studies/hgsoc/results/tables/moranI_P3.csv`, `cnv_cluster_markers_P3.csv`; `results/stats/cliffs_delta_cnv_vs_background.csv`, `celltype_marker_moran_ranks.csv` |
| Supplementary Table 1 | `scripts/build_supplementary_tables.py` | `E/case_studies/oscc/results/stats/sample_level_tests.csv`, `effect_size_summary.csv` |
| Supplementary Tables 2, 3, 4 | `scripts/build_supplementary_tables.py` | curated content in the script (method inventory of release 1.5.9; 31 test scenarios; comparison of agents) |
| Supplementary Tables 5, 6 | `scripts/build_submission_package.py` | curated content in the script; cases 2-4 of Table 6 are observed in `E/round1/results/ablation/e2e/ablation_e2e_raw.jsonl` |
| Supplementary Table 7 | `E/case_studies/oscc/scripts/oscc157_postrun.py` (step C) with `E/case_studies/deconvolution_concordance/scripts/mdeconv_analyze.py` | `E/case_studies/deconvolution_concordance/results/runs/`, `results/analysis/` |
| Supplementary Table 8 | `scripts/build_expB_supplementary_tables.py`; `E/backends/scripts/expB_analysis.py` | `E/backends/results/analysis/` |
| Source Data | `scripts/build_submission_package.py` (Figure 5 sheets: `oscc157_postrun.py`, step D; Figure 3 and 4 sheets: the figure scripts) | `source_data/` |

Numbers given only in the text come from the same tables:

| Result | Where |
|---|---|
| Typed interface: 20 tools, 449 parameters, 375 constrained (83.5%), 74 free text; 65 methods in 13 categories | `E/schema_constraints/constraint_surface_chatspatial_1.5.9.csv` (from `run_constraint_surface_analysis()` in `E/round1/scripts/reproducibility_analysis.py`, run on ChatSpatial 1.5.9, commit 141f245); method list in `supplementary_tables/Supplementary_Table_2_Integrated_Methods.csv` |
| Schema ablation, prompt sensitivity, cross-system comparison, lymph-node benchmark, CARD concordance (160 calls, Pearson r = 1.000) | `E/round1/README.md` |
| Eight backends: 1,280 of 1,280 correct tools under full schemas, 5 silent failures in 3,840 trials, description ablation, odds ratios, method choice | `E/backends/results/analysis/expB_summary.txt`, `rates_by_backend_condition.csv`, `bbias_method_counts.csv`, `trial_outcomes.csv.gz` |
| Assumption-violation probes (0, 21 and 3 of 40 silent failures for SpatialAgent; 0 to 3 of 40 for ChatSpatial) | `E/refusals_and_head_to_head/results/probes_{sonnet,gemma4,qwen38}/probe_summary.json`, `probe_per_run.csv`; judge and audit records in `results/judge/` |
| DLPFC head-to-head (mean ARI 0.461, 0.445, 0.404, 0.378; cross-replicate ARI 0.776, 0.538, 0.625, 0.990) | `E/refusals_and_head_to_head/results/h2h_v159/h2h_summary.json` |
| Parameter robustness (2,724 configurations per section; default 0.420, rule 0.441, best 0.565; Kendall's W) | `E/robustness/results/evaluation_summary.txt` |
| OSCC and HGSOC numbers | `E/case_studies/README.md` |

## Rerunning

Figures (from the repository root; written to `figures/`, Figure 3 and 4 source data to `source_data/`):

```
cd figures
python figureA_robustness.py
python figureB_backends.py
python figure3_hgsoc.py
python figure4.py
OSCC_GEO_DIR=/path/to/GSE208253 python figure2_oscc.py
```

Statistics from the committed tables: `python experiments/robustness/scripts/evaluate.py dlpfc155 noise155 platforms155`,
`python experiments/case_studies/compute_case_study_stats.py`, and the commands in each experiment README.

The experiments themselves need the raw data, a Slurm cluster and, for the LLM experiments, API access or a GPU for
the open-weight models. Each experiment README gives the order of the scripts.

LLM-free reproduction: the reproduction image of release 1.5.9 provides one command, `chatspatial-reproduce`, that
downloads the input archive (https://github.com/cafferychen777/ChatSpatial/releases/tag/repro-inputs-2026-10-08),
re-executes 30 recorded spatial-domain calls on three DLPFC sections and the 160 recorded calls of the CARD
concordance without any LLM, and reports the deviation of each value from its recorded reference:

```
docker pull cafferyyang777/chatspatial@sha256:ac3ba1ed8235c3d53ca72005d8ad2556cac007f5ddb932ae3e1d1241fa2cf9d8
docker run --rm -v "$PWD/out:/outputs" cafferyyang777/chatspatial@sha256:ac3ba1ed8235c3d53ca72005d8ad2556cac007f5ddb932ae3e1d1241fa2cf9d8 chatspatial-reproduce
```

The release image without the reproduction layer is
`cafferyyang777/chatspatial@sha256:a75d48c87200182c24063f28a2433ecda8848a3ee616ef1884f170c56b366e1e`, mirrored as
`ghcr.io/cafferychen777/chatspatial` with the same digest. The script, Dockerfile and reference values of the
reproduction are in `experiments/round1/reproduction/` and `experiments/round1/results/`.

## Raw data

All datasets are public.

| Dataset | Source | Used in |
|---|---|---|
| OSCC Visium, 12 sections | GEO GSE208253 (Arora et al. 2023) | Case study (Figure 5), schema ablation, CARD concordance, backend study, deconvolution concordance |
| HNSCC single-cell reference | GEO GSE103322 (Puram et al. 2017) | OSCC deconvolution |
| HGSOC Visium, 8 sections, and single-cell reference | GEO GSE211956 (Denisenko et al. 2024) | Case study (Figure 6) |
| DLPFC Visium, 12 sections | spatialLIBD (Maynard et al. 2021), `spatialLIBD::fetch_data("sce")` | Head-to-head, parameter robustness, backend study, reproduction |
| STARmap mouse visual cortex, section 20180505_BY3_1kgenes | Wang et al. 2018 | Parameter robustness |
| MERFISH, Allen Brain Cell Atlas C57BL6J-638850, section 47 | Yao et al. 2023 | Parameter robustness |
| Human lymph node Visium | 10x Genomics public datasets | Lymph-node benchmark (Figure 2j,k) |
| MERFISH mouse hypothalamus slice; pancreas endocrinogenesis | squidpy and scVelo example datasets | Backend study (spatially variable genes; trajectory) |

Scripts that download or prepare these inputs are named in each experiment README
(`E/robustness/scripts/dlpfc_prepare_data.py`, `stage_external.py`; `E/backends/scripts/expB_fixtures.py`;
`E/refusals_and_head_to_head/scripts/expBX_fixtures.py`; `E/case_studies/*/scripts/*prep*`).

## License

The code in this repository is released under the MIT license, as is ChatSpatial.
