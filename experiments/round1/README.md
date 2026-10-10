# Round 1 experiments

These experiments were run for the first submission (February to May 2026) and are still reported in the resubmitted manuscript. The language-model experiments used ChatSpatial 1.2.10 (`pip install chatspatial==1.2.10`). The LLM-free reproduction image in `reproduction/` is built on the published ChatSpatial 1.5.9 image.

Run the scripts from this directory (`python scripts/<name>.py`) so that sibling imports resolve. Results are written to `results/`. Scripts that execute ChatSpatial on data expect a ChatSpatial checkout with its `data/` folder (set `CHATSPATIAL_CODE_DIR`); the end-to-end and DLPFC drivers also need STAgent and SpatialAgent checkouts (`CHATSPATIAL_BENCHMARKS_DIR`, or `STAGENT_ROOT` and `SPATIALAGENT_ROOT`). API keys are read from the environment or from a `.env` file at the repository root: `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`, default `https://api.openai.com/v1`).

## Scripts and the manuscript results they back

| Script | Result table(s) in `results/` | Manuscript |
|---|---|---|
| `reproducibility_analysis.py` | `reproducibility_analysis.csv` | Parameter constraint counts per tool (Results, typed schemas) |
| `ablation_invocation.py` | `ablation/invocation/ablation_invocation_raw.jsonl`, `ablation_invocation_results.csv`, `ablation_invocation_summary.txt` | 720-trial invocation ablation: Fig. 4a, validation rates 98.3/93.8/20.4% (manipulation check), 694 of 720 parsed |
| `ablation_e2e.py` | `ablation/e2e/ablation_e2e_raw.jsonl`, `ablation_e2e_results.csv` | 270-trial end-to-end ablation on OSCC Sample 1: Fig. 4b; 6 of 87 bare-schema calls stopped at the dependency check; Supplementary Table 6 cases 2-4 |
| `ablation_analysis.py` | `ablation/analysis/ablation_metrics_bootstrap.csv`, `ablation_statistical_tests.csv`, `ablation_summary.txt` | Fig. 4c concordance (ARI, Jaccard@100, Pearson r) with bootstrap CIs |
| `prompt_sensitivity.py` | `sensitivity/prompt_sensitivity_raw.jsonl`, `prompt_sensitivity_results.csv` | 375-trial prompt sensitivity: Fig. 4d, 95.7% correct tool |
| `prompt_sensitivity_analysis.py` | `sensitivity/analysis/sensitivity_cross_paraphrase.csv`, `sensitivity_summary.txt` | Fig. 4e, 98.7% cross-paraphrase agreement |
| `cross_system_comparison.py` | `cross_system/cross_system_raw.jsonl`, `cross_system_results.csv` | 480 new trials with STAgent and SpatialAgent contexts (1,200 with the ablation) |
| `cross_system_analysis.py` | `cross_system/analysis/cross_system_metrics.csv`, `cross_system_aggregate.csv` | Fig. 4f,g: 98.3% versus 87.5% and 90.4% tool selection; 100% versus 87.8% and 94.5% consistency |
| `e2e_benchmark.py` (with `e2e_driver_stagent.py`, `e2e_driver_spatialagent.py`) | `e2e_benchmark/e2e_benchmark_raw.jsonl` | 45-trial lymph node benchmark: Fig. 4h (15, 11 and 12 completed trials) |
| `e2e_benchmark_analysis.py` | `e2e_benchmark/e2e_benchmark_results.csv`, `e2e_benchmark_summary.txt` | Fig. 4h,i summary |
| `e2e_pairwise_concordance.py` | `e2e_benchmark/e2e_pairwise_concordance.csv` | Fig. 4i points (pairwise ARI, Jaccard@100, Jaccard@50; SpatialAgent ARI 0.45 and Jaccard@50 0.32) |
| `casestudy_reproducibility.py` | `casestudy_reproducibility/casestudy_repro_raw.jsonl`, `casestudy_repro_metrics.csv`, `casestudy_repro_summary.txt` | 160-trial CARD concordance on OSCC Samples 1 and 9: 80 of 80 valid with full schemas, Pearson r = 1.000, 0 of 80 valid without schemas |
| `dlpfc_prepare_data.py` | `dlpfc_benchmark/ground_truth/*_labels.csv` | Inputs and expert labels of DLPFC sections 151507, 151669 and 151673 |
| `dlpfc_benchmark.py`, `dlpfc_benchmark_analysis.py` | `dlpfc_benchmark/dlpfc_benchmark_raw.jsonl`, `dlpfc_benchmark_results.csv`, `dlpfc_benchmark_per_trial.csv` | Three-section benchmark with Claude Sonnet 4. Its comparison of systems is replaced in the manuscript by the 12-section benchmark (`experiments/refusals_and_head_to_head`); its 30 recorded ChatSpatial calls and their ARI are the reference values of the LLM-free reproduction |

Models: Gemini 2.5 Flash, Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) and GPT-5.4 (`gpt-5.4`) for the ablation, prompt sensitivity and cross-system experiments; the same three and Claude Sonnet 4.5 for the CARD concordance; Claude Sonnet 4 (`claude-sonnet-4-20250514`) for the lymph node and three-section DLPFC benchmarks. All calls used temperature 1.0.

The `*_raw.jsonl` files hold one record per trial, including the model response. Local file paths in them are shortened to `<workspace>/`.

## LLM-free reproduction (`reproduction/`)

`reproduction/reproduce_key_results.py` is the `chatspatial-reproduce` command of the reproduction image named in Code availability. It downloads the input archive (https://github.com/cafferychen777/ChatSpatial/releases/tag/repro-inputs-2026-10-08), re-executes the 30 recorded `identify_spatial_domains` calls of `dlpfc_benchmark_raw.jsonl` and the 160 recorded calls of `casestudy_repro_raw.jsonl` without any LLM, and compares each value with the reference values in `results/dlpfc_benchmark/` and `results/casestudy_reproducibility/`. The published image is

```
docker pull cafferyyang777/chatspatial@sha256:ac3ba1ed8235c3d53ca72005d8ad2556cac007f5ddb932ae3e1d1241fa2cf9d8
docker run --rm -v "$PWD/out:/outputs" cafferyyang777/chatspatial@sha256:ac3ba1ed8235c3d53ca72005d8ad2556cac007f5ddb932ae3e1d1241fa2cf9d8 chatspatial-reproduce
```

To build it from this directory, see the header of `reproduction/Dockerfile` (build context `experiments/round1`).

## Raw data

- Lymph node: 10x Genomics public dataset "Human Lymph Node" (Visium, 4,035 spots), expected at `$CHATSPATIAL_CODE_DIR/data/processed_datasets/visium/human_lymph_node.h5ad`.
- OSCC Samples 1 and 9: GSE208253, processed as in the case study, expected under `$CHATSPATIAL_CODE_DIR/data/geo_data/analyzed_projects/GSE208253/processed_h5ad/`; HNSCC reference GSE103322 under `$CHATSPATIAL_CODE_DIR/data/reference_datasets/puram_hnscc/`. The prepared files are also in the reproduction input archive.
- DLPFC: `spatialLIBD::fetch_data()`, exported by `dlpfc_prepare_data.py`.

## Superseded (`superseded/`)

Single-model and cross-model determinism (`determinism_experiment.py`, `determinism_multimodel.py`) and the code-generation baseline (`codegen_hallucination.py`) were part of the first submission and are not reported in the resubmission. They are kept with their result tables for reference.
