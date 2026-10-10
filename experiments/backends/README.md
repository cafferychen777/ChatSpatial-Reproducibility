# Experiment B: tool selection and execution across eight language-model backends

Manuscript: Results section "Tool selection is determined by the typed interface and holds across eight language
models", Methods "Backend dependence", Figure B (`figures/figureB_backends.py`), Supplementary Table 8 and the
`Supplementary_Table_expB_*.csv` files.

The evaluation rules were written before the runs and are in `expB_rules_2026-10-08.txt` (design, per-trial outcome
definitions, fixtures, statistics, and a dated change log).

## Design in brief

Eight prompts (the prompts of the schema ablation, `experiments/round1/scripts/ablation_invocation.py`) were sent to
eight backends under two constraint mechanisms (JSON object requested in the prompt; native tool calling) and three
levels of schema information (full, bare, minimal), ten repetitions per cell (3,840 trials). Two further arms used
native tool calling with full schemas: a description ablation (descriptions of two tools exchanged, with real or
neutral tool names) and seven open-ended prompts that name no method. Every call that was parseable, named the
expected tool and passed Pydantic validation was executed with ChatSpatial 1.5.5 on a fixed dataset for its prompt.

Backends: Claude Haiku 4.5 (`claude-haiku-4-5-20251001`), Claude Haiku 5.5, Claude Sonnet 5.5 and Claude Opus 5.5
(Anthropic API, 8 October 2026); `gpt-6-sol` and `gpt-6.1-sol` (9 October 2026); `google/gemma-4-12B-it` and
`Qwen/Qwen3.8-27B-FP8` served with vLLM 0.31.0 on one NVIDIA H100 GPU inside the same Slurm job as the driver
(9 October 2026).

## Scripts (`scripts/`)

| Script | Role |
|---|---|
| `expB_common.py` | Conditions: JSON system prompts (imported unchanged from `ablation_invocation.py`), native tool definitions for full, bare and minimal schemas, description-ablation and method-choice arms. Standard library only. |
| `expB_invocation.py` | LLM calls only. One JSONL record per trial with the full request (without credentials), response, timing and parsed call. Backends: `anthropic` (`ANTHROPIC_API_KEY`), `openai` (`OPENAI_API_KEY`; endpoint `OPENAI_BASE_URL`, default `https://api.openai.com/v1`), `openai_compat` (a vLLM server given with `--base-url`). Resumable. |
| `expB_fixtures.py` | Builds the execution datasets (DLPFC 151673; OSCC Sample 1 with marker-rule cell types and gene positions; Puram et al. reference; squidpy MERFISH slice; scVelo pancreas). Input file names are listed in its docstring. |
| `expB_execute.py` | Validates every call against the ChatSpatial Pydantic models, executes each unique validated call once in an isolated subprocess (45 min limit), checks the output, and writes `exec_cache.jsonl` and `trial_outcomes.csv`. |
| `expB_analysis.py` | Rates, cluster bootstrap over prompts, error decomposition, invocation determinism, cross-backend and version-pair agreement, description ablation, method choice, and the GLMMs (R: `lme4`, `brglm2` under separation). Writes `analysis/*.csv` and `expB_summary.txt`. |
| `compat/sitecustomize.py` | Restores the sparse-matrix `.A` alias that SpaGCN 1.2.7 uses (placed on `PYTHONPATH` of the execution subprocesses). |
| `pyg_shims/` | Masks four optional compiled torch-geometric extensions that crashed on import in the cluster environment (they are optional for STAGATE and GraphST). |
| `expB_build_env_v155.sbatch` | Builds the execution environment: Python 3.12, conda-forge R 4.4.3, `chatspatial[full]==1.5.5`, `rpy2==3.5.17`. It takes the R packages spacexr, CellChat, SPARK and CARD (GitHub: dmcable/spacexr, jinworks/CellChat, xzhoulab/SPARK, YMa-lab/CARD) and lme4 and brglm2 (CRAN) from the R libraries `$CHATSPATIAL_WORK_DIR/expB/no_user_rlib` and `rlib_v155`, which are installed with `remotes::install_github` and `install.packages` for R 4.4.3. |
| `expB_anthropic.sbatch` | Slurm array, one task per Claude model, all three arms. |
| `expB_vllm_job.sbatch` + `expB_ow_driver.sh` | One GPU job per open-weight model: starts vLLM on 127.0.0.1 inside the job, runs a pilot gate, then all arms. |
| `expB_rerun_v155.sbatch` | Executes every recorded call under ChatSpatial 1.5.5 (no LLM calls) and runs the analysis. |

## Run order

Set `CHATSPATIAL_WORK_DIR` to a cluster work directory; invocation records, fixtures and execution outputs go to
`$CHATSPATIAL_WORK_DIR/expB/`. Submit the Slurm files from `experiments/backends/scripts/` after creating
`expB_logs/` there.

1. `sbatch expB_build_env_v155.sbatch` (4 h, 8 CPUs, 16 GB).
2. `python expB_fixtures.py` on a compute node, with the inputs in `$CHATSPATIAL_WORK_DIR/expB/inputs/`
   (DLPFC 151673 from `experiments/round1/scripts/dlpfc_prepare_data.py`).
3. LLM calls:
   - Claude: `sbatch expB_anthropic.sbatch` (1 CPU, 1 GB per model).
   - GPT: `python expB_invocation.py --backend openai --model gpt-6-sol --experiment {main,desc,bias} --workers 3`
     and the same for `gpt-6.1-sol` (no cluster resources needed).
   - Open-weight: `MODEL_KEY=gemma4-12b DRIVER_CMD='bash expB_ow_driver.sh' sbatch expB_vllm_job.sbatch`, and
     `MODEL_KEY=qwen38-27b` (one H100, 8 CPUs, 64 GB, about 2 h each).
4. `sbatch expB_rerun_v155.sbatch` (12 CPUs, 64 GB, up to 12 h; GPU not required).
5. Figure B and the Supplementary Table 8 values: `python figures/figureB_backends.py` from the repository root.

`python scripts/expB_analysis.py` with no environment variables reruns the analysis from the tables in `results/`
(R with lme4 and brglm2 is needed for the GLMM section) and writes to `results/analysis/`.

## Results (`results/`)

| File | Content |
|---|---|
| `trial_outcomes.csv.gz` | One row per trial (6,320 rows: 3,840 main, 1,920 description ablation, 560 method choice): backend, arm, mechanism, schema level, prompt, repetition, parse status, raw and validated parameters, tool correctness, execution outcome, completion, silent failure, explicit failure. |
| `exec_cache.jsonl.gz` | One record per unique executed call with its outcome and the output check details. |
| `analysis/rates_by_backend_condition.csv` | Rates per backend, mechanism and schema level with cluster-bootstrap intervals (Fig. B a,b; Supplementary Table 8). |
| `analysis/bmis_decomposition.csv` | Format failures versus genuine mis-selections. |
| `analysis/invocation_determinism_cells.csv`, `analysis/binv_per_backend.csv` | Modal-call share and pass^10 (Fig. B e; Supplementary Table 8). |
| `analysis/cross_backend_agreement.csv` | Agreement on constrained parameters, including the two version pairs. |
| `analysis/bdesc_by_backend.csv` | Description ablation (Fig. B d; Supplementary Table 8). |
| `analysis/bbias_method_counts.csv` | Method choice on open-ended prompts (Fig. B f). |
| `analysis/glmm_input.csv`, `analysis/expB_summary.txt` | GLMM input and the full printed analysis, including the odds ratios of Fig. B c. |

The full per-trial model responses (about 40 MB of JSONL) are not stored here; `trial_outcomes.csv.gz` holds the
parsed call and every outcome of each trial, and `expB_invocation.py` regenerates the responses.
