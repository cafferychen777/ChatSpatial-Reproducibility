# Assumption-violation probes and DLPFC head-to-head (Experiment B-X)

This directory holds the code and result tables of three analyses in the manuscript:

- **Assumption-violation probes** (Results, "ChatSpatial refuses analyses the data cannot support and states the reason"; Methods, "Assumption-violation probes"). Eight requests that each violate a documented precondition, sent five times to ChatSpatial, STAgent and SpatialAgent, driven by Claude Sonnet 5.5, Gemma 4 12B or Qwen3.8 27B. ChatSpatial 1.5.6.
- **DLPFC head-to-head** (Results, "ChatSpatial matches the accuracy of code-generating agents and returns far more reproducible results"; Figure 2a,b; Methods, "Ground-truth benchmark"). All 12 DLPFC sections, ten replicates per section, Claude Sonnet 5.5. ChatSpatial 1.5.9 as an agent with all 20 tools and as a single `identify_spatial_domains` call; STAgent (git 054cc64) and SpatialAgent (git e51ec0f) with their original configuration.
- **DLPFC head-to-head with an open-weight model** (Results, section 2.3, "ChatSpatial matches the accuracy of code-generating agents and returns far more reproducible results", Qwen3.8 27B paragraph; Methods, "Ground-truth benchmark"). The same 12 sections, ten replicates and four systems, driven by Qwen3.8 27B (`Qwen/Qwen3.8-27B-FP8`) served with vLLM 0.31.0 on one NVIDIA GB10 GPU host, thinking off, served default sampling. ChatSpatial 1.5.9 as an agent and as a single call; STAgent and SpatialAgent with their original configuration.

The rules were written before any run and are in `expBX_rules_2026-10-09.txt`, with a dated change log in its section 6. Two entries matter for reading the versions: the ChatSpatial arms ran on 1.5.6 instead of the planned 1.5.5 (a fix that lets tool error text reach the MCP client), and the two ChatSpatial arms of the head-to-head were re-run in full on 1.5.9 after a bug fix in the Gaussian mixture that replaces R mclust (tag `v159`). The competitor arms of the head-to-head were not re-run.

The Qwen3.8 27B arm of the head-to-head has its own rules, `h2h_qwen_rules_2026-10-10.txt`, written before any run of the arm (pointer entry of 2026-10-10 in section 6 of the parent rules). Its change log records the harness fixes after the smoke test (the open-weight client path now exports head-to-head labels with the code of the Sonnet path; the nested `params` argument, which Qwen sends as a JSON string, is decoded before the export; the success rule accepts the OpenAI finish reason `stop` as the equivalent of `end_turn`) and the addition of the single-call arm, written before any run of it. Smoke-test runs are excluded from all analyses.

## Result tables (`results/`)

| Path | Content | Manuscript |
|---|---|---|
| `h2h_v159/h2h_per_run.csv` | one row per run (480): ARI, NMI, method, labels column, coverage | Figure 2a,b; head-to-head text |
| `h2h_v159/h2h_cross_rep.csv` | cross-replicate ARI per section and system | Figure 2b; cross-replicate ARI |
| `h2h_v159/h2h_summary.json` | system means, section-bootstrap CIs, paired differences, sign-flip P, method counts | mean ARI 0.461 / 0.445 / 0.404 / 0.378; cross-replicate ARI 0.776 / 0.538 / 0.625 / 0.990; paired differences; SpaGCN 66 and STAGATE 53 runs |
| `h2h_qwen38/h2h_per_run.csv` | one row per run (480) of the Qwen3.8 27B arm, same columns as `h2h_v159` | Qwen3.8 27B head-to-head text |
| `h2h_qwen38/h2h_cross_rep.csv` | cross-replicate ARI per section and system, Qwen3.8 27B | cross-replicate ARI 0.550 / 0.264 / 0.361 / 1.000 |
| `h2h_qwen38/h2h_summary.json` | same fields as `h2h_v159/h2h_summary.json`; `scorable` is the number of runs that returned labels and `ari_itt_mean` the mean ARI with a run without labels counted as 0 | runs with labels 120 / 58 / 98 of 120 (ChatSpatial agent / STAgent / SpatialAgent); mean ARI with no labels counted as 0: 0.312 (95% CI 0.274 to 0.348) / 0.106 / 0.215; paired differences +0.206 (0.156 to 0.254) and +0.097 (0.053 to 0.140); on runs with labels +0.098 and +0.050; single call 0.375 (0.327 to 0.418) |
| `h2h_v159/fig2ab_trials.csv`, `fig2ab_sections.csv`, `fig2b_summary.csv` | tables written by `figures/figure4.py` for panels a and b | Figure 2a,b source data |
| `probes_sonnet/probe_per_run.csv`, `probe_summary.json` | per-run judged and audited class; silent failures per arm with probe-bootstrap CI; differences and sign-flip tests | Claude Sonnet 5.5: 0 of 40 in every arm; 40 refusals by ChatSpatial; 11 valid completions by the agents |
| `probes_gemma4/...` | same, Gemma 4 12B | ChatSpatial 3 of 40, STAgent 1, SpatialAgent 21; difference -45 points (-78 to -10), P = 0.094 |
| `probes_qwen38/...` | same, Qwen3.8 27B | ChatSpatial 1 of 40, STAgent 2, SpatialAgent 3 |
| `judge/judgements*.jsonl` | judge output per run (rubric fields and evidence quotes) | classification |
| `judge/audit_result_*.jsonl`, `judge/audit_overrides.jsonl` | hand-audit verdicts and the overrides with reasons (Sonnet 0, Gemma 4, Qwen 1) | audited class |
| `tasks/*.jsonl` | task lists that were run (`all_full`, `rerun_quota`, `cs159_h2h`, `ow_*_probe`, `ow_qwen38_h2h`, `ow_qwen38_h2h_single`) and analysed (`probe_full`, `h2h_v159`, `ow_*_probe`, `ow_qwen38_h2h_all`) | |

The per-run records (transcripts, API logs, generated code and outputs; about 770 MB) are not included. They are regenerated by the run scripts below.

## Scripts (`scripts/`)

| Script | Role |
|---|---|
| `expBX_common.py` | shared definitions: the eight probes (request text, violated precondition, requested output), the DLPFC prompts, task lists, run directories, limits (45 min per run, 16,000 output tokens, 30 agent turns) |
| `expBX_fixtures.py` | builds the probe inputs and the 12 head-to-head inputs (raw counts in `layers['counts']`, full-resolution pixel coordinates, no labels) |
| `expBX_make_tasks.py` | writes task lists that interleave systems within each replicate |
| `expBX_run_task.py` | runs one task in an isolated container (Apptainer on the CPU cluster, Docker on the GPU host of the open-weight arm): read-only input and code, private run directory and HOME |
| `expBX_chatspatial_agent.py` | ChatSpatial agent arm: all tools of the MCP server with their published schemas, native tool calling, tools called through an in-memory MCP client session; Anthropic API or, for the open-weight arms, the OpenAI-compatible endpoint of vLLM; both paths export the head-to-head labels with the same code |
| `expBX_single_call.py` | single-call ChatSpatial arm of the head-to-head (round-1 protocol); imports the schema text and parser from `ablation_invocation.py`; the same call goes to the OpenAI-compatible vLLM endpoint when `EXPBX_BACKEND=openai` (Qwen3.8 27B arm) |
| `ablation_invocation.py` | full-schema system text, response instruction, tool-name alias map and JSON parser of the round-1 ablation (only the parts the single-call arm uses) |
| `expBX_driver_stagent.py`, `expBX_driver_spatialagent.py`, `expBX_competitor_common.py` | competitor drivers with the original configuration and full API logging |
| `expBX_judge.py` | judge (Claude Sonnet 5.5): fills the rubric fields for each probe run; the class is computed by a fixed rule |
| `expBX_audit_export.py`, `expBX_audit_extra.py` | export the runs for the hand audit (all judged silent failures, a seeded 20% of the others, and all valid completions) |
| `expBX_analysis.py` | `h2h` and `probe` analyses (scoring, cross-replicate ARI, cluster bootstraps, sign-flip tests) |
| `expBX_build_v156.sbatch`, `expBX_build_v159.sbatch` | build the ChatSpatial 1.5.6 and 1.5.9 environments |
| `expBX_array.sbatch` | one task per Slurm array element (4 CPUs, 40 GB, no GPU) |
| `expBX_py.sbatch` | runs the judge and the analysis on a compute node |
| `expBX_openweight.sbatch` | open-weight arm: vLLM 0.31.0 on one NVIDIA GB10 (localhost endpoint) and the probe runs of all three systems, four at a time |
| `expBX_build_v159_openweight.sbatch` | builds the aarch64 ChatSpatial 1.5.9 environment on the GPU host (same recipe as `expBX_build_v159.sbatch`) |
| `expBX_openweight_h2h.sbatch` | Qwen3.8 27B head-to-head: vLLM 0.31.0 with the serve settings of the Qwen probe arm and the runs of the ChatSpatial agent, STAgent and SpatialAgent, four at a time, in chained 24-hour jobs (a run starts only if at least 47 minutes remain) |
| `expBX_openweight_h2h_single.sbatch` | the same job for the single-call arm; with `GATE=1` it fails unless every run finished with labels (gate after the smoke run) |
| `run_one_in_job.sh` | runs one task inside a running head-to-head job, with the environment of the job script |
| `expBX_ow_h2h_progress.py` | progress of the Qwen3.8 27B head-to-head queue (called by the job scripts) |
| `expBX_ow_h2h_score.py` | scores the Qwen3.8 27B head-to-head runs on the GPU host with the unchanged `expBX_analysis.h2h` code |

## Run order

Set `CHATSPATIAL_WORK_DIR` to the cluster work directory; the experiment lives in `$CHATSPATIAL_WORK_DIR/expBX` (`EXPBX_ROOT` overrides it), with `scripts/`, `fixtures/`, `competitors/` (STAgent and SpatialAgent checkouts with their venvs), `images/` and `runs/`. The Anthropic key is read from `ANTHROPIC_API_KEY`.

1. Environments: `expBX_build_v156.sbatch`, `expBX_build_v159.sbatch` (`pip install "chatspatial[full]==1.5.6"` and `==1.5.9` on top of the experiment-B environment with R 4.4 and rpy2).
2. Inputs: `python expBX_fixtures.py dlpfc probes` (needs the experiment-A DLPFC inputs in `$CHATSPATIAL_WORK_DIR/chatspatial_expA/data/dlpfc`).
3. Tasks: `python3 expBX_make_tasks.py probe_full probe`, `python3 expBX_make_tasks.py h2h_full h2h`, `python3 expBX_make_tasks.py cs159_h2h h2h --systems chatspatial,chatspatial_single --tag v159`.
4. Runs: `sbatch --array=0-N%8 expBX_array.sbatch $CHATSPATIAL_WORK_DIR/expBX/tasks/<list>.jsonl`; for the 1.5.9 re-run set `EXPBX_CS_ENV=$CHATSPATIAL_WORK_DIR/conda_envs/chatspatial_bx159`. Open-weight arm: `sbatch expBX_openweight.sbatch gemma4-12b` and `... qwen38-27b`. Qwen3.8 27B head-to-head: `sbatch expBX_build_v159_openweight.sbatch`; task lists with `expBX_make_tasks.py ow_qwen38_h2h h2h --systems chatspatial,stagent,spatialagent --tag ow_qwen38_h2h` and `expBX_make_tasks.py ow_qwen38_h2h_single h2h --systems chatspatial_single --tag ow_qwen38_h2h`; then `sbatch expBX_openweight_h2h.sbatch ow_qwen38_h2h` (repeated with `--dependency=afterany` until the queue is done) and `sbatch expBX_openweight_h2h_single.sbatch ow_qwen38_h2h_single`.
5. Judge and audit: `sbatch expBX_py.sbatch expBX_judge.py [--tasks tasks/ow_gemma4_probe.jsonl]`, then `expBX_audit_export.py`; overrides go to `judge/audit_overrides.jsonl`.
6. Analysis: `sbatch expBX_py.sbatch expBX_analysis.py probe [--tasks ...]` and `expBX_analysis.py h2h --tasks tasks/h2h_v159.jsonl` (the competitor runs of `h2h_full` with the ChatSpatial runs of `cs159_h2h`). Outputs go to `$EXPBX_ROOT/analysis/` (`probe_full` at the top level, other task lists in a subdirectory named after the list), which correspond to `results/probes_sonnet/`, `results/probes_gemma4/`, `results/probes_qwen38/` and `results/h2h_v159/` here. The Qwen3.8 27B head-to-head is scored on the GPU host with `EXPBX_ROOT=<mirror>$CHATSPATIAL_WORK_DIR/expBX python expBX_ow_h2h_score.py ow_qwen38_h2h_all <ground-truth directory>` (the same `h2h` analysis); its output `analysis/ow_qwen38_h2h_all/` corresponds to `results/h2h_qwen38/`.

## Changes to the competitor code

The competitors were run unchanged except for the client changes that the backend requires (rules 1.2 and section 6):

1. no temperature is sent (Claude Sonnet 5.5 rejects an explicit temperature);
2. `max_tokens` is 16,000;
3. when a request would end with an assistant message (both agents send one after every executed step), a transient user message "Continue." is appended for that API call only; the agent state is unchanged;
4. SpatialAgent: when a response holds a list of blocks (thinking and text), the text blocks are joined before its tag parser runs;
5. STAgent: thinking blocks are skipped when its message history is converted to text (otherwise the next request is refused);
6. open-weight arms only (probes and the Qwen3.8 27B head-to-head): STAgent gets two model entries in `src/config.py` for the locally served models (temperature not set, parallel tool calls off); SpatialAgent is pointed at the Anthropic-compatible endpoint of vLLM through its existing `ANTHROPIC_API_URL` setting, without code changes.

Google and OpenAI keys are set to dummy values and no search key is given, as in the round-1 runs, so the literature and retrieval tools of the agents are unavailable to every arm.
