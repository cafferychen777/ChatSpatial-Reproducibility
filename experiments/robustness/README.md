# Experiment A: schema-derived parameter robustness

Manuscript: Results section 2.6 "Schema-derived parameter spaces make robustness analysis systematic", Methods
"Schema-derived parameter robustness and label-free selection", Figure 4 (`figures/figureA_robustness.py`).

The selection rule, the grid rules, the outcomes and the statistics were fixed before any ground-truth evaluation.
They are in `PREREGISTRATION.md`, with two dated addenda (final analysis on the released ChatSpatial 1.5.5; re-run
of the units affected by the Gaussian-mixture fix of ChatSpatial 1.5.9) and one dated correction.

## Software

- Environment: `chatspatial[full]==1.5.5` and `spagcn-modern==1.4.0`, CPU PyTorch, Python 3.12
  (`scripts/build_env155.sbatch`).
- Units that use the Gaussian-mixture substitute for mclust EEE (every STAGATE unit, and every GraphST unit that
  clusters with mclust, including the GraphST expert setting) were re-run with the ChatSpatial 1.5.9 code
  (commits 3b4e7be and 141f245), with spagcn-modern 1.4.1. The 1.5.9 code is put in front of the 1.5.5
  installation with `EXPA_PYTHONPATH_PREFIX` (see `scripts/run_array.sbatch` and `scripts/gmm_stage2.sbatch`).
  The `identify_spatial_domains` schema is identical in releases 1.5.5 to 1.5.9, so the grid is the same.

## Data

- DLPFC: all 12 Visium sections of spatialLIBD (Maynard et al. 2021), built by `scripts/dlpfc_prepare_data.py`
  from the SingleCellExperiment that `spatialLIBD::fetch_data("sce")` downloads.
- STARmap mouse visual cortex, section 20180505_BY3_1kgenes (figshare file 40038688), and section 47 of the Allen
  Brain Cell Atlas MERFISH dataset C57BL6J-638850 (public S3 bucket of the atlas), staged by
  `scripts/stage_external.py`.

## Scripts (`scripts/`)

| Script | Role |
|---|---|
| `grid.py` | Reads the typed parameter model of `identify_spatial_domains` and generates the grid of each method (rules R1 to R7). |
| `common.py` | Datasets, data degradation (count thinning, spot removal), preprocessing, the tool call, post-processing variants and metrics. |
| `make_tasks.py` | Writes one task file per Slurm array element for a stage (`dlpfc`, `noise`, `platforms`). |
| `run_task.py` | Runs one task: every grid configuration of one method on one section, scored against the annotation, plus the two 80% subsamples for the stability criterion. |
| `cache_hook/sitecustomize.py` | Memoizes STAGATE and GraphST training on identical inputs (seed included in the key). |
| `select_rule.py` | The pre-specified label-free rule; ground-truth columns are dropped before it runs. Adds the default, expert and best (oracle) strategies. |
| `evaluate.py`, `pooled.py` | All reported numbers (configurations per section, ARI ranges, strategy means with cluster-bootstrap CIs, retention under noise, Kendall's W with permutation P). |
| `export_figA.py` | Writes the small tables of Figure 4 to `results/`. |
| `incomplete_tasks.py`, `purge_affected.py`, `gmm_stage2.sbatch` | Find unfinished tasks; remove and re-run the units affected by the Gaussian-mixture fix. |
| `launch_v155.sh`, `run_array.sbatch`, `noise_stage.sbatch`, `final_eval.sh` | Submit the stages `dlpfc155`, `platforms155` and `noise155` and run the final evaluation. |
| `build_env155.sbatch`, `prep_dlpfc.sbatch`, `stage_external.sbatch` | Environment and input data. |

## Run order

Set `CHATSPATIAL_WORK_DIR` to the cluster work directory and copy `scripts/` to
`$CHATSPATIAL_WORK_DIR/chatspatial_expA/scripts/`. Submit from there.

1. `sbatch build_env155.sbatch`; `sbatch prep_dlpfc.sbatch`; `sbatch stage_external.sbatch`.
2. `bash launch_v155.sh` (DLPFC grid, then rule selection and the noise stage, and the STARmap and MERFISH grid).
   The DLPFC stage used up to 60 tasks with 4 CPUs and 24 GB and 24 tasks with 16 CPUs and 48 GB at a time;
   every tool call had a 3,600 s limit.
3. Re-run of the affected units with the 1.5.9 code (`EXPA_PYTHONPATH_PREFIX` set), then `sbatch gmm_stage2.sbatch`.
4. `bash final_eval.sh TAG` prints every reported number; `python export_figA.py` writes the Figure 4 tables.

Without a cluster run directory, `select_rule.py`, `evaluate.py`, `pooled.py` and `export_figA.py` read the
per-configuration tables in `results/` instead, so the evaluation and Figure 4 can be recomputed from this
repository (`python scripts/evaluate.py dlpfc155 noise155 platforms155`).

## Results (`results/`)

| File | Content | Manuscript |
|---|---|---|
| `configurations_dlpfc155.csv.gz` | Every configuration on the 12 DLPFC sections (98,208 rows): configuration, number of domains, coherence, stability, ARI, NMI, runtime, status | Fig. 4a,b; parameter sensitivity; 2,724 configurations per section |
| `configurations_platforms155.csv.gz` | The same for STARmap and MERFISH | Fig. 4b,d; platform rankings |
| `configurations_noise155.csv.gz` | Default, expert, rule and best configurations under count thinning and spot removal | Fig. 4c |
| `strategy_scores_all.csv` | ARI and NMI of each strategy per sample and method | Fig. 4b,d; mean ARI 0.420 (default), 0.434 (expert), 0.441 (rule), 0.565 (best) |
| `sensitivity_dlpfc.csv`, `sensitivity_platforms.csv` | Minimum, median and maximum ARI over all configurations per section and method | Fig. 4a; median ranges 0.25 to 0.53 |
| `noise_retention.csv` | ARI under degradation relative to clean data | Fig. 4c; 95%, 86%, 74%, 68% retained |
| `platform_method_means.csv` | Mean ARI per platform, strategy and method | Fig. 4d; Kendall's W |
| `evaluation_summary.txt` | Output of `final_eval.sh` for the final stages | All numbers of the Results paragraph, including the paired difference rule minus default, 16 of 72 near-optimal defaults, and Kendall's W 0.37 (P = 0.42) and 0.29 (P = 0.57) |

In the per-configuration tables, the cluster work directory in error messages is written as
`$CHATSPATIAL_WORK_DIR`.
