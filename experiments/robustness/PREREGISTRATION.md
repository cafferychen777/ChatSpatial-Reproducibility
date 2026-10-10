# Experiment A — pre-registration of the label-free selection rule

Frozen: 2026-10-08, before any ground-truth evaluation of the grid. Experiment-A ground truth had been used only for software sanity checks: the recorded May 2026 SpaGCN call on 151673 (ARI 0.4523) and the end-to-end test on 151673, which scored one default and two grid configurations per method. Nothing in Sections 2 to 4 may change after the pilot results are seen. A change after this point is reported as post hoc and is confirmed on the held-out platforms (Section 6).

Code: `scripts/` (grid.py, common.py, run_task.py, select_rule.py, make_tasks.py). On the cluster, results are written to `$CHATSPATIAL_WORK_DIR/chatspatial_expA/results/`.

## 1. Data and task

- Task: spatial-domain identification with `identify_spatial_domains`. ChatSpatial runs from branch `spagcn-127-compat`, and SpaGCN runs in `spagcn_compat="spagcn-1.2.7"` mode from the spagcn-modern fork branch `compat-spagcn-1.2.7`.
- DLPFC: all 12 Visium sections (Maynard et al. 2021, spatialLIBD). Ground truth is `layer_guess_reordered`, with 7 layers, or 5 layers for 151669–151672. Unannotated spots are excluded from scoring but kept in the input.
  - The primary input uses full-resolution pixel coordinates, the Visium convention of ChatSpatial's loader.
  - The May 2026 benchmark used array coordinates. The SpaGCN default is also run on array coordinates for continuity and is reported alongside.
- STARmap mouse visual cortex: Wang et al. 2018, section 20180505_BY3_1kgenes, 1,207 cells, 7 labels (SDMBench copy).
- MERFISH: Allen Brain Cell Atlas C57BL6J-638850 (Yao et al. 2023), section .47, one hemisphere. The field of view is the isocortex plus fiber-tract cells within 300 µm of it. Labels are CCF layers collapsed to L1, L2/3, L4, L5, L6a, L6b and WM. This collapse is our derivation from `parcellation_substructure`.
- Preprocessing is the May 2026 recipe for every input:
  - filter genes (min_cells=10), normalize to 1e4, log1p;
  - 2,000 seurat_v3 HVGs, or all genes when at most 2,000 remain;
  - PCA with 50 components.
  - The expression neighbour graph is not precomputed, so that `cluster_n_neighbors` takes effect.

## 2. Configurations

- Grids are generated from the `SpatialDomainParameters` schema by `grid.py`, using rules R1–R7 in its docstring.
- Methods: spagcn, leiden, louvain, stagate, graphst, banksy. AESTETIK is excluded because it needs a precomputed morphology embedding.
- Tool calls per section: SpaGCN 35, Leiden 75, Louvain 75, STAGATE 28, GraphST 84, BANKSY 150 (a seeded subset of the 810-point factorial). That is 447 calls in total.
- Each call is scored with 6 post-processing variants (unrefined, or refined at thresholds 0.1/0.3/0.5/0.7/0.9), giving 2,682 configurations per section.
- One method seed is used: the schema default.
- **Schema default**: every parameter at its schema default. The output is the refined labels (`refine_domains=True`, threshold 0.5).
- **Expert**: the setting recommended by each method's paper or official tutorial. K is the number of annotated domains in the section. This strategy therefore uses annotation-count knowledge, as published benchmarks do.

  | Method | Expert setting | Output labels |
  |---|---|---|
  | SpaGCN | `n_domains=K`, other parameters at tutorial values (equal to the defaults) | refined |
  | STAGATE | `n_domains=K`, `rad_cutoff` = radius giving 6 mean neighbours (≈150 px on DLPFC, as in the tutorial) | unrefined |
  | GraphST | `n_domains=K`, mclust, GraphST refinement with radius 50 | unrefined |
  | BANKSY | λ=0.2 and k=18 on Visium, λ=0.8 and k=15 otherwise; m=1; 20 PCs; resolution bisected to K domains | unrefined |
  | Leiden / Louvain | resolution bisected to K domains | unrefined |

- **Oracle**: the configuration with the highest ARI over all grid configurations of that method and section. It is used only to measure the gap and is never proposed as a method.

## 3. The label-free rule (frozen)

The rule chooses, for each section and method, one run-level configuration from the grid. Post-processing stays at the schema default (refined at 0.5), because spatial smoothing would directly inflate the coherence criterion.

1. **Stability S.** Use two random 80% spot subsamples of the raw counts (seeded). Preprocess each subsample again and run the same configuration on it. S is the mean ARI between the subsample labels and the full-data labels on the shared spots.
2. **Spatial coherence C.** C is the chance-corrected agreement (Cohen's kappa form) between each spot's label and the labels of its 6 nearest spatial neighbours: (A − E)/(1 − E). Here A is the mean neighbour agreement and E is the sum of squared cluster proportions.
3. **Eligibility.** Exclude runs that failed or that returned fewer than 3 domains.
4. **Score.** Rank S and C separately as percentiles among the eligible configurations of that section and method, with ties averaged. Select the configuration with the highest score = rank(S) + rank(C).
5. **Ties.** Break ties by higher S, then by fewer parameters changed from the default, then by `call_id`.
6. **No ground truth.** `select_rule.py` drops every `gt_*` column before selecting.

## 4. Outcomes and statistics

- **Metrics.** The primary outcome is ARI against the annotation on annotated spots. NMI is secondary.
- **Near-optimal tolerance.** δ = 0.05 ARI. A default is near-optimal if ARI_default ≥ ARI_oracle − δ.
- **Unit of analysis.** The unit is the sample (section). Methods are averaged within a sample.
- **Confidence intervals.** 95% CIs come from a cluster bootstrap over samples (10,000 resamples), with all methods of a resampled sample kept together. Paired differences are rule − default, rule − expert and oracle − rule.
- **"Matched or exceeded".** A sample counts if its rule ARI, averaged over methods, is at least its default ARI minus 0.01.
- **Gap recovered.** (mean rule − mean default) / (mean oracle − mean default).
- **Sensitivity.** For each method and sample, compute the ARI range (max − min) over all grid configurations. A5 is the largest across methods of the median range over samples.
- **A12 and A13.** SpaGCN on the 12 DLPFC sections, under schema defaults and under the rule. The mean has a bootstrap CI over sections. The array-coordinate default, scored as in May 2026 (unrefined `domain_key`), is reported alongside for continuity with 0.374.
- **Cross-platform concordance.** Rank the 6 methods within each platform by mean ARI under the rule: DLPFC (mean of 12), STARmap and MERFISH. Compute Kendall's W across the 3 platforms, with a permutation p-value (10,000 permutations of ranks within platform). W under defaults is also reported.
- **Failure cases (A11).** List the samples and methods where the rule is more than δ below the default. Describe each with the number of domains selected against the number annotated, and with S and C against the oracle configuration.

## 5. Noise axis

- Noise is applied to raw counts before preprocessing and is seeded by `crc32(dataset|section|condition)`.
  - UMI thinning keeps each UMI with probability 0.5, 0.25 or 0.1 (binomial thinning of the integer counts).
  - Spot dropout removes a random 10% or 30% of spots.
  - Cells left with zero counts are removed.
- **Pilot conditions.** For each of the 12 DLPFC sections, 6 methods and 5 conditions, the pilot runs:
  - the schema default;
  - the expert setting, re-tuned on the degraded data;
  - the rule-selected configuration chosen on clean data;
  - the clean-data oracle configuration.
- **Retention.** Retention = mean ARI under noise / mean ARI on clean data for the same strategy, across sections and methods. N-UMI is the default's retention at 10% UMIs; N-DROP is the default's retention at 30% dropout. N-RULE is the rule − default paired difference at every noise level, with a bootstrap CI over sections.
- **Extension.** Re-applying the rule to the degraded data requires the full grid with stability replicates on every noise condition. It is run only after GO.

## 6. Go / adjust / pivot gate (DLPFC pilot, 12 sections × 6 methods)

- **GO.** All three must hold:
  - the rule matched or exceeded the default in at least 7 of 12 sections;
  - the gap recovered is at least 25%;
  - the 95% CI of the mean paired difference (rule − default) has a lower bound above 0.
  - Then extend to STARmap and MERFISH (`make_tasks.py platforms`) and run the noise extension.
- **PIVOT.** The upper bound of the CI of rule − default is below 0, or the rule matched the default in at most 4 of 12 sections.
  - Drop automatic selection. Report the sensitivity atlas and the robustness of the defaults.
- **ADJUST.** Any other outcome.
  - The pre-listed alternatives, in order, are S only, C only, and S + C with post-processing included in the candidate set.
  - Any alternative chosen at this point is post hoc. Its selection is made on DLPFC and is confirmed only on STARmap and MERFISH, which act as held-out platforms.
  - If no alternative passes, reframe the result as a sensitivity atlas that marks where defaults are unreliable.

## 7. Implementation notes (fixed before the pilot)

- **Runtime cap.** `timeout` is held at 7,200 s instead of the 600 s default. It is a runtime cap, not a method parameter, and CPU training of STAGATE and GraphST exceeds 600 s.
- **Input precision.** Inputs are float64, as in the May 2026 file. With float32, SpaGCN on 151673 changes from ARI 0.4523 to 0.4499.
- **Training cache.** GraphST and STAGATE training is memoized on its exact inputs (features, graph, seed, epochs). Configurations that differ only in clustering or refinement parameters therefore share one trained embedding.
- **GraphST seed workaround.** GraphST is re-seeded with its own `fix_seed(graphst_random_seed)` before training. ChatSpatial seeds it in the parent process but trains it in an unseeded spawned worker. That is a reported package bug: two identical calls gave labels that agreed only at ARI 0.57.

## 8. What is reported regardless of outcome

All strategies and all samples are reported, including the rule's failure cases.

## Addendum, 2026-10-09: final analysis on the released ChatSpatial 1.5.5

Recorded on 2026-10-09, before any 1.5.5 run. The decision was made after the pilot reached GO (rule 0.393 vs default 0.354, 10 of 12 sections, gap recovered 29%). It responds to the 1.5.5 release, which fixes the package bugs the pilot exposed. It is not based on the per-configuration pilot results.

- **Software.** The final analysis uses the released `chatspatial[full]==1.5.5` and `spagcn-modern==1.4.0`. SpaGCN runs in the 1.2.7-compatible mode, which is the default.
- **Grid.** The final grid is regenerated by `grid.py` directly from the 1.5.5 `SpatialDomainParameters` schema. Effective defaults of Optional fields are read from the field descriptions; method applicability is checked against them and agrees with the code. Conditional parameters are pruned as before: `graphst_radius` applies only with `graphst_refinement`.
  - `stagate_rad_cutoff` is enumerated under R6+R7 as {default, 0.25, 0.5, 2, 4} × the data-derived default radius. That default is ChatSpatial's own derivation, about 6 neighbours per spot. It replaces the pilot's declared data-scale ladder {50, 1.1, 2, 3 × NN distance}.
  - All other axes are unchanged.
  - Calls per section: SpaGCN 35, Leiden 75, Louvain 75, STAGATE 35, GraphST 84, BANKSY 150. The total is 454, up from 447 in the pilot. With 6 post-processing variants per call, A4 = 2,724 configurations per section, up from 2,682.
- **Workarounds removed.** The package's own behaviour is what gets measured, so three pilot workarounds are dropped:
  - re-seeding GraphST in the worker (1.5.5 seeds it);
  - removing the stored neighbour graph (1.5.5 honours `cluster_n_neighbors`);
  - the manual STAGATE expert radius. The STAGATE expert setting is now `n_domains=K` with the package's default radius, which reproduces the tutorials' first-ring graph.
- **Kept.**
  - float64 inputs;
  - memoized GraphST/STAGATE training on identical inputs, with the seed in the key;
  - a runtime cap of 3,600 s for every backend, which is the package's own cap for STAGATE/GraphST and avoids spurious timeouts on the 13k-cell MERFISH section.
- **Unchanged.** The selection rule (Section 3), δ, the outcome metrics and statistics (Section 4), the noise design (Section 5) and the gate (Section 6) are all unchanged.
- **Result tags.** Results go to new tags `dlpfc155`, `noise155` and `platforms155`. The pilot results (`pilot`, `noise`) are kept and reported alongside.

## Addendum, 2026-10-09 (second entry): GMM bug fix and re-run of affected units

Recorded on 2026-10-09, before the re-run.

- **The bug.** ChatSpatial's sklearn substitute for mclust EEE (`utils/compute.py::gmm_clustering`) started EM with `init_params="k-means++"`. It collapsed to degenerate fits. On 151673 the effect was STAGATE ARI 0.144 instead of 0.647 and GraphST-mclust 0.223 instead of 0.567.
- **The fix.** ChatSpatial 1.5.9 (commits 3b4e7be and 141f245) starts EM from a full k-means partition.
- **Re-run scope.** Every unit that went through this function is removed (CSV backup in `backups/gmm_fix_2026-10-09/`) and re-run with 1.5.9:
  - all STAGATE units, on every platform and for every strategy;
  - every GraphST unit with `graphst_clustering_method="mclust"`, including the GraphST expert setting.
- **Unchanged inputs.**
  - Trained embeddings are unchanged, because the training code did not change. The re-run therefore reuses the cached STAGATE and GraphST training.
  - SpaGCN, Leiden, Louvain, BANKSY and GraphST with Leiden/Louvain clustering are untouched.
  - The schema did not change, so the grid (A4 = 2,724 per section) is unchanged.
- **Downstream re-runs.** The frozen rule is re-applied to the corrected DLPFC results, because the selected STAGATE/GraphST configurations may change. The noise units are then re-run for STAGATE (all strategies) and for the GraphST expert, rule and oracle strategies.
- **Unchanged.** The selection rule, δ, the outcome metrics and the gate are unchanged.
- **Reporting.** Numbers from before and after the fix are both reported.
- **SpaGCN version.** SpaGCN runs with spagcn-modern 1.4.1, which 1.5.9 requires. That version changes only spots with zero counts in the selected genes and is otherwise label-identical.

## Correction, 2026-10-10: coordinate units

The DLPFC "pixel coordinates" (obs columns `pxl_col_in_fullres` / `pxl_row_in_fullres`) are spatialLIBD's `imagecol` / `imagerow`. These are low-resolution image pixels, not full-resolution pixels: the spot spacing is about 6.2 units, against about 138 in full resolution.

- The geometry is the same isotropic layout at a different scale. Every method in the grid is either scale-free (k-nearest-neighbour graphs, SpaGCN's p-driven bandwidth, refinement) or, in 1.5.5, derives the STAGATE radius from the spot spacing. No 1.5.5 result changes.
- One earlier description was wrong. In the pilot, the pre-release default `stagate_rad_cutoff=50` gave a dense graph of about 8 spot spacings, not zero neighbours.
