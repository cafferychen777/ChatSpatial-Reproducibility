# Case studies: OSCC and HGSOC, and deconvolution method concordance

Manuscript: Results section "Case studies run through recorded ChatSpatial calls recover the central structure of
two published cancer analyses", Figures 5 and 6, Supplementary Tables 1 and 7, Methods "OSCC case study", "HGSOC
case study" and "Deconvolution method concordance".

Both case studies were run with ChatSpatial 1.5.7 (`pip install "chatspatial[full]==1.5.7"`). A client started the
ChatSpatial MCP server over stdio (`python -m chatspatial server --transport stdio`) and issued every analysis step
as a tool call. Each call is recorded with its arguments, result, software versions and timestamps in the tool-call
log (`*_tool_calls.jsonl`). The few steps that ChatSpatial has no tool for (labelling regions by a marker score or a
proportion threshold, gene positions, pooling of sections) ran on datasets exported by `export_data` and were
returned with `reload_data`; they are recorded in the same log as `post_processing` records.

## OSCC (`oscc/`; Figure 5, Supplementary Table 1)

Data: GEO GSE208253, Samples 1 to 12 (GSM6339631 to GSM6339642; `filtered_feature_bc_matrix.h5`,
`tissue_positions_list.csv`, `scalefactors_json.json`, `tissue_hires_image.png`), and the HNSCC single-cell reference
of Puram et al., GEO GSE103322 (`GSE103322_HNSCC_all_data.txt.gz`).

| Script | Role |
|---|---|
| `scripts/oscc156_prep_ref.py` | Writes the Puram et al. reference as h5ad (10 cell types, values as deposited). |
| `scripts/oscc157_mcp_driver.py` | The case study as 137 MCP tool calls: load, preprocess, embeddings, Leiden domains, TC/LE labelling (client side), FlashDeconv deconvolution, TC versus LE marker genes, Moran's I, pooling (client side) and CellChat. |
| `scripts/oscc157_tc_le.py` | The two client-side steps: TC/LE assignment from marker scores, and the pooled input for CellChat. |
| `scripts/oscc157_collect.py` | Collects the tables in `results/tables/` from the exported datasets. |
| `scripts/oscc157_postrun.py` | Runs the statistics (`../compute_case_study_stats.py`), Supplementary Table 1 (`scripts/build_supplementary_tables.py`), Supplementary Table 7 with the TC/LE labels of this run, and the Figure 5 source data. |
| `scripts/oscc157_build_env.sbatch`, `scripts/oscc157_run.sbatch` | Environment (ChatSpatial 1.5.7 with R 4.4, rpy2 and CellChat 2.2.0) and the run (8 CPUs, 64 GB). |

| Result | Content | Manuscript |
|---|---|---|
| `results/oscc157_tool_calls.jsonl` | The 137 recorded tool calls and the post-processing records | Methods |
| `results/tables/qc_statistics.csv`, `tc_le_domain_scores.csv` | QC per section; TC/LE scores and assignment per Leiden domain | 4 to 9 domains per sample; Sample 6 has no LE spots |
| `results/tables/spots/`, `proportions/` | Per spot: domain, TC/LE label, FN1; FlashDeconv proportions | Fig. 5a-e,g |
| `results/tables/deg_tc_vs_le_top500.csv`, `consensus_*_upregulated_genes.csv` | `find_markers` results and genes higher in TC or LE in at least 9 samples | Results text (AQP3, KRT6A, ..., VIM, COL1A2, ...) |
| `results/tables/cellchat_*.csv` | CellChat interactions (pooled TC and LE spots; LE spots only) | Fig. 5f; 105 Fibroblast to Cancer interactions in 17 pathways |
| `results/stats/` | Outputs of `../compute_case_study_stats.py` | Fig. 5d,e,h; Supplementary Table 1; Moran's I ranks and means |

The Moran's I table of all 12 sections (`moranI_results_all_samples.csv`, 40 MB) is committed once, as
`source_data/Fig5h_morans.csv`; `compute_case_study_stats.py` reads it from there.

## HGSOC (`hgsoc/`; Figure 6)

Data: GEO GSE211956, Visium sections P1 to P8 (GSM6506110 to GSM6506117), and the single-cell reference of the same
series (raw counts of GSM6506105 to GSM6506109 with the deposited 8 cell types); Ensembl release 93 GTF for gene
positions.

| Script | Role |
|---|---|
| `scripts/hgsoc157_prep_inputs.py`, `hgsoc157_prep_reference.py` | Gene-position table from Ensembl 93; reference with cell type names that spacexr accepts. |
| `scripts/hgsoc157_mcp_driver.py` | The case study as 59 MCP tool calls: load, preprocess, RCTD deconvolution, low-tumor labelling and gene positions (client side), infercnvpy with CNV clusters, and for P3 the CNV-cluster marker genes and Moran's I. |
| `scripts/hgsoc157_post.py` | The client-side labelling and gene-position step. |
| `scripts/hgsoc157_collect.py` | Collects the tables in `results/tables/`. |
| `scripts/hgsoc157_stats.py` | Statistics of Figure 6c,d,f and the text, written to `results/stats/`. |
| `scripts/*.sbatch` | Input preparation and the run (16 CPUs, 64 GB). |

| Result | Content | Manuscript |
|---|---|---|
| `results/hgsoc157_tool_calls.jsonl` | The 59 recorded tool calls and the post-processing records | Methods |
| `results/tables/spots/` | Per spot: coordinates, RCTD proportions, tumor label, CNV score and cluster | Fig. 6a-d |
| `results/tables/cnv_cluster_markers_P3.csv`, `moranI_P3.csv` | CNV-cluster marker genes and Moran's I of P3 | Fig. 6e,f |
| `results/stats/hgsoc_patient_summary.csv`, `cnv_score_per_patient_stats.csv` | Tumor burden, CNV clusters, mean CNV score per patient | Fig. 6c,d; 5 to 10 CNV clusters; 17% to 97% tumor-enriched spots |
| `results/stats/cliffs_delta_cnv_vs_background.csv`, `celltype_marker_moran_ranks.csv` | Moran's I of the 95 marker genes versus the other 19,085 genes (per-gene values in `source_data/Fig6f_cnv_vs_background.csv`, written by `figures/figure3_hgsoc.py`); ranks of KRT7, CD14, CDH5 | Fig. 6e,f; Cliff's delta 0.99, P = 5e-63; 91 of 95 in the top 2% |

## Deconvolution method concordance (`deconvolution_concordance/`; Supplementary Table 7)

CARD 1.1, RCTD (spacexr 2.2.1, full mode) and FlashDeconv 0.2.0 were each called through ChatSpatial 1.5.5 with
default parameters on the spots of Figure 5, with the Puram et al. reference converted to TPM/10 and rounded to
integers. The rules were fixed before the runs (`mdeconv_rules_2026-10-09.txt`).

| Script | Role |
|---|---|
| `scripts/mdeconv_prep.py` | Integer and log-scale references. |
| `scripts/mdeconv_run.py` | Deconvolves one sample with the three methods (and the FlashDeconv run on the log-scale reference). |
| `scripts/mdeconv_jobs.sh`, `mdeconv_card_rerun.sh`, `mdeconv_music.sbatch` | Slurm submission (4 CPUs, 40 GB per sample); CARD needs the R package MuSiC. |
| `scripts/mdeconv_analyze.py` | Per-sample and patient-level tests and the decision rule. |

`results/runs/` holds the per-spot proportions of every method and sample with the TC/LE labels of the ChatSpatial
1.5.7 run (relabelled by `oscc157_postrun.py`, step C; `results/spot_check_chatspatial157.csv` confirms that the spot
sets agree). `results/analysis/` is the output of `mdeconv_analyze.py` on these tables, from which
Supplementary Table 7 is formatted.

## Recomputing the statistics from this repository

```
python experiments/case_studies/compute_case_study_stats.py          # Fig. 5 and Supplementary Table 1 statistics
python experiments/case_studies/hgsoc/scripts/hgsoc157_stats.py \
    experiments/case_studies/hgsoc/results/tables experiments/case_studies/hgsoc/results/stats P3
python experiments/case_studies/deconvolution_concordance/scripts/mdeconv_analyze.py \
    experiments/case_studies/deconvolution_concordance/results/runs \
    experiments/case_studies/deconvolution_concordance/results/analysis
```

Re-running the case studies themselves needs the raw GEO files and a cluster node. Set `CHATSPATIAL_WORK_DIR`, place
the inputs under `$CHATSPATIAL_WORK_DIR/oscc157/input/visium/s1 ... s12` and
`$CHATSPATIAL_WORK_DIR/hgsoc157/input/visium/P1 ... P8` (references under `ref/`), copy the scripts to
`$CHATSPATIAL_WORK_DIR/{oscc157,hgsoc157}/scripts/`, and submit the `*_run.sbatch` files. In the tool-call logs, the
cluster work directory is written as `$CHATSPATIAL_WORK_DIR`.
