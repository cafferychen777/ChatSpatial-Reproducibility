#!/usr/bin/env python3
"""Build experiment-B execution fixtures (rules: ../expB_rules_2026-10-08.txt, section 3).

Reads the input datasets from $EXPB_INPUT_DIR (default $CHATSPATIAL_WORK_DIR/expB/inputs):
  oscc_s1_processed.h5ad        OSCC Sample 1 (GSE208253), QC-filtered, raw counts in layers["counts"]
  puram_hnscc_reference.h5ad    Puram et al. HNSCC scRNA-seq reference (GSE103322), obs["cell_type"]
  squidpy_merfish.h5ad          squidpy.datasets.merfish() (mouse hypothalamus)
  endocrinogenesis_day15.h5ad   scvelo.datasets.pancreas() with RNA velocity (scVelo stochastic mode)
  dlpfc_151673.h5ad             DLPFC section 151673 (spatialLIBD), written by
                                experiments/round1/scripts/dlpfc_prepare_data.py
  gene_order.txt                gene, chromosome, start, end (genomic positions used for CNV inference)
Writes $EXPB_FIXTURES (default $CHATSPATIAL_WORK_DIR/expB/fixtures)/<name>.h5ad. Fixtures represent the dataset state a user would
have reached before issuing the prompt (loaded and preprocessed with ChatSpatial defaults where
the source is not already processed).
"""
import asyncio
import os
import shutil
import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc

warnings.filterwarnings("ignore")
if not (os.environ.get("CHATSPATIAL_WORK_DIR") or (os.environ.get("EXPB_INPUT_DIR") and os.environ.get("EXPB_FIXTURES"))):
    raise SystemExit("Set CHATSPATIAL_WORK_DIR, or EXPB_INPUT_DIR and EXPB_FIXTURES.")
WORK = Path(os.environ.get("CHATSPATIAL_WORK_DIR", ".")) / "expB"
IN = Path(os.environ.get("EXPB_INPUT_DIR", WORK / "inputs"))
FIX = Path(os.environ.get("EXPB_FIXTURES", WORK / "fixtures"))
FIX.mkdir(parents=True, exist_ok=True)

OSCC_S1 = IN / "oscc_s1_processed.h5ad"
PURAM = IN / "puram_hnscc_reference.h5ad"
MERFISH = IN / "squidpy_merfish.h5ad"
PANCREAS = IN / "endocrinogenesis_day15.h5ad"
DLPFC = IN / "dlpfc_151673.h5ad"
GENE_ORDER = IN / "gene_order.txt"

MARKERS = {
    "Cancer cells": ["KRT5", "KRT14", "KRT6A", "KRT17", "SFN"],
    "Fibroblasts": ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM"],
    "T cells": ["CD3D", "CD3E", "CD2", "TRAC", "CD52"],
    "B cells": ["MS4A1", "CD79A", "CD79B", "IGHM", "CD19"],
    "Macrophages": ["CD68", "CD163", "C1QA", "C1QB", "LYZ"],
    "Endothelial cells": ["PECAM1", "VWF", "CDH5", "CLDN5"],
}


def oscc_annotated():
    a = ad.read_h5ad(OSCC_S1)
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    scores = {}
    for ct, genes in MARKERS.items():
        idx = [a.var_names.get_loc(g) for g in genes if g in a.var_names]
        scores[ct] = ((X[:, idx] - mu[idx]) / sd[idx]).mean(1)
        print(f"  {ct}: {len(idx)}/{len(genes)} markers present")
    names = list(scores)
    M = np.vstack([scores[n] for n in names]).T
    a.obs["cell_type"] = np.array(names)[M.argmax(1)]
    a.obs["cell_type"] = a.obs["cell_type"].astype("category")
    print("  cell_type counts:", a.obs["cell_type"].value_counts().to_dict())
    # genomic positions for CNV inference (gene order file of the HGSOC case study; ChatSpatial requires them)
    import pandas as pd
    go = pd.read_csv(GENE_ORDER, sep="\t", header=None,
                     names=["gene", "chromosome", "start", "end"]).drop_duplicates("gene").set_index("gene")
    hit = a.var_names.intersection(go.index)
    a.var["chromosome"] = go.reindex(a.var_names)["chromosome"].values
    a.var["start"] = go.reindex(a.var_names)["start"].values
    a.var["end"] = go.reindex(a.var_names)["end"].values
    print(f"  genomic positions for {len(hit)}/{a.n_vars} genes")
    a.write_h5ad(FIX / "oscc_s1_celltype.h5ad")


def puram_ref():
    r = ad.read_h5ad(PURAM)
    r.obs["celltype"] = r.obs["cell_type"].astype(str).astype("category")
    r.write_h5ad(FIX / "sc_ref_puram.h5ad")
    print("  ref", r.shape, "X max", float(r.X.max()))


async def chatspatial_step(adata, steps):
    from chatspatial import server
    data_id = await server.data_manager.create_dataset(adata, prefix="fixture", metadata={"type": "other"})
    for name, params in steps:
        fn = getattr(server, name)
        res = await fn(data_id, params=params)
        print(f"  {name}: {str(res)[:200]}")
    return (await server.data_manager.get_dataset(data_id))["adata"]


def merfish_slice():
    from chatspatial.models.data import PreprocessingParameters
    a = ad.read_h5ad(MERFISH)
    br = a.obs["Bregma"].value_counts().idxmax()
    s = a[a.obs["Bregma"] == br].copy()
    print("  MERFISH slice Bregma", br, s.shape, "X max", float(s.X.max()), "integer?",
          bool(np.allclose(s.X[:200].toarray() if hasattr(s.X, 'toarray') else s.X[:200], np.round(s.X[:200].toarray() if hasattr(s.X, 'toarray') else s.X[:200]))))
    s = asyncio.run(chatspatial_step(s, [("preprocess_data", PreprocessingParameters(normalization="none"))]))  # X is already volume-normalized (non-integer)
    s.write_h5ad(FIX / "merfish_slice.h5ad")


def pancreas_velocity():
    from chatspatial.models.data import PreprocessingParameters, RNAVelocityParameters
    a = ad.read_h5ad(PANCREAS)
    a.obsm["spatial"] = a.obsm["X_umap"].copy()  # no physical coordinates; documented limitation
    a = asyncio.run(chatspatial_step(a, [("preprocess_data", PreprocessingParameters()),
                                          ("analyze_velocity_data", RNAVelocityParameters())]))
    a.write_h5ad(FIX / "pancreas_velocity.h5ad")


def dlpfc():
    # The spatial-domain fixture is the DLPFC 151673 file of the DLPFC benchmark, unchanged.
    shutil.copyfile(DLPFC, FIX / "dlpfc_151673.h5ad")


if __name__ == "__main__":
    which = sys.argv[1:] or ["oscc", "ref", "merfish", "pancreas", "dlpfc"]
    if "dlpfc" in which:
        dlpfc()
    if "oscc" in which:
        oscc_annotated()
    if "ref" in which:
        puram_ref()
    if "merfish" in which:
        merfish_slice()
    if "pancreas" in which:
        pancreas_velocity()
