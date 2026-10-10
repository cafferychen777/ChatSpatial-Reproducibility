#!/usr/bin/env python
"""Inspect the HGSOC inputs and write the gene-position table (cluster compute node only).

1. Reference: GSE211956_scRNA_reference_RAW_counts.h5ad (GEO scRNA-seq counts of GSM6506105-GSM6506109 with the
   deposited cell-type labels): cells, genes, label column, cells per type, integer check.
2. Gene positions: Ensembl release 93 GTF (the annotation of the Cell Ranger GRCh38-3.0.0 reference that produced
   the GEO feature files), gene records on chromosomes 1-22, X, Y -> ref/gene_positions_ensembl93.tsv with columns
   gene_id, gene_name, chromosome (chr-prefixed), start, end.
3. Coverage of the Visium feature IDs by the table.
Usage: hgsoc157_prep_inputs.py WORKDIR
"""
import gzip
import re
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

W = Path(sys.argv[1])
ref = ad.read_h5ad(W / "ref" / "GSE211956_scRNA_reference_RAW_counts.h5ad")
print("reference", ref.shape)
print("obs columns", list(ref.obs.columns))
for c in ref.obs.columns:
    if ref.obs[c].nunique() < 40:
        print("--", c, ref.obs[c].value_counts().to_dict())
X = ref.X
vals = X.data[:100000] if sparse.issparse(X) else np.asarray(X).ravel()[:100000]
print("X type", type(X), "dtype", X.dtype, "integer-valued", bool(np.all(np.mod(vals, 1) == 0)), "max", float(vals.max()))
print("raw", ref.raw is not None, "layers", list(ref.layers.keys()), "var", list(ref.var.columns)[:10], ref.var_names[:5].tolist())
print("uns", list(ref.uns.keys()))

chroms = {str(i) for i in range(1, 23)} | {"X", "Y"}
rows = []
with gzip.open(W / "ref" / "Homo_sapiens.GRCh38.93.gtf.gz", "rt") as f:
    for line in f:
        if line.startswith("#"):
            continue
        p = line.split("\t", 9)
        if p[2] != "gene" or p[0] not in chroms:
            continue
        gid = re.search(r'gene_id "([^"]+)"', p[8]).group(1)
        gn = re.search(r'gene_name "([^"]+)"', p[8])
        rows.append((gid, gn.group(1) if gn else gid, "chr" + p[0], int(p[3]), int(p[4])))
pos = pd.DataFrame(rows, columns=["gene_id", "gene_name", "chromosome", "start", "end"])
pos.to_csv(W / "ref" / "gene_positions_ensembl93.tsv", sep="\t", index=False)
print("gene positions", len(pos))
for k in range(1, 9):
    feat = pd.read_csv(W / "input" / "visium" / f"P{k}" / "filtered_feature_bc_matrix" / "features.tsv.gz", sep="\t",
                       header=None)
    print(f"P{k} features", len(feat), "with position", int(feat[0].isin(pos.gene_id).sum()))
print("PREP_DONE")
