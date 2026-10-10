#!/usr/bin/env python
"""Write the reference for deconvolve_data (cluster compute node only).

Input: GSE211956_scRNA_reference_RAW_counts.h5ad (16,249 cells x 19,915 genes, raw integer counts, 8 deposited
cell types in obs['cell_type']). spacexr refuses cell-type names containing '/', and ChatSpatial passes the labels
unchanged, so the labels are written with '/' and spaces replaced by '_' (as in the earlier pipeline:
'B/plasma cells' -> 'B_plasma_cells', 'Tumour cells' -> 'Tumour_cells'). Counts are unchanged (stored as CSR).
Usage: hgsoc157_prep_reference.py WORKDIR
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

W = Path(sys.argv[1])
a = ad.read_h5ad(W / "ref" / "GSE211956_scRNA_reference_RAW_counts.h5ad")
X = sparse.csr_matrix(a.X)
lab = a.obs["cell_type"].astype(str).str.replace("/", "_", regex=False).str.replace(" ", "_", regex=False)
b = ad.AnnData(X=X, obs=a.obs[["sample"]].assign(cell_type=lab.astype("category"),
                                                 cell_type_deposited=a.obs["cell_type"].astype(str)),
               var=a.var.copy())
assert np.array_equal(b.X.toarray().sum(axis=1), np.asarray(a.X).sum(axis=1))
b.write_h5ad(W / "ref" / "gse211956_reference_rctd.h5ad")
print(b)
print(b.obs["cell_type"].value_counts().to_dict())
print("REF_DONE")
