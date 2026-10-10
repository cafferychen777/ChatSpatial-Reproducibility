#!/usr/bin/env python
"""Build the Puram et al. HNSCC reference as h5ad, exactly as the case study did (cluster compute node only).

Same construction as the original case-study pipeline (first submission) step 5: cell-type labels from the row
"non-cancer cell type" (label 0 -> "Cancer"), the five metadata rows dropped, quotes stripped from gene names,
expression kept as deposited on GEO (log2(TPM/10 + 1)), cells x genes, var_names made unique.
The h5ad is the file a user hands to ChatSpatial load_data(data_type="generic").
Usage: oscc156_prep_ref.py GSE103322_HNSCC_all_data.txt.gz OUT.h5ad
"""
import sys

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

src, out = sys.argv[1], sys.argv[2]
raw = pd.read_csv(src, sep="\t", index_col=0, low_memory=False)
cell_types = raw.loc["non-cancer cell type"].values.astype(str)
cell_types = np.where(np.isin(cell_types, ["0", "0.0"]), "Cancer", cell_types)
meta_rows = [
    "processed by Maxima enzyme",
    "Lymph node",
    "classified  as cancer cell",
    "classified as non-cancer cells",
    "non-cancer cell type",
]
expr = raw.loc[~raw.index.isin(meta_rows)].apply(pd.to_numeric, errors="coerce")
expr = expr.dropna(how="all")
expr.index = expr.index.str.strip("'")

X = expr.T.values.astype(np.float64)
obs = pd.DataFrame({"cell_type": pd.Categorical(cell_types)}, index=expr.columns.astype(str))
var = pd.DataFrame(index=pd.Index(expr.index.astype(str)))
ref = ad.AnnData(X=sparse.csr_matrix(X), obs=obs, var=var)
ref.var_names_make_unique()
print("cells", ref.n_obs, "genes", ref.n_vars)
print(obs["cell_type"].value_counts().to_string())
print("value range %.3f..%.3f (log2(TPM/10+1) as deposited)" % (X.min(), X.max()))
ref.write_h5ad(out)
print("PREP_DONE", out)
