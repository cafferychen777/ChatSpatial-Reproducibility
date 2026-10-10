#!/usr/bin/env python
"""M-DECONV step 1: build the Puram reference once (cluster compute node only).

Labels are built exactly as in the original case-study pipeline (first submission) (row
"non-cancer cell type", 0 -> "Cancer"). Two matrices are written:
  ref_log.h5ad : the GEO values, log2(TPM/10 + 1), as the manuscript pipeline used them
  ref_int.h5ad : round(2^x - 1), integer TPM/10, given to CARD, RCTD and FlashDeconv
See ../mdeconv_rules_2026-10-09.txt.
"""
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

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

X = expr.T.values.astype(np.float64)  # cells x genes
obs = pd.DataFrame({"cell_type": pd.Categorical(cell_types)}, index=expr.columns.astype(str))
var = pd.DataFrame(index=pd.Index(expr.index.astype(str)))

ref_log = ad.AnnData(X=sparse.csr_matrix(X), obs=obs.copy(), var=var.copy())
ref_log.var_names_make_unique()

tpm10 = np.power(2.0, X) - 1.0
per_cell = tpm10.sum(axis=1)
X_int = np.rint(np.clip(tpm10, 0, None)).astype(np.int32)
ref_int = ad.AnnData(X=sparse.csr_matrix(X_int), obs=obs.copy(), var=var.copy())
ref_int.var_names_make_unique()

print("cells", ref_log.n_obs, "genes", ref_log.n_vars)
print("cell types:\n", obs["cell_type"].value_counts().to_string())
print("log matrix: min %.3f max %.3f" % (X.min(), X.max()))
print("sum of 2^x-1 per cell: median %.0f, 5%% %.0f, 95%% %.0f (1e5 expected for TPM/10)"
      % (np.median(per_cell), np.percentile(per_cell, 5), np.percentile(per_cell, 95)))
print("integer matrix: per-cell total median %.0f, nonzero fraction %.3f"
      % (np.median(X_int.sum(axis=1)), (X_int > 0).mean()))

ref_log.write_h5ad(out / "ref_log.h5ad")
ref_int.write_h5ad(out / "ref_int.h5ad")
print("PREP_DONE")
