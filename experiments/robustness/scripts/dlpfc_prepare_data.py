#!/usr/bin/env python3
"""
DLPFC Ground-Truth Benchmark — Data Preparation (all 12 sections)

Source: Maynard et al. 2021 (Nat Neurosci 24:425-436), spatialLIBD. The
SingleCellExperiment behind ``spatialLIBD::fetch_data("spe")`` (``spe`` is
``sce_to_spe(fetch_data("sce"))``) is downloaded from the same URL that
spatialLIBD uses, exported per section by R (counts as MatrixMarket), and
assembled in Python. This avoids the spatialLIBD/zellkonverter dependency
chain, so it also runs on a Linux cluster without Bioconductor.

Per section it writes
  {out}/samples_raw/{sid}.h5ad       raw integer counts, all genes, no preprocessing
                                     (input to the experiment-A noise axis)
  {out}/samples/{sid}.h5ad           benchmark input (identical recipe to the
                                     May 2026 three-section benchmark; NO ground truth)
  {out}/ground_truth/{sid}_labels.csv  expert layer annotation (layer_guess_reordered)

Spatial coordinates: obsm["spatial"] = (array_col, array_row), as in the
original three-section build; full-resolution pixel coordinates are kept in
obs["pxl_col_in_fullres"], obs["pxl_row_in_fullres"].

Usage:
  python dlpfc_prepare_data.py --out-dir $CHATSPATIAL_WORK_DIR/chatspatial_expA/data/dlpfc [--samples 151673 151507]
  (default samples: all 12)
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse as sp

SCRIPT_DIR = Path(__file__).parent
DEFAULT_OUT = SCRIPT_DIR.parent / "data" / "dlpfc_benchmark"

ALL_SAMPLES = [
    "151507", "151508", "151509", "151510",  # donor Br5292
    "151669", "151670", "151671", "151672",  # donor Br5595 (L3-L6 + WM only)
    "151673", "151674", "151675", "151676",  # donor Br8100
]

SCE_URL = (
    "https://www.dropbox.com/s/f4wcvtdq428y73p/"
    "Human_DLPFC_Visium_processedData_sce_scran_spatialLIBD.Rdata?dl=1"
)

KEEP_OBS = ("barcode_id", "sample_id", "in_tissue", "array_row", "array_col",
            "cell_count", "expr_chrM", "expr_chrM_ratio")


def build_r_script(rdata: Path, export_dir: Path, samples: list[str]) -> str:
    sample_vec = ", ".join(f'"{s}"' for s in samples)
    return f"""
suppressPackageStartupMessages(library(SingleCellExperiment))
suppressPackageStartupMessages(library(Matrix))
load("{rdata}")   # object: sce
cd <- as.data.frame(colData(sce))
sid_col <- if ("sample_name" %in% colnames(cd)) "sample_name" else "sample_id"
cat("Samples:", paste(unique(cd[[sid_col]]), collapse=", "), "\\n")
cat("colData:", paste(colnames(cd), collapse=", "), "\\n")
rd <- as.data.frame(rowData(sce))
write.csv(data.frame(gene_id=rownames(sce), rd, check.names=FALSE),
          file.path("{export_dir}", "genes.csv"), row.names=FALSE)
for (sid in c({sample_vec})) {{
  keep <- as.character(cd[[sid_col]]) == sid
  sub <- sce[, keep]
  cat(sid, ":", ncol(sub), "spots\\n")
  m <- as(counts(sub), "CsparseMatrix")
  writeMM(m, file.path("{export_dir}", paste0(sid, "_counts.mtx")))
  obs <- as.data.frame(colData(sub))
  obs <- obs[, sapply(obs, function(x) is.atomic(x)), drop=FALSE]
  obs$spot_name <- colnames(sub)
  write.csv(obs, file.path("{export_dir}", paste0(sid, "_obs.csv")), row.names=FALSE)
}}
cat("R export done\\n")
"""


def download_sce(dest: Path) -> Path:
    rdata = dest / "Human_DLPFC_Visium_processedData_sce_scran_spatialLIBD.Rdata"
    if not rdata.exists():
        print(f"Downloading {SCE_URL}")
        subprocess.run(["wget", "-q", "-O", str(rdata) + ".part", SCE_URL], check=True)
        Path(str(rdata) + ".part").rename(rdata)
    return rdata


def benchmark_preprocess(adata: ad.AnnData) -> ad.AnnData:
    """Preprocessing recipe of the May 2026 DLPFC benchmark (unchanged).

    Expects raw integer counts in X. Stores them in layers["counts"].
    Used for every experiment-A input, including the noise conditions.
    """
    adata.layers["counts"] = adata.X.copy()
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3",
                                layer="counts")
    sc.pp.pca(adata, n_comps=50, use_highly_variable=True)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
    return adata


def assemble_sample(export_dir: Path, sid: str, genes: pd.DataFrame) -> tuple:
    m = scipy.io.mmread(export_dir / f"{sid}_counts.mtx")
    X = sp.csr_matrix(m.T).astype(np.float64)  # spots x genes (float64 as the May-2026 build)
    obs = pd.read_csv(export_dir / f"{sid}_obs.csv", low_memory=False)
    # spot name: barcode (spatialLIBD colnames are barcodes, possibly sample-suffixed)
    if "barcode" in obs.columns:
        names = obs["barcode"].astype(str)
    else:
        names = obs["spot_name"].astype(str).str.replace(f"_{sid}$", "", regex=True)
    obs.index = pd.Index(names.values)
    gt = obs["layer_guess_reordered"] if "layer_guess_reordered" in obs.columns else None
    # coordinates (sce colData: row/col = array, imagerow/imagecol = fullres px)
    arow = obs["array_row"] if "array_row" in obs.columns else obs["row"]
    acol = obs["array_col"] if "array_col" in obs.columns else obs["col"]
    prow = obs["pxl_row_in_fullres"] if "pxl_row_in_fullres" in obs.columns else obs.get("imagerow")
    pcol = obs["pxl_col_in_fullres"] if "pxl_col_in_fullres" in obs.columns else obs.get("imagecol")
    clean = pd.DataFrame(index=obs.index)
    clean["sample_id"] = sid
    clean["array_row"] = arow.values.astype(int)
    clean["array_col"] = acol.values.astype(int)
    for c in KEEP_OBS:
        if c in obs.columns and c not in clean.columns:
            clean[c] = obs[c].values
    if prow is not None:
        clean["pxl_row_in_fullres"] = np.asarray(prow, dtype=float)
        clean["pxl_col_in_fullres"] = np.asarray(pcol, dtype=float)
    var = genes.copy()
    var.index = pd.Index(var["gene_id"].astype(str).values)
    adata = ad.AnnData(X=X, obs=clean, var=var)
    adata.obsm["spatial"] = clean[["array_col", "array_row"]].values.astype(float)
    gt_df = pd.DataFrame({"ground_truth": gt.values if gt is not None else np.nan},
                         index=pd.Index(obs.index, name="spot"))
    return adata, gt_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--samples", nargs="+", default=ALL_SAMPLES)
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="where to keep the downloaded sce Rdata (default: out-dir/raw)")
    ap.add_argument("--rscript", default="Rscript")
    args = ap.parse_args()

    out = args.out_dir
    for d in ("samples", "samples_raw", "ground_truth"):
        (out / d).mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir or (out / "raw")
    cache.mkdir(parents=True, exist_ok=True)

    todo = [s for s in args.samples
            if not ((out / "samples" / f"{s}.h5ad").exists()
                    and (out / "samples_raw" / f"{s}.h5ad").exists()
                    and (out / "ground_truth" / f"{s}_labels.csv").exists())]
    print(f"To build: {todo}")
    if todo:
        rdata = download_sce(cache)
        with tempfile.TemporaryDirectory(prefix="dlpfc_", dir=cache) as tmp:
            export_dir = Path(tmp)
            r_path = export_dir / "export_dlpfc.R"
            r_path.write_text(build_r_script(rdata, export_dir, todo))
            subprocess.run([args.rscript, "--no-save", "--no-restore", str(r_path)],
                           check=True)
            genes = pd.read_csv(export_dir / "genes.csv", low_memory=False)
            genes = genes.loc[:, ~genes.columns.duplicated()]
            for sid in todo:
                adata, gt_df = assemble_sample(export_dir, sid, genes)
                gt_df.to_csv(out / "ground_truth" / f"{sid}_labels.csv")
                adata.write_h5ad(out / "samples_raw" / f"{sid}.h5ad", compression="gzip")
                pre = benchmark_preprocess(adata.copy())
                pre.write_h5ad(out / "samples" / f"{sid}.h5ad")
                vc = gt_df["ground_truth"].value_counts(dropna=False).to_dict()
                print(f"  {sid}: {adata.n_obs} spots x {adata.n_vars} genes (raw), "
                      f"{pre.n_vars} after filter; labels {vc}")

    print("\nVerification")
    for sid in args.samples:
        raw = ad.read_h5ad(out / "samples_raw" / f"{sid}.h5ad", backed="r")
        gt = pd.read_csv(out / "ground_truth" / f"{sid}_labels.csv", index_col=0)
        leak = [c for c in raw.obs.columns
                if any(k in c.lower() for k in ("layer", "ground", "manual", "cluster"))]
        print(f"  {sid}: {raw.n_obs} spots, {gt['ground_truth'].notna().sum()} labeled, "
              f"{gt['ground_truth'].nunique()} layers, leak_cols={leak}")


if __name__ == "__main__":
    main()
