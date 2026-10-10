#!/usr/bin/env python3
"""Stage STARmap (visual cortex) and ABC-atlas MERFISH (one cortical section) for experiment A.

Outputs per dataset under {root}/data/{name}/:
  samples_raw/{sid}.h5ad      raw integer counts, obsm["spatial"]
  ground_truth/{sid}_labels.csv  index=cell, column ground_truth
Run on a compute node (sbatch/srun). Downloads are skipped if files exist.
"""
import argparse
import os
import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ABC = "https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com"
ABC_FILES = {
    "raw.h5ad": f"{ABC}/expression_matrices/MERFISH-C57BL6J-638850/20230830/C57BL6J-638850-raw.h5ad",
    "cells.csv": f"{ABC}/metadata/MERFISH-C57BL6J-638850-CCF/20231215/views/cell_metadata_with_parcellation_annotation.csv",
    "gene.csv": f"{ABC}/metadata/MERFISH-C57BL6J-638850/20241115/gene.csv",
}
STARMAP_URL = "https://ndownloader.figshare.com/files/40038688"


def wget(url, dest: Path):
    if not dest.exists():
        subprocess.run(["wget", "-q", "-c", "-O", str(dest) + ".part", url], check=True)
        Path(str(dest) + ".part").rename(dest)


def stage_starmap(root: Path):
    out = root / "data" / "starmap"
    for d in ("raw", "samples_raw", "ground_truth"):
        (out / d).mkdir(parents=True, exist_ok=True)
    f = out / "raw" / "STARmap_20180505_BY3_1k.h5ad"
    wget(STARMAP_URL, f)
    a = ad.read_h5ad(f)
    X = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
    assert np.allclose(X, np.round(X)), "STARmap counts are not integers"
    b = ad.AnnData(X=sp.csr_matrix(np.round(X).astype(np.float32)),
                   obs=pd.DataFrame(index=a.obs_names.astype(str)),
                   var=pd.DataFrame(index=a.var_names.astype(str)))
    b.obsm["spatial"] = np.asarray(a.obsm["spatial"], dtype=float)
    sid = "BY3_1k"
    b.write_h5ad(out / "samples_raw" / f"{sid}.h5ad")
    pd.DataFrame({"ground_truth": a.obs["label"].astype(str).values},
                 index=pd.Index(b.obs_names, name="spot")).to_csv(out / "ground_truth" / f"{sid}_labels.csv")
    print("STARmap:", b.shape, a.obs["label"].value_counts().to_dict())


def abc_layer(row) -> str | None:
    """Collapse CCF parcellation to cortical layer / white matter (our derivation)."""
    div = str(row["parcellation_division"])
    sub = str(row["parcellation_substructure"])
    if str(row["parcellation_category"]) == "fiber tracts":
        return "WM"
    if div == "Isocortex":
        for suffix, lab in (("2/3", "L2/3"), ("6a", "L6a"), ("6b", "L6b"),
                            ("1", "L1"), ("4", "L4"), ("5", "L5")):
            if sub.endswith(suffix):
                return lab
        return None
    return None


def stage_abc(root: Path, section: str, summary_only: bool):
    out = root / "data" / "abc_merfish"
    for d in ("raw", "samples_raw", "ground_truth"):
        (out / d).mkdir(parents=True, exist_ok=True)
    for name, url in ABC_FILES.items():
        wget(url, out / "raw" / name)
    m = pd.read_csv(out / "raw" / "cells.csv", dtype={"cell_label": str}).set_index("cell_label")
    summ = m.groupby("brain_section_label").agg(n=("x_ccf", "size"), ap=("x_ccf", "median"),
                                                n_iso=("parcellation_division", lambda s: (s == "Isocortex").sum()))
    print(summ.to_string())
    if summary_only:
        return
    s = m[m["brain_section_label"] == section].copy()
    s = s[s["z_ccf"] < 5.7]  # one hemisphere
    s["layer"] = s.apply(abc_layer, axis=1)
    iso = s[s["parcellation_division"] == "Isocortex"]
    # field of view: isocortex cells plus fiber-tract cells within 300 um of isocortex
    from sklearn.neighbors import NearestNeighbors
    xy_iso = iso[["x_reconstructed", "y_reconstructed"]].values
    ft = s[s["layer"] == "WM"]
    nn = NearestNeighbors(n_neighbors=1).fit(xy_iso)
    d, _ = nn.kneighbors(ft[["x_reconstructed", "y_reconstructed"]].values)
    keep = iso.index.union(ft.index[d[:, 0] <= 0.3])  # reconstructed coords are in mm
    s = s.loc[keep]
    a = ad.read_h5ad(out / "raw" / "raw.h5ad", backed="r")
    names = a.obs["cell_label"].astype(str).values if "cell_label" in a.obs else a.obs_names.astype(str).values
    mask = np.isin(names, s.index.values)
    sub = a[mask].to_memory()
    sub.obs_names = pd.Index(names[mask])
    genes = pd.read_csv(out / "raw" / "gene.csv")
    sym = dict(zip(genes["gene_identifier"], genes["gene_symbol"])) if "gene_symbol" in genes else {}
    keep_genes = ~sub.var_names.str.startswith("Blank")
    if "gene_symbol" in sub.var.columns:
        keep_genes &= ~sub.var["gene_symbol"].astype(str).str.startswith("Blank").values
    sub = sub[:, keep_genes].copy()
    X = sub.X.tocsr() if sp.issparse(sub.X) else sp.csr_matrix(sub.X)
    assert np.allclose(X.data, np.round(X.data)), "ABC counts are not integers"
    b = ad.AnnData(X=X.astype(np.float32), obs=pd.DataFrame(index=sub.obs_names),
                   var=pd.DataFrame(index=sub.var_names.astype(str)))
    if sym:
        b.var["gene_symbol"] = [sym.get(g, g) for g in b.var_names]
    s = s.loc[b.obs_names]
    b.obsm["spatial"] = s[["x_reconstructed", "y_reconstructed"]].values.astype(float) * 1000.0  # um
    sid = section.replace("C57BL6J-638850.", "sec")
    b.write_h5ad(out / "samples_raw" / f"{sid}.h5ad")
    pd.DataFrame({"ground_truth": s["layer"].values,
                  "parcellation_substructure": s["parcellation_substructure"].values},
                 index=pd.Index(b.obs_names, name="spot")).to_csv(out / "ground_truth" / f"{sid}_labels.csv")
    print("ABC:", b.shape, s["layer"].value_counts(dropna=False).to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(os.environ.get("CHATSPATIAL_WORK_DIR", ".")) / "chatspatial_expA")
    ap.add_argument("--abc-section", default="C57BL6J-638850.47")
    ap.add_argument("--abc-summary-only", action="store_true")
    ap.add_argument("--skip-starmap", action="store_true")
    args = ap.parse_args()
    if not args.skip_starmap:
        stage_starmap(args.root)
    stage_abc(args.root, args.abc_section, args.abc_summary_only)


if __name__ == "__main__":
    main()
