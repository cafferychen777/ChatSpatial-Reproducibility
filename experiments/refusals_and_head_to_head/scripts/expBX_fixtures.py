#!/usr/bin/env python3
"""Build experiment B-X inputs on the cluster (run by sbatch in the chatspatial_v155 env).

DLPFC head-to-head input (one directory per section, so that SpatialAgent, which is given a
directory, sees only that section):
  fixtures/dlpfc/{sid}/{sid}.h5ad = experiment-A benchmark input samples/{sid}.h5ad (May-2026
  recipe: raw counts in layers['counts'], log-normalized X, 2,000 HVGs, PCA, neighbours) with
  obsm['spatial'] replaced by full-resolution pixel coordinates (pxl_col_in_fullres,
  pxl_row_in_fullres), matching the experiment-A pre-registration. Array coordinates stay in obs.

Probe fixtures: fixtures/probes/{probe_id}/<neutral file names>, see expBX_common.PROBES.
All fixtures are made read-only after writing.
"""
import os
import shutil
import stat
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent))
import expBX_common as C  # noqa: E402

EXPB_FIX = C.WORK / "expB" / "fixtures"
MOUSE_HNE = C.ROOT / "data" / "visium_hne_adata.h5ad"
# human DLPFC single-nucleus reference with integer UMI counts (the reference of the Aug-2026 head-to-head T5)
DLPFC_REF = C.ROOT / "data" / "dlpfc_151673_reference.h5ad"  # squidpy visium_hne_adata (mouse brain, raw counts in .raw)


def ro(path: Path):
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def write(a: ad.AnnData, path: Path):
    if path.parent.exists():
        path.parent.chmod(0o755)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o755)
    if path.exists():
        path.chmod(0o644)
        path.unlink()
    a.write_h5ad(path, compression="gzip")
    ro(path)
    X = a.X[: min(200, a.n_obs)]
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    print(f"  wrote {path} {a.shape} X max {X.max():.3f} integer={np.allclose(X, np.round(X))} "
          f"layers={list(a.layers)} raw={a.raw is not None} obs={list(a.obs.columns)} obsm={list(a.obsm)} "
          f"uns={list(a.uns)}", flush=True)


def lognorm_only(a: ad.AnnData) -> ad.AnnData:
    """Log-normalized X only: normalize_total(1e4) + log1p of raw counts; no counts anywhere."""
    b = ad.AnnData(X=sp.csr_matrix(a.X, dtype=np.float32), obs=a.obs.copy(), var=a.var.copy())
    if "spatial" in a.obsm:
        b.obsm["spatial"] = np.asarray(a.obsm["spatial"], dtype=float)
    sc.pp.normalize_total(b, target_sum=1e4)
    sc.pp.log1p(b)
    return b


def dlpfc_inputs():
    for sid in C.DLPFC_SECTIONS:
        a = ad.read_h5ad(C.EXPA_DLPFC / "samples" / f"{sid}.h5ad")
        a.obsm["spatial"] = a.obs[["pxl_col_in_fullres", "pxl_row_in_fullres"]].values.astype(float)
        leak = [c for c in a.obs.columns if any(k in c.lower() for k in ("layer", "ground", "manual", "cluster", "domain"))]
        assert not leak, leak
        write(a, C.dlpfc_input(sid))
        C.dlpfc_input(sid).parent.chmod(0o555)


def dlpfc_raw_151673() -> ad.AnnData:
    a = ad.read_h5ad(C.EXPA_DLPFC / "samples_raw" / "151673.h5ad")
    a.var_names_make_unique()
    a.obsm["spatial"] = a.obs[["pxl_col_in_fullres", "pxl_row_in_fullres"]].values.astype(float)
    keep = [c for c in ("sample_id", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres")
            if c in a.obs.columns]
    a.obs = a.obs[keep].copy()
    return a


def oscc_counts() -> ad.AnnData:
    a = ad.read_h5ad(EXPB_FIX / "oscc_s1_celltype.h5ad")
    X = a.layers["counts"]
    b = ad.AnnData(X=sp.csr_matrix(X, dtype=np.float32), obs=pd.DataFrame(index=a.obs_names),
                   var=pd.DataFrame(index=a.var_names))
    b.obs["sample_id"] = "OSCC_S1"
    b.obsm["spatial"] = np.asarray(a.obsm["spatial"], dtype=float)
    return b


def reference() -> ad.AnnData:
    r = ad.read_h5ad(DLPFC_REF)
    X0 = r.X[:500]
    X0 = X0.toarray() if sp.issparse(X0) else np.asarray(X0)
    assert np.allclose(X0, np.round(X0)), "reference must hold integer counts"
    rng = np.random.default_rng(C.SEED)
    idx = []
    for ct, grp in r.obs.groupby("cell_type", observed=True).groups.items():
        g = np.asarray(r.obs_names.get_indexer(grp))
        idx.extend(sorted(rng.choice(g, size=min(len(g), 400), replace=False)))
    idx = np.sort(np.array(idx))
    s = r[idx].copy()
    out = ad.AnnData(X=sp.csr_matrix(s.X, dtype=np.float32), obs=pd.DataFrame(index=s.obs_names),
                     var=pd.DataFrame(index=s.var_names))
    out.obs["cell_type"] = s.obs["cell_type"].astype(str).astype("category").values
    print("  reference", out.shape, "var head", list(out.var_names[:3]), "cell types:",
          out.obs["cell_type"].value_counts().to_dict())
    return out


def mouse_brain() -> ad.AnnData:
    a = ad.read_h5ad(MOUSE_HNE)
    raw = a.raw.to_adata()
    b = ad.AnnData(X=sp.csr_matrix(raw.X, dtype=np.float32), obs=pd.DataFrame(index=a.obs_names),
                   var=pd.DataFrame(index=raw.var_names))
    b.obs["cluster"] = a.obs["cluster"].astype(str).astype("category").values
    b.obsm["spatial"] = np.asarray(a.obsm["spatial"], dtype=float)
    print("  mouse clusters:", b.obs["cluster"].value_counts().to_dict())
    return b


def probes():
    ref = reference()
    dl = dlpfc_raw_151673()
    shared = set(ref.var_names) & set(dl.var_names)
    print(f"  reference/spatial shared genes: {len(shared)}")
    for pid in ("P1_rctd_no_counts", "P2_card_no_counts"):
        d = C.probe_dir(pid)
        write(lognorm_only(dl), d / "dlpfc_151673_visium.h5ad")
        write(ref, d / "brain_snrna_reference.h5ad")
    write(lognorm_only(dl), C.probe_dir("P3_spark_no_counts") / "dlpfc_151673_visium.h5ad")
    write(mouse_brain(), C.probe_dir("P4_human_lr_on_mouse") / "mouse_brain_visium.h5ad")
    write(dl, C.probe_dir("P5_method_unavailable") / "dlpfc_151673_visium.h5ad")
    # 15 in-tissue spots nearest the tissue centroid (deterministic)
    xy = dl.obsm["spatial"]
    d2 = ((xy - xy.mean(0)) ** 2).sum(1)
    tiny = dl[np.sort(np.argsort(d2)[:15])].copy()
    write(tiny, C.probe_dir("P6_more_domains_than_spots") / "dlpfc_subset_visium.h5ad")
    nc = dl.copy()
    del nc.obsm["spatial"]
    nc.obs = nc.obs[["sample_id"]].copy() if "sample_id" in nc.obs else pd.DataFrame(index=nc.obs_names)
    write(nc, C.probe_dir("P7_no_spatial_coordinates") / "dlpfc_section_visium.h5ad")
    d = C.probe_dir("P8_missing_label_column")
    write(dl, d / "dlpfc_151673_visium.h5ad")
    write(ref, d / "brain_snrna_reference.h5ad")
    for p in C.PROBES:
        pd_ = C.probe_dir(p["id"])
        for x in pd_.iterdir():
            if x.name not in p["files"].values():
                x.chmod(0o644)
                x.unlink()
        files = sorted(x.name for x in pd_.iterdir())
        assert files == sorted(p["files"].values()), (p["id"], files)
        pd_.chmod(0o555)


if __name__ == "__main__":
    which = sys.argv[1:] or ["dlpfc", "probes"]
    if "dlpfc" in which:
        dlpfc_inputs()
    if "probes" in which:
        probes()
    print("FIXTURES_DONE")
