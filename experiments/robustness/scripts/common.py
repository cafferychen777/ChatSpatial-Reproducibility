"""Experiment A shared code: datasets, degradation (noise axis), preprocessing,
the identify_spatial_domains call, post-processing variants and metrics.

Everything runs on cluster compute nodes. ROOT is the work directory
($EXPA_ROOT, default $CHATSPATIAL_WORK_DIR/chatspatial_expA).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("EXPA_ROOT") or Path(os.environ.get("CHATSPATIAL_WORK_DIR", ".")) / "chatspatial_expA")
# Per-configuration tables committed with this repository (one gzipped CSV per stage)
REPO_RESULTS = HERE.parent / "results"
sys.path.insert(0, str(HERE))

DLPFC_SECTIONS = ["151507", "151508", "151509", "151510", "151669", "151670",
                  "151671", "151672", "151673", "151674", "151675", "151676"]
DATASETS = {
    # primary DLPFC analysis: full-resolution pixel coordinates (the Visium
    # convention of ChatSpatial's loader / scanpy.read_visium)
    "dlpfc": {"dir": "data/dlpfc", "sections": DLPFC_SECTIONS, "coords": "pixel", "platform": "Visium"},
    # continuity with the May 2026 three-section benchmark (array coordinates)
    "dlpfc_array": {"dir": "data/dlpfc", "sections": DLPFC_SECTIONS, "coords": "array", "platform": "Visium"},
    "starmap": {"dir": "data/starmap", "sections": ["BY3_1k"], "coords": "obsm", "platform": "STARmap"},
    "abc_merfish": {"dir": "data/abc_merfish", "sections": ["sec47"], "coords": "obsm", "platform": "MERFISH"},
}
NOISE_CONDITIONS = ["umi0.5", "umi0.25", "umi0.1", "drop0.1", "drop0.3"]
STABILITY_REPS = ["sub0.8_r1", "sub0.8_r2"]  # 80% spot subsamples for the stability criterion
KAPPA_K = 6


def stable_seed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


# --------------------------------------------------------------------------- data
def load_raw(dataset: str, sid: str):
    import anndata as ad
    cfg = DATASETS[dataset]
    a = ad.read_h5ad(ROOT / cfg["dir"] / "samples_raw" / f"{sid}.h5ad")
    a.var_names_make_unique()
    if cfg["coords"] == "pixel":
        a.obsm["spatial"] = a.obs[["pxl_col_in_fullres", "pxl_row_in_fullres"]].values.astype(float)
    elif cfg["coords"] == "array":
        a.obsm["spatial"] = a.obs[["array_col", "array_row"]].values.astype(float)
    return a


def load_gt(dataset: str, sid: str) -> pd.Series:
    cfg = DATASETS[dataset]
    gt = pd.read_csv(ROOT / cfg["dir"] / "ground_truth" / f"{sid}_labels.csv", index_col=0)
    return gt["ground_truth"]


def degrade(adata, condition: str, dataset: str, sid: str):
    """Noise axis / stability subsampling, applied to RAW counts before preprocessing.

    umi{f}:   binomial thinning, each UMI kept with probability f.
    drop{p}:  remove a random fraction p of spots/cells.
    sub0.8_r{k}: keep a random 80% of spots (stability replicate k).
    Cells left with zero counts are removed (reported in n_obs).
    """
    import scipy.sparse as sp
    if condition == "clean":
        return adata
    rng = np.random.default_rng(stable_seed(dataset, sid, condition))
    a = adata.copy()
    if condition.startswith("umi"):
        f = float(condition[3:])
        X = sp.csr_matrix(a.X)
        X.data = rng.binomial(np.round(X.data).astype(np.int64), f).astype(np.float32)
        X.eliminate_zeros()
        a.X = X
    elif condition.startswith("drop") or condition.startswith("sub"):
        frac_keep = 1 - float(condition[4:]) if condition.startswith("drop") else float(condition[3:].split("_")[0])
        n_keep = int(round(a.n_obs * frac_keep))
        idx = np.sort(rng.choice(a.n_obs, size=n_keep, replace=False))
        a = a[idx].copy()
    else:
        raise ValueError(condition)
    tot = np.asarray(a.X.sum(axis=1)).ravel()
    if (tot == 0).any():
        a = a[tot > 0].copy()
    return a


def preprocess(adata):
    """May-2026 benchmark recipe (dlpfc_prepare_data.benchmark_preprocess).

    For gene panels with <= 2000 genes after filtering (STARmap, MERFISH) all genes
    are marked highly variable. The precomputed expression neighbour graph is
    removed afterwards so that cluster_n_neighbors takes effect (ChatSpatial's
    ensure_neighbors reuses an existing graph; see report).
    """
    import scanpy as sc
    from dlpfc_prepare_data import benchmark_preprocess
    a = adata.copy()
    # float64 as in the May-2026 input (zellkonverter export); float32 shifts
    # SpaGCN labels on 151673 (ARI 0.4499 vs 0.4523), see sanity_151673.py
    a.X = a.X.astype(np.float64)
    probe = a.copy()
    sc.pp.filter_genes(probe, min_cells=10)
    if probe.n_vars > 2000:
        a = benchmark_preprocess(a)
    else:
        a.layers["counts"] = a.X.copy()
        sc.pp.filter_genes(a, min_cells=10)
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        a.var["highly_variable"] = True
        n_comps = min(50, a.n_vars - 1, a.n_obs - 1)
        sc.pp.pca(a, n_comps=n_comps, use_highly_variable=True)
    if os.environ.get("EXPA_STRIP_NEIGHBORS") == "1":
        # pilot only: pre-release ChatSpatial ignored cluster_n_neighbors when a
        # graph existed; 1.5.5 rebuilds it when an explicit k differs.
        a.uns.pop("neighbors", None)
        for k in ("connectivities", "distances"):
            if k in a.obsp:
                del a.obsp[k]
    return a


def median_nn_distance(coords: np.ndarray) -> float:
    from sklearn.neighbors import NearestNeighbors
    d, _ = NearestNeighbors(n_neighbors=2).fit(coords).kneighbors(coords)
    return float(np.median(d[:, 1]))


def radius_for_mean_neighbours(coords: np.ndarray, target: float = 6.0) -> float:
    """Smallest radius whose mean neighbour count reaches `target` (STAGATE expert rule)."""
    from sklearn.neighbors import NearestNeighbors
    k = int(target * 3) + 1
    d, _ = NearestNeighbors(n_neighbors=min(k, len(coords))).fit(coords).kneighbors(coords)
    d = np.sort(d[:, 1:].ravel())
    # mean count within r = (#pairs with distance <= r) / n
    n = len(coords)
    idx = int(np.ceil(target * n)) - 1
    return float(d[min(idx, len(d) - 1)]) * 1.0001


# --------------------------------------------------------------------------- tool call
class BenchCtx:
    """Minimal ToolContext, as in the May 2026 DLPFC benchmark driver."""

    def __init__(self, datasets):
        self._datasets = datasets
        self.warnings = []

    async def get_adata(self, data_id):
        return self._datasets[data_id]

    async def get_dataset_info(self, data_id):
        return {"adata": self._datasets[data_id], "name": data_id, "type": "spatial"}

    async def set_adata(self, data_id, adata):
        self._datasets[data_id] = adata

    async def add_dataset(self, adata, prefix="data", name=None, metadata=None):
        new_id = f"{prefix}_{len(self._datasets)}"
        self._datasets[new_id] = adata
        return new_id

    async def info(self, msg):
        pass

    async def warning(self, msg):
        self.warnings.append(str(msg)[:300])

    async def error(self, msg):
        pass

    def debug(self, msg):
        pass

    def log_config(self, title, config):
        pass


def call_tool(adata_pre, method: str, cfg: dict) -> dict:
    """Run identify_spatial_domains exactly as ChatSpatial does; return unrefined + tool-refined labels."""
    from chatspatial.models.data import SpatialDomainParameters
    from chatspatial.tools.spatial_domains import identify_spatial_domains
    cfg = dict(cfg)
    # runtime cap only (held parameter); unset -> the package's own default
    if os.environ.get("EXPA_TIMEOUT"):
        cfg.setdefault("timeout", int(os.environ["EXPA_TIMEOUT"]))
    params = SpatialDomainParameters(method=method, **cfg)
    ctx = BenchCtx({"d": adata_pre})
    t0 = time.time()
    res = asyncio.run(identify_spatial_domains("d", ctx, params))
    out = ctx._datasets["d"]
    labels = out.obs[res.domain_key].astype(str).values
    refined = out.obs[res.refined_domain_key].astype(str).values if res.refined_domain_key else None
    return {"labels": labels, "refined_tool": refined, "runtime": time.time() - t0,
            "warnings": ctx.warnings, "params": params.model_dump(mode="json")}


def refine(labels: np.ndarray, coords: np.ndarray, threshold: float) -> np.ndarray:
    """ChatSpatial's own refinement (tools/spatial_domains._refine_spatial_domains)."""
    import anndata as ad
    from chatspatial.tools.spatial_domains import _refine_spatial_domains
    a = ad.AnnData(obs=pd.DataFrame({"lab": pd.Categorical(labels)},
                                    index=[str(i) for i in range(len(labels))]))
    a.obsm["spatial"] = coords
    return np.asarray(_refine_spatial_domains(a, "lab", threshold=threshold)).astype(str)


# --------------------------------------------------------------------------- metrics
def spatial_kappa(labels: np.ndarray, coords: np.ndarray, k: int = KAPPA_K) -> float:
    """Chance-corrected agreement between each spot's label and its k nearest spatial neighbours."""
    from sklearn.neighbors import NearestNeighbors
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(coords)
    lab = pd.factorize(labels)[0]
    nb = lab[idx[:, 1:]]
    A = float((nb == lab[:, None]).mean())
    p = np.bincount(lab) / len(lab)
    E = float((p ** 2).sum())
    return (A - E) / (1 - E) if E < 1 else float("nan")


def gt_scores(obs_names, labels, gt: pd.Series) -> tuple[float, float, int]:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    s = pd.Series(labels, index=obs_names)
    g = gt.reindex(s.index)
    ok = g.notna().values
    if ok.sum() < 10:
        return float("nan"), float("nan"), int(ok.sum())
    return (adjusted_rand_score(g[ok].astype(str), s[ok]),
            normalized_mutual_info_score(g[ok].astype(str), s[ok]), int(ok.sum()))


def ari(a, b) -> float:
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(a, b))


def short_id(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def jdump(d) -> str:
    return json.dumps(d, sort_keys=True, default=str)
