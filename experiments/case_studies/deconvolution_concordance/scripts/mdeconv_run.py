#!/usr/bin/env python
"""M-DECONV step 2: deconvolve one OSCC sample with ChatSpatial 1.5.5 (cluster compute node only).

Runs CARD, RCTD and FlashDeconv on the integer reference, plus the FlashDeconv anchor run on the
log-scale reference, through chatspatial.tools.deconvolution.deconvolve_spatial_data with schema
defaults. Writes one CSV per run: spot proportions + tc_le_region.
Usage: mdeconv_run.py SAMPLE_ID H5AD_DIR REF_DIR OUT_DIR
"""
import asyncio
import json
import sys
import time
import traceback
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

sid, h5ad_dir, ref_dir, out_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
out_dir.mkdir(parents=True, exist_ok=True)

RUNS = [  # (run label, method, reference file)
    ("card", "card", "ref_int.h5ad"),
    ("rctd", "rctd", "ref_int.h5ad"),
    ("flashdeconv", "flashdeconv", "ref_int.h5ad"),
    ("flashdeconv_logref", "flashdeconv", "ref_log.h5ad"),
]
only = set(sys.argv[5].split(",")) if len(sys.argv) > 5 else None


class Ctx:
    """Minimal ToolContext (same behaviour as the experiment scripts)."""

    def __init__(self, datasets):
        self._d = datasets

    async def get_adata(self, data_id):
        return self._d[data_id]

    async def set_adata(self, data_id, adata):
        self._d[data_id] = adata

    async def get_dataset_info(self, data_id):
        return {"adata": self._d[data_id], "name": data_id, "type": "visium"}

    async def add_dataset(self, adata, prefix="data", name=None, metadata=None):
        new_id = f"{prefix}_{len(self._d)}"
        self._d[new_id] = adata
        return new_id

    async def info(self, msg):
        print("  info:", msg, flush=True)

    async def warning(self, msg):
        print("  warning:", msg, flush=True)

    async def error(self, msg):
        print("  error:", msg, flush=True)

    def debug(self, msg):
        pass

    def log_config(self, title, config):
        pass


def spatial_input() -> tuple[ad.AnnData, pd.DataFrame]:
    a = ad.read_h5ad(h5ad_dir / f"{sid}_processed.h5ad")
    counts = a.layers["counts"]
    counts = sparse.csr_matrix(counts)
    sp = ad.AnnData(
        X=counts.copy(),
        obs=pd.DataFrame({"tc_le_region": a.obs["tc_le_region"].astype(str).values}, index=a.obs_names),
        var=pd.DataFrame(index=a.var_names.copy()),
        obsm={"spatial": np.asarray(a.obsm["spatial"], dtype=float)},
    )
    sp.layers["counts"] = counts.copy()
    sp.var_names_make_unique()
    # Manuscript proportions (FlashDeconv, stored under the deconvolution_card_ prefix)
    cols = [c for c in a.obs.columns if c.startswith("deconvolution_card_")]
    ms = a.obs[cols].copy()
    ms.columns = [c.replace("deconvolution_card_", "") for c in cols]
    ms["tc_le_region"] = a.obs["tc_le_region"].astype(str).values
    return sp, ms


async def run_one(sp, ref, method):
    import chatspatial
    from chatspatial.models.data import DeconvolutionParameters
    from chatspatial.tools.deconvolution import deconvolve_spatial_data

    ctx = Ctx({"oscc": sp.copy(), "puram_ref": ref})
    params = DeconvolutionParameters(method=method, reference_data_id="puram_ref", cell_type_key="cell_type")
    await deconvolve_spatial_data("oscc", ctx, params)
    res = await ctx.get_adata("oscc")
    key = f"deconvolution_{method}"
    props = res.obsm[key]
    cts = list(res.uns[f"{key}_cell_types"])
    df = pd.DataFrame(np.asarray(props), index=res.obs_names, columns=cts)
    return df, chatspatial.__version__


def main():
    import chatspatial

    print(f"{sid}: chatspatial {chatspatial.__version__}", flush=True)
    sp, ms = spatial_input()
    ms.to_csv(out_dir / f"{sid}__manuscript.csv")
    print(f"{sid}: {sp.n_obs} spots x {sp.n_vars} genes; regions {sp.obs['tc_le_region'].value_counts().to_dict()}",
          flush=True)
    refs = {}
    log = []
    for label, method, ref_file in RUNS:
        if only and label not in only:
            continue
        if ref_file not in refs:
            refs[ref_file] = ad.read_h5ad(ref_dir / ref_file)
        t0 = time.time()
        rec = {"sample": sid, "run": label, "method": method, "reference": ref_file}
        try:
            df, ver = asyncio.run(run_one(sp, refs[ref_file].copy(), method))
            df["tc_le_region"] = sp.obs["tc_le_region"].values
            df.to_csv(out_dir / f"{sid}__{label}.csv")
            rec.update(status="ok", seconds=round(time.time() - t0, 1), chatspatial=ver,
                       cell_types=[c for c in df.columns if c != "tc_le_region"],
                       row_sum_median=float(df.drop(columns="tc_le_region").sum(axis=1).median()))
        except Exception as e:  # reported, never hidden
            rec.update(status="failed", seconds=round(time.time() - t0, 1),
                       error=f"{type(e).__name__}: {e}"[:2000])
            traceback.print_exc()
        print(json.dumps(rec), flush=True)
        log.append(rec)
    with open(out_dir / f"{sid}__runlog.json", "w") as f:
        json.dump(log, f, indent=1)


if __name__ == "__main__":
    main()
