"""Client-side post-processing steps of the OSCC ChatSpatial 1.5.7 run (imported by oscc157_mcp_driver.py).

assign_tc_le: Tumor Core / Leading Edge assignment from the Leiden domains returned by identify_spatial_domains.
  The rule is the case study's rule (the original case-study pipeline (first submission), step 4), applied unchanged to the
  ChatSpatial outputs: for each Leiden domain, the TC score is the mean log-normalized expression (adata.X written by
  preprocess_data, normalize_total 1e4 + log1p, all genes after QC) of KRT14, KRT5, MKI67, PCNA, TOP2A over the
  domain's spots, and the LE score the same mean for VIM, FN1, COL1A1, COL1A2, SPARC, TGFBI. The two domains with
  the highest TC score are Tumor Core; the two domains with the highest LE score, excluding Tumor Core domains,
  are Leading Edge; all other spots are "Other". ChatSpatial has no tool that labels clusters by a marker-score
  rule, so this step runs on the exported dataset, and the labelled dataset is returned to the server with
  reload_data.

build_pooled: input for CellChat. The case study ran CellChat once on the TC and LE spots of all 12 sections
  pooled, each spot labelled by its dominant deconvolved cell type (03_run_analysis.py step 11). This step collects
  those spots from the ChatSpatial exports (raw counts from layers["counts"], label from
  obs["dominant_celltype_flashdeconv"] written by deconvolve_data), keeps the genes present in all 12 sections, and
  writes one h5ad that is loaded with load_data and analysed with analyze_cell_communication.
"""
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

LE_MARKERS = ["VIM", "FN1", "COL1A1", "COL1A2", "SPARC", "TGFBI"]
TC_MARKERS = ["KRT14", "KRT5", "MKI67", "PCNA", "TOP2A"]

TC_LE_RULE = (
    "Per Leiden domain: TC score = mean log-normalized expression of KRT14, KRT5, MKI67, PCNA, TOP2A; LE score = "
    "mean of VIM, FN1, COL1A1, COL1A2, SPARC, TGFBI (markers absent from the section are skipped). Top 2 domains by "
    "TC score -> TC; top 2 by LE score, excluding TC domains -> LE; remaining spots -> Other. Same rule as the case "
    "study (03_run_analysis.py step 4)."
)
POOL_RULE = (
    "Spots labelled TC or LE (or LE only for the secondary run) from the 12 final exports; raw counts from "
    "layers['counts']; genes kept if present in all 12 sections; label = obs['dominant_celltype_flashdeconv']; "
    "barcodes prefixed with the section id. Same construction as the case study (03_run_analysis.py step 11)."
)


def assign_tc_le(src: Path, dst: Path, domain_key: str) -> dict:
    a = ad.read_h5ad(src)
    norm = a.uns.get("preprocessing", {}).get("normalization")
    if norm != "log":
        raise ValueError(f"{src}: expected log-normalized X from preprocess_data, found {norm!r}")
    X = a.X
    domains = a.obs[domain_key].astype(str)
    le_genes = [g for g in LE_MARKERS if g in a.var_names]
    tc_genes = [g for g in TC_MARKERS if g in a.var_names]
    le_idx = [a.var_names.get_loc(g) for g in le_genes]
    tc_idx = [a.var_names.get_loc(g) for g in tc_genes]
    scores = {}
    for d in domains.unique():
        mask = (domains == d).values
        sub = X[mask]
        sub = sub.toarray() if sparse.issparse(sub) else np.asarray(sub)
        scores[d] = {
            "n_spots": int(mask.sum()),
            "le_score": float(sub[:, le_idx].mean()) if le_idx else 0.0,
            "tc_score": float(sub[:, tc_idx].mean()) if tc_idx else 0.0,
        }
    sdf = pd.DataFrame(scores).T
    top_tc = sdf.nlargest(2, "tc_score").index.tolist()
    top_le = [d for d in sdf.nlargest(2, "le_score").index.tolist() if d not in top_tc]
    region = pd.Series("Other", index=a.obs_names)
    region[domains.isin(top_le).values] = "LE"
    region[domains.isin(top_tc).values] = "TC"
    a.obs["tc_le_region"] = pd.Categorical(region.values, categories=["TC", "LE", "Other"])
    sdf["assignment"] = ["TC" if d in top_tc else ("LE" if d in top_le else "Other") for d in sdf.index]
    a.uns["tc_le_assignment"] = {
        "rule": TC_LE_RULE, "domain_key": domain_key, "tc_markers_used": tc_genes, "le_markers_used": le_genes,
        "tc_domains": [str(d) for d in top_tc], "le_domains": [str(d) for d in top_le],
    }
    a.write_h5ad(dst)
    table = sdf.reset_index().rename(columns={"index": "domain"})
    return {
        "rule": TC_LE_RULE, "domain_key": domain_key, "tc_domains": top_tc, "le_domains": top_le,
        "tc_markers_used": tc_genes, "le_markers_used": le_genes,
        "n_TC": int((region == "TC").sum()), "n_LE": int((region == "LE").sum()),
        "n_Other": int((region == "Other").sum()), "domain_scores": table.to_dict(orient="records"),
    }


def build_pooled(paths: list, dst: Path, regions: tuple) -> dict:
    parts, genes_common, rows = [], None, []
    for p in paths:
        p = Path(p)
        sid = p.stem
        a = ad.read_h5ad(p)
        mask = a.obs["tc_le_region"].astype(str).isin(regions).values
        sub = a[mask]
        counts = sub.layers["counts"]
        counts = sparse.csr_matrix(counts)
        obs = pd.DataFrame({
            "cell_type": sub.obs["dominant_celltype_flashdeconv"].astype(str).values,
            "region": sub.obs["tc_le_region"].astype(str).values,
            "sample": sid,
        }, index=[f"{sid}_{bc}" for bc in sub.obs_names])
        genes = list(map(str, sub.var_names))
        genes_common = set(genes) if genes_common is None else genes_common & set(genes)
        parts.append((obs, genes, counts))
        rows.append({"sample": sid, "n_spots": int(mask.sum()),
                     **{f"n_{r}": int((obs["region"] == r).sum()) for r in regions}})
    common = sorted(genes_common)
    mats, obss = [], []
    for obs, genes, counts in parts:
        pos = pd.Index(genes).get_indexer(common)
        mats.append(counts[:, pos])
        obss.append(obs)
    X = sparse.vstack(mats).tocsr()
    obs = pd.concat(obss)
    for c in ("cell_type", "region", "sample"):
        obs[c] = pd.Categorical(obs[c])
    pooled = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=pd.Index(common)))
    pooled.write_h5ad(dst)
    return {"rule": POOL_RULE, "regions": list(regions), "n_spots": int(pooled.n_obs), "n_genes": int(pooled.n_vars),
            "cell_type_counts": obs["cell_type"].value_counts().to_dict(), "per_sample": rows,
            "counts_are_integer": bool(np.allclose(X.data, np.round(X.data)))}
