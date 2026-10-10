"""Client-side post-processing of the HGSOC ChatSpatial 1.5.7 run (applied to datasets exported by the tools).

ChatSpatial has no tool that labels spots by a threshold on a deconvolved proportion and no tool that annotates
genomic positions (analyze_cnv requires adata.var chromosome/start/end and tells the user to add them), so both
steps are applied here to the section exported with export_data; the result is returned with reload_data.

Rule 1 (spot label, Denisenko et al. convention): spots whose RCTD tumour-cell proportion (deconvolve_data,
method rctd) is below 0.15 are labelled 'low_tumor' and serve as the copy-number reference; all other spots are
labelled 'tumor_enriched'.
Rule 2 (gene positions): chromosome, start and end of each gene from Ensembl release 93 (the annotation of the
Cell Ranger GRCh38-3.0.0 reference behind the GEO feature files), matched on the Ensembl gene ID in
adata.var['gene_ids']; genes on other contigs or without a match are left without a position.
"""
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

TUMOR_THRESHOLD = 0.15
RULE = ("rctd_tumor_class = 'low_tumor' if RCTD tumour-cell proportion < 0.15 else 'tumor_enriched'; "
        "var chromosome/start/end from Ensembl 93 GTF gene records (chr1-22, X, Y) matched on gene_ids")


def deconv_table(a: ad.AnnData, key: str = "deconvolution_rctd") -> pd.DataFrame:
    cts = [str(c) for c in a.uns[f"{key}_cell_types"]] if f"{key}_cell_types" in a.uns else None
    m = a.obsm[key]
    if isinstance(m, pd.DataFrame):
        return m.astype(float)
    return pd.DataFrame(np.asarray(m, dtype=float), index=a.obs_names, columns=cts)


def tumour_column(cols) -> str:
    hits = [c for c in cols if "tumo" in c.lower()]
    if len(hits) != 1:
        raise ValueError(f"cannot identify the tumour cell type among {list(cols)}")
    return hits[0]


def label_and_annotate(p_in: Path, p_out: Path, gene_pos_tsv: Path) -> dict:
    a = ad.read_h5ad(p_in)
    props = deconv_table(a)
    tcol = tumour_column(props.columns)
    tw = props[tcol].to_numpy()
    zero = props.sum(axis=1).to_numpy() <= 0
    a.obs["rctd_tumor_class"] = pd.Categorical(np.where(tw < TUMOR_THRESHOLD, "low_tumor", "tumor_enriched"),
                                               categories=["low_tumor", "tumor_enriched"])
    pos = pd.read_csv(gene_pos_tsv, sep="\t").drop_duplicates("gene_id").set_index("gene_id")
    gid = a.var["gene_ids"].astype(str) if "gene_ids" in a.var else pd.Series(a.var_names, index=a.var_names)
    m = pos.reindex(gid.values)
    a.var["chromosome"] = pd.Categorical(m["chromosome"].values)
    a.var["start"] = m["start"].values.astype(float)
    a.var["end"] = m["end"].values.astype(float)
    a.write_h5ad(p_out)
    return {"rule": RULE, "tumour_cell_type": tcol, "n_spots": int(a.n_obs),
            "n_low_tumor": int((tw < TUMOR_THRESHOLD).sum()), "n_tumor_enriched": int((tw >= TUMOR_THRESHOLD).sum()),
            "n_spots_zero_rctd_weight": int(zero.sum()), "n_genes": int(a.n_vars),
            "n_genes_with_position": int(m["chromosome"].notna().sum()),
            "mean_tumour_proportion": float(np.mean(tw))}
