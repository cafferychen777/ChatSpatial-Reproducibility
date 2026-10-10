#!/usr/bin/env python3
"""Prompt text, schema text and parser of the schema-enforcement ablation (round 1), as used by the
single-call ChatSpatial arm of the DLPFC head-to-head (expBX_single_call.py).

Only the parts that the single-call arm imports are kept here: the full-schema system text
(SCHEMA_FULL), the response-format instruction (RESPONSE_INSTRUCTION), the tool-name alias map
(resolve_tool_name) and the JSON parser (parse_json). The text is identical to the round-1 script.
"""

import json
import re

# ---------------------------------------------------------------------------
# Schema Variants (the ablation variable)
# ---------------------------------------------------------------------------

# Condition A: Full Schema — types + enums + descriptions + method guidance
# (same as the proven schema from determinism_multimodel.py)
SCHEMA_FULL = """
You are an AI assistant that selects the appropriate analysis tool and parameters for spatial transcriptomics workflows. You have access to the following MCP tools:

## Available Tools

### 1. identify_spatial_domains
Identify spatial domains and tissue architecture.
Parameters:
- method: one of ["spagcn", "leiden", "louvain", "stagate", "graphst", "banksy"] (default: "spagcn"). 'spagcn' uses histology images. 'banksy' uses spatial feature augmentation. 'stagate'/'graphst' use deep learning for spatial embedding.
- n_domains: integer 1-50 (default: 7). Number of spatial domains to identify.
- resolution: float (default: 0.5). Leiden/Louvain clustering resolution. Higher values produce more clusters.

### 2. deconvolve_data
Deconvolve spatial spots to estimate cell type proportions.
Parameters:
- method: one of ["flashdeconv", "cell2location", "rctd", "destvi", "stereoscope", "spotlight", "tangram", "card"] (default: "flashdeconv"). All methods require reference_data_id and cell_type_key.
- reference_data_id: string (REQUIRED). ID of loaded single-cell reference dataset.
- cell_type_key: string (REQUIRED). Column in reference data with cell type labels.
- use_gpu: boolean (default: false). GPU acceleration. Supported by cell2location, destvi, stereoscope, tangram.

### 3. find_spatial_genes
Identify spatially variable genes.
Parameters:
- method: one of ["spatialde", "sparkx", "flashs"] (default: "sparkx"). 'sparkx' requires R. 'flashs' is Python-native and ultra-fast. 'spatialde' uses Gaussian process kernels.
- n_top_genes: integer or null (default: null = all significant). Number of top genes to return.
- filter_mt_genes: boolean (default: true). Filter mitochondrial genes from results.

### 4. analyze_cell_communication
Analyze cell-cell communication patterns.
Parameters:
- method: one of ["liana", "cellphonedb", "cellchat_r", "fastccc"] (default: "fastccc"). 'cellchat_r' calls R-native CellChat. 'liana' supports multiple backends. 'fastccc' is Python-native and fast.
- species: one of ["human", "mouse", "zebrafish"] (REQUIRED). Species for ligand-receptor database.
- cell_type_key: string (REQUIRED). Column with cell type annotations.

### 5. analyze_spatial_statistics
Analyze spatial statistics and autocorrelation.
Parameters:
- analysis_type: one of ["neighborhood", "co_occurrence", "ripley", "moran", "local_moran", "geary", "centrality", "getis_ord"] (default: "neighborhood"). Type of spatial statistic to compute.
- genes: list of strings or null. Specific genes to analyze.
- n_top_genes: integer 1-500 (default: 20). Number of top genes for gene-level statistics.

### 6. visualize_data
Visualize spatial transcriptomics data.
Parameters:
- plot_type: one of ["feature", "expression", "deconvolution", "communication", "trajectory", "velocity", "statistics", "enrichment", "cnv"] (default: "feature"). Type of visualization.
- feature: string or list of strings or null. Feature(s) to visualize.
- basis: one of ["spatial", "umap", "pca"] (default: "spatial"). Coordinate space for plotting.

### 7. analyze_trajectory_data
Infer cellular trajectories and pseudotime.
Parameters:
- method: one of ["cellrank", "palantir", "dpt"] (default: "cellrank"). 'cellrank' uses velocity + expression. 'palantir' uses diffusion maps. 'dpt' is simpler diffusion pseudotime.
- spatial_weight: float 0-1 (default: 0.5). Weight of spatial coordinates in trajectory computation.

### 8. analyze_cnv
Analyze copy number variations.
Parameters:
- method: one of ["infercnvpy", "numbat"] (default: "infercnvpy"). 'infercnvpy' is Python-native. 'numbat' requires R.
- reference_key: string (REQUIRED). Column in adata.obs for identifying reference cells.
- reference_categories: list of strings (REQUIRED). Cell types to use as normal reference.
"""

# Condition B: Bare Schema — types + enums + bounds, NO descriptions

RESPONSE_INSTRUCTION = """
Based on the user's request, select the most appropriate tool and parameters.
Respond with ONLY a JSON object (no markdown, no explanation):
{
  "tool_name": "<tool_name>",
  "parameters": {
    "<param1>": <value1>,
    ...
  }
}
Include only parameters that differ from defaults or are required.
"""


# Tool name aliases for Condition C (generous mapping)
TOOL_ALIASES: dict[str, list[str]] = {
    "identify_spatial_domains": [
        "identify_spatial_domains", "spatial_domains", "spatial_clustering",
        "cluster_spatial", "find_domains", "find_spatial_domains",
    ],
    "deconvolve_data": [
        "deconvolve_data", "deconvolution", "deconvolve",
        "cell_type_deconvolution", "estimate_proportions",
        "estimate_cell_type_proportions",
    ],
    "find_spatial_genes": [
        "find_spatial_genes", "spatial_genes", "svg_detection", "find_svg",
        "spatially_variable_genes", "identify_spatial_genes",
        "spatial_variable_genes", "find_spatially_variable_genes",
    ],
    "analyze_cell_communication": [
        "analyze_cell_communication", "cell_communication",
        "cellchat", "cell_chat", "ligand_receptor",
        "analyze_communication", "cell_cell_communication",
    ],
    "analyze_spatial_statistics": [
        "analyze_spatial_statistics", "spatial_statistics",
        "spatial_autocorrelation", "morans_i", "moran_i",
        "compute_spatial_statistics",
    ],
    "visualize_data": [
        "visualize_data", "visualize", "plot", "spatial_plot",
        "create_plot", "visualization",
    ],
    "analyze_trajectory_data": [
        "analyze_trajectory_data", "trajectory", "trajectory_analysis",
        "analyze_trajectory", "pseudotime",
    ],
    "analyze_cnv": [
        "analyze_cnv", "cnv", "cnv_analysis", "infer_cnv",
        "copy_number", "cnv_inference",
    ],
}

# Build reverse lookup
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for canonical, aliases in TOOL_ALIASES.items():
    for alias in aliases:
        _ALIAS_TO_CANONICAL[alias.lower()] = canonical


def resolve_tool_name(raw_name: str | None) -> str | None:
    """Map a raw tool name to canonical name via aliases. Returns None if unresolvable."""
    if raw_name is None:
        return None
    return _ALIAS_TO_CANONICAL.get(raw_name.lower().strip())



def parse_json(text: str) -> dict | None:
    """Parse JSON from LLM response, handling markdown blocks and thinking tags."""
    text = text.strip()
    # Strip thinking tags (Gemini 2.5)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Strip markdown code blocks
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None

