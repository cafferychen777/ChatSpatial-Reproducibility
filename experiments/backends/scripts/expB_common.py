#!/usr/bin/env python3
"""Experiment B (2026-10): shared condition definitions.

Factorial: mechanism (json = prompt-requested JSON, the original ablation;
native = provider tool-calling API) x information (full / bare / minimal)
x backend, on the original 8 ablation prompts, 10 replicates per cell.

The JSON-mechanism system prompts are the ORIGINAL strings from
ablation_invocation.py (imported, not copied). The native-mechanism tool
definitions carry the same information as the corresponding text condition,
rendered as JSON Schema:
  full    tool description + per-parameter description, type, enum, default,
          bounds, required
  bare    no tool description; per-parameter type, enum, default, bounds,
          required (no descriptions)
  minimal one-sentence tool description; no parameter schema (an empty object
          schema that accepts any properties)

Additional arms (native + full only):
  desc_swap       real tool names; the tool-level descriptions of the prompt's
                  target tool and one fixed partner tool are swapped
  neutral_intact  tool names replaced by neutral ids (tool_01..tool_08);
                  descriptions intact (control for neutral_swap)
  neutral_swap    neutral ids + the same description swap
  bias            7 open-ended prompts that name no method (method choice)

This module only needs the standard library (it runs on the GPU cluster too).
"""
from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
# The original ablation prompts and schema text: experiments/round1/scripts/ablation_invocation.py
sys.path.insert(1, str(Path(__file__).resolve().parents[2] / "round1" / "scripts"))
import ablation_invocation as _orig  # noqa: E402  (original conditions/prompts)

SCHEMA_TEXT = {
    "full": _orig.SCHEMA_FULL,
    "bare": _orig.SCHEMA_BARE,
    "minimal": _orig.SCHEMA_MINIMAL,
}
RESPONSE_INSTRUCTION = _orig.RESPONSE_INSTRUCTION
TEST_PROMPTS = _orig.TEST_PROMPTS
parse_json_text = _orig.parse_json
resolve_tool_name = _orig.resolve_tool_name
TEMPERATURE = _orig.TEMPERATURE  # 1.0
N_REPS = _orig.N_TRIALS  # 10

NATIVE_SYSTEM = (
    "You are an AI assistant that selects the appropriate analysis tool and "
    "parameters for spatial transcriptomics workflows. You have access to the "
    "tools provided.\n\n"
    "Based on the user's request, select the most appropriate tool and "
    "parameters and call it. Call exactly one tool. "
    "Include only parameters that differ from defaults or are required."
)

# ---------------------------------------------------------------------------
# Structured mirror of SCHEMA_FULL (same text, same parameters)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "identify_spatial_domains",
        "desc": "Identify spatial domains and tissue architecture.",
        "minimal": "Identify spatial domains and tissue architecture in spatial transcriptomics data.",
        "params": {
            "method": {"type": "string", "enum": ["spagcn", "leiden", "louvain", "stagate", "graphst", "banksy"],
                       "default": "spagcn",
                       "description": "'spagcn' uses histology images. 'banksy' uses spatial feature augmentation. 'stagate'/'graphst' use deep learning for spatial embedding."},
            "n_domains": {"type": "integer", "minimum": 1, "maximum": 50, "default": 7,
                          "description": "Number of spatial domains to identify."},
            "resolution": {"type": "number", "default": 0.5,
                           "description": "Leiden/Louvain clustering resolution. Higher values produce more clusters."},
        },
        "required": [],
    },
    {
        "name": "deconvolve_data",
        "desc": "Deconvolve spatial spots to estimate cell type proportions.",
        "minimal": "Estimate cell type proportions at each spatial location using a reference dataset.",
        "params": {
            "method": {"type": "string", "enum": ["flashdeconv", "cell2location", "rctd", "destvi", "stereoscope", "spotlight", "tangram", "card"],
                       "default": "flashdeconv",
                       "description": "All methods require reference_data_id and cell_type_key."},
            "reference_data_id": {"type": "string", "description": "ID of loaded single-cell reference dataset."},
            "cell_type_key": {"type": "string", "description": "Column in reference data with cell type labels."},
            "use_gpu": {"type": "boolean", "default": False,
                        "description": "GPU acceleration. Supported by cell2location, destvi, stereoscope, tangram."},
        },
        "required": ["reference_data_id", "cell_type_key"],
    },
    {
        "name": "find_spatial_genes",
        "desc": "Identify spatially variable genes.",
        "minimal": "Identify genes with statistically significant spatial expression patterns.",
        "params": {
            "method": {"type": "string", "enum": ["spatialde", "sparkx", "flashs"], "default": "sparkx",
                       "description": "'sparkx' requires R. 'flashs' is Python-native and ultra-fast. 'spatialde' uses Gaussian process kernels."},
            "n_top_genes": {"type": ["integer", "null"], "default": None,
                            "description": "Number of top genes to return (null = all significant)."},
            "filter_mt_genes": {"type": "boolean", "default": True,
                                "description": "Filter mitochondrial genes from results."},
        },
        "required": [],
    },
    {
        "name": "analyze_cell_communication",
        "desc": "Analyze cell-cell communication patterns.",
        "minimal": "Analyze cell-cell communication and ligand-receptor interactions.",
        "params": {
            "method": {"type": "string", "enum": ["liana", "cellphonedb", "cellchat_r", "fastccc"], "default": "fastccc",
                       "description": "'cellchat_r' calls R-native CellChat. 'liana' supports multiple backends. 'fastccc' is Python-native and fast."},
            "species": {"type": "string", "enum": ["human", "mouse", "zebrafish"],
                        "description": "Species for ligand-receptor database."},
            "cell_type_key": {"type": "string", "description": "Column with cell type annotations."},
        },
        "required": ["species", "cell_type_key"],
    },
    {
        "name": "analyze_spatial_statistics",
        "desc": "Analyze spatial statistics and autocorrelation.",
        "minimal": "Compute spatial statistics and autocorrelation measures.",
        "params": {
            "analysis_type": {"type": "string",
                              "enum": ["neighborhood", "co_occurrence", "ripley", "moran", "local_moran", "geary", "centrality", "getis_ord"],
                              "default": "neighborhood", "description": "Type of spatial statistic to compute."},
            "genes": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Specific genes to analyze."},
            "n_top_genes": {"type": "integer", "minimum": 1, "maximum": 500, "default": 20,
                            "description": "Number of top genes for gene-level statistics."},
        },
        "required": [],
    },
    {
        "name": "visualize_data",
        "desc": "Visualize spatial transcriptomics data.",
        "minimal": "Visualize spatial transcriptomics data and analysis results.",
        "params": {
            "plot_type": {"type": "string",
                          "enum": ["feature", "expression", "deconvolution", "communication", "trajectory", "velocity", "statistics", "enrichment", "cnv"],
                          "default": "feature", "description": "Type of visualization."},
            "feature": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}, {"type": "null"}],
                        "description": "Feature(s) to visualize."},
            "basis": {"type": "string", "enum": ["spatial", "umap", "pca"], "default": "spatial",
                      "description": "Coordinate space for plotting."},
        },
        "required": [],
    },
    {
        "name": "analyze_trajectory_data",
        "desc": "Infer cellular trajectories and pseudotime.",
        "minimal": "Infer cellular trajectories and pseudotime ordering.",
        "params": {
            "method": {"type": "string", "enum": ["cellrank", "palantir", "dpt"], "default": "cellrank",
                       "description": "'cellrank' uses velocity + expression. 'palantir' uses diffusion maps. 'dpt' is simpler diffusion pseudotime."},
            "spatial_weight": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5,
                               "description": "Weight of spatial coordinates in trajectory computation."},
        },
        "required": [],
    },
    {
        "name": "analyze_cnv",
        "desc": "Analyze copy number variations.",
        "minimal": "Analyze copy number variations from spatial transcriptomics data.",
        "params": {
            "method": {"type": "string", "enum": ["infercnvpy", "numbat"], "default": "infercnvpy",
                       "description": "'infercnvpy' is Python-native. 'numbat' requires R."},
            "reference_key": {"type": "string", "description": "Column in adata.obs for identifying reference cells."},
            "reference_categories": {"type": "array", "items": {"type": "string"},
                                     "description": "Cell types to use as normal reference."},
        },
        "required": ["reference_key", "reference_categories"],
    },
]
TOOL_NAMES = [t["name"] for t in TOOLS]

# Fixed description-swap partner for each tool (symmetric), defined 2026-10-08
SWAP_PARTNER = {
    "identify_spatial_domains": "find_spatial_genes",
    "find_spatial_genes": "identify_spatial_domains",
    "deconvolve_data": "analyze_cnv",
    "analyze_cnv": "deconvolve_data",
    "analyze_cell_communication": "analyze_trajectory_data",
    "analyze_trajectory_data": "analyze_cell_communication",
    "analyze_spatial_statistics": "visualize_data",
    "visualize_data": "analyze_spatial_statistics",
}
NEUTRAL_NAME = {name: f"tool_{i + 1:02d}" for i, name in enumerate(TOOL_NAMES)}
NEUTRAL_TO_REAL = {v: k for k, v in NEUTRAL_NAME.items()}

# Open-ended prompts that name no method (B-BIAS), defined 2026-10-08
BIAS_PROMPTS = [
    {"id": "bias_domains", "prompt": "Identify spatial domains in this Visium dataset",
     "expected_tool": "identify_spatial_domains", "method_param": "method"},
    {"id": "bias_deconv", "prompt": "Deconvolve cell types in this Visium dataset using the reference I loaded (ref_id='sc_ref', cell_type_key='celltype')",
     "expected_tool": "deconvolve_data", "method_param": "method"},
    {"id": "bias_svg", "prompt": "Find spatially variable genes in this Visium dataset",
     "expected_tool": "find_spatial_genes", "method_param": "method"},
    {"id": "bias_ccc", "prompt": "Analyze cell-cell communication in this human tumor dataset (cell_type_key='cell_type')",
     "expected_tool": "analyze_cell_communication", "method_param": "method"},
    {"id": "bias_autocorr", "prompt": "Compute spatial autocorrelation for the top marker genes",
     "expected_tool": "analyze_spatial_statistics", "method_param": "analysis_type"},
    {"id": "bias_traj", "prompt": "Infer the developmental trajectory and pseudotime of these cells",
     "expected_tool": "analyze_trajectory_data", "method_param": "method"},
    {"id": "bias_cnv", "prompt": "Infer copy number variations in this tumor using immune cells as reference (cell_type column='cell_type', reference categories 'T cells' and 'B cells')",
     "expected_tool": "analyze_cnv", "method_param": "method"},
]


def _param_schema(p: dict, with_desc: bool) -> dict:
    out = {k: v for k, v in p.items() if k != "description"}
    if with_desc and "description" in p:
        out["description"] = p["description"]
    return out


def native_tools(info: str, arm: str = "main", target_tool: str | None = None) -> list[dict]:
    """Provider-neutral tool list: [{name, description|None, input_schema}]."""
    tools = []
    descs = {t["name"]: t["desc"] for t in TOOLS}
    if arm in ("desc_swap", "neutral_swap"):
        assert target_tool is not None
        partner = SWAP_PARTNER[target_tool]
        descs[target_tool], descs[partner] = descs[partner], descs[target_tool]
    for t in TOOLS:
        name = NEUTRAL_NAME[t["name"]] if arm.startswith("neutral") else t["name"]
        if info == "full":
            schema = {"type": "object",
                      "properties": {k: _param_schema(v, True) for k, v in t["params"].items()},
                      "required": t["required"]}
            desc = descs[t["name"]]
        elif info == "bare":
            schema = {"type": "object",
                      "properties": {k: _param_schema(v, False) for k, v in t["params"].items()},
                      "required": t["required"]}
            desc = None
        elif info == "minimal":
            schema = {"type": "object", "properties": {}}
            desc = t["minimal"]
        else:
            raise ValueError(info)
        tools.append({"name": name, "description": desc, "input_schema": schema})
    return tools


def to_anthropic_tools(tools: list[dict]) -> list[dict]:
    out = []
    for t in tools:
        d = {"name": t["name"], "input_schema": t["input_schema"]}
        if t["description"]:
            d["description"] = t["description"]
        out.append(d)
    return out


def to_openai_tools(tools: list[dict]) -> list[dict]:
    out = []
    for t in tools:
        f = {"name": t["name"], "parameters": t["input_schema"]}
        if t["description"]:
            f["description"] = t["description"]
        out.append({"type": "function", "function": f})
    return out


def map_tool_name(raw: str | None, arm: str) -> str | None:
    """Canonical tool name. JSON mechanism uses the original alias table;
    native arms map exact names (and neutral ids back to real names)."""
    if raw is None:
        return None
    raw = raw.strip()
    if arm.startswith("neutral"):
        return NEUTRAL_TO_REAL.get(raw)
    if raw in TOOL_NAMES:
        return raw
    return resolve_tool_name(raw)


def build_trials(experiment: str, prompts_filter: set | None = None, reps: int = N_REPS) -> list[dict]:
    """Enumerate trials: dicts with trial_key, experiment, mechanism, info, arm, prompt fields."""
    trials = []
    if experiment == "main":
        for mech in ("json", "native"):
            for info in ("full", "bare", "minimal"):
                for p in TEST_PROMPTS:
                    for rep in range(reps):
                        trials.append(dict(experiment="main", mechanism=mech, info=info, arm="main",
                                           prompt_id=p["id"], prompt=p["prompt"], expected_tool=p["expected_tool"], rep=rep))
    elif experiment == "desc":
        for arm in ("desc_swap", "neutral_intact", "neutral_swap"):
            for p in TEST_PROMPTS:
                for rep in range(reps):
                    trials.append(dict(experiment="desc", mechanism="native", info="full", arm=arm,
                                       prompt_id=p["id"], prompt=p["prompt"], expected_tool=p["expected_tool"], rep=rep))
    elif experiment == "bias":
        for p in BIAS_PROMPTS:
            for rep in range(reps):
                trials.append(dict(experiment="bias", mechanism="native", info="full", arm="main",
                                   prompt_id=p["id"], prompt=p["prompt"], expected_tool=p["expected_tool"], rep=rep))
    else:
        raise ValueError(experiment)
    if prompts_filter:
        trials = [t for t in trials if t["prompt_id"] in prompts_filter]
    for t in trials:
        t["trial_key"] = f"{t['experiment']}|{t['arm']}|{t['mechanism']}|{t['info']}|{t['prompt_id']}|{t['rep']}"
    return trials


def request_spec(trial: dict) -> tuple[str, list[dict] | None]:
    """(system prompt, provider-neutral tools or None) for a trial."""
    if trial["mechanism"] == "json":
        return SCHEMA_TEXT[trial["info"]] + RESPONSE_INSTRUCTION, None
    return NATIVE_SYSTEM, native_tools(trial["info"], trial["arm"], trial["expected_tool"])


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()


if __name__ == "__main__":
    for e in ("main", "desc", "bias"):
        print(e, len(build_trials(e)))
    print(json.dumps(to_openai_tools(native_tools("full"))[:1], indent=1))
