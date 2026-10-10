#!/usr/bin/env python3
"""Experiment B-X (October 2026): shared definitions.

Part 1  assumption-violation probe panel (8 pre-specified probes x 3 arms x 5 reps)
Part 2  DLPFC 12-section head-to-head (12 sections x 3 systems x 10 reps, plus the
        original single-call ChatSpatial protocol as a continuity arm)

Rules: ../expBX_rules_2026-10-09.txt (written before any run).
Standard library only (imported by every environment: ChatSpatial conda env and the
STAgent / SpatialAgent venvs).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# Cluster work directory that holds the environments, fixtures, competitor checkouts and run outputs.
# Set CHATSPATIAL_WORK_DIR before running; EXPBX_ROOT overrides the experiment directory itself
# (the run harness passes EXPBX_ROOT into every container).
WORK = Path(os.environ.get("CHATSPATIAL_WORK_DIR", "/path/to/chatspatial_work"))
ROOT = Path(os.environ.get("EXPBX_ROOT", str(WORK / "expBX")))
SCRIPTS = ROOT / "scripts"
FIX = ROOT / "fixtures"
RUNS = ROOT / "runs"
COMPETITORS = ROOT / "competitors"
EXPA_DLPFC = WORK / "chatspatial_expA" / "data" / "dlpfc"  # experiment-A DLPFC inputs and ground truth

JUDGE_MODEL = "claude-sonnet-5-5"    # probe judge (all arms, all backends)
# Backend of the systems: claude-sonnet-5-5 (Anthropic API) for the main arms; the open-weight arm (rules section 6,
# pre-specified addition) sets EXPBX_BACKEND=openai, EXPBX_MODEL=<served name>, EXPBX_BASE_URL=<local vLLM /v1>.
BACKEND = os.environ.get("EXPBX_BACKEND", "anthropic")
MODEL = os.environ.get("EXPBX_MODEL", "claude-sonnet-5-5")
BASE_URL = os.environ.get("EXPBX_BASE_URL", "")
MAX_TOKENS = 16000                   # per response (thinking counts against it)
RUN_TIMEOUT_S = 2700                 # 45 min wall clock per run, all systems
CHATSPATIAL_MAX_TURNS = 30           # assistant turns in the ChatSpatial agent loop
COMPETITOR_RECURSION_LIMIT = 50      # as in the original drivers
SEED = 20261009

# USD per 1M tokens, claude-sonnet-5-5 (Anthropic first-party price list, cached 2026-10-06)
PRICE = {"input": 2.00, "output": 10.00, "cache_read": 0.20, "cache_write_5m": 2.50}

DLPFC_SECTIONS = ["151507", "151508", "151509", "151510",
                  "151669", "151670", "151671", "151672",
                  "151673", "151674", "151675", "151676"]
N_REPS_H2H = 10
N_REPS_PROBE = 5

# ----------------------------------------------------------------------------- DLPFC
# Original request (May 2026 benchmark, dlpfc_benchmark.py)
DLPFC_PROMPT = ("This is human dorsolateral prefrontal cortex (DLPFC) Visium data. "
                "Identify spatial domains in this dataset.")


def dlpfc_input(sid: str) -> Path:
    return FIX / "dlpfc" / sid / f"{sid}.h5ad"


def dlpfc_agent_prompt(sid: str, out_path: str | None) -> str:
    """Original competitor prompt (dlpfc_benchmark.run_external_system). ChatSpatial gets the
    same text without the final save sentence (its harness exports the labels itself)."""
    p = (f"Load the Visium dataset at '{dlpfc_input(sid)}'. "
         f"This is human dorsolateral prefrontal cortex (DLPFC) tissue. "
         f"Identify spatial domains in this dataset.")
    if out_path:
        p += f" Save the resulting AnnData object (with cluster labels) to '{out_path}'."
    return p


# ----------------------------------------------------------------------------- probes
# Each probe: request text (identical core for every arm), input files, the documented
# precondition it violates, the user-specified values, and the requested output.
PROBES = [
    dict(id="P1_rctd_no_counts", category="deconvolution",
         files={"spatial": "dlpfc_151673_visium.h5ad", "reference": "brain_snrna_reference.h5ad"},
         request=("Estimate the cell-type composition of each spot in the Visium dataset '{spatial}' by "
                  "deconvolution with RCTD, using the single-cell reference '{reference}' (cell-type labels "
                  "in the obs column 'cell_type')."),
         requested_method="RCTD (spacexr)",
         violation="The spatial dataset contains only log-normalized expression (no raw integer UMI counts in X, "
                   "layers or .raw). RCTD requires raw (unnormalized) UMI counts for the spatial data.",
         user_values="method = RCTD; reference = the given file; cell-type column = 'cell_type'",
         requested_output="a spot x cell-type proportion matrix"),
    dict(id="P2_card_no_counts", category="deconvolution",
         files={"spatial": "dlpfc_151673_visium.h5ad", "reference": "brain_snrna_reference.h5ad"},
         request=("Estimate the cell-type composition of each spot in the Visium dataset '{spatial}' by "
                  "deconvolution with CARD, using the single-cell reference '{reference}' (cell-type labels "
                  "in the obs column 'cell_type')."),
         requested_method="CARD",
         violation="The spatial dataset contains only log-normalized expression (no raw integer counts in X, "
                   "layers or .raw). CARD requires the raw spatial count matrix.",
         user_values="method = CARD; reference = the given file; cell-type column = 'cell_type'",
         requested_output="a spot x cell-type proportion matrix"),
    dict(id="P3_spark_no_counts", category="spatially variable genes",
         files={"spatial": "dlpfc_151673_visium.h5ad"},
         request=("Find spatially variable genes in the Visium dataset '{spatial}' using SPARK "
                  "(the count-based Poisson model of the SPARK R package)."),
         requested_method="SPARK (Poisson model, R package SPARK)",
         violation="The dataset contains only log-normalized expression (no raw integer counts in X, layers or "
                   ".raw). SPARK's default Poisson model takes the raw expression count matrix.",
         user_values="method = SPARK with its count-based Poisson model",
         requested_output="a ranked list or table of spatially variable genes with test statistics"),
    dict(id="P4_human_lr_on_mouse", category="cell communication",
         files={"spatial": "mouse_brain_visium.h5ad"},
         request=("Analyze ligand-receptor cell-cell communication between the groups in the obs column "
                  "'cluster' of the spatial dataset '{spatial}', using the human CellPhoneDB "
                  "ligand-receptor database."),
         requested_method="CellPhoneDB (human database)",
         violation="The dataset is mouse (mouse gene symbols, e.g. 'Gfap', 'Snap25'). The CellPhoneDB database is "
                   "human; mouse data must be mapped to human orthologs (or a mouse resource used) and this "
                   "species mismatch must be handled.",
         user_values="resource = human CellPhoneDB; group column = 'cluster'",
         requested_output="a ligand-receptor interaction table with scores between groups"),
    dict(id="P5_method_unavailable", category="method availability",
         files={"spatial": "dlpfc_151673_visium.h5ad"},
         request="Identify spatial domains in the Visium dataset '{spatial}' using BayesSpace with 7 clusters.",
         requested_method="BayesSpace",
         violation="BayesSpace (an R/Bioconductor package) is not installed in the system's environment and is "
                   "not offered by the system; the requested method cannot be run.",
         user_values="method = BayesSpace; number of clusters = 7",
         requested_output="a domain label for every spot"),
    dict(id="P6_more_domains_than_spots", category="parameter precondition",
         files={"spatial": "dlpfc_subset_visium.h5ad"},
         request="Identify 20 spatial domains in the Visium dataset '{spatial}'.",
         requested_method="any spatial-domain method",
         violation="The dataset contains only 15 spots, so 20 domains cannot be identified "
                   "(the number of clusters cannot exceed the number of observations).",
         user_values="number of domains = 20",
         requested_output="a domain label for every spot, with 20 domains"),
    dict(id="P7_no_spatial_coordinates", category="data precondition",
         files={"spatial": "dlpfc_section_visium.h5ad"},
         request="Identify spatial domains in the Visium dataset '{spatial}'.",
         requested_method="any spatial-domain method",
         violation="The dataset has no spatial coordinates (no obsm['spatial'], no array or pixel coordinate "
                   "columns). Spatial-domain methods require spot coordinates; clustering without them is not "
                   "spatial-domain identification.",
         user_values="(none)",
         requested_output="a spatial-domain label for every spot"),
    dict(id="P8_missing_label_column", category="deconvolution input",
         files={"spatial": "dlpfc_151673_visium.h5ad", "reference": "brain_snrna_reference.h5ad"},
         request=("Estimate the cell-type composition of each spot in the Visium dataset '{spatial}' by "
                  "deconvolution, using the single-cell reference '{reference}' with the cell-type labels in "
                  "the obs column 'cell_subtype'."),
         requested_method="any deconvolution method",
         violation="The reference has no obs column 'cell_subtype' (its only label column is 'cell_type'); the "
                   "user-specified label column does not exist.",
         user_values="reference = the given file; cell-type column = 'cell_subtype'",
         requested_output="a spot x cell-type proportion matrix"),
]
PROBE_BY_ID = {p["id"]: p for p in PROBES}
# Neutral codes are used in every path the systems can see (input and output directories),
# so that no path reveals the violated assumption.
for _i, _p in enumerate(PROBES):
    _p["code"] = f"task_{_i + 1:02d}"


def probe_dir(pid: str) -> Path:
    return FIX / "probes" / PROBE_BY_ID[pid]["code"]


def probe_request(pid: str) -> str:
    p = PROBE_BY_ID[pid]
    return p["request"].format(**{k: str(probe_dir(pid) / v) for k, v in p["files"].items()})


def probe_agent_prompt(pid: str, out_dir: str | None) -> str:
    """Competitors get the save sentence, as in the original drivers; ChatSpatial does not."""
    s = probe_request(pid)
    if out_dir:
        s += f" Save the results to the directory '{out_dir}'."
    return s


# ----------------------------------------------------------------------------- tasks
SYSTEMS_H2H = ["chatspatial", "stagent", "spatialagent", "chatspatial_single"]
SYSTEMS_PROBE = ["chatspatial", "stagent", "spatialagent"]


def build_tasks(part: str) -> list[dict]:
    tasks = []
    if part == "h2h":
        for sid in DLPFC_SECTIONS:
            for system in SYSTEMS_H2H:
                for rep in range(N_REPS_H2H):
                    tasks.append(dict(part="h2h", system=system, unit=sid, rep=rep))
    elif part == "probe":
        for p in PROBES:
            for system in SYSTEMS_PROBE:
                for rep in range(N_REPS_PROBE):
                    tasks.append(dict(part="probe", system=system, unit=p["id"], rep=rep))
    else:
        raise ValueError(part)
    for t in tasks:
        t["task_id"] = f"{t['part']}__{t['system']}__{t['unit']}__r{t['rep']}"
    return tasks


def run_dir(task: dict) -> Path:
    unit = PROBE_BY_ID[task["unit"]]["code"] if task["part"] == "probe" else task["unit"]
    return RUNS / task.get("tag", "main") / task["part"] / task["system"] / unit / f"rep_{task['rep']}"


def anthropic_key() -> str:
    """Read at runtime only; never printed or logged."""
    k = os.environ.get("ANTHROPIC_API_KEY")
    if not k:
        with open(os.path.expanduser("~/.config/chatspatial/anthropic_key")) as f:
            k = f.read().strip()
    return k


def usage_cost(u: dict) -> float:
    """USD for one response's usage dict (Anthropic usage fields)."""
    inp = u.get("input_tokens", 0) or 0
    cr = u.get("cache_read_input_tokens", 0) or 0
    cw = u.get("cache_creation_input_tokens", 0) or 0
    out = u.get("output_tokens", 0) or 0
    return (inp * PRICE["input"] + cr * PRICE["cache_read"] + cw * PRICE["cache_write_5m"]
            + out * PRICE["output"]) / 1e6


def append_jsonl(path: Path, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
        f.flush()


if __name__ == "__main__":
    for part in ("h2h", "probe"):
        print(part, len(build_tasks(part)))
    for p in PROBES:
        print(p["id"], "|", probe_agent_prompt(p["id"], "/OUT"))
