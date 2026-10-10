#!/usr/bin/env python
"""HGSOC case study (GSE211956, 8 Visium sections P1-P8) run end to end through ChatSpatial 1.5.7 MCP tools.

Runs on a cluster compute node only. Starts the ChatSpatial MCP server over stdio
(`python -m chatspatial server --transport stdio`), the same server an MCP client such as Claude Desktop
talks to, and issues every analysis step of the case study as a tool call:

  load_data(reference)                                  GSE211956 scRNA-seq reference (raw counts, 8 cell types)
  per section P1..P8:
    load_data(GEO section, data_type="visium")
    preprocess_data(QC, log normalization, HVGs)        case-study values, see PREPROCESS below
    deconvolve_data(method="rctd", full mode)           GSE211956 reference, cell_type_key="cell_type"
    export_data                                         -> runs/step1_rctd/{pid}.h5ad
    [post-processing, client side]                      low-tumor label (RCTD tumour proportion < 0.15) and
                                                        Ensembl 93 gene positions (hgsoc157_post.py)
    reload_data
    analyze_cnv(method="infercnvpy", reference = low-tumor spots, cluster_cells=True)
    P3 only (cross-method follow-up):
      find_markers(group_key="cnv_clusters")            Wilcoxon, one cluster versus the rest, top 20 genes
      analyze_spatial_statistics(analysis_type="moran") all genes of the section, 100 permutations
    export_data                                         -> runs/final_h5ad/{pid}.h5ad

Every tool call is appended to runs/hgsoc157_tool_calls.jsonl as it happens: tool, arguments, UTC start/end,
seconds, error flag, the tool's structured result and server log messages. Client-side post-processing steps are
recorded as "post_processing" records with their rule, inputs, outputs and summary. The first record holds the
software versions (Python packages, R and spacexr).
Usage: hgsoc157_mcp_driver.py WORKDIR
"""
import asyncio
import datetime as dt
import importlib.metadata as md
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import anndata as ad
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hgsoc157_post import label_and_annotate  # noqa: E402

W = Path(sys.argv[1])
RUN = Path(os.environ.get("HGSOC_RUN_DIR", W / "runs"))
D_STEP1 = RUN / "step1_rctd"
D_LAB = RUN / "step1_labelled"
D_FINAL = RUN / "final_h5ad"
for d in (D_STEP1, D_LAB, D_FINAL):
    d.mkdir(parents=True, exist_ok=True)
LOG_PATH = RUN / "hgsoc157_tool_calls.jsonl"
SAMPLES = os.environ.get("HGSOC_SAMPLES", " ".join(f"P{i}" for i in range(1, 9))).split()
FOLLOWUP = os.environ.get("HGSOC_FOLLOWUP", "P3").split()
REF_PATH = W / "ref" / "gse211956_reference_rctd.h5ad"  # written by hgsoc157_prep_reference.py
GENE_POS = W / "ref" / "gene_positions_ensembl93.tsv"
VISIUM = W / "input" / "visium"
GSM = {f"P{i}": f"GSM{6506109 + i}" for i in range(1, 9)}

# Case-study values recorded in the earlier pipeline (Visium_replication/run_step1_preprocessing.py and
# run_step2_deconvolution.py: spots with < 400 detected genes removed as in Denisenko et al., no mitochondrial
# filter, normalization to 10,000 counts and log transformation, 3,000 HVGs; RCTD full mode); everything else is
# the ChatSpatial 1.5.7 default.
PREPROCESS = {"filter_cells_min_genes": 400, "filter_mito_pct": None, "normalization": "log",
              "normalize_target_sum": 10000.0, "n_hvgs": 3000}
DECONV = {"method": "rctd", "cell_type_key": "cell_type", "rctd_mode": "full", "rctd_max_cores": 16}
CNV = {"method": "infercnvpy", "reference_key": "rctd_tumor_class", "reference_categories": ["low_tumor"],
       "cluster_cells": True}
MARKERS = {"group_key": "cnv_clusters", "method": "wilcoxon", "n_top_genes": 20}
MORAN = {"analysis_type": "moran", "moran_n_perms": 100, "n_jobs": 8}

SERVER = StdioServerParameters(
    command=sys.executable,
    args=["-m", "chatspatial", "server", "--transport", "stdio"],
    env={k: v for k, v in os.environ.items()},
)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def versions() -> dict:
    out = {}
    for p in ["chatspatial", "infercnvpy", "mcp", "scanpy", "squidpy", "anndata", "numpy", "scipy", "pandas",
              "scikit-learn", "statsmodels", "leidenalg", "igraph", "python-igraph", "numba", "pydantic", "rpy2"]:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    out["python"] = platform.python_version()
    try:
        r = subprocess.run(["Rscript", "-e", 'suppressMessages(library(spacexr)); cat(R.version.string, "|", '
                            'as.character(packageVersion("spacexr")))'], capture_output=True, text=True, timeout=300)
        rv, sx = (r.stdout.strip().split("|") + ["", ""])[:2]
        out["R"], out["spacexr"] = rv.strip(), sx.strip()
    except Exception as e:  # recorded, not fatal
        out["R"] = f"unavailable: {e}"
    return out


class Recorder:
    def __init__(self, path: Path):
        self.path = path
        self.seq = 0
        self.messages: list = []

    def write(self, rec: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    async def on_log(self, params) -> None:
        data = params.data
        try:
            json.dumps(data)
        except TypeError:
            data = str(data)
        self.messages.append({"level": str(params.level), "data": data})

    async def call(self, session: ClientSession, tool: str, arguments: dict, step: str, sample=None):
        self.seq += 1
        self.messages = []
        t0, s0 = now(), time.time()
        res = await session.call_tool(tool, arguments)
        t1 = now()
        structured = getattr(res, "structured_content", None)
        if structured is None:
            structured = getattr(res, "structuredContent", None)
        texts = [c.text for c in (res.content or []) if getattr(c, "type", None) == "text"]
        is_error = bool(getattr(res, "is_error", getattr(res, "isError", False)))
        self.write({
            "record": "tool_call", "seq": self.seq, "step": step, "sample": sample, "tool": tool,
            "arguments": arguments, "started_utc": t0, "finished_utc": t1, "seconds": round(time.time() - s0, 3),
            "is_error": is_error, "result": structured if structured is not None else texts,
            "server_messages": self.messages,
        })
        print(f"[{self.seq}] {sample or ''} {tool} {'ERROR' if is_error else 'ok'} {round(time.time() - s0, 1)}s",
              flush=True)
        if is_error:
            raise RuntimeError(f"{tool} failed: {texts}")
        if structured is not None:
            return structured.get("result", structured) if isinstance(structured, dict) and set(structured) == {
                "result"} else structured
        try:
            return json.loads(texts[0])
        except Exception:
            return texts

    def post(self, step: str, sample, rule: str, inputs, outputs, summary) -> None:
        self.write({"record": "post_processing", "step": step, "sample": sample, "utc": now(), "rule": rule,
                    "inputs": inputs, "outputs": outputs, "summary": summary})


def data_id_of(result) -> str:
    for key in ("id", "data_id"):
        if isinstance(result, dict) and key in result:
            return result[key]
    raise KeyError(f"no data id in {str(result)[:300]}")


async def run_section(rec: Recorder, session: ClientSession, pid: str, ref_id: str) -> None:
    loaded = await rec.call(session, "load_data", {
        "data_path": str(VISIUM / pid), "data_type": "visium", "name": pid}, "step1_load", pid)
    did = data_id_of(loaded)
    await rec.call(session, "preprocess_data", {"data_id": did, "params": PREPROCESS}, "step1_preprocess", pid)
    await rec.call(session, "deconvolve_data", {"data_id": did, "params": {**DECONV, "reference_data_id": ref_id}},
                   "step1_rctd", pid)
    p1 = D_STEP1 / f"{pid}.h5ad"
    await rec.call(session, "export_data", {"data_id": did, "path": str(p1)}, "step1_export", pid)

    # Client-side post-processing: low-tumor label and gene positions (rules in hgsoc157_post.py).
    p2 = D_LAB / f"{pid}.h5ad"
    summary = label_and_annotate(p1, p2, GENE_POS)
    rec.post("step2_label_and_gene_positions", pid, summary.pop("rule"), [str(p1), str(GENE_POS)], [str(p2)],
             summary)
    await rec.call(session, "reload_data", {"data_id": did, "path": str(p2)}, "step2_reload", pid)
    await rec.call(session, "analyze_cnv", {"data_id": did, "params": CNV}, "step2_cnv", pid)

    if pid in FOLLOWUP:
        await rec.call(session, "find_markers", {"data_id": did, "params": MARKERS}, "step4_cnv_cluster_markers", pid)
        genes = [str(g) for g in ad.read_h5ad(p2, backed="r").var_names]
        await rec.call(session, "analyze_spatial_statistics", {"data_id": did, "params": {**MORAN, "genes": genes}},
                       "step4_moran", pid)
    await rec.call(session, "export_data", {"data_id": did, "path": str(D_FINAL / f"{pid}.h5ad")},
                   "final_export", pid)


async def main() -> None:
    rec = Recorder(LOG_PATH)
    rec.write({
        "record": "session", "started_utc": now(),
        "purpose": "HGSOC case study (GSE211956, 8 Visium sections) run end to end through ChatSpatial MCP tools: "
                   "QC and preprocessing, RCTD deconvolution with the GSE211956 scRNA-seq reference, copy-number "
                   "inference with infercnvpy using low-tumor spots as reference, CNV clustering, CNV-cluster marker "
                   "genes and Moran's I for P3",
        "versions": versions(), "server_command": [SERVER.command, *SERVER.args], "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "inputs": {"visium_sections": str(VISIUM), "reference_h5ad": str(REF_PATH), "gene_positions": str(GENE_POS),
                   "visium_source": "GEO GSE211956 GSM6506110-GSM6506117 (P1-P8): matrix.mtx.gz, barcodes.tsv.gz, "
                                    "features.tsv.gz, tissue_positions_list.csv, scalefactors_json.json, images",
                   "sample_map": GSM,
                   "reference_source": "GEO GSE211956 scRNA-seq counts GSM6506105-GSM6506109 with the deposited "
                                       "cell-type labels (8 types), raw counts; labels written with / and spaces replaced by _ "
                                       "(spacexr refuses / in cell-type names), hgsoc157_prep_reference.py",
                   "gene_position_source": "Ensembl release 93 Homo_sapiens.GRCh38.93.gtf.gz gene records"},
        "fixed_parameters": {"preprocess": PREPROCESS, "deconvolution": DECONV, "cnv": CNV, "markers": MARKERS,
                             "moran": {**MORAN, "genes": "all genes of the section after QC"},
                             "followup_samples": FOLLOWUP},
    })
    async with stdio_client(SERVER) as streams:
        read, write = streams[0], streams[1]
        async with ClientSession(read, write, logging_callback=rec.on_log) as session:
            init = await session.initialize()
            tools = await session.list_tools()
            rec.write({"record": "server", "server_info": init.server_info.model_dump()
                       if hasattr(init, "server_info") else str(init), "tools": sorted(t.name for t in tools.tools)})
            ref = await rec.call(session, "load_data", {
                "data_path": str(REF_PATH), "data_type": "generic", "name": "gse211956_scrna_reference"},
                "step1_load_reference")
            ref_id = data_id_of(ref)
            for pid in SAMPLES:
                await run_section(rec, session, pid, ref_id)
    rec.write({"record": "session_end", "finished_utc": now()})
    print("DRIVER_DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
