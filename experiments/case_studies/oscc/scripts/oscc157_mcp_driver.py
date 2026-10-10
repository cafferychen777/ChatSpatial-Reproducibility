#!/usr/bin/env python
"""OSCC case study (GSE208253, 12 Visium sections) run end to end through ChatSpatial 1.5.7 MCP tools.

Runs on a cluster compute node only. Starts the ChatSpatial MCP server over stdio
(`python -m chatspatial server --transport stdio`), the same server an MCP client such as Claude Desktop
talks to, and issues every analysis step of the case study as a tool call:

  load_data(reference)                                 Puram et al. HNSCC reference (h5ad, oscc156_prep_ref.py)
  per section s1..s12:
    load_data(raw GEO section, data_type="visium")
    preprocess_data(QC, log normalization, HVGs)       case-study thresholds, see PREPROCESS below
    compute_embeddings(PCA, neighbor graph)            50 PCs, 15 neighbors (case-study values)
    identify_spatial_domains(method="leiden")          Leiden on the expression + spatial composite graph, res 0.8
    export_data                                        -> runs/step1_domains/{sid}.h5ad
    [post-processing, client side]                     TC/LE assignment from the Leiden domains (oscc157_tc_le.py)
    reload_data                                        section with obs["tc_le_region"]
    deconvolve_data(method="flashdeconv")              Puram reference, cell_type_key="cell_type"
    find_markers(tc_le_region, LE vs TC) and (TC vs LE)  Wilcoxon, top 500 genes per direction
    analyze_spatial_statistics(analysis_type="moran")  all genes of the section, 6 spatial neighbors, 100 perms
    export_data                                        -> runs/final_h5ad/{sid}.h5ad
  [post-processing, client side]                       pool the TC and LE spots of the 12 sections (raw counts,
                                                       dominant FlashDeconv cell type as label)
  load_data(pooled) + analyze_cell_communication(method="cellchat_r", species="human")
  load_data(pooled LE spots) + analyze_cell_communication(...)   secondary: Leading Edge spots only

Every tool call is appended to runs/oscc157_tool_calls.jsonl as it happens: tool, arguments, UTC start/end,
seconds, error flag, the tool's structured result and server log messages. Client-side post-processing steps are
recorded as "post_processing" records with their rule, inputs, outputs and summary. The first record holds the
software versions (Python packages, R and CellChat).
Usage: oscc157_mcp_driver.py WORKDIR
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
from oscc157_tc_le import assign_tc_le, build_pooled  # noqa: E402

W = Path(sys.argv[1])
RUN = W / os.environ.get("OSCC_RUN_DIR", "runs")
D_STEP1 = RUN / "step1_domains"
D_TCLE = RUN / "step1_tc_le"
D_FINAL = RUN / "final_h5ad"
D_POOL = RUN / "cellchat"
for d in (D_STEP1, D_TCLE, D_FINAL, D_POOL):
    d.mkdir(parents=True, exist_ok=True)
LOG_PATH = RUN / "oscc157_tool_calls.jsonl"
SAMPLES = os.environ.get("OSCC_SAMPLES", " ".join(f"s{i}" for i in range(1, 13))).split()
REF_PATH = W / "ref" / "puram_hnscc_log.h5ad"
VISIUM = W / "input" / "visium"

# Case-study values (the original case-study pipeline (first submission)); everything else is the ChatSpatial 1.5.7 default.
PREPROCESS = {"filter_cells_min_genes": 200, "filter_genes_min_cells": 10, "filter_mito_pct": 25.0,
              "normalization": "log", "normalize_target_sum": 10000.0, "n_hvgs": 3000}
EMBED = {"n_pcs": 50, "n_neighbors": 15}
DOMAINS = {"method": "leiden", "resolution": 0.8}
DECONV = {"method": "flashdeconv", "cell_type_key": "cell_type"}
DEG = {"group_key": "tc_le_region", "method": "wilcoxon", "n_top_genes": 500}
MORAN = {"analysis_type": "moran", "n_neighbors": 6, "moran_n_perms": 100, "n_jobs": 8}
CELLCHAT = {"method": "cellchat_r", "species": "human", "cell_type_key": "cell_type", "cellchat_type": "triMean",
            "cellchat_min_cells": 10, "cellchat_population_size": False, "cellchat_distance_use": False,
            "cellchat_db_category": "All"}

SERVER = StdioServerParameters(
    command=sys.executable,
    args=["-m", "chatspatial", "server", "--transport", "stdio"],
    env={k: v for k, v in os.environ.items()},
)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def versions() -> dict:
    out = {}
    for p in ["chatspatial", "flashdeconv", "mcp", "scanpy", "squidpy", "anndata", "numpy", "scipy", "pandas",
              "scikit-learn", "statsmodels", "leidenalg", "igraph", "python-igraph", "numba", "pydantic", "rpy2"]:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    out["python"] = platform.python_version()
    try:
        r = subprocess.run(["Rscript", "-e", 'suppressMessages(library(CellChat)); cat(R.version.string, "|", '
                            'as.character(packageVersion("CellChat")))'], capture_output=True, text=True, timeout=300)
        rv, cc = (r.stdout.strip().split("|") + ["", ""])[:2]
        out["R"], out["CellChat"] = rv.strip(), cc.strip()
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


async def run_section(rec: Recorder, session: ClientSession, sid: str, ref_id: str) -> None:
    gsm = f"GSM{6339630 + int(sid[1:])}"
    loaded = await rec.call(session, "load_data", {
        "data_path": str(VISIUM / sid / "filtered_feature_bc_matrix.h5"), "data_type": "visium", "name": sid},
        "step1_load", sid)
    did = data_id_of(loaded)
    await rec.call(session, "preprocess_data", {"data_id": did, "params": PREPROCESS}, "step1_preprocess", sid)
    await rec.call(session, "compute_embeddings", {"data_id": did, "params": EMBED}, "step1_embeddings", sid)
    dom = await rec.call(session, "identify_spatial_domains", {"data_id": did, "params": DOMAINS},
                         "step1_leiden_domains", sid)
    domain_key = dom["domain_key"]
    p1 = D_STEP1 / f"{sid}.h5ad"
    await rec.call(session, "export_data", {"data_id": did, "path": str(p1)}, "step1_export", sid)

    # Client-side post-processing: TC/LE assignment from the Leiden domains (rule in oscc157_tc_le.py).
    p2 = D_TCLE / f"{sid}.h5ad"
    summary = assign_tc_le(p1, p2, domain_key)
    rec.post("step1_tc_le_assignment", sid, summary.pop("rule"), [str(p1)], [str(p2)], summary)
    await rec.call(session, "reload_data", {"data_id": did, "path": str(p2)}, "step1_reload_tc_le", sid)

    await rec.call(session, "deconvolve_data", {"data_id": did, "params": {**DECONV, "reference_data_id": ref_id}},
                   "step2_deconvolution", sid)

    n_tc, n_le = summary["n_TC"], summary["n_LE"]
    if n_tc >= 3 and n_le >= 3:
        for g1, g2 in (("LE", "TC"), ("TC", "LE")):
            await rec.call(session, "find_markers", {"data_id": did, "params": {**DEG, "group1": g1, "group2": g2}},
                           f"step4_deg_{g1}_vs_{g2}", sid)
    else:
        rec.post("step4_deg_skipped", sid, "find_markers needs both TC and LE spots", [], [],
                 {"n_TC": n_tc, "n_LE": n_le})

    genes = [str(g) for g in ad.read_h5ad(p2, backed="r").var_names]
    await rec.call(session, "analyze_spatial_statistics", {"data_id": did, "params": {**MORAN, "genes": genes}},
                   "step4_moran", sid)
    await rec.call(session, "export_data", {"data_id": did, "path": str(D_FINAL / f"{sid}.h5ad")},
                   "final_export", sid)


async def main() -> None:
    rec = Recorder(LOG_PATH)
    rec.write({
        "record": "session", "started_utc": now(),
        "purpose": "OSCC case study (GSE208253, 12 Visium sections) run end to end through ChatSpatial MCP tools: "
                   "QC and preprocessing, Leiden spatial domains, TC/LE assignment, FlashDeconv deconvolution with "
                   "the Puram et al. HNSCC reference, TC vs LE marker genes, Moran's I, CellChat",
        "versions": versions(), "server_command": [SERVER.command, *SERVER.args], "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "inputs": {"visium_sections": str(VISIUM), "reference_h5ad": str(REF_PATH),
                   "visium_source": "GEO GSE208253 GSM6339631-GSM6339642: filtered_feature_bc_matrix.h5, "
                                    "tissue_positions_list.csv, scalefactors_json.json, tissue_hires_image.png",
                   "reference_source": "GEO GSE103322_HNSCC_all_data.txt.gz, values as deposited (log2(TPM/10+1)), "
                                       "labels from row 'non-cancer cell type' (0 -> Cancer)"},
        "fixed_parameters": {"preprocess": PREPROCESS, "embeddings": EMBED, "domains": DOMAINS, "deconvolution": DECONV,
                             "deg": DEG, "moran": {**MORAN, "genes": "all genes of the section after QC"},
                             "cellchat": CELLCHAT},
    })
    async with stdio_client(SERVER) as streams:
        read, write = streams[0], streams[1]
        async with ClientSession(read, write, logging_callback=rec.on_log) as session:
            init = await session.initialize()
            tools = await session.list_tools()
            rec.write({"record": "server", "server_info": init.server_info.model_dump()
                       if hasattr(init, "server_info") else str(init), "tools": sorted(t.name for t in tools.tools)})

            ref = await rec.call(session, "load_data", {
                "data_path": str(REF_PATH), "data_type": "generic", "name": "puram_hnscc_reference"},
                "step2_load_reference")
            ref_id = data_id_of(ref)
            for sid in SAMPLES:
                await run_section(rec, session, sid, ref_id)

            # Client-side post-processing: pool TC and LE spots of the 12 sections for CellChat.
            for label, regions in (("tc_le", ("TC", "LE")), ("le_only", ("LE",))):
                pooled = D_POOL / f"pooled_{label}.h5ad"
                summary = build_pooled([D_FINAL / f"{s}.h5ad" for s in SAMPLES], pooled, regions)
                rec.post(f"step3_pool_{label}", None, summary.pop("rule"),
                         [str(D_FINAL / f"{s}.h5ad") for s in SAMPLES], [str(pooled)], summary)
                pl = await rec.call(session, "load_data", {
                    "data_path": str(pooled), "data_type": "generic", "name": f"oscc_pooled_{label}"},
                    f"step3_load_pooled_{label}")
                pid = data_id_of(pl)
                await rec.call(session, "analyze_cell_communication", {"data_id": pid, "params": CELLCHAT},
                               f"step3_cellchat_{label}")
                await rec.call(session, "export_data", {"data_id": pid, "path": str(D_POOL / f"cellchat_{label}.h5ad")},
                               f"step3_export_{label}")
    rec.write({"record": "session_end", "finished_utc": now()})
    print("DRIVER_DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
