#!/usr/bin/env python3
"""Continuity arm 'chatspatial_single': the original May-2026 DLPFC protocol
(round-1 dlpfc_benchmark.py, run_chatspatial), re-run with the current backend.

One LLM call: system = the original SCHEMA_FULL + RESPONSE_INSTRUCTION text, user = the original
open DLPFC prompt; the JSON reply is parsed with the original parser, validated against
the installed ChatSpatial release's SpatialDomainParameters (1.5.9 for the reported runs) and executed once with identify_spatial_domains on a
fresh copy of the section's input. Changes from May 2026 (rules file): model claude-sonnet-5-5
with provider-default sampling (no temperature) and default thinking; max_tokens 16000 (512 in
May; thinking tokens count against max_tokens); full-resolution pixel coordinates.
Scored label = domain_key (unrefined), as in May 2026; refined labels kept when produced.

Usage: python expBX_single_call.py --sid 151673 --out_dir DIR
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import expBX_common as C  # noqa: E402


class BenchCtx:
    """Minimal ToolContext (as in dlpfc_benchmark.py / expA common.py)."""

    def __init__(self, datasets):
        self._datasets = datasets
        self.warnings = []

    async def get_adata(self, data_id):
        return self._datasets[data_id]

    async def get_dataset_info(self, data_id):
        return {"adata": self._datasets[data_id], "name": data_id, "type": "visium"}

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tlog = out / "transcript.jsonl"
    result = {"system": "chatspatial_single", "model": C.MODEL, "success": False, "error": "",
              "temperature": "provider_default", "thinking": "provider_default"}
    t0 = time.time()
    try:
        import anthropic
        from ablation_invocation import RESPONSE_INSTRUCTION, SCHEMA_FULL, parse_json, resolve_tool_name
        client = anthropic.Anthropic(api_key=C.anthropic_key(), max_retries=6, timeout=600)
        resp = client.messages.create(model=C.MODEL, max_tokens=C.MAX_TOKENS,
                                      system=SCHEMA_FULL + RESPONSE_INSTRUCTION,
                                      messages=[{"role": "user", "content": C.DLPFC_PROMPT}])
        rd = resp.model_dump()
        C.append_jsonl(tlog, {"event": "api_response", "id": rd.get("id"), "model": rd.get("model"),
                              "stop_reason": rd.get("stop_reason"), "usage": rd.get("usage"),
                              "content": rd.get("content")})
        result["usage"] = rd.get("usage")
        result["cost_usd"] = C.usage_cost(rd.get("usage") or {})
        text = "\n".join(b.text for b in resp.content if b.type == "text")
        result["llm_response"] = text
        parsed = parse_json(text)
        if parsed is None:
            result["error"] = "json_parse_failed"
            raise SystemExit
        tool = resolve_tool_name(parsed.get("tool_name"))
        params_dict = parsed.get("parameters", {}) or {}
        result["invoked_tool"], result["parsed_params"] = tool, params_dict
        if tool != "identify_spatial_domains":
            result["error"] = f"wrong_tool:{tool}"
            raise SystemExit
        import anndata as ad
        from chatspatial.models.data import SpatialDomainParameters
        from chatspatial.tools.spatial_domains import identify_spatial_domains
        params = SpatialDomainParameters(**params_dict)
        result["method"] = params.method
        result["validated_params"] = params.model_dump(mode="json")
        adata = ad.read_h5ad(C.dlpfc_input(args.sid))
        adata.var_names_make_unique()
        ctx = BenchCtx({"bench_data": adata})
        tr = asyncio.run(identify_spatial_domains("bench_data", ctx, params))
        a2 = ctx._datasets["bench_data"]
        df = a2.obs[[tr.domain_key]].rename(columns={tr.domain_key: "label"}).astype(str)
        if getattr(tr, "refined_domain_key", None) and tr.refined_domain_key in a2.obs:
            df["label_refined"] = a2.obs[tr.refined_domain_key].astype(str).values
        df.to_csv(out / "labels.csv")
        result.update(success=True, domain_key=tr.domain_key, refined_domain_key=getattr(tr, "refined_domain_key", None),
                      n_labels=int(df["label"].nunique()), labels_file=str(out / "labels.csv"),
                      tool_warnings=ctx.warnings)
    except SystemExit:
        pass
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)[:1000]}"
        result["traceback"] = traceback.format_exc()[-3000:]
    result["wall_time"] = time.time() - t0
    (out / "result.json").write_text(json.dumps(result, indent=1, default=str))
    print("SINGLE_DONE", result["success"], result["error"][:200], flush=True)


if __name__ == "__main__":
    main()
