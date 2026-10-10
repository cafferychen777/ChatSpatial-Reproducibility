#!/usr/bin/env python3
"""SpatialAgent driver for experiment B-X (one trial; runs inside the SpatialAgent venv on the cluster).

Same configuration as the original round-1 driver e2e_driver_spatialagent.py (data_path = the
directory holding the input, save_path = run directory, tool_retrieval=True,
tool_retrieval_method="llm", auto_interpret_figures=False, act_timeout=600, recursion limit 50),
with full logging (api_log.jsonl for every Anthropic call incl. internal ones, the complete
message list). Backend: claude-sonnet-5-5 with temperature=None (not sent; the model rejects
explicit temperature), default thinking, max_tokens 16000 (8192 in make_llm's default).
"""
import argparse
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import expBX_common as C  # noqa: E402
from expBX_competitor_common import (assistant_texts, install_api_logger, list_files,  # noqa: E402
                                     serialize_messages, tool_calls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--log_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    out = args.output_dir
    os.makedirs(out, exist_ok=True)
    if C.BACKEND == "anthropic":
        os.environ["ANTHROPIC_API_KEY"] = C.anthropic_key()
    else:  # open-weight arm: OpenAI-compatible local vLLM endpoint (client configuration only)
        os.environ["OPENAI_BASE_URL"] = C.BASE_URL
        os.environ["OPENAI_API_BASE"] = C.BASE_URL
        os.environ["OPENAI_API_KEY"] = "EMPTY"
        os.environ["CUSTOM_MODEL_BASE_URL"] = C.BASE_URL
        os.environ["CUSTOM_MODEL_API_KEY"] = "EMPTY"
        # SpatialAgent's existing custom-endpoint branch: ChatAnthropic -> vLLM's Anthropic-compatible /v1/messages
        os.environ["ANTHROPIC_API_URL"] = C.BASE_URL.rstrip("/").removesuffix("/v1")
        os.environ["ANTHROPIC_API_KEY"] = "EMPTY"
    stats = install_api_logger(os.path.join(args.log_dir, "api_log.jsonl"))
    sys.path.insert(0, str(C.COMPETITORS / "SpatialAgent"))

    result = {"system": "spatialagent", "model": C.MODEL, "success": False, "error": "", "prompt": args.prompt}
    t0 = time.time()
    msgs = []
    try:
        from spatialagent.agent import SpatialAgent, make_llm
        llm = make_llm(C.MODEL, temperature=None, track_cost=False, max_tokens=C.MAX_TOKENS)
        agent = SpatialAgent(llm=llm, data_path=os.path.dirname(args.data_path), save_path=out,
                             tool_retrieval=True, tool_retrieval_method="llm",
                             auto_interpret_figures=False, act_timeout=600)
        res = agent.run(user_query=args.prompt, config={"recursion_limit": C.COMPETITOR_RECURSION_LIMIT})
        msgs = res.get("messages", [])
        result["next_step"] = res.get("next_step", "unknown")
        result["success"] = True  # original rule: no exception
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)[:1500]}"
        result["traceback"] = traceback.format_exc()[-3000:]
    result["wall_time"] = time.time() - t0
    result["n_steps"] = len(msgs)
    texts = assistant_texts(msgs)
    result["assistant_texts"] = texts
    result["final_text"] = next((t for t in reversed(texts) if t.strip()), "")
    result["tool_calls"] = tool_calls(msgs)
    result["output_files"] = list_files(out)
    result["api_calls"] = stats["n_calls"]
    result["usage"] = stats["usage"]
    result["cost_usd"] = C.usage_cost(stats["usage"]) if C.BACKEND == "anthropic" else 0.0
    result["temperature"] = "provider_default"
    result["backend"] = C.BACKEND
    with open(os.path.join(args.log_dir, "messages.json"), "w") as f:
        json.dump(serialize_messages(msgs), f, default=str)
    with open(os.path.join(args.log_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=1, default=str)
    print(f"SPATIALAGENT_DONE success={result['success']} {result['error'][:200]}", flush=True)


if __name__ == "__main__":
    main()
