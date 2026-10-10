#!/usr/bin/env python3
"""STAgent driver for experiment B-X (one trial; runs inside the STAgent venv on the cluster).

Same invocation as the original round-1 driver e2e_driver_stagent.py (invoke_our_graph,
STAGENT_DATA_PATH / STAGENT_PLOT_DIR, recursion limit 50, emit_conflict_message=False, cwd = the
run directory), with full logging: every Anthropic API response (api_log.jsonl), the complete
message list, every tool call with its arguments (the generated code) and output.
Backend: claude-sonnet-5-5, no temperature sent (config entry with temperature None), default
thinking, max_tokens 16000.
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

    src = str(C.COMPETITORS / "STAgent" / "src")
    env_file = os.path.join(src, ".env")
    if os.path.exists(env_file):
        from dotenv import load_dotenv
        load_dotenv(env_file, override=False)
    os.environ["STAGENT_DATA_PATH"] = args.data_path
    os.environ["STAGENT_PLOT_DIR"] = out
    os.environ.setdefault("STAGENT_RECURSION_LIMIT", str(C.COMPETITOR_RECURSION_LIMIT))
    sys.path.insert(0, src)

    result = {"system": "stagent", "model": C.MODEL, "success": False, "error": "", "prompt": args.prompt}
    t0 = time.time()
    msgs = []
    try:
        from langchain_core.messages import HumanMessage
        from graph_unified import invoke_our_graph
        res = invoke_our_graph(messages=[HumanMessage(content=args.prompt)], model_name=C.MODEL,
                               session_id=f"bx_{int(time.time())}", emit_conflict_message=False)
        msgs = res.get("messages", [])
        # original success rule (e2e_driver_stagent.py)
        infra = {"conversation_histories", "research_reports", "driver_result.json", ".DS_Store",
                 "api_log.jsonl", "messages.json", "result.json", "driver_stdout.log"}
        real = [f for f in os.listdir(out) if f not in infra and os.path.isfile(os.path.join(out, f))]
        texts = "\n".join(assistant_texts(msgs)[-10:]).lower()
        api_err = any(k in texts for k in ["apiconnectionerror", "api connection error",
                                            "model provider request failed", "rate limit"])
        if api_err and not real:
            result["error"] = "API error during execution, no output produced"
        elif not real and len(msgs) <= 3:
            result["error"] = "No real output files produced (only infrastructure dirs)"
        else:
            result["success"] = True
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
    print(f"STAGENT_DONE success={result['success']} {result['error'][:200]}", flush=True)


if __name__ == "__main__":
    main()
