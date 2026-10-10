#!/usr/bin/env python3
"""ChatSpatial arm of experiment B-X: one trial = one conversation.

The LLM (claude-sonnet-5-5, provider defaults: no temperature, default adaptive thinking and
effort) gets every tool of the installed ChatSpatial MCP server (1.5.6 for the probes,
1.5.9 for the head-to-head) with its full published JSON
schema (native tool calling, tool_choice auto) and the server's own instructions, as an MCP
client would present them. Tools run in-process through the MCP server's an in-memory MCP client
session (mcp.client.Client over the real protocol). A tool error reaches the model as the
is_error tool_result the MCP client receives.

Usage: python expBX_chatspatial_agent.py --prompt "..." --out_dir DIR [--h2h]
Writes DIR/transcript.jsonl (every API response, every tool call and result, incrementally),
DIR/result.json, and for --h2h DIR/labels.csv (labels of the last successful
identify_spatial_domains call).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import expBX_common as C  # noqa: E402

SYSTEM_PREFIX = (
    "You are an assistant for spatial transcriptomics analysis. You are connected to the "
    "ChatSpatial MCP server and can use its tools. Carry out the user's request with these tools "
    "and report the outcome to the user."
)


def content_to_blocks(content) -> list[dict]:
    blocks = []
    for c in content or []:
        t = getattr(c, "type", None)
        if t == "text":
            blocks.append({"type": "text", "text": c.text})
        elif t == "image":
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": c.mimeType
                                                       if hasattr(c, "mimeType") else c.mime_type, "data": c.data}})
        else:
            blocks.append({"type": "text", "text": str(c)})
    return blocks or [{"type": "text", "text": "(empty result)"}]


async def call_tool(mcpc, name: str, args: dict):
    """Call through the in-memory MCP client session -> (blocks, is_error, raw_text)."""
    try:
        res = await mcpc.call_tool(name, args)
        is_error = bool(getattr(res, "is_error", False) or getattr(res, "isError", False))
        blocks = content_to_blocks(getattr(res, "content", None))
    except Exception as e:  # transport-level failure (not a tool error)
        is_error = True
        blocks = [{"type": "text", "text": f"Error executing tool {name}: {type(e).__name__}: {e}"}]
    raw = "\n".join(b.get("text", "[image]") for b in blocks)
    return blocks, is_error, raw


async def main_async(args) -> dict:
    if C.BACKEND == "anthropic":
        import anthropic
    from chatspatial import __version__ as cs_version
    from chatspatial import server

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tlog = out / "transcript.jsonl"
    from mcp.client import Client
    mcp = server.mcp
    client_cm = Client(mcp)  # real MCP protocol, in-memory transport
    mcpc = await client_cm.__aenter__()
    tools_mcp = (await mcpc.list_tools()).tools
    tools = []
    for t in tools_mcp:
        schema = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None)
        d = {"name": t.name, "input_schema": schema}
        if t.description:
            d["description"] = t.description
        tools.append(d)
    tools[-1]["cache_control"] = {"type": "ephemeral"}
    instructions = mcp.instructions or ""
    system = [{"type": "text", "text": SYSTEM_PREFIX + ("\n\nServer instructions:\n" + instructions if instructions else ""),
               "cache_control": {"type": "ephemeral"}}]
    C.append_jsonl(tlog, {"event": "setup", "chatspatial_version": cs_version, "model": C.MODEL,
                          "n_tools": len(tools), "tool_names": [t["name"] for t in tools],
                          "system": system[0]["text"], "user_prompt": args.prompt,
                          "backend": C.BACKEND, "base_url": C.BASE_URL})

    if C.BACKEND == "openai":
        return await openai_loop(args, out, tlog, mcpc, client_cm, server, tools, system[0]["text"])
    client = anthropic.Anthropic(api_key=C.anthropic_key(), max_retries=6, timeout=600)
    messages = [{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]
    usage_tot = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
                 "cache_creation_input_tokens": 0}
    cost = 0.0
    tool_calls = []
    stop_reason = None
    n_turns = 0
    t0 = time.time()
    while n_turns < C.CHATSPATIAL_MAX_TURNS:
        n_turns += 1
        # rolling cache breakpoint on the newest message (max 4 breakpoints: tools, system, this one)
        for m in messages:
            if isinstance(m["content"], list):
                for b in m["content"]:
                    if isinstance(b, dict):
                        b.pop("cache_control", None)
        last = messages[-1]["content"][-1]
        if isinstance(last, dict) and last.get("type") in ("text", "tool_result"):
            last["cache_control"] = {"type": "ephemeral"}
        resp = client.messages.create(model=C.MODEL, max_tokens=C.MAX_TOKENS, system=system,
                                      tools=tools, messages=messages)
        rd = resp.model_dump()
        u = rd.get("usage") or {}
        for k in usage_tot:
            usage_tot[k] += u.get(k) or 0
        cost += C.usage_cost(u)
        stop_reason = resp.stop_reason
        C.append_jsonl(tlog, {"event": "api_response", "turn": n_turns, "t": time.time() - t0,
                              "id": rd.get("id"), "model": rd.get("model"), "stop_reason": stop_reason,
                              "stop_details": rd.get("stop_details"), "usage": u, "content": rd.get("content")})
        # append the full assistant content unchanged (thinking blocks included)
        messages.append({"role": "assistant", "content": resp.content})
        uses = [b for b in rd["content"] if b.get("type") == "tool_use"]
        if stop_reason != "tool_use" or not uses:
            break
        results = []
        for b in uses:
            ts = time.time()
            blocks, is_err, raw = await call_tool(mcpc, b["name"], b.get("input") or {})
            rec = {"turn": n_turns, "id": b["id"], "name": b["name"], "input": b.get("input"),
                   "is_error": is_err, "result_text": raw[:20000], "seconds": time.time() - ts}
            tool_calls.append(rec)
            C.append_jsonl(tlog, {"event": "tool_result", **rec})
            results.append({"type": "tool_result", "tool_use_id": b["id"], "content": blocks, "is_error": is_err})
        messages.append({"role": "user", "content": results})

    # assistant prose shown to the user (all turns) and the final message
    protocol = getattr(mcpc, "protocol_version", None)
    texts = []
    for m in messages:
        if m["role"] == "assistant":
            texts.append("\n".join(b.text for b in m["content"] if getattr(b, "type", None) == "text"))
    final_text = texts[-1] if texts else ""
    result = {"system": "chatspatial", "model": C.MODEL, "stop_reason": stop_reason, "n_turns": n_turns,
              "hit_turn_cap": n_turns >= C.CHATSPATIAL_MAX_TURNS and stop_reason == "tool_use",
              "n_tool_calls": len(tool_calls), "n_tool_errors": sum(t["is_error"] for t in tool_calls),
              "tool_sequence": [t["name"] for t in tool_calls], "final_text": final_text,
              "assistant_texts": texts, "usage": usage_tot, "cost_usd": cost, "wall_time": time.time() - t0,
              "temperature": "provider_default", "thinking": "provider_default", "mcp_protocol": protocol}

    if args.h2h:
        # labels of the last successful identify_spatial_domains call
        dom = [t for t in tool_calls if t["name"] == "identify_spatial_domains" and not t["is_error"]]
        result["domain_calls"] = [{"input": t["input"], "is_error": t["is_error"]} for t in tool_calls
                                  if t["name"] == "identify_spatial_domains"]
        if dom:
            last = dom[-1]
            try:
                info = json.loads(last["result_text"])
            except Exception:
                info = {}
            data_id = (last["input"] or {}).get("data_id")
            params = (last["input"] or {}).get("params") or {}
            result["method"] = params.get("method", "default")
            result["domain_params"] = params
            key = info.get("domain_key")
            rkey = info.get("refined_domain_key")
            result["domain_key"], result["refined_domain_key"] = key, rkey
            try:
                ds = await server.data_manager.get_dataset(data_id)
                ad_ = ds["adata"] if isinstance(ds, dict) else ds.adata
                if key and key in ad_.obs:
                    # primary label = the tool's final output (refined labels when the call produced
                    # them, the experiment-A convention); unrefined labels kept for sensitivity
                    prim = rkey if (rkey and rkey in ad_.obs) else key
                    df = ad_.obs[[prim]].rename(columns={prim: "label"}).astype(str)
                    df["label_unrefined"] = ad_.obs[key].astype(str).values
                    df.to_csv(out / "labels.csv")
                    result["labels_file"] = str(out / "labels.csv")
                    result["label_column_used"] = prim
                    result["n_labels"] = int(df["label"].nunique())
            except Exception as e:
                result["label_export_error"] = f"{type(e).__name__}: {e}"
    try:
        await client_cm.__aexit__(None, None, None)
    except Exception:
        pass
    return result


async def openai_loop(args, out, tlog, mcpc, client_cm, server, tools, system_text) -> dict:
    """Open-weight arm: the same loop against an OpenAI-compatible local vLLM endpoint (native tool calling).
    Tool results are returned as tool messages (text; an image result is replaced by a one-line note)."""
    import httpx
    otools = []
    for t in tools:
        f = {"name": t["name"], "parameters": t["input_schema"]}
        if t.get("description"):
            f["description"] = t["description"]
        otools.append({"type": "function", "function": f})
    messages = [{"role": "system", "content": system_text}, {"role": "user", "content": args.prompt}]
    usage_tot = {"input_tokens": 0, "output_tokens": 0}
    tool_calls_log, texts = [], []
    stop_reason, n_turns, t0 = None, 0, time.time()
    http = httpx.Client(timeout=1800)
    while n_turns < C.CHATSPATIAL_MAX_TURNS:
        n_turns += 1
        body = {"model": C.MODEL, "messages": messages, "tools": otools, "tool_choice": "auto",
                "max_tokens": C.MAX_TOKENS}
        for attempt in range(3):
            r = http.post(C.BASE_URL.rstrip("/") + "/chat/completions", json=body)
            if r.status_code < 500:
                break
            time.sleep(10)
        if r.status_code != 200:
            C.append_jsonl(tlog, {"event": "api_error", "turn": n_turns, "status": r.status_code, "text": r.text[:3000]})
            stop_reason = f"http_{r.status_code}"
            break
        d = r.json()
        u = d.get("usage") or {}
        usage_tot["input_tokens"] += u.get("prompt_tokens") or 0
        usage_tot["output_tokens"] += u.get("completion_tokens") or 0
        ch = d["choices"][0]
        msg = ch["message"]
        stop_reason = ch.get("finish_reason")
        content = []
        if msg.get("content"):
            content.append({"type": "text", "text": msg["content"]})
        for tc in msg.get("tool_calls") or []:
            content.append({"type": "tool_use", "id": tc.get("id"), "name": tc["function"]["name"],
                            "input": tc["function"].get("arguments")})
        C.append_jsonl(tlog, {"event": "api_response", "turn": n_turns, "t": time.time() - t0, "id": d.get("id"),
                              "model": d.get("model"), "stop_reason": stop_reason, "usage": u, "content": content,
                              "reasoning": (msg.get("reasoning_content") or msg.get("reasoning") or "")[:6000]})
        texts.append(msg.get("content") or "")
        am = {"role": "assistant", "content": msg.get("content") or ""}
        if msg.get("tool_calls"):
            am["tool_calls"] = msg["tool_calls"]
        messages.append(am)
        if not msg.get("tool_calls"):
            break
        for tc in msg["tool_calls"]:
            name = tc["function"]["name"]
            try:
                targs = json.loads(tc["function"].get("arguments") or "{}")
                if not isinstance(targs, dict):
                    raise ValueError("arguments are not a JSON object")
                ts = time.time()
                blocks, is_err, raw = await call_tool(mcpc, name, targs)
            except Exception as e:
                targs, is_err, raw, ts = tc["function"].get("arguments"), True, f"Invalid tool arguments: {e}", time.time()
            raw = raw.replace("[image]", "[image output omitted: this client is text-only]")
            rec = {"turn": n_turns, "id": tc.get("id"), "name": name, "input": targs, "is_error": is_err,
                   "result_text": raw[:20000], "seconds": time.time() - ts}
            tool_calls_log.append(rec)
            C.append_jsonl(tlog, {"event": "tool_result", **rec})
            messages.append({"role": "tool", "tool_call_id": tc.get("id"),
                             "content": (("Error: " if is_err else "") + raw)[:60000]})
    final_text = texts[-1] if texts else ""
    try:
        await client_cm.__aexit__(None, None, None)
    except Exception:
        pass
    return {"system": "chatspatial", "model": C.MODEL, "backend": C.BACKEND, "stop_reason": stop_reason,
            "n_turns": n_turns, "hit_turn_cap": n_turns >= C.CHATSPATIAL_MAX_TURNS and bool(tool_calls_log),
            "n_tool_calls": len(tool_calls_log), "n_tool_errors": sum(t["is_error"] for t in tool_calls_log),
            "tool_sequence": [t["name"] for t in tool_calls_log], "final_text": final_text,
            "assistant_texts": texts, "usage": usage_tot, "cost_usd": 0.0, "wall_time": time.time() - t0,
            "temperature": "server_default", "thinking": "server_default"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--h2h", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir)
    try:
        result = asyncio.run(main_async(args))
        result["harness_error"] = None
    except Exception as e:
        result = {"system": "chatspatial", "harness_error": f"{type(e).__name__}: {e}",
                  "traceback": traceback.format_exc()[-3000:]}
    (out / "result.json").write_text(json.dumps(result, indent=1, default=str))
    print("CHATSPATIAL_DONE", result.get("stop_reason"), result.get("harness_error"), flush=True)


if __name__ == "__main__":
    main()
