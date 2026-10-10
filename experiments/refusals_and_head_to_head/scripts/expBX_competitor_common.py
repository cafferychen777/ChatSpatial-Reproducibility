"""Shared helpers for the STAgent / SpatialAgent drivers (run inside each competitor venv).

install_api_logger(path): wraps the Anthropic SDK's Messages.create so that every request made
by the agent (including internal calls such as SpatialAgent's LLM tool retrieval) is logged
with its full response content, stop reason and token usage, incrementally (a killed run keeps
its log). The key is never logged.
"""
from __future__ import annotations

import json
import os
import time


def _dump(obj):
    try:
        return obj.model_dump()
    except Exception:
        return str(obj)


def _last_msg(msgs):
    """The newest request message (tool results / observations), truncated, for the record."""
    try:
        m = (msgs or [])[-1]
        c = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        if isinstance(c, list):
            out = []
            for b in c:
                if isinstance(b, dict):
                    if b.get("type") == "tool_result":
                        cc = b.get("content")
                        out.append("[tool_result] " + (cc if isinstance(cc, str) else json.dumps(cc, default=str)))
                    elif b.get("type") == "text":
                        out.append(b.get("text", ""))
                    elif b.get("type") == "image":
                        out.append("[image]")
            c = "\n".join(out)
        role = m.get("role") if isinstance(m, dict) else None
        return {"role": role, "content": (c or "")[:5000]}
    except Exception as e:
        return {"error": str(e)}


def install_api_logger(path: str):
    import anthropic.resources.messages as m
    stats = {"n_calls": 0, "usage": {}, "t0": time.time()}

    def log(rec):
        with open(path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def wrap(cls, label):
        orig = cls.create

        def create(self, *args, **kwargs):
            t = time.time()
            try:
                resp = orig(self, *args, **kwargs)
            except Exception as e:
                log({"event": "api_error", "where": label, "t": t - stats["t0"],
                     "error": f"{type(e).__name__}: {str(e)[:2000]}",
                     "request_model": kwargs.get("model"), "temperature_sent": "temperature" in kwargs})
                raise
            stats["n_calls"] += 1
            if hasattr(resp, "parse") and not hasattr(resp, "usage"):
                try:
                    resp_p = resp.parse()
                except Exception:
                    resp_p = None
            else:
                resp_p = resp
            if resp_p is not None and hasattr(resp_p, "usage"):
                resp_log = resp_p
                d = _dump(resp_log)
                u = d.get("usage") or {}
                for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
                    stats["usage"][k] = stats["usage"].get(k, 0) + (u.get(k) or 0)
                log({"event": "api_response", "where": label, "t": t - stats["t0"], "seconds": time.time() - t,
                     "id": d.get("id"), "model": d.get("model"), "stop_reason": d.get("stop_reason"),
                     "stop_details": d.get("stop_details"), "usage": u, "content": d.get("content"),
                     "n_request_messages": len(kwargs.get("messages") or []),
                     "temperature_sent": kwargs.get("temperature", "not_sent"),
                     "thinking_sent": kwargs.get("thinking", "not_sent"),
                     "n_tools": len(kwargs.get("tools") or []),
                     "last_request_message": _last_msg(kwargs.get("messages"))})
            else:  # streaming object
                log({"event": "api_stream", "where": label, "t": t - stats["t0"]})
            return resp

        cls.create = create

    wrap(m.Messages, "messages")
    try:
        import anthropic.resources.beta.messages.messages as bm
        wrap(bm.Messages, "beta.messages")
    except Exception:
        pass
    try:  # open-weight arm (OpenAI-compatible client)
        import openai.resources.chat.completions as oc
        orig = oc.Completions.create

        def ocreate(self, *args, **kwargs):
            t = time.time()
            try:
                resp = orig(self, *args, **kwargs)
            except Exception as e:
                log({"event": "api_error", "where": "openai.chat", "t": t - stats["t0"],
                     "error": f"{type(e).__name__}: {str(e)[:2000]}", "request_model": kwargs.get("model")})
                raise
            try:  # logging must never change the call's behaviour
                parsed = resp.parse() if hasattr(resp, "parse") and not hasattr(resp, "choices") else resp
                d = _dump(parsed)
                d = d if isinstance(d, dict) else {}
                stats["n_calls"] += 1
                u = d.get("usage") or {}
                uu = {"input_tokens": u.get("prompt_tokens", 0) or 0, "output_tokens": u.get("completion_tokens", 0) or 0}
                for k, v in uu.items():
                    stats["usage"][k] = stats["usage"].get(k, 0) + v
                ch = (d.get("choices") or [{}])[0]
                msg = ch.get("message") or {}
                content = []
                if msg.get("content"):
                    content.append({"type": "text", "text": msg["content"]})
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    content.append({"type": "tool_use", "name": fn.get("name"), "input": fn.get("arguments")})
                log({"event": "api_response", "where": "openai.chat", "t": t - stats["t0"], "seconds": time.time() - t,
                     "id": d.get("id"), "model": d.get("model"), "stop_reason": ch.get("finish_reason"),
                     "usage": uu, "content": content,
                     "reasoning": (msg.get("reasoning_content") or msg.get("reasoning") or "")[:4000],
                     "n_request_messages": len(kwargs.get("messages") or []),
                     "temperature_sent": kwargs.get("temperature", "not_sent"),
                     "n_tools": len(kwargs.get("tools") or []),
                     "last_request_message": _last_msg(kwargs.get("messages"))})
            except Exception as e:
                log({"event": "log_error", "where": "openai.chat", "error": f"{type(e).__name__}: {e}"})
            return resp

        oc.Completions.create = ocreate
    except Exception:
        pass
    return stats


def serialize_messages(msgs) -> list:
    try:
        from langchain_core.messages import messages_to_dict
        return messages_to_dict(msgs)
    except Exception:
        return [{"type": type(m).__name__, "content": getattr(m, "content", str(m))} for m in msgs]


def assistant_texts(msgs) -> list[str]:
    """Natural-language text of every AI message (what the user sees), excluding tool calls."""
    out = []
    for m in msgs:
        if type(m).__name__ not in ("AIMessage", "AIMessageChunk"):
            continue
        c = getattr(m, "content", "")
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, list):
            out.append("\n".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text"))
    return out


def tool_calls(msgs) -> list[dict]:
    calls = {}
    order = []
    for m in msgs:
        for tc in getattr(m, "tool_calls", None) or []:
            calls[tc.get("id")] = {"name": tc.get("name"), "args": tc.get("args"), "output": None}
            order.append(tc.get("id"))
        if type(m).__name__ == "ToolMessage":
            tid = getattr(m, "tool_call_id", None)
            if tid in calls:
                c = m.content
                calls[tid]["output"] = c if isinstance(c, str) else json.dumps(c, default=str)
    return [calls[i] for i in order]


def list_files(d: str) -> list[str]:
    res = []
    for root, _, files in os.walk(d):
        for f in files:
            res.append(os.path.relpath(os.path.join(root, f), d))
    return sorted(res)
