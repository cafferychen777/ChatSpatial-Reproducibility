#!/usr/bin/env python3
"""Experiment B invocation driver (LLM calls only; no ChatSpatial import).

Usage examples
  python expB_invocation.py --backend anthropic --model claude-haiku-4-5-20251001 --experiment main
  python expB_invocation.py --backend openai --model gpt-6-sol --experiment main --workers 3
  python expB_invocation.py --backend openai_compat --base-url http://127.0.0.1:8000/v1 \
      --model qwen --label Qwen/Qwen3-... --experiment main

Writes one JSONL line per trial to $EXPB_DIR/invocation/<label>__<experiment>.jsonl
(or --out). Resumable: trials already present (by trial_key, without error) are skipped.
Each record holds the full request body (minus credentials), the full response,
timestamps, latency, and the parsed call.

Fixed settings (2026-10-08): temperature 1.0 where the API accepts it
(otherwise the provider default, recorded); tool_choice "auto" for every
native request (Claude Sonnet/Opus 5.5 reject forced tool choice, so "auto"
is used uniformly); one user turn; each model's default reasoning/thinking
configuration; max output tokens 4096.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

import expB_common as C

OUT_DIR = Path(os.environ.get("EXPB_DIR") or Path(os.environ.get("CHATSPATIAL_WORK_DIR", ".")) / "expB") / "invocation"
MAX_TOKENS = 4096
MAX_RETRIES = 5

# Models whose API rejects an explicit temperature (provider default is used)
NO_TEMPERATURE = {"claude-sonnet-5-5", "claude-opus-5-5"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


_KEY_CACHE: dict = {}


def anthropic_key() -> str:
    """Key from $ANTHROPIC_API_KEY, else read at runtime from $ANTHROPIC_KEY_FILE
    (default ~/.config/chatspatial/anthropic_key). Never logged."""
    if "k" not in _KEY_CACHE:
        k = os.environ.get("ANTHROPIC_API_KEY")
        if not k:
            path = os.path.expanduser(os.environ.get("ANTHROPIC_KEY_FILE", "~/.config/chatspatial/anthropic_key"))
            k = open(path).read().strip()
        _KEY_CACHE["k"] = k
    return _KEY_CACHE["k"]


def build_body(backend: str, model: str, system: str, tools, prompt: str, temperature_ok: bool) -> dict:
    if backend == "anthropic":
        body = {"model": model, "max_tokens": MAX_TOKENS, "system": system,
                "messages": [{"role": "user", "content": prompt}]}
        if tools is not None:
            body["tools"] = C.to_anthropic_tools(tools)
            body["tool_choice"] = {"type": "auto"}
        if temperature_ok:
            body["temperature"] = C.TEMPERATURE
        return body
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
    if backend == "openai":
        body["max_completion_tokens"] = MAX_TOKENS
    else:
        body["max_tokens"] = MAX_TOKENS
    if tools is not None:
        body["tools"] = C.to_openai_tools(tools)
        body["tool_choice"] = "auto"
    if temperature_ok:
        body["temperature"] = C.TEMPERATURE
    return body


def post(backend: str, base_url: str, body: dict) -> tuple[int, dict | str]:
    if backend == "anthropic":
        url = "https://api.anthropic.com/v1/messages"
        headers = {"x-api-key": anthropic_key(), "anthropic-version": "2023-06-01",
                   "content-type": "application/json"}
    elif backend == "openai":
        root = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        url = root.rstrip("/") + "/chat/completions"
        headers = {"content-type": "application/json"}
        if os.environ.get("OPENAI_API_KEY"):
            headers["Authorization"] = f"Bearer {os.environ['OPENAI_API_KEY']}"
    else:
        url = base_url.rstrip("/") + "/chat/completions"
        headers = {"content-type": "application/json"}
    r = requests.post(url, headers=headers, json=body, timeout=300)
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, r.text


def parse_response(backend: str, mechanism: str, arm: str, resp: dict) -> dict:
    """Extract text, tool calls and the parsed call. parse_status in
    {ok, no_tool_call, bad_arguments, unparseable, malformed, empty}."""
    text_parts, calls = [], []
    if backend == "anthropic":
        for b in resp.get("content", []):
            if b.get("type") == "text":
                text_parts.append(b.get("text", ""))
            elif b.get("type") == "tool_use":
                calls.append({"name": b.get("name"), "arguments": b.get("input"), "raw_arguments": None})
        stop = resp.get("stop_reason")
    else:
        ch = (resp.get("choices") or [{}])[0]
        msg = ch.get("message", {}) or {}
        if msg.get("content"):
            text_parts.append(msg["content"])
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            raw = fn.get("arguments")
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError):
                args = None
            calls.append({"name": fn.get("name"), "arguments": args, "raw_arguments": raw})
        stop = ch.get("finish_reason")
    text = "\n".join(text_parts)
    out = {"text": text, "tool_calls": calls, "n_tool_calls": len(calls), "stop_reason": stop,
           "raw_tool_name": None, "canonical_tool": None, "params": None}
    if mechanism == "native":
        if not calls:
            out["parse_status"] = "no_tool_call"
            return out
        c0 = calls[0]
        out["raw_tool_name"] = c0["name"]
        out["canonical_tool"] = C.map_tool_name(c0["name"], arm)
        if not isinstance(c0["arguments"], dict):
            out["parse_status"] = "bad_arguments"
            return out
        out["params"] = c0["arguments"]
        out["parse_status"] = "ok"
        return out
    # json mechanism: original parser on the text output
    t = C.strip_think(text)
    if not t.strip():
        out["parse_status"] = "empty"
        return out
    parsed = C.parse_json_text(t)
    if parsed is None:
        out["parse_status"] = "unparseable"
        return out
    out["raw_tool_name"] = parsed.get("tool_name")
    out["canonical_tool"] = C.map_tool_name(parsed.get("tool_name"), arm)
    params = parsed.get("parameters", {})
    if parsed.get("tool_name") is None or not isinstance(params, dict):
        out["parse_status"] = "malformed"
        out["params"] = params if isinstance(params, dict) else None
        return out
    out["params"] = params
    out["parse_status"] = "ok"
    return out


def run_trial(trial: dict, args, temperature_ok: bool) -> dict:
    system, tools = C.request_spec(trial)
    body = build_body(args.backend, args.model, system, tools, trial["prompt"], temperature_ok)
    rec = dict(trial)
    rec.update(backend=args.backend, model=args.model, label=args.label, request=body,
               temperature=body.get("temperature", "provider_default"))
    err = None
    for attempt in range(MAX_RETRIES):
        t0 = time.time()
        rec["t_start"] = now()
        try:
            status, resp = post(args.backend, args.base_url, body)
        except requests.RequestException as e:
            status, resp = -1, f"{type(e).__name__}: {e}"
        rec["latency_s"] = round(time.time() - t0, 3)
        rec["t_end"] = now()
        rec["http_status"] = status
        if status == 200 and isinstance(resp, dict):
            rec["response"] = resp
            rec["response_model"] = resp.get("model")
            rec["usage"] = resp.get("usage")
            rec.update(parse_response(args.backend, trial["mechanism"], trial["arm"], resp))
            rec["error"] = None
            return rec
        err = resp if isinstance(resp, str) else json.dumps(resp)[:2000]
        if status in (400, 401, 403, 404):
            break
        if status == 429:
            time.sleep(30)
        time.sleep(min(60, 3 * 2 ** attempt))
    rec["response"] = None
    rec["error"] = err
    rec["parse_status"] = "api_error"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["anthropic", "openai", "openai_compat"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default=None, help="backend label used in file names / analysis")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--experiment", default="main", choices=["main", "desc", "bias"])
    ap.add_argument("--prompts", default=None, help="comma-separated prompt ids")
    ap.add_argument("--reps", type=int, default=C.N_REPS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-temperature", action="store_true")
    args = ap.parse_args()
    args.label = args.label or args.model
    out = Path(args.out) if args.out else OUT_DIR / f"{args.label.replace('/', '_')}__{args.experiment}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    temperature_ok = not (args.no_temperature or args.model in NO_TEMPERATURE)

    trials = C.build_trials(args.experiment, set(args.prompts.split(",")) if args.prompts else None, args.reps)
    done = set()
    if out.exists():
        for line in open(out):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error") is None:
                done.add(r["trial_key"])
    todo = [t for t in trials if t["trial_key"] not in done]
    print(f"{args.label} {args.experiment}: {len(trials)} trials, {len(done)} done, {len(todo)} to run -> {out}", flush=True)
    lock = threading.Lock()
    n_ok = n_err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_trial, t, args, temperature_ok) for t in todo]
        for i, f in enumerate(as_completed(futs), 1):
            rec = f.result()
            with lock, open(out, "a") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            if rec["error"] is None:
                n_ok += 1
            else:
                n_err += 1
                print(f"  ERROR {rec['trial_key']}: {str(rec['error'])[:300]}", flush=True)
            if i % 50 == 0:
                print(f"  {i}/{len(todo)} ok={n_ok} err={n_err}", flush=True)
    print(f"done ok={n_ok} err={n_err}", flush=True)


if __name__ == "__main__":
    main()
