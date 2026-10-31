#!/usr/bin/env python3
"""A stand-in for llama-server in teacher_ab_run.sh tests (stdlib only, loopback only).

Takes llama-server's flags (-m, --alias, --host, --port, -c, --parallel,
--kv-unified, --chat-template-file and the sampling flags matter), answers
/health, /props (model path, per-slot context, chat template, sampler
defaults), /v1/models (the alias) and /v1/chat/completions: the strict grader's
prompt gets a JSON verdict, the compactor's prompt a summary, anything else a
reply in the llama.cpp streaming wire: a wiki_query call first when tools are
offered, then a text answer quoting the prompt. Every request body is appended,
with the alias, to the JSONL file named by FAKE_LLAMA_LOG. A prompt holding the
text in FAKE_LLAMA_FAIL_<ALIAS> (alias upper-cased, non-alphanumerics as _) gets
a short text-only answer instead (a task the arm fails). FAKE_LLAMA_TRUNCATE_FIRST=N
cuts the first N agent replies off mid-stream the way llama-server does on a
server error (a ``data: {"error": ...}`` event, no finish_reason).
FAKE_LLAMA_AGENT_DELAY (seconds) delays every agent reply; FAKE_LLAMA_HEALTH_DELAY
keeps /health at 503 for that long after the start.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def arg(flag: str, default: str = "") -> str:
    argv = sys.argv[1:]
    return argv[argv.index(flag) + 1] if flag in argv else default


ALIAS = arg("--alias", "fake")
PORT = int(arg("--port", "18080"))
HOST = arg("--host", "127.0.0.1")
MODEL = arg("-m", "")
SLOTS = int(arg("--parallel", "1"))
CTX = int(arg("-c", "4096"))
N_CTX_SLOT = CTX if "--kv-unified" in sys.argv[1:] else CTX // max(1, SLOTS)
TEMPLATE_FILE = arg("--chat-template-file", "")
TEMPLATE = (open(TEMPLATE_FILE, encoding="utf-8").read() if TEMPLATE_FILE
            else os.environ.get("FAKE_LLAMA_TEMPLATE", "{% for m in messages %}{{ m.content }}{% endfor %}"))
LOG = os.environ.get("FAKE_LLAMA_LOG", "")
FAIL = os.environ.get("FAKE_LLAMA_FAIL_" + re.sub(r"[^A-Z0-9]", "_", ALIAS.upper()), "")
TRUNCATE = [int(os.environ.get("FAKE_LLAMA_TRUNCATE_FIRST", "0") or 0)]
AGENT_DELAY = float(os.environ.get("FAKE_LLAMA_AGENT_DELAY", "0") or 0)
READY_AT = time.time() + float(os.environ.get("FAKE_LLAMA_HEALTH_DELAY", "0") or 0)
LOCK = threading.Lock()


def props() -> dict:
    params = {"temperature": float(arg("--temp", "0.8")), "top_k": int(arg("--top-k", "40")),
              "top_p": float(arg("--top-p", "0.95")), "min_p": float(arg("--min-p", "0.05")),
              "repeat_penalty": 1.0, "presence_penalty": 0.0, "dry_multiplier": 0.0, "xtc_probability": 0.0,
              "samplers": ["penalties", "dry", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"],
              "chat_format": "Hermes 2 Pro", "n_predict": -1}
    return {"default_generation_settings": {"params": params, "n_ctx": N_CTX_SLOT}, "total_slots": SLOTS,
            "model_alias": ALIAS, "model_path": MODEL, "chat_template": TEMPLATE, "build_info": "fake"}


def text_of(content) -> str:  # noqa: ANN001
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def last_user(messages: list) -> tuple[int, str]:
    for k in range(len(messages) - 1, -1, -1):
        m = messages[k]
        if m.get("role") == "user" and not text_of(m.get("content")).startswith("[Compacted summary"):
            return k, text_of(m.get("content"))
    return 0, ""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a) -> None:  # noqa: ANN002
        pass

    def _json(self, code: int, obj) -> None:  # noqa: ANN001
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            if time.time() < READY_AT:
                self._json(503, {"error": {"message": "Loading model"}})
            else:
                self._json(200, {"status": "ok"})
        elif self.path == "/props":
            self._json(200, props())
        elif self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": ALIAS}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        msgs = body.get("messages") or []
        if msgs and text_of(msgs[0].get("content")).startswith("You are a strict grader"):
            kind = "judge"
            reply = {"content": json.dumps({"verdict": "pass", "completed": True, "tools_ok": True,
                                            "grounded": True, "recovery_ok": None, "reason": "fine"})}
        elif any("You are a conversation compactor" in text_of(m.get("content")) for m in msgs):
            kind = "summary"
            reply = {"content": "Earlier: the user read several files."}
        else:
            kind = "agent"
            if AGENT_DELAY:
                time.sleep(AGENT_DELAY)
            k, prompt = last_user(msgs)
            step = sum(1 for m in msgs[k + 1:] if m.get("role") == "assistant")
            if FAIL and FAIL in prompt:
                reply = {"content": "No."}
            elif body.get("tools") and step < 2:
                reply = {"tool_calls": [("wiki_query", {"query": "service host"})]}
            else:
                reply = {"content": "Here is what I found about your request. Request: " + prompt[-2000:]}
        if LOG:
            with open(LOG, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"alias": ALIAS, "kind": kind, "body": body}) + "\n")
        if not body.get("stream"):
            self._json(200, {"model": ALIAS, "choices": [{"message": {"role": "assistant",
                                                                      "content": reply.get("content")},
                                                          "finish_reason": "stop"}]})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        with LOCK:
            cut = kind == "agent" and TRUNCATE[0] > 0
            if cut:
                TRUNCATE[0] -= 1
        if cut:  # llama-server on a failed decode mid-stream: an error event, no finish_reason
            self.wfile.write(('data: ' + json.dumps({"choices": [{"delta": {"content": "Here is wh"}}],
                                                     "model": ALIAS}) + "\n\n").encode())
            self.wfile.write(('data: ' + json.dumps({"error": {"code": 500, "type": "server_error",
                              "message": "Context size has been exceeded."}}) + "\n\n").encode())
            self.wfile.flush()
            return
        chunks = []
        if reply.get("content"):
            chunks.append({"choices": [{"delta": {"content": reply["content"]}}]})
        for i, (name, args) in enumerate(reply.get("tool_calls") or []):
            chunks.append({"choices": [{"delta": {"tool_calls": [{
                "index": i, "id": f"call_{os.getpid()}_{id(body)}_{i}",
                "function": {"name": name, "arguments": json.dumps(args)}}]}}]})
        chunks.append({"choices": [{"delta": {}, "finish_reason": "tool_calls" if reply.get("tool_calls")
                                    else "stop"}]})
        chunks.append({"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 20}})
        for c in chunks:
            c["model"] = ALIAS
            self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


if __name__ == "__main__":
    if HOST not in ("127.0.0.1", "localhost"):
        sys.exit("loopback only")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
