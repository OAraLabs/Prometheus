#!/usr/bin/env python3
"""FIRSTLIGHT stub inference server — an OpenAI-compatible model that isn't one.

Speaks exactly enough of the llama.cpp / OpenAI surface for a fresh
Prometheus install to detect it (``GET /v1/models``) and run agent turns
against it (``POST /v1/chat/completions``, streaming and non-streaming).
The "model" is a two-step script, stateless by construction:

  * request WITHOUT a tool result in its messages -> a ``tool_calls``
    response asking for ``glob {"pattern": "*"}`` (an always-advertised,
    read-only tool);
  * request WITH a tool result (any ``role: "tool"`` message) -> a final
    text answer carrying ``FIRSTLIGHT-COMPLETE``.

So the completion marker appearing in a transcript PROVES a full
model->tool->model round trip happened — the assertion rides the protocol,
not log scraping. Used by scripts/firstlight_harness.py; stdlib only; no
imports from src/prometheus (the harness contract).

Mutation modes (``--mode``) exist so the harness's own failure reporting
can be tested — each breaks exactly one step of the acceptance flow:

  normal        healthy two-step model (default)
  models-500    GET /v1/models returns 500 -> setup detects nothing
  no-final      NEVER returns final text -> the agent loop cannot conclude
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FINAL_MARKER = "FIRSTLIGHT-COMPLETE"
MODEL_ID = "firstlight-stub-model"

MODE = "normal"


def _completion_body(*, want_tool_call: bool) -> dict:
    """The assistant message for one round, non-streaming shape."""
    if want_tool_call:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": f"call_fl_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": "glob", "arguments": json.dumps({"pattern": "*"})},
            }],
        }
        finish = "tool_calls"
    else:
        message = {
            "role": "assistant",
            "content": f"{FINAL_MARKER}: the stub model saw the tool result and is done.",
        }
        finish = "stop"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18},
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet; the harness captures I/O itself
        pass

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- detection surface -------------------------------------------------
    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            if MODE == "models-500":
                self._send_json(500, {"error": "firstlight mutation: models-500"})
                return
            self._send_json(200, {
                "object": "list",
                "data": [{"id": MODEL_ID, "object": "model", "owned_by": "firstlight"}],
            })
            return
        if self.path.rstrip("/") in ("", "/health"):
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": f"no route {self.path}"})

    # -- completion surface ------------------------------------------------
    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send_json(404, {"error": f"no route {self.path}"})
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON"})
            return

        saw_tool_result = any(
            isinstance(m, dict) and m.get("role") == "tool"
            for m in req.get("messages", [])
        )
        want_tool_call = (MODE == "no-final") or not saw_tool_result

        if not req.get("stream"):
            self._send_json(200, _completion_body(want_tool_call=want_tool_call))
            return

        # SSE stream: role delta, payload delta(s), finish, usage, [DONE] —
        # the standard chunk shapes llama.cpp emits with stream_options.
        cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        def chunk(delta: dict, finish: str | None = None) -> dict:
            return {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }

        events: list[dict] = [chunk({"role": "assistant"})]
        if want_tool_call:
            events.append(chunk({"tool_calls": [{
                "index": 0,
                "id": f"call_fl_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": "glob", "arguments": json.dumps({"pattern": "*"})},
            }]}))
            events.append(chunk({}, finish="tool_calls"))
        else:
            events.append(chunk({"content": f"{FINAL_MARKER}: the stub model "}))
            events.append(chunk({"content": "saw the tool result and is done."}))
            events.append(chunk({}, finish="stop"))
        if (req.get("stream_options") or {}).get("include_usage"):
            events.append({
                "id": cid, "object": "chat.completion.chunk",
                "created": int(time.time()), "model": MODEL_ID, "choices": [],
                "usage": {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18},
            })

        payload = b"".join(
            b"data: " + json.dumps(e).encode() + b"\n\n" for e in events
        ) + b"data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    global MODE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--mode", choices=["normal", "models-500", "no-final"],
                        default="normal")
    args = parser.parse_args()
    MODE = args.mode
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"firstlight stub model listening on 127.0.0.1:{args.port} mode={MODE}",
          flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
