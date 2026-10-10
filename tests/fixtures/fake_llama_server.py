"""A fake llama-server (or OpenAI-compatible cloud endpoint) for worker tests.

Serves the endpoints the providers use — ``/props``, ``/v1/models`` and
streamed ``/v1/chat/completions`` (also ``/chat/completions`` for a base URL
that already ends in ``/v1``) — on a loopback port, in a thread of the test
process, so a spawned ``python -m prometheus`` child can reach it too.

Prompt tokens are estimated as chars/4, the heuristic Prometheus's own
estimator uses, so an overflow here is never an estimator disagreement. A
prompt over ``n_ctx`` gets llama.cpp's real ``exceed_context_size_error`` 400.

The scripted "model":

* a conversation whose latest user text contains ``SPAWN_WORKER`` is a
  PARENT: it calls ``parent_tool`` once with ``parent_args``, then replies;
* any other conversation is a WORKER: it calls ``write_file`` with a ~9 KB
  body ``writes_per_turn`` times (an absolute path under ``work_dir``), then
  replies;
* a request with no tools (a summarizer call) gets a one-line summary.

``rows`` holds one dict of METADATA per chat request: the model asked for,
whether an Authorization header arrived (presence only), whether it belongs
to a worker, the user-text turn count, the token estimate and the outcome.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

WORKER_MARK = "You are worker"
SPAWN_MARK = "SPAWN_WORKER"
BODY = ("def step():\n    return 'filler line for the worker tests'\n" * 170)[:9000]


def _est_tokens(payload: dict[str, Any]) -> int:
    chars = 0
    for m in payload.get("messages", []):
        content = m.get("content")
        if isinstance(content, str):
            chars += len(content)
        elif isinstance(content, list):
            chars += sum(len(p.get("text", "")) for p in content if isinstance(p, dict))
        for tc in m.get("tool_calls") or []:
            chars += len(tc.get("function", {}).get("arguments", ""))
    chars += len(json.dumps(payload.get("tools") or []))
    return chars // 4


class FakeLlamaServer:
    def __init__(
        self,
        *,
        n_ctx: int = 32768,
        model_id: str = "fake-local-27b",
        writes_per_turn: int = 4,
        work_dir: Path | str = "/nonexistent",
        parent_tool: str = "task_create",
        parent_args: dict[str, Any] | None = None,
    ) -> None:
        self.n_ctx = n_ctx
        self.model_id = model_id
        self.writes_per_turn = writes_per_turn
        self.work_dir = str(work_dir)
        self.parent_tool = parent_tool
        self.parent_args = parent_args or {}
        self.rows: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def __enter__(self) -> "FakeLlamaServer":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    def chat_rows(self, *, worker: bool | None = None) -> list[dict[str, Any]]:
        with self._lock:
            rows = list(self.rows)
        return rows if worker is None else [r for r in rows if r["is_worker"] is worker]

    # -- the handler -------------------------------------------------------

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *args: object) -> None:  # quiet
                pass

            def _json(self, code: int, obj: dict[str, Any]) -> None:
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                if self.path.startswith("/props"):
                    return self._json(200, {
                        "default_generation_settings": {"n_ctx": fake.n_ctx},
                        "modalities": {"vision": False},
                        "model_path": f"/models/{fake.model_id}.gguf",
                        "chat_template": "{{ messages }}",
                        "chat_template_caps": {"supports_tools": True,
                                               "supports_tool_calls": True},
                    })
                if self.path.endswith("/models"):
                    return self._json(200, {"object": "list",
                                            "data": [{"id": fake.model_id}]})
                return self._json(200, {"status": "ok"})

            def do_POST(self) -> None:  # noqa: N802
                n = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(n) or b"{}")
                msgs = payload.get("messages", [])
                tokens = _est_tokens(payload)
                has_tools = bool(payload.get("tools"))
                user_texts = [m for m in msgs if m.get("role") == "user"
                              and isinstance(m.get("content"), str)]
                writes = 0
                for m in reversed(msgs):
                    if m.get("role") == "user" and isinstance(m.get("content"), str):
                        break
                    if m.get("role") == "assistant" and m.get("tool_calls"):
                        writes += 1
                row: dict[str, Any] = {
                    "model": payload.get("model"),
                    "auth_header_present": bool(self.headers.get("Authorization")),
                    "is_worker": bool(user_texts) and str(
                        user_texts[0].get("content", "")).startswith(WORKER_MARK),
                    "user_text_msgs": len(user_texts),
                    "est_prompt_tokens": tokens,
                    "has_tools": has_tools,
                }
                if tokens > fake.n_ctx:
                    row["outcome"] = "400"
                    fake._record(row)
                    return self._json(400, {"error": {
                        "code": 400,
                        "message": f"request ({tokens} tokens) exceeds the available "
                                   f"context size ({fake.n_ctx} tokens), try increasing it",
                        "type": "exceed_context_size_error",
                        "n_prompt_tokens": tokens, "n_ctx": fake.n_ctx}})

                last_user = str(user_texts[-1].get("content", "")) if user_texts else ""
                if not has_tools:
                    row["outcome"] = "summary"
                    delta: dict[str, Any] = {"content": "Summary: step files were written."}
                    finish = "stop"
                elif SPAWN_MARK in last_user:
                    if writes == 0 and not any(m.get("role") == "tool" for m in msgs):
                        row["outcome"] = f"parent {fake.parent_tool}"
                        delta = {"tool_calls": [{
                            "index": 0, "id": "call_spawn", "type": "function",
                            "function": {"name": fake.parent_tool,
                                         "arguments": json.dumps(fake.parent_args)}}]}
                        finish = "tool_calls"
                    else:
                        row["outcome"] = "parent reply"
                        delta = {"content": "Worker started."}
                        finish = "stop"
                elif writes < fake.writes_per_turn:
                    k = sum(1 for m in msgs if m.get("role") == "assistant"
                            and m.get("tool_calls"))
                    row["outcome"] = f"write_file {k}"
                    args = json.dumps({"path": f"{fake.work_dir}/step_{k}.py",
                                       "content": BODY})
                    delta = {"tool_calls": [{
                        "index": 0, "id": f"call_{k}", "type": "function",
                        "function": {"name": "write_file", "arguments": args}}]}
                    finish = "tool_calls"
                else:
                    row["outcome"] = "reply"
                    delta = {"content": "Done with this step."}
                    finish = "stop"
                fake._record(row)

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for chunk in (
                    {"model": fake.model_id,
                     "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                    {"model": fake.model_id,
                     "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
                    {"model": fake.model_id, "choices": [],
                     "usage": {"prompt_tokens": tokens, "completion_tokens": 50}},
                ):
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")

        return Handler

    def _record(self, row: dict[str, Any]) -> None:
        with self._lock:
            self.rows.append(row)
