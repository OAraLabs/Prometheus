"""A fake Ollama server with the window behaviour measured on this box (2026-10-16).

Ollama 0.23.0 serving qwen2.5:7b-instruct, asked through its OpenAI-compatible
``/v1/chat/completions`` the way Prometheus asks:

* ``POST /api/show`` reports the TRAINED length (``model_info.*.context_length``,
  32768 for qwen2.5, 262144 for qwen3.5:9b on the 4090's Ollama 0.17.4).
* What a loaded model SERVES is its ``num_ctx``, set when it loads: from the
  request's ``options.num_ctx`` (native ``/api/chat`` only; the OpenAI
  endpoint has no options), else the server default (``OLLAMA_CONTEXT_LENGTH``
  or the VRAM tier: 4k under 24 GiB, 32k for 24-48 GiB, 256k above). ``/api/ps``
  reports it as ``context_length``.
* A prompt over the served window is NOT refused: prompt_tokens comes back as
  exactly the window (36.8k sent, 32768 reported) and the FRONT of the prompt
  is gone (the marker at the top of the system prompt was not recalled), while
  the tool list, which the template renders after the system text, survived.

This fake keeps those rules. ``requests`` records every chat request's
``num_ctx`` (None when absent), the tokens sent and the tokens it kept.
"""

from __future__ import annotations

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from tests.support.scripted_llama import rendered_chars


class FakeOllama:
    def __init__(self, *, model: str = "qwen3.5:9b", trained_ctx: int = 262_144, served_ctx: int = 4096,
                 chars_per_token: float = 3.73) -> None:
        self.model = model
        self.trained_ctx = trained_ctx
        self.served_ctx = served_ctx
        self.chars_per_token = chars_per_token
        self.loaded_ctx: int | None = None
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def __enter__(self) -> "FakeOllama":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *args: object) -> None:
                pass

            def _json(self, code: int, obj: dict[str, Any]) -> None:
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                if self.path.startswith("/api/version"):
                    return self._json(200, {"version": "0.17.4"})
                if self.path.startswith("/api/tags"):
                    return self._json(200, {"models": [{"name": fake.model, "model": fake.model}]})
                if self.path.startswith("/api/ps"):
                    loaded = [] if fake.loaded_ctx is None else [
                        {"name": fake.model, "model": fake.model, "context_length": fake.loaded_ctx}]
                    return self._json(200, {"models": loaded})
                if self.path.rstrip("/").endswith("/models"):
                    return self._json(200, {"object": "list", "data": [{"id": fake.model, "object": "model"}]})
                return self._json(200, {"status": "ok"})

            def do_POST(self) -> None:  # noqa: N802
                n = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(n) or b"{}")
                if self.path.startswith("/api/show"):
                    arch = "qwen35"
                    return self._json(200, {"model_info": {f"{arch}.context_length": fake.trained_ctx},
                                            "parameters": "", "capabilities": ["completion", "tools"]})
                if not (self.path.endswith("/chat/completions") or self.path.startswith("/api/chat")):
                    return self._json(404, {"error": "not found"})
                options = payload.get("options") or {}
                num_ctx = options.get("num_ctx") if isinstance(options, dict) else None
                if num_ctx is None:
                    num_ctx = payload.get("num_ctx")
                if fake.loaded_ctx is None:  # the first request loads the model
                    fake.loaded_ctx = int(num_ctx) if num_ctx else fake.served_ctx
                sent = math.ceil(rendered_chars(payload) / fake.chars_per_token)
                kept = min(sent, fake.loaded_ctx)
                with fake._lock:
                    fake.requests.append({"num_ctx": num_ctx, "sent_tokens": sent, "kept_tokens": kept,
                                          "loaded_ctx": fake.loaded_ctx, "path": self.path,
                                          "max_tokens": payload.get("max_tokens")})
                usage = {"prompt_tokens": kept, "completion_tokens": 3, "total_tokens": kept + 3}
                if not payload.get("stream"):
                    return self._json(200, {"model": fake.model, "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                        "usage": usage})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for chunk in (
                    {"model": fake.model, "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": None}]},
                    {"model": fake.model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                    {"model": fake.model, "choices": [], "usage": usage},
                ):
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")

        return Handler
