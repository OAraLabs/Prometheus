"""Logging reverse proxy between a scratch daemon and a llama.cpp server.

Forwards every request unchanged and streams the reply back. For each
``POST .../chat/completions`` it appends one JSONL record: the request body as
sent and a digest of the reply (finish reason, usage, server timings, tool
calls, content). Everything else (``/props``, ``/tokenize``, ``/health``) is
forwarded without a record.

The upstream address is read in-process from a Prometheus config
(``model.base_url``) so it never lands in a shell history, a log or this
record. Only scheme, host and port are used; the record never contains them.

Usage: python proxy.py <listen_port> <capture.jsonl> <config_with_upstream.yaml>
"""

from __future__ import annotations

import json
import sys
import time
from urllib.parse import urlparse

import httpx
import uvicorn
import yaml
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

PORT = int(sys.argv[1])
LOG = sys.argv[2]
with open(sys.argv[3], encoding="utf-8") as _fh:
    _url = urlparse((yaml.safe_load(_fh) or {})["model"]["base_url"])
UPSTREAM = f"{_url.scheme}://{_url.netloc}"
del _url

_seq = [0]
_client = httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=10.0))
_HOP = {"host", "content-length", "transfer-encoding", "content-encoding", "connection"}


def _write(rec: dict) -> None:
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def _digest_sse(raw: bytes) -> dict:
    """Fold an OpenAI-shape SSE stream into one summary."""
    content: list[str] = []
    reasoning_chars = 0
    calls: dict[int, dict] = {}
    finish = usage = timings = None
    for line in raw.decode("utf-8", "replace").splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        usage = chunk.get("usage") or usage
        timings = chunk.get("timings") or timings
        for choice in chunk.get("choices") or []:
            finish = choice.get("finish_reason") or finish
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            reasoning_chars += len(delta.get("reasoning_content") or "")
            for tc in delta.get("tool_calls") or []:
                slot = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                fn = tc.get("function") or {}
                slot["name"] += fn.get("name") or ""
                slot["arguments"] += fn.get("arguments") or ""
    return {
        "finish_reason": finish,
        "usage": usage,
        "timings": timings,
        "content": "".join(content),
        "reasoning_chars": reasoning_chars,
        "tool_calls": [calls[i] for i in sorted(calls)],
    }


async def proxy(request: Request) -> Response:
    path = request.url.path
    url = UPSTREAM + path + (f"?{request.url.query}" if request.url.query else "")
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP}
    if not (request.method == "POST" and path.endswith("/chat/completions")):
        upstream = await _client.request(request.method, url, content=body, headers=headers)
        return Response(
            upstream.content,
            status_code=upstream.status_code,
            headers={k: v for k, v in upstream.headers.items() if k.lower() not in _HOP},
        )
    _seq[0] += 1
    seq = _seq[0]
    try:
        req_json = json.loads(body)
    except ValueError:
        req_json = {"_unparsed_chars": len(body)}
    started = time.time()
    upstream = await _client.send(
        _client.build_request("POST", url, content=body, headers=headers), stream=True
    )
    buf = bytearray()

    async def relay():
        try:
            async for chunk in upstream.aiter_raw():
                buf.extend(chunk)
                yield chunk
        finally:
            await upstream.aclose()
            if req_json.get("stream"):
                digest = _digest_sse(bytes(buf))
            else:
                digest = {"raw_chars": len(buf)}
            _write({
                "seq": seq, "t": started, "dt": round(time.time() - started, 2),
                "status": upstream.status_code, "request": req_json, "response": digest,
            })

    return StreamingResponse(
        relay(),
        status_code=upstream.status_code,
        headers={k: v for k, v in upstream.headers.items() if k.lower() not in _HOP},
    )


app = Starlette(routes=[Route("/{path:path}", proxy, methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"])])

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
