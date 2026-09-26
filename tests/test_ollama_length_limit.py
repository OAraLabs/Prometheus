"""Ollama must be SENT the length limit where its endpoint reads one.

The provider posts to ollama's OpenAI-compatible ``/v1/chat/completions``,
whose request type has a top-level ``max_tokens`` (mapped server-side to
``num_predict``) and NO ``options`` field — so ``options.num_predict`` was
dropped without an error and every ollama turn generated without a limit.
Measured on ollama 0.23.0 with one prompt: ``options: {num_predict: 5}``
returned 150 tokens with ``finish_reason: stop``; ``max_tokens: 5`` returned
5 with ``finish_reason: length``.

Driven through the REAL ``_call_once`` against a mock server that behaves as
that measurement did: it honours ``max_tokens`` and ignores ``options``.
"""

from __future__ import annotations

import json

import httpx
import pytest

from prometheus.engine.messages import ConversationMessage
from prometheus.providers.base import ApiMessageCompleteEvent, ApiMessageRequest

_WORDS = ["one", " two", " three", " four", " five", " six"]


def _mock_ollama(monkeypatch) -> list[dict]:
    """Route httpx.AsyncClient through a mock ollama; returns the POSTed bodies."""
    seen: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/show":
            # The capability question (WP-X.35): unknown here, so the chat
            # request is exactly the one this test was written against.
            return httpx.Response(404, json={"error": "not found"})
        body = json.loads(req.content)
        seen.append(body)
        limit = body.get("max_tokens")  # the only length field the endpoint reads
        words = _WORDS if limit is None else _WORDS[:limit]
        finish = "length" if len(words) < len(_WORDS) else "stop"
        chunks = [{"choices": [{"index": 0, "delta": {"content": w}, "finish_reason": None}]}
                  for w in words]
        chunks.append({"choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": finish}]})
        sse = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, content=sse.encode(),
                              headers={"content-type": "text/event-stream"})

    real = httpx.AsyncClient

    def client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    return seen


@pytest.mark.asyncio
async def test_ollama_length_limit_reaches_the_server(monkeypatch):
    from prometheus.providers.ollama import OllamaProvider

    seen = _mock_ollama(monkeypatch)
    request = ApiMessageRequest(
        model="m", messages=[ConversationMessage.from_user_text("count")], max_tokens=2,
    )
    done = None
    async for ev in OllamaProvider(base_url="http://unit.test:1").stream_message(request):
        if isinstance(ev, ApiMessageCompleteEvent):
            done = ev
    assert done is not None, "stream ended without a completion event"

    text = "".join(b.text for b in done.message.content if hasattr(b, "text"))
    assert (text, done.stop_reason) == ("one two", "length"), (
        f"the length limit never reached ollama: got {text!r}, stop={done.stop_reason!r}"
    )
    assert seen[0]["max_tokens"] == 2
    assert "options" not in seen[0], (
        "ollama's OpenAI-compatible endpoint has no `options` field; anything sent there is dropped"
    )
