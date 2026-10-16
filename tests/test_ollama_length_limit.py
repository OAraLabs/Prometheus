"""Ollama must be SENT the length limit where its endpoint reads one.

The provider used to post to ollama's OpenAI-compatible ``/v1/chat/completions``,
whose request type has a top-level ``max_tokens`` and NO ``options`` field, so
``options.num_predict`` was dropped without an error and every ollama turn
generated without a limit (ollama 0.23.0: ``options: {num_predict: 5}``
returned 150 tokens with ``finish_reason: stop``; ``max_tokens: 5`` returned 5
with ``finish_reason: length``). It now posts to the native ``/api/chat``
(audit 2026-10-16, for ``options.num_ctx``), where the limit is
``options.num_predict`` and a top-level ``max_tokens`` is the field ignored.

Driven through the REAL ``_call_once`` against a mock server that behaves as
the native route does: it honours ``options.num_predict`` and ignores
``max_tokens``.
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
        if req.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        body = json.loads(req.content)
        seen.append(body)
        limit = (body.get("options") or {}).get("num_predict")  # the native route's length field
        words = _WORDS if limit is None else _WORDS[:limit]
        finish = "length" if len(words) < len(_WORDS) else "stop"
        lines = [{"message": {"role": "assistant", "content": w}, "done": False} for w in words]
        lines.append({"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": finish,
                      "prompt_eval_count": 9, "eval_count": len(words)})
        return httpx.Response(200, content="".join(json.dumps(x) + "\n" for x in lines).encode(),
                              headers={"content-type": "application/x-ndjson"})

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
    assert seen[0]["options"]["num_predict"] == 2
    assert "max_tokens" not in seen[0], "the native route has no top-level max_tokens; it would be dropped"
