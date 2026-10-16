"""F1 follow-through: streamed completions must REQUEST usage accounting.

The live proof after PR #31 landed showed every agent_loop usage row with
input_tokens=0/output_tokens=0 — the providers parse the OpenAI-compat
``usage`` chunk, but llama.cpp (and OpenAI-compat servers generally) only
SEND it when the request carries ``stream_options: {"include_usage": true}``.
Accounting-only: generation, sampling, prompt, and schemas are unaffected.
"""

from __future__ import annotations

from prometheus.engine.messages import ConversationMessage
from prometheus.providers.base import ApiMessageRequest


def _request() -> ApiMessageRequest:
    return ApiMessageRequest(
        model="m",
        messages=[ConversationMessage.from_user_text("q")],
        max_tokens=16,
    )


def test_llama_cpp_payload_requests_usage():
    from prometheus.providers.llama_cpp import LlamaCppProvider

    provider = LlamaCppProvider(base_url="http://unit.test:1")
    payload = provider._build_request_payload(_request())
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}


def test_payload_builders_share_the_request(monkeypatch):
    # openai_compat and stub build their payload inline; assert by
    # source-level contract: the dict literal carries stream_options
    # adjacent to stream. (A capture-the-httpx-body harness would be
    # heavier than the one-line invariant deserves.)
    import inspect

    from prometheus.providers import openai_compat, stub

    for mod in (openai_compat, stub):
        src = inspect.getsource(mod)
        assert '"stream_options": {"include_usage": True}' in src, (
            f"{mod.__name__} streams without requesting usage accounting"
        )


# ---------------------------------------------------------------------------
# Ollama — on the native /api/chat route (audit 2026-10-16) the final line
# always carries prompt_eval_count / eval_count: nothing to ask for. Driven
# through its REAL _call_once against a mock server that answers as 0.23.0
# does; the "older server" mode sends a final line with no counts.
# ---------------------------------------------------------------------------

import json  # noqa: E402

import httpx  # noqa: E402
import pytest  # noqa: E402


def _ollama_lines(counts: bool) -> bytes:
    final = {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"}
    if counts:
        final.update({"prompt_eval_count": 31, "eval_count": 2})
    lines = [{"message": {"role": "assistant", "content": "hi"}, "done": False}, final]
    return "".join(json.dumps(x) + "\n" for x in lines).encode()


def _mock_ollama(monkeypatch, *, counts: bool) -> list[dict]:
    """Route every httpx.AsyncClient through a mock ollama; returns the list
    the POSTed bodies are captured into."""
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
        return httpx.Response(200, content=_ollama_lines(counts),
                              headers={"content-type": "application/x-ndjson"})

    real = httpx.AsyncClient

    def client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    return seen


async def _complete(provider):
    from prometheus.providers.base import ApiMessageCompleteEvent

    done = None
    async for ev in provider.stream_message(_request()):
        if isinstance(ev, ApiMessageCompleteEvent):
            done = ev
    assert done is not None, "stream ended without a completion event"
    return done


@pytest.mark.asyncio
async def test_ollama_records_the_counts_its_final_line_carries(monkeypatch):
    from prometheus.providers.ollama import OllamaProvider

    seen = _mock_ollama(monkeypatch, counts=True)
    done = await _complete(OllamaProvider(base_url="http://unit.test:1"))
    assert "stream_options" not in seen[0]           # a /v1 field; the native route always counts
    assert (done.usage.input_tokens, done.usage.output_tokens) == (31, 2)


@pytest.mark.asyncio
async def test_ollama_server_without_counts_still_streams(monkeypatch):
    """A final line without counts: the turn must complete, with usage left at
    0/0 (not an error)."""
    from prometheus.providers.ollama import OllamaProvider

    _mock_ollama(monkeypatch, counts=False)
    done = await _complete(OllamaProvider(base_url="http://unit.test:1"))
    assert "".join(b.text for b in done.message.content if hasattr(b, "text")) == "hi"
    assert (done.usage.input_tokens, done.usage.output_tokens) == (0, 0)
