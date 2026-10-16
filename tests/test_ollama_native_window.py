"""Ollama is asked for the window Prometheus budgets (audit 2026-10-16, fix 5b).

The OpenAI-compatible route has no ``options``, so a model served Ollama's own
default window while Prometheus budgeted the trained one, and an over-window
prompt lost its start with no error. On the native ``/api/chat``:

* a request the loop budgeted (``context_window``) sends exactly that as
  ``options.num_ctx``, never above the model's trained length (warned once);
* a request with no budget follows the window the model is loaded at
  (``/api/ps``), so it never forces a reload, and sends none when nothing is
  loaded;
* a prompt that filled the whole window is reported;
* history goes over in the native shape: a call's arguments as an object, a
  tool result naming its call and tool, a picture's base64 in ``images``;
* an ``{"error": ...}`` line inside a 200 stream is an error, not an empty reply.

The far side, through the real daemon: tests/test_local_gate_context_window.py.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from prometheus.engine.messages import (
    ConversationMessage,
    ImageBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from prometheus.providers.base import ApiMessageCompleteEvent, ApiMessageRequest
from prometheus.providers.ollama import OllamaProvider, OllamaStreamError

MODEL = "qwen3.5:9b"


def _done(prompt_tokens: int = 30) -> bytes:
    lines = [{"model": MODEL, "message": {"role": "assistant", "content": "ok"}, "done": False},
             {"model": MODEL, "message": {"role": "assistant", "content": ""}, "done": True,
              "done_reason": "stop", "prompt_eval_count": prompt_tokens, "eval_count": 1}]
    return "".join(json.dumps(x) + "\n" for x in lines).encode()


class _Server:
    def __init__(self, *, trained: int = 262_144, loaded: int | None = None, body: bytes | None = None):
        self.trained, self.loaded, self.body = trained, loaded, body or _done()
        self.chats: list[dict] = []
        self.ps_calls = 0

    def handler(self, req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["completion", "tools"],
                                             "model_info": {"qwen35.context_length": self.trained},
                                             "details": {"family": "qwen35"}})
        if req.url.path == "/api/ps":
            self.ps_calls += 1
            rows = [] if self.loaded is None else [{"name": MODEL, "model": MODEL, "context_length": self.loaded}]
            return httpx.Response(200, json={"models": rows})
        assert req.url.path == "/api/chat", req.url.path
        self.chats.append(json.loads(req.content))
        return httpx.Response(200, content=self.body, headers={"content-type": "application/x-ndjson"})


@pytest.fixture
def serve(monkeypatch):
    def install(**kw) -> _Server:
        server = _Server(**kw)
        real = httpx.AsyncClient

        def client(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(server.handler)
            return real(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", client)
        return server

    return install


def _request(**kw) -> ApiMessageRequest:
    return ApiMessageRequest(model=MODEL, messages=kw.pop("messages", [ConversationMessage.from_user_text("hi")]),
                             max_tokens=64, **kw)


async def _run(provider: OllamaProvider, request: ApiMessageRequest) -> ApiMessageCompleteEvent:
    done = None
    async for ev in provider.stream_message(request):
        if isinstance(ev, ApiMessageCompleteEvent):
            done = ev
    assert done is not None
    return done


@pytest.mark.asyncio
async def test_a_budgeted_request_asks_for_exactly_its_window(serve):
    server = serve(loaded=4096)
    await _run(OllamaProvider(base_url="http://unit.test"), _request(context_window=24_000))
    assert server.chats[0]["options"] == {"num_predict": 64, "num_ctx": 24_000}
    assert server.ps_calls == 0, "a budgeted request does not need to ask what is loaded"


@pytest.mark.asyncio
async def test_a_budget_above_the_trained_length_asks_for_the_trained_length_and_warns_once(serve, caplog):
    server = serve(trained=32_768)
    p = OllamaProvider(base_url="http://unit.test")
    with caplog.at_level(logging.WARNING):
        await _run(p, _request(context_window=72_000))
        await _run(p, _request(context_window=72_000))
    assert [c["options"]["num_ctx"] for c in server.chats] == [32_768, 32_768]
    warned = [r.getMessage() for r in caplog.records if "trained for 32768" in r.getMessage()]
    assert len(warned) == 1 and "72000" in warned[0]


@pytest.mark.asyncio
async def test_an_unbudgeted_request_follows_the_loaded_window(serve):
    server = serve(loaded=8192)
    await _run(OllamaProvider(base_url="http://unit.test"), _request())
    assert server.chats[0]["options"]["num_ctx"] == 8192


@pytest.mark.asyncio
async def test_an_unbudgeted_request_to_an_unloaded_model_sends_no_window(serve):
    server = serve(loaded=None)
    await _run(OllamaProvider(base_url="http://unit.test"), _request())
    assert "num_ctx" not in server.chats[0]["options"]


@pytest.mark.asyncio
async def test_a_prompt_that_filled_the_window_is_reported(serve, caplog):
    serve(body=_done(prompt_tokens=24_000))
    with caplog.at_level(logging.WARNING):
        await _run(OllamaProvider(base_url="http://unit.test"), _request(context_window=24_000))
    assert any("filled the whole 24000-token window" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_history_goes_over_in_the_native_shape(serve):
    server = serve()
    history = [
        ConversationMessage.from_user_text("read a.txt"),
        ConversationMessage(role="assistant", content=[
            TextBlock(text="Reading."),
            ToolUseBlock(id="call_1", name="read_file", input={"path": "a.txt"})]),
        ConversationMessage(role="user", content=[ToolResultBlock(tool_use_id="call_1", content="hello")]),
        ConversationMessage(role="user", content=[TextBlock(text="and this?"),
                                                  ImageBlock(media_type="image/png", data="iVBORw0KGgo=")]),
    ]
    p = OllamaProvider(base_url="http://unit.test")
    p.supports_vision = True
    await _run(p, _request(messages=history))
    sent = server.chats[0]["messages"]
    assert sent[1] == {"role": "assistant", "content": "Reading.", "tool_calls": [
        {"id": "call_1", "function": {"name": "read_file", "arguments": {"path": "a.txt"}}}]}
    assert sent[2] == {"role": "tool", "content": "hello", "tool_call_id": "call_1", "tool_name": "read_file"}
    assert sent[3] == {"role": "user", "content": "and this?", "images": ["iVBORw0KGgo="]}


@pytest.mark.asyncio
async def test_an_error_line_in_the_stream_is_an_error(serve):
    serve(body=b'{"error":"model runner has unexpectedly stopped"}\n')
    with pytest.raises(OllamaStreamError, match="unexpectedly stopped"):
        await _run(OllamaProvider(base_url="http://unit.test"), _request())
