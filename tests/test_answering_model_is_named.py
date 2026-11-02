"""Every chat response names the model that ANSWERED, not the one that was configured.

A client labels each turn with a model. Before this it could only label it with
the configured primary (``/v1`` echoed the requested key; ``/api/chat`` and
``chat_done`` named nothing), so a turn a fallback served was labelled as the
primary's. The loop already knew better: each round records what the server
reported, and the fallback decision names its model.

Pinned through the real ``run_loop`` with a primary that fails terminally and
a fallback that answers, on every surface a client reads:

* ``AssistantTurnComplete.served_model`` and ``RunResult.served_model``;
* ``served_by`` on ``POST /api/chat``, on the bridge's ``chat_done`` frame, and
  in the ``prometheus`` object of ``/v1/chat/completions`` (streamed or not).

``/v1``'s ``model`` stays the key the client asked for: an OpenAI client matches
on it, and requested vs served are kept apart on purpose (``ProviderDegraded``).
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import AsyncIterator

import pytest

from prometheus.engine.agent_loop import AgentLoop, LoopContext, run_loop
from prometheus.engine.fallback import FallbackTarget
from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.engine.stream_events import AssistantTurnComplete
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiStreamEvent,
    ModelProvider,
)
from prometheus.tools.base import ToolRegistry

PRIMARY = "primary-model"
FALLBACK = "fallback-model"
REPLY = "The answer is 4."
TOKEN = "served-by-test-token"


class _Unauthorized(ModelProvider):
    """401 on every call: a terminal failure, so the turn degrades to the fallback."""

    def __init__(self) -> None:
        self._suppress_thinking = True
        self.calls = 0

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        self.calls += 1

        class R:
            status_code = 401

        class AuthError(Exception):
            response = R()

        raise AuthError("expired credential")
        yield  # pragma: no cover - makes this an async generator


class _Answers(ModelProvider):
    """Answers once; ``reports`` is the model name the server puts on its reply, if any."""

    def __init__(self, reports: str | None = None) -> None:
        self._suppress_thinking = True
        self.reports = reports
        self.requests: list[ApiMessageRequest] = []

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        self.requests.append(request)
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[TextBlock(text=REPLY)]),
            usage=UsageSnapshot(input_tokens=10, output_tokens=4),
            stop_reason="stop",
            served_model=self.reports,
        )


def _fallback(provider: ModelProvider) -> FallbackTarget:
    return FallbackTarget(model=FALLBACK, provider_name="llama_cpp", provider=provider, is_local_backend=True)


def _degrading_context(**extra) -> tuple[LoopContext, _Unauthorized, _Answers]:
    primary, local = _Unauthorized(), _Answers()
    context = LoopContext(
        provider=primary, model=PRIMARY, system_prompt="sys", max_tokens=256,
        tool_registry=ToolRegistry(), fallback=_fallback(local), **extra,
    )
    return context, primary, local


def _turns(context: LoopContext) -> list[AssistantTurnComplete]:
    out: list[AssistantTurnComplete] = []

    async def go() -> None:
        async for event, _usage in run_loop(context, [ConversationMessage.from_user_text("2+2?")]):
            if isinstance(event, AssistantTurnComplete):
                out.append(event)

    asyncio.run(go())
    return out


# ── the loop ─────────────────────────────────────────────────────────


def test_a_turn_the_fallback_served_names_the_fallback():
    context, primary, local = _degrading_context()
    turns = _turns(context)
    assert primary.calls == 1 and len(local.requests) == 1, "the fallback served the turn"
    assert turns[-1].served_model == FALLBACK


def test_the_model_the_server_reports_wins_over_the_configured_name():
    """A router or proxy behind one URL answers with whatever it picked; its reply says which."""
    provider = _Answers(reports="what-the-server-ran")
    context = LoopContext(provider=provider, model=PRIMARY, system_prompt="sys", max_tokens=256,
                          tool_registry=ToolRegistry())
    assert _turns(context)[-1].served_model == "what-the-server-ran"


def test_a_server_that_reports_nothing_is_named_by_the_model_asked_of_it():
    context = LoopContext(provider=_Answers(), model=PRIMARY, system_prompt="sys", max_tokens=256,
                          tool_registry=ToolRegistry())
    assert _turns(context)[-1].served_model == PRIMARY


def test_run_async_carries_the_answering_model():
    local = _Answers()
    loop = AgentLoop(provider=_Unauthorized(), model=PRIMARY, max_tokens=256, fallback=_fallback(local))
    result = asyncio.run(loop.run_async(system_prompt="sys", user_message="2+2?"))
    assert REPLY in result.text
    assert result.served_model == FALLBACK


# ── POST /api/chat ───────────────────────────────────────────────────


def test_api_chat_names_the_answering_model(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from prometheus.engine.session import SessionManager
    from prometheus.web.server import create_app

    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path))
    app = create_app(
        {}, session_mgr=SessionManager(),
        agent_loop=AgentLoop(provider=_Unauthorized(), model=PRIMARY, max_tokens=256,
                             fallback=_fallback(_Answers())),
    )
    app.state.ws_bridge = SimpleNamespace(loop_context=SimpleNamespace(system_prompt="sys"))

    response = TestClient(app).post("/api/chat", json={"session_id": "s1", "content": "2+2?"})

    assert response.status_code == 200, response.text
    assert response.json()["served_by"] == FALLBACK


# ── the bridge's chat_done ───────────────────────────────────────────


class _FakeSession:
    def __init__(self) -> None:
        self.messages: list = [ConversationMessage.from_user_text("2+2?")]

    def get_messages(self) -> list:
        return self.messages

    def persist_loop_result(self, original_len: int) -> None:
        return None


class _Recorder:
    def __init__(self) -> None:
        self.frames: list[dict] = []

    async def send(self, raw: str) -> None:
        self.frames.append(json.loads(raw))

    def of(self, kind: str) -> list[dict]:
        return [f["payload"] for f in self.frames if f["type"] == kind]


@pytest.mark.asyncio
async def test_chat_done_names_the_answering_model():
    from prometheus.web.ws_server import WebSocketBridge

    context, _primary, _local = _degrading_context(session_id="beacon:served")
    bridge = WebSocketBridge(loop_context=context)
    rec = _Recorder()
    bridge._clients.add(rec)

    await bridge._run_agent("beacon:served", _FakeSession())

    done = rec.of("chat_done")
    assert done, f"no chat_done: {[f['type'] for f in rec.frames]}"
    assert done[0]["served_by"] == FALLBACK


# ── /v1/chat/completions ─────────────────────────────────────────────


def _v1_client():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from prometheus.web.server import create_app

    app = create_app({"web": {"api_token": TOKEN}})
    context, _primary, _local = _degrading_context()
    app.state.ws_bridge = SimpleNamespace(loop_context=context)
    return TestClient(app)


def test_v1_names_the_answering_model_beside_the_requested_key():
    resp = _v1_client().post(
        "/v1/chat/completions", headers={"Authorization": f"Bearer {TOKEN}"},
        json={"model": "local", "messages": [{"role": "user", "content": "2+2?"}]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model"] == "local", "the requested key is still echoed for OpenAI clients"
    assert body["prometheus"]["served_by"] == FALLBACK


def test_v1_stream_names_the_answering_model_on_the_final_chunk():
    resp = _v1_client().post(
        "/v1/chat/completions", headers={"Authorization": f"Bearer {TOKEN}"},
        json={"messages": [{"role": "user", "content": "2+2?"}], "stream": True},
    )
    assert resp.status_code == 200, resp.text
    chunks = [json.loads(line[6:]) for line in resp.text.splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    final = chunks[-1]
    assert final["choices"][0]["finish_reason"] == "stop"
    assert final["prometheus"]["served_by"] == FALLBACK
