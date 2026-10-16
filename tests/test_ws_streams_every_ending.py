"""Every ending reaches the WebSocket client (audit 2026-10-16, gate item d).

The loop writes some endings itself (the repeat guard, the iteration cap, an
empty reply twice, the context pre-flight, a context overflow, the circuit
breaker, a boundary escape, the divergence halt) and yields them as an
``AssistantTurnComplete`` with no text deltas. The bridge forwarded only
deltas, so Beacon showed an empty bubble while the text sat in history.

Pinned here, against the frames the bridge broadcasts:

* a round whose text never streamed sends it, whole, before ``chat_done``;
* a round that streamed its text is not sent twice;
* a round that streamed the start of its text sends the rest;
* a final text that was rewritten (not an extension of what streamed) sends
  nothing more;
* ``chat_done`` names the loop's stop reason, and nothing when the model chose
  to stop.

The far side (the real repeat guard, through the real daemon) is
tests/test_local_gate_integration.py's gate item d.
"""

from __future__ import annotations

import json

import pytest

from prometheus.engine.agent_loop import _make_assistant_msg
from prometheus.engine.stream_events import (
    AssistantTextDelta,
    AssistantTurnComplete,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)
from prometheus.engine.usage import UsageSnapshot
from prometheus.web.ws_server import WebSocketBridge
from tests.support.doubles import register_double

HALT = "Halted: no progress. `tool_search` was called 3 times with identical arguments."


class _FakeSession:
    def __init__(self) -> None:
        self.messages: list = []

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

    def deltas(self) -> list[str]:
        return [p["content"] for p in self.of("chat_delta")]


def _complete(text: str, stop_reason: str | None = None) -> AssistantTurnComplete:
    return AssistantTurnComplete(message=_make_assistant_msg(text), usage=UsageSnapshot(), stop_reason=stop_reason)


async def _run(monkeypatch, events: list) -> _Recorder:
    import prometheus.engine.agent_loop as agent_loop_mod

    @register_double("ws_endings.run_loop", replaces="prometheus.engine.agent_loop.run_loop")
    async def run_loop(context, messages, mode="agent", session_id=None, tool_choice=None, surface=None):
        for ev in events:
            yield ev, None

    monkeypatch.setattr(agent_loop_mod, "run_loop", run_loop)
    bridge = WebSocketBridge(loop_context=object())
    rec = _Recorder()
    bridge._clients.add(rec)
    await bridge._run_agent("beacon:endings", _FakeSession())
    return rec


@pytest.mark.asyncio
async def test_an_ending_that_never_streamed_is_sent_before_chat_done(monkeypatch):
    rec = await _run(monkeypatch, [_complete(HALT, "unproductive_repeat")])
    assert rec.deltas() == [HALT]
    kinds = [f["type"] for f in rec.frames]
    assert kinds.index("chat_delta") < kinds.index("chat_done")
    assert rec.of("chat_done")[0]["stop_reason"] == "unproductive_repeat"


@pytest.mark.asyncio
async def test_a_streamed_round_is_not_sent_twice(monkeypatch):
    rec = await _run(monkeypatch, [AssistantTextDelta(text="Hello there."), _complete("Hello there.")])
    assert rec.deltas() == ["Hello there."]
    assert "stop_reason" not in rec.of("chat_done")[0]


@pytest.mark.asyncio
async def test_a_partly_streamed_round_sends_the_rest(monkeypatch):
    rec = await _run(monkeypatch, [AssistantTextDelta(text="Hello"), _complete("Hello there.")])
    assert rec.deltas() == ["Hello", " there."]


@pytest.mark.asyncio
async def test_a_rewritten_final_text_sends_nothing_more(monkeypatch):
    # The stream showed the words; the final text dropped a markup span.
    rec = await _run(monkeypatch, [AssistantTextDelta(text="Done. <function=x>junk"), _complete("Done.")])
    assert rec.deltas() == ["Done. <function=x>junk"]


@pytest.mark.asyncio
async def test_a_halt_after_a_tool_round_is_sent_once(monkeypatch):
    rec = await _run(monkeypatch, [
        AssistantTextDelta(text="Let me look."),
        _complete("Let me look."),
        ToolExecutionStarted(tool_name="glob", tool_input={"pattern": "*"}, tool_use_id="t1"),
        ToolExecutionCompleted(tool_name="glob", output="(no matches)", is_error=False, tool_use_id="t1"),
        _complete(HALT, "unproductive_repeat"),
    ])
    assert rec.deltas() == ["Let me look.", HALT]
    assert rec.of("chat_done")[0]["stop_reason"] == "unproductive_repeat"
