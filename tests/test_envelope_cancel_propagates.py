"""A Stop propagates out of ``LLMCallEnvelope.call`` whatever its ``on_failure``.

``call()`` caught ``CancelledError`` in its ``except BaseException`` block and,
under ``on_failure="return_none"`` (or ``"log_only"``), RETURNED instead of
re-raising. Since Python 3.11 a coroutine that catches the cancellation and
carries on is never cancelled again, so the Stop was simply gone. Seen while
building the Beacon #283 compaction frame: the context compactor's envelope is
``return_none``, so a Stop that landed during a summarizer call came back as
"summarizer returned empty", the compactor failed loud, and the agent loop
went on to the provider call — the turn did not stop.

The policy is for a failed CALL. A cancelled one is not a failure (the row
still says ``partial`` / ``cancelled``, as before); it is the task being told
to stop, and only the task's owner may decide to swallow that.

Canned providers only — no live model.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator

import pytest

from prometheus.context.compactor import ContextCompactor
from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.learning.llm_envelope import LLMCallEnvelope
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiStreamEvent,
    ApiTextDeltaEvent,
    ModelProvider,
)
from prometheus.telemetry.tracker import ToolCallTelemetry
from prometheus.web.ws_server import WebSocketBridge
from tests.support.doubles import register_double


@register_double("cancel_propagates.hanging", replaces="prometheus.providers.base.ModelProvider")
class _HangingProvider(ModelProvider):
    """Streams one delta, then waits forever: the call a user Stops."""

    def __init__(self) -> None:
        self.requests: list[ApiMessageRequest] = []
        self.started = asyncio.Event()

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        self.requests.append(request)
        yield ApiTextDeltaEvent(text="summary so f")
        self.started.set()
        await asyncio.Event().wait()


@register_double("cancel_propagates.raising", replaces="prometheus.providers.base.ModelProvider")
class _RaisingProvider(ModelProvider):
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        yield ApiTextDeltaEvent(text="partial")
        raise self._exc


@register_double("cancel_propagates.model", replaces="prometheus.providers.base.ModelProvider")
class _ModelProvider(ModelProvider):
    """The turn's own model: answers at once, and records that it was asked."""

    def __init__(self) -> None:
        self.requests: list[ApiMessageRequest] = []

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        self.requests.append(request)
        yield ApiTextDeltaEvent(text="answer")
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[TextBlock(text="answer")]),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason="stop",
        )


def _silent(tel: ToolCallTelemetry) -> list[tuple]:
    return tel._conn.execute(
        "SELECT subsystem, operation, exception_type FROM silent_failures"
    ).fetchall()


def _runs(tel: ToolCallTelemetry, subsystem: str) -> list[tuple]:
    return tel._conn.execute(
        "SELECT operation, outcome, summary_json FROM subsystem_runs WHERE subsystem = ?",
        (subsystem,),
    ).fetchall()


async def _until(cond, what: str, timeout: float = 3.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while not cond():
        assert asyncio.get_event_loop().time() < deadline, what
        await asyncio.sleep(0.01)


# --------------------------------------------------------------------------- #
# The envelope
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("on_failure", ["raise", "return_none", "log_only"])
def test_a_call_cancelled_mid_stream_reraises_under_every_policy(tmp_path: Path, on_failure):
    tel = ToolCallTelemetry(db_path=tmp_path / "tel.db")
    env = LLMCallEnvelope(subsystem="context_compactor", telemetry=tel, on_failure=on_failure)
    provider = _HangingProvider()

    async def _main():
        task = asyncio.create_task(env.call(
            provider=provider, model="m", prompt="p", operation="summarize_span"))
        await provider.started.wait()
        task.cancel()  # what the WS interrupt does to the turn awaiting it
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_main())
    # Still recorded, as a Stop: a partial run row, no failure row.
    assert _silent(tel) == []
    runs = _runs(tel, "context_compactor")
    assert [(op, outcome) for op, outcome, _ in runs] == [("summarize_span", "partial")]
    assert json.loads(runs[0][2]) == {"reason": "cancelled", "exception_type": "CancelledError"}


@pytest.mark.parametrize("on_failure", ["return_none", "log_only"])
def test_a_keyboard_interrupt_reraises_too(tmp_path: Path, on_failure):
    tel = ToolCallTelemetry(db_path=tmp_path / "tel.db")
    env = LLMCallEnvelope(subsystem="memory_extractor", telemetry=tel, on_failure=on_failure)
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(env.call(provider=_RaisingProvider(KeyboardInterrupt()),
                             model="m", prompt="p", operation="extract_memory_batch"))
    assert _silent(tel) == []
    assert [outcome for _, outcome, _ in _runs(tel, "memory_extractor")] == ["partial"]


def test_a_real_failure_still_follows_the_policy(tmp_path: Path):
    # The other half of the contract: only a Stop escapes return_none.
    tel = ToolCallTelemetry(db_path=tmp_path / "tel.db")
    env = LLMCallEnvelope(subsystem="memory_extractor", telemetry=tel, on_failure="return_none")
    out = asyncio.run(env.call(provider=_RaisingProvider(RuntimeError("backend fell over")),
                               model="m", prompt="p", operation="extract_memory_batch"))
    assert out is None
    assert _silent(tel) == [("memory_extractor", "extract_memory_batch", "RuntimeError")]
    assert [outcome for _, outcome, _ in _runs(tel, "memory_extractor")] == ["failed"]


# --------------------------------------------------------------------------- #
# A Stop during compaction ends the turn
# --------------------------------------------------------------------------- #


def _history(n_turns: int, filler: int = 400) -> list[ConversationMessage]:
    msgs: list[ConversationMessage] = []
    for i in range(n_turns):
        msgs.append(ConversationMessage.from_user_text(f"U{i:04d} " + "x" * filler))
        msgs.append(ConversationMessage(
            role="assistant", content=[TextBlock(text=f"A{i:04d} " + "y" * filler)]))
    msgs.append(ConversationMessage.from_user_text("and now?"))
    return msgs


def _compactor(summarizer: ModelProvider) -> ContextCompactor:
    # 91 turns of ~800 chars against a 20k window: well over the threshold.
    return ContextCompactor(
        provider=summarizer,
        model="local-27b",
        effective_limit=3000,
        cloud_default_limit=20_000,
        protect_recent_turns=3,
    )


def _context(model: ModelProvider, compactor: ContextCompactor) -> LoopContext:
    return LoopContext(
        provider=model,
        model="cloud-model",
        system_prompt="You are the cancel-propagation fixture agent.",
        max_tokens=512,
        compactor=compactor,
        session_id="desktop:t",
    )


@pytest.mark.asyncio
async def test_the_fixture_compacts_before_the_model_call():
    # Guards the two tests below: compaction really runs a summarizer call on
    # this history, and an uncancelled turn does go on to the model.
    summarizer, model = _ModelProvider(), _ModelProvider()
    async for _event, _usage in run_loop(_context(model, _compactor(summarizer)), _history(90)):
        pass
    assert summarizer.requests, "the history never triggered compaction"
    assert len(model.requests) == 1


@pytest.mark.asyncio
async def test_a_stop_during_a_summarizer_call_ends_the_loop_before_the_model_call():
    summarizer, model = _HangingProvider(), _ModelProvider()
    context = _context(model, _compactor(summarizer))

    async def _turn():
        async for _event, _usage in run_loop(context, _history(90)):
            pass

    task = asyncio.ensure_future(_turn())
    await asyncio.wait_for(summarizer.started.wait(), 3.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert model.requests == [], "the turn went on to the model after a Stop"


class _FakeSession:
    def __init__(self, messages: list) -> None:
        self.messages = messages

    def get_messages(self) -> list:
        return self.messages

    def persist_loop_result(self, original_len: int) -> None:
        pass


class _Recorder:
    def __init__(self) -> None:
        self.frames: list[dict] = []

    async def send(self, raw: str) -> None:
        self.frames.append(json.loads(raw))

    def of(self, kind: str) -> list[dict]:
        return [f for f in self.frames if f["type"] == kind]


@pytest.mark.asyncio
async def test_the_ws_stop_button_during_compaction_ends_the_turn():
    summarizer, model = _HangingProvider(), _ModelProvider()
    bridge = WebSocketBridge(loop_context=_context(model, _compactor(summarizer)))
    ws = _Recorder()
    bridge._clients.add(ws)

    task = asyncio.ensure_future(bridge._run_agent("desktop:t", _FakeSession(_history(90))))
    await asyncio.wait_for(summarizer.started.wait(), 3.0)
    await _until(lambda: "desktop:t" in bridge._turn_tasks, "the turn never registered")

    assert bridge.interrupt_turn("desktop:t") is True
    await asyncio.wait_for(task, 3.0)  # a user Stop returns; it does not raise

    assert model.requests == [], "the turn went on to the model after a Stop"
    [done] = ws.of("chat_done")
    assert done["payload"]["interrupted"] is True
    assert ws.of("chat_delta") == []
