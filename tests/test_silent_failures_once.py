"""silent_failures: a Stop is not a failure, and each failure is written once.

Live (849 rows, 2026-06-03 to 2026-09-29):

* 13 rows were cancellations: user Stops ("turn interrupted by user") and
  shutdowns. The envelope's ``except BaseException`` wrote a failure row for
  ``CancelledError``; only ``GeneratorExit`` was exempt.
* 92 of the 95 ``web_bridge`` rows had an ``agent_loop`` row for the same
  failure within 2 s: the envelope recorded the provider error and re-raised
  it, and the WS bridge's ``_run_agent`` handler recorded the same exception
  object again.

Telemetry-only: what propagates, and what the user sees, is unchanged.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator

import httpx
import pytest

from prometheus.engine.agent_loop import LoopContext
from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.engine.session import SessionManager
from prometheus.learning.llm_envelope import LLMCallEnvelope
from prometheus.providers.base import (
    ApiMessageRequest,
    ApiStreamEvent,
    ApiTextDeltaEvent,
    ModelProvider,
)
from prometheus.telemetry import tracker
from prometheus.telemetry.tracker import ToolCallTelemetry
from tests.support.doubles import register_double


class _RaisingProvider(ModelProvider):
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        yield ApiTextDeltaEvent(text="partial")
        raise self._exc


class _HangingProvider(ModelProvider):
    """Streams one delta, then waits forever: the round a user Stops."""

    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]:
        yield ApiTextDeltaEvent(text="thinking…")
        self.started.set()
        await asyncio.Event().wait()


def _request() -> ApiMessageRequest:
    return ApiMessageRequest(
        model="fixture-model",
        messages=[ConversationMessage(role="user", content=[TextBlock(text="q")])],
        max_tokens=64,
    )


def _silent(tel: ToolCallTelemetry) -> list[tuple]:
    return tel._conn.execute(
        "SELECT subsystem, operation, exception_type FROM silent_failures ORDER BY timestamp"
    ).fetchall()


def _runs(tel: ToolCallTelemetry, subsystem: str = "agent_loop") -> list[tuple]:
    return tel._conn.execute(
        "SELECT operation, outcome, summary_json FROM subsystem_runs"
        " WHERE subsystem = ? AND operation != 'tool_advertisement'",
        (subsystem,),
    ).fetchall()


def _http_400() -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://backend:8080/v1/chat/completions")
    response = httpx.Response(400, request=request, json={"error": {"message": "bad"}})
    return httpx.HTTPStatusError("Client error '400 Bad Request'", request=request,
                                 response=response)


# --------------------------------------------------------------------------- #
# A Stop is not a failure
# --------------------------------------------------------------------------- #


class TestCancellationIsNotAFailure:
    def test_a_user_stop_mid_stream_writes_no_failure_row(self, tmp_path: Path):
        tel = ToolCallTelemetry(db_path=tmp_path / "tel.db")
        env = LLMCallEnvelope(subsystem="agent_loop", telemetry=tel)
        provider = _HangingProvider()

        async def _turn():
            async for _ in env.stream(provider=provider, request=_request(),
                                      operation="loop_round", round_index=0,
                                      session_id="beacon:s1"):
                pass

        async def _main():
            task = asyncio.create_task(_turn())
            await provider.started.wait()
            task.cancel()  # what the WS interrupt does to a running turn
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(_main())
        assert _silent(tel) == []
        runs = _runs(tel)
        assert [(op, outcome) for op, outcome, _ in runs] == [("loop_round", "partial")]
        assert json.loads(runs[0][2]) == {"reason": "cancelled",
                                          "exception_type": "CancelledError"}

    def test_a_keyboard_interrupt_is_a_stop_too(self, tmp_path: Path):
        tel = ToolCallTelemetry(db_path=tmp_path / "tel.db")
        env = LLMCallEnvelope(subsystem="agent_loop", telemetry=tel)

        async def _turn():
            async for _ in env.stream(provider=_RaisingProvider(KeyboardInterrupt()),
                                      request=_request()):
                pass

        with pytest.raises(KeyboardInterrupt):
            asyncio.run(_turn())
        assert _silent(tel) == []
        assert [outcome for _, outcome, _ in _runs(tel)] == ["partial"]

    def test_a_cancelled_call_writes_no_failure_row(self, tmp_path: Path):
        """The non-streaming path (memory extractor, curator, ...) follows the same rule."""
        tel = ToolCallTelemetry(db_path=tmp_path / "tel.db")
        env = LLMCallEnvelope(subsystem="memory_extractor", telemetry=tel, on_failure="raise")
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(env.call(provider=_RaisingProvider(asyncio.CancelledError()),
                                 model="m", prompt="p", operation="extract_memory_batch"))
        assert _silent(tel) == []
        assert [outcome for _, outcome, _ in _runs(tel, "memory_extractor")] == ["partial"]

    def test_a_real_provider_error_is_still_a_failure(self, tmp_path: Path):
        tel = ToolCallTelemetry(db_path=tmp_path / "tel.db")
        env = LLMCallEnvelope(subsystem="agent_loop", telemetry=tel)
        exc = _http_400()

        async def _turn():
            async for _ in env.stream(provider=_RaisingProvider(exc), request=_request()):
                pass

        with pytest.raises(httpx.HTTPStatusError) as caught:
            asyncio.run(_turn())
        assert caught.value is exc, "propagates unchanged, by identity"
        assert _silent(tel) == [("agent_loop", "loop_round", "HTTPStatusError")]
        assert [outcome for _, outcome, _ in _runs(tel)] == ["failed"]
        from prometheus.telemetry.tracker import silent_failure_recorded

        assert silent_failure_recorded(exc)


# --------------------------------------------------------------------------- #
# Each failure once
# --------------------------------------------------------------------------- #


class TestRecordedMark:
    def test_the_mark_is_set_only_by_a_written_row(self, tmp_path: Path):
        from prometheus.telemetry.tracker import silent_failure_recorded

        tel = ToolCallTelemetry(db_path=tmp_path / "tel.db")
        exc = RuntimeError("boom")
        assert not silent_failure_recorded(exc)
        tel.record_silent_failure("curator", "run_once", exc)
        assert silent_failure_recorded(exc)

    def test_a_refused_write_leaves_no_mark(self, tmp_path: Path):
        from prometheus.telemetry.tracker import silent_failure_recorded

        tel = ToolCallTelemetry(db_path=tmp_path / "tel.db")
        tel._conn.execute("DROP TABLE silent_failures")
        exc = RuntimeError("boom")
        tel.record_silent_failure("curator", "run_once", exc)  # never raises
        assert not silent_failure_recorded(exc)


def _bridge(loop_context) -> object:
    from prometheus.web.ws_server import WebSocketBridge

    bridge = WebSocketBridge(session_mgr=SessionManager(), loop_context=loop_context,
                             agent_state_ref={"state": "thinking"})

    async def _noop(*a, **k):
        return None

    bridge.broadcast = _noop
    return bridge


@register_double(
    "silent_failures_once.loop_raises_outside_the_envelope",
    replaces="prometheus.engine.agent_loop.run_loop",
)
class _LoopRaisesOutsideTheEnvelope:
    """The loop itself fails (no provider call): only the bridge can record it."""

    def __call__(self, ctx, messages, *, mode="agent", session_id=None, tool_choice=None,
                 surface=None):
        async def gen():
            raise RuntimeError("Exceeded maximum turn limit (200)")
            yield  # pragma: no cover

        return gen()


class TestWebBridgeRecordsOnce:
    def test_a_provider_error_through_the_real_loop_is_one_row(self, tmp_path, monkeypatch):
        tel = ToolCallTelemetry(db_path=tmp_path / "tel.db")
        monkeypatch.setattr(tracker, "_telemetry_singleton", tel)
        ctx = LoopContext(provider=_RaisingProvider(_http_400()), model="fixture-model",
                          system_prompt="sys", max_tokens=64, telemetry=tel,
                          session_id="beacon:s1")
        bridge = _bridge(ctx)
        session = bridge.session_mgr.get_or_create("beacon:s1")
        session.add_user_message("hello")

        asyncio.run(bridge._run_agent("beacon:s1", session))

        assert _silent(tel) == [("agent_loop", "loop_round", "HTTPStatusError")], \
            "one failure, one row: the bridge must not write the envelope's failure again"

    def test_a_failure_the_envelope_never_saw_is_still_recorded(self, tmp_path, monkeypatch):
        import prometheus.engine.agent_loop as al

        tel = ToolCallTelemetry(db_path=tmp_path / "tel.db")
        monkeypatch.setattr(tracker, "_telemetry_singleton", tel)
        monkeypatch.setattr(al, "run_loop", _LoopRaisesOutsideTheEnvelope())
        bridge = _bridge(object())
        session = bridge.session_mgr.get_or_create("beacon:s2")
        session.add_user_message("hello")

        asyncio.run(bridge._run_agent("beacon:s2", session))

        assert _silent(tel) == [("web_bridge", "_run_agent", "RuntimeError")]
