"""SPRINT-loop-envelope (F1) — the agent loop runs inside LLMCallEnvelope.

Bakeoff finding F1: the loop's model calls bypassed the envelope entirely;
UsageSnapshot was produced each round and never persisted (ws_server discards
it, AgentLoop keeps only the last one in memory). The wrap must be
behavior-preserving — identical request payloads, identical streaming
behavior, identical error propagation — while making "prometheus tokens =
null" structurally impossible.

The payload-equivalence fixture (tests/fixtures/loop_envelope_prewrap_request
.json) was recorded by scripts/record_loop_envelope_fixture.py at pre-wrap
commit 1367334 (run_loop calling provider.stream_message directly). The
scenario is deliberately duplicated here, self-contained: if the two copies
drift, the equality assertion fails loudly.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator

import pytest
from pydantic import BaseModel

from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolUseBlock,
)
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
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult

FIXTURE = Path(__file__).parent / "fixtures" / "loop_envelope_prewrap_request.json"


# ---------------------------------------------------------------------------
# Scenario building blocks (mirrors scripts/record_loop_envelope_fixture.py)
# ---------------------------------------------------------------------------


def _assistant_text(text: str) -> ConversationMessage:
    return ConversationMessage(role="assistant", content=[TextBlock(text=text)])


def _complete(
    message: ConversationMessage,
    *,
    input_tokens: int = 123,
    output_tokens: int = 7,
    stop_reason: str | None = "stop",
) -> ApiMessageCompleteEvent:
    return ApiMessageCompleteEvent(
        message=message,
        usage=UsageSnapshot(
            input_tokens=input_tokens, output_tokens=output_tokens
        ),
        stop_reason=stop_reason,
    )


class ScriptedProvider(ModelProvider):
    """Captures every request; plays one scripted event list per call."""

    def __init__(self, turns: list[list[ApiStreamEvent]]) -> None:
        self.requests: list[ApiMessageRequest] = []
        self._turns = turns

    async def stream_message(
        self, request: ApiMessageRequest
    ) -> AsyncIterator[ApiStreamEvent]:
        self.requests.append(request)
        for event in self._turns[len(self.requests) - 1]:
            yield event


class RaisingProvider(ModelProvider):
    """Raises mid-stream after one delta."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    async def stream_message(
        self, request: ApiMessageRequest
    ) -> AsyncIterator[ApiStreamEvent]:
        yield ApiTextDeltaEvent(text="partial")
        raise self._exc


class _EchoArgs(BaseModel):
    text: str = ""


class EchoTool(BaseTool):
    name = "echo"
    description = "Echo the input back."
    input_model = _EchoArgs

    async def execute(self, arguments: _EchoArgs, context) -> ToolResult:
        return ToolResult(output=f"echo: {arguments.text}")


def _context(
    provider: ModelProvider,
    tel: ToolCallTelemetry | None,
    *,
    registry: ToolRegistry | None = None,
    session_id: str | None = "sess-fixture",
) -> LoopContext:
    return LoopContext(
        provider=provider,
        model="fixture-model",
        system_prompt="You are the loop-envelope fixture agent.",
        max_tokens=512,
        tool_registry=registry,
        telemetry=tel,
        session_id=session_id,
    )


def _user(text: str) -> list[ConversationMessage]:
    return [ConversationMessage.from_user_text(text)]


def _canonical(request: ApiMessageRequest) -> str:
    # Same serialization as scripts/record_loop_envelope_fixture.py.
    return json.dumps(
        {
            "model": request.model,
            "system_prompt": request.system_prompt,
            "max_tokens": request.max_tokens,
            "tools": request.tools,
            "suppress_thinking": request.suppress_thinking,
            "messages": [m.model_dump(mode="json") for m in request.messages],
        },
        sort_keys=True,
        indent=1,
    )


async def _drain(context: LoopContext, messages: list[ConversationMessage]):
    async for _event, _usage in run_loop(context, messages):
        pass


def _usage_rows(tel: ToolCallTelemetry) -> list[tuple]:
    return tel._conn.execute(
        "SELECT operation, outcome, input_tokens, output_tokens, round_index,"
        " session_id, model, thinking, summary_json"
        " FROM subsystem_runs WHERE subsystem = 'agent_loop'"
        " ORDER BY round_index"
    ).fetchall()


# ---------------------------------------------------------------------------
# Payload equivalence — the recorded-fixture diff
# ---------------------------------------------------------------------------


class TestPayloadEquivalence:

    def test_request_identical_to_prewrap_fixture(self, tmp_path: Path):
        """The wrapped loop sends byte-identical requests to the pre-wrap loop."""
        fixture = json.loads(FIXTURE.read_text())
        provider = ScriptedProvider([[
            ApiTextDeltaEvent(text="OK"),
            _complete(_assistant_text("OK")),
        ]])
        tel = ToolCallTelemetry(db_path=tmp_path / "tel.db")
        asyncio.run(_drain(
            _context(provider, tel),
            _user("fixture turn — reply without tools"),
        ))
        assert len(provider.requests) == 1
        assert _canonical(provider.requests[0]) == fixture["request_canonical"], (
            "post-wrap request differs from the pre-wrap recording — "
            "the envelope wrap changed the payload"
        )

    def test_fixture_was_recorded_prewrap(self):
        # Provenance guard: the fixture must come from the commit where
        # run_loop still called provider.stream_message directly.
        fixture = json.loads(FIXTURE.read_text())
        assert fixture["recorded_at_sha"].startswith("1367334")


# ---------------------------------------------------------------------------
# Envelope stream() unit behavior — passthrough by identity
# ---------------------------------------------------------------------------


class TestStreamPassthrough:

    def test_events_and_request_pass_through_by_identity(self, tmp_path: Path):
        sent_events: list[ApiStreamEvent] = [
            ApiTextDeltaEvent(text="a"),
            ApiTextDeltaEvent(text="b"),
            _complete(_assistant_text("ab")),
        ]
        provider = ScriptedProvider([sent_events])
        envelope = LLMCallEnvelope(
            "agent_loop", telemetry=ToolCallTelemetry(db_path=tmp_path / "t.db")
        )
        request = ApiMessageRequest(
            model="m", messages=_user("hi"), max_tokens=8
        )

        async def collect() -> list[ApiStreamEvent]:
            return [
                e
                async for e in envelope.stream(
                    provider=provider, request=request
                )
            ]

        received = asyncio.run(collect())
        assert len(received) == len(sent_events)
        for got, sent in zip(received, sent_events):
            assert got is sent  # identity, not equality — zero transformation
        assert provider.requests[0] is request  # request never copied/rebuilt

    def test_telemetry_none_is_safe(self):
        provider = ScriptedProvider([[_complete(_assistant_text("x"))]])
        envelope = LLMCallEnvelope("agent_loop", telemetry=None)

        async def collect():
            return [
                e
                async for e in envelope.stream(
                    provider=provider,
                    request=ApiMessageRequest(model="m", messages=_user("q")),
                )
            ]

        assert len(asyncio.run(collect())) == 1


# ---------------------------------------------------------------------------
# Usage rows — the F1 acceptance ("tokens = null" structurally impossible)
# ---------------------------------------------------------------------------


class TestUsageRows:

    def test_fixture_turn_writes_usage_row(self, tmp_path: Path):
        provider = ScriptedProvider([[
            ApiTextDeltaEvent(text="OK"),
            _complete(_assistant_text("OK")),
        ]])
        tel = ToolCallTelemetry(db_path=tmp_path / "tel.db")
        asyncio.run(_drain(_context(provider, tel), _user("q")))

        rows = _usage_rows(tel)
        assert len(rows) == 1
        op, outcome, in_tok, out_tok, round_ix, sess, model, thinking, summary = rows[0]
        assert op == "loop_round"
        assert outcome == "success"
        assert in_tok == 123
        assert out_tok == 7
        assert round_ix == 0
        assert sess == "sess-fixture"
        assert model == "fixture-model"
        assert thinking is None  # ScriptedProvider exposes no thinking knob
        assert json.loads(summary)["stop_reason"] == "stop"

    def test_two_rounds_write_two_rows(self, tmp_path: Path):
        registry = ToolRegistry()
        registry.register(EchoTool())
        tool_call = ConversationMessage(
            role="assistant",
            content=[ToolUseBlock(id="t1", name="echo", input={"text": "hi"})],
        )
        provider = ScriptedProvider([
            [_complete(tool_call, input_tokens=100, output_tokens=10)],
            [_complete(_assistant_text("done"), input_tokens=150, output_tokens=5)],
        ])
        tel = ToolCallTelemetry(db_path=tmp_path / "tel.db")
        asyncio.run(_drain(
            _context(provider, tel, registry=registry), _user("use echo")
        ))

        rows = _usage_rows(tel)
        assert [(r[4], r[2], r[3]) for r in rows] == [
            (0, 100, 10),
            (1, 150, 5),
        ]
        assert all(r[1] == "success" for r in rows)

    def test_thinking_flag_provider_default_and_override(self, tmp_path: Path):
        provider = ScriptedProvider([
            [_complete(_assistant_text("a"))],
            [_complete(_assistant_text("b"))],
        ])
        provider._suppress_thinking = True  # what LlamaCppProvider carries
        tel = ToolCallTelemetry(db_path=tmp_path / "tel.db")
        envelope = LLMCallEnvelope("agent_loop", telemetry=tel)

        async def one(request: ApiMessageRequest) -> None:
            async for _ in envelope.stream(
                provider=provider, request=request, round_index=0
            ):
                pass

        # Provider default: suppressed → thinking recorded as 0.
        asyncio.run(one(ApiMessageRequest(model="m", messages=_user("q"))))
        # Per-call override back INTO thinking → recorded as 1.
        asyncio.run(one(ApiMessageRequest(
            model="m", messages=_user("q"), suppress_thinking=False
        )))

        flags = [r[7] for r in _usage_rows(tel)]
        assert flags == [0, 1]

    def test_record_run_old_signature_unchanged(self, tmp_path: Path):
        # Every pre-existing caller passes no usage kwargs → NULL columns.
        tel = ToolCallTelemetry(db_path=tmp_path / "tel.db")
        tel.record_run(
            subsystem="curator", operation="run_once", outcome="success"
        )
        row = tel._conn.execute(
            "SELECT input_tokens, output_tokens, round_index, session_id,"
            " model, thinking FROM subsystem_runs WHERE subsystem='curator'"
        ).fetchone()
        assert row == (None, None, None, None, None, None)


# ---------------------------------------------------------------------------
# Failure paths — capture without changing loop error behavior
# ---------------------------------------------------------------------------


class TestFailurePaths:

    def test_provider_exception_recorded_and_propagates_unchanged(
        self, tmp_path: Path
    ):
        exc = RuntimeError("provider died mid-round")
        tel = ToolCallTelemetry(db_path=tmp_path / "tel.db")
        with pytest.raises(RuntimeError, match="provider died mid-round"):
            asyncio.run(_drain(
                _context(RaisingProvider(exc), tel), _user("q")
            ))

        sf = tel._conn.execute(
            "SELECT subsystem, operation, exception_type, exception_msg"
            " FROM silent_failures"
        ).fetchall()
        assert len(sf) == 1
        assert sf[0][0] == "agent_loop"
        assert sf[0][1] == "loop_round"
        assert sf[0][2] == "RuntimeError"
        assert "provider died mid-round" in sf[0][3]

        rows = _usage_rows(tel)
        assert len(rows) == 1  # exactly one failed row — no double-count
        assert rows[0][1] == "failed"
        assert json.loads(rows[0][8])["exception_type"] == "RuntimeError"

    def test_empty_stream_keeps_loop_error_and_records_row(self, tmp_path: Path):
        provider = ScriptedProvider([[]])  # stream ends with no events
        tel = ToolCallTelemetry(db_path=tmp_path / "tel.db")
        with pytest.raises(
            RuntimeError, match="Model stream finished without a final message"
        ):
            asyncio.run(_drain(_context(provider, tel), _user("q")))

        rows = _usage_rows(tel)
        assert len(rows) == 1
        assert rows[0][1] == "failed"
        assert json.loads(rows[0][8])["reason"] == "empty_stream"

    def test_empty_content_complete_event_flagged(self, tmp_path: Path):
        # Collapse-arc shape: an empty complete event (no prose, no tool calls).
        # It must still be FLAGGED (observable). The empty-response guard retries
        # once — here the retry recovers, so the empty turn never reaches the
        # history but its emptiness is recorded on the first round.
        provider = ScriptedProvider([
            [_complete(
                ConversationMessage(role="assistant", content=[]),
                stop_reason="stop",
            )],
            [_complete(
                ConversationMessage(role="assistant", content=[TextBlock(text="recovered")]),
                stop_reason="stop",
            )],
        ])
        tel = ToolCallTelemetry(db_path=tmp_path / "tel.db")
        asyncio.run(_drain(_context(provider, tel), _user("q")))

        rows = _usage_rows(tel)
        assert len(rows) == 2  # the empty call + the retry
        assert rows[0][1] == "success"  # the call itself succeeded (empty != crash)
        assert json.loads(rows[0][8])["empty_content"] is True  # empty STILL flagged
        assert len(provider.requests) == 2  # retried once, then recovered
