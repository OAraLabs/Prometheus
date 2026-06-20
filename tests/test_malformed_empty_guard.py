"""malformed_empty guard — provider-boundary drop + loop feedback retry.

D1 (2026-06 tool-calling diagnostics): llama.cpp's native tool-call stream
can deliver entries with an empty function name; these became
``ToolUseBlock(name="")``, flowed to the registry as ``Unknown tool: `` with
no useful feedback, and burned turns until the circuit breaker killed them
(232 telemetry rows, all failures). The guard has two layers:

1. ``_parse_assistant_message`` drops empty-name entries — loud telemetry
   (``malformed_empty`` under sentinel tool ``_malformed``), never a
   ToolUseBlock.
2. ``run_loop`` sees ``dropped_malformed`` on the complete event; when a
   turn produced nothing but dropped calls, it injects structured guidance
   and retries, bounded by the circuit breaker.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import BaseModel

from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
from prometheus.providers.stub import _parse_assistant_message
from prometheus.telemetry.tracker import (
    ToolCallTelemetry,
    set_telemetry_handle,
)
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult


# ---------------------------------------------------------------------------
# Layer 1 — provider parse boundary
# ---------------------------------------------------------------------------


def _choice(tool_calls, content=None):
    return {"message": {"content": content, "tool_calls": tool_calls}}


class TestParseBoundaryDrop:

    def test_empty_name_entry_never_becomes_tool_use_block(self):
        msg, dropped = _parse_assistant_message(
            _choice([{"id": "x1", "function": {"name": "", "arguments": ""}}]),
            model="gemma-test",
        )
        assert dropped == 1
        assert msg.tool_uses == []

    def test_whitespace_and_missing_name_both_drop(self):
        msg, dropped = _parse_assistant_message(
            _choice([
                {"id": "x1", "function": {"name": "   ", "arguments": "{}"}},
                {"id": "x2", "function": {"arguments": "{}"}},
            ]),
            model="gemma-test",
        )
        assert dropped == 2
        assert msg.tool_uses == []

    def test_valid_entries_survive_alongside_dropped(self):
        msg, dropped = _parse_assistant_message(
            _choice([
                {"id": "x1", "function": {"name": "", "arguments": ""}},
                {"id": "x2", "function": {"name": "bash", "arguments": '{"command": "ls"}'}},
            ]),
            model="gemma-test",
        )
        assert dropped == 1
        assert [t.name for t in msg.tool_uses] == ["bash"]
        assert msg.tool_uses[0].input == {"command": "ls"}

    def test_valid_name_with_empty_arguments_is_not_dropped(self):
        # No-arg tools legitimately stream empty arguments — pydantic
        # downstream judges those, the boundary guard must not.
        msg, dropped = _parse_assistant_message(
            _choice([{"id": "x1", "function": {"name": "cron_list", "arguments": ""}}]),
            model="gemma-test",
        )
        assert dropped == 0
        assert [t.name for t in msg.tool_uses] == ["cron_list"]
        assert msg.tool_uses[0].input == {}

    def test_text_content_is_preserved_when_calls_drop(self):
        msg, dropped = _parse_assistant_message(
            _choice(
                [{"id": "x1", "function": {"name": "", "arguments": ""}}],
                content="partial prose",
            ),
            model="gemma-test",
        )
        assert dropped == 1
        assert msg.text == "partial prose"


class TestParseBoundaryTelemetry:

    @pytest.fixture
    def tel(self, tmp_path: Path):
        t = ToolCallTelemetry(db_path=tmp_path / "tel.db")
        set_telemetry_handle(t)
        yield t
        set_telemetry_handle(None)

    def test_drop_records_malformed_empty_row_with_raw_payload(self, tel):
        _parse_assistant_message(
            _choice([{"id": "srv123", "function": {"name": "", "arguments": ""}}]),
            model="gemma-test",
        )
        rep = tel.report()
        assert "_malformed" in rep["tools"]
        row = tel._conn.execute(
            "SELECT model, success, error_type, error_detail, raw_model_output "
            "FROM tool_calls WHERE tool_name = '_malformed'"
        ).fetchone()
        assert row[0] == "gemma-test"
        assert row[1] == 0
        assert row[2] == "malformed_empty"
        # the raw entry is preserved for forensics — D1 had to dig the LCM
        assert "srv123" in row[3]
        assert "srv123" in row[4]

    def test_malformed_rows_count_in_overall_rate_not_excluded(self, tel):
        # These are REAL model failures: they stay in denominators
        # (unlike _loop_transition echoes).
        tel.record(model="m", tool_name="bash", success=True)
        _parse_assistant_message(
            _choice([{"id": "x", "function": {"name": "", "arguments": ""}}]),
            model="m",
        )
        rep = tel.report()
        assert rep["total_calls"] == 2
        assert rep["overall_success_rate"] == 0.5

    def test_no_telemetry_handle_does_not_break_parsing(self):
        set_telemetry_handle(None)
        msg, dropped = _parse_assistant_message(
            _choice([{"id": "x", "function": {"name": "", "arguments": ""}}]),
            model="m",
        )
        assert dropped == 1
        assert msg.tool_uses == []


# ---------------------------------------------------------------------------
# Layer 2 — loop feedback retry, breaker-bounded
# ---------------------------------------------------------------------------


class _EmptyInput(BaseModel):
    pass


class _ScriptedProvider(ModelProvider):
    """Yields scripted (message, dropped_malformed) complete events."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    async def stream_message(self, request):  # noqa: ANN001
        self.calls += 1
        msg, dropped = self.script.pop(0) if self.script else (
            ConversationMessage(role="assistant", content=[TextBlock(text="fallthrough")]),
            0,
        )
        yield ApiMessageCompleteEvent(
            message=msg,
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason="stop",
            dropped_malformed=dropped,
        )


def _empty_assistant():
    return ConversationMessage(role="assistant", content=[])


def _text_assistant(text):
    return ConversationMessage(role="assistant", content=[TextBlock(text=text)])


def _registry():
    class _Echo(BaseTool):
        name = "echo"
        description = "echoes"
        input_model = _EmptyInput

        async def execute(self, arguments, context):  # noqa: ANN001
            return ToolResult(output="ok")

    reg = ToolRegistry()
    reg.register(_Echo())
    return reg


def _drive(provider, registry):
    messages = [ConversationMessage.from_user_text("hi")]
    ctx = LoopContext(
        provider=provider,
        model="stub",
        system_prompt="",
        max_tokens=256,
        tool_registry=registry,
    )

    async def go():
        async for _ in run_loop(ctx, messages):
            pass

    asyncio.run(go())
    return messages


class TestLoopMalformedFeedback:

    def test_all_dropped_turn_gets_feedback_and_retries(self):
        provider = _ScriptedProvider([
            (_empty_assistant(), 3),          # collapse turn — all calls dropped
            (_text_assistant("recovered"), 0),  # model recovers after feedback
        ])
        messages = _drive(provider, _registry())

        assert provider.calls == 2
        feedback = [
            m for m in messages
            if m.role == "user" and m.provenance == "orchestrator"
        ]
        assert len(feedback) == 1
        assert "malformed tool call" in feedback[0].text
        assert "echo" in feedback[0].text  # available tools listed
        assert messages[-1].text == "recovered"
        # #65: the empty malformed turn must NOT enter history — a content-less
        # assistant message 400s the next request. Only feedback + recovery
        # turns survive; no empty assistant turn is committed.
        assert not any(
            m.role == "assistant" and not m.text.strip() and not m.tool_uses
            for m in messages
        )

    def test_repeated_collapse_trips_breaker_after_three(self):
        provider = _ScriptedProvider([
            (_empty_assistant(), 1),
            (_empty_assistant(), 1),
            (_empty_assistant(), 1),
            (_text_assistant("should never be reached"), 0),
        ])
        messages = _drive(provider, _registry())

        # 3 identical malformed turns → breaker trips on the third.
        assert provider.calls == 3
        assert "Circuit breaker tripped" in messages[-1].text

    def test_dropped_calls_with_prose_end_turn_normally(self):
        # If the model produced text alongside the dropped garbage, the
        # turn ends normally — no synthetic feedback, no retry.
        provider = _ScriptedProvider([
            (_text_assistant("here is my answer"), 2),
        ])
        messages = _drive(provider, _registry())

        assert provider.calls == 1
        assert all(m.provenance != "orchestrator" for m in messages)
        assert messages[-1].text == "here is my answer"
