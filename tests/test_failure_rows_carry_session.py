"""Every row the loop writes to ``tool_calls`` carries the turn's session, and a failed
call records what was called (WP-X.21 T1, T3, T15; docs/audits/TELEMETRY-GAPS.md).

* T1: only the main path passed a session. The failure writers (permission denied,
  unknown tool, validation, timeout, a tool that raised, …) passed nothing, so every
  failure row since the column appeared was session-less, and the Instinct corpus
  and the skill audit could not count failures per conversation.
* T3: ``_loop_transition`` rows — why each round ended — carried no session either,
  so no reader could say whose turn ended on a breaker trip or the iteration cap.
* T15: six failure writers dropped the call's input; Beacon's tool feed showed a
  denied or timed-out call with none.

Pinned here through a real ``run_loop``, against the rows actually written. The
rule is the main path's own: the turn's conversation, a record-only run's record
id, and nothing at all on an ephemeral turn.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest
from pydantic import BaseModel

from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.permissions.checker import PermissionDecision
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
from prometheus.telemetry.tracker import ToolCallTelemetry
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult

SESSION = "telegram:77"


class _PathInput(BaseModel):
    path: str


class _Read(BaseTool):
    name = "read_file"
    description = "reads a file"
    input_model = _PathInput

    async def execute(self, arguments, context):  # noqa: ANN001
        return ToolResult(output="contents", is_error=False)


class _Raises(BaseTool):
    name = "flaky"
    description = "raises"
    input_model = _PathInput

    async def execute(self, arguments, context):  # noqa: ANN001
        raise RuntimeError("boom")


class _Calls(ModelProvider):
    """Round 1 calls *tool* with a path; round 2 answers."""

    def __init__(self, tool: str) -> None:
        self.tool = tool
        self.calls = 0

    async def stream_message(self, request):  # noqa: ANN001
        self.calls += 1
        content = ([ToolUseBlock(id="t1", name=self.tool, input={"path": "notes.txt"})]
                   if self.calls == 1 else [TextBlock(text="done")])
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=content),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1), stop_reason="stop")


class _DenyAll:
    def evaluate(self, tool_name, **kw):  # noqa: ANN001
        return PermissionDecision.deny("denied by test")


def _run(tmp_path, tool: str, *, gate=None, session: str | None = SESSION,
         record_session_id: str | None = None) -> sqlite3.Connection:
    registry = ToolRegistry()
    registry.register(_Read())
    registry.register(_Raises())
    db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db)
    ctx = LoopContext(provider=_Calls(tool), model="stub-model", system_prompt="", max_tokens=128,
                      tool_registry=registry, telemetry=tel, permission_checker=gate)

    async def drain() -> None:
        async for _ in run_loop(ctx, [ConversationMessage.from_user_text("go")], session_id=session,
                                record_session_id=record_session_id):
            pass

    asyncio.run(drain())
    tel.close()
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    return con


def _failure(con) -> sqlite3.Row:
    [row] = con.execute("SELECT error_type, session_id, parsed_tool_call FROM tool_calls"
                        " WHERE tool_name != '_loop_transition'").fetchall()
    return row


def _transitions(con) -> set:
    return {r[0] for r in con.execute(
        "SELECT session_id FROM tool_calls WHERE tool_name = '_loop_transition'")}


@pytest.mark.parametrize("tool,gate,error_type", [
    ("read_file", _DenyAll(), "permission_denied"),
    ("no_such_tool", None, "unknown_tool"),
    ("flaky", None, "tool_exception"),
])
def test_a_failed_call_records_its_session_and_what_was_called(tmp_path, tool, gate, error_type):
    row = _failure(_run(tmp_path, tool, gate=gate))
    assert row["error_type"] == error_type
    assert row["session_id"] == SESSION, "the failure writers passed no session"
    assert json.loads(row["parsed_tool_call"]) == {"name": tool, "input": {"path": "notes.txt"}}, (
        "the failure writers dropped the call's input")


def test_a_rounds_transitions_carry_the_turns_session(tmp_path):
    con = _run(tmp_path, "read_file")
    assert _transitions(con) == {SESSION}, "_loop_transition rows were session-less"


def test_an_ephemeral_turn_records_no_session_and_no_input(tmp_path):
    from prometheus.config.ephemeral import set_session_ephemeral

    set_session_ephemeral(SESSION, True)
    con = _run(tmp_path, "read_file", gate=_DenyAll())
    row = _failure(con)
    assert (row["session_id"], row["parsed_tool_call"]) == (None, None)
    assert _transitions(con) == {None}


def test_a_record_only_runs_failures_and_transitions_carry_its_record_id(tmp_path):
    """A run with no session of its own (#608): its failure rows and transitions
    carry its record id, like its successful calls."""
    own = "subagent:telegram:77:agent-3"
    con = _run(tmp_path, "read_file", gate=_DenyAll(), session=None, record_session_id=own)
    assert _failure(con)["session_id"] == own
    assert _transitions(con) == {own}


@pytest.fixture(autouse=True)
def _no_global_handle(monkeypatch):
    import prometheus.telemetry.tracker as tracker

    monkeypatch.setattr(tracker, "_telemetry_singleton", None)
