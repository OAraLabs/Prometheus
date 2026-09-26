"""A golden call's context is the conversation just before the call, however
long the session.

The exporter's resolver pairs each golden tool call with the conversation that
preceded it. It read ``get_messages(session_id, limit=500)``, which returns a
session's 500 LOWEST ``turn_index`` rows, and only then kept the rows before the
call. So every call after a session's 500th row was paired with the
conversation around row 500 instead of its own. On the mini's 2026-09-26
snapshot that was 2,473 of 5,521 golden calls, and 3,200 of the 5,519 lines
already exported carry a context this fix changes.

These run against a real ``LCMConversationStore``: the defect lived in the SQL
the resolver relied on, which no fake store can reproduce.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from prometheus.memory.lcm_conversation_store import LCMConversationStore
from prometheus.memory.lcm_types import MessagePart
from prometheus.sentinel.golden_trace_exporter import (
    CONTEXT_MESSAGE_LIMIT,
    GoldenTraceExporter,
    lcm_context_resolver,
)
from prometheus.telemetry.tracker import ToolCallTelemetry

SESSION = "telegram:long-chat"
ROWS = 1200  # well past the 500 the old read returned


def _long_session(store: LCMConversationStore, *, t0: float, rows: int = ROWS) -> None:
    """User and assistant rows alternating; row i is ``m<i>``, persisted at t0 + i."""
    for i in range(rows):
        store.insert_message(MessagePart(
            session_id=SESSION, turn_index=i,
            role="user" if i % 2 == 0 else "assistant",
            content=f"m{i}", timestamp=t0 + i,
        ))


def _contents(context: list[dict[str, str]]) -> list[str]:
    return [m["content"] for m in context]


@pytest.mark.parametrize("last_row", [100, 498, 500, 502, 1100, 1198])
def test_a_call_gets_the_rows_just_before_it(tmp_path, last_row):
    """Early or late in the session, the context is the 12 rows before the call.

    Through row 499 the old read held every row before the call, so 100 and
    498 resolved correctly before this fix too and must still. From row 500 on
    it returned rows 0-499 whatever the call, so every later call resolved to
    m487-m498.
    """
    store = LCMConversationStore(tmp_path / "lcm.db")
    _long_session(store, t0=1_000_000.0)

    got = lcm_context_resolver(store)(
        {"session_id": SESSION, "timestamp": 1_000_000.0 + last_row + 0.5}
    )

    first = last_row - CONTEXT_MESSAGE_LIMIT + 1
    assert _contents(got) == [f"m{i}" for i in range(first, last_row + 1)]


def test_the_daemon_export_pairs_a_late_call_with_its_own_context(tmp_path):
    """End to end through ``GoldenTraceExporter.run_once``, the path the daemon runs."""
    tel = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
    tel.record(
        model="claude-sonnet", tool_name="bash", success=True, retries=0,
        raw_model_output="Listing the files.",
        parsed_tool_call=json.dumps({"name": "bash", "input": {"command": "ls"}}),
        provider="anthropic", session_id=SESSION,
    )
    call_ts = tel.get_golden_traces()[0]["timestamp"]

    store = LCMConversationStore(tmp_path / "lcm.db")
    # m0-m1100 persisted before the call (the last one a user row), m1101-m1199 after.
    _long_session(store, t0=call_ts - 1100.5)

    exporter = GoldenTraceExporter(
        telemetry=tel, signal_bus=None,
        config={"output_dir": str(tmp_path / "trajectories")},
        conversation_store=store,
    )
    path = asyncio.run(exporter.run_once())

    assert path is not None, "the late call was skipped as untrainable"
    with open(path, encoding="utf-8") as fh:
        (line,) = fh.read().splitlines()
    *context, target = json.loads(line)["messages"]
    assert target["tool_calls"][0]["function"]["name"] == "bash"
    assert _contents(context) == [f"m{i}" for i in range(1089, 1101)]


def test_messages_before_reads_the_newest_rows_before_a_time_in_prompt_order(tmp_path):
    """Strictly before the time; the newest by (turn_index, rowid); returned ascending."""
    store = LCMConversationStore(tmp_path / "lcm.db")
    for turn_index, content, ts in [  # insertion order, so rowid order
        (0, "a", 1.0),
        (1, "b", 2.0),
        # Sent mid-turn: persisted before the turn's tail, at its prompt
        # position. turn_index puts it after d, where the model saw it.
        (4, "sent mid-turn", 3.0),
        (2, "c", 4.0),
        (3, "d", 5.0),
        # A numbering that restarted reused 2. rowid breaks the tie.
        (2, "c again", 6.0),
        (5, "at the call", 7.0),
        (6, "after the call", 8.0),
    ]:
        store.insert_message(MessagePart(
            session_id=SESSION, turn_index=turn_index, role="user",
            content=content, timestamp=ts,
        ))
    store.insert_message(MessagePart(
        session_id="another-session", turn_index=3, role="user",
        content="someone else", timestamp=2.5,
    ))

    def read(limit: int) -> list[str]:
        return [m.content for m in store.messages_before(SESSION, 7.0, limit=limit)]

    assert read(100) == ["a", "b", "c", "c again", "d", "sent mid-turn"]
    assert read(3) == ["c again", "d", "sent mid-turn"]


def test_rows_the_resolver_drops_do_not_starve_the_context(tmp_path):
    """The read window must be deeper than the 12-message cap.

    Tool calls and tool results persist with no flat text, so an agentic turn
    leaves a run of rows the resolver drops. A window the size of the cap
    would come back empty after one.
    """
    store = LCMConversationStore(tmp_path / "lcm.db")
    store.insert_message(MessagePart(
        session_id=SESSION, turn_index=0, role="user",
        content="fix the build", timestamp=1.0,
    ))
    for i in range(1, 101):
        store.insert_message(MessagePart(
            session_id=SESSION, turn_index=i,
            role="assistant" if i % 2 else "user", content="",
            content_json=json.dumps([{"type": "tool_use" if i % 2 else "tool_result"}]),
            timestamp=1.0 + i,
        ))

    got = lcm_context_resolver(store)({"session_id": SESSION, "timestamp": 200.0})

    assert _contents(got) == ["fix the build"]
