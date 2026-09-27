"""A lucky guess is a marker, not a call (WP-X.21 T2, docs/audits/TELEMETRY-GAPS.md).

When the model calls a DEFERRED tool by name (registered, but not among the
schemas this run advertised), the loop notes it. It used to note it as a second
``tool_calls`` row, ``success=1`` and ``error_type='lucky_guess'``, written just
before the call's own row: one call, two "calls", and the marker had no session
id. Every per-call reader counted both — the telemetry report, ``/health``, the
tool feed, the dashboard's totals, the model ladder's success rate.

Pinned here, against the rows actually written:

* the loop writes ONE ``tool_calls`` row for the call, and the marker as a
  ``subsystem_runs`` row (``agent_loop`` / ``lucky_guess``) with the run's
  session, the model and the tool's name — no session on an ephemeral turn;
* a tool the run advertised writes no marker at all;
* every per-call reader skips the marker rows written before this change, and
  the dashboard counts lucky guesses from both places.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time

import pytest
from pydantic import BaseModel

from prometheus.context.dynamic_tools import DynamicToolLoader
from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
from prometheus.telemetry.dashboard import ToolDashboard
from prometheus.telemetry.tracker import ToolCallTelemetry
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult

SESSION = "desktop:lucky"


class _EmptyInput(BaseModel):
    pass


def _tool(tool_name: str) -> BaseTool:
    class _T(BaseTool):
        name = tool_name
        description = f"{tool_name} test tool"
        input_model = _EmptyInput

        async def execute(self, arguments, context):  # noqa: ANN001
            return ToolResult(output="ok", is_error=False)

    return _T()


class _OneCallProvider(ModelProvider):
    """Round 1 calls *tool*; round 2 answers in prose and ends the turn."""

    def __init__(self, tool: str) -> None:
        self.tool = tool
        self.calls = 0

    async def stream_message(self, request):  # noqa: ANN001
        self.calls += 1
        if self.calls == 1:
            content = [ToolUseBlock(id="t1", name=self.tool, input={})]
        else:
            content = [TextBlock(text="done")]
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=content),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason="stop",
        )


def _run(tmp_path, tool: str, session_id: str = SESSION) -> sqlite3.Connection:
    """Run one turn whose first round calls *tool*. Returns a read handle on the
    telemetry the loop wrote."""
    from prometheus.adapter import ModelAdapter

    registry = ToolRegistry()
    for name in ("bash", "read_file", "image_generate"):
        registry.register(_tool(name))
    # Deferred loading ON, advertising bash + read_file only: image_generate is
    # registered but not advertised, so calling it is a lucky guess.
    loader = DynamicToolLoader(registry, {"enabled": True, "always_loaded": ["bash", "read_file"]})
    db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db)
    ctx = LoopContext(
        provider=_OneCallProvider(tool),
        model="stub-model",
        system_prompt="",
        max_tokens=128,
        tool_registry=registry,
        tool_loader=loader,
        adapter=ModelAdapter(tier="light"),
        telemetry=tel,
    )

    async def drain() -> None:
        async for _ in run_loop(ctx, [ConversationMessage.from_user_text("go")], session_id=session_id):
            pass

    asyncio.run(drain())
    tel.close()
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    return con


def _calls(con: sqlite3.Connection) -> list[sqlite3.Row]:
    return con.execute(
        "SELECT tool_name, success, error_type, session_id FROM tool_calls"
        " WHERE tool_name != '_loop_transition' ORDER BY rowid"
    ).fetchall()


def _markers(con: sqlite3.Connection) -> list[sqlite3.Row]:
    return con.execute(
        "SELECT session_id, model, outcome, summary_json FROM subsystem_runs"
        " WHERE subsystem = 'agent_loop' AND operation = 'lucky_guess' ORDER BY rowid"
    ).fetchall()


# ---------------------------------------------------------------------------
# The writer
# ---------------------------------------------------------------------------


def test_a_lucky_guess_writes_one_call_row_and_a_marker_run_row(tmp_path):
    con = _run(tmp_path, "image_generate")

    calls = _calls(con)
    assert [(c["tool_name"], c["success"], c["error_type"], c["session_id"]) for c in calls] == [
        ("image_generate", 1, None, SESSION)
    ], "one call must be one tool_calls row — the marker used to be a second, 'successful' one"

    markers = _markers(con)
    assert len(markers) == 1
    m = markers[0]
    assert m["session_id"] == SESSION
    assert m["model"] == "stub-model"
    assert m["outcome"] == "success"
    assert json.loads(m["summary_json"]) == {"tool": "image_generate"}


def test_the_marker_carries_no_session_on_an_ephemeral_turn(tmp_path):
    from prometheus.config.ephemeral import set_session_ephemeral

    set_session_ephemeral(SESSION, True)
    con = _run(tmp_path, "image_generate")

    markers = _markers(con)
    assert len(markers) == 1
    assert markers[0]["session_id"] is None, "an ephemeral turn's rows name no session"
    assert [c["session_id"] for c in _calls(con)] == [None]


def test_a_tool_the_run_advertised_writes_no_marker(tmp_path):
    con = _run(tmp_path, "bash")

    assert [(c["tool_name"], c["error_type"]) for c in _calls(con)] == [("bash", None)]
    assert _markers(con) == []


# ---------------------------------------------------------------------------
# The readers: marker rows written before this change are not calls
# ---------------------------------------------------------------------------


def _history(tmp_path) -> ToolCallTelemetry:
    """A telemetry DB holding one FAILED bash call, and the marker the old
    writer put before it: a success=1 bash row with error_type='lucky_guess'."""
    tel = ToolCallTelemetry(tmp_path / "telemetry.db")
    tel.record(
        model="m", tool_name="bash", success=True, error_type="lucky_guess",
        error_detail="Tool bash called without being in prompt schema",
    )
    tel.record(model="m", tool_name="bash", success=False, error_type="tool_error",
               error_detail="boom", session_id=SESSION)
    return tel


def test_report_counts_the_call_and_not_its_old_marker(tmp_path):
    tel = _history(tmp_path)
    rep = tel.report()
    assert rep["total_calls"] == 1
    assert rep["overall_success_rate"] == 0.0, "the marker's success=1 must not rescue a failed call"
    assert rep["tools"]["bash"]["calls"] == 1
    assert rep["models"]["m"]["bash"]["calls"] == 1
    assert "lucky_guess" not in rep["tools"]["bash"]["error_types"]


def test_health_summary_counts_the_call_and_not_its_old_marker(tmp_path):
    tel = _history(tmp_path)
    h = tel.health_summary(since=time.time() - 3600)["tool_calls"]
    assert (h["total"], h["failures"], h["success_rate"]) == (1, 1, 0.0)


def test_recent_tool_calls_lists_the_call_and_not_its_old_marker(tmp_path):
    tel = _history(tmp_path)
    rows = tel.recent_tool_calls(limit=10)
    assert [(r["tool_name"], r["error_type"]) for r in rows] == [("bash", "tool_error")]


def test_dashboard_counts_lucky_guesses_from_both_places_and_not_as_calls(tmp_path):
    tel = _history(tmp_path)
    # A marker in the new shape, as the loop now writes it.
    tel.record_run(subsystem="agent_loop", operation="lucky_guess", outcome="success",
                   session_id=SESSION, model="m", summary={"tool": "read_file"})
    tel.close()
    dash = ToolDashboard(db_path=tmp_path / "telemetry.db")
    try:
        stats = dash.get_stats(hours=24)
    finally:
        dash.close()
    assert stats["lucky_guesses"] == 2, "one old marker row + one new marker run"
    assert stats["total_calls"] == 1
    assert stats["overall_success_rate"] == 0.0
    assert stats["success_rate_by_tool"] == {"bash": 0.0}
    assert stats["most_called"] == [{"tool_name": "bash", "calls": 1}]


def test_ladder_harvest_counts_the_call_and_not_its_old_marker(tmp_path):
    from prometheus.gym.ladder.record import harvest_run_metrics

    tel = _history(tmp_path)
    tel.close()
    con = sqlite3.connect(tmp_path / "telemetry.db")
    m = harvest_run_metrics(con, SESSION, window=(time.time() - 3600, time.time() + 1))
    assert m["tool_calls"] == 1, "the session-less marker must not join the run by time window"
    assert m["tool_calls_unattributed"] == 0
    assert m["tool_call_success"] == 0.0


@pytest.fixture(autouse=True)
def _no_global_handle(monkeypatch):
    # The loop writes through context.telemetry; keep any global handle a
    # previous test left behind out of the provider paths.
    import prometheus.telemetry.tracker as tracker

    monkeypatch.setattr(tracker, "_telemetry_singleton", None)
