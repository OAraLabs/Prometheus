"""Runs with no session of their own record under one, and nothing else changes
(WP-X.21 T4, docs/audits/TELEMETRY-GAPS.md).

Two entry points ran the loop with no session: POST ``/api/chat`` (the
conversation is ``web:<id>`` in LCM, but ``run_async`` got none) and the
subagent spawner. Every row such a run wrote — tool calls, rounds, the
advertisement — was session-less, so no per-session reader could find the run.

The fix is RECORD-ONLY (Will's ruling): a ``record_session_id`` that reaches the
telemetry writers and nothing else. The loop's own session, which decides the
permission origin, stays unset, so the origin stays ``system`` — the stricter
one, no user present to approve anything. The router's override lookup, the
profile and workspace resolvers, checkpoints and the compactor see what they
saw before. A subagent records under its OWN derived id,
``subagent:<parent session>:<spawn id>`` (``subagent:none:<spawn id>`` with no
parent), never the parent's: under the parent's, the golden-trace exporter
would pair the subagent's calls with the parent's conversation. The parent is
named in the run's advertisement summary.

Pinned here against the rows actually written.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from prometheus.engine.agent_loop import AgentLoop
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.permissions.checker import ORIGIN_SYSTEM, PermissionDecision, origin_from_session_id
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
from prometheus.telemetry.tracker import ToolCallTelemetry
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolRegistry, ToolResult


class _EmptyInput(BaseModel):
    pass


class _Bash(BaseTool):
    name = "bash"
    description = "test tool"
    input_model = _EmptyInput

    async def execute(self, arguments, context):  # noqa: ANN001
        return ToolResult(output="ok", is_error=False)


class _OneCallProvider(ModelProvider):
    """Round 1 calls bash; round 2 answers and ends the turn."""

    def __init__(self) -> None:
        self.calls = 0

    async def stream_message(self, request):  # noqa: ANN001
        self.calls += 1
        content = ([ToolUseBlock(id=f"t{self.calls}", name="bash", input={})]
                   if self.calls == 1 else [TextBlock(text="done")])
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=content),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason="stop",
        )


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(_Bash())
    return reg


def _rows(db) -> dict:
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    out = {
        "calls": [r["session_id"] for r in con.execute(
            "SELECT session_id FROM tool_calls WHERE tool_name != '_loop_transition' ORDER BY rowid")],
        "rounds": [r["session_id"] for r in con.execute(
            "SELECT session_id FROM subsystem_runs WHERE subsystem='agent_loop'"
            " AND operation='loop_round' ORDER BY rowid")],
        "ads": [(r["session_id"], json.loads(r["summary_json"])) for r in con.execute(
            "SELECT session_id, summary_json FROM subsystem_runs WHERE subsystem='agent_loop'"
            " AND operation='tool_advertisement' ORDER BY rowid")],
    }
    con.close()
    assert out["calls"] and out["rounds"] and out["ads"], "the run must write all three kinds of row"
    return out


# ---------------------------------------------------------------------------
# AgentLoop.run_async: the POST /api/chat shape
# ---------------------------------------------------------------------------


def test_a_run_with_no_session_files_its_rows_under_the_record_id(tmp_path):
    db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db)
    loop = AgentLoop(provider=_OneCallProvider(), model="stub-model",
                     tool_registry=_registry(), telemetry=tel)
    asyncio.run(loop.run_async("sys", "go", record_session_id="web:s1"))
    tel.close()

    rows = _rows(db)
    assert set(rows["calls"]) == {"web:s1"}
    assert set(rows["rounds"]) == {"web:s1"}
    assert [s for s, _ in rows["ads"]] == ["web:s1"]


class _RecordingGate:
    def __init__(self) -> None:
        self.origins: list[str] = []

    def evaluate(self, tool_name, **kw):  # noqa: ANN001
        self.origins.append(kw.get("origin"))
        return PermissionDecision.allow("test")


class _RecordingRouter:
    def __init__(self) -> None:
        self.sessions: list = []

    def route(self, text, context=None):  # noqa: ANN001
        self.sessions.append((context or {}).get("session_id"))
        return SimpleNamespace(reason="primary", provider_name="", model_name="",
                               provider=None, adapter=None, backend=None)


def test_the_record_id_reaches_telemetry_and_nothing_else(tmp_path):
    """Record-only: the gate still sees origin 'system', and the router, the
    profile resolver and the workspace resolver see the no-session they saw
    before."""
    gate, router = _RecordingGate(), _RecordingRouter()
    profiles: list = []
    workspaces: list = []

    def profile_resolver(session_id=None):  # noqa: ANN001
        profiles.append(session_id)
        return None

    def workspace_resolver(session_id):  # noqa: ANN001
        workspaces.append(session_id)
        return None

    tel = ToolCallTelemetry(tmp_path / "telemetry.db")
    loop = AgentLoop(provider=_OneCallProvider(), model="stub-model", tool_registry=_registry(),
                     telemetry=tel, permission_checker=gate, model_router=router,
                     profile_resolver=profile_resolver, workspace_resolver=workspace_resolver)
    asyncio.run(loop.run_async("sys", "go", record_session_id="web:s1"))
    tel.close()

    assert gate.origins and set(gate.origins) == {ORIGIN_SYSTEM}
    assert router.sessions == [None]
    assert profiles == [None]
    assert workspaces == [], "no session of its own, so no workspace lookup — as before"


def test_an_ephemeral_record_id_is_not_used(tmp_path):
    """The run is not made ephemeral (record-only), so its content columns are
    written as they always were — which is exactly why an ephemeral
    conversation's id must not be put beside them."""
    from prometheus.config.ephemeral import set_session_ephemeral

    set_session_ephemeral("web:s1", True)
    db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db)
    loop = AgentLoop(provider=_OneCallProvider(), model="stub-model",
                     tool_registry=_registry(), telemetry=tel)
    asyncio.run(loop.run_async("sys", "go", record_session_id="web:s1"))
    tel.close()

    rows = _rows(db)
    assert set(rows["calls"]) == {None}
    assert set(rows["rounds"]) == {None}


def test_a_run_with_its_own_session_ignores_the_record_id(tmp_path):
    db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db)
    loop = AgentLoop(provider=_OneCallProvider(), model="stub-model",
                     tool_registry=_registry(), telemetry=tel)
    asyncio.run(loop.run_async("sys", "go", session_id="telegram:42", record_session_id="web:s1"))
    tel.close()

    assert set(_rows(db)["calls"]) == {"telegram:42"}


# ---------------------------------------------------------------------------
# Subagents: their own derived id, the parent in the summary, origin 'system'
# ---------------------------------------------------------------------------


def _spawn(tmp_path, **kw):
    from prometheus.coordinator.subagent import SubagentSpawner

    db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db)
    spawner = SubagentSpawner(provider=_OneCallProvider(), parent_tool_registry=_registry(),
                              model="stub-model", telemetry=tel)
    result = asyncio.run(spawner.spawn(task="go", tools_subset=["bash"], **kw))
    tel.close()
    assert result.success, result.error
    return result, _rows(db)


def test_a_subagent_records_under_its_own_id_and_names_its_parent(tmp_path):
    result, rows = _spawn(tmp_path, parent_session_id="telegram:42")
    own = f"subagent:telegram:42:{result.agent_id}"
    assert set(rows["calls"]) == {own}, "never the parent's id: the exporter would pair them"
    assert set(rows["rounds"]) == {own}
    [(ad_session, summary)] = rows["ads"]
    assert ad_session == own
    assert summary["parent_session"] == "telegram:42"


def test_a_subagent_with_no_parent_records_under_none(tmp_path):
    result, rows = _spawn(tmp_path)
    own = f"subagent:none:{result.agent_id}"
    assert set(rows["calls"]) == {own}
    [(_, summary)] = rows["ads"]
    assert summary["parent_session"] is None


@pytest.mark.parametrize("parent", ["telegram:42", "web:s1", "cli", "web", "desktop:abc", None])
def test_a_subagent_id_classifies_as_system(parent):
    """Will's check: whatever the parent, the derived id is never a user origin
    (not that the loop feeds it to the gate — it does not)."""
    assert origin_from_session_id(f"subagent:{parent or 'none'}:sub_1a2b3c4d") == ORIGIN_SYSTEM


def test_the_agent_tool_passes_the_calling_turns_session_as_the_parent():
    from prometheus.tools.builtin.agent import AgentTool, AgentToolInput

    seen: list = []

    class _Spawner:
        async def spawn(self, **kw):  # noqa: ANN001
            seen.append(kw.get("parent_session_id"))
            return SimpleNamespace(success=True, text="ok", agent_id="sub_1", agent_type="t", turns=1)

    def ctx(ephemeral: bool) -> ToolExecutionContext:
        from pathlib import Path

        return ToolExecutionContext(cwd=Path("."), metadata={
            "subagent_spawner": _Spawner(), "effective_session_id": "desktop:abc",
            "session_id": "web", "ephemeral": ephemeral})

    args = AgentToolInput(description="d", prompt="p", subagent_type="general-purpose")
    asyncio.run(AgentTool().execute(args, ctx(False)))
    asyncio.run(AgentTool().execute(args, ctx(True)))
    assert seen == ["desktop:abc", None], "an ephemeral turn's subagent names no parent"


def test_an_escalation_subagent_names_the_turn_as_its_parent(monkeypatch):
    import prometheus.coordinator.subagent as subagent_mod
    from prometheus.engine.agent_loop import LoopContext, _try_escalate_tool_call

    seen: list = []

    class _Spawner:
        def __init__(self, *a, **kw):  # noqa: ANN001
            pass

        async def spawn(self, **kw):  # noqa: ANN001
            seen.append(kw.get("parent_session_id"))
            return SimpleNamespace(success=True, text="ok", error=None)

    monkeypatch.setattr(subagent_mod, "SubagentSpawner", _Spawner)
    router = SimpleNamespace(get_escalation_decision=lambda: SimpleNamespace(
        provider=object(), model_name="m", adapter=None, provider_name="p"))
    ctx = LoopContext(provider=_OneCallProvider(), model="m", system_prompt="", max_tokens=8,
                      model_router=router)
    block = asyncio.run(_try_escalate_tool_call(ctx, "bash", {}, "t1", "err",
                                                parent_session_id="telegram:42"))
    assert block is not None and not block.is_error
    assert seen == ["telegram:42"]


@pytest.fixture(autouse=True)
def _no_global_handle(monkeypatch):
    import prometheus.telemetry.tracker as tracker

    monkeypatch.setattr(tracker, "_telemetry_singleton", None)
