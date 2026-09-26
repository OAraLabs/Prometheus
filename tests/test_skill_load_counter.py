"""Every ``skill`` load is recorded, so skill use can be measured.

Before this nothing counted a load. The Curator read file mtime as "last
used", the Telegram/Beacon lists showed mtime under that label, and the
skill-usage audit (docs/audits/SKILL-USAGE.md) had to reconstruct loads from
``tool_calls`` rows — which drop the session on every failure path. A load is
now one ``subsystem_runs`` row (no new table or column: the parity harness
dumps every table, so a schema change would move every golden).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
from prometheus.skills.registry import SkillRegistry
from prometheus.skills.types import SkillDefinition
from prometheus.telemetry.tracker import (
    SKILL_LOAD_OPERATION,
    SKILL_LOAD_SUBSYSTEM,
    ToolCallTelemetry,
    get_telemetry_handle,
    set_telemetry_handle,
)
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolRegistry, ToolResult
from prometheus.tools.builtin import skill as skill_mod
from prometheus.tools.builtin.skill import SkillTool, SkillToolInput


def _registry() -> SkillRegistry:
    reg = SkillRegistry()
    reg.register(SkillDefinition(
        name="release-check", description="Check a release before tagging it",
        content="---\nname: release-check\n---\n# Steps\n1. check", source="auto",
        path="/cfg/skills/auto/release-check.md",
    ))
    reg.register(SkillDefinition(
        name="commit", description="Stage and commit", content="# Commit",
        source="builtin", path="/pkg/skills/builtin/commit.md",
    ))
    return reg


@pytest.fixture
def telemetry(tmp_path, monkeypatch):
    tel = ToolCallTelemetry(tmp_path / "telemetry.db")
    previous = get_telemetry_handle()
    set_telemetry_handle(tel)
    monkeypatch.setattr(skill_mod, "load_skill_registry", lambda cwd=None: _registry())
    yield tel
    set_telemetry_handle(previous)


def _load_rows(tel: ToolCallTelemetry) -> list[tuple]:
    return tel._conn.execute(
        "SELECT outcome, session_id, summary_json FROM subsystem_runs "
        "WHERE subsystem = ? AND operation = ? ORDER BY rowid",
        (SKILL_LOAD_SUBSYSTEM, SKILL_LOAD_OPERATION),
    ).fetchall()


def _run(tool: SkillTool, name: str, **metadata) -> ToolResult:
    ctx = ToolExecutionContext(cwd=Path.cwd(), metadata=metadata)
    return asyncio.run(tool.execute(SkillToolInput(name=name), ctx))


class TestTheLoadIsRecorded:
    def test_a_load_records_name_source_file_and_the_turns_own_session(self, telemetry):
        result = _run(SkillTool(), "release-check",
                      session_id="web", effective_session_id="beacon:abc", ephemeral=False)
        assert not result.is_error
        assert "# Steps" in result.output
        [(outcome, session, summary)] = _load_rows(telemetry)
        assert outcome == "success"
        # effective_session_id, NOT the shared web routing namespace (#458).
        assert session == "beacon:abc"
        assert json.loads(summary) == {"skill": "release-check", "source": "auto",
                                       "file": "release-check"}

    def test_without_an_effective_session_the_context_session_is_used(self, telemetry):
        _run(SkillTool(), "commit", session_id="telegram:42")
        [(_, session, summary)] = _load_rows(telemetry)
        assert session == "telegram:42"
        assert json.loads(summary)["source"] == "builtin"

    def test_an_ephemeral_load_is_counted_without_its_session(self, telemetry):
        _run(SkillTool(), "commit", session_id="beacon:e", effective_session_id="beacon:e",
             ephemeral=True)
        [(outcome, session, _)] = _load_rows(telemetry)
        assert outcome == "success"
        assert session is None

    def test_a_miss_is_recorded_as_failed_and_is_not_a_load(self, telemetry):
        result = _run(SkillTool(), "no-such-skill", effective_session_id="telegram:1")
        assert result.is_error
        assert "Skill not found" in result.output
        [(outcome, _, summary)] = _load_rows(telemetry)
        assert outcome == "failed"
        assert json.loads(summary) == {"skill": "no-such-skill", "reason": "not_found"}
        assert telemetry.skill_load_stats() == {}

    def test_no_telemetry_handle_still_returns_the_skill(self, monkeypatch):
        previous = get_telemetry_handle()
        set_telemetry_handle(None)
        monkeypatch.setattr(skill_mod, "load_skill_registry", lambda cwd=None: _registry())
        try:
            result = _run(SkillTool(), "commit")
        finally:
            set_telemetry_handle(previous)
        assert result.output == "# Commit"

    def test_a_broken_telemetry_write_never_breaks_the_load(self, telemetry, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("disk full")
        monkeypatch.setattr(telemetry, "record_run", boom)
        assert _run(SkillTool(), "commit").output == "# Commit"


class TestLoadStats:
    def test_counts_last_load_and_source_per_skill(self, tmp_path):
        tel = ToolCallTelemetry(tmp_path / "t.db")
        for skill, ts in (("a", 100.0), ("a", 300.0), ("b", 200.0)):
            tel.record_run(SKILL_LOAD_SUBSYSTEM, SKILL_LOAD_OPERATION, "success",
                           summary={"skill": skill, "source": "auto", "file": skill})
            tel._conn.execute("UPDATE subsystem_runs SET timestamp = ? WHERE rowid = "
                              "(SELECT MAX(rowid) FROM subsystem_runs)", (ts,))
        tel.record_run(SKILL_LOAD_SUBSYSTEM, SKILL_LOAD_OPERATION, "failed",
                       summary={"skill": "a", "reason": "not_found"})
        stats = tel.skill_load_stats()
        assert stats == {
            "a": {"loads": 2, "last_loaded_at": 300.0, "source": "auto", "file": "a"},
            "b": {"loads": 1, "last_loaded_at": 200.0, "source": "auto", "file": "b"},
        }

    def test_no_loads_is_an_empty_dict(self, tmp_path):
        assert ToolCallTelemetry(tmp_path / "t.db").skill_load_stats() == {}


# ---------------------------------------------------------------------------
# The loop hands every tool the turn's own session and the ephemeral flag
# ---------------------------------------------------------------------------

class _ProbeInput(BaseModel):
    pass


class _ProbeTool(BaseTool):
    name = "probe"
    description = "records its execution metadata"
    input_model = _ProbeInput

    def __init__(self) -> None:
        self.seen: list[dict] = []

    def is_read_only(self, arguments: BaseModel) -> bool:
        return True

    async def execute(self, arguments, context):  # noqa: ANN001
        self.seen.append(dict(context.metadata))
        return ToolResult(output="ok")


class _CallsProbeOnce(ModelProvider):
    def __init__(self) -> None:
        self.calls = 0

    async def stream_message(self, request):  # noqa: ANN001
        self.calls += 1
        content = ([ToolUseBlock(id="p1", name="probe", input={})] if self.calls == 1
                   else [TextBlock(text="done")])
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=content),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1), stop_reason="stop",
        )


def _run_probe(session_id: str, *, ephemeral: bool, monkeypatch) -> dict:
    import prometheus.engine.agent_loop as al

    monkeypatch.setattr(al, "is_session_ephemeral", lambda sid: ephemeral)
    probe = _ProbeTool()
    registry = ToolRegistry()
    registry.register(probe)
    ctx = LoopContext(provider=_CallsProbeOnce(), model="stub", system_prompt="S",
                      max_tokens=64, tool_registry=registry, session_id="web")

    async def go():
        async for _ in run_loop(ctx, [ConversationMessage.from_user_text("go")],
                                session_id=session_id):
            pass

    asyncio.run(go())
    [seen] = probe.seen
    return seen


def test_tools_see_the_turns_effective_session(monkeypatch):
    seen = _run_probe("beacon:xyz", ephemeral=False, monkeypatch=monkeypatch)
    assert seen["effective_session_id"] == "beacon:xyz"
    assert seen["ephemeral"] is False


def test_tools_see_the_ephemeral_flag(monkeypatch):
    seen = _run_probe("beacon:eph", ephemeral=True, monkeypatch=monkeypatch)
    assert seen["ephemeral"] is True


# ---------------------------------------------------------------------------
# /api/status shows loads per skill
# ---------------------------------------------------------------------------

def test_api_status_reports_loads_per_skill(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from prometheus.engine.session import SessionManager
    from prometheus.web.server import create_app

    tel = ToolCallTelemetry(tmp_path / "t.db")
    for skill in ("a", "a", "b"):
        tel.record_run(SKILL_LOAD_SUBSYSTEM, SKILL_LOAD_OPERATION, "success",
                       summary={"skill": skill, "source": "auto", "file": skill})
    client = TestClient(create_app({}, session_mgr=SessionManager(), telemetry=tel))
    body = client.get("/api/status").json()
    block = body["skill_loads"]
    assert block["total"] == 3
    assert [(s["name"], s["loads"]) for s in block["skills"]] == [("a", 2), ("b", 1)]
    assert all(isinstance(s["last_loaded_at"], float) for s in block["skills"])


def test_api_status_without_telemetry_says_so():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from prometheus.engine.session import SessionManager
    from prometheus.web.server import create_app

    body = TestClient(create_app({}, session_mgr=SessionManager())).get("/api/status").json()
    assert body["skill_loads"] == {"available": False}
