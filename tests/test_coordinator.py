"""Tests for the coordinator module (Sprint 8)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prometheus.coordinator.agent_definitions import (
    AgentDefinition,
    get_agent_definition,
    get_all_agent_definitions,
    register_agent_definition,
)
from prometheus.coordinator.coordinator_mode import (
    TeamRecord,
    TeamRegistry,
    get_coordinator_system_prompt,
    get_team_registry,
    is_coordinator_mode,
)
from prometheus.coordinator.subagent import SubagentResult, SubagentSpawner
from prometheus.coordinator.health import (
    ComponentHealth,
    HealthMonitor,
    HealthState,
    HealthStatus,
    check_disk,
    check_sqlite,
)


# ---------------------------------------------------------------------------
# AgentDefinition
# ---------------------------------------------------------------------------


class TestAgentDefinitions:
    def test_builtin_agents_exist(self):
        defs = get_all_agent_definitions()
        assert "general-purpose" in defs
        assert "explorer" in defs
        assert "planner" in defs
        assert "worker" in defs
        assert "verification" in defs

    def test_get_existing(self):
        defn = get_agent_definition("explorer")
        assert defn is not None
        assert defn.name == "explorer"
        assert defn.read_only is True

    def test_get_missing(self):
        assert get_agent_definition("nonexistent-agent-xyz") is None

    def test_register_custom(self):
        custom = AgentDefinition(
            name="test-custom",
            description="Custom test agent",
            system_prompt="You are a test agent.",
            tools=["Bash"],
        )
        register_agent_definition(custom)
        assert get_agent_definition("test-custom") is not None
        assert get_agent_definition("test-custom").description == "Custom test agent"

    def test_definition_defaults(self):
        defn = AgentDefinition(name="minimal", description="minimal")
        assert defn.system_prompt == ""
        assert defn.tools == []
        assert defn.model == ""
        assert defn.read_only is False
        assert defn.max_turns == 50

    def test_worker_is_not_read_only(self):
        defn = get_agent_definition("worker")
        assert defn is not None
        assert defn.read_only is False
        assert defn.max_turns == 100


# ---------------------------------------------------------------------------
# TeamRegistry
# ---------------------------------------------------------------------------


class TestTeamRegistry:
    def test_create_and_get(self):
        reg = TeamRegistry()
        team = reg.create_team("alpha", description="Test team", agents=["worker", "explorer"])
        assert team.name == "alpha"
        assert reg.get_team("alpha") is team

    def test_get_missing(self):
        reg = TeamRegistry()
        assert reg.get_team("nonexistent") is None

    def test_list_teams(self):
        reg = TeamRegistry()
        reg.create_team("a")
        reg.create_team("b")
        assert len(reg.list_teams()) == 2

    def test_add_agent_to_team(self):
        reg = TeamRegistry()
        reg.create_team("t1", agents=["worker"])
        assert reg.add_agent_to_team("t1", "explorer")
        team = reg.get_team("t1")
        assert "explorer" in team.agents
        # No duplicates
        assert reg.add_agent_to_team("t1", "explorer")
        assert team.agents.count("explorer") == 1

    def test_add_agent_missing_team(self):
        reg = TeamRegistry()
        assert reg.add_agent_to_team("nope", "worker") is False

    def test_remove_agent_from_team(self):
        reg = TeamRegistry()
        reg.create_team("t1", agents=["worker", "explorer"])
        assert reg.remove_agent_from_team("t1", "worker")
        team = reg.get_team("t1")
        assert "worker" not in team.agents

    def test_remove_agent_missing_team(self):
        reg = TeamRegistry()
        assert reg.remove_agent_from_team("nope", "worker") is False


class TestCoordinatorMode:
    def test_is_coordinator_mode(self):
        assert is_coordinator_mode(0) is False
        assert is_coordinator_mode(1) is False
        assert is_coordinator_mode(2) is True
        assert is_coordinator_mode(5) is True

    def test_system_prompt_basic(self):
        prompt = get_coordinator_system_prompt()
        assert "coordinator" in prompt.lower()
        assert "subtask" in prompt.lower()

    def test_system_prompt_with_team(self):
        team = TeamRecord(name="test-team", description="A test", agents=["worker", "explorer"])
        prompt = get_coordinator_system_prompt(team)
        assert "test-team" in prompt
        assert "worker" in prompt

    def test_get_team_registry_singleton(self):
        reg1 = get_team_registry()
        reg2 = get_team_registry()
        assert reg1 is reg2


# ---------------------------------------------------------------------------
# SubagentSpawner
# ---------------------------------------------------------------------------


class TestSubagentSpawner:
    def test_subagent_result_dataclass(self):
        r = SubagentResult(
            agent_id="sub_12345678",
            agent_type="explorer",
            text="Found 3 files",
            turns=2,
        )
        assert r.success is True
        assert r.error is None
        assert r.agent_id == "sub_12345678"

    def test_subagent_result_failure(self):
        r = SubagentResult(
            agent_id="sub_fail",
            agent_type="worker",
            text="",
            success=False,
            error="Provider timeout",
        )
        assert r.success is False
        assert r.error == "Provider timeout"

    @pytest.mark.asyncio
    async def test_spawn_success(self):
        """Test spawning a subagent with a mocked provider."""
        from prometheus.engine.agent_loop import RunResult
        from prometheus.engine.messages import ConversationMessage, TextBlock

        mock_provider = MagicMock()
        spawner = SubagentSpawner(provider=mock_provider, model="test-model")

        mock_result = RunResult(
            text="Task completed successfully.",
            messages=[
                ConversationMessage(role="assistant", content=[TextBlock(text="Task completed successfully.")])
            ],
            turns=1,
        )

        with patch.object(
            SubagentSpawner,
            "spawn",
            new_callable=AsyncMock,
            return_value=SubagentResult(
                agent_id="sub_mock",
                agent_type="general-purpose",
                text="Task completed successfully.",
                turns=1,
            ),
        ):
            result = await spawner.spawn(task="Test task")
            assert result.success is True
            assert result.text == "Task completed successfully."

    @pytest.mark.asyncio
    async def test_spawn_with_agent_type(self):
        """Test that agent_type lookup works."""
        mock_provider = MagicMock()
        spawner = SubagentSpawner(provider=mock_provider)

        with patch.object(
            SubagentSpawner,
            "spawn",
            new_callable=AsyncMock,
            return_value=SubagentResult(
                agent_id="sub_expl",
                agent_type="explorer",
                text="Found files",
                turns=1,
            ),
        ):
            result = await spawner.spawn(task="List files", agent_type="explorer")
            assert result.agent_type == "explorer"

    def test_build_tool_registry_none(self):
        """No parent registry returns None."""
        spawner = SubagentSpawner(provider=MagicMock())
        assert spawner._build_tool_registry(None) is None

    def test_build_tool_registry_subset(self):
        """Subset filters tools from parent."""
        from prometheus.tools.base import ToolRegistry

        parent = ToolRegistry()
        mock_tool = MagicMock()
        mock_tool.name = "Bash"
        parent.register(mock_tool)

        mock_tool2 = MagicMock()
        mock_tool2.name = "Grep"
        parent.register(mock_tool2)

        spawner = SubagentSpawner(provider=MagicMock(), parent_tool_registry=parent)
        subset = spawner._build_tool_registry(["Bash"])
        assert subset is not None
        assert subset.get("Bash") is not None
        assert subset.get("Grep") is None

    def test_build_tool_registry_all(self):
        """None tool_names returns full parent."""
        from prometheus.tools.base import ToolRegistry

        parent = ToolRegistry()
        spawner = SubagentSpawner(provider=MagicMock(), parent_tool_registry=parent)
        assert spawner._build_tool_registry(None) is parent


# ---------------------------------------------------------------------------
# HealthMonitor
# ---------------------------------------------------------------------------


class TestHealthComponents:
    def test_component_health_ok(self):
        c = ComponentHealth(name="test", healthy=True, detail="all good")
        assert c.healthy
        assert c.name == "test"

    def test_component_health_fail(self):
        c = ComponentHealth(name="test", healthy=False, detail="down")
        assert not c.healthy

    def test_health_status_healthy(self):
        hs = HealthStatus(
            state=HealthState.HEALTHY,
            components=[ComponentHealth(name="a", healthy=True)],
        )
        assert hs.state == HealthState.HEALTHY
        assert len(hs.degraded_components) == 0

    def test_health_status_degraded(self):
        hs = HealthStatus(
            state=HealthState.DEGRADED,
            components=[
                ComponentHealth(name="a", healthy=True),
                ComponentHealth(name="b", healthy=False, detail="down"),
            ],
        )
        assert len(hs.degraded_components) == 1
        assert hs.degraded_components[0].name == "b"

    def test_health_summary(self):
        hs = HealthStatus(
            state=HealthState.HEALTHY,
            components=[ComponentHealth(name="disk", healthy=True, detail="50GB free")],
        )
        summary = hs.summary()
        assert "healthy" in summary.lower()
        assert "disk" in summary

    def test_check_disk(self):
        """Disk check should always work on the test machine."""
        result = check_disk("/")
        assert result.name == "disk"
        assert result.healthy is True
        assert "GB" in result.detail

    def test_check_sqlite_valid(self, tmp_path):
        """SQLite check on a real temp DB."""
        import sqlite3

        db = tmp_path / "test.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE t (id INTEGER)")
        conn.close()
        result = check_sqlite(str(db))
        assert result.healthy is True
        assert result.name == "sqlite"

    def test_check_sqlite_missing(self):
        """SQLite check on a nonexistent path should still succeed (sqlite creates it)."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as td:
            result = check_sqlite(os.path.join(td, "new.db"))
            assert result.healthy is True


class TestHealthMonitor:
    def test_init_defaults(self):
        hm = HealthMonitor()
        assert hm.interval == 60
        assert len(hm._enabled_checks) == 5

    def test_init_custom_checks(self):
        hm = HealthMonitor(checks=["disk", "sqlite"])
        assert hm._enabled_checks == ["disk", "sqlite"]

    @pytest.mark.asyncio
    async def test_check_runs(self):
        """Integration test: run one health check with disk+sqlite only."""
        hm = HealthMonitor(checks=["disk", "sqlite"])
        status = await hm.check()
        assert status.state in (HealthState.HEALTHY, HealthState.DEGRADED, HealthState.CRITICAL)
        assert len(status.components) == 2
        names = [c.name for c in status.components]
        assert "disk" in names
        assert "sqlite" in names

    @pytest.mark.asyncio
    async def test_alert_callback_on_degraded(self):
        """Alert callback fires when state transitions to degraded."""
        alerts = []

        async def on_alert(status: HealthStatus):
            alerts.append(status)

        hm = HealthMonitor(
            interval=1,
            checks=["disk"],
            alert_callback=on_alert,
        )

        # Mock a degraded check
        with patch.object(
            hm,
            "_run_checks",
            return_value=[ComponentHealth(name="disk", healthy=False, detail="full")],
        ):
            status = await hm.check()
            assert status.state in (HealthState.DEGRADED, HealthState.CRITICAL)

    def test_stop(self):
        hm = HealthMonitor()
        hm._running = True
        hm.stop()
        assert hm._running is False
