"""Tests for ToolSearchTool skill search integration."""

import json
import pytest

from prometheus.tools.base import ToolRegistry
from prometheus.tools.builtin.bash import BashTool
from prometheus.tools.builtin.grep import GrepTool
from prometheus.tools.tool_search import ToolSearchTool, ToolSearchInput
from prometheus.tools.base import ToolExecutionContext
from prometheus.skills.types import SkillDefinition
from prometheus.skills.registry import SkillRegistry
from pathlib import Path


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(BashTool())
    reg.register(GrepTool())
    return reg


@pytest.fixture
def skill_registry():
    sr = SkillRegistry()
    sr.register(SkillDefinition(
        name="docker-deploy",
        description="Deploy containers with Docker and docker-compose",
        content="# Docker Deploy\nFull instructions here...",
        source="user",
        path="/home/will/.prometheus/skills/docker-deploy.md",
    ))
    sr.register(SkillDefinition(
        name="commit",
        description="Stage, write, and push a well-formed git commit",
        content="# Commit\nFull commit skill...",
        source="builtin",
        path="/home/will/Prometheus/src/prometheus/skills/builtin/commit.md",
    ))
    sr.register(SkillDefinition(
        name="debug",
        description="Systematic debugging workflow",
        content="# Debug\nFull debug skill...",
        source="builtin",
        path=None,
    ))
    return sr


@pytest.fixture
def ctx():
    return ToolExecutionContext(cwd=Path.cwd())


@pytest.fixture
def tool(registry, skill_registry):
    t = ToolSearchTool()
    t.set_registry(registry)
    t.set_skill_registry(skill_registry)
    return t


class TestSkillSearch:
    @pytest.mark.asyncio
    async def test_search_matches_skill_name(self, tool, ctx):
        result = await tool.execute(ToolSearchInput(query="docker"), ctx)
        data = json.loads(result.output)
        skill_results = [r for r in data if r.get("type") == "skill"]
        assert len(skill_results) >= 1
        assert any("docker" in r["name"] for r in skill_results)

    @pytest.mark.asyncio
    async def test_search_matches_tool_name(self, tool, ctx):
        result = await tool.execute(ToolSearchInput(query="bash"), ctx)
        data = json.loads(result.output)
        tool_results = [r for r in data if r.get("type") == "tool"]
        assert len(tool_results) >= 1
        assert any("bash" in r["name"] for r in tool_results)

    @pytest.mark.asyncio
    async def test_search_returns_both_types(self, tool, ctx):
        result = await tool.execute(ToolSearchInput(query="commit"), ctx)
        data = json.loads(result.output)
        types = {r.get("type") for r in data}
        # "commit" is a skill name, should appear
        assert "skill" in types

    @pytest.mark.asyncio
    async def test_select_skill_by_name(self, tool, ctx):
        result = await tool.execute(ToolSearchInput(query="docker-deploy", action="select"), ctx)
        data = json.loads(result.output)
        assert data["type"] == "skill"
        assert data["name"] == "docker-deploy"
        assert "description" in data
        assert "hint" in data

    @pytest.mark.asyncio
    async def test_select_tool_still_works(self, tool, ctx):
        result = await tool.execute(ToolSearchInput(query="bash", action="select"), ctx)
        data = json.loads(result.output)
        assert data["type"] == "tool"
        assert data["name"] == "bash"

    @pytest.mark.asyncio
    async def test_select_nonexistent_shows_skills(self, tool, ctx):
        result = await tool.execute(ToolSearchInput(query="nonexistent", action="select"), ctx)
        assert result.is_error
        data = json.loads(result.output)
        # Available list should include skills
        assert any("[skill]" in item for item in data["available"])

    @pytest.mark.asyncio
    async def test_empty_query_lists_skills(self, tool, ctx):
        result = await tool.execute(ToolSearchInput(query=""), ctx)
        data = json.loads(result.output)
        assert "tools" in data
        assert "skills" in data
        assert "docker-deploy" in data["skills"]

    @pytest.mark.asyncio
    async def test_search_without_skill_registry(self, registry, ctx):
        """Works fine without skill registry — just searches tools."""
        t = ToolSearchTool()
        t.set_registry(registry)
        result = await t.execute(ToolSearchInput(query="bash"), ctx)
        data = json.loads(result.output)
        assert any("bash" in r.get("name", "") for r in data)


class TestPromptAssemblySkills:
    def test_prompt_no_longer_lists_skills(self):
        from prometheus.context.prompt_assembler import build_runtime_system_prompt
        skills = [{"name": f"skill_{i}", "description": f"Desc {i}"} for i in range(50)]
        prompt = build_runtime_system_prompt(
            cwd="/tmp", config={}, skills=skills,
        )
        # Should NOT contain individual skill listings
        assert "skill_0" not in prompt
        assert "skill_49" not in prompt
        # Should contain the one-liner
        assert "tool_search" in prompt
        assert "skill tool" in prompt

    def test_prompt_without_skills_has_no_skills_section(self):
        from prometheus.context.prompt_assembler import build_runtime_system_prompt
        prompt = build_runtime_system_prompt(cwd="/tmp", config={})
        assert "tool_search to find skills" not in prompt

    def test_token_savings(self):
        from prometheus.context.token_estimation import estimate_tokens
        from prometheus.context.prompt_assembler import build_runtime_system_prompt

        skills = [{"name": f"skill_{i}", "description": f"Description for skill number {i}"} for i in range(95)]

        # The new prompt with one-liner
        prompt_new = build_runtime_system_prompt(cwd="/tmp", config={}, skills=skills)
        tokens_new = estimate_tokens(prompt_new)

        # Estimate old: the one-liner is ~30 tokens, old listing would be ~6300
        # We verify the new prompt's skills section is small
        skills_text = "Use tool_search to find skills"
        assert skills_text in prompt_new
        # The full listing of 95 skills would be ~25K chars / ~6300 tokens
        # New prompt should be well under that
        assert tokens_new < 10000  # generous ceiling — base prompt + one line
