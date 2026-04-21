"""Tests for Layer 1 bootstrap system — SOUL.md, AGENTS.md, MEMORY.md, USER.md loading."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from prometheus.context.prompt_assembler import (
    _load_bootstrap_file,
    _load_memory_and_user,
    build_runtime_system_prompt,
)
from prometheus.context.system_prompt import SYSTEM_PROMPT_DYNAMIC_BOUNDARY


@pytest.fixture()
def bootstrap_dir(tmp_path: Path):
    """Create a temporary bootstrap directory with SOUL.md and AGENTS.md."""
    soul = tmp_path / "SOUL.md"
    soul.write_text("# Prometheus Identity\n\nYou are Prometheus.", encoding="utf-8")

    agents = tmp_path / "AGENTS.md"
    agents.write_text("# Agent Registry\n\nSpawn subagents for parallel work.", encoding="utf-8")

    return tmp_path


@pytest.fixture()
def memory_dir(tmp_path: Path):
    """Create a temporary config dir with MEMORY.md and USER.md."""
    memory = tmp_path / "MEMORY.md"
    memory.write_text("Alice prefers concise responses\nProject Prometheus is a local agent harness", encoding="utf-8")

    user = tmp_path / "USER.md"
    user.write_text("Senior engineer building AI infrastructure", encoding="utf-8")

    return tmp_path


# ---------------------------------------------------------------------------
# _load_bootstrap_file
# ---------------------------------------------------------------------------


class TestLoadBootstrapFile:
    def test_loads_existing_file(self, bootstrap_dir: Path) -> None:
        with patch("prometheus.context.prompt_assembler.get_config_dir", return_value=bootstrap_dir):
            content = _load_bootstrap_file("SOUL.md")
        assert content is not None
        assert "Prometheus Identity" in content

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        with patch("prometheus.context.prompt_assembler.get_config_dir", return_value=tmp_path):
            content = _load_bootstrap_file("NONEXISTENT.md")
        assert content is None

    def test_returns_none_for_empty_file(self, tmp_path: Path) -> None:
        empty = tmp_path / "EMPTY.md"
        empty.write_text("", encoding="utf-8")
        with patch("prometheus.context.prompt_assembler.get_config_dir", return_value=tmp_path):
            content = _load_bootstrap_file("EMPTY.md")
        assert content is None

    def test_strips_whitespace(self, tmp_path: Path) -> None:
        padded = tmp_path / "PADDED.md"
        padded.write_text("\n\n  hello world  \n\n", encoding="utf-8")
        with patch("prometheus.context.prompt_assembler.get_config_dir", return_value=tmp_path):
            content = _load_bootstrap_file("PADDED.md")
        assert content == "hello world"


# ---------------------------------------------------------------------------
# SOUL.md in system prompt
# ---------------------------------------------------------------------------


class TestSoulMdInPrompt:
    def test_soul_appears_in_static_section(self, bootstrap_dir: Path, tmp_path: Path) -> None:
        with patch("prometheus.context.prompt_assembler.get_config_dir", return_value=bootstrap_dir):
            prompt = build_runtime_system_prompt(cwd=str(tmp_path))

        static, _dynamic = prompt.split(SYSTEM_PROMPT_DYNAMIC_BOUNDARY)
        assert "Prometheus Identity" in static

    def test_soul_appears_first(self, bootstrap_dir: Path, tmp_path: Path) -> None:
        with patch("prometheus.context.prompt_assembler.get_config_dir", return_value=bootstrap_dir):
            prompt = build_runtime_system_prompt(cwd=str(tmp_path))

        # SOUL.md content should appear before the base system prompt text
        soul_pos = prompt.find("Prometheus Identity")
        base_pos = prompt.find("sovereign AI agent")
        assert soul_pos < base_pos, "SOUL.md must appear before base system prompt"

    def test_missing_soul_doesnt_crash(self, tmp_path: Path) -> None:
        with patch("prometheus.context.prompt_assembler.get_config_dir", return_value=tmp_path):
            prompt = build_runtime_system_prompt(cwd=str(tmp_path))
        assert SYSTEM_PROMPT_DYNAMIC_BOUNDARY in prompt
        assert "Prometheus" in prompt  # base prompt still works

    def test_config_disables_soul(self, bootstrap_dir: Path, tmp_path: Path) -> None:
        config = {"bootstrap": {"load_soul": False}}
        with patch("prometheus.context.prompt_assembler.get_config_dir", return_value=bootstrap_dir):
            prompt = build_runtime_system_prompt(cwd=str(tmp_path), config=config)
        assert "Prometheus Identity" not in prompt


# ---------------------------------------------------------------------------
# AGENTS.md in system prompt
# ---------------------------------------------------------------------------


class TestAgentsMdInPrompt:
    def test_agents_appears_in_static_section(self, bootstrap_dir: Path, tmp_path: Path) -> None:
        with patch("prometheus.context.prompt_assembler.get_config_dir", return_value=bootstrap_dir):
            prompt = build_runtime_system_prompt(cwd=str(tmp_path))

        static, _dynamic = prompt.split(SYSTEM_PROMPT_DYNAMIC_BOUNDARY)
        assert "Agent Registry" in static

    def test_missing_agents_doesnt_crash(self, tmp_path: Path) -> None:
        with patch("prometheus.context.prompt_assembler.get_config_dir", return_value=tmp_path):
            prompt = build_runtime_system_prompt(cwd=str(tmp_path))
        assert SYSTEM_PROMPT_DYNAMIC_BOUNDARY in prompt

    def test_config_disables_agents(self, bootstrap_dir: Path, tmp_path: Path) -> None:
        config = {"bootstrap": {"load_agents": False}}
        with patch("prometheus.context.prompt_assembler.get_config_dir", return_value=bootstrap_dir):
            prompt = build_runtime_system_prompt(cwd=str(tmp_path), config=config)
        assert "Agent Registry" not in prompt


# ---------------------------------------------------------------------------
# Bootstrap ordering in static section
# ---------------------------------------------------------------------------


class TestBootstrapOrdering:
    def test_soul_before_agents(self, bootstrap_dir: Path, tmp_path: Path) -> None:
        with patch("prometheus.context.prompt_assembler.get_config_dir", return_value=bootstrap_dir):
            prompt = build_runtime_system_prompt(cwd=str(tmp_path))

        soul_pos = prompt.find("Prometheus Identity")
        agents_pos = prompt.find("Agent Registry")
        assert soul_pos < agents_pos, "SOUL.md must appear before AGENTS.md"

    def test_bootstrap_before_tool_schemas(self, bootstrap_dir: Path, tmp_path: Path) -> None:
        with patch("prometheus.context.prompt_assembler.get_config_dir", return_value=bootstrap_dir):
            prompt = build_runtime_system_prompt(cwd=str(tmp_path))

        static, _ = prompt.split(SYSTEM_PROMPT_DYNAMIC_BOUNDARY)
        soul_pos = static.find("Prometheus Identity")
        env_pos = static.find("# Environment")
        assert soul_pos < env_pos, "Bootstrap files must appear before environment info"


# ---------------------------------------------------------------------------
# MEMORY.md + USER.md in dynamic section
# ---------------------------------------------------------------------------


class TestMemoryInPrompt:
    def test_memory_appears_in_dynamic_section(self, memory_dir: Path, tmp_path: Path) -> None:
        with patch("prometheus.context.prompt_assembler.get_config_dir", return_value=memory_dir), \
             patch("prometheus.memory.hermes_memory_tool.get_config_dir", return_value=memory_dir):
            prompt = build_runtime_system_prompt(cwd=str(tmp_path))

        _, dynamic = prompt.split(SYSTEM_PROMPT_DYNAMIC_BOUNDARY)
        assert "Memory" in dynamic

    def test_explicit_memory_content_takes_precedence(self, tmp_path: Path) -> None:
        prompt = build_runtime_system_prompt(
            cwd=str(tmp_path),
            memory_content="Custom memory content here",
        )
        _, dynamic = prompt.split(SYSTEM_PROMPT_DYNAMIC_BOUNDARY)
        assert "Custom memory content here" in dynamic

    def test_empty_memory_files_no_section(self, tmp_path: Path) -> None:
        # Empty config dir — no MEMORY.md or USER.md
        with patch("prometheus.context.prompt_assembler.get_config_dir", return_value=tmp_path):
            prompt = build_runtime_system_prompt(cwd=str(tmp_path))
        _, dynamic = prompt.split(SYSTEM_PROMPT_DYNAMIC_BOUNDARY)
        # "# Memory" section should not appear if both files are empty
        assert "# Memory" not in dynamic


# ---------------------------------------------------------------------------
# format_memory_for_prompt
# ---------------------------------------------------------------------------


class TestFormatMemoryForPrompt:
    def test_format_returns_both_sections(self, memory_dir: Path) -> None:
        with patch("prometheus.memory.hermes_memory_tool.get_config_dir", return_value=memory_dir):
            from prometheus.memory.hermes_memory_tool import format_memory_for_prompt
            content = format_memory_for_prompt()
        assert "Memory" in content
        assert "User Model" in content

    def test_format_empty_files_returns_empty(self, tmp_path: Path) -> None:
        # Create empty files
        (tmp_path / "MEMORY.md").write_text("", encoding="utf-8")
        (tmp_path / "USER.md").write_text("", encoding="utf-8")
        with patch("prometheus.memory.hermes_memory_tool.get_config_dir", return_value=tmp_path):
            from prometheus.memory.hermes_memory_tool import format_memory_for_prompt
            content = format_memory_for_prompt()
        assert content == ""


# ---------------------------------------------------------------------------
# Token budget awareness
# ---------------------------------------------------------------------------


class TestTokenBudget:
    def test_bootstrap_files_within_budget(self) -> None:
        """SOUL.md and AGENTS.md should stay under ~800 and ~400 tokens respectively.

        Rough estimate: 1 token ~ 4 chars for English text.
        """
        soul_path = Path.home() / ".prometheus" / "SOUL.md"
        agents_path = Path.home() / ".prometheus" / "AGENTS.md"

        if soul_path.exists():
            soul_chars = len(soul_path.read_text(encoding="utf-8"))
            # ~800 tokens * 4 chars/token = ~3200 chars, allow some margin
            assert soul_chars < 4000, f"SOUL.md too large: {soul_chars} chars (~{soul_chars // 4} tokens)"

        if agents_path.exists():
            agents_chars = len(agents_path.read_text(encoding="utf-8"))
            # ~400 tokens * 4 chars/token = ~1600 chars, allow some margin
            assert agents_chars < 2400, f"AGENTS.md too large: {agents_chars} chars (~{agents_chars // 4} tokens)"
