"""Tests for context / prompt assembly: system_prompt, prompt_assembler, prometheusmd."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from prometheus.context.system_prompt import (
    SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
    build_system_prompt,
)
from prometheus.context.prompt_assembler import build_runtime_system_prompt
from prometheus.context.prometheusmd import (
    CONVENTION_FILES,
    discover_project_files,
    discover_prometheus_md_files,
)


class TestSystemPromptBoundary:
    """SYSTEM_PROMPT_DYNAMIC_BOUNDARY is a distinct separator."""

    def test_boundary_present(self):
        assert isinstance(SYSTEM_PROMPT_DYNAMIC_BOUNDARY, str)
        assert len(SYSTEM_PROMPT_DYNAMIC_BOUNDARY) > 10

    def test_boundary_contains_marker(self):
        assert "DYNAMIC_BOUNDARY" in SYSTEM_PROMPT_DYNAMIC_BOUNDARY


class TestBuildSystemPrompt:
    """build_system_prompt composes static system prompt with environment info."""

    def test_includes_environment_section(self):
        prompt = build_runtime_system_prompt(cwd=".", config={})
        assert "# Environment" in prompt
        assert "OS:" in prompt

    def test_boundary_in_runtime_prompt(self):
        prompt = build_runtime_system_prompt(cwd=".", config={})
        # Dynamic boundary is injected by runtime assembler, not build_system_prompt
        assert SYSTEM_PROMPT_DYNAMIC_BOUNDARY in prompt

    def test_custom_prompt_overrides_base(self):
        custom = "# Custom Prompt\nThis is my custom system prompt."
        prompt = build_system_prompt(custom_prompt=custom)
        assert "This is my custom system prompt" in prompt
        # Custom prompt replaces the base identity section
        assert "# Custom Prompt" in prompt

    def test_environment_has_shell(self):
        prompt = build_runtime_system_prompt(cwd=".", config={})
        assert "Shell:" in prompt

    def test_environment_has_python_version(self):
        prompt = build_runtime_system_prompt(cwd=".", config={})
        assert "Python:" in prompt


class TestRuntimePromptAssembler:
    """build_runtime_system_prompt combines static + dynamic sections."""

    def test_assembles_full_prompt(self):
        prompt = build_runtime_system_prompt(cwd=".", config={})
        assert prompt is not None
        assert len(prompt) > 100

    def test_boundary_in_runtime_prompt(self):
        prompt = build_runtime_system_prompt(cwd=".", config={})
        # Dynamic boundary is injected by runtime assembler, not build_system_prompt
        assert SYSTEM_PROMPT_DYNAMIC_BOUNDARY in prompt

    def test_includes_reasoning_settings(self):
        prompt = build_runtime_system_prompt(cwd=".", config={})
        assert "# Reasoning Settings" in prompt

    def test_custom_effort_setting(self):
        prompt = build_runtime_system_prompt(cwd=".", config={"effort": "high"})
        assert "Effort: high" in prompt

    def test_custom_passes_setting(self):
        prompt = build_runtime_system_prompt(cwd=".", config={"passes": 3})
        assert "Passes: 3" in prompt

    def test_includes_memory_etiquette(self):
        prompt = build_runtime_system_prompt(cwd=".", config={})
        # Should include memory etiquette or memory section
        assert "# Memory" in prompt or "memory" in prompt.lower()

    def test_memory_content_override(self):
        custom_memory = "# Custom Memory\n\nFact: User likes coffee"
        prompt = build_runtime_system_prompt(
            cwd=".", config={}, memory_content=custom_memory
        )
        assert "User likes coffee" in prompt

    def test_task_state_injected(self):
        prompt = build_runtime_system_prompt(
            cwd=".", config={}, task_state="Working on feature X"
        )
        assert "Working on feature X" in prompt
        assert "# Current Task State" in prompt


class TestConventionFileDiscovery:
    """discover_project_files finds convention files walking upward."""

    def test_no_files_found(self, tmp_path):
        files = discover_project_files(str(tmp_path))
        assert files == []

    def test_finds_prometheus_md(self, tmp_path):
        (tmp_path / "PROMETHEUS.md").write_text("# Test Project\n")
        files = discover_project_files(str(tmp_path))
        assert len(files) == 1
        assert files[0][0].name == "PROMETHEUS.md"

    def test_finds_claude_md_when_no_prometheus(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("# Claude Project\n")
        files = discover_project_files(str(tmp_path))
        assert len(files) == 1
        assert files[0][0].name == "CLAUDE.md"

    def test_finds_hermes_md(self, tmp_path):
        (tmp_path / "HERMES.md").write_text("# Hermes Project\n")
        files = discover_project_files(str(tmp_path))
        assert len(files) == 1
        assert files[0][0].name == "HERMES.md"

    def test_finds_agents_md(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("# Agents Project\n")
        files = discover_project_files(str(tmp_path))
        assert len(files) == 1
        assert files[0][0].name == "AGENTS.md"

    def test_finds_cursorrules(self, tmp_path):
        (tmp_path / ".cursorrules").write_text("# Cursor rules\n")
        files = discover_project_files(str(tmp_path))
        assert len(files) == 1
        assert files[0][0].name == ".cursorrules"

    def test_finds_windsurfrules(self, tmp_path):
        (tmp_path / ".windsurfrules").write_text("# Windsurf rules\n")
        files = discover_project_files(str(tmp_path))
        assert len(files) == 1
        assert files[0][0].name == ".windsurfrules"

    def test_legacy_alias_returns_paths_only(self, tmp_path):
        (tmp_path / "PROMETHEUS.md").write_text("# Test\n")
        paths = discover_prometheus_md_files(str(tmp_path))
        assert len(paths) == 1
        assert isinstance(paths[0], Path)
        assert paths[0].name == "PROMETHEUS.md"

    def test_legacy_alias_first_match_wins(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("# Parent\n")
        child = tmp_path / "sub"
        child.mkdir()
        (child / "PROMETHEUS.md").write_text("# Child\n")
        paths = discover_prometheus_md_files(str(child))
        assert len(paths) == 1
        assert paths[0].name == "PROMETHEUS.md"


class TestStackingMode:
    """Stacking mode collects files from all directory levels."""

    def test_stack_collects_parent_and_child(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("# Parent\n")
        child = tmp_path / "sub"
        child.mkdir()
        (child / "PROMETHEUS.md").write_text("# Child\n")

        files = discover_project_files(str(child), stack=True)
        assert len(files) == 2
        assert files[0][0].name == "PROMETHEUS.md"
        assert files[1][0].name == "CLAUDE.md"

    def test_stack_prevents_duplicate(self, tmp_path):
        (tmp_path / "PROMETHEUS.md").write_text("# Parent\n")
        child = tmp_path / "sub"
        child.mkdir()
        (child / "PROMETHEUS.md").write_text("# Child\n")

        files = discover_project_files(str(child), stack=True)
        assert len(files) == 2
        assert files[0][0].parent.name == "sub"
        assert files[1][0].parent.name == tmp_path.name

    def test_legacy_no_stack_stops_at_first(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("# Parent\n")
        child = tmp_path / "sub"
        child.mkdir()
        (child / "PROMETHEUS.md").write_text("# Child\n")

        files = discover_project_files(str(child), stack=False)
        assert len(files) == 1
        assert files[0][0].name == "PROMETHEUS.md"

    def test_prometheus_beats_claude_at_same_level(self, tmp_path):
        (tmp_path / "PROMETHEUS.md").write_text("# Primary\n")
        (tmp_path / "CLAUDE.md").write_text("# Secondary\n")
        files = discover_project_files(str(tmp_path))
        assert len(files) == 1
        assert files[0][0].name == "PROMETHEUS.md"


class TestPrometheusRules:
    """Discovery of .prometheus/rules/*.md files."""

    def test_rules_collected(self, tmp_path):
        rules_dir = tmp_path / ".prometheus" / "rules"
        rules_dir.mkdir(parents=True)
        (rules_dir / "security.md").write_text("# Security Rules\n")
        (rules_dir / "naming.md").write_text("# Naming Rules\n")

        files = discover_project_files(str(tmp_path))
        assert len(files) == 2
        names = [f[0].name for f in files]
        assert "security.md" in names
        assert "naming.md" in names

    def test_rules_sorted(self, tmp_path):
        rules_dir = tmp_path / ".prometheus" / "rules"
        rules_dir.mkdir(parents=True)
        (rules_dir / "zzz.md").write_text("z")
        (rules_dir / "aaa.md").write_text("a")

        files = discover_project_files(str(tmp_path))
        names = [f[0].name for f in files]
        assert names == ["aaa.md", "zzz.md"]

    def test_rules_stacked_with_convention_file(self, tmp_path):
        (tmp_path / "PROMETHEUS.md").write_text("# Main\n")
        rules_dir = tmp_path / ".prometheus" / "rules"
        rules_dir.mkdir(parents=True)
        (rules_dir / "security.md").write_text("# Security\n")

        files = discover_project_files(str(tmp_path))
        assert len(files) == 2
        names = [f[0].name for f in files]
        assert "PROMETHEUS.md" in names
        assert "security.md" in names


class TestRuntimePromptIncludesProjectFiles:
    """build_runtime_system_prompt includes project files in dynamic section."""

    def test_project_file_loaded(self, tmp_path, monkeypatch):
        """When in a project dir, PROMETHEUS.md appears in the prompt."""
        (tmp_path / "PROMETHEUS.md").write_text("# My Project\n- Rule 1\n")
        # Mock the cwd used by the assembler
        monkeypatch.setattr(
            "prometheus.context.prompt_assembler.Path.cwd",
            lambda: tmp_path,
        )
        prompt = build_runtime_system_prompt(config={}, cwd=str(tmp_path))
        assert "# My Project" in prompt

    def test_legacy_no_stack_via_config(self, tmp_path, monkeypatch):
        """stack_project_files=False reverts to first-match-wins."""
        (tmp_path / "CLAUDE.md").write_text("# Parent\n")
        child = tmp_path / "sub"
        child.mkdir()
        (child / "PROMETHEUS.md").write_text("# Child\n")

        prompt = build_runtime_system_prompt(
            config={"context": {"stack_project_files": False}},
            cwd=str(child),
        )
        assert "# Child" in prompt
        assert "# Parent" not in prompt


class TestSkillsSection:
    """The skills hint is emitted only when a skills list is supplied."""

    def test_skills_hint_emitted_when_skills_present(self):
        skills = [{"name": "commit", "description": "Git commits"}]
        prompt = build_runtime_system_prompt(cwd=".", config={}, skills=skills)
        assert "tool_search" in prompt
        assert "skill tool" in prompt

    def test_no_skills_hint_when_empty(self):
        prompt = build_runtime_system_prompt(cwd=".", config={}, skills=None)
        assert "tool_search to find skills" not in prompt

    def test_daemon_call_sites_pass_skills(self):
        # Wiring guard (AST): every daemon prompt-build site must pass
        # skills=skills_for_prompt() so gateway conversations learn the
        # skill library exists. Regression for the CLI-passes/daemon-
        # doesn't gap.
        import ast

        src = (Path(__file__).parent.parent / "src" / "prometheus" / "daemon.py").read_text()
        tree = ast.parse(src)
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "build_runtime_system_prompt"
        ]
        assert len(calls) == 4, f"expected 4 daemon prompt-build sites, found {len(calls)}"
        for call in calls:
            kw_names = [kw.arg for kw in call.keywords]
            assert "skills" in kw_names, "daemon prompt-build site missing skills= argument"

class TestSkillsTiering:
    """Item 1+2: core skills render inline, tail stays tool_search-only."""

    def _prompt(self, skills):
        return build_runtime_system_prompt(cwd=".", config={}, skills=skills)

    def test_core_skills_render_name_and_description(self):
        skills = [
            {"name": "commit", "description": "Git commit workflow", "core": True},
            {"name": "debug", "description": "Debug workflow", "core": True},
            {"name": "aider", "description": "Reachable on demand", "core": False},
        ]
        prompt = self._prompt(skills)
        assert "# Available Skills" in prompt
        assert "**commit**: Git commit workflow" in prompt
        assert "**debug**: Debug workflow" in prompt
        # tail skill NOT rendered inline — only reachable via tool_search
        assert "aider" not in prompt
        assert "tool_search" in prompt
        assert "1 additional skill available on demand" in prompt

    def test_no_core_skills_keeps_count_only_hint(self):
        skills = [{"name": "x", "description": "d", "core": False}]
        prompt = self._prompt(skills)
        assert "**x**" not in prompt
        assert "1 additional skill available on demand" in prompt
        assert "tool_search" in prompt

    def test_all_core_no_count_hint(self):
        skills = [{"name": "commit", "description": "d", "core": True}]
        prompt = self._prompt(skills)
        assert "**commit**: d" in prompt
        assert "additional skill" not in prompt
        assert "tool_search" not in prompt

    def test_skills_dicts_without_core_key_treated_as_tail(self):
        # Back-compat: callers passing legacy {"name","description"} dicts
        # must not crash and must land in the tool_search tier.
        skills = [{"name": "legacy", "description": "old shape"}]
        prompt = self._prompt(skills)
        assert "legacy" not in prompt.split("## Core")[0] if "## Core" in prompt else True
        assert "tool_search" in prompt

    def test_daemon_built_prompt_contains_skills_section(self):
        """Admission guard (item 5): the real registry feeds the assembler
        and the rendered prompt carries a skills section with the builtin
        core skills visible."""
        from prometheus.skills.loader import skills_for_prompt

        entries = skills_for_prompt()
        assert entries is not None
        prompt = self._prompt(entries)
        assert "# Available Skills" in prompt
        assert "**commit**" in prompt and "**debug**" in prompt and "**plan**" in prompt

