"""Tests for project file discovery: prometheusmd module.

Covers:
  - Single PROMETHEUS.md discovery
  - Multi-agent convention file recognition (HERMES.md, CLAUDE.md, etc.)
  - Stacking mode (collect from all directory levels)
  - Legacy first-match-wins mode
  - .prometheus/rules/*.md collection
  - Deduplication
  - Priority ordering (PROMETHEUS.md beats CLAUDE.md at same level)
"""

from pathlib import Path

import pytest

from prometheus.context.prometheusmd import (
    CONVENTION_FILES,
    discover_project_files,
    discover_prometheus_md_files,
    load_project_files_prompt,
)


class TestConventionFiles:
    """Verify CONVENTION_FILES has the expected entries."""

    def test_has_prometheus(self):
        assert "PROMETHEUS.md" in CONVENTION_FILES

    def test_has_hermes(self):
        assert "HERMES.md" in CONVENTION_FILES
        assert ".hermes.md" in CONVENTION_FILES

    def test_has_claude(self):
        assert "CLAUDE.md" in CONVENTION_FILES

    def test_has_agents(self):
        assert "AGENTS.md" in CONVENTION_FILES

    def test_has_cursor(self):
        assert ".cursorrules" in CONVENTION_FILES

    def test_has_windsurf(self):
        assert ".windsurfrules" in CONVENTION_FILES

    def test_prometheus_is_primary(self):
        assert CONVENTION_FILES[0] == "PROMETHEUS.md"


class TestDiscoverProjectFiles:
    """discover_project_files walks upward collecting convention files."""

    def test_empty_directory(self, tmp_path):
        files = discover_project_files(str(tmp_path))
        assert files == []

    def test_single_prometheus_md(self, tmp_path):
        (tmp_path / "PROMETHEUS.md").write_text("# Test Project\n")
        files = discover_project_files(str(tmp_path))
        assert len(files) == 1
        assert files[0][0].name == "PROMETHEUS.md"

    def test_prometheus_beats_claude_at_same_level(self, tmp_path):
        (tmp_path / "PROMETHEUS.md").write_text("# Primary\n")
        (tmp_path / "CLAUDE.md").write_text("# Secondary\n")
        files = discover_project_files(str(tmp_path))
        # Only one file per directory level
        assert len(files) == 1
        assert files[0][0].name == "PROMETHEUS.md"

    def test_claude_only_when_no_prometheus(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("# Just Claude\n")
        files = discover_project_files(str(tmp_path))
        assert len(files) == 1
        assert files[0][0].name == "CLAUDE.md"

    def test_hermes_md_recognized(self, tmp_path):
        (tmp_path / "HERMES.md").write_text("# Hermes\n")
        files = discover_project_files(str(tmp_path))
        assert len(files) == 1
        assert files[0][0].name == "HERMES.md"

    def test_cursorrules_recognized(self, tmp_path):
        (tmp_path / ".cursorrules").write_text("# Cursor\n")
        files = discover_project_files(str(tmp_path))
        assert len(files) == 1
        assert files[0][0].name == ".cursorrules"

    def test_agents_md_recognized(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("# Agents\n")
        files = discover_project_files(str(tmp_path))
        assert len(files) == 1
        assert files[0][0].name == "AGENTS.md"


class TestStackingMode:
    """Stacking mode collects files from all directory levels."""

    def test_stack_collects_from_parent_and_child(self, tmp_path):
        # Parent has CLAUDE.md, child has PROMETHEUS.md
        (tmp_path / "CLAUDE.md").write_text("# Parent\n")
        child = tmp_path / "sub"
        child.mkdir()
        (child / "PROMETHEUS.md").write_text("# Child\n")

        files = discover_project_files(str(child), stack=True)
        assert len(files) == 2
        # Most specific first
        assert files[0][0].name == "PROMETHEUS.md"
        assert files[1][0].name == "CLAUDE.md"

    def test_stack_prevents_duplicate(self, tmp_path):
        # Same file at parent, child has its own
        (tmp_path / "PROMETHEUS.md").write_text("# Parent\n")
        child = tmp_path / "sub"
        child.mkdir()
        (child / "PROMETHEUS.md").write_text("# Child\n")

        files = discover_project_files(str(child), stack=True)
        assert len(files) == 2
        assert files[0][0].name == "PROMETHEUS.md"
        assert files[1][0].name == "PROMETHEUS.md"
        # Different paths
        assert files[0][0].parent.name == "sub"
        assert files[1][0].parent.name == tmp_path.name

    def test_legacy_no_stack_stops_at_first(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("# Parent\n")
        child = tmp_path / "sub"
        child.mkdir()
        (child / "PROMETHEUS.md").write_text("# Child\n")

        files = discover_project_files(str(child), stack=False)
        assert len(files) == 1
        assert files[0][0].name == "PROMETHEUS.md"  # Child found first (deepest)


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


class TestLegacyAlias:
    """discover_prometheus_md_files returns paths only, first-match-wins."""

    def test_returns_paths_only(self, tmp_path):
        (tmp_path / "PROMETHEUS.md").write_text("# Test\n")
        paths = discover_prometheus_md_files(str(tmp_path))
        assert len(paths) == 1
        assert isinstance(paths[0], Path)
        assert paths[0].name == "PROMETHEUS.md"

    def test_first_match_wins(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("# Parent\n")
        child = tmp_path / "sub"
        child.mkdir()
        (child / "PROMETHEUS.md").write_text("# Child\n")

        paths = discover_prometheus_md_files(str(child))
        assert len(paths) == 1
        assert paths[0].name == "PROMETHEUS.md"


class TestLoadPrompt:
    """load_project_files_prompt formats files into system prompt text."""

    def test_none_when_empty(self, tmp_path):
        result = load_project_files_prompt(str(tmp_path))
        assert result is None

    def test_includes_directory_label(self, tmp_path):
        (tmp_path / "PROMETHEUS.md").write_text("# Test Project\nRule 1\n")
        result = load_project_files_prompt(str(tmp_path))
        assert result is not None
        assert "# Project Instructions" in result
        assert "```md" in result
        assert "Rule 1" in result

    def test_stack_mode_labels_multiple(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("# Parent\n")
        child = tmp_path / "sub"
        child.mkdir()
        (child / "PROMETHEUS.md").write_text("# Child\n")

        result = load_project_files_prompt(str(child), stack=True)
        assert result is not None
        assert "Parent" in result
        assert "Child" in result
        assert "```md" in result

    def test_truncates_long_files(self, tmp_path):
        (tmp_path / "PROMETHEUS.md").write_text("# " + "x" * 20000)
        result = load_project_files_prompt(str(tmp_path), max_chars_per_file=1000)
        assert result is not None
        assert "...[truncated]..." in result


class TestLabelGeneration:
    """Labels are directory names for context."""

    def test_label_is_directory_name(self, tmp_path):
        (tmp_path / "PROMETHEUS.md").write_text("# Test\n")
        files = discover_project_files(str(tmp_path))
        assert files[0][1] == tmp_path.name

    def test_label_in_stacked_result(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text("# Parent\n")
        child = tmp_path / "myproject"
        child.mkdir()
        (child / "PROMETHEUS.md").write_text("# Child\n")

        files = discover_project_files(str(child), stack=True)
        labels = {f[0].name: f[1] for f in files}
        assert labels["PROMETHEUS.md"] == "myproject"
        assert labels["CLAUDE.md"] == tmp_path.name


class TestExcludedConventions:
    """`exclude` keeps a name that another config already gates out.

    AGENTS.md is the live case: `bootstrap.load_agents: false` turns the agent
    registry off, and convention-file discovery would otherwise hand it straight
    back the moment an AGENTS.md sits at or above cwd.
    """

    def test_excluded_name_is_not_discovered(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("# Agent Registry\n")
        assert discover_project_files(str(tmp_path)) != []
        assert discover_project_files(str(tmp_path), exclude=("AGENTS.md",)) == []

    def test_exclude_is_case_insensitive(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("# Agent Registry\n")
        assert discover_project_files(str(tmp_path), exclude=("agents.md",)) == []

    def test_exclude_leaves_other_conventions_alone(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("# Agent Registry\n")
        (tmp_path / "PROMETHEUS.md").write_text("# Project\n")
        files = discover_project_files(str(tmp_path), exclude=("AGENTS.md",))
        assert [p.name for p, _ in files] == ["PROMETHEUS.md"]

    def test_excluded_name_absent_from_prompt(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("# Agent Registry\n")
        assert load_project_files_prompt(str(tmp_path), exclude=("AGENTS.md",)) is None
