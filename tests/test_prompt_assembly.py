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
from prometheus.context.prometheusmd import discover_prometheus_md_files
from prometheus.context.environment import EnvironmentInfo, get_environment_info


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    def test_build_system_prompt_contains_identity(self) -> None:
        prompt = build_system_prompt(cwd=os.getcwd())
        assert "Prometheus" in prompt

    def test_system_prompt_dynamic_boundary(self) -> None:
        # The boundary constant should be the expected marker string
        assert "SYSTEM_PROMPT_DYNAMIC_BOUNDARY" in SYSTEM_PROMPT_DYNAMIC_BOUNDARY
        assert SYSTEM_PROMPT_DYNAMIC_BOUNDARY == "--- SYSTEM_PROMPT_DYNAMIC_BOUNDARY ---"


# ---------------------------------------------------------------------------
# build_runtime_system_prompt
# ---------------------------------------------------------------------------


class TestBuildRuntimeSystemPrompt:
    def test_build_runtime_system_prompt_structure(self, tmp_path: Path) -> None:
        prompt = build_runtime_system_prompt(cwd=str(tmp_path))

        # Should contain the static section (identity)
        assert "Prometheus" in prompt

        # Should contain the dynamic boundary
        assert SYSTEM_PROMPT_DYNAMIC_BOUNDARY in prompt

        # Should have content after the boundary (at minimum the reasoning settings)
        parts = prompt.split(SYSTEM_PROMPT_DYNAMIC_BOUNDARY)
        assert len(parts) == 2
        static_part, dynamic_part = parts
        assert len(static_part.strip()) > 0
        assert len(dynamic_part.strip()) > 0

        # The dynamic section should include reasoning settings
        assert "Reasoning Settings" in dynamic_part or "Effort" in dynamic_part


# ---------------------------------------------------------------------------
# PROMETHEUS.md discovery
# ---------------------------------------------------------------------------


class TestPrometheusMdDiscovery:
    def test_prometheus_md_discovery(self, tmp_path: Path) -> None:
        # Create a PROMETHEUS.md in the temp directory
        pmd = tmp_path / "PROMETHEUS.md"
        pmd.write_text("# Project Rules\n\nAlways use type hints.\n")

        files = discover_prometheus_md_files(str(tmp_path))
        assert any(f == pmd for f in files)

    def test_prometheus_md_nested_discovery(self, tmp_path: Path) -> None:
        # Create a nested directory structure with PROMETHEUS.md at multiple levels
        parent_pmd = tmp_path / "PROMETHEUS.md"
        parent_pmd.write_text("# Parent rules\n")

        child_dir = tmp_path / "subdir"
        child_dir.mkdir()
        child_pmd = child_dir / "PROMETHEUS.md"
        child_pmd.write_text("# Child rules\n")

        # Discover from the child dir -- should find both
        files = discover_prometheus_md_files(str(child_dir))
        paths = [f.resolve() for f in files]
        assert child_pmd.resolve() in paths
        assert parent_pmd.resolve() in paths

        # Child (more specific) should come before parent (less specific)
        child_idx = paths.index(child_pmd.resolve())
        parent_idx = paths.index(parent_pmd.resolve())
        assert child_idx < parent_idx


# ---------------------------------------------------------------------------
# Environment info
# ---------------------------------------------------------------------------


class TestEnvironmentInfo:
    def test_environment_info(self) -> None:
        env = get_environment_info()
        assert isinstance(env, EnvironmentInfo)
        assert env.os_name  # non-empty
        assert env.cwd  # non-empty
        assert env.shell  # non-empty
        assert env.python_version  # non-empty
        assert env.date  # non-empty, e.g. "2026-04-04"
        assert env.platform_machine  # e.g. "arm64", "x86_64"
