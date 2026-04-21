"""Tests for agent profiles — ProfileStore, AgentProfile, filter_tools_by_profile."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from prometheus.config.profiles import (
    AgentProfile,
    ProfileStore,
    filter_tools_by_profile,
    get_profile_store,
)


# ---------------------------------------------------------------------------
# Builtin profiles
# ---------------------------------------------------------------------------


class TestBuiltinProfiles:
    def test_all_builtins_load(self) -> None:
        store = ProfileStore(custom_dir=Path("/tmp/empty_profiles_dir_test"))
        names = store.names()
        assert "full" in names
        assert "coder" in names
        assert "research" in names
        assert "assistant" in names
        assert "minimal" in names

    def test_full_profile_has_all_tools(self) -> None:
        store = ProfileStore(custom_dir=Path("/tmp/empty_profiles_dir_test"))
        full = store.get("full")
        assert full is not None
        assert full.tools is None  # None = all tools

    def test_coder_profile_has_limited_tools(self) -> None:
        store = ProfileStore(custom_dir=Path("/tmp/empty_profiles_dir_test"))
        coder = store.get("coder")
        assert coder is not None
        assert coder.tools is not None
        assert "bash" in coder.tools
        assert "file_read" in coder.tools
        assert "wiki_query" not in coder.tools

    def test_minimal_profile_has_fewest_tools(self) -> None:
        store = ProfileStore(custom_dir=Path("/tmp/empty_profiles_dir_test"))
        minimal = store.get("minimal")
        assert minimal is not None
        assert minimal.tools == ["bash", "file_read"]

    def test_research_profile_excludes_mutations(self) -> None:
        store = ProfileStore(custom_dir=Path("/tmp/empty_profiles_dir_test"))
        research = store.get("research")
        assert research is not None
        assert "file_write" not in (research.tools or [])
        assert "bash" not in (research.tools or [])

    def test_each_profile_has_description(self) -> None:
        store = ProfileStore(custom_dir=Path("/tmp/empty_profiles_dir_test"))
        for profile in store.list_profiles():
            assert profile.description, f"{profile.name} has no description"


# ---------------------------------------------------------------------------
# Custom profiles
# ---------------------------------------------------------------------------


class TestCustomProfiles:
    def test_custom_profile_loads_from_yaml(self, tmp_path: Path) -> None:
        custom_dir = tmp_path / "profiles"
        custom_dir.mkdir()
        (custom_dir / "myprofile.yaml").write_text(
            "name: myprofile\n"
            "description: Custom test profile\n"
            "bootstrap_files:\n  - SOUL.md\n"
            "tools:\n  - bash\n  - grep\n",
            encoding="utf-8",
        )
        store = ProfileStore(custom_dir=custom_dir)
        p = store.get("myprofile")
        assert p is not None
        assert p.description == "Custom test profile"
        assert p.tools == ["bash", "grep"]
        assert p.bootstrap_files == ["SOUL.md"]

    def test_custom_overrides_builtin(self, tmp_path: Path) -> None:
        custom_dir = tmp_path / "profiles"
        custom_dir.mkdir()
        (custom_dir / "coder.yaml").write_text(
            "name: coder\n"
            "description: My custom coder\n"
            "tools:\n  - bash\n",
            encoding="utf-8",
        )
        store = ProfileStore(custom_dir=custom_dir)
        coder = store.get("coder")
        assert coder is not None
        assert coder.description == "My custom coder"
        assert coder.tools == ["bash"]

    def test_invalid_yaml_skipped(self, tmp_path: Path) -> None:
        custom_dir = tmp_path / "profiles"
        custom_dir.mkdir()
        (custom_dir / "bad.yaml").write_text("not: valid: yaml: [[", encoding="utf-8")
        store = ProfileStore(custom_dir=custom_dir)
        # Should load builtins without crashing
        assert len(store.names()) >= 5

    def test_missing_name_skipped(self, tmp_path: Path) -> None:
        custom_dir = tmp_path / "profiles"
        custom_dir.mkdir()
        (custom_dir / "noname.yaml").write_text(
            "description: Missing name field\n", encoding="utf-8"
        )
        store = ProfileStore(custom_dir=custom_dir)
        assert store.get("noname") is None


# ---------------------------------------------------------------------------
# filter_tools_by_profile
# ---------------------------------------------------------------------------


class TestFilterToolsByProfile:
    def _schemas(self, *names: str) -> list[dict]:
        return [{"name": n, "description": f"{n} tool"} for n in names]

    def test_none_tools_returns_all(self) -> None:
        profile = AgentProfile(name="full", tools=None)
        schemas = self._schemas("bash", "grep", "wiki_query")
        filtered = filter_tools_by_profile(schemas, profile)
        assert len(filtered) == 3

    def test_tools_list_filters(self) -> None:
        profile = AgentProfile(name="coder", tools=["bash", "grep"])
        schemas = self._schemas("bash", "grep", "wiki_query", "file_read")
        filtered = filter_tools_by_profile(schemas, profile)
        assert [s["name"] for s in filtered] == ["bash", "grep"]

    def test_exclude_tools(self) -> None:
        profile = AgentProfile(name="test", tools=None, exclude_tools=["wiki_query"])
        schemas = self._schemas("bash", "grep", "wiki_query")
        filtered = filter_tools_by_profile(schemas, profile)
        assert [s["name"] for s in filtered] == ["bash", "grep"]

    def test_include_and_exclude(self) -> None:
        profile = AgentProfile(
            name="test",
            tools=["bash", "grep", "wiki_query"],
            exclude_tools=["wiki_query"],
        )
        schemas = self._schemas("bash", "grep", "wiki_query", "file_read")
        filtered = filter_tools_by_profile(schemas, profile)
        assert [s["name"] for s in filtered] == ["bash", "grep"]

    def test_max_tool_schemas(self) -> None:
        profile = AgentProfile(name="test", tools=None, max_tool_schemas=2)
        schemas = self._schemas("a", "b", "c", "d")
        filtered = filter_tools_by_profile(schemas, profile)
        assert len(filtered) == 2


# ---------------------------------------------------------------------------
# Profile controls bootstrap file loading
# ---------------------------------------------------------------------------


class TestProfileBootstrapFiles:
    def test_profile_controls_bootstrap_in_prompt(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "SOUL.md").write_text("# Soul Content", encoding="utf-8")
        (config_dir / "AGENTS.md").write_text("# Agents Content", encoding="utf-8")
        (config_dir / "ANATOMY.md").write_text(
            "# Anatomy\n\n## Active Configuration\nSome infra\n",
            encoding="utf-8",
        )

        # Profile that only loads SOUL.md
        profile = AgentProfile(name="lean", bootstrap_files=["SOUL.md"])

        with patch("prometheus.context.prompt_assembler.get_config_dir", return_value=config_dir):
            from prometheus.context.prompt_assembler import build_runtime_system_prompt
            from prometheus.context.system_prompt import SYSTEM_PROMPT_DYNAMIC_BOUNDARY

            prompt = build_runtime_system_prompt(cwd=str(tmp_path), profile=profile)

        static, _ = prompt.split(SYSTEM_PROMPT_DYNAMIC_BOUNDARY)
        assert "Soul Content" in static
        assert "Agents Content" not in static

    def test_full_profile_loads_all(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "SOUL.md").write_text("# Soul", encoding="utf-8")
        (config_dir / "AGENTS.md").write_text("# Agents", encoding="utf-8")

        profile = AgentProfile(name="full", bootstrap_files=["SOUL.md", "AGENTS.md"])

        with patch("prometheus.context.prompt_assembler.get_config_dir", return_value=config_dir):
            from prometheus.context.prompt_assembler import build_runtime_system_prompt
            from prometheus.context.system_prompt import SYSTEM_PROMPT_DYNAMIC_BOUNDARY

            prompt = build_runtime_system_prompt(cwd=str(tmp_path), profile=profile)

        static, _ = prompt.split(SYSTEM_PROMPT_DYNAMIC_BOUNDARY)
        assert "# Soul" in static
        assert "# Agents" in static

    def test_no_profile_uses_legacy_config(self, tmp_path: Path) -> None:
        """Without a profile, legacy bootstrap config toggles work."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "SOUL.md").write_text("# Soul", encoding="utf-8")

        config = {"bootstrap": {"load_soul": False}}
        with patch("prometheus.context.prompt_assembler.get_config_dir", return_value=config_dir):
            from prometheus.context.prompt_assembler import build_runtime_system_prompt

            prompt = build_runtime_system_prompt(cwd=str(tmp_path), config=config)

        assert "# Soul" not in prompt


# ---------------------------------------------------------------------------
# /profile command
# ---------------------------------------------------------------------------


class TestCmdProfile:
    def test_show_profiles(self) -> None:
        from prometheus.gateway.commands import cmd_profile
        text = cmd_profile()
        assert "full" in text
        assert "coder" in text
        assert "Available profiles:" in text

    def test_switch_profile(self) -> None:
        from prometheus.gateway.commands import cmd_profile
        text = cmd_profile(arg="coder")
        assert "Switched to: coder" in text
        assert "bash" in text

    def test_unknown_profile(self) -> None:
        from prometheus.gateway.commands import cmd_profile
        text = cmd_profile(arg="nonexistent")
        assert "Unknown profile" in text
        assert "full" in text  # suggests available

    def test_shows_current_profile(self) -> None:
        from prometheus.gateway.commands import cmd_profile
        text = cmd_profile(current="coder")
        assert "Current profile: coder" in text
