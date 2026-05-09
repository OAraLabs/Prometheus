"""Tests for the WEAVE-PRESS Printing Press CLI registry + hook.

Pure-function coverage of the registry. Network-dependent paths
(``go install``, ``git pull``) are tested via mocked subprocesses or
filesystem fixtures — no live network calls.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prometheus.tools.printing_press import (
    CLIRecord,
    InstallResult,
    PrintingPressRegistry,
    _extract_frontmatter,
    detect_command_not_found,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_fake_library(root: Path, *clis: dict) -> Path:
    """Build a tmp directory shaped like the real printing-press-library."""
    (root / "cli-skills").mkdir(parents=True)
    for spec in clis:
        skill_dir = root / "cli-skills" / spec["skill_name"]
        skill_dir.mkdir(parents=True)
        skill_md = skill_dir / "SKILL.md"
        bin_name = spec.get("bin_name", spec["skill_name"] + "-cli")
        module = spec.get(
            "module",
            f"github.com/mvanhorn/printing-press-library/library/"
            f"{spec.get('category', 'misc')}/{spec['skill_name']}/cmd/{bin_name}",
        )
        kind = spec.get("kind", "go")
        skill_md.write_text(
            f"---\n"
            f"name: {spec['skill_name']}\n"
            f"description: \"{spec.get('description', 'A test CLI')}\"\n"
            f"metadata:\n"
            f"  openclaw:\n"
            f"    install:\n"
            f"      - kind: {kind}\n"
            f"        bins: [{bin_name}]\n"
            f"        module: {module}\n"
            f"---\n\n"
            f"# {spec['skill_name']}\n",
            encoding="utf-8",
        )
    return root


# ---------------------------------------------------------------------------
# Frontmatter parser
# ---------------------------------------------------------------------------


class TestFrontmatterParser:
    def test_extracts_simple_frontmatter(self):
        text = "---\nname: foo\ndescription: bar\n---\n\nbody"
        out = _extract_frontmatter(text)
        assert out == {"name": "foo", "description": "bar"}

    def test_no_frontmatter_returns_none(self):
        assert _extract_frontmatter("just a body, no fences") is None

    def test_unclosed_frontmatter_returns_none(self):
        assert _extract_frontmatter("---\nname: foo\nno closing fence") is None

    def test_malformed_yaml_returns_none(self):
        assert _extract_frontmatter("---\nname: : :\n---\n") is None

    def test_nested_metadata(self):
        text = (
            "---\n"
            "name: x\n"
            "metadata:\n"
            "  openclaw:\n"
            "    install:\n"
            "      - kind: go\n"
            "        bins: [x-cli]\n"
            "---\n"
        )
        out = _extract_frontmatter(text)
        assert out["metadata"]["openclaw"]["install"][0]["kind"] == "go"


# ---------------------------------------------------------------------------
# detect_command_not_found
# ---------------------------------------------------------------------------


class TestDetectCommandNotFound:
    def test_bash_line_form(self):
        assert detect_command_not_found(
            "bash: line 1: cal-com-pp-cli: command not found"
        ) == "cal-com-pp-cli"

    def test_short_form(self):
        assert detect_command_not_found(
            "slack-pp-cli: command not found"
        ) == "slack-pp-cli"

    def test_alt_colon_form(self):
        assert detect_command_not_found(
            "command not found: pp-calendar"
        ) == "pp-calendar"

    def test_clean_output_returns_none(self):
        assert detect_command_not_found("hello world\n") is None

    def test_empty_string(self):
        assert detect_command_not_found("") is None

    def test_multiline_with_other_errors(self):
        out = (
            "doing some setup\n"
            "bash: line 4: pp-roam: command not found\n"
            "exit 127\n"
        )
        assert detect_command_not_found(out) == "pp-roam"


# ---------------------------------------------------------------------------
# Registry: discovery / search / installed
# ---------------------------------------------------------------------------


class TestRegistryDiscovery:
    def test_finds_local_library(self, tmp_path):
        lib = _build_fake_library(
            tmp_path / "lib",
            {"skill_name": "pp-slack", "category": "productivity",
             "bin_name": "slack-pp-cli"},
        )
        reg = PrintingPressRegistry(library_path=lib)
        assert reg.is_available()
        assert reg.library_path == lib.resolve()

    def test_no_library_anywhere(self, tmp_path, monkeypatch):
        # Point all default search locations at empty dirs
        from prometheus.tools import printing_press as pp_mod
        monkeypatch.setattr(
            pp_mod, "_DEFAULT_LIBRARY_LOCATIONS",
            (tmp_path / "nope1", tmp_path / "nope2"),
        )
        reg = PrintingPressRegistry()
        assert not reg.is_available()
        assert reg.list_available() == []
        assert reg.search("anything") == []

    def test_lists_available_clis(self, tmp_path):
        lib = _build_fake_library(
            tmp_path / "lib",
            {"skill_name": "pp-slack", "category": "productivity",
             "bin_name": "slack-pp-cli"},
            {"skill_name": "pp-cal-com", "category": "productivity",
             "bin_name": "cal-com-pp-cli"},
            {"skill_name": "pp-roam", "category": "knowledge",
             "bin_name": "roam-pp-cli"},
        )
        reg = PrintingPressRegistry(library_path=lib)
        records = reg.list_available()
        assert len(records) == 3
        names = {r.name for r in records}
        assert names == {"slack", "cal-com", "roam"}

    def test_skips_non_go_install_kinds(self, tmp_path):
        lib = _build_fake_library(
            tmp_path / "lib",
            {"skill_name": "pp-slack", "bin_name": "slack-pp-cli", "kind": "go"},
            {"skill_name": "pp-something-npm", "bin_name": "npm-cli", "kind": "npm"},
        )
        reg = PrintingPressRegistry(library_path=lib)
        records = reg.list_available()
        # Only the go entry survives — npm install kinds are out of scope
        assert len(records) == 1
        assert records[0].name == "slack"

    def test_extracts_category_from_module_path(self, tmp_path):
        lib = _build_fake_library(
            tmp_path / "lib",
            {"skill_name": "pp-slack", "bin_name": "slack-pp-cli",
             "category": "productivity"},
        )
        reg = PrintingPressRegistry(library_path=lib)
        rec = reg.list_available()[0]
        assert rec.category == "productivity"


class TestRegistrySearch:
    @pytest.fixture
    def registry(self, tmp_path):
        lib = _build_fake_library(
            tmp_path / "lib",
            {"skill_name": "pp-cal-com", "category": "productivity",
             "bin_name": "cal-com-pp-cli",
             "description": "Cal.com calendar API"},
            {"skill_name": "pp-slack", "category": "productivity",
             "bin_name": "slack-pp-cli",
             "description": "Slack workspace messaging"},
            {"skill_name": "pp-airbnb", "category": "travel",
             "bin_name": "airbnb-pp-cli",
             "description": "Airbnb hosting CLI"},
        )
        return PrintingPressRegistry(library_path=lib)

    def test_exact_name_match(self, registry):
        out = registry.search("slack")
        assert out and out[0].name == "slack"

    def test_substring_in_description(self, registry):
        out = registry.search("calendar")
        assert out and out[0].name == "cal-com"

    def test_case_insensitive(self, registry):
        out = registry.search("SLACK")
        assert out and out[0].name == "slack"

    def test_empty_query(self, registry):
        assert registry.search("") == []
        assert registry.search("   ") == []

    def test_no_match(self, registry):
        assert registry.search("xyzzy_quux") == []

    def test_search_limit_respected(self, registry):
        out = registry.search("pp", limit=2)
        assert len(out) <= 2

    def test_search_matches_bin_name(self, registry):
        # bin name lookup
        out = registry.search("airbnb-pp-cli")
        assert out and out[0].bin_name == "airbnb-pp-cli"


class TestIsInstalled:
    def test_uses_shutil_which(self, tmp_path, monkeypatch):
        reg = PrintingPressRegistry(library_path=tmp_path / "missing")
        monkeypatch.setattr(
            "prometheus.tools.printing_press.shutil.which",
            lambda name: "/some/path" if name == "exists" else None,
        )
        assert reg.is_installed("exists")
        assert not reg.is_installed("does-not-exist")

    def test_falls_back_to_go_bin_dir(self, tmp_path, monkeypatch):
        # shutil.which returns None but binary lives in ~/go/bin/<name>
        fake_gobin = tmp_path / "go-bin"
        fake_gobin.mkdir()
        (fake_gobin / "stub-cli").write_text("#!/bin/sh\necho hi\n")
        monkeypatch.setattr(
            "prometheus.tools.printing_press._GO_BIN_DIR", fake_gobin,
        )
        monkeypatch.setattr(
            "prometheus.tools.printing_press.shutil.which",
            lambda name: None,
        )
        reg = PrintingPressRegistry(library_path=tmp_path / "missing")
        assert reg.is_installed("stub-cli")
        assert not reg.is_installed("absent-cli")

    def test_empty_name(self, tmp_path):
        reg = PrintingPressRegistry(library_path=tmp_path / "missing")
        assert not reg.is_installed("")


# ---------------------------------------------------------------------------
# Registry: install (subprocess mocked)
# ---------------------------------------------------------------------------


class TestInstall:
    @pytest.fixture
    def registry(self, tmp_path, monkeypatch):
        lib = _build_fake_library(
            tmp_path / "lib",
            {"skill_name": "pp-slack", "category": "productivity",
             "bin_name": "slack-pp-cli"},
        )
        skills_dest = tmp_path / "user-skills"
        # Ensure go is "available" for the no-go-on-PATH guard
        monkeypatch.setattr(
            "prometheus.tools.printing_press.shutil.which",
            lambda name: "/usr/bin/go" if name == "go" else None,
        )
        # Pretend ~/go/bin exists and the binary lands there
        gobin = tmp_path / "go-bin"
        gobin.mkdir()
        monkeypatch.setattr(
            "prometheus.tools.printing_press._GO_BIN_DIR", gobin,
        )
        return PrintingPressRegistry(
            library_path=lib, skills_dest=skills_dest,
        ), gobin, skills_dest

    @pytest.mark.asyncio
    async def test_install_copies_skill(self, registry):
        reg, gobin, skills_dest = registry

        async def fake_subprocess(*args, **kwargs):
            # Simulate `go install` succeeding by creating the target binary
            (gobin / "slack-pp-cli").write_text("#!/bin/sh\n")
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(b"", b""))
            proc.returncode = 0
            return proc

        with patch(
            "prometheus.tools.printing_press.asyncio.create_subprocess_exec",
            side_effect=fake_subprocess,
        ):
            result = await reg.install("slack")

        assert result.success, result.error
        assert result.cli_name == "slack"
        assert result.bin_name == "slack-pp-cli"
        assert result.skill_installed
        # Skill landed at the expected destination
        assert (skills_dest / "pp-slack.md").is_file()
        assert "name: pp-slack" in (skills_dest / "pp-slack.md").read_text()

    @pytest.mark.asyncio
    async def test_install_fires_reload_callback(self, registry):
        reg, gobin, _ = registry
        callback = MagicMock()
        reg.set_reload_callback(callback)

        async def fake_subprocess(*args, **kwargs):
            (gobin / "slack-pp-cli").write_text("#!/bin/sh\n")
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(b"", b""))
            proc.returncode = 0
            return proc

        with patch(
            "prometheus.tools.printing_press.asyncio.create_subprocess_exec",
            side_effect=fake_subprocess,
        ):
            await reg.install("slack")
        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_install_no_go_on_path_returns_error(self, tmp_path, monkeypatch):
        lib = _build_fake_library(
            tmp_path / "lib",
            {"skill_name": "pp-slack", "bin_name": "slack-pp-cli"},
        )
        # Pretend Go is NOT on PATH
        monkeypatch.setattr(
            "prometheus.tools.printing_press.shutil.which",
            lambda name: None,
        )
        reg = PrintingPressRegistry(library_path=lib)
        result = await reg.install("slack")
        assert not result.success
        assert "go" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_install_unknown_cli_returns_error(self, registry):
        reg, _, _ = registry
        result = await reg.install("absolutely-not-real")
        assert not result.success
        assert "no cli matching" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_install_subprocess_failure(self, registry):
        reg, _, _ = registry

        async def fake_subprocess(*args, **kwargs):
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(b"", b"go: errored\n"))
            proc.returncode = 1
            return proc

        with patch(
            "prometheus.tools.printing_press.asyncio.create_subprocess_exec",
            side_effect=fake_subprocess,
        ):
            result = await reg.install("slack")
        assert not result.success
        assert "go install failed" in (result.error or "")


# ---------------------------------------------------------------------------
# Hot-reload (skills/registry.py.reload_user_skills)
# ---------------------------------------------------------------------------


class TestSkillHotReload:
    def test_reload_picks_up_new_skill(self, tmp_path, monkeypatch):
        # Point ~/.prometheus/ at tmp_path so user-skills dir is isolated
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(
            "prometheus.config.paths.get_config_dir",
            lambda: tmp_path / ".prometheus",
        )
        skills_dir = tmp_path / ".prometheus" / "skills"
        skills_dir.mkdir(parents=True)

        # Build a registry with no skills initially
        from prometheus.skills.registry import SkillRegistry
        registry = SkillRegistry()
        assert registry.list_skills() == []

        # Drop a new SKILL.md after construction
        (skills_dir / "pp-slack.md").write_text(
            "---\nname: pp-slack\ndescription: testing\n---\n# pp-slack\n",
            encoding="utf-8",
        )
        added = registry.reload_user_skills()
        assert added >= 1
        names = {s.name for s in registry.list_skills()}
        assert "pp-slack" in names

    def test_reload_idempotent_for_unchanged_skills(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "prometheus.config.paths.get_config_dir",
            lambda: tmp_path / ".prometheus",
        )
        skills_dir = tmp_path / ".prometheus" / "skills"
        skills_dir.mkdir(parents=True)
        (skills_dir / "x.md").write_text(
            "---\nname: x\ndescription: y\n---\n", encoding="utf-8",
        )
        from prometheus.skills.registry import SkillRegistry
        registry = SkillRegistry()
        registry.reload_user_skills()
        # Second reload with no on-disk changes should add 0
        added = registry.reload_user_skills()
        assert added == 0


# ---------------------------------------------------------------------------
# Suggestion hook (_maybe_suggest_printing_press)
# ---------------------------------------------------------------------------


class TestSuggestionHook:
    @pytest.mark.asyncio
    async def test_suggests_when_command_not_found_and_match_exists(self, tmp_path):
        from prometheus.engine.agent_loop import _maybe_suggest_printing_press
        lib = _build_fake_library(
            tmp_path / "lib",
            {"skill_name": "pp-cal-com", "category": "productivity",
             "bin_name": "cal-com-pp-cli", "description": "Cal.com CLI"},
        )
        reg = PrintingPressRegistry(library_path=lib)
        out = await _maybe_suggest_printing_press(
            reg, "bash: line 1: cal-com-pp-cli: command not found"
        )
        assert out is not None
        assert "cal-com" in out
        assert "/press install" in out

    @pytest.mark.asyncio
    async def test_no_suggestion_for_clean_output(self, tmp_path):
        from prometheus.engine.agent_loop import _maybe_suggest_printing_press
        lib = _build_fake_library(
            tmp_path / "lib",
            {"skill_name": "pp-slack", "bin_name": "slack-pp-cli"},
        )
        reg = PrintingPressRegistry(library_path=lib)
        out = await _maybe_suggest_printing_press(reg, "ok\nhello\n")
        assert out is None

    @pytest.mark.asyncio
    async def test_no_suggestion_when_library_unavailable(self, tmp_path, monkeypatch):
        from prometheus.engine.agent_loop import _maybe_suggest_printing_press
        from prometheus.tools import printing_press as pp_mod
        monkeypatch.setattr(
            pp_mod, "_DEFAULT_LIBRARY_LOCATIONS",
            (tmp_path / "nope",),
        )
        reg = PrintingPressRegistry()
        assert not reg.is_available()
        out = await _maybe_suggest_printing_press(
            reg, "bash: line 1: pp-slack: command not found"
        )
        assert out is None

    @pytest.mark.asyncio
    async def test_no_suggestion_when_already_installed(
        self, tmp_path, monkeypatch
    ):
        from prometheus.engine.agent_loop import _maybe_suggest_printing_press
        lib = _build_fake_library(
            tmp_path / "lib",
            {"skill_name": "pp-slack", "bin_name": "slack-pp-cli"},
        )
        # Pretend the binary IS installed
        monkeypatch.setattr(
            "prometheus.tools.printing_press.shutil.which",
            lambda name: "/usr/bin/" + name if name == "slack-pp-cli" else None,
        )
        reg = PrintingPressRegistry(library_path=lib)
        out = await _maybe_suggest_printing_press(
            reg, "slack-pp-cli: command not found"
        )
        # Already installed so no suggestion (the failure is something else)
        assert out is None

    @pytest.mark.asyncio
    async def test_strips_pp_cli_suffix_for_search(self, tmp_path):
        """The model often types ``pp-cal-com-cli``; we strip suffixes
        to find the underlying CLI name."""
        from prometheus.engine.agent_loop import _maybe_suggest_printing_press
        lib = _build_fake_library(
            tmp_path / "lib",
            {"skill_name": "pp-cal-com", "category": "productivity",
             "bin_name": "cal-com-pp-cli"},
        )
        reg = PrintingPressRegistry(library_path=lib)
        out = await _maybe_suggest_printing_press(
            reg, "bash: line 1: pp-cal-com-cli: command not found"
        )
        assert out is not None
        assert "cal-com" in out


# ---------------------------------------------------------------------------
# update_library
# ---------------------------------------------------------------------------


class TestUpdateLibrary:
    @pytest.mark.asyncio
    async def test_no_library_returns_false(self, tmp_path, monkeypatch):
        from prometheus.tools import printing_press as pp_mod
        monkeypatch.setattr(
            pp_mod, "_DEFAULT_LIBRARY_LOCATIONS",
            (tmp_path / "nope",),
        )
        reg = PrintingPressRegistry()
        ok, msg = await reg.update_library()
        assert ok is False
        assert "not cloned" in msg

    @pytest.mark.asyncio
    async def test_non_git_path_returns_false(self, tmp_path):
        # cli-skills/ exists, but no .git/
        (tmp_path / "lib" / "cli-skills").mkdir(parents=True)
        reg = PrintingPressRegistry(library_path=tmp_path / "lib")
        ok, msg = await reg.update_library()
        assert ok is False
        assert "not a git" in msg
