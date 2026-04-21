"""Tests for the migration tool (cli/migrate.py).

All tests use tmp_path fixtures — never touches the real home directory.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from prometheus.cli.migrate import (
    HermesMigrator,
    MigrationOptions,
    OpenClawMigrator,
    _deep_get,
    _deep_set,
    detect_sources,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _hermes_tree(root: Path) -> Path:
    """Create a minimal Hermes directory tree."""
    hermes = root / ".hermes"
    hermes.mkdir()
    (hermes / "config.yaml").write_text(
        "model:\n  provider: openrouter\n  default: qwen3.5-32b\n"
        "gateway:\n  telegram:\n    token: bot-token-123\n    enabled: true\n"
    )
    (hermes / "SOUL.md").write_text("# Hermes Soul\nI am Hermes.")
    (hermes / "AGENTS.md").write_text("# Hermes Agents")
    mem = hermes / "memories"
    mem.mkdir()
    (mem / "MEMORY.md").write_text("- fact one\n- fact two\n")
    (mem / "USER.md").write_text("- user likes Python")
    daily = mem / "daily"
    daily.mkdir()
    (daily / "2026-04-01.md").write_text("did stuff")
    skills = hermes / "skills"
    skills.mkdir()
    s1 = skills / "deploy"
    s1.mkdir()
    (s1 / "SKILL.md").write_text("# Deploy skill")
    cron = hermes / "cron"
    cron.mkdir()
    (cron / "backup.yaml").write_text("schedule: daily")
    (hermes / ".env").write_text("OPENAI_API_KEY=sk-test\nANTHROPIC_API_KEY=sk-ant-test\n")
    return hermes


def _openclaw_tree(root: Path) -> Path:
    """Create a minimal OpenClaw directory tree."""
    oc = root / ".openclaw"
    oc.mkdir()
    workspace = root / "clawd"
    workspace.mkdir()
    import json
    config = {
        "agents": {"main": {"model": "anthropic/claude-sonnet-4-20250514", "workspace": str(workspace)}},
        "channels": {"telegram": {"enabled": True}},
    }
    (oc / "openclaw.json").write_text(json.dumps(config))
    (workspace / "SOUL.md").write_text("# OpenClaw Soul")
    (workspace / "MEMORY.md").write_text("- claw memory")
    (workspace / "USER.md").write_text("- claw user pref")
    skills = workspace / "skills"
    skills.mkdir()
    (skills / "research.md").write_text("# Research skill")
    memory = workspace / "memory"
    memory.mkdir()
    (memory / "notes.md").write_text("daily note")
    (oc / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-claw\n")
    return oc


def _opts(src: str, src_path: Path, dst: Path, **kw) -> MigrationOptions:
    return MigrationOptions(source=src, source_path=src_path, dest_path=dst, **kw)


def _redirect_remap_items(report, tmp_path: Path) -> None:
    """Redirect config remap items to tmp_path to avoid writing the real config."""
    config_path = tmp_path / "config" / "prometheus.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    for item in report.items:
        if item.action == "remap":
            item.dest_path = config_path


# ------------------------------------------------------------------
# detect_sources
# ------------------------------------------------------------------

class TestDetectSources:
    def test_finds_hermes(self, tmp_path: Path):
        _hermes_tree(tmp_path)
        with patch("prometheus.cli.migrate.Path.home", return_value=tmp_path):
            sources = detect_sources()
        assert "hermes" in sources
        assert sources["hermes"] == tmp_path / ".hermes"

    def test_finds_openclaw(self, tmp_path: Path):
        _openclaw_tree(tmp_path)
        with patch("prometheus.cli.migrate.Path.home", return_value=tmp_path):
            sources = detect_sources()
        assert "openclaw" in sources

    def test_finds_clawdbot_legacy(self, tmp_path: Path):
        cb = tmp_path / ".clawdbot"
        cb.mkdir()
        (cb / "clawdbot.json").write_text("{}")
        with patch("prometheus.cli.migrate.Path.home", return_value=tmp_path):
            sources = detect_sources()
        assert "openclaw" in sources
        assert sources["openclaw"] == cb

    def test_empty_when_nothing_exists(self, tmp_path: Path):
        with patch("prometheus.cli.migrate.Path.home", return_value=tmp_path):
            sources = detect_sources()
        assert sources == {}


# ------------------------------------------------------------------
# Hermes migrator — scan
# ------------------------------------------------------------------

class TestHermesScan:
    def test_scans_identity_files(self, tmp_path: Path):
        hermes = _hermes_tree(tmp_path)
        dst = tmp_path / "prom"
        m = HermesMigrator(_opts("hermes", hermes, dst))
        report = m.scan()
        cats = [i.category for i in report.items]
        assert "identity" in cats
        soul = [i for i in report.items if "SOUL" in i.description]
        assert len(soul) == 1

    def test_scans_memory_files(self, tmp_path: Path):
        hermes = _hermes_tree(tmp_path)
        dst = tmp_path / "prom"
        m = HermesMigrator(_opts("hermes", hermes, dst))
        report = m.scan()
        mem_items = [i for i in report.items if i.category == "memory"]
        names = [i.dest_path.name for i in mem_items if i.dest_path.name.endswith(".md")]
        assert "MEMORY.md" in names
        assert "USER.md" in names

    def test_scans_daily_notes(self, tmp_path: Path):
        hermes = _hermes_tree(tmp_path)
        dst = tmp_path / "prom"
        m = HermesMigrator(_opts("hermes", hermes, dst))
        report = m.scan()
        daily = [i for i in report.items if "Daily" in i.description]
        assert len(daily) == 1

    def test_scans_skills(self, tmp_path: Path):
        hermes = _hermes_tree(tmp_path)
        dst = tmp_path / "prom"
        m = HermesMigrator(_opts("hermes", hermes, dst))
        report = m.scan()
        skills = [i for i in report.items if i.category == "skills"]
        assert len(skills) >= 1

    def test_scans_config(self, tmp_path: Path):
        hermes = _hermes_tree(tmp_path)
        dst = tmp_path / "prom"
        m = HermesMigrator(_opts("hermes", hermes, dst))
        report = m.scan()
        config = [i for i in report.items if i.action == "remap"]
        assert len(config) == 1

    def test_scans_cron(self, tmp_path: Path):
        hermes = _hermes_tree(tmp_path)
        dst = tmp_path / "prom"
        m = HermesMigrator(_opts("hermes", hermes, dst))
        report = m.scan()
        cron = [i for i in report.items if "Cron" in i.description]
        assert len(cron) == 1

    def test_marks_conflicts(self, tmp_path: Path):
        hermes = _hermes_tree(tmp_path)
        dst = tmp_path / "prom"
        dst.mkdir()
        (dst / "SOUL.md").write_text("existing soul")
        m = HermesMigrator(_opts("hermes", hermes, dst))
        report = m.scan()
        soul = [i for i in report.items if "SOUL" in i.description][0]
        assert soul.status == "conflict"

    def test_secrets_excluded_by_default(self, tmp_path: Path):
        hermes = _hermes_tree(tmp_path)
        dst = tmp_path / "prom"
        m = HermesMigrator(_opts("hermes", hermes, dst))
        report = m.scan()
        secrets = [i for i in report.items if i.category == "secrets"]
        assert len(secrets) == 0

    def test_secrets_included_with_full_preset(self, tmp_path: Path):
        hermes = _hermes_tree(tmp_path)
        dst = tmp_path / "prom"
        m = HermesMigrator(_opts("hermes", hermes, dst, preset="full"))
        report = m.scan()
        secrets = [i for i in report.items if i.category == "secrets"]
        assert len(secrets) == 1
        assert secrets[0].action == "manual"


# ------------------------------------------------------------------
# Hermes migrator — execute
# ------------------------------------------------------------------

class TestHermesExecute:
    def _execute_safe(self, migrator: HermesMigrator, tmp_path: Path) -> "MigrationReport":
        """Execute migration with remap items redirected to tmp_path."""
        report = migrator.scan()
        _redirect_remap_items(report, tmp_path)
        migrator._execute_items(report)
        return report

    def test_copies_files(self, tmp_path: Path):
        hermes = _hermes_tree(tmp_path)
        dst = tmp_path / "prom"
        m = HermesMigrator(_opts("hermes", hermes, dst))
        report = self._execute_safe(m, tmp_path)
        assert (dst / "SOUL.md").exists()
        assert (dst / "SOUL.md").read_text() == "# Hermes Soul\nI am Hermes."
        assert (dst / "MEMORY.md").exists()
        assert len(report.migrated) > 0

    def test_dry_run_no_changes(self, tmp_path: Path):
        hermes = _hermes_tree(tmp_path)
        dst = tmp_path / "prom"
        m = HermesMigrator(_opts("hermes", hermes, dst, dry_run=True))
        report = m.execute()
        assert not (dst / "SOUL.md").exists()
        assert all(i.status == "pending" for i in report.items
                    if i.action not in ("skip", "manual"))

    def test_skips_conflicts_without_overwrite(self, tmp_path: Path):
        hermes = _hermes_tree(tmp_path)
        dst = tmp_path / "prom"
        dst.mkdir()
        (dst / "SOUL.md").write_text("keep me")
        m = HermesMigrator(_opts("hermes", hermes, dst))
        report = self._execute_safe(m, tmp_path)
        assert (dst / "SOUL.md").read_text() == "keep me"
        soul = [i for i in report.items if "SOUL" in i.description][0]
        assert soul.status == "conflict"

    def test_overwrites_with_archive(self, tmp_path: Path):
        hermes = _hermes_tree(tmp_path)
        dst = tmp_path / "prom"
        dst.mkdir()
        (dst / "SOUL.md").write_text("old soul")
        m = HermesMigrator(_opts("hermes", hermes, dst, overwrite=True))
        report = self._execute_safe(m, tmp_path)
        assert (dst / "SOUL.md").read_text() == "# Hermes Soul\nI am Hermes."
        archive_dirs = list((dst / "migration" / "hermes").iterdir())
        assert len(archive_dirs) == 1
        archive = archive_dirs[0] / "archive" / "SOUL.md"
        assert archive.exists()
        assert archive.read_text() == "old soul"

    def test_creates_migration_report(self, tmp_path: Path):
        hermes = _hermes_tree(tmp_path)
        dst = tmp_path / "prom"
        m = HermesMigrator(_opts("hermes", hermes, dst))
        self._execute_safe(m, tmp_path)
        reports = list((dst / "migration" / "hermes").rglob("migration_report.yaml"))
        assert len(reports) == 1
        data = yaml.safe_load(reports[0].read_text())
        assert data["source"] == "hermes"
        assert data["summary"]["migrated"] > 0

    def test_config_remap(self, tmp_path: Path):
        hermes = _hermes_tree(tmp_path)
        dst = tmp_path / "prom"
        config_path = tmp_path / "config" / "prometheus.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text("system:\n  name: Prometheus\n")
        m = HermesMigrator(_opts("hermes", hermes, dst))
        report = m.scan()
        for item in report.items:
            if item.action == "remap":
                item.dest_path = config_path
        m._execute_items(report)
        result = yaml.safe_load(config_path.read_text())
        assert result["model"]["provider"] == "openai"  # openrouter -> openai
        assert result["gateway"]["telegram_token"] == "bot-token-123"

    def test_skill_conflict_rename(self, tmp_path: Path):
        hermes = _hermes_tree(tmp_path)
        dst = tmp_path / "prom"
        imported = dst / "skills" / "imported" / "deploy"
        imported.mkdir(parents=True)
        (imported / "SKILL.md").write_text("existing")
        m = HermesMigrator(_opts("hermes", hermes, dst, skill_conflict="rename"))
        report = m.scan()
        skill_items = [i for i in report.items if i.category == "skills"]
        renamed = [i for i in skill_items if "hermes" in str(i.dest_path)]
        assert len(renamed) >= 1


# ------------------------------------------------------------------
# OpenClaw migrator
# ------------------------------------------------------------------

class TestOpenClawScan:
    def test_finds_workspace(self, tmp_path: Path):
        _openclaw_tree(tmp_path)
        oc = tmp_path / ".openclaw"
        dst = tmp_path / "prom"
        m = OpenClawMigrator(_opts("openclaw", oc, dst))
        assert m.workspace == tmp_path / "clawd"

    def test_scans_workspace_files(self, tmp_path: Path):
        _openclaw_tree(tmp_path)
        oc = tmp_path / ".openclaw"
        dst = tmp_path / "prom"
        m = OpenClawMigrator(_opts("openclaw", oc, dst))
        report = m.scan()
        descs = [i.description for i in report.items]
        assert any("SOUL" in d for d in descs)
        assert any("MEMORY" in d for d in descs)

    def test_scans_skills(self, tmp_path: Path):
        _openclaw_tree(tmp_path)
        oc = tmp_path / ".openclaw"
        dst = tmp_path / "prom"
        m = OpenClawMigrator(_opts("openclaw", oc, dst))
        report = m.scan()
        skills = [i for i in report.items if i.category == "skills"]
        assert len(skills) >= 1

    def test_scans_config(self, tmp_path: Path):
        _openclaw_tree(tmp_path)
        oc = tmp_path / ".openclaw"
        dst = tmp_path / "prom"
        m = OpenClawMigrator(_opts("openclaw", oc, dst))
        report = m.scan()
        config = [i for i in report.items if i.action == "remap"]
        assert len(config) == 1

    def test_executes_copy(self, tmp_path: Path):
        _openclaw_tree(tmp_path)
        oc = tmp_path / ".openclaw"
        dst = tmp_path / "prom"
        m = OpenClawMigrator(_opts("openclaw", oc, dst))
        report = m.scan()
        _redirect_remap_items(report, tmp_path)
        m._execute_items(report)
        assert (dst / "SOUL.md").exists()
        assert (dst / "SOUL.md").read_text() == "# OpenClaw Soul"
        assert len(report.migrated) > 0


# ------------------------------------------------------------------
# Memory overflow
# ------------------------------------------------------------------

class TestMemoryOverflow:
    def test_small_memory_copied_directly(self, tmp_path: Path):
        hermes = tmp_path / ".hermes"
        hermes.mkdir()
        (hermes / "config.yaml").write_text("model:\n  provider: ollama\n")
        mem = hermes / "memories"
        mem.mkdir()
        (mem / "MEMORY.md").write_text("- short fact\n")
        dst = tmp_path / "prom"
        m = HermesMigrator(_opts("hermes", hermes, dst))
        m.execute()
        assert (dst / "MEMORY.md").read_text() == "- short fact\n"

    def test_large_memory_trimmed(self, tmp_path: Path):
        hermes = tmp_path / ".hermes"
        hermes.mkdir()
        (hermes / "config.yaml").write_text("model:\n  provider: ollama\n")
        mem = hermes / "memories"
        mem.mkdir()
        # Create a MEMORY.md that exceeds the 12K char limit
        lines = [f"- fact number {i}: " + "x" * 100 for i in range(200)]
        big_content = "\n".join(lines)
        assert len(big_content) > 12_000
        (mem / "MEMORY.md").write_text(big_content)
        dst = tmp_path / "prom"
        m = HermesMigrator(_opts("hermes", hermes, dst))
        m.execute()
        result = (dst / "MEMORY.md").read_text()
        assert len(result) <= 12_000
        # Most recent facts (end of file) should be kept
        assert "fact number 199" in result
        # Overflow should be archived
        overflow = list((dst / "migration").rglob("MEMORY.md.overflow"))
        assert len(overflow) == 1
        assert "fact number 0" in overflow[0].read_text()


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

class TestUtils:
    def test_deep_get(self):
        d = {"a": {"b": {"c": 42}}}
        assert _deep_get(d, "a.b.c") == 42
        assert _deep_get(d, "a.x") is None
        assert _deep_get(d, "z") is None

    def test_deep_set(self):
        d: dict = {}
        _deep_set(d, "a.b.c", 42)
        assert d == {"a": {"b": {"c": 42}}}

    def test_deep_set_existing(self):
        d = {"a": {"b": 1}}
        _deep_set(d, "a.b", 2)
        assert d["a"]["b"] == 2
