"""BackupVault — tarball creation, manifest, retention, restore."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import tarfile
from datetime import date
from pathlib import Path

import pytest

from prometheus.symbiote.backup_vault import (
    BackupSnapshot,
    BackupVault,
    RestoreResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _project_tree(root: Path) -> Path:
    """Build a minimal Prometheus-shaped project tree under ``root``."""
    (root / "src" / "prometheus").mkdir(parents=True, exist_ok=True)
    (root / "src" / "prometheus" / "__init__.py").write_text(
        "__version__ = '0.0.0'\n"
    )
    (root / "src" / "prometheus" / "core.py").write_text(
        "def add(a, b):\n    return a + b\n"
    )
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "test_dummy.py").write_text(
        "def test_x():\n    assert True\n"
    )
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "config" / "prometheus.yaml").write_text(
        "system: {name: P}\n"
    )
    (root / "PROMETHEUS.md").write_text("# Prometheus\n")
    # Junk that should be excluded
    (root / "src" / "prometheus" / "__pycache__").mkdir(exist_ok=True)
    (root / "src" / "prometheus" / "__pycache__" / "x.pyc").write_text("nope")
    (root / ".git").mkdir(exist_ok=True)
    (root / ".git" / "config").write_text("[core]\n")
    return root


def _vault(tmp_path: Path) -> BackupVault:
    project = _project_tree(tmp_path / "project")
    return BackupVault(
        project_root=project,
        vault_root=tmp_path / "vault",
        max_backups=10,
        include_identity=False,  # avoid reaching into ~/.prometheus during tests
    )


# ---------------------------------------------------------------------------
# create_snapshot
# ---------------------------------------------------------------------------


class TestCreateSnapshot:
    @pytest.mark.asyncio
    async def test_creates_tarball(self, tmp_path):
        vault = _vault(tmp_path)
        snap = await vault.create_snapshot(
            description="x", source="manual", capture_test_status=False,
        )
        assert isinstance(snap, BackupSnapshot)
        assert snap.backup_id.startswith("v1_")
        assert snap.version_number == 1
        assert snap.source == "manual"
        assert Path(snap.tarball_path).exists()
        assert snap.size_bytes > 0
        assert snap.file_count > 0

    @pytest.mark.asyncio
    async def test_tarball_contains_expected_files(self, tmp_path):
        vault = _vault(tmp_path)
        snap = await vault.create_snapshot(capture_test_status=False)
        with tarfile.open(snap.tarball_path) as tar:
            names = {m.name for m in tar.getmembers()}
        assert "src/prometheus/__init__.py" in names
        assert "src/prometheus/core.py" in names
        assert "tests/test_dummy.py" in names
        assert "config/prometheus.yaml" in names
        assert "PROMETHEUS.md" in names

    @pytest.mark.asyncio
    async def test_excludes_pycache_and_git(self, tmp_path):
        vault = _vault(tmp_path)
        snap = await vault.create_snapshot(capture_test_status=False)
        with tarfile.open(snap.tarball_path) as tar:
            names = {m.name for m in tar.getmembers()}
        assert not any(".git" in n.split("/") for n in names)
        assert not any("__pycache__" in n.split("/") for n in names)
        assert not any(n.endswith(".pyc") for n in names)

    @pytest.mark.asyncio
    async def test_version_numbers_auto_increment(self, tmp_path):
        vault = _vault(tmp_path)
        snap1 = await vault.create_snapshot(capture_test_status=False)
        snap2 = await vault.create_snapshot(capture_test_status=False)
        snap3 = await vault.create_snapshot(capture_test_status=False)
        assert snap1.version_number == 1
        assert snap2.version_number == 2
        assert snap3.version_number == 3

    @pytest.mark.asyncio
    async def test_manifest_db_tracks_snapshots(self, tmp_path):
        vault = _vault(tmp_path)
        await vault.create_snapshot(capture_test_status=False)
        await vault.create_snapshot(capture_test_status=False)
        snaps = vault.list_snapshots()
        assert len(snaps) == 2
        # Newest first.
        assert snaps[0].version_number > snaps[1].version_number

    @pytest.mark.asyncio
    async def test_prometheus_md_hash_recorded(self, tmp_path):
        vault = _vault(tmp_path)
        snap = await vault.create_snapshot(capture_test_status=False)
        # Hash of "# Prometheus\n"
        expected = hashlib.sha256(b"# Prometheus\n").hexdigest()
        assert snap.prometheus_md_hash == expected

    @pytest.mark.asyncio
    async def test_capture_test_status_skipped_when_false(self, tmp_path):
        vault = _vault(tmp_path)
        snap = await vault.create_snapshot(capture_test_status=False)
        assert snap.test_status == "unknown"


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------


class TestRetention:
    @pytest.mark.asyncio
    async def test_keeps_manual_and_morph_snapshots_indefinitely(self, tmp_path):
        vault = _vault(tmp_path)
        vault._max_backups = 2
        # Create five snapshots: 2 manual, 2 morph, 1 scheduled.
        await vault.create_snapshot(source="manual", capture_test_status=False)
        await vault.create_snapshot(source="manual", capture_test_status=False)
        await vault.create_snapshot(source="symbiote_morph", capture_test_status=False)
        await vault.create_snapshot(source="symbiote_morph", capture_test_status=False)
        await vault.create_snapshot(source="scheduled", capture_test_status=False)
        snaps = vault.list_snapshots(limit=20)
        sources = {s.source for s in snaps}
        assert "manual" in sources
        assert "symbiote_morph" in sources

    @pytest.mark.asyncio
    async def test_multiple_snapshots_same_day_all_retained(self, tmp_path):
        vault = _vault(tmp_path)
        vault._max_backups = 1
        await vault.create_snapshot(source="scheduled", capture_test_status=False)
        await vault.create_snapshot(source="scheduled", capture_test_status=False)
        await vault.create_snapshot(source="scheduled", capture_test_status=False)
        # Today's backups are exempt — all should still be present.
        snaps = vault.list_snapshots(limit=20)
        assert len(snaps) == 3


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------


class TestRestore:
    @pytest.mark.asyncio
    async def test_dry_run_does_not_modify_files(self, tmp_path):
        vault = _vault(tmp_path)
        snap = await vault.create_snapshot(capture_test_status=False)
        # Modify the live tree after backup.
        (vault._project_root / "src" / "prometheus" / "core.py").write_text(
            "def add(a, b):\n    return 999\n"
        )
        # Dry run.
        result = await vault.restore_snapshot(snap.backup_id, dry_run=True)
        assert result.dry_run is True
        # core.py was changed AFTER the snapshot, so dry-run reports it.
        assert any("core.py" in p for p in result.files_changed)
        # File still has the modified content.
        assert "999" in (vault._project_root / "src" / "prometheus" / "core.py").read_text()

    @pytest.mark.asyncio
    async def test_restore_replaces_changed_files(self, tmp_path):
        vault = _vault(tmp_path)
        snap = await vault.create_snapshot(capture_test_status=False)
        core_path = vault._project_root / "src" / "prometheus" / "core.py"
        core_path.write_text("def add(a, b):\n    return 999\n")
        result = await vault.restore_snapshot(
            snap.backup_id, capture_test_status=False,
        )
        assert result.error == ""
        assert result.files_restored > 0
        # The restored file should match the original.
        assert "return a + b" in core_path.read_text()

    @pytest.mark.asyncio
    async def test_pre_restore_backup_created(self, tmp_path):
        vault = _vault(tmp_path)
        snap = await vault.create_snapshot(capture_test_status=False)
        before = len(vault.list_snapshots(limit=20))
        await vault.restore_snapshot(snap.backup_id, capture_test_status=False)
        after = vault.list_snapshots(limit=20)
        # Should have one extra snapshot, source="pre_restore"
        assert len(after) == before + 1
        assert any(s.source == "pre_restore" for s in after)

    @pytest.mark.asyncio
    async def test_restore_unknown_backup_id_fails_gracefully(self, tmp_path):
        vault = _vault(tmp_path)
        result = await vault.restore_snapshot("v999_nonexistent")
        assert result.error
        assert "Unknown backup_id" in result.error

    @pytest.mark.asyncio
    async def test_restore_corrupt_tarball_reports_error(self, tmp_path):
        vault = _vault(tmp_path)
        snap = await vault.create_snapshot(capture_test_status=False)
        # Corrupt the tarball.
        Path(snap.tarball_path).write_bytes(b"not a tarball")
        result = await vault.restore_snapshot(snap.backup_id)
        assert result.error
        assert "extract" in result.error.lower()


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


class TestListing:
    @pytest.mark.asyncio
    async def test_get_snapshot_by_id(self, tmp_path):
        vault = _vault(tmp_path)
        snap = await vault.create_snapshot(capture_test_status=False)
        loaded = vault.get_snapshot(snap.backup_id)
        assert loaded is not None
        assert loaded.backup_id == snap.backup_id

    @pytest.mark.asyncio
    async def test_get_snapshot_unknown_returns_none(self, tmp_path):
        vault = _vault(tmp_path)
        assert vault.get_snapshot("nope") is None

    @pytest.mark.asyncio
    async def test_get_latest(self, tmp_path):
        vault = _vault(tmp_path)
        s1 = await vault.create_snapshot(capture_test_status=False)
        s2 = await vault.create_snapshot(capture_test_status=False)
        latest = vault.get_latest()
        assert latest is not None
        assert latest.backup_id == s2.backup_id

    @pytest.mark.asyncio
    async def test_get_latest_when_empty(self, tmp_path):
        vault = _vault(tmp_path)
        assert vault.get_latest() is None
