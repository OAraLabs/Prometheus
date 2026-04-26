"""MorphEngine — daemon-manager detection, candidate staging, swap, rollback."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from prometheus.symbiote.backup_vault import BackupVault
from prometheus.symbiote.morph import (
    MorphEngine,
    MorphReport,
    SwapResult,
    _pid_alive,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _project_tree(root: Path) -> Path:
    (root / "src" / "prometheus").mkdir(parents=True, exist_ok=True)
    (root / "src" / "prometheus" / "__init__.py").write_text(
        "__version__ = '0.0.0'\n"
    )
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "test_smoke.py").write_text(
        "def test_smoke():\n    assert 1 + 1 == 2\n"
    )
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "config" / "prometheus.yaml").write_text("system: {name: P}\n")
    (root / "PROMETHEUS.md").write_text("# Prometheus\n")
    return root


def _engine(tmp_path: Path, **kwargs) -> MorphEngine:
    project = _project_tree(tmp_path / "project")
    vault = BackupVault(
        project_root=project,
        vault_root=tmp_path / "vault",
        include_identity=False,
    )
    return MorphEngine(
        backup_vault=vault,
        project_root=project,
        candidate_root=tmp_path / "candidate",
        post_mortem_root=tmp_path / "post_mortem",
        health_check_timeout=2,
        health_check_interval=1,
        consecutive_passes_required=1,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# _detect_daemon_manager — CRITICAL no-systemd-assumption tests
# ---------------------------------------------------------------------------


class TestDetectDaemonManager:
    """Detection order: systemctl is-active → pidfile → pkill."""

    def test_returns_pidfile_when_lockfile_alive(self, tmp_path, monkeypatch):
        engine = _engine(tmp_path)

        # Mock systemctl to return inactive (so step 1 fails).
        async def fake_active():
            return False
        monkeypatch.setattr(MorphEngine, "_systemctl_is_active", staticmethod(fake_active))

        # Make the lockfile point at our own PID (definitely alive).
        from prometheus.symbiote import morph as morph_mod
        lock = tmp_path / "daemon.lock"
        lock.write_text(json.dumps({"pid": os.getpid()}))
        monkeypatch.setattr(morph_mod, "_daemon_lock_path", lambda: lock)

        manager = asyncio.run(engine._detect_daemon_manager())
        assert manager == "pidfile"

    def test_returns_pkill_when_no_lockfile_no_systemd(self, tmp_path, monkeypatch):
        engine = _engine(tmp_path)

        async def fake_active():
            return False
        monkeypatch.setattr(MorphEngine, "_systemctl_is_active", staticmethod(fake_active))

        from prometheus.symbiote import morph as morph_mod
        nonexistent = tmp_path / "nonexistent.lock"
        monkeypatch.setattr(morph_mod, "_daemon_lock_path", lambda: nonexistent)

        manager = asyncio.run(engine._detect_daemon_manager())
        assert manager == "pkill"

    def test_returns_systemd_when_systemctl_active(self, tmp_path, monkeypatch):
        engine = _engine(tmp_path)

        async def fake_active():
            return True
        monkeypatch.setattr(MorphEngine, "_systemctl_is_active", staticmethod(fake_active))
        manager = asyncio.run(engine._detect_daemon_manager())
        assert manager == "systemd"

    def test_skips_pidfile_when_pid_dead(self, tmp_path, monkeypatch):
        engine = _engine(tmp_path)

        async def fake_active():
            return False
        monkeypatch.setattr(MorphEngine, "_systemctl_is_active", staticmethod(fake_active))

        from prometheus.symbiote import morph as morph_mod
        lock = tmp_path / "daemon.lock"
        # Use PID 1 only as fallback if 999999 doesn't work — but typically
        # PID 999999 is unused. Validate via _pid_alive.
        dead_pid = 999_999
        if _pid_alive(dead_pid):
            pytest.skip(f"PID {dead_pid} is unexpectedly alive on this host")
        lock.write_text(json.dumps({"pid": dead_pid}))
        monkeypatch.setattr(morph_mod, "_daemon_lock_path", lambda: lock)

        manager = asyncio.run(engine._detect_daemon_manager())
        assert manager == "pkill"

    def test_does_not_hang_when_systemctl_missing(self, tmp_path, monkeypatch):
        """Critical: systemctl missing must NOT hang the detection."""
        # Patch asyncio.create_subprocess_exec to raise FileNotFoundError
        # when called with 'systemctl'.
        engine = _engine(tmp_path)
        from prometheus.symbiote import morph as morph_mod

        nonexistent = tmp_path / "no_lock"
        monkeypatch.setattr(morph_mod, "_daemon_lock_path", lambda: nonexistent)

        original = asyncio.create_subprocess_exec

        async def patched(*args, **kwargs):
            if args and args[0] == "systemctl":
                raise FileNotFoundError("systemctl missing")
            return await original(*args, **kwargs)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", patched)

        start = time.monotonic()
        manager = asyncio.run(engine._detect_daemon_manager())
        elapsed = time.monotonic() - start

        assert manager == "pkill"
        assert elapsed < 5.0, f"detection took {elapsed:.1f}s — must not hang"

    def test_caches_result(self, tmp_path, monkeypatch):
        engine = _engine(tmp_path)
        call_count = {"n": 0}

        async def fake_active():
            call_count["n"] += 1
            return True

        monkeypatch.setattr(MorphEngine, "_systemctl_is_active", staticmethod(fake_active))
        asyncio.run(engine._detect_daemon_manager())
        asyncio.run(engine._detect_daemon_manager())
        assert call_count["n"] == 1, "result should be cached after first call"

    def test_override_skips_detection(self, tmp_path):
        engine = _engine(tmp_path, daemon_manager_override="pidfile")
        manager = asyncio.run(engine._detect_daemon_manager())
        assert manager == "pidfile"

    def test_override_invalid_falls_through(self, tmp_path, monkeypatch):
        engine = _engine(tmp_path, daemon_manager_override="not_a_real_strategy")

        async def fake_active():
            return False
        monkeypatch.setattr(MorphEngine, "_systemctl_is_active", staticmethod(fake_active))
        from prometheus.symbiote import morph as morph_mod
        nonexistent = tmp_path / "nope"
        monkeypatch.setattr(morph_mod, "_daemon_lock_path", lambda: nonexistent)
        manager = asyncio.run(engine._detect_daemon_manager())
        assert manager == "pkill"


# ---------------------------------------------------------------------------
# Candidate staging
# ---------------------------------------------------------------------------


class TestStageCandidate:
    @pytest.mark.asyncio
    async def test_stage_copies_live_tree(self, tmp_path):
        engine = _engine(tmp_path)
        await engine._stage_candidate()
        cand = engine._candidate_root
        assert (cand / "src" / "prometheus" / "__init__.py").exists()
        assert (cand / "tests" / "test_smoke.py").exists()

    @pytest.mark.asyncio
    async def test_stage_excludes_pycache(self, tmp_path):
        engine = _engine(tmp_path)
        # Add a __pycache__ dir to the live tree.
        cache = engine._project_root / "src" / "prometheus" / "__pycache__"
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "x.pyc").write_text("nope")
        await engine._stage_candidate()
        cand_pycache = engine._candidate_root / "src" / "prometheus" / "__pycache__"
        assert not cand_pycache.exists()


# ---------------------------------------------------------------------------
# prepare_candidate
# ---------------------------------------------------------------------------


class TestPrepareCandidate:
    @pytest.mark.asyncio
    async def test_creates_morph_report_with_backup(self, tmp_path):
        engine = _engine(tmp_path)
        # Skip the candidate test run for speed.
        report = await engine.prepare_candidate(capture_test_status=False)
        assert isinstance(report, MorphReport)
        assert report.backup_snapshot_id  # snapshot was created
        assert report.tests_passed is True  # because we skipped
        assert report.ready_to_swap is True

    @pytest.mark.asyncio
    async def test_blocks_when_tests_fail(self, tmp_path):
        engine = _engine(tmp_path)

        # Patch the candidate test runner to return failure.
        async def fake_tests(self):
            return False, "FAILED 1 test"
        with patch.object(MorphEngine, "_run_candidate_tests", fake_tests):
            report = await engine.prepare_candidate(capture_test_status=True)
        assert report.ready_to_swap is False
        assert report.tests_passed is False
        assert "test" in report.reason_if_not_ready.lower()


# ---------------------------------------------------------------------------
# Swap + rollback (mocked stop/start/health)
# ---------------------------------------------------------------------------


class TestSwap:
    @pytest.mark.asyncio
    async def test_swap_aborts_when_not_ready(self, tmp_path):
        engine = _engine(tmp_path)
        report = MorphReport(
            backup_snapshot_id="vX",
            candidate_path=str(engine._candidate_root),
            graft_applied=False,
            tests_passed=False,
            tests_output="",
            ready_to_swap=False,
            reason_if_not_ready="tests failed",
        )
        result = await engine.execute_swap(report)
        assert not result.success
        assert "Not ready to swap" in result.error

    @pytest.mark.asyncio
    async def test_swap_aborts_when_candidate_outside_root(self, tmp_path):
        engine = _engine(tmp_path)
        report = MorphReport(
            backup_snapshot_id="vX",
            candidate_path=str(tmp_path / "elsewhere"),  # outside CANDIDATE_ROOT
            graft_applied=False,
            tests_passed=True,
            tests_output="",
            ready_to_swap=True,
        )
        result = await engine.execute_swap(report)
        assert not result.success
        assert "outside CANDIDATE_ROOT" in result.error

    @pytest.mark.asyncio
    async def test_swap_aborts_when_stop_fails(self, tmp_path):
        engine = _engine(tmp_path, daemon_manager_override="pkill")
        # Stage candidate first.
        await engine._stage_candidate()
        report = MorphReport(
            backup_snapshot_id="vX",
            candidate_path=str(engine._candidate_root),
            graft_applied=False,
            tests_passed=True,
            tests_output="",
            ready_to_swap=True,
        )

        async def fake_stop(self, manager):
            return False

        with patch.object(MorphEngine, "_stop_daemon", fake_stop):
            result = await engine.execute_swap(report)
        assert not result.success
        assert "stop daemon" in result.error.lower()

    @pytest.mark.asyncio
    async def test_swap_rolls_back_on_unhealthy(self, tmp_path):
        engine = _engine(tmp_path, daemon_manager_override="pkill")
        await engine._stage_candidate()
        report = MorphReport(
            backup_snapshot_id="vX",
            candidate_path=str(engine._candidate_root),
            graft_applied=False,
            tests_passed=True,
            tests_output="",
            ready_to_swap=True,
        )

        # Stub stop/start/health to simulate an unhealthy swap.
        async def stop_ok(self, manager):
            return True

        async def start_ok(self, manager):
            return True

        async def unhealthy(self):
            return False, "fake unhealthy"

        with patch.object(MorphEngine, "_stop_daemon", stop_ok), \
             patch.object(MorphEngine, "_start_daemon", start_ok), \
             patch.object(MorphEngine, "_health_check_watchdog", unhealthy):
            result = await engine.execute_swap(report)
        assert result.rolled_back is True
        assert "fake unhealthy" in result.rollback_reason
        # Live tree restored: check src/prometheus exists.
        assert (engine._project_root / "src" / "prometheus").exists()
        # Failed candidate preserved.
        post_mortem_dirs = list(engine._post_mortem_root.glob("failed_*"))
        assert post_mortem_dirs
        reason_path = post_mortem_dirs[0] / "REASON.txt"
        assert reason_path.exists()
        assert "fake unhealthy" in reason_path.read_text()


# ---------------------------------------------------------------------------
# Path-traversal guard on _candidate_under_root
# ---------------------------------------------------------------------------


class TestPathTraversalGuard:
    def test_rejects_path_outside_candidate_root(self, tmp_path):
        engine = _engine(tmp_path)
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        assert engine._candidate_under_root(outside) is False

    def test_rejects_traversal_via_dot_dot(self, tmp_path):
        engine = _engine(tmp_path)
        sneaky = engine._candidate_root / ".." / "elsewhere"
        # Even though the literal string starts with the candidate root,
        # resolving it lands outside.
        assert engine._candidate_under_root(sneaky) is False

    def test_accepts_path_inside_candidate_root(self, tmp_path):
        engine = _engine(tmp_path)
        engine._candidate_root.mkdir(parents=True, exist_ok=True)
        nested = engine._candidate_root / "src"
        nested.mkdir()
        assert engine._candidate_under_root(nested) is True
