"""Tests for Sprint 15 GRAFT: scoped daemon lock."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from prometheus.gateway.status import (
    acquire_daemon_lock,
    release_daemon_lock,
)


class TestScopedLock:

    def test_acquire_and_release(self, tmp_path, monkeypatch):
        monkeypatch.setattr("prometheus.gateway.status.get_config_dir", lambda: tmp_path)
        ok, reason = acquire_daemon_lock()
        assert ok
        assert reason == ""
        lock_file = tmp_path / "daemon.lock"
        assert lock_file.exists()
        data = json.loads(lock_file.read_text())
        assert data["pid"] == os.getpid()

        release_daemon_lock()
        assert not lock_file.exists()

    def test_double_acquire_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr("prometheus.gateway.status.get_config_dir", lambda: tmp_path)
        ok1, _ = acquire_daemon_lock()
        assert ok1

        ok2, reason = acquire_daemon_lock()
        assert not ok2
        assert "already running" in reason.lower() or "race" in reason.lower()

        release_daemon_lock()

    def test_stale_lock_cleaned(self, tmp_path, monkeypatch):
        monkeypatch.setattr("prometheus.gateway.status.get_config_dir", lambda: tmp_path)
        # Write a lock with a dead PID
        lock_file = tmp_path / "daemon.lock"
        lock_file.write_text(json.dumps({
            "pid": 99999999,  # almost certainly not running
            "start_time": None,
            "started_at": 0,
            "argv": "fake",
        }))

        ok, reason = acquire_daemon_lock()
        assert ok, f"Should have cleaned stale lock, got: {reason}"

        release_daemon_lock()

    def test_release_wrong_pid(self, tmp_path, monkeypatch):
        monkeypatch.setattr("prometheus.gateway.status.get_config_dir", lambda: tmp_path)
        lock_file = tmp_path / "daemon.lock"
        lock_file.write_text(json.dumps({"pid": 1, "start_time": None}))

        # We are NOT pid 1, so release should not delete
        release_daemon_lock()
        assert lock_file.exists()
