"""A stale daemon lock is recognised when its PID now belongs to another process.

WHAT WAS WRONG (WP-X.24)
------------------------
``acquire_daemon_lock`` tells a live holder from a recycled PID by comparing
the start time recorded in the lock with the start time of whatever process
has that PID now. ``_process_start_time`` read ``/proc/<pid>/stat``, which
exists only on Linux. On macOS it returned None for every process, so every
lock was written with ``start_time: null``, and on the next start any live
process that happened to have the old PID made the daemon refuse to run
("Daemon appears to be running") until someone deleted the lock by hand.

The first test below is the reproduction: a lock written the way this
platform's own code writes it, by a process that is gone (here, recorded as
this test process's start time), whose PID now belongs to a newer, unrelated
process. It passes on Linux on origin/main and fails on macOS there.

The two "still blocks" tests guard the other direction: the fix must never
delete the lock of the daemon that actually holds it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

from prometheus.gateway import status as S


@pytest.fixture
def lock_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("prometheus.gateway.status.get_config_dir", lambda: tmp_path)
    yield tmp_path
    S.release_daemon_lock()


@pytest.fixture
def newer_process():
    """A live process started NOW: the one that inherited the stale PID."""
    before = time.time()
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        yield proc, before
    finally:
        proc.kill()
        proc.wait()


def _write_lock(lock_dir, **record):
    record.setdefault("argv", "prometheus daemon")
    (lock_dir / "daemon.lock").write_text(json.dumps(record))


def test_a_lock_whose_pid_now_belongs_to_a_newer_process_is_stale(lock_dir, newer_process):
    """THE REPRODUCTION. On macOS, origin/main refuses to start here."""
    proc, before = newer_process
    _write_lock(
        lock_dir,
        pid=proc.pid,
        # What the dead daemon would have recorded for ITSELF, by this
        # platform's own code: None on macOS before the fix.
        start_time=S._process_start_time(os.getpid()),
        started_at=before - 60,
    )

    ok, reason = S.acquire_daemon_lock()

    assert ok, f"a stale lock blocked startup: {reason!r}"
    assert json.loads((lock_dir / "daemon.lock").read_text())["pid"] == os.getpid()


def test_a_legacy_lock_without_a_start_time_is_stale_when_the_pid_is_newer(
        lock_dir, newer_process):
    """Locks written before the fix on macOS carry start_time null. The lock
    also records when it was written, and a process that started AFTER that
    cannot be the one that wrote it."""
    proc, before = newer_process
    _write_lock(lock_dir, pid=proc.pid, start_time=None, started_at=before - 60)

    ok, reason = S.acquire_daemon_lock()

    assert ok, f"a stale legacy lock blocked startup: {reason!r}"


def test_the_process_that_holds_the_lock_still_blocks(lock_dir, newer_process):
    """The holder, verified by start time, not assumed."""
    proc, _ = newer_process
    _write_lock(lock_dir, pid=proc.pid,
                start_time=S._process_start_time(proc.pid), started_at=time.time())

    ok, reason = S.acquire_daemon_lock()

    assert not ok
    assert "already running" in reason.lower(), reason


def test_a_legacy_lock_whose_process_could_be_the_holder_still_blocks(
        lock_dir, newer_process):
    """No start time recorded, and the live process started BEFORE the lock
    was written, so it may well be the holder. Refuse, as before."""
    proc, _ = newer_process
    time.sleep(0.05)
    _write_lock(lock_dir, pid=proc.pid, start_time=None, started_at=time.time())

    ok, reason = S.acquire_daemon_lock()

    assert not ok
    assert "running" in reason.lower(), reason


def test_the_start_time_is_known_on_this_platform():
    """None here is the defect: every lock then carries null and PID reuse
    cannot be told apart from the holder."""
    assert S._process_start_time(os.getpid()) is not None
