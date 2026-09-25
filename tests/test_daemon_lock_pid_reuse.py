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

The first test below is the reproduction: a lock written the way a release
before this fix writes it, by a process that is gone (recorded here as this
test process's start time), whose PID now belongs to a newer, unrelated
process. It passes on Linux on origin/main and fails on macOS there.

The fix keeps the lock's ``start_time`` exactly as before (/proc ticks, null
on macOS) and puts the exact start time in a new ``started_epoch`` key. An
older release on the same machine reads ``start_time`` only; had it become a
number it cannot reproduce there, that release would delete a LIVE daemon's
lock as stale. The "still blocks" and "older release" tests guard that
direction: the fix must never cost the lock its holder.
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
    """THE REPRODUCTION. A lock as a release before this fix writes it (no
    started_epoch; start_time null on macOS). On macOS, origin/main refuses to
    start here."""
    proc, before = newer_process
    _write_lock(
        lock_dir,
        pid=proc.pid,
        start_time=S._process_start_time(os.getpid()),  # the dead daemon's own
        started_at=before - 60,
    )

    ok, reason = S.acquire_daemon_lock()

    assert ok, f"a stale lock blocked startup: {reason!r}"
    assert json.loads((lock_dir / "daemon.lock").read_text())["pid"] == os.getpid()


def test_a_current_format_lock_whose_pid_is_now_newer_is_stale(lock_dir, newer_process):
    """The same, for a lock this release writes: exact start times differ."""
    proc, before = newer_process
    _write_lock(
        lock_dir,
        pid=proc.pid,
        start_time=S._process_start_time(os.getpid()),
        started_epoch=S._process_started_epoch(os.getpid()),
        started_at=before - 60,
    )

    ok, reason = S.acquire_daemon_lock()

    assert ok, f"a stale lock blocked startup: {reason!r}"


def test_the_process_that_holds_the_lock_still_blocks(lock_dir, newer_process):
    """The holder, verified by its exact start time, not assumed."""
    proc, _ = newer_process
    _write_lock(lock_dir, pid=proc.pid,
                start_time=S._process_start_time(proc.pid),
                started_epoch=S._process_started_epoch(proc.pid),
                started_at=time.time())

    ok, reason = S.acquire_daemon_lock()

    assert not ok
    assert "already running" in reason.lower(), reason


def _pre_fix_start_time(pid: int) -> float | None:
    """A FROZEN copy of the pre-fix reader (origin/main b1a697f), deliberately
    not the module's: this test must still fail if the module's reader changes."""
    from pathlib import Path

    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat.rsplit(")", 1)[-1].split()
        return float(fields[19])
    except (OSError, IndexError, ValueError):
        return None


def _pre_fix_decision(lock: dict) -> str:
    """What a release before this fix concludes about a lock: its decision
    logic, verbatim, with its own reader."""
    pid = lock.get("pid", -1)
    if not S._process_alive(pid):
        return "stale"
    old_start = lock.get("start_time")
    current = _pre_fix_start_time(pid)
    if old_start is not None and current is not None and old_start == current:
        return "running"
    if old_start is None:
        return "running"  # "appears to be running"
    return "stale"


def test_an_older_release_still_sees_this_releases_lock_as_held(lock_dir):
    """Two installs can share one config dir (a checkout and a Homebrew
    install). This release's lock must not look stale to the older one."""
    ok, reason = S.acquire_daemon_lock()
    assert ok, reason
    lock = json.loads((lock_dir / "daemon.lock").read_text())

    assert lock["start_time"] == _pre_fix_start_time(os.getpid())
    assert isinstance(lock["started_epoch"], float)
    assert _pre_fix_decision(lock) == "running"


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
    """None here is the defect: PID reuse cannot be told apart from the holder."""
    assert S._process_started_epoch(os.getpid()) is not None
