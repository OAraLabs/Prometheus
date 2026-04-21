"""Scoped daemon lock — prevents duplicate daemon instances.

Donor pattern: NousResearch/hermes-agent gateway/status.py.
Adapted for Prometheus: lock at ~/.prometheus/daemon.lock, /proc-based stale detection.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from prometheus.config.paths import get_config_dir

logger = logging.getLogger(__name__)


def _lock_path() -> Path:
    return get_config_dir() / "daemon.lock"


def _read_lock() -> dict | None:
    """Read lock file, return record dict or None."""
    p = _lock_path()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _process_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but we can't signal it
    return True


def _process_start_time(pid: int) -> float | None:
    """Read process start time from /proc on Linux."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        # Field 22 (0-indexed: 21) is starttime in clock ticks
        fields = stat.rsplit(")", 1)[-1].split()
        return float(fields[19])  # starttime is field 22, after the ')' split index 19
    except (OSError, IndexError, ValueError):
        return None


def acquire_daemon_lock() -> tuple[bool, str]:
    """Acquire the daemon lock.

    Returns (True, "") on success, or (False, reason) if another daemon is running.
    """
    existing = _read_lock()
    if existing is not None:
        old_pid = existing.get("pid", -1)
        if _process_alive(old_pid):
            # Verify it's actually the same process (not a recycled PID)
            old_start = existing.get("start_time")
            current_start = _process_start_time(old_pid)
            if old_start is not None and current_start is not None and old_start == current_start:
                return False, f"Daemon already running (PID {old_pid})"
            if old_start is None:
                # Can't verify — assume it's running
                return False, f"Daemon appears to be running (PID {old_pid})"
        # Stale lock — clean it up
        logger.info("Removing stale daemon lock (PID %d no longer running)", old_pid)
        _lock_path().unlink(missing_ok=True)

    # Write new lock atomically
    record = {
        "pid": os.getpid(),
        "start_time": _process_start_time(os.getpid()),
        "started_at": time.time(),
        "argv": " ".join(os.sys.argv),
    }
    try:
        fd = os.open(str(_lock_path()), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(fd, json.dumps(record, indent=2).encode())
        os.close(fd)
    except FileExistsError:
        return False, "Lock file appeared during acquisition (race condition)"

    logger.info("Daemon lock acquired (PID %d)", os.getpid())
    return True, ""


def release_daemon_lock() -> None:
    """Release the daemon lock if we own it."""
    existing = _read_lock()
    if existing is None:
        return
    if existing.get("pid") == os.getpid():
        _lock_path().unlink(missing_ok=True)
        logger.info("Daemon lock released")
    else:
        logger.warning(
            "Not releasing lock — owned by PID %d, we are PID %d",
            existing.get("pid", -1),
            os.getpid(),
        )
