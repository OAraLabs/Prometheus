"""Scoped daemon lock — prevents duplicate daemon instances.

Donor pattern: NousResearch/hermes-agent gateway/status.py.
Adapted for Prometheus: lock at ~/.prometheus/daemon.lock, stale detection by
process start time (/proc on Linux, sysctl on macOS).
"""

from __future__ import annotations

import json
import logging
import os
import struct
import sys
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
    """The process's start time, as recorded in the lock and compared for equality.

    Linux: ``starttime`` from /proc/<pid>/stat, in clock ticks since boot.
    macOS: the kernel's start time from sysctl, in epoch seconds.

    This used to be the /proc read alone. macOS has no /proc, so it returned
    None for every process: every lock there recorded ``start_time: null``,
    and a stale lock whose PID had been reused by any other process blocked
    the daemon from starting ("appears to be running") until the lock was
    deleted by hand (WP-X.24).
    """
    if sys.platform == "darwin":
        return _darwin_start_time(pid)
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        # Field 22 (0-indexed: 21) is starttime in clock ticks
        fields = stat.rsplit(")", 1)[-1].split()
        return float(fields[19])  # starttime is field 22, after the ')' split index 19
    except (OSError, IndexError, ValueError):
        return None


def _darwin_start_time(pid: int) -> float | None:
    """Start time of ``pid`` on macOS, epoch seconds with microseconds.

    ``sysctl({CTL_KERN, KERN_PROC, KERN_PROC_PID, pid})`` fills a
    ``struct kinfo_proc``, whose first member is ``kp_proc``, whose first
    member is the ``p_starttime`` timeval: so the time is at offset 0. No
    subprocess, no dependency. An empty result means no such process.
    """
    try:
        import ctypes
        import ctypes.util

        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        mib = (ctypes.c_int * 4)(1, 14, 1, pid)  # CTL_KERN, KERN_PROC, KERN_PROC_PID
        buf = ctypes.create_string_buffer(1024)  # sizeof(struct kinfo_proc) is 648
        size = ctypes.c_size_t(ctypes.sizeof(buf))
        if libc.sysctl(mib, 4, buf, ctypes.byref(size), None, ctypes.c_size_t(0)) != 0:
            return None
        if size.value < 16:
            return None
        sec, usec = struct.unpack_from("@qi", buf, 0)
        return sec + usec / 1_000_000
    except (OSError, AttributeError, ValueError, struct.error):
        return None


def _process_started_epoch(pid: int) -> float | None:
    """When ``pid`` started, in epoch seconds, for comparing with ``started_at``."""
    if sys.platform == "darwin":
        return _darwin_start_time(pid)
    ticks = _process_start_time(pid)
    if ticks is None:
        return None
    try:
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith("btime "):
                return int(line.split()[1]) + ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        pass
    return None


def _started_after_lock_was_written(pid: int, lock_written_at: object) -> bool:
    """True when the live ``pid`` started after the lock was written.

    Such a process cannot be the one that wrote the lock: the PID was reused.
    A second of slack keeps clock granularity from ever reading the real
    holder, which started before it wrote the lock, as a newcomer.
    """
    if not isinstance(lock_written_at, (int, float)):
        return False
    started = _process_started_epoch(pid)
    return started is not None and started > float(lock_written_at) + 1.0


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
                # No start time in the lock: written on macOS before WP-X.24,
                # or where it could not be read. The lock still says WHEN it
                # was written, and a process that started later is not its
                # writer. Otherwise it can't be verified: assume it's running.
                if not _started_after_lock_was_written(old_pid, existing.get("started_at")):
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
