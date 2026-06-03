"""Background cron scheduler — embeddable async loop.

Source: Adapted from OpenHarness services/cron_scheduler.py (MIT).
Original path: OpenHarness/src/openharness/services/cron_scheduler.py
Modified: Import paths changed to prometheus.*; removed standalone daemon fork
(Prometheus daemon.py handles lifecycle).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prometheus.config.paths import get_data_dir, get_logs_dir
from prometheus.gateway.cron_service import (
    load_cron_jobs,
    mark_job_run,
    validate_cron_expression,
)

logger = logging.getLogger(__name__)

TICK_INTERVAL_SECONDS = 30
"""How often the scheduler checks for due jobs."""

# ---------------------------------------------------------------------------
# Failure notifications (audit follow-up: cron failures must surface)
# ---------------------------------------------------------------------------
#
# The Heartbeat task-notification path watches BackgroundTaskManager only —
# cron jobs run via asyncio.create_subprocess_exec and never appear there. So
# without this hook, a failing cron job (like a broken daily briefing) is
# silent unless the user explicitly polls cron_list. The daemon wires a
# notifier at startup; if none is set, this is a no-op.

NOTIFY_COOLDOWN_SECONDS = 3600
"""Per-job throttle. Without it, a chronically broken hourly job would push
24 identical messages a day. One per hour per job is enough to surface
regressions without spamming."""

_NOTIFIER_GATEWAY: Any | None = None
_NOTIFIER_CHAT_ID: int | None = None
_LAST_NOTIFY: dict[str, float] = {}


def set_cron_notifier(gateway: Any | None, chat_id: int | None) -> None:
    """Register a Telegram-style gateway + destination chat id.

    Passing ``None`` for either argument disables notifications (the daemon
    does that when no chat target is configured). Subsequent successful calls
    replace the prior wiring; the per-job throttle map is NOT cleared so a
    restart-driven re-wiring doesn't re-spam in-flight failures.
    """
    global _NOTIFIER_GATEWAY, _NOTIFIER_CHAT_ID
    _NOTIFIER_GATEWAY = gateway
    _NOTIFIER_CHAT_ID = chat_id


async def _maybe_notify_failure(entry: dict[str, Any]) -> None:
    """Push a failure message if a notifier is wired and the throttle allows.

    No-op on success, no-op when the gateway/chat are unset, and throttled per
    job name by NOTIFY_COOLDOWN_SECONDS. Send errors are logged, never raised,
    so a flaky Telegram can't kill the scheduler.
    """
    if entry.get("status") == "success":
        return
    if _NOTIFIER_GATEWAY is None or _NOTIFIER_CHAT_ID is None:
        return
    name = str(entry.get("name", "?"))
    now = time.time()
    if now - _LAST_NOTIFY.get(name, 0) < NOTIFY_COOLDOWN_SECONDS:
        logger.info("Cron failure notification throttled for %r", name)
        return
    stderr_tail = "\n".join(
        (entry.get("stderr") or "").strip().splitlines()[-5:]
    ) or "(no stderr)"
    cmd_preview = str(entry.get("command", ""))[:120]
    text = (
        f"⚠️ Cron job failed: {name}\n"
        f"status={entry.get('status')} rc={entry.get('returncode')}\n"
        f"command: {cmd_preview}\n"
        f"stderr (last 5 lines):\n{stderr_tail}"
    )
    try:
        await _NOTIFIER_GATEWAY.send(_NOTIFIER_CHAT_ID, text)
        _LAST_NOTIFY[name] = now
        logger.info("Cron failure notification sent for %r", name)
    except Exception:
        logger.exception("Failed to send cron failure notification for %r", name)


# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------


def get_history_path() -> Path:
    """Return the path to the cron execution history file."""
    return get_data_dir() / "cron_history.jsonl"


def append_history(entry: dict[str, Any]) -> None:
    """Append one execution record to the history log."""
    path = get_history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def load_history(
    *, limit: int = 50, job_name: str | None = None
) -> list[dict[str, Any]]:
    """Load the most recent execution history entries."""
    path = get_history_path()
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if job_name and entry.get("name") != job_name:
            continue
        entries.append(entry)
    return entries[-limit:]


# ---------------------------------------------------------------------------
# PID file helpers
# ---------------------------------------------------------------------------


def get_pid_path() -> Path:
    """Return the scheduler PID file path."""
    return get_data_dir() / "cron_scheduler.pid"


def read_pid() -> int | None:
    """Read the PID of a running scheduler, or None."""
    path = get_pid_path()
    if not path.exists():
        return None
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None
    try:
        os.kill(pid, 0)
    except OSError:
        logger.debug("Removed stale scheduler PID file (pid=%d)", pid)
        path.unlink(missing_ok=True)
        return None
    return pid


def write_pid() -> None:
    """Write the current process PID."""
    path = get_pid_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(os.getpid()) + "\n", encoding="utf-8")


def remove_pid() -> None:
    """Remove the PID file."""
    get_pid_path().unlink(missing_ok=True)


def is_scheduler_running() -> bool:
    """Return True if a scheduler process is alive."""
    return read_pid() is not None


def stop_scheduler() -> bool:
    """Send SIGTERM to the running scheduler. Returns True if killed."""
    pid = read_pid()
    if pid is None:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        remove_pid()
        return False
    for _ in range(10):
        try:
            os.kill(pid, 0)
        except OSError:
            remove_pid()
            return True
        time.sleep(0.2)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    remove_pid()
    return True


# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------


async def execute_job(job: dict[str, Any]) -> dict[str, Any]:
    """Run a single cron job and return a history entry."""
    name = job["name"]
    command = job["command"]
    cwd = Path(job.get("cwd") or ".").expanduser()
    started_at = datetime.now(timezone.utc)

    logger.info("Executing cron job %r: %s", name, command)
    try:
        process = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "-lc",
            command,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=300,
        )
    except asyncio.TimeoutError:
        try:
            process.kill()
            await process.wait()
        except Exception:
            pass
        entry = {
            "name": name,
            "command": command,
            "started_at": started_at.isoformat(),
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "returncode": -1,
            "status": "timeout",
            "stdout": "",
            "stderr": "Job timed out after 300s",
        }
        mark_job_run(name, success=False)
        append_history(entry)
        await _maybe_notify_failure(entry)
        return entry
    except Exception as exc:
        entry = {
            "name": name,
            "command": command,
            "started_at": started_at.isoformat(),
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "returncode": -1,
            "status": "error",
            "stdout": "",
            "stderr": str(exc),
        }
        mark_job_run(name, success=False)
        append_history(entry)
        await _maybe_notify_failure(entry)
        return entry

    success = process.returncode == 0
    entry = {
        "name": name,
        "command": command,
        "started_at": started_at.isoformat(),
        "ended_at": datetime.now(timezone.utc).isoformat(),
        "returncode": process.returncode,
        "status": "success" if success else "failed",
        "stdout": (
            stdout.decode("utf-8", errors="replace")[-2000:] if stdout else ""
        ),
        "stderr": (
            stderr.decode("utf-8", errors="replace")[-2000:] if stderr else ""
        ),
    }
    mark_job_run(name, success=success)
    append_history(entry)
    await _maybe_notify_failure(entry)
    logger.info(
        "Job %r finished: %s (rc=%s)", name, entry["status"], process.returncode
    )
    return entry


# ---------------------------------------------------------------------------
# Scheduler loop
# ---------------------------------------------------------------------------


def _jobs_due(
    jobs: list[dict[str, Any]], now: datetime
) -> list[dict[str, Any]]:
    """Return jobs whose next_run is at or before *now*."""
    due: list[dict[str, Any]] = []
    for job in jobs:
        if not job.get("enabled", True):
            continue
        schedule = job.get("schedule", "")
        if not validate_cron_expression(schedule):
            continue
        next_run_str = job.get("next_run")
        if not next_run_str:
            continue
        try:
            next_run = datetime.fromisoformat(next_run_str)
            if next_run.tzinfo is None:
                next_run = next_run.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
        if next_run <= now:
            due.append(job)
    return due


async def run_scheduler_loop(*, once: bool = False) -> None:
    """Main scheduler loop. Runs until cancelled or *once* is True (test mode)."""
    shutdown = asyncio.Event()

    def _on_signal() -> None:
        logger.info("Cron scheduler received shutdown signal")
        shutdown.set()

    try:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _on_signal)
    except (NotImplementedError, RuntimeError):
        pass  # signal handlers may not work in all contexts

    write_pid()
    logger.info(
        "Cron scheduler started (pid=%d, tick=%ds)",
        os.getpid(),
        TICK_INTERVAL_SECONDS,
    )

    try:
        while not shutdown.is_set():
            now = datetime.now(timezone.utc)
            jobs = load_cron_jobs()
            due = _jobs_due(jobs, now)

            if due:
                logger.info("Tick: %d job(s) due", len(due))
                results = await asyncio.gather(
                    *(execute_job(job) for job in due), return_exceptions=True
                )
                for result in results:
                    if isinstance(result, BaseException):
                        logger.error(
                            "Unexpected error executing cron job: %s", result
                        )

            if once:
                break

            try:
                await asyncio.wait_for(
                    shutdown.wait(), timeout=TICK_INTERVAL_SECONDS
                )
            except asyncio.TimeoutError:
                pass
    finally:
        remove_pid()
        logger.info("Cron scheduler stopped")


def scheduler_status() -> dict[str, Any]:
    """Return a status dict about the scheduler."""
    pid = read_pid()
    log_path = get_logs_dir() / "cron_scheduler.log"
    jobs = load_cron_jobs()
    enabled = [j for j in jobs if j.get("enabled", True)]
    return {
        "running": pid is not None,
        "pid": pid,
        "total_jobs": len(jobs),
        "enabled_jobs": len(enabled),
        "log_file": str(log_path),
        "history_file": str(get_history_path()),
    }
