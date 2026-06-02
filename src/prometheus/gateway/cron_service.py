"""Local cron-style registry helpers.

Source: Adapted from OpenHarness services/cron.py (MIT).
Original path: OpenHarness/src/openharness/services/cron.py
Modified: Import paths changed to prometheus.config.paths.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from croniter import croniter

from prometheus.config.paths import get_cron_registry_path

DEFAULT_CRON_TIMEZONE = "America/New_York"


def load_cron_jobs() -> list[dict[str, Any]]:
    """Load stored cron jobs."""
    path = get_cron_registry_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def save_cron_jobs(jobs: list[dict[str, Any]]) -> None:
    """Persist cron jobs to disk."""
    path = get_cron_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jobs, indent=2) + "\n", encoding="utf-8")


def validate_cron_expression(expression: str) -> bool:
    """Return True if the expression is a valid cron schedule."""
    return croniter.is_valid(expression)


def _default_cron_tz() -> ZoneInfo:
    """Resolve the cron timezone.

    Reads ``gateway.cron_timezone`` from prometheus.yaml when the config file is
    reachable; otherwise defaults to ``America/New_York``. A missing config is a
    normal condition (use the default), but a malformed config or an invalid
    timezone name is allowed to raise — no silent fallback.
    """
    tz_name = DEFAULT_CRON_TIMEZONE
    cfg_path = Path("config/prometheus.yaml")
    if not cfg_path.exists():
        from prometheus.config.paths import get_config_dir

        cfg_path = get_config_dir() / "prometheus.yaml"
    if cfg_path.exists():
        import yaml

        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        configured = (data.get("gateway") or {}).get("cron_timezone")
        if configured:
            tz_name = str(configured)
    return ZoneInfo(tz_name)


def next_run_time(
    expression: str,
    base: datetime | None = None,
    tz: ZoneInfo | None = None,
) -> datetime:
    """Return the next run time for a cron expression, evaluated in local time.

    The expression is interpreted in *tz* (default: ``gateway.cron_timezone`` or
    ``America/New_York``), NOT UTC — so ``"30 7 * * *"`` means 07:30 *local*.
    A naive *base* is treated as local; an aware *base* (e.g. the UTC ``now``
    that ``mark_job_run`` passes) is converted into *tz* first. The result is
    tz-aware, so ``isoformat()`` preserves the offset and the scheduler's
    ``next_run <= datetime.now(timezone.utc)`` compares the correct instant.
    """
    tz = tz or _default_cron_tz()
    if base is None:
        base = datetime.now(tz)
    elif base.tzinfo is None:
        base = base.replace(tzinfo=tz)
    else:
        base = base.astimezone(tz)
    return croniter(expression, base).get_next(datetime)


def upsert_cron_job(job: dict[str, Any]) -> None:
    """Insert or replace one cron job.

    Automatically sets ``enabled`` to True and computes ``next_run`` when the
    schedule is a valid cron expression.
    """
    job.setdefault("enabled", True)
    job.setdefault("created_at", datetime.now(timezone.utc).isoformat())

    schedule = job.get("schedule", "")
    if validate_cron_expression(schedule):
        job["next_run"] = next_run_time(schedule).isoformat()

    jobs = [
        existing
        for existing in load_cron_jobs()
        if existing.get("name") != job.get("name")
    ]
    jobs.append(job)
    jobs.sort(key=lambda item: str(item.get("name", "")))
    save_cron_jobs(jobs)


def delete_cron_job(name: str) -> bool:
    """Delete one cron job by name."""
    jobs = load_cron_jobs()
    filtered = [job for job in jobs if job.get("name") != name]
    if len(filtered) == len(jobs):
        return False
    save_cron_jobs(filtered)
    return True


def get_cron_job(name: str) -> dict[str, Any] | None:
    """Return one cron job by name."""
    for job in load_cron_jobs():
        if job.get("name") == name:
            return job
    return None


def set_job_enabled(name: str, enabled: bool) -> bool:
    """Enable or disable a cron job. Returns False if job not found."""
    jobs = load_cron_jobs()
    for job in jobs:
        if job.get("name") == name:
            job["enabled"] = enabled
            save_cron_jobs(jobs)
            return True
    return False


def mark_job_run(name: str, *, success: bool) -> None:
    """Update last_run and recompute next_run after a job executes."""
    jobs = load_cron_jobs()
    now = datetime.now(timezone.utc)
    for job in jobs:
        if job.get("name") == name:
            job["last_run"] = now.isoformat()
            job["last_status"] = "success" if success else "failed"
            schedule = job.get("schedule", "")
            if validate_cron_expression(schedule):
                job["next_run"] = next_run_time(schedule, now).isoformat()
            save_cron_jobs(jobs)
            return
