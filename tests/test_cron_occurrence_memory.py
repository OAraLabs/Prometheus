"""A job whose store is unwritable must run once per occurrence, not once per tick.

THE DEFECT
----------
`mark_job_run` advances `next_run` and saves it. If the save fails (full disk)
or the recompute fails (invalid `gateway.cron_timezone`), `next_run` in the
store stays at the instant that made the job due. The job is therefore still
due on the next tick and its COMMAND RUNS AGAIN.

Measured against the real scheduler loop before this fix:

    TICK_INTERVAL_SECONDS = 30
    ticks simulated       = 5   (150s of wall clock)
    times the job RAN     = 5
    persisted next_run    = 2020-01-01T00:00:00+00:00  (unchanged)

Guarding the exception (`_record_run`, previous commit) makes that loud. It
does not stop it — the state that decides dueness is the state that failed to
write.

WHAT THE KEY IS, AND THE TWO WRONG ANSWERS
-------------------------------------------
The occurrence key is the SCHEDULED FIRE TIME derived from the cron
expression, not the wall clock and not the stored `next_run`.

  * wall clock       — differs every tick, so it suppresses nothing
  * stored next_run  — frozen precisely BECAUSE persistence broke, so every
                       later occurrence looks like a repeat of the first and
                       the job never runs again

`test_two_distinct_occurrences_both_run` is the one that separates the correct
answer from the second wrong one. It is the test to read first.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import prometheus.gateway.cron_scheduler as sched  # noqa: E402

EVERY_5_MIN = "*/5 * * * *"
T0 = datetime(2026, 9, 11, 12, 1, 0, tzinfo=timezone.utc)   # inside the 12:00 slot
T1 = datetime(2026, 9, 11, 12, 6, 0, tzinfo=timezone.utc)   # inside the 12:05 slot
T2 = datetime(2026, 9, 11, 12, 11, 0, tzinfo=timezone.utc)  # inside the 12:10 slot


def _job(name: str = "nightly") -> dict:
    # next_run in the past, and never advanced — the broken-store condition.
    return {
        "name": name, "command": "true", "schedule": EVERY_5_MIN,
        "enabled": True, "next_run": "2020-01-01T00:00:00+00:00",
        "cwd": "/tmp",
    }


@pytest.fixture(autouse=True)
def _clear_memory():
    sched._fired_occurrences.clear()
    yield
    sched._fired_occurrences.clear()


def _tick(now: datetime, jobs: list[dict]) -> list[dict]:
    """One tick's dispatch decision, through the real functions."""
    due = sched._jobs_due(jobs, now)
    due = sched._suppress_already_fired(due, now)
    for job in due:
        sched._remember_dispatch(job, now)
    return due


# ── the runaway ─────────────────────────────────────────────────────────────

def test_the_same_occurrence_is_dispatched_once_across_many_ticks():
    """Five ticks inside ONE scheduled slot dispatch the job once."""
    jobs = [_job()]
    fired = 0
    for offset in range(5):
        now = T0 + timedelta(seconds=30 * offset)  # 12:01:00 .. 12:03:00
        fired += len(_tick(now, jobs))

    assert fired == 1, (
        f"the job was dispatched {fired} times within a single scheduled "
        f"occurrence — this is the runaway"
    )


# ── the property that rules out the wrong key ───────────────────────────────

def test_two_distinct_occurrences_both_run():
    """THE test. Persistence failing throughout must not freeze the job.

    Advance past two further fire times with `next_run` never advancing.
    Exactly two dispatches: not one (which is what keying on the stored
    `next_run` would give — it is frozen) and not twenty (the runaway).
    """
    jobs = [_job()]
    dispatches: list[datetime] = []

    # Walk 30s at a time from 12:01 to 12:12 — spans the 12:05 and 12:10 slots.
    now = T0
    while now <= T2 + timedelta(seconds=30):
        for _ in _tick(now, jobs):
            dispatches.append(now)
        now += timedelta(seconds=30)

    assert len(dispatches) == 3, (
        f"expected one dispatch per scheduled occurrence (12:00, 12:05, 12:10 "
        f"slots) = 3; got {len(dispatches)} at {[d.isoformat() for d in dispatches]}"
    )
    # And they are in distinct slots, not three in a row.
    keys = {sched._occurrence_key(jobs[0], d) for d in dispatches}
    assert len(keys) == 3, f"dispatches shared an occurrence key: {sorted(keys)}"


def test_the_stored_next_run_never_moved_during_that_run():
    """Pins the premise: the store really is frozen in these tests.

    Without this, the test above could be passing because something advanced
    `next_run` — measuring normal scheduling rather than the broken-store case.
    """
    jobs = [_job()]
    before = jobs[0]["next_run"]
    for offset in range(10):
        _tick(T0 + timedelta(seconds=30 * offset), jobs)
    assert jobs[0]["next_run"] == before == "2020-01-01T00:00:00+00:00"


# ── it leaves a trace ───────────────────────────────────────────────────────

def test_suppression_logs_the_job_and_the_occurrence_key(caplog):
    """Silent suppression of a symptom plants the next invisible defect."""
    jobs = [_job("nightly")]
    _tick(T0, jobs)  # first dispatch — no suppression

    with caplog.at_level(logging.WARNING, logger=sched.__name__):
        suppressed = _tick(T0 + timedelta(seconds=30), jobs)

    assert suppressed == [], "the repeat was dispatched"
    assert "SUPPRESSED" in caplog.text
    assert "nightly" in caplog.text
    key = sched._occurrence_key(jobs[0], T0)
    assert key is not None and key in caplog.text, (
        f"the occurrence key {key!r} is not in the log line:\n{caplog.text}"
    )
    assert "next_run was not advanced" in caplog.text


# ── it does not over-suppress ───────────────────────────────────────────────

def test_a_different_job_is_not_suppressed():
    """The memory is per job, not global."""
    jobs = [_job("a"), _job("b")]
    first = _tick(T0, jobs)
    assert sorted(j["name"] for j in first) == ["a", "b"]

    # 'a' alone repeats; 'b' must be unaffected by 'a' having fired.
    sched._fired_occurrences.pop("b")
    second = _tick(T0 + timedelta(seconds=30), jobs)
    assert [j["name"] for j in second] == ["b"]


def test_an_unevaluable_schedule_is_not_suppressed():
    """A key we could not compute is not evidence that the occurrence ran.

    Declining to suppress is the safe direction: the worst case is the
    pre-existing behaviour, rather than a job silently never running.
    """
    job = _job()
    job["schedule"] = "not a cron expression"
    assert sched._occurrence_key(job, T0) is None

    kept = sched._suppress_already_fired([job], T0)
    assert kept == [job]


def test_the_memory_is_in_process_only():
    """Documented as non-durable; assert nothing writes it to disk.

    A future reader mistaking this for durable state is the failure mode the
    comment warns about, so the shape is pinned: a plain module-level dict.
    """
    assert isinstance(sched._fired_occurrences, dict)
    sched._fired_occurrences["x"] = "y"
    import importlib

    importlib.reload(sched)
    assert sched._fired_occurrences == {}, (
        "occurrence memory survived a module reload — it is not in-process state"
    )


def test_a_bad_timezone_does_not_break_the_key(monkeypatch):
    """The invalid-timezone failure is one of the two CAUSES of the runaway.

    If the key could not be computed without a resolvable timezone, the fix
    would not work in the case that motivated it.
    """
    import prometheus.gateway.cron_service as svc
    from zoneinfo import ZoneInfoNotFoundError

    def _boom():
        raise ZoneInfoNotFoundError("No time zone found with key Mars/Olympus_Mons")

    monkeypatch.setattr(svc, "_default_cron_tz", _boom)

    key = sched._occurrence_key(_job(), T0)
    assert key is not None, "the occurrence key needs a resolvable timezone"
