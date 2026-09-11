"""A job whose bookkeeping fails must not take down the tick.

THE DEFECT
----------
`mark_job_run` and `append_history` were called bare at all four exit paths of
`execute_job` (blocked / timeout / error / success). Either can raise —
`save_cron_jobs` on a full disk, `next_run_time` on an invalid
`gateway.cron_timezone` — and the exception escaped into the scheduler's
`asyncio.gather(..., return_exceptions=True)`.

Two consequences: the history row and the failure notification for that job
were SKIPPED (both come after `mark_job_run`), and the tick logged a bare
"Unexpected error executing cron job".

⚠ WHAT THIS FILE DOES *NOT* TEST
---------------------------------
It does not test that the job stops re-executing, because this fix does not
stop that. When `mark_job_run` fails, `next_run` is not advanced in the store,
the job is still due on the next tick, and its command runs again. Guarding
the exception makes that loud instead of silent.

The runaway is stopped by the in-process occurrence memory — see
`tests/test_cron_occurrence_memory.py`. Keeping the two apart is deliberate: a
test file named for a fix it does not perform is how a half-fix gets mistaken
for a whole one.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import prometheus.gateway.cron_scheduler as sched  # noqa: E402


@pytest.fixture
def history():
    return []


@pytest.fixture
def scheduler(monkeypatch, history, tmp_path):
    """Patch the bookkeeping sinks; leave `_record_run` itself real."""
    monkeypatch.setattr(sched, "append_history", lambda entry: history.append(entry))

    async def _no_notify(entry):
        return None

    monkeypatch.setattr(sched, "_maybe_notify_failure", _no_notify)
    monkeypatch.setattr(sched, "vet_cron_command", lambda cmd, cwd=None: (True, ""))
    monkeypatch.setattr(sched, "resolve_cron_cwd", lambda cwd: str(tmp_path))
    return sched


def _job(name: str, command: str = "true") -> dict:
    return {"name": name, "command": command, "schedule": "*/5 * * * *",
            "enabled": True, "next_run": "2020-01-01T00:00:00+00:00"}


def test_a_persist_failure_does_not_escape_execute_job(scheduler, monkeypatch, caplog):
    """The exception must be contained and NAMED, not propagated."""
    def _boom(name, *, success):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(sched, "mark_job_run", _boom)

    with caplog.at_level(logging.ERROR, logger=sched.__name__):
        entry = asyncio.run(sched.execute_job(_job("nightly")))

    assert entry["status"] == "success", entry
    assert "could not persist last_run/next_run" in caplog.text
    assert "No space left on device" in caplog.text
    assert "nightly" in caplog.text


def test_the_history_row_still_lands_when_persistence_fails(
    scheduler, history, monkeypatch
):
    """`append_history` came AFTER `mark_job_run` and was skipped with it.

    The history row is the only durable record that the command ran at all;
    losing it to an unrelated failure is how a run becomes invisible.
    """
    monkeypatch.setattr(
        sched, "mark_job_run",
        lambda name, *, success: (_ for _ in ()).throw(OSError("disk")),
    )

    asyncio.run(sched.execute_job(_job("nightly")))

    assert len(history) == 1, "the history row was lost with the persist failure"
    assert history[0]["name"] == "nightly"


def test_the_failure_notification_still_fires(scheduler, monkeypatch):
    """Same reasoning: it sits after the bookkeeping and went down with it."""
    notified = []

    async def _notify(entry):
        notified.append(entry)

    monkeypatch.setattr(sched, "_maybe_notify_failure", _notify)
    monkeypatch.setattr(
        sched, "mark_job_run",
        lambda name, *, success: (_ for _ in ()).throw(OSError("disk")),
    )

    asyncio.run(sched.execute_job(_job("nightly", command="false")))

    assert notified, "the failure notification was skipped"


def test_other_jobs_in_the_same_tick_still_run(scheduler, monkeypatch, history):
    """One job's broken bookkeeping must not cost the others their turn.

    `asyncio.gather(return_exceptions=True)` already prevented one raise from
    cancelling siblings, so this asserts the property end to end rather than
    assuming gather's semantics hold at this call site.
    """
    def _selective(name, *, success):
        if name == "broken":
            raise OSError(28, "No space left on device")

    monkeypatch.setattr(sched, "mark_job_run", _selective)

    async def go():
        return await asyncio.gather(
            sched.execute_job(_job("broken")),
            sched.execute_job(_job("healthy-a")),
            sched.execute_job(_job("healthy-b")),
            return_exceptions=True,
        )

    results = asyncio.run(go())

    raised = [r for r in results if isinstance(r, BaseException)]
    assert not raised, f"a persist failure escaped into the tick: {raised}"
    assert sorted(h["name"] for h in history) == [
        "broken", "healthy-a", "healthy-b",
    ], history


def test_a_history_failure_does_not_lose_the_run_either(scheduler, monkeypatch, caplog):
    """Both sinks are guarded independently.

    Guarding only `mark_job_run` would leave `append_history` able to take the
    tick down — the same defect, one line lower.
    """
    monkeypatch.setattr(sched, "mark_job_run", lambda name, *, success: None)
    monkeypatch.setattr(
        sched, "append_history",
        lambda entry: (_ for _ in ()).throw(OSError("disk")),
    )

    with caplog.at_level(logging.ERROR, logger=sched.__name__):
        entry = asyncio.run(sched.execute_job(_job("nightly")))

    assert entry["status"] == "success"
    assert "could not append the history row" in caplog.text
