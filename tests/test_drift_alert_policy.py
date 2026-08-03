"""The merged-but-dark nudge: suppressed during maintenance, persisted across restarts.

Two defects, both about *when* the nudge fires rather than whether it dedupes
(it always did — ``_stale_notified_for`` keyed on ``tree_head``):

A. **It fired inside an armed maintenance window** — i.e. during the
   merge-to-restart gap, which is exactly when drift is expected. Every deploy
   generated one.
B. **The dedupe state was in-memory**, so it reset on restart — the precise
   event that resolves drift and therefore precedes the next one.

The subtle part is A's blind spot: suppressing must NOT mark the drift as
notified, or a merge that lands inside a window and is never followed by a
restart is silenced *forever*. These tests pin that ordering explicitly,
because it is the failure mode that would only surface weeks later.

Real SQLite windows and real state files throughout — no mocks. The bug class
here is state that does not survive a process boundary, which a mock cannot
observe.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from prometheus.gateway.heartbeat import Heartbeat


def _window(path, *, minutes: float | None) -> str:
    """Write a maintenance row. ``minutes=None`` writes a cleared window."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS maintenance ("
        "id INTEGER PRIMARY KEY CHECK (id = 1), until_ts TEXT, reason TEXT, "
        "set_at TEXT, findings_pending INTEGER NOT NULL DEFAULT 0)"
    )
    until = (
        None
        if minutes is None
        else (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()
    )
    conn.execute("DELETE FROM maintenance")
    conn.execute(
        "INSERT INTO maintenance (id, until_ts, reason, set_at) VALUES (1, ?, ?, ?)",
        (until, "test", datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()
    return str(path)


@pytest.fixture
def hb(tmp_path, monkeypatch):
    """A Heartbeat whose state file and git HEAD are under our control."""
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path / "cfg"))

    def make(*, maintenance_db: str | None = None, boot="aaaaaaaa", tree="bbbbbbbb"):
        monkeypatch.setattr(
            "prometheus.gateway.heartbeat.git_head_sha", lambda: tree
        )
        h = Heartbeat(boot_sha=boot, maintenance_db=maintenance_db)
        h._sent: list[str] = []

        async def _notify(msg):
            h._sent.append(msg)

        h._notify = _notify  # type: ignore[method-assign]
        return h

    return make


@pytest.mark.asyncio
async def test_drift_nudges_when_no_window_is_configured(hb):
    """Unchanged out of the box: no maintenance_db means no suppression."""
    h = hb()
    await h._check_staleness()
    assert len(h._sent) == 1, "a plain drift must still nudge"
    assert "aaaaaaaa" in h._sent[0] and "bbbbbbbb" in h._sent[0]


@pytest.mark.asyncio
async def test_drift_is_suppressed_inside_an_armed_window(hb, tmp_path):
    """The deploy-window noise this PR exists to remove."""
    db = _window(tmp_path / "m.db", minutes=30)
    h = hb(maintenance_db=db)
    await h._check_staleness()
    assert h._sent == [], (
        "the nudge fired inside an armed maintenance window — that is the "
        "merge-to-restart gap, where drift is expected"
    )


@pytest.mark.asyncio
async def test_suppression_does_not_mark_the_drift_as_notified(hb, tmp_path):
    """THE BLIND SPOT. Suppressed must not mean silenced-forever."""
    db = _window(tmp_path / "m.db", minutes=30)
    h = hb(maintenance_db=db)
    await h._check_staleness()

    assert h._stale_notified_for is None, (
        "suppression recorded the tree_head as notified — a merge landing "
        "inside a window and never followed by a restart would then be "
        "silenced permanently"
    )


@pytest.mark.asyncio
async def test_it_nudges_once_the_window_closes(hb, tmp_path):
    """Window-close re-evaluation, falling out of the ordering."""
    path = tmp_path / "m.db"
    db = _window(path, minutes=30)
    h = hb(maintenance_db=db)

    await h._check_staleness()
    assert h._sent == [], "precondition: suppressed while armed"

    _window(path, minutes=None)  # operator cleared the window
    await h._check_staleness()

    assert len(h._sent) == 1, (
        "the drift was never re-evaluated after the window closed — this is "
        "the silenced-forever case"
    )


@pytest.mark.asyncio
async def test_an_expired_window_does_not_suppress(hb, tmp_path):
    """until_ts in the past is not an armed window."""
    db = _window(tmp_path / "m.db", minutes=-5)
    h = hb(maintenance_db=db)
    await h._check_staleness()
    assert len(h._sent) == 1, "an expired window suppressed a nudge"


@pytest.mark.asyncio
async def test_unreadable_window_fails_open(hb, tmp_path):
    """Suppression is a convenience, so it degrades toward noisy, not silent."""
    missing = str(tmp_path / "does-not-exist.db")
    h = hb(maintenance_db=missing)
    await h._check_staleness()
    assert len(h._sent) == 1, (
        "an unreadable maintenance window suppressed the nudge — swallowing a "
        "drift alert is worse than a redundant one"
    )


@pytest.mark.asyncio
async def test_dedupe_survives_a_restart(hb, tmp_path):
    """Defect B: in-memory state reset on the very event that precedes re-drift."""
    h1 = hb()
    await h1._check_staleness()
    assert len(h1._sent) == 1

    h2 = hb()  # a fresh process, same config dir
    await h2._check_staleness()
    assert h2._sent == [], (
        "the nudge repeated after a restart for the SAME unchanged drift — "
        "the dedupe state did not survive the process boundary"
    )


@pytest.mark.asyncio
async def test_a_new_commit_nudges_again(hb):
    """The dedupe must not swallow genuinely new drift."""
    h = hb(tree="bbbbbbbb")
    await h._check_staleness()
    assert len(h._sent) == 1

    h2 = hb(tree="cccccccc")  # someone merged again
    await h2._check_staleness()
    assert len(h2._sent) == 1, "a NEW tree_head must nudge even after a prior one"


@pytest.mark.asyncio
async def test_returning_to_sync_clears_the_persisted_state(hb, tmp_path):
    """After a restart onto the new code, a later drift must nudge again."""
    h = hb(tree="bbbbbbbb")
    await h._check_staleness()
    state = tmp_path / "cfg" / "data" / "drift_notified.json"
    assert json.loads(state.read_text())["notified_for"] == "bbbbbbbb"

    insync = hb(boot="bbbbbbbb", tree="bbbbbbbb")  # restarted onto it
    await insync._check_staleness()
    assert insync._sent == [], "in-sync must be silent"
    assert json.loads(state.read_text())["notified_for"] is None, (
        "returning to sync did not clear the persisted state, so a later "
        "drift onto the same SHA would be wrongly deduped"
    )
