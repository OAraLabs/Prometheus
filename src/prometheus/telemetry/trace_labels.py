"""Trace labels: what a stored tool call's "pass" does and does not say.

A golden trace was labelled "pass" when its tool call ran without an error.
That is one fact about the CALL. It says nothing about whether the turn the
call sat in got anywhere, and the distillation audit (2026-10-30) found
"pass" read as task success: re-judged by a strong model, about a fifth of
the passes were wrong calls, and many sat in turns the loop force-stopped.
So the trace stores carry three labels, kept apart:

=================  ==========================================================
``tool_ok``        1/0: the tool call executed without error. The old "pass",
                   and ``tool_calls.success``'s meaning. Known at write time.
``turn_outcome``   How the turn the call sat in ENDED (vocabulary below), from
                   the turn's own end state. Known once the turn ends.
``judged_ok``      1/0, or NULL: a judge's verdict on the call, filled later
                   (``record_judgement``), with ``judged_by`` / ``judged_at``.
=================  ==========================================================

A success RATE reads ``turn_outcome`` or ``judged_ok``, never ``tool_ok``
alone. A tool's own reliability (did ``bash`` run) is a ``tool_ok`` rate and
says so in its name.

``turn_outcome`` (the turn's end state, from the ``turns`` row):

=============  ==============================================================
``completed``  the turn ended on its own: the model stopped, no loop stop,
               no error. NOT "the task succeeded": nobody judged it.
``stopped``    the loop or the user stopped it (circuit breaker, iteration
               cap, repeat guard, Stop button, ...): ``FORCED_STOP_REASONS``.
``errored``    it ended in an error: ``ERROR_TERMINAL_REASONS`` or an error
               response.
``abandoned``  it started and never recorded an end (the process died or
               restarted mid-turn), and ``ABANDONED_AFTER_SECONDS`` passed.
``unknown``    the call cannot be linked to a turn: written before telemetry
               v2 (no ``turn_id``), or its turn row is missing.
NULL           not known YET: the turn is still running.
=============  ==============================================================

THIS ``abandoned`` IS NOT ``turns.outcome = 'abandoned'``. That one is a USER
signal (telemetry/outcomes.py): no message followed a finished turn within
30 minutes, which usually means the user got what they needed, and that
module says it must never be a negative label. A turn like that ended on its
own, so its calls are ``completed`` here. The user signal stays on the turn
row (``turns.outcome``) and rides along in v2 exports as ``turn_signal``.

The labels are written in three places, all idempotent and all "only while
NULL", so none overwrites another:

* turn end: ``label_turn_calls`` (queued on the telemetry v2 writer) and
  ``pair_capture.label_turn`` stamp every call and pair of the turn;
* the heartbeat sweep: ``sweep_unended`` calls a turn that never ended
  ``abandoned``;
* ``oara trace-labels``: ``backfill_telemetry`` / ``backfill_training`` for
  rows written before this existed (dry run by default).
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

from prometheus.telemetry.outcomes import ERROR_TERMINAL_REASONS

log = logging.getLogger(__name__)

COMPLETED = "completed"
STOPPED = "stopped"
ERRORED = "errored"
ABANDONED = "abandoned"
UNKNOWN = "unknown"

#: Every value ``turn_outcome`` may hold (NULL aside: not known yet).
TURN_OUTCOMES: tuple[str, ...] = (COMPLETED, STOPPED, ERRORED, ABANDONED, UNKNOWN)

#: A turn that started this long ago and never recorded an end was cut off.
#: Generous on purpose: a worker's run is capped at an hour, and calling a
#: live turn abandoned would be a label nobody can take back.
ABANDONED_AFTER_SECONDS = 6 * 3600

#: The label columns, as each trace store adds them (additive, nullable).
#: ``tool_ok`` is NULL only on a row that predates it and was not backfilled.
LABEL_COLUMNS: list[tuple[str, str]] = [
    ("tool_ok", "INTEGER"),
    ("turn_outcome", "TEXT"),
    ("judged_ok", "INTEGER"),
    ("judged_by", "TEXT"),        # the judge's model or rubric id
    ("judged_at", "REAL"),
]


def turn_outcome(
    *,
    started_at: float | None,
    ended_at: float | None,
    forced_stop_reason: str | None,
    terminal_kind: str | None,
    now: float | None = None,
) -> str | None:
    """A turn's end state from its ``turns`` row, or None while it may still be running."""
    if ended_at is None:
        now = time.time() if now is None else now
        if started_at is not None and now - started_at >= ABANDONED_AFTER_SECONDS:
            return ABANDONED
        return None
    if forced_stop_reason in ERROR_TERMINAL_REASONS or terminal_kind == "error":
        return ERRORED
    if forced_stop_reason is not None:
        return STOPPED
    return COMPLETED


def _turn_outcome_sql(alias: str = "t") -> tuple[str, list[Any]]:
    """``turn_outcome`` as a SQL CASE over a ``turns`` row, with its bind values.

    The same rule as :func:`turn_outcome` for an ENDED turn (the unended case
    needs ``now`` and is handled by the callers).
    """
    marks = ", ".join("?" for _ in ERROR_TERMINAL_REASONS)
    sql = (
        f"CASE WHEN {alias}.forced_stop_reason IN ({marks})"
        f" OR {alias}.terminal_kind = 'error' THEN '{ERRORED}'"
        f" WHEN {alias}.forced_stop_reason IS NOT NULL THEN '{STOPPED}'"
        f" ELSE '{COMPLETED}' END"
    )
    return sql, sorted(ERROR_TERMINAL_REASONS)


# --------------------------------------------------------------------------- #
# Ongoing labelling (the daemon)
# --------------------------------------------------------------------------- #


def label_turn_calls(conn: sqlite3.Connection, turn_id: str, outcome: str) -> None:
    """Stamp ``outcome`` on every ``tool_calls`` row of ``turn_id`` not labelled yet.

    Runs on the telemetry v2 writer thread (``TelemetryV2Writer.call``), after
    the turn's end upsert, so it never waits on the turn path.
    """
    conn.execute(
        "UPDATE tool_calls SET turn_outcome = ? WHERE turn_id = ? AND turn_outcome IS NULL",
        (outcome, turn_id))


def sweep_unended(conn: sqlite3.Connection, *, now: float | None = None) -> int:
    """``abandoned`` for the calls of turns that started long ago and never ended.

    Returns how many calls were labelled. Idempotent; on the writer thread
    from the heartbeat sweep.
    """
    now = time.time() if now is None else now
    cur = conn.execute(
        "UPDATE tool_calls SET turn_outcome = ? WHERE turn_outcome IS NULL AND turn_id IN ("
        " SELECT t.turn_id FROM turns t WHERE t.ended_at IS NULL AND t.started_at < ?)",
        (ABANDONED, now - ABANDONED_AFTER_SECONDS))
    return cur.rowcount or 0


def queue_sweep_unended(telemetry: Any) -> None:
    """Heartbeat: queue :func:`sweep_unended` on the telemetry v2 writer. Never raises."""
    try:
        get = getattr(telemetry, "v2_writer", None) if telemetry is not None else None
        writer = get() if callable(get) else None
        if writer is not None:
            writer.call("trace_labels_sweep", lambda conn: sweep_unended(conn))
    except Exception:
        log.warning("trace labels: could not queue the unended-turn sweep", exc_info=True)


# --------------------------------------------------------------------------- #
# Backfill (``oara trace-labels``)
# --------------------------------------------------------------------------- #


@dataclass
class BackfillReport:
    """What a backfill found (dry run) or wrote (apply), per store."""

    store: str
    rows: int = 0
    tool_ok_set: int = 0
    turn_outcome: dict[str, int] = field(default_factory=dict)
    still_running: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "store": self.store,
            "rows": self.rows,
            "tool_ok_set": self.tool_ok_set,
            "turn_outcome": dict(sorted(self.turn_outcome.items())),
            "still_running": self.still_running,
        }


def _has_columns(conn: sqlite3.Connection, table: str, cols: list[str]) -> bool:
    have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    return all(c in have for c in cols)


def _turn_states(conn: sqlite3.Connection) -> dict[str, tuple]:
    """turn_id -> (started_at, ended_at, forced_stop_reason, terminal_kind)."""
    try:
        rows = conn.execute(
            "SELECT turn_id, started_at, ended_at, forced_stop_reason, terminal_kind FROM turns"
        ).fetchall()
    except sqlite3.DatabaseError:
        return {}
    return {r[0]: tuple(r[1:]) for r in rows}


def _outcome_for(turn_id: str | None, ts: float | None, turns: dict[str, tuple],
                 now: float) -> str | None:
    if not turn_id:
        return UNKNOWN
    state = turns.get(turn_id)
    if state is None:
        # A turn's start row is queued and its calls are written at once, so a
        # brand-new call can briefly have no turn row. Old ones never will.
        if ts is not None and now - ts < ABANDONED_AFTER_SECONDS:
            return None
        return UNKNOWN
    started_at, ended_at, reason, kind = state
    return turn_outcome(started_at=started_at, ended_at=ended_at,
                        forced_stop_reason=reason, terminal_kind=kind, now=now)


def backfill_telemetry(conn: sqlite3.Connection, *, apply: bool,
                       now: float | None = None) -> BackfillReport:
    """Label ``tool_calls`` rows that have no labels yet. Never rewrites a set label.

    ``tool_ok`` copies ``success`` (same meaning, new name). ``turn_outcome``
    comes from the row's ``turns`` row; a row with no ``turn_id`` (written
    before telemetry v2) or no turn row is ``unknown``. A dry run counts
    what it would write and writes nothing.
    """
    now = time.time() if now is None else now
    report = BackfillReport(store="telemetry.db/tool_calls")
    if not _has_columns(conn, "tool_calls", ["tool_ok", "turn_outcome"]):
        raise RuntimeError("telemetry.db has no label columns yet: open it with this build first")
    turns = _turn_states(conn)
    rows = conn.execute(
        "SELECT rowid, success, tool_ok, turn_id, timestamp, turn_outcome FROM tool_calls"
        " WHERE tool_ok IS NULL OR turn_outcome IS NULL").fetchall()
    report.rows = len(rows)
    updates: list[tuple[Any, Any, int]] = []
    for rowid, success, tool_ok, turn_id, ts, current in rows:
        new_ok = tool_ok if tool_ok is not None else (1 if success else 0)
        if tool_ok is None:
            report.tool_ok_set += 1
        outcome = current if current is not None else _outcome_for(turn_id, ts, turns, now)
        if current is None:
            if outcome is None:
                report.still_running += 1
            else:
                report.turn_outcome[outcome] = report.turn_outcome.get(outcome, 0) + 1
        updates.append((new_ok, outcome, rowid))
    if apply and updates:
        conn.executemany(
            "UPDATE tool_calls SET tool_ok = COALESCE(tool_ok, ?),"
            " turn_outcome = COALESCE(turn_outcome, ?) WHERE rowid = ?", updates)
        conn.commit()
    return report


#: Pair sources whose ``chosen`` call is banked only AFTER it executed without
#: error (agent_loop ``_capture_success_pairs``; the miner pairs a failure with
#: a ``success = 1`` call). ``levenshtein_repair`` / ``schema_repair`` are
#: banked when the adapter repairs a call, before it runs, so their
#: ``tool_ok`` is not known and stays NULL.
EXECUTED_PAIR_SOURCES = frozenset({
    "retry_success", "self_correction", "malformed_recovery", "cloud_golden",
})


def pair_tool_ok(pair_source: str, meta: dict[str, Any] | None) -> int | None:
    """``tool_ok`` for a pair: 1 when its chosen call is known to have run, else None."""
    source = pair_source
    if pair_source == "gym_harvest" and isinstance(meta, dict):
        source = str(meta.get("origin_source") or "")
    return 1 if source in EXECUTED_PAIR_SOURCES else None


def backfill_training(training: sqlite3.Connection, telemetry: sqlite3.Connection | None, *,
                      apply: bool, now: float | None = None) -> BackfillReport:
    """Label ``training_pairs`` rows that have no labels yet, from telemetry.db's turns.

    ``tool_ok``: 1 where the pair's source banks only a call that ran
    (``pair_tool_ok``), else left NULL. ``turn_outcome``: from the pair's
    ``turn_id``; no ``turn_id`` (mined pairs, rows from before telemetry
    v2), no telemetry, or no turn row is ``unknown``. Never rewrites a set
    label.
    """
    import json

    now = time.time() if now is None else now
    report = BackfillReport(store="training.db/training_pairs")
    if not _has_columns(training, "training_pairs", ["tool_ok", "turn_outcome"]):
        raise RuntimeError("training.db has no label columns yet: open it with this build first")
    turns = _turn_states(telemetry) if telemetry is not None else {}
    rows = training.execute(
        "SELECT rowid, pair_source, meta, tool_ok, turn_id, timestamp, turn_outcome"
        " FROM training_pairs WHERE tool_ok IS NULL OR turn_outcome IS NULL").fetchall()
    report.rows = len(rows)
    updates: list[tuple[Any, Any, int]] = []
    for rowid, source, meta_json, tool_ok, turn_id, ts, current in rows:
        new_ok = tool_ok
        if tool_ok is None:
            try:
                meta = json.loads(meta_json) if meta_json else None
            except ValueError:
                meta = None
            new_ok = pair_tool_ok(source, meta)
            if new_ok is not None:
                report.tool_ok_set += 1
        outcome = current if current is not None else _outcome_for(turn_id, ts, turns, now)
        if current is None:
            if outcome is None:
                report.still_running += 1
            else:
                report.turn_outcome[outcome] = report.turn_outcome.get(outcome, 0) + 1
        if (new_ok, outcome) != (tool_ok, current):
            updates.append((new_ok, outcome, rowid))
    if apply and updates:
        training.executemany(
            "UPDATE training_pairs SET tool_ok = COALESCE(tool_ok, ?),"
            " turn_outcome = COALESCE(turn_outcome, ?) WHERE rowid = ?", updates)
        training.commit()
    return report


# --------------------------------------------------------------------------- #
# Judgements
# --------------------------------------------------------------------------- #


def record_judgement(conn: sqlite3.Connection, table: str, row_id: str, *, ok: bool,
                     judged_by: str, at: float | None = None) -> bool:
    """Store a judge's verdict on one trace row. True if the row exists.

    ``table`` is ``tool_calls`` (telemetry.db, keyed by ``id``) or
    ``training_pairs`` (training.db, keyed by ``id``). A later verdict
    replaces an earlier one: a re-judge with a revised rubric is the point.
    """
    if table not in ("tool_calls", "training_pairs"):
        raise ValueError(f"not a trace table: {table!r}")
    if not judged_by:
        raise ValueError("a judgement must say who judged")
    cur = conn.execute(
        f"UPDATE {table} SET judged_ok = ?, judged_by = ?, judged_at = ? WHERE id = ?",
        (1 if ok else 0, judged_by, time.time() if at is None else at, row_id))
    conn.commit()
    return bool(cur.rowcount)


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def outcome_rates(counts: dict[str | None, int]) -> dict[str, Any]:
    """Turn-outcome shares over the calls whose turn has an end state.

    ``completed_rate`` is completed / (completed + stopped + errored +
    abandoned). ``unknown`` and not-yet-known (None) calls are counted but
    kept out of the denominator, so a corpus that is mostly unlinkable says
    so instead of averaging over nothing.
    """
    known = {k: counts.get(k, 0) for k in (COMPLETED, STOPPED, ERRORED, ABANDONED)}
    denom = sum(known.values())
    return {
        **known,
        UNKNOWN: counts.get(UNKNOWN, 0),
        "pending": counts.get(None, 0),
        "linked": denom,
        "completed_rate": (known[COMPLETED] / denom) if denom else None,
    }
