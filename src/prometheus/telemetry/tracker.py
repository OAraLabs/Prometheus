"""ToolCallTelemetry — per-model, per-tool success/retry/latency tracking.

Storage: SQLite at ~/.prometheus/telemetry.db (or a path you specify).

Sprint 4 additions: two new tables for autonomous-subsystem observability.

- ``silent_failures``  — every exception caught by ``LLMCallEnvelope`` or
  ``record_silent_failure(...)`` calls from autonomous subsystems lands here
  with subsystem name, operation, exception type/message, full traceback.
  Closes the gap that hid PR #1 / ed8f1a6 for weeks.
- ``subsystem_runs``   — liveness/outcome companion. Every Curator pass,
  MemoryExtractor cycle, GEPA cycle, etc. writes a row so "no successful
  run in 7 days" detects hangs even when no exception is thrown.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import traceback as _traceback
from pathlib import Path
from typing import Any
from uuid import uuid4

from prometheus.telemetry.db import connect_telemetry_db

log = logging.getLogger(__name__)


# Cloud provider names that qualify for golden-trace capture.
# Kept in sync with ProviderRegistry.is_cloud() but duplicated here to
# avoid a telemetry → providers dependency. Drift is unlikely — this set
# only changes when a new cloud backend is added to the project.
_CLOUD_PROVIDERS: frozenset[str] = frozenset(
    {"openai", "anthropic", "gemini", "xai", "groq"}
)


# D3 denominator honesty: these error types are POLICY outcomes — the
# SecurityGate or a hook refusing to run a well-formed call. They are not
# model tool-calling failures, so success-rate denominators must exclude
# them (they're still surfaced separately as ``denials`` counts).
POLICY_ERROR_TYPES: frozenset[str] = frozenset(
    {"permission_denied", "hook_blocked"}
)


_SCHEMA_SQL_TABLES = """
CREATE TABLE IF NOT EXISTS tool_calls (
    id                TEXT PRIMARY KEY,
    timestamp         REAL NOT NULL,
    model             TEXT NOT NULL,
    tool_name         TEXT NOT NULL,
    success           INTEGER NOT NULL,   -- 0 or 1
    retries           INTEGER NOT NULL DEFAULT 0,
    latency_ms        REAL NOT NULL DEFAULT 0.0,
    error_type        TEXT,
    error_detail      TEXT,
    -- Golden Trace Capture sprint additions (nullable for backcompat):
    raw_model_output  TEXT,                -- raw text the model produced BEFORE adapter parsing
    parsed_tool_call  TEXT,                -- validated tool call as JSON {"name": ..., "input": {...}}
    is_golden         INTEGER NOT NULL DEFAULT 0, -- 1 = cloud + success + zero retries + captured raw
    repairs           INTEGER NOT NULL DEFAULT 0  -- M2: adapter repairs applied (fuzzy name, coercion, ...)
);

-- Circuit Breaker Self-Diagnosis sprint: per-trip diagnostic rows.
-- Written by _CircuitBreaker.diagnose_and_recover() when the breaker trips.
-- SENTINEL's _check_tool_patterns can later query this for richer
-- alerting (not wired in this sprint).
CREATE TABLE IF NOT EXISTS circuit_breaker_diagnostics (
    id                TEXT PRIMARY KEY,
    timestamp         REAL NOT NULL,
    model_id          TEXT NOT NULL,
    adapter_tier      TEXT NOT NULL,
    tool_name         TEXT NOT NULL,
    failure_category  TEXT NOT NULL,
    config_drift      INTEGER NOT NULL DEFAULT 0,   -- 0 or 1
    raw_sample        TEXT,                          -- first 500 chars of failed output
    recovered         INTEGER NOT NULL DEFAULT 0,    -- 0 or 1
    recovery_method   TEXT,                          -- "tier_bump", "none", etc.
    golden_reference  TEXT                           -- Golden Trace sprint: best-match golden parsed_tool_call
);

-- Sprint 4 (Silent Failure Eradication): every exception caught inside an
-- autonomous subsystem (Curator, SkillCreator, SkillRefiner, MemoryExtractor,
-- GEPA, SENTINEL phases, ...) writes a row here. Closes the gap that hid
-- PR #1 / ed8f1a6 (a ValidationError swallowed inside _call_model for an
-- unknown duration).
CREATE TABLE IF NOT EXISTS silent_failures (
    id              TEXT PRIMARY KEY,
    timestamp       REAL NOT NULL,
    subsystem       TEXT NOT NULL,        -- "curator" | "skill_creator" | ...
    operation       TEXT,                  -- "_call_model" | "run_once" | ...
    exception_type  TEXT NOT NULL,         -- type(exc).__name__
    exception_msg   TEXT,                  -- str(exc) [:2000]
    traceback       TEXT,                  -- traceback.format_exc() [:8000]
    context         TEXT                   -- optional JSON: skill_path, model_id, ...
);

-- Sprint 4: liveness companion. Every Curator pass, MemoryExtractor cycle,
-- GEPA cycle etc. writes one row so /health can detect hangs even when no
-- exception is thrown ("no successful Curator run in 7 days").
CREATE TABLE IF NOT EXISTS subsystem_runs (
    id              TEXT PRIMARY KEY,
    timestamp       REAL NOT NULL,
    subsystem       TEXT NOT NULL,
    operation       TEXT,
    duration_ms     REAL,
    outcome         TEXT NOT NULL,         -- "success" | "partial" | "failed" | "skipped"
    summary_json    TEXT,                  -- arbitrary JSON the subsystem wants to surface
    -- SPRINT-loop-envelope (F1) additions (nullable for backcompat):
    input_tokens    INTEGER,               -- UsageSnapshot.input_tokens for LLM calls
    output_tokens   INTEGER,               -- UsageSnapshot.output_tokens for LLM calls
    round_index     INTEGER,               -- loop turn number (0-based) for agent_loop rows
    session_id      TEXT,                  -- LoopContext.session_id for agent_loop rows
    model           TEXT,                  -- model id the call was made with
    thinking        INTEGER                -- effective flag: 1 on, 0 suppressed, NULL unknown
);

-- SignalBus Persistence sprint: every emission on the in-process SignalBus
-- (skill_created, skill_refined, memory_updated, curator_report, dream_*,
-- idle_*, ...) writes one row so event history survives daemon restarts.
-- The in-memory ``deque(maxlen=500)`` on SignalBus remains the hot cache;
-- this table is the cold tail + the source of truth for /events and the
-- Beacon ``/api/events/recent`` endpoint.
--
-- ``timestamp`` is ISO8601 TEXT here (not REAL like the other tables) per
-- the sprint spec — sortable lexicographically and human-readable in raw
-- queries. ``read_at`` is reserved for a future "has this surfaced to the
-- user" marker; nullable for backcompat with this sprint.
CREATE TABLE IF NOT EXISTS signal_events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT NOT NULL,        -- ISO8601 UTC
    signal_type       TEXT NOT NULL,        -- ActivitySignal.kind: "skill_created", ...
    payload           TEXT NOT NULL,        -- JSON blob of ActivitySignal.payload
    source_subsystem  TEXT NOT NULL,        -- ActivitySignal.source: "SkillCreator", ...
    read_at           TEXT                  -- nullable: when surfaced to user (reserved)
);
"""

# Indexes run AFTER _migrate_schema so they can reference columns that are
# newly added to pre-existing tables. (CREATE INDEX on a missing column
# fails even with IF NOT EXISTS.)
_SCHEMA_SQL_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_tool_calls_model ON tool_calls (model);
CREATE INDEX IF NOT EXISTS idx_tool_calls_tool ON tool_calls (tool_name);
CREATE INDEX IF NOT EXISTS idx_tool_calls_golden ON tool_calls (is_golden);

CREATE INDEX IF NOT EXISTS idx_cb_diag_timestamp ON circuit_breaker_diagnostics (timestamp);
CREATE INDEX IF NOT EXISTS idx_cb_diag_model ON circuit_breaker_diagnostics (model_id);
CREATE INDEX IF NOT EXISTS idx_cb_diag_tool ON circuit_breaker_diagnostics (tool_name);

-- Sprint 4 indexes
CREATE INDEX IF NOT EXISTS idx_silent_failures_ts ON silent_failures (timestamp);
CREATE INDEX IF NOT EXISTS idx_silent_failures_subsystem ON silent_failures (subsystem);
CREATE INDEX IF NOT EXISTS idx_subsystem_runs_ts ON subsystem_runs (timestamp);
CREATE INDEX IF NOT EXISTS idx_subsystem_runs_subsystem ON subsystem_runs (subsystem);

-- SignalBus Persistence sprint: (signal_type, timestamp DESC) is the natural
-- read pattern for /events filtered queries and Beacon's recent-events
-- hydration; the composite index serves both.
CREATE INDEX IF NOT EXISTS idx_signal_events_type_time
    ON signal_events (signal_type, timestamp DESC);
"""


# Golden Trace Capture sprint: migration map for existing DBs.
# Key = table name, value = list of (column_name, column_type_sql) tuples
# that must exist. On init, any missing column gets ALTER TABLE ADD COLUMN'd.
_EXPECTED_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "tool_calls": [
        ("raw_model_output", "TEXT"),
        ("parsed_tool_call", "TEXT"),
        ("is_golden", "INTEGER NOT NULL DEFAULT 0"),
        ("repairs", "INTEGER NOT NULL DEFAULT 0"),
    ],
    "circuit_breaker_diagnostics": [
        ("golden_reference", "TEXT"),
    ],
    # SPRINT-loop-envelope (F1): per-call usage columns on the envelope's
    # table, so the agent loop's rounds and the autonomous subsystems share
    # one query surface. NULL on rows from callers that don't carry usage
    # (curator, skill_creator, ... — their write path is unchanged).
    # ``thinking`` is the EFFECTIVE thinking flag for the call: 1 = thinking
    # on, 0 = suppressed, NULL = provider doesn't expose the knob.
    "subsystem_runs": [
        ("input_tokens", "INTEGER"),
        ("output_tokens", "INTEGER"),
        ("round_index", "INTEGER"),
        ("session_id", "TEXT"),
        ("model", "TEXT"),
        ("thinking", "INTEGER"),
    ],
}


class ToolCallTelemetry:
    """Record and report tool-call outcomes.

    Usage:
        tel = ToolCallTelemetry("~/.prometheus/telemetry.db")
        tel.record(
            model="qwen2.5-coder-32b",
            tool_name="bash",
            success=True,
            retries=0,
            latency_ms=142.3,
        )
        report = tel.report()
        # {"models": {"qwen2.5-coder-32b": {"bash": {"calls": 1, "success_rate": 1.0, ...}}}}
    """

    def __init__(self, db_path: str | Path = "~/.prometheus/telemetry.db") -> None:
        self._db_path = Path(db_path).expanduser().resolve()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # Shared WAL + busy_timeout setup (see telemetry.db) so the daemon
        # writer, the coding-subprocess writer, and dashboard/live-stream
        # readers share one concurrency-safe substrate. Write/commit logic
        # below is unchanged.
        self._conn = connect_telemetry_db(self._db_path)
        # Three-phase init:
        # 1. Create tables IF NOT EXISTS (fresh DBs get the full schema here)
        # 2. Migrate any existing DB that predates later column additions
        # 3. Create indexes (runs AFTER migration so indexes on newly-added
        #    columns don't fail against freshly-migrated pre-existing tables)
        self._conn.executescript(_SCHEMA_SQL_TABLES)
        self._migrate_schema()
        self._conn.executescript(_SCHEMA_SQL_INDEXES)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Schema migration (Golden Trace Capture sprint)
    # ------------------------------------------------------------------

    def _migrate_schema(self) -> None:
        """Add any expected columns missing from existing tables.

        Called on every __init__. Uses PRAGMA table_info to detect existing
        columns so ALTER TABLE is only issued when necessary (idempotent).
        Never raises — if the DB is read-only or a column add fails, we
        log and continue; downstream code handles NULL new-column values.
        """
        for table, expected_cols in _EXPECTED_COLUMNS.items():
            try:
                rows = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
            except sqlite3.DatabaseError:
                continue
            existing = {row[1] for row in rows}  # row[1] = column name
            for col_name, col_sql in expected_cols:
                if col_name in existing:
                    continue
                try:
                    self._conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN {col_name} {col_sql}"
                    )
                except sqlite3.DatabaseError:
                    # Column may have been added by a concurrent connection
                    # or the DB might be read-only. Best-effort only.
                    pass

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(
        self,
        model: str,
        tool_name: str,
        success: bool,
        retries: int = 0,
        latency_ms: float = 0.0,
        error_type: str | None = None,
        error_detail: str | None = None,
        *,
        raw_model_output: str | None = None,
        parsed_tool_call: str | None = None,
        provider: str = "",
        repairs: int = 0,
    ) -> None:
        """Record a single tool-call outcome.

        Golden Trace Capture sprint additions (all keyword-only for clarity):
          - ``raw_model_output``: the text the model produced BEFORE adapter
            parsing (enforcer/formatter). Captured only when the caller can
            supply it — typically the main success path in the agent loop.
          - ``parsed_tool_call``: JSON string of the validated tool call
            (shape ``{"name": ..., "input": {...}}``).
          - ``provider``: provider name string (``"anthropic"``, ``"openai"``,
            ``"llama_cpp"``, ...). Used to compute ``is_golden``.

        ``is_golden`` is computed internally, not passed in: True iff the
        provider is cloud AND ``success`` AND ``retries == 0`` AND
        ``raw_model_output`` was supplied. Only cloud wins count as "teacher
        model" examples worth keeping.
        """
        is_golden = (
            provider in _CLOUD_PROVIDERS
            and success
            and retries == 0
            and raw_model_output is not None
        )
        self._conn.execute(
            """
            INSERT INTO tool_calls
              (id, timestamp, model, tool_name, success, retries, latency_ms,
               error_type, error_detail,
               raw_model_output, parsed_tool_call, is_golden, repairs)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uuid4().hex,
                time.time(),
                model,
                tool_name,
                1 if success else 0,
                retries,
                latency_ms,
                error_type,
                error_detail,
                raw_model_output,
                parsed_tool_call,
                1 if is_golden else 0,
                int(repairs),
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Circuit Breaker diagnostic writes (Circuit Breaker Self-Diagnosis sprint)
    # ------------------------------------------------------------------

    def record_diagnosis(
        self,
        model_id: str,
        adapter_tier: str,
        tool_name: str,
        failure_category: str,
        config_drift: bool,
        raw_sample: str | None,
        recovered: bool,
        recovery_method: str,
        *,
        golden_reference: str | None = None,
    ) -> None:
        """Record a single circuit-breaker diagnostic event.

        Called by _CircuitBreaker.diagnose_and_recover() when the breaker
        trips. Gives SENTINEL's _check_tool_patterns real diagnostic data
        beyond "a failure happened".

        Golden Trace Capture sprint: ``golden_reference`` carries the
        ``parsed_tool_call`` JSON from the best-match golden trace for this
        tool (or None if no golden trace exists). Stored for later analysis
        of "what would a cloud teacher have done differently".
        """
        sample = (raw_sample or "")[:500]
        self._conn.execute(
            """
            INSERT INTO circuit_breaker_diagnostics
              (id, timestamp, model_id, adapter_tier, tool_name,
               failure_category, config_drift, raw_sample,
               recovered, recovery_method, golden_reference)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                uuid4().hex,
                time.time(),
                model_id,
                adapter_tier,
                tool_name,
                failure_category,
                1 if config_drift else 0,
                sample,
                1 if recovered else 0,
                recovery_method,
                golden_reference,
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Sprint 4 (Silent Failure Eradication) — autonomous-subsystem writes
    # ------------------------------------------------------------------

    def record_silent_failure(
        self,
        subsystem: str,
        operation: str,
        exc: BaseException,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Record an exception caught inside an autonomous subsystem.

        Best-effort: this helper never raises. If the DB is read-only or
        a parallel write fails, we log at WARN and move on — the goal is
        to make silent failures observable, not to crash the daemon
        because telemetry is unhappy.

        Args:
            subsystem: short tag like ``"curator"``, ``"skill_creator"``,
                ``"memory_extractor"``, ``"gepa"``.
            operation: optional sub-operation tag such as ``"_call_model"``
                or ``"run_once"`` — helps disambiguate which path failed.
            exc: the exception instance.
            context: optional JSON-serialisable dict (skill path, model
                id, batch size, etc.). Stored as a JSON string.
        """
        try:
            ctx_json = json.dumps(context, default=str) if context else None
        except Exception:
            ctx_json = None
        try:
            tb_text = "".join(_traceback.format_exception(type(exc), exc, exc.__traceback__))
        except Exception:
            tb_text = ""
        try:
            self._conn.execute(
                """
                INSERT INTO silent_failures
                  (id, timestamp, subsystem, operation,
                   exception_type, exception_msg, traceback, context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid4().hex,
                    time.time(),
                    subsystem,
                    operation,
                    type(exc).__name__,
                    str(exc)[:2000],
                    tb_text[:8000],
                    ctx_json,
                ),
            )
            self._conn.commit()
        except Exception:
            # Never let telemetry plumbing crash a subsystem path. The whole
            # point of this table is observability — a write failure here
            # would be ironic but not load-bearing.
            log.warning(
                "ToolCallTelemetry.record_silent_failure: write failed for "
                "subsystem=%s operation=%s",
                subsystem, operation, exc_info=True,
            )

    def record_run(
        self,
        subsystem: str,
        operation: str,
        outcome: str,
        duration_ms: float = 0.0,
        summary: dict[str, Any] | None = None,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        round_index: int | None = None,
        session_id: str | None = None,
        model: str | None = None,
        thinking: bool | None = None,
    ) -> None:
        """Record one autonomous-subsystem cycle / pass / invocation.

        The companion to :meth:`record_silent_failure`. Every Curator pass,
        MemoryExtractor cycle, etc. writes one row regardless of outcome
        so ``/health`` can detect a hung subsystem even when no exception
        is thrown.

        ``outcome`` must be one of ``"success"`` | ``"partial"`` |
        ``"failed"`` | ``"skipped"`` — anything else is coerced to
        ``"failed"`` defensively.

        The keyword-only usage fields (SPRINT-loop-envelope, F1) carry
        per-LLM-call accounting for rows written by
        :meth:`LLMCallEnvelope.stream` — the agent loop's rounds. They
        default to ``None`` so every pre-existing caller is unchanged;
        ``thinking`` is stored as 1/0/NULL (NULL = the provider doesn't
        expose a thinking knob, e.g. stubs and cloud providers).
        """
        if outcome not in {"success", "partial", "failed", "skipped"}:
            outcome = "failed"
        try:
            summary_json = json.dumps(summary, default=str) if summary else None
        except Exception:
            summary_json = None
        try:
            self._conn.execute(
                """
                INSERT INTO subsystem_runs
                  (id, timestamp, subsystem, operation,
                   duration_ms, outcome, summary_json,
                   input_tokens, output_tokens, round_index,
                   session_id, model, thinking)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid4().hex,
                    time.time(),
                    subsystem,
                    operation,
                    float(duration_ms),
                    outcome,
                    summary_json,
                    input_tokens,
                    output_tokens,
                    round_index,
                    session_id,
                    model,
                    None if thinking is None else int(thinking),
                ),
            )
            self._conn.commit()
        except Exception:
            log.warning(
                "ToolCallTelemetry.record_run: write failed for "
                "subsystem=%s operation=%s",
                subsystem, operation, exc_info=True,
            )

        # Feed the cost tracker from this single LLM-usage seam (audit: the
        # tracker was reported on but never fed → always $0). Only LLM-call rows
        # carry token counts; the handle is None unless a cloud provider is
        # active, so this is a no-op on the local box. Never let cost accounting
        # break telemetry.
        if input_tokens is not None and model:
            try:
                from prometheus.telemetry.cost import get_cost_tracker_handle

                handle = get_cost_tracker_handle()
                if handle is not None:
                    handle.record(model, input_tokens or 0, output_tokens or 0)
            except Exception:
                log.debug("cost tracker feed skipped", exc_info=True)

    # ------------------------------------------------------------------
    # SignalBus Persistence sprint — signal_events writer + reader
    # ------------------------------------------------------------------

    def record_signal_event(
        self,
        signal_type: str,
        payload: dict[str, Any] | None,
        source_subsystem: str,
        *,
        timestamp_iso: str | None = None,
    ) -> int | None:
        """Persist one SignalBus emission to the ``signal_events`` table.

        Called synchronously by ``SignalBus.emit`` BEFORE broadcasting to
        in-process subscribers. The contract is: if this method returns
        a row id, the event is durable; if it returns None, persistence
        failed (but a silent_failure row was already written, and the
        broadcast will continue regardless — per the sprint spec, "live
        event stream must never be blocked by persistence").

        Args:
            signal_type: ``ActivitySignal.kind`` (e.g. ``"skill_created"``).
            payload: ``ActivitySignal.payload`` dict — serialised to JSON.
            source_subsystem: ``ActivitySignal.source`` (e.g. ``"SkillCreator"``).
            timestamp_iso: ISO8601 UTC string. Defaults to ``datetime.utcnow()``
                rendered. Callers that already have a unix-timestamp from
                ``ActivitySignal.timestamp`` can pre-convert and pass it
                so the persisted row and the broadcast share the same
                wall-clock moment to the microsecond.

        Returns:
            The new row's ``id`` on success, ``None`` on failure.
        """
        from datetime import datetime, timezone

        ts = timestamp_iso or datetime.now(timezone.utc).isoformat()
        try:
            payload_json = json.dumps(payload or {}, default=str)
        except Exception:
            payload_json = "{}"

        try:
            cur = self._conn.execute(
                """
                INSERT INTO signal_events
                  (timestamp, signal_type, payload, source_subsystem)
                VALUES (?, ?, ?, ?)
                """,
                (ts, signal_type, payload_json, source_subsystem),
            )
            self._conn.commit()
            return int(cur.lastrowid) if cur.lastrowid is not None else None
        except Exception as exc:
            log.warning(
                "ToolCallTelemetry.record_signal_event: write failed for "
                "signal_type=%s source=%s",
                signal_type, source_subsystem, exc_info=True,
            )
            # Surface to /health via silent_failures so a flaky DB doesn't
            # hide. Best-effort — record_silent_failure is itself best-effort.
            try:
                self.record_silent_failure(
                    subsystem="signal_bus",
                    operation="persist_event",
                    exc=exc,
                    context={
                        "signal_type": signal_type,
                        "source": source_subsystem,
                    },
                )
            except Exception:
                pass
            return None

    def signal_events_since(
        self,
        since: str | None = None,
        *,
        signal_type: str | None = None,
        signal_types: list[str] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return ``signal_events`` rows, newest first.

        Args:
            since: optional ISO8601 lower bound. ``None`` = no lower bound.
            signal_type: single-type filter (convenience for the common case).
            signal_types: multi-type filter; takes precedence over
                ``signal_type`` if both are supplied.
            limit: max rows to return.

        The composite index ``idx_signal_events_type_time`` makes the
        single-type + ``ORDER BY timestamp DESC`` path index-only.
        """
        query_parts = [
            "SELECT id, timestamp, signal_type, payload, source_subsystem, "
            "read_at FROM signal_events"
        ]
        where: list[str] = []
        params: list[Any] = []
        if since is not None:
            where.append("timestamp >= ?")
            params.append(since)
        if signal_types:
            placeholders = ",".join(["?"] * len(signal_types))
            where.append(f"signal_type IN ({placeholders})")
            params.extend(signal_types)
        elif signal_type is not None:
            where.append("signal_type = ?")
            params.append(signal_type)
        if where:
            query_parts.append("WHERE " + " AND ".join(where))
        query_parts.append("ORDER BY timestamp DESC LIMIT ?")
        params.append(max(1, int(limit)))
        query = " ".join(query_parts)

        try:
            rows = self._conn.execute(query, tuple(params)).fetchall()
        except sqlite3.DatabaseError:
            return []

        out: list[dict[str, Any]] = []
        for row in rows:
            try:
                payload_obj = json.loads(row[3]) if row[3] else {}
            except Exception:
                payload_obj = {}
            out.append({
                "id": int(row[0]),
                "timestamp": row[1],
                "signal_type": row[2],
                "payload": payload_obj,
                "source_subsystem": row[4],
                "read_at": row[5],
            })
        return out

    def signal_event_by_id(self, event_id: int) -> dict[str, Any] | None:
        """Return a single ``signal_events`` row by id, or ``None``."""
        try:
            row = self._conn.execute(
                "SELECT id, timestamp, signal_type, payload, "
                "source_subsystem, read_at FROM signal_events WHERE id = ?",
                (int(event_id),),
            ).fetchone()
        except sqlite3.DatabaseError:
            return None
        if row is None:
            return None
        try:
            payload_obj = json.loads(row[3]) if row[3] else {}
        except Exception:
            payload_obj = {}
        return {
            "id": int(row[0]),
            "timestamp": row[1],
            "signal_type": row[2],
            "payload": payload_obj,
            "source_subsystem": row[4],
            "read_at": row[5],
        }

    # ------------------------------------------------------------------
    # Sprint 4 — readers for /health and audits
    # ------------------------------------------------------------------

    def silent_failures_since(
        self,
        since: float,
        *,
        subsystem: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Return silent failures with ``timestamp >= since``, newest first."""
        query = (
            "SELECT id, timestamp, subsystem, operation, exception_type, "
            "exception_msg, traceback, context FROM silent_failures "
            "WHERE timestamp >= ?"
        )
        params: list[Any] = [since]
        if subsystem is not None:
            query += " AND subsystem = ?"
            params.append(subsystem)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(max(1, int(limit)))
        try:
            rows = self._conn.execute(query, tuple(params)).fetchall()
        except sqlite3.DatabaseError:
            return []
        return [
            {
                "id": row[0],
                "timestamp": row[1],
                "subsystem": row[2],
                "operation": row[3],
                "exception_type": row[4],
                "exception_msg": row[5],
                "traceback": row[6],
                "context": row[7],
            }
            for row in rows
        ]

    def runs_since(
        self,
        since: float,
        *,
        subsystem: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Return subsystem-run rows with ``timestamp >= since``, newest first."""
        query = (
            "SELECT id, timestamp, subsystem, operation, duration_ms, "
            "outcome, summary_json FROM subsystem_runs WHERE timestamp >= ?"
        )
        params: list[Any] = [since]
        if subsystem is not None:
            query += " AND subsystem = ?"
            params.append(subsystem)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(max(1, int(limit)))
        try:
            rows = self._conn.execute(query, tuple(params)).fetchall()
        except sqlite3.DatabaseError:
            return []
        return [
            {
                "id": row[0],
                "timestamp": row[1],
                "subsystem": row[2],
                "operation": row[3],
                "duration_ms": row[4],
                "outcome": row[5],
                "summary_json": row[6],
            }
            for row in rows
        ]

    def health_summary(self, since: float) -> dict[str, Any]:
        """Compact aggregate used by the ``/health`` command.

        Returns::

            {
                "since": <unix-ts>,
                "tool_calls": {"total": N, "failures": N, "denials": N,
                               "success_rate": 0-1},
                    # denials = policy refusals (SecurityGate / hooks); they
                    # count in `total` but not in `failures` or the rate.
                "subsystems": {
                    "<name>": {
                        "runs": N, "success": N, "partial": N, "failed": N,
                        "skipped": N, "silent_failures": N,
                        "last_run_at": ts | None,
                        "last_outcome": "success" | ...,
                    },
                    ...
                },
                "recent_silent_failures": [ <silent_failure dict>, ... ],
                    # 5 newest, across all subsystems
            }
        """
        out: dict[str, Any] = {
            "since": since,
            "tool_calls": {
                "total": 0, "failures": 0, "denials": 0, "success_rate": 0.0,
            },
            "subsystems": {},
            "recent_silent_failures": [],
        }
        try:
            # M1: exclude the synthetic ``_loop_transition`` rows the agent loop
            # writes per iteration — they echo every real tool failure, so
            # counting them double-counts failures in the success rate.
            # D3: policy denials count toward ``total`` (the call happened) but
            # not toward ``failures`` or the success-rate denominator.
            _ph = ",".join("?" * len(POLICY_ERROR_TYPES))
            t_total, t_succ, t_denied = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(success), 0), "
                f"COALESCE(SUM(error_type IN ({_ph})), 0) "
                "FROM tool_calls WHERE timestamp >= ? "
                "AND tool_name != '_loop_transition'",
                (*POLICY_ERROR_TYPES, since),
            ).fetchone() or (0, 0, 0)
            t_total, t_succ, t_denied = (
                int(t_total or 0), int(t_succ or 0), int(t_denied or 0),
            )
            judged = t_total - t_denied
            out["tool_calls"]["total"] = t_total
            out["tool_calls"]["failures"] = judged - t_succ
            out["tool_calls"]["denials"] = t_denied
            out["tool_calls"]["success_rate"] = (
                float(t_succ) / float(judged) if judged else 0.0
            )
        except sqlite3.DatabaseError:
            pass

        # subsystem_runs aggregated by name + outcome
        try:
            for subsystem, outcome, cnt, last_ts in self._conn.execute(
                "SELECT subsystem, outcome, COUNT(*), MAX(timestamp) "
                "FROM subsystem_runs WHERE timestamp >= ? "
                "GROUP BY subsystem, outcome",
                (since,),
            ).fetchall():
                bucket = out["subsystems"].setdefault(
                    subsystem,
                    {"runs": 0, "success": 0, "partial": 0, "failed": 0,
                     "skipped": 0, "silent_failures": 0,
                     "last_run_at": None, "last_outcome": None},
                )
                bucket["runs"] += int(cnt or 0)
                if outcome in ("success", "partial", "failed", "skipped"):
                    bucket[outcome] += int(cnt or 0)
                if last_ts is not None and (
                    bucket["last_run_at"] is None or last_ts > bucket["last_run_at"]
                ):
                    bucket["last_run_at"] = float(last_ts)
                    bucket["last_outcome"] = outcome
        except sqlite3.DatabaseError:
            pass

        # silent_failures grouped by subsystem
        try:
            for subsystem, cnt in self._conn.execute(
                "SELECT subsystem, COUNT(*) FROM silent_failures "
                "WHERE timestamp >= ? GROUP BY subsystem",
                (since,),
            ).fetchall():
                bucket = out["subsystems"].setdefault(
                    subsystem,
                    {"runs": 0, "success": 0, "partial": 0, "failed": 0,
                     "skipped": 0, "silent_failures": 0,
                     "last_run_at": None, "last_outcome": None},
                )
                bucket["silent_failures"] = int(cnt or 0)
        except sqlite3.DatabaseError:
            pass

        # Most-recent 5 silent failures across all subsystems
        out["recent_silent_failures"] = self.silent_failures_since(since, limit=5)
        return out

    # ------------------------------------------------------------------
    # Golden Trace Capture sprint — query + export
    # ------------------------------------------------------------------

    def get_golden_traces(
        self,
        tool_name: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Return recent golden tool-call traces.

        Each row is a dict with: ``model``, ``tool_name``,
        ``raw_model_output``, ``parsed_tool_call``, ``timestamp``.
        Filtered by ``tool_name`` if provided. Ordered newest-first.
        """
        query = (
            "SELECT model, tool_name, raw_model_output, parsed_tool_call, timestamp"
            " FROM tool_calls WHERE is_golden = 1"
        )
        params: list[Any] = []
        if tool_name is not None:
            query += " AND tool_name = ?"
            params.append(tool_name)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(max(1, int(limit)))

        try:
            rows = self._conn.execute(query, tuple(params)).fetchall()
        except sqlite3.DatabaseError:
            return []

        return [
            {
                "model": row[0],
                "tool_name": row[1],
                "raw_model_output": row[2],
                "parsed_tool_call": row[3],
                "timestamp": row[4],
            }
            for row in rows
        ]

    def export_golden_traces(
        self,
        tool_name: str | None = None,
        limit: int = 100,
        format: str = "jsonl",
        output_dir: str | Path = "~/.prometheus",
    ) -> Path:
        """Export golden traces to a JSONL file suitable for fine-tuning.

        Each line is a chat-completion training example in the
        ``{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}``
        shape — the standard format for OpenAI fine-tuning and Axolotl.

        Args:
            tool_name: if provided, export only traces for this tool
            limit: max traces to export (default 100)
            format: reserved; currently only "jsonl" is supported
            output_dir: directory to write to (default ~/.prometheus)

        Returns:
            Path to the written JSONL file, e.g.
            ``~/.prometheus/golden_traces_1713885600.jsonl``
        """
        if format != "jsonl":
            raise ValueError(f"Unsupported format: {format!r} (only 'jsonl' is supported)")

        traces = self.get_golden_traces(tool_name=tool_name, limit=limit)
        out_dir = Path(output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time())
        path = out_dir / f"golden_traces_{stamp}.jsonl"

        with path.open("w", encoding="utf-8") as fh:
            for trace in traces:
                user_content = (
                    f"Call the `{trace['tool_name']}` tool appropriately.\n"
                    f"Reference parsed call: {trace['parsed_tool_call'] or '{}'}"
                )
                assistant_content = trace["raw_model_output"] or ""
                example = {
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content},
                    ],
                    "_meta": {
                        "model": trace["model"],
                        "tool_name": trace["tool_name"],
                        "timestamp": trace["timestamp"],
                    },
                }
                fh.write(json.dumps(example, ensure_ascii=False) + "\n")

        return path

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def report(self, since: float | None = None) -> dict[str, Any]:
        """Return aggregated success rates per model and per tool.

        Returns a dict structured as::

            {
                "models": {
                    "<model_name>": {
                        "<tool_name>": {
                            "calls": int,        # excludes policy denials
                            "successes": int,
                            "failures": int,
                            "denials": int,
                            "success_rate": float,   # 0.0 – 1.0
                            "avg_retries": float,
                            "avg_latency_ms": float,
                        },
                        ...
                    },
                    ...
                },
                "tools": {
                    "<tool_name>": {
                        "calls": int,        # excludes policy denials
                        "denials": int,      # SecurityGate / hook refusals
                        "success_rate": float,
                        "avg_retries": float,
                        "avg_latency_ms": float,
                        "error_types": {"<type>": int, ...},
                    },
                    ...
                },
                "total_calls": int,          # excludes policy denials
                "total_denials": int,
                "overall_success_rate": float,
            }
        """
        # M1: drop synthetic ``_loop_transition`` rows so the per-tool breakdown
        # doesn't list a fake "tool" and the totals aren't inflated by the loop
        # echo of every real tool call.
        query = (
            "SELECT model, tool_name, success, retries, latency_ms, error_type"
            " FROM tool_calls WHERE tool_name != '_loop_transition'"
        )
        params: tuple = ()
        if since is not None:
            query += " AND timestamp >= ?"
            params = (since,)
        rows = self._conn.execute(query, params).fetchall()

        if not rows:
            return {
                "models": {},
                "tools": {},
                "total_calls": 0,
                "total_denials": 0,
                "overall_success_rate": 0.0,
            }

        # Aggregate
        models: dict[str, dict[str, dict[str, Any]]] = {}
        tools: dict[str, dict[str, Any]] = {}
        total = 0
        total_success = 0

        total_denials = 0

        for model, tool_name, success, retries, latency_ms, error_type in rows:
            # per-model per-tool
            model_data = models.setdefault(model, {})
            mt = model_data.setdefault(
                tool_name,
                {"calls": 0, "successes": 0, "failures": 0, "denials": 0,
                 "total_retries": 0, "total_latency_ms": 0.0},
            )
            # per-tool
            td = tools.setdefault(
                tool_name,
                {"calls": 0, "successes": 0, "denials": 0, "total_retries": 0,
                 "total_latency_ms": 0.0, "error_types": {}},
            )

            # D3: policy denials are surfaced as `denials`, not failures —
            # they never enter calls/successes or the success-rate math.
            if error_type in POLICY_ERROR_TYPES:
                mt["denials"] += 1
                td["denials"] += 1
                total_denials += 1
                continue

            total += 1
            total_success += success

            mt["calls"] += 1
            mt["successes"] += success
            mt["failures"] += 1 - success
            mt["total_retries"] += retries
            mt["total_latency_ms"] += latency_ms

            td["calls"] += 1
            td["successes"] += success
            td["total_retries"] += retries
            td["total_latency_ms"] += latency_ms
            if error_type:
                td["error_types"][error_type] = td["error_types"].get(error_type, 0) + 1

        # Finalise per-model
        for model_data in models.values():
            for mt in model_data.values():
                c = mt["calls"]
                mt["success_rate"] = mt["successes"] / c if c else 0.0
                mt["avg_retries"] = mt["total_retries"] / c if c else 0.0
                mt["avg_latency_ms"] = mt["total_latency_ms"] / c if c else 0.0
                del mt["total_retries"], mt["total_latency_ms"]

        # Finalise per-tool
        for td in tools.values():
            c = td["calls"]
            td["success_rate"] = td["successes"] / c if c else 0.0
            td["avg_retries"] = td["total_retries"] / c if c else 0.0
            td["avg_latency_ms"] = td["total_latency_ms"] / c if c else 0.0
            del td["total_retries"], td["total_latency_ms"], td["successes"]

        return {
            "models": models,
            "tools": tools,
            "total_calls": total,
            "total_denials": total_denials,
            "overall_success_rate": total_success / total if total else 0.0,
        }

    @property
    def db_path(self) -> Path:
        """Filesystem path of the telemetry SQLite db.

        Lets read-only consumers (e.g. the coding live-stream tailer) open their
        OWN connection to the same file via ``connect_telemetry_db`` instead of
        sharing this writer's connection.
        """
        return self._db_path

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Sprint 4 A3 — module-level singleton for /health and other command handlers.
# ---------------------------------------------------------------------------

_telemetry_singleton: "ToolCallTelemetry | None" = None


def set_telemetry_handle(tel: "ToolCallTelemetry | None") -> None:
    """Register the running ``ToolCallTelemetry`` for command-level access.

    Mirrors the ``set_curator`` / ``set_sentinel_components`` pattern.
    Called once from ``scripts/daemon.py`` after the tracker is built so
    ``gateway/commands.py::cmd_health`` can read live telemetry without
    threading the handle through every command-dispatch layer.
    """
    global _telemetry_singleton
    _telemetry_singleton = tel


def get_telemetry_handle() -> "ToolCallTelemetry | None":
    """Return the registered telemetry handle (None if not wired)."""
    return _telemetry_singleton
