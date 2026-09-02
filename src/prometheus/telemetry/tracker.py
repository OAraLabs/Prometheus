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
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from prometheus.telemetry.db import connect_telemetry_db

log = logging.getLogger(__name__)


# Cloud provider names that qualify for golden-trace capture. Mirrors
# ProviderRegistry.is_cloud(), duplicated to avoid a telemetry → providers
# dependency.
#
# The previous comment here said "drift is unlikely — this set only changes
# when a new cloud backend is added". That is exactly what happened and it
# drifted anyway: `qwen` shipped in 2026-08 and was never added, so the
# busiest cloud model on this box was not classified as cloud at all. It
# still produced golden rows ONLY because OpenAICompatProvider set no
# provider_name and the class-name fallback mislabelled every
# OpenAI-compatible provider as "openai". Now that the registry passes the
# real name through, an unclassified provider silently stops being captured
# — so both sets below are asserted complete against
# ProviderRegistry.list_providers() by a drift guard in tests/test_wiring.py.
_CLOUD_PROVIDERS: frozenset[str] = frozenset(
    {
        "openai", "anthropic", "gemini", "xai", "groq",
        # CLOUD EXPANSION (2026-07)
        "deepseek", "kimi", "glm", "mimo",
        # Alibaba (2026-08) — the omission described above.
        "qwen",
    }
)

# Local backends. Explicit rather than "everything not cloud" on purpose:
# an unclassified name must fail the drift guard, not silently land in
# whichever bucket the default happens to be. Mislabelling a LOCAL provider
# as cloud is the worse direction — it would file student output as teacher
# exemplars and quietly poison the training corpus.
_LOCAL_PROVIDERS: frozenset[str] = frozenset({"llama_cpp", "ollama", "stub"})


# D3 denominator honesty: these error types are POLICY outcomes — the
# SecurityGate or a hook refusing to run a well-formed call. They are not
# model tool-calling failures, so success-rate denominators must exclude
# them (they're still surfaced separately as ``denials`` counts).
POLICY_ERROR_TYPES: frozenset[str] = frozenset(
    {"permission_denied", "hook_blocked"}
)

# Same denominator-honesty argument, different cause: the tool RAN to
# completion and reported a non-zero exit status (``pytest`` with failing
# tests, ``grep`` finding nothing, a build that legitimately fails). The call
# was well-formed and executed — the failure is in the world, not in tool
# calling — so it must not count against the model's tool-call success rate.
# It is still surfaced separately, because a rising count is real signal.
#
# Measured impact: on the EMBERFALL baseline this single conflation reported
# bash at 82% success when every one of the "failures" was a correct execution.
EXECUTED_ERROR_TYPES: frozenset[str] = frozenset({"nonzero_exit"})

# Everything that must be kept out of the success-rate denominator.
NON_CALL_FAILURE_TYPES: frozenset[str] = POLICY_ERROR_TYPES | EXECUTED_ERROR_TYPES

# The agent loop writes one synthetic ``tool_calls`` row per turn transition. It is NOT a tool
# call: no inputs, no model decision, nothing a reader of tool history wants to see. It is ~48%
# of the table on a live daemon, so every per-tool reader excludes it — and the rule lived only
# as a repeated string literal plus prose until /api/tools/recent shipped without it and served
# a feed that was half loop echoes. Named so a fourth reader cannot miss it.
SYNTHETIC_TOOL_NAME = "_loop_transition"


# FOUNDATION 1.3: the telemetry schema version. The first real version
# number this DB has ever had — before it, the only mechanism was the
# additive _EXPECTED_COLUMNS map, which silently tolerates a rollback
# reading a newer DB. Version 1 = schema_meta exists + node_id columns.
# Bumping this requires a migration step in __init__ and a changelog row
# in docs/FOUNDATION.md.
TELEMETRY_SCHEMA_VERSION = 1


class TelemetrySchemaError(RuntimeError):
    """The DB was written by a newer schema than this build knows."""


_SCHEMA_SQL_TABLES = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

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
    repairs           INTEGER NOT NULL DEFAULT 0, -- M2: adapter repairs applied (fuzzy name, coercion, ...)
    -- What the server said actually served this call, echoed back in the
    -- completion response. SEPARATE from `model`, which is the name the
    -- caller REQUESTED. They disagree whenever a harness passes a config
    -- string that no longer matches the loaded model — which is why
    -- `gemma4-26b` rows kept being written months after the server moved to
    -- Qwen. NULL when the provider does not echo one.
    served_model      TEXT
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
-- Beacon's Tool Feed backfill reads `ORDER BY timestamp DESC LIMIT ?`. Without
-- this the tail is a full scan + sort of every call ever recorded, which only
-- gets worse as the table grows.
CREATE INDEX IF NOT EXISTS idx_tool_calls_ts ON tool_calls (timestamp DESC);

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
    # The SERVER's own words on a failed HTTP call. str(exc) on an httpx or
    # requests error is the CLIENT's summary — "Client error '400 Bad Request'
    # for url ..." — which names the status and nothing about the cause.
    "silent_failures": [
        ("response_body", "TEXT"),
    ],
    "tool_calls": [
        ("raw_model_output", "TEXT"),
        ("parsed_tool_call", "TEXT"),
        ("is_golden", "INTEGER NOT NULL DEFAULT 0"),
        ("repairs", "INTEGER NOT NULL DEFAULT 0"),
        ("served_model", "TEXT"),
        # Fine-tuning capture (2026-08). Without these a golden row records
        # WHAT was called but nothing about the situation that prompted it,
        # so no trainable example can be reconstructed from it afterwards.
        # NULL on every row written before this shipped — those rows are not
        # recoverable and are skipped by the export.
        ("session_id", "TEXT"),      # joins to lcm_messages for the context
        ("tool_schema", "TEXT"),     # JSON: the schema the model actually saw
        # FOUNDATION 1.3: which machine produced this trace — the node's
        # public key (config/node_identity.py). A latency number without
        # its hardware is not comparable across machines. NULL on rows
        # from before node identity existed (honest: there was no node to
        # name) and on writers running before first identity generation.
        # NOTE: "node_id" also names a summary-DAG concept in
        # memory/lcm_summary_store.py — different database; the telemetry
        # name is the fleet-facing one from the spec.
        ("node_id", "TEXT"),
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
        # Prompt-cache accounting (per round). The agent loop re-sends a
        # near-identical prefix every round, so the cached fraction is the
        # single biggest lever on cost — but it was previously invisible.
        # NULL means "the provider reported nothing about caching", which is
        # deliberately distinct from 0 ("cache was cold this round").
        ("cached_input_tokens", "INTEGER"),
        ("cache_write_tokens", "INTEGER"),
        # FOUNDATION 1.3 — same column, same meaning as on tool_calls.
        ("node_id", "TEXT"),
    ],
}


# Resolves a golden trace's INPUT half — the conversation that preceded the
# call. Takes the trace row, returns chat messages ([{role, content}, ...])
# or [] when unrecoverable. Injected rather than imported so telemetry keeps
# no dependency on the conversation store; the daemon supplies the LCM-backed
# implementation (see sentinel/golden_trace_exporter.lcm_context_resolver).
ContextResolver = "Callable[[dict[str, Any]], list[dict[str, str]]]"

# Column order shared by both golden-trace queries.
_GOLDEN_COLUMNS = (
    "rowid", "model", "tool_name", "raw_model_output", "parsed_tool_call",
    "timestamp", "session_id", "tool_schema",
)


def _golden_row_to_dict(row: tuple) -> dict[str, Any]:
    """Map a golden-trace row to the dict shape the export consumes."""
    return dict(zip(_GOLDEN_COLUMNS, row))


@dataclass(frozen=True)
class GoldenExport:
    """One incremental golden-trace export.

    ``last_rowid`` is the watermark the caller persists to resume from; it is
    the last row of the batch that was READ, so a batch capped by ``limit``
    resumes mid-backlog rather than skipping what it did not reach.

    ``path`` is None when every trace in the batch was untrainable. The
    watermark must still advance in that case — those rows can never become
    exportable, so re-reading them each cycle would stall the cursor on a
    permanent backlog. ``count`` is examples WRITTEN, not traces read.
    """

    path: Path | None
    count: int
    last_rowid: int


def _response_body(exc: BaseException) -> str | None:
    """The server's own response body, when the exception carries one.

    WHY THIS EXISTS. On 2026-08-17 a Beacon turn died on a 400 from the
    inference server and ``silent_failures`` recorded it faithfully — twice —
    yet the row could not diagnose it: ``exception_msg`` held only httpx's
    "Client error '400 Bad Request' for url ...". The server's body said
    ``Failed to tokenize prompt``, which is a MEDIA-MARKER rejection and not a
    context overflow, and that word went only to journald. From the DB alone
    the two are indistinguishable, and overflow is the wrong guess to leave
    lying around — it sends you to the context budget instead of the tool
    result that poisoned the prompt.

    Duck-typed on ``exc.response`` so httpx and requests both work without
    importing either. Never raises: ``.text`` on an httpx response whose
    stream was never read raises ``ResponseNotRead``, and a telemetry helper
    must not turn a recorded failure into an unrecorded one.
    """
    resp = getattr(exc, "response", None)
    if resp is None:
        return None
    for attr in ("text", "content"):
        try:
            val = getattr(resp, attr, None)
        except Exception:
            continue
        if not val:
            continue
        if isinstance(val, (bytes, bytearray)):
            try:
                val = bytes(val).decode("utf-8", "replace")
            except Exception:
                continue
        return str(val)[:4000]
    return None


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

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            # Resolve through get_config_dir() like every other store. The
            # old default was a hardcoded "~/.prometheus/telemetry.db",
            # which ignored PROMETHEUS_CONFIG_DIR while `prometheus
            # reset-data` honoured it — the writer and the eraser
            # disagreeing about where the DB lives.
            from prometheus.config.paths import get_config_dir
            db_path = get_config_dir() / "telemetry.db"
        self._db_path = Path(db_path).expanduser().resolve()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # Shared WAL + busy_timeout setup (see telemetry.db) so the daemon
        # writer, the coding-subprocess writer, and dashboard/live-stream
        # readers share one concurrency-safe substrate. Write/commit logic
        # below is unchanged.
        self._conn = connect_telemetry_db(self._db_path)
        # FOUNDATION 1.3: refuse a DB written by a NEWER schema before
        # touching it. Unlike _migrate_schema below (best-effort, never
        # raises), this raise is deliberate: the newer-DB-older-binary case
        # is a rollback, and additively "migrating" it would misread rows
        # whose meaning this build predates.
        stored = self._read_stored_schema_version()
        if stored is not None and stored > TELEMETRY_SCHEMA_VERSION:
            self._conn.close()
            raise TelemetrySchemaError(
                f"{self._db_path} has telemetry schema_version {stored}; "
                f"this build reads schema_version {TELEMETRY_SCHEMA_VERSION}. "
                "Newer DB, older binary — refusing rather than misreading. "
                "Upgrade Prometheus, or move the DB aside."
            )
        # Three-phase init:
        # 1. Create tables IF NOT EXISTS (fresh DBs get the full schema here)
        # 2. Migrate any existing DB that predates later column additions
        # 3. Create indexes (runs AFTER migration so indexes on newly-added
        #    columns don't fail against freshly-migrated pre-existing tables)
        self._conn.executescript(_SCHEMA_SQL_TABLES)
        self._migrate_schema()
        self._conn.executescript(_SCHEMA_SQL_INDEXES)
        # A DB at or below the current version is stamped current — the
        # additive migration above IS the upgrade path. A legacy DB (no
        # schema_meta) adopts silently: unlike the vault, this store is
        # machine-owned with a single writer, so adoption is not a policy
        # decision, it is bookkeeping.
        self._conn.execute(
            "INSERT OR IGNORE INTO schema_meta (key, value) VALUES "
            "('created_at', datetime('now'))"
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO schema_meta (key, value) VALUES "
            "('created_by', ?)", (f"prometheus telemetry v{TELEMETRY_SCHEMA_VERSION}",)
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO schema_meta (key, value) VALUES "
            "('schema_version', ?)", (str(TELEMETRY_SCHEMA_VERSION),)
        )
        self._conn.commit()
        # FOUNDATION 1.3: every trace row names the machine that produced
        # it. Resolved lazily in _current_node_id() — a read of node.pub,
        # never a mint: telemetry asking for identity must not create one
        # (the entry points do that, at first run). NULL honestly until
        # an identity exists.
        self._node_id: str | None = None

    def _current_node_id(self) -> str | None:
        """The node ID to stamp on rows. Cached once found.

        Lazy rather than snapshotted at __init__ because construction order
        is not guaranteed: the CLI builds its telemetry before the entry
        point mints first-run identity, and a snapshot would stamp NULL for
        that entire first session.
        """
        if self._node_id is None:
            from prometheus.config.node_identity import get_node_pubkey
            self._node_id = get_node_pubkey()
        return self._node_id

    def _read_stored_schema_version(self) -> int | None:
        """The schema_meta version, or None for a legacy/fresh DB."""
        try:
            row = self._conn.execute(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'"
            ).fetchone()
        except sqlite3.DatabaseError:
            return None
        if row is None:
            return None
        try:
            return int(row[0])
        except (TypeError, ValueError):
            return None

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
        served_model: str | None = None,
        session_id: str | None = None,
        tool_schema: str | None = None,
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

        ``served_model`` is what the server echoed back as having served this
        call, captured per-call from the completion response. ``model`` stays
        the name the caller REQUESTED — the two are recorded side by side and
        neither overwrites the other, because their disagreement is itself the
        finding: it is exactly how six out-of-daemon harnesses kept writing
        ``gemma4-26b`` for months after the server moved to Qwen. Cost is
        deliberately unaffected — ``CostTracker`` keys off the requested
        string and is not touched by this field.

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
               raw_model_output, parsed_tool_call, is_golden, repairs,
               served_model, session_id, tool_schema, node_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                served_model,
                session_id,
                tool_schema,
                self._current_node_id(),
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

        When ``exc`` carries an HTTP response, its body is stored in
        ``response_body`` — see :func:`_response_body` for why a row without it
        could not tell a media-marker rejection from a context overflow.
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
                   exception_type, exception_msg, traceback, context,
                   response_body)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    _response_body(exc),
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
        cached_input_tokens: int | None = None,
        cache_write_tokens: int | None = None,
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
                   session_id, model, thinking,
                   cached_input_tokens, cache_write_tokens, node_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    cached_input_tokens,
                    cache_write_tokens,
                    self._current_node_id(),
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

    def usage_rollup(
        self,
        *,
        days: int | None = None,
    ) -> dict[str, Any]:
        """Token usage aggregated per model, plus a daily series — the cost view's data.

        Reads ``subsystem_runs``, NOT ``tool_calls``: the token counts live there (written by
        ``LLMCallEnvelope`` through :meth:`record_run`), and ``tool_calls`` has no token columns
        at all. A cost feature that reached for the obvious table would find nothing and conclude
        the data did not exist.

        Returns raw counts only. Pricing and billing classification are applied by the caller, so
        this method has no opinion about money and stays correct when the pricing table changes.

        ``tracking_since`` is the first row that carried tokens — a dashboard must be able to say
        what window it is summing, because rows older than that exist and have no token data.
        """
        window_start: float | None = None
        if days is not None and days > 0:
            window_start = time.time() - (days * 86400)

        where = "WHERE input_tokens IS NOT NULL"
        params: list[Any] = []
        if window_start is not None:
            where += " AND timestamp >= ?"
            params.append(window_start)

        try:
            per_model = self._conn.execute(
                "SELECT COALESCE(model, ''), COUNT(*), "
                "COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0), "
                "COALESCE(SUM(cached_input_tokens), 0), MIN(timestamp), MAX(timestamp) "
                f"FROM subsystem_runs {where} GROUP BY COALESCE(model, '') "
                "ORDER BY SUM(input_tokens) DESC",
                tuple(params),
            ).fetchall()
            daily = self._conn.execute(
                "SELECT date(timestamp, 'unixepoch'), "
                "COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0), COUNT(*) "
                f"FROM subsystem_runs {where} GROUP BY 1 ORDER BY 1",
                tuple(params),
            ).fetchall()
            first_ever = self._conn.execute(
                "SELECT MIN(timestamp) FROM subsystem_runs WHERE input_tokens IS NOT NULL"
            ).fetchone()
        except sqlite3.DatabaseError:
            return {"tracking_since": None, "models": [], "daily": []}

        return {
            "tracking_since": (first_ever[0] if first_ever else None),
            "window_days": days,
            "models": [
                {
                    "model": r[0],
                    "runs": r[1],
                    "input_tokens": r[2],
                    "output_tokens": r[3],
                    "cached_input_tokens": r[4],
                    "first_seen": r[5],
                    "last_seen": r[6],
                }
                for r in per_model
            ],
            "daily": [
                {"day": r[0], "input_tokens": r[1], "output_tokens": r[2], "runs": r[3]}
                for r in daily
            ],
        }

    def recent_tool_calls(
        self,
        *,
        limit: int = 100,
        tool_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return ``tool_calls`` rows, newest first — Beacon's Tool Feed backfill.

        The live ``tool_call_start``/``tool_call_end`` WS frames are pure
        fan-out: ``WebSocketServer.broadcast`` never persists, and it returns
        early when no client is attached. So a desktop client that was not
        running (or not yet connected) has no way to learn a call happened,
        and its Tool Feed opened empty next to a telemetry card reporting the
        lifetime ``total_calls`` from THIS table. This reader closes that gap.

        Unlike the signal tail, these rows describe COMPLETED calls only —
        there is no start/end pairing to redo, and no pending state to model.

        ``parsed_tool_call`` is the validated ``{"name", "input"}`` JSON the
        adapter produced; the ``input`` half is what the feed shows as the
        call's inputs. It is nullable (older rows, and calls that never got
        far enough to parse), so it decodes defensively to ``None``.
        """
        query = (
            "SELECT id, timestamp, model, tool_name, success, retries, "
            "latency_ms, error_type, error_detail, parsed_tool_call "
            "FROM tool_calls WHERE tool_name != ?"
        )
        params: list[Any] = [SYNTHETIC_TOOL_NAME]
        if tool_name is not None:
            query += " AND tool_name = ?"
            params.append(tool_name)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(max(1, int(limit)))

        try:
            rows = self._conn.execute(query, tuple(params)).fetchall()
        except sqlite3.DatabaseError:
            return []

        out: list[dict[str, Any]] = []
        for row in rows:
            inputs: Any = None
            if row[9]:
                try:
                    parsed = json.loads(row[9])
                    if isinstance(parsed, dict):
                        inputs = parsed.get("input")
                except Exception:
                    inputs = None
            out.append({
                "call_id": row[0],
                # REAL epoch seconds here (this table predates the ISO-text
                # convention signal_events uses) — clients parse both.
                "timestamp": row[1],
                "model": row[2],
                "tool_name": row[3],
                "success": bool(row[4]),
                "retries": int(row[5] or 0),
                "latency_ms": float(row[6] or 0.0),
                "error_type": row[7],
                "error_detail": row[8],
                "inputs": inputs,
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
            # Also excludes ``nonzero_exit`` (the tool ran; the command
            # reported failure) — see NON_CALL_FAILURE_TYPES.
            _ph = ",".join("?" * len(NON_CALL_FAILURE_TYPES))
            t_total, t_succ, t_denied = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(success), 0), "
                f"COALESCE(SUM(error_type IN ({_ph})), 0) "
                "FROM tool_calls WHERE timestamp >= ? "
                "AND tool_name != ?",
                (*NON_CALL_FAILURE_TYPES, since, SYNTHETIC_TOOL_NAME),
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
            "SELECT rowid, model, tool_name, raw_model_output, parsed_tool_call,"
            " timestamp, session_id, tool_schema"
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

        return [_golden_row_to_dict(row) for row in rows]

    def export_golden_traces(
        self,
        tool_name: str | None = None,
        limit: int = 100,
        format: str = "jsonl",
        output_dir: str | Path = "~/.prometheus",
        context_resolver: "ContextResolver | None" = None,
    ) -> Path:
        """Export the most recent golden traces on demand (the CLI path).

        Each line is a tool-calling training example: the preceding
        conversation, then an assistant turn carrying the call as
        ``tool_calls``, plus the ``tools`` schema the model saw — the shape
        OpenAI fine-tuning and Axolotl consume. See :meth:`_build_example`
        for why the old user/assistant split was not trainable.

        Deliberately NOT incremental — see :meth:`export_new_golden_traces`.
        A human running ``oara export-traces`` asking for the last N
        should get them, not an empty file because the daemon already
        exported those rows.

        Args:
            tool_name: if provided, export only traces for this tool
            limit: max traces to export (default 100)
            format: reserved; currently only "jsonl" is supported
            output_dir: directory to write to (default ~/.prometheus)
            context_resolver: recovers each trace's preceding conversation.
                Without it every row is skipped as untrainable, so the CLI
                and daemon both supply one.

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

        written = self._write_golden_jsonl(traces, path, context_resolver)
        skipped = len(traces) - written
        if skipped:
            # Loud, not silent: rows recorded before session_id capture
            # shipped have no recoverable context and can never be exported.
            # A quiet drop would read as "there was nothing to export".
            log.warning(
                "export_golden_traces: %d of %d traces skipped as untrainable "
                "(no recoverable context or no parsed call)", skipped, len(traces),
            )
        return path

    @staticmethod
    def _build_example(
        trace: dict[str, Any],
        context_messages: list[dict[str, str]],
    ) -> dict[str, Any] | None:
        """Shape one golden trace as a tool-calling fine-tuning example.

        THE TARGET IS THE TOOL CALL. The previous shape had this inverted and
        was not trainable in either half:

        * the ``user`` turn was synthesised as ``Call the `X` tool
          appropriately. Reference parsed call: {the real call}`` — the answer
          verbatim in the prompt. That teaches copying from context, and no
          such reference exists at inference time.
        * the ``assistant`` turn was ``raw_model_output``, which for cloud
          providers is only the PROSE preamble: they return the call as a
          structured field, never in the text stream. Measured over the 1375
          golden rows on this box — 836 prose-only, 522 EMPTY, 17 containing
          any tool-call JSON. ~98.7% had a target that was not the call.

        Now the call goes in the assistant turn as an OpenAI-style
        ``tool_calls`` entry (prose kept as ``content``, which is the real
        shape of such a turn), the schema the model saw goes in ``tools``,
        and the input half is the ACTUAL preceding conversation.

        Returns None when the example would not be trainable — no context, or
        no parsed call to learn. Skipping is deliberate: a corpus of
        unlearnable rows is worse than a smaller honest one, and the caller
        counts what it drops.
        """
        raw_call = trace.get("parsed_tool_call")
        if not raw_call or not context_messages:
            return None
        try:
            call = json.loads(raw_call)
            arguments = json.dumps(call.get("input", {}), ensure_ascii=False)
        except (TypeError, ValueError):
            return None

        assistant: dict[str, Any] = {
            "role": "assistant",
            # Empty prose is normal and fine now: the signal is the call.
            "content": trace.get("raw_model_output") or "",
            "tool_calls": [
                {
                    "id": f"call_{trace.get('rowid', 0)}",
                    "type": "function",
                    "function": {
                        "name": call.get("name", trace["tool_name"]),
                        "arguments": arguments,
                    },
                }
            ],
        }

        example: dict[str, Any] = {
            "messages": [*context_messages, assistant],
            "_meta": {
                "model": trace["model"],
                "tool_name": trace["tool_name"],
                "timestamp": trace["timestamp"],
                "session_id": trace.get("session_id"),
            },
        }

        schema = trace.get("tool_schema")
        if schema:
            try:
                example["tools"] = [
                    {"type": "function", "function": json.loads(schema)}
                ]
            except (TypeError, ValueError):
                pass
        return example

    def _write_golden_jsonl(
        self,
        traces: list[dict[str, Any]],
        path: Path,
        context_resolver: "ContextResolver | None" = None,
    ) -> int:
        """Write *traces* as fine-tuning JSONL. Returns the number written.

        Rows whose context cannot be recovered are skipped rather than
        emitted in a degraded shape — see :meth:`_build_example`.
        """
        written = 0
        with path.open("w", encoding="utf-8") as fh:
            for trace in traces:
                context_messages: list[dict[str, str]] = []
                if context_resolver is not None:
                    try:
                        context_messages = context_resolver(trace) or []
                    except Exception:
                        log.debug("context resolution failed", exc_info=True)
                        context_messages = []
                example = self._build_example(trace, context_messages)
                if example is None:
                    continue
                fh.write(json.dumps(example, ensure_ascii=False) + "\n")
                written += 1
        return written

    def _golden_rows_after(
        self,
        since_rowid: int,
        limit: int,
        tool_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Golden traces with ``rowid > since_rowid``, OLDEST first.

        rowid, not timestamp, is the cursor: it is unique and monotonic per
        insert, so a batch boundary cannot split or duplicate rows that share
        a timestamp — and it is immune to a clock that steps backwards.
        Ascending order matters too, so the watermark advances through the
        backlog in order instead of skipping to the newest N.
        """
        query = (
            "SELECT rowid, model, tool_name, raw_model_output, parsed_tool_call,"
            " timestamp, session_id, tool_schema"
            " FROM tool_calls WHERE is_golden = 1 AND rowid > ?"
        )
        params: list[Any] = [int(since_rowid)]
        if tool_name is not None:
            query += " AND tool_name = ?"
            params.append(tool_name)
        query += " ORDER BY rowid ASC LIMIT ?"
        params.append(max(1, int(limit)))

        try:
            rows = self._conn.execute(query, tuple(params)).fetchall()
        except sqlite3.DatabaseError:
            return []

        return [_golden_row_to_dict(row) for row in rows]

    def export_new_golden_traces(
        self,
        *,
        since_rowid: int = 0,
        limit: int = 1000,
        tool_name: str | None = None,
        format: str = "jsonl",
        output_dir: str | Path = "~/.prometheus/trajectories/",
        context_resolver: "ContextResolver | None" = None,
    ) -> GoldenExport | None:
        """Export only golden traces newer than *since_rowid*.

        The INCREMENTAL counterpart to :meth:`export_golden_traces`, which
        dumps the most recent N on demand and is right for a human running
        ``oara export-traces``. A pipeline needs the other contract:
        each cycle writes only what the last one did not, so the corpus
        accumulates instead of restating itself.

        That distinction is not academic. The daemon loop previously called
        the dump-N method on every cycle AND once at startup, so each restart
        re-wrote the same trailing rows to a new timestamped file — 279 files
        holding 18 distinct payloads, ~94% of 36 MB duplicated, and a corpus
        that looked far larger than the training signal in it.

        Returns None when nothing new exists — no empty file is written,
        since an empty export is not an event worth recording.
        """
        if format != "jsonl":
            raise ValueError(f"Unsupported format: {format!r} (only 'jsonl' is supported)")

        traces = self._golden_rows_after(since_rowid, limit, tool_name)
        if not traces:
            return None

        out_dir = Path(output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        last_rowid = int(traces[-1]["rowid"])
        # The rowid is in the NAME because a bare `int(time.time())` stamp has
        # one-second resolution: two exports in the same second resolve to the
        # same path and the second silently OVERWRITES the first, advancing
        # the watermark past traces that no longer exist on disk. Back-to-back
        # cycles are normal now (draining a backlog capped by `limit`), so
        # that collision is reachable, and it loses data in the one direction
        # a training corpus cannot recover from. rowid is strictly increasing
        # across exports, so it cannot collide.
        path = out_dir / f"golden_traces_{int(time.time())}_{last_rowid}.jsonl"
        written = self._write_golden_jsonl(traces, path, context_resolver)

        if written == 0:
            # Every row in the batch was untrainable. Leave no empty file,
            # but DO advance the watermark: these rows will never become
            # exportable (their context is gone), so re-reading them every
            # cycle would stall the cursor forever on a permanent backlog.
            path.unlink(missing_ok=True)
            log.warning(
                "golden export: all %d traces in rowid range (%d, %d] were "
                "untrainable — skipping past them",
                len(traces), since_rowid, last_rowid,
            )
            return GoldenExport(path=None, count=0, last_rowid=last_rowid)

        # The batch's own last rowid, not a separately-queried MAX: a row
        # inserted between the two queries would otherwise be skipped
        # forever. Advancing only past what this batch READ also makes a
        # truncated batch (limit reached) resume correctly next cycle.
        if written < len(traces):
            log.warning(
                "golden export: %d of %d traces skipped as untrainable",
                len(traces) - written, len(traces),
            )
        return GoldenExport(path=path, count=written, last_rowid=last_rowid)

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
            " FROM tool_calls WHERE tool_name != ?"
        )
        params: tuple = (SYNTHETIC_TOOL_NAME,)
        if since is not None:
            query += " AND timestamp >= ?"
            params = (SYNTHETIC_TOOL_NAME, since)  # APPEND — do not drop the exclusion bind
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

            # The tool executed and the command exited non-zero. Reported in
            # its own bucket, kept out of the success-rate math.
            if error_type in EXECUTED_ERROR_TYPES:
                mt["nonzero_exits"] = mt.get("nonzero_exits", 0) + 1
                td["nonzero_exits"] = td.get("nonzero_exits", 0) + 1
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
