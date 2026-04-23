"""ToolCallTelemetry — per-model, per-tool success/retry/latency tracking.

Storage: SQLite at ~/.prometheus/telemetry.db (or a path you specify).
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any
from uuid import uuid4


# Cloud provider names that qualify for golden-trace capture.
# Kept in sync with ProviderRegistry.is_cloud() but duplicated here to
# avoid a telemetry → providers dependency. Drift is unlikely — this set
# only changes when a new cloud backend is added to the project.
_CLOUD_PROVIDERS: frozenset[str] = frozenset(
    {"openai", "anthropic", "gemini", "xai", "groq"}
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
    is_golden         INTEGER NOT NULL DEFAULT 0  -- 1 = cloud + success + zero retries + captured raw
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
"""


# Golden Trace Capture sprint: migration map for existing DBs.
# Key = table name, value = list of (column_name, column_type_sql) tuples
# that must exist. On init, any missing column gets ALTER TABLE ADD COLUMN'd.
_EXPECTED_COLUMNS: dict[str, list[tuple[str, str]]] = {
    "tool_calls": [
        ("raw_model_output", "TEXT"),
        ("parsed_tool_call", "TEXT"),
        ("is_golden", "INTEGER NOT NULL DEFAULT 0"),
    ],
    "circuit_breaker_diagnostics": [
        ("golden_reference", "TEXT"),
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
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
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
               raw_model_output, parsed_tool_call, is_golden)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                            "calls": int,
                            "successes": int,
                            "failures": int,
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
                        "calls": int,
                        "success_rate": float,
                        "avg_retries": float,
                        "avg_latency_ms": float,
                        "error_types": {"<type>": int, ...},
                    },
                    ...
                },
                "total_calls": int,
                "overall_success_rate": float,
            }
        """
        query = (
            "SELECT model, tool_name, success, retries, latency_ms, error_type"
            " FROM tool_calls"
        )
        params: tuple = ()
        if since is not None:
            query += " WHERE timestamp >= ?"
            params = (since,)
        rows = self._conn.execute(query, params).fetchall()

        if not rows:
            return {
                "models": {},
                "tools": {},
                "total_calls": 0,
                "overall_success_rate": 0.0,
            }

        # Aggregate
        models: dict[str, dict[str, dict[str, Any]]] = {}
        tools: dict[str, dict[str, Any]] = {}
        total = 0
        total_success = 0

        for model, tool_name, success, retries, latency_ms, error_type in rows:
            total += 1
            total_success += success

            # per-model per-tool
            model_data = models.setdefault(model, {})
            mt = model_data.setdefault(
                tool_name,
                {"calls": 0, "successes": 0, "failures": 0,
                 "total_retries": 0, "total_latency_ms": 0.0},
            )
            mt["calls"] += 1
            mt["successes"] += success
            mt["failures"] += 1 - success
            mt["total_retries"] += retries
            mt["total_latency_ms"] += latency_ms

            # per-tool
            td = tools.setdefault(
                tool_name,
                {"calls": 0, "successes": 0, "total_retries": 0,
                 "total_latency_ms": 0.0, "error_types": {}},
            )
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
            "overall_success_rate": total_success / total if total else 0.0,
        }

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
