"""ToolDashboard — structured queries against the telemetry SQLite DB.

Provides high-level stats (success rates, latency, circuit-breaker trips,
lucky guesses, adapter repairs) without coupling to ToolCallTelemetry.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from prometheus.telemetry.db import connect_telemetry_db
from prometheus.telemetry.tracker import POLICY_ERROR_TYPES

# SQL fragment placeholders for the policy error types (D3 denominator
# honesty — see tracker.POLICY_ERROR_TYPES).
_POLICY_PH = ",".join("?" * len(POLICY_ERROR_TYPES))
_POLICY_PARAMS = tuple(POLICY_ERROR_TYPES)


class ToolDashboard:
    """Read-only dashboard over the telemetry database.

    Opens its own connection to the same SQLite file used by
    :class:`~prometheus.telemetry.tracker.ToolCallTelemetry` so the two
    classes are fully decoupled.

    Usage::

        dash = ToolDashboard()
        stats = dash.get_stats(hours=24)
        print(stats["success_rate_by_tool"])
    """

    def __init__(self, db_path: str | Path = "~/.prometheus/telemetry.db") -> None:
        self._db_path = Path(db_path).expanduser().resolve()
        # Shared WAL + busy_timeout setup (see telemetry.db) — the reader side
        # of the same concurrency-safe substrate the writer uses.
        self._conn = connect_telemetry_db(self._db_path)
        self._conn.row_factory = sqlite3.Row

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_stats(self, hours: int = 24) -> dict[str, Any]:
        """Return structured stats for the last *hours* hours.

        Keys returned:

        * ``success_rate_by_tool``  – ``{tool_name: float}``
        * ``most_called``           – top-10 tools by call count (list of dicts)
        * ``avg_latency_by_tool``   – ``{tool_name: float}``
        * ``circuit_breaker_trips`` – count of ``_loop_transition`` records
          with ``error_type='circuit_breaker_trip'``
        * ``lucky_guesses``         – count of records with
          ``error_type='lucky_guess'``
        * ``adapter_repairs``       – count of records where ``retries > 0``
        * ``total_calls``           – real tool calls in window (synthetic
          ``_loop_transition`` loop echoes excluded)
        * ``total_denials``         – policy refusals (SecurityGate / hooks);
          included in ``total_calls``, excluded from success rates
        * ``overall_success_rate``  – float 0.0 – 1.0 over non-denied calls
        """
        cutoff = time.time() - hours * 3600

        success_rate_by_tool = self._success_rate_by_tool(cutoff)
        most_called = self._most_called(cutoff)
        avg_latency_by_tool = self._avg_latency_by_tool(cutoff)
        circuit_breaker_trips = self._count_circuit_breaker_trips(cutoff)
        lucky_guesses = self._count_lucky_guesses(cutoff)
        adapter_repairs = self._count_adapter_repairs(cutoff)
        total_calls, total_denials, overall_success_rate = self._totals(cutoff)

        return {
            "success_rate_by_tool": success_rate_by_tool,
            "most_called": most_called,
            "avg_latency_by_tool": avg_latency_by_tool,
            "circuit_breaker_trips": circuit_breaker_trips,
            "lucky_guesses": lucky_guesses,
            "adapter_repairs": adapter_repairs,
            "total_calls": total_calls,
            "total_denials": total_denials,
            "overall_success_rate": overall_success_rate,
        }

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _success_rate_by_tool(self, cutoff: float) -> dict[str, float]:
        # M1: synthetic _loop_transition rows are loop echoes, not tools.
        # D3: policy denials are excluded from the rate denominator.
        rows = self._conn.execute(
            f"""
            SELECT tool_name,
                   CAST(SUM(success) AS REAL) / COUNT(*) AS rate
              FROM tool_calls
             WHERE timestamp >= ?
               AND tool_name != '_loop_transition'
               AND (error_type IS NULL OR error_type NOT IN ({_POLICY_PH}))
             GROUP BY tool_name
            """,
            (cutoff, *_POLICY_PARAMS),
        ).fetchall()
        return {r["tool_name"]: r["rate"] for r in rows}

    def _most_called(self, cutoff: float) -> list[dict[str, Any]]:
        # Denied calls still count as calls (volume metric); loop echoes don't.
        rows = self._conn.execute(
            """
            SELECT tool_name, COUNT(*) AS calls
              FROM tool_calls
             WHERE timestamp >= ?
               AND tool_name != '_loop_transition'
             GROUP BY tool_name
             ORDER BY calls DESC
             LIMIT 10
            """,
            (cutoff,),
        ).fetchall()
        return [{"tool_name": r["tool_name"], "calls": r["calls"]} for r in rows]

    def _avg_latency_by_tool(self, cutoff: float) -> dict[str, float]:
        """Mean execution time per tool, over rows that were actually MEASURED.

        `latency_ms` is `REAL NOT NULL DEFAULT 0.0`, so "nobody measured this" and "measured as
        zero" are the same stored value. Real execution is timed with `time.monotonic()`, which
        does not return an exact 0.0 — so a 0.0 is always the placeholder, written by the record
        sites for calls that never ran.

        `NULLIF(latency_ms, 0)` is the load-bearing guard: AVG skips NULL, so every never-ran row
        drops out WITHOUT anyone maintaining a list of the ways a call can fail to execute. The
        previous enumeration had already fallen out of sync — `validation_failed`, `unknown_tool`
        and `input_validation` all mean "never ran" and none were in POLICY_ERROR_TYPES, so they
        were being averaged in as zero-duration executions (measured: lcm_grep 0.64 -> 0.85ms).

        Both filters below are, TODAY, redundant with NULLIF: every `_loop_transition` row and
        every policy-denial row is unmeasured, so NULLIF already drops them (measured: 7,000 and
        147 rows respectively, all 0.0). Mutation testing proved it — deleting either one changed
        no observable behaviour on realistic data.

        They are kept anyway, and the distinction matters if one ever stops being unmeasured:
          * `tool_name != '_loop_transition'` — loop bookkeeping is not a tool, so it must not be
            a row in a per-TOOL table even if something starts timing it.
          * the POLICY_ERROR_TYPES clause — a denial that somehow carried a duration still did not
            execute.
        Neither is load-bearing for the bug this method had; NULLIF is.

        A tool with NO measured rows is OMITTED rather than reported as 0.0 — "we never timed
        this" is not "this is instant", which is the whole point of the change.
        """
        rows = self._conn.execute(
            f"""
            SELECT tool_name, AVG(NULLIF(latency_ms, 0)) AS avg_lat
              FROM tool_calls
             WHERE timestamp >= ?
               AND tool_name != '_loop_transition'
               AND (error_type IS NULL OR error_type NOT IN ({_POLICY_PH}))
             GROUP BY tool_name
            """,
            (cutoff, *_POLICY_PARAMS),
        ).fetchall()
        # AVG over an all-NULL group is NULL. Dropping the key keeps the declared dict[str, float]
        # honest instead of smuggling a None through it.
        return {r["tool_name"]: r["avg_lat"] for r in rows if r["avg_lat"] is not None}

    def _count_circuit_breaker_trips(self, cutoff: float) -> int:
        row = self._conn.execute(
            """
            SELECT COUNT(*) AS cnt
              FROM tool_calls
             WHERE timestamp >= ?
               AND tool_name = '_loop_transition'
               AND error_type = 'circuit_breaker_trip'
            """,
            (cutoff,),
        ).fetchone()
        return row["cnt"]

    def _count_lucky_guesses(self, cutoff: float) -> int:
        row = self._conn.execute(
            """
            SELECT COUNT(*) AS cnt
              FROM tool_calls
             WHERE timestamp >= ?
               AND error_type = 'lucky_guess'
            """,
            (cutoff,),
        ).fetchone()
        return row["cnt"]

    def _count_adapter_repairs(self, cutoff: float) -> int:
        row = self._conn.execute(
            """
            SELECT COUNT(*) AS cnt
              FROM tool_calls
             WHERE timestamp >= ?
               AND retries > 0
            """,
            (cutoff,),
        ).fetchone()
        return row["cnt"]

    def _totals(self, cutoff: float) -> tuple[int, int, float]:
        # total counts every real tool call (denials included — the model
        # made the call); the success rate is judged over non-denied calls.
        row = self._conn.execute(
            f"""
            SELECT COUNT(*) AS total,
                   COALESCE(SUM(error_type IN ({_POLICY_PH})), 0) AS denials,
                   COALESCE(
                       CAST(SUM(success) AS REAL)
                       / NULLIF(COUNT(*) - COALESCE(SUM(error_type IN ({_POLICY_PH})), 0), 0),
                       0.0
                   ) AS rate
              FROM tool_calls
             WHERE timestamp >= ?
               AND tool_name != '_loop_transition'
            """,
            (*_POLICY_PARAMS, *_POLICY_PARAMS, cutoff),
        ).fetchone()
        return row["total"], row["denials"], row["rate"]
