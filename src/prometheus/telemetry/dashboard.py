"""ToolDashboard — structured queries against the telemetry SQLite DB.

Provides high-level stats (success rates, latency, circuit-breaker trips,
lucky guesses, adapter repairs) without coupling to ToolCallTelemetry.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any


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
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
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
        * ``total_calls``           – total row count in window
        * ``overall_success_rate``  – float 0.0 – 1.0
        """
        cutoff = time.time() - hours * 3600

        success_rate_by_tool = self._success_rate_by_tool(cutoff)
        most_called = self._most_called(cutoff)
        avg_latency_by_tool = self._avg_latency_by_tool(cutoff)
        circuit_breaker_trips = self._count_circuit_breaker_trips(cutoff)
        lucky_guesses = self._count_lucky_guesses(cutoff)
        adapter_repairs = self._count_adapter_repairs(cutoff)
        total_calls, overall_success_rate = self._totals(cutoff)

        return {
            "success_rate_by_tool": success_rate_by_tool,
            "most_called": most_called,
            "avg_latency_by_tool": avg_latency_by_tool,
            "circuit_breaker_trips": circuit_breaker_trips,
            "lucky_guesses": lucky_guesses,
            "adapter_repairs": adapter_repairs,
            "total_calls": total_calls,
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
        rows = self._conn.execute(
            """
            SELECT tool_name,
                   CAST(SUM(success) AS REAL) / COUNT(*) AS rate
              FROM tool_calls
             WHERE timestamp >= ?
             GROUP BY tool_name
            """,
            (cutoff,),
        ).fetchall()
        return {r["tool_name"]: r["rate"] for r in rows}

    def _most_called(self, cutoff: float) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT tool_name, COUNT(*) AS calls
              FROM tool_calls
             WHERE timestamp >= ?
             GROUP BY tool_name
             ORDER BY calls DESC
             LIMIT 10
            """,
            (cutoff,),
        ).fetchall()
        return [{"tool_name": r["tool_name"], "calls": r["calls"]} for r in rows]

    def _avg_latency_by_tool(self, cutoff: float) -> dict[str, float]:
        rows = self._conn.execute(
            """
            SELECT tool_name, AVG(latency_ms) AS avg_lat
              FROM tool_calls
             WHERE timestamp >= ?
             GROUP BY tool_name
            """,
            (cutoff,),
        ).fetchall()
        return {r["tool_name"]: r["avg_lat"] for r in rows}

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

    def _totals(self, cutoff: float) -> tuple[int, float]:
        row = self._conn.execute(
            """
            SELECT COUNT(*) AS total,
                   COALESCE(CAST(SUM(success) AS REAL) / NULLIF(COUNT(*), 0), 0.0)
                       AS rate
              FROM tool_calls
             WHERE timestamp >= ?
            """,
            (cutoff,),
        ).fetchone()
        return row["total"], row["rate"]
