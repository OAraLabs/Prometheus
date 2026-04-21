"""ToolCallTelemetry — per-model, per-tool success/retry/latency tracking.

Storage: SQLite at ~/.prometheus/telemetry.db (or a path you specify).
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any
from uuid import uuid4


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tool_calls (
    id           TEXT PRIMARY KEY,
    timestamp    REAL NOT NULL,
    model        TEXT NOT NULL,
    tool_name    TEXT NOT NULL,
    success      INTEGER NOT NULL,   -- 0 or 1
    retries      INTEGER NOT NULL DEFAULT 0,
    latency_ms   REAL NOT NULL DEFAULT 0.0,
    error_type   TEXT,
    error_detail TEXT
);

CREATE INDEX IF NOT EXISTS idx_tool_calls_model ON tool_calls (model);
CREATE INDEX IF NOT EXISTS idx_tool_calls_tool ON tool_calls (tool_name);
"""


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
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

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
    ) -> None:
        """Record a single tool-call outcome."""
        self._conn.execute(
            """
            INSERT INTO tool_calls
              (id, timestamp, model, tool_name, success, retries, latency_ms, error_type, error_detail)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            ),
        )
        self._conn.commit()

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
