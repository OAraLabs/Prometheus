"""GymStore — run-level records, keyed by (series, experiment, task, run).

Separate file from live telemetry.db on purpose: gym runs include INDUCED
failures; mixing them into the live tool_calls table would distort the
dashboards this sprint just made honest. The pipeline still records through
a REAL ToolCallTelemetry instance — it just points at this database.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

DEFAULT_GYM_DB = "~/.prometheus/data/gym.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS gym_runs (
    series          TEXT NOT NULL,
    experiment      TEXT NOT NULL,
    task_id         TEXT NOT NULL,
    run_idx         INTEGER NOT NULL,
    timestamp       REAL NOT NULL,
    model           TEXT NOT NULL,
    category        TEXT NOT NULL,
    success         INTEGER NOT NULL,
    fail_reasons    TEXT,
    tools_called    TEXT,            -- JSON list of tool names in order
    latency_ms      REAL NOT NULL DEFAULT 0,
    retries         INTEGER NOT NULL DEFAULT 0,
    repairs         INTEGER NOT NULL DEFAULT 0,
    dropped_malformed INTEGER NOT NULL DEFAULT 0,
    feedback_retries  INTEGER NOT NULL DEFAULT 0,
    breaker_tripped   INTEGER NOT NULL DEFAULT 0,
    error           TEXT,            -- harness-level error (timeout, crash)
    manifest_sha    TEXT NOT NULL,
    taskset_sha     TEXT NOT NULL,
    PRIMARY KEY (series, experiment, task_id, run_idx)
);
"""


class GymStore:
    def __init__(self, db_path: str | Path = DEFAULT_GYM_DB) -> None:
        p = Path(db_path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(p))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def record_run(self, **row: Any) -> None:
        cols = (
            "series", "experiment", "task_id", "run_idx", "timestamp", "model",
            "category", "success", "fail_reasons", "tools_called", "latency_ms",
            "retries", "repairs", "dropped_malformed", "feedback_retries",
            "breaker_tripped", "error", "manifest_sha", "taskset_sha",
        )
        row.setdefault("timestamp", time.time())
        values = [row.get(c) for c in cols]
        self._conn.execute(
            f"INSERT OR REPLACE INTO gym_runs ({', '.join(cols)}) "
            f"VALUES ({', '.join('?' * len(cols))})",
            values,
        )
        self._conn.commit()

    def runs(self, series: str, experiment: str) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM gym_runs WHERE series = ? AND experiment = ? "
            "ORDER BY task_id, run_idx",
            (series, experiment),
        ).fetchall()

    def experiments(self, series: str) -> list[str]:
        return [
            r[0] for r in self._conn.execute(
                "SELECT DISTINCT experiment FROM gym_runs WHERE series = ? "
                "ORDER BY experiment",
                (series,),
            ).fetchall()
        ]

    def close(self) -> None:
        self._conn.close()
