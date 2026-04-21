"""Trend tracking for evaluation scores over time.

Stores per-run summary rows in a SQLite database so you can track
score improvements and regressions across nightly runs.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS eval_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    task_count INTEGER NOT NULL,
    completed INTEGER NOT NULL,
    errored INTEGER NOT NULL,
    avg_latency_ms REAL NOT NULL,
    metric_averages TEXT NOT NULL
);
"""


@dataclass
class TrendRow:
    """A single row from the trend history."""

    timestamp: str
    task_count: int
    completed: int
    errored: int
    avg_latency_ms: float
    metric_averages: dict[str, float]


class TrendTracker:
    """Track evaluation scores over time in SQLite."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(
            db_path or Path.home() / ".prometheus" / "eval_results" / "trends.db"
        )
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def record(self, summary: dict[str, Any]) -> None:
        """Record a run summary from EvalRunner._compute_summary()."""
        self._conn.execute(
            "INSERT INTO eval_runs (timestamp, task_count, completed, errored, "
            "avg_latency_ms, metric_averages) VALUES (?, ?, ?, ?, ?, ?)",
            (
                datetime.now().isoformat(),
                summary.get("total_tasks", 0),
                summary.get("completed", 0),
                summary.get("errored", 0),
                summary.get("avg_latency_ms", 0.0),
                json.dumps(summary.get("metric_averages", {})),
            ),
        )
        self._conn.commit()
        log.info("Trend recorded to %s", self._db_path)

    def get_latest(self, n: int = 10) -> list[TrendRow]:
        """Get the most recent N runs."""
        cursor = self._conn.execute(
            "SELECT timestamp, task_count, completed, errored, avg_latency_ms, "
            "metric_averages FROM eval_runs ORDER BY id DESC LIMIT ?",
            (n,),
        )
        rows = []
        for row in cursor.fetchall():
            rows.append(
                TrendRow(
                    timestamp=row[0],
                    task_count=row[1],
                    completed=row[2],
                    errored=row[3],
                    avg_latency_ms=row[4],
                    metric_averages=json.loads(row[5]),
                )
            )
        return rows

    def get_previous(self) -> TrendRow | None:
        """Get the most recent run (before the current one)."""
        rows = self.get_latest(1)
        return rows[0] if rows else None

    def format_trend_comparison(
        self, current: dict[str, float], previous: TrendRow | None
    ) -> str:
        """Format a comparison between current and previous run scores.

        Returns a multi-line string like:
            Task Completion: 0.850  (+0.130 vs prev)
            Tool Usage:      0.920  (new)
        """
        if not current:
            return "  (no metrics)"

        lines = []
        prev_avgs = previous.metric_averages if previous else {}
        max_name_len = max(len(n) for n in current)

        for name, score in current.items():
            padded = name.ljust(max_name_len)
            if name in prev_avgs:
                delta = score - prev_avgs[name]
                sign = "+" if delta >= 0 else ""
                lines.append(f"    {padded}: {score:.3f}  ({sign}{delta:.3f} vs prev)")
            else:
                lines.append(f"    {padded}: {score:.3f}  (new)")

        return "\n".join(lines)

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
