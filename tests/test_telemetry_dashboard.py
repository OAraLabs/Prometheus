"""Tests for ToolDashboard — structured telemetry queries."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from uuid import uuid4

import pytest

from prometheus.telemetry.dashboard import ToolDashboard


# -- Schema (duplicated intentionally so tests don't couple to tracker.py) ----

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tool_calls (
    id           TEXT PRIMARY KEY,
    timestamp    REAL NOT NULL,
    model        TEXT NOT NULL,
    tool_name    TEXT NOT NULL,
    success      INTEGER NOT NULL,
    retries      INTEGER NOT NULL DEFAULT 0,
    latency_ms   REAL NOT NULL DEFAULT 0.0,
    error_type   TEXT,
    error_detail TEXT
);
"""


def _insert(
    conn: sqlite3.Connection,
    *,
    tool_name: str = "bash",
    model: str = "qwen2.5",
    success: int = 1,
    retries: int = 0,
    latency_ms: float = 100.0,
    error_type: str | None = None,
    error_detail: str | None = None,
    timestamp: float | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO tool_calls
          (id, timestamp, model, tool_name, success, retries, latency_ms,
           error_type, error_detail)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            uuid4().hex,
            timestamp if timestamp is not None else time.time(),
            model,
            tool_name,
            success,
            retries,
            latency_ms,
            error_type,
            error_detail,
        ),
    )
    conn.commit()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Create a fresh telemetry DB and return its path."""
    p = tmp_path / "telemetry.db"
    conn = sqlite3.connect(str(p))
    conn.executescript(_SCHEMA_SQL)
    conn.close()
    return p


@pytest.fixture
def dash(db_path: Path) -> ToolDashboard:
    d = ToolDashboard(db_path=db_path)
    yield d
    d.close()


# ---------------------------------------------------------------------------
# Empty DB
# ---------------------------------------------------------------------------


class TestEmptyDB:

    def test_empty_returns_zeroes(self, dash: ToolDashboard):
        stats = dash.get_stats(hours=24)
        assert stats["total_calls"] == 0
        assert stats["overall_success_rate"] == 0.0
        assert stats["success_rate_by_tool"] == {}
        assert stats["most_called"] == []
        assert stats["avg_latency_by_tool"] == {}
        assert stats["circuit_breaker_trips"] == 0
        assert stats["lucky_guesses"] == 0
        assert stats["adapter_repairs"] == 0


# ---------------------------------------------------------------------------
# Populated DB
# ---------------------------------------------------------------------------


class TestPopulatedDB:

    def test_success_rate_by_tool(self, db_path: Path):
        conn = sqlite3.connect(str(db_path))
        _insert(conn, tool_name="bash", success=1)
        _insert(conn, tool_name="bash", success=1)
        _insert(conn, tool_name="bash", success=0)
        _insert(conn, tool_name="read", success=1)
        conn.close()

        dash = ToolDashboard(db_path=db_path)
        stats = dash.get_stats(hours=24)
        assert abs(stats["success_rate_by_tool"]["bash"] - 2 / 3) < 0.001
        assert stats["success_rate_by_tool"]["read"] == 1.0
        dash.close()

    def test_most_called_ordering(self, db_path: Path):
        conn = sqlite3.connect(str(db_path))
        for _ in range(5):
            _insert(conn, tool_name="bash")
        for _ in range(3):
            _insert(conn, tool_name="read")
        _insert(conn, tool_name="write")
        conn.close()

        dash = ToolDashboard(db_path=db_path)
        stats = dash.get_stats(hours=24)
        top = stats["most_called"]
        assert top[0]["tool_name"] == "bash"
        assert top[0]["calls"] == 5
        assert top[1]["tool_name"] == "read"
        assert top[1]["calls"] == 3
        assert top[2]["tool_name"] == "write"
        assert top[2]["calls"] == 1
        dash.close()

    def test_most_called_caps_at_10(self, db_path: Path):
        conn = sqlite3.connect(str(db_path))
        for i in range(15):
            _insert(conn, tool_name=f"tool_{i:02d}")
        conn.close()

        dash = ToolDashboard(db_path=db_path)
        stats = dash.get_stats(hours=24)
        assert len(stats["most_called"]) == 10
        dash.close()

    def test_avg_latency_by_tool(self, db_path: Path):
        conn = sqlite3.connect(str(db_path))
        _insert(conn, tool_name="bash", latency_ms=100.0)
        _insert(conn, tool_name="bash", latency_ms=200.0)
        _insert(conn, tool_name="read", latency_ms=50.0)
        conn.close()

        dash = ToolDashboard(db_path=db_path)
        stats = dash.get_stats(hours=24)
        assert stats["avg_latency_by_tool"]["bash"] == 150.0
        assert stats["avg_latency_by_tool"]["read"] == 50.0
        dash.close()

    def test_circuit_breaker_trips(self, db_path: Path):
        conn = sqlite3.connect(str(db_path))
        _insert(conn, tool_name="_loop_transition", success=0,
                error_type="circuit_breaker_trip")
        _insert(conn, tool_name="_loop_transition", success=0,
                error_type="circuit_breaker_trip")
        # This one should NOT count — wrong tool_name
        _insert(conn, tool_name="bash", success=0,
                error_type="circuit_breaker_trip")
        # This one should NOT count — wrong error_type
        _insert(conn, tool_name="_loop_transition", success=0,
                error_type="other_error")
        conn.close()

        dash = ToolDashboard(db_path=db_path)
        stats = dash.get_stats(hours=24)
        assert stats["circuit_breaker_trips"] == 2
        dash.close()

    def test_lucky_guesses(self, db_path: Path):
        conn = sqlite3.connect(str(db_path))
        _insert(conn, tool_name="bash", error_type="lucky_guess")
        _insert(conn, tool_name="read", error_type="lucky_guess")
        _insert(conn, tool_name="bash", error_type="other")
        conn.close()

        dash = ToolDashboard(db_path=db_path)
        stats = dash.get_stats(hours=24)
        assert stats["lucky_guesses"] == 2
        dash.close()

    def test_adapter_repairs(self, db_path: Path):
        conn = sqlite3.connect(str(db_path))
        _insert(conn, retries=0)
        _insert(conn, retries=1)
        _insert(conn, retries=3)
        conn.close()

        dash = ToolDashboard(db_path=db_path)
        stats = dash.get_stats(hours=24)
        assert stats["adapter_repairs"] == 2
        dash.close()

    def test_total_calls_and_overall_success(self, db_path: Path):
        conn = sqlite3.connect(str(db_path))
        _insert(conn, success=1)
        _insert(conn, success=1)
        _insert(conn, success=0)
        _insert(conn, success=1)
        conn.close()

        dash = ToolDashboard(db_path=db_path)
        stats = dash.get_stats(hours=24)
        assert stats["total_calls"] == 4
        assert abs(stats["overall_success_rate"] - 0.75) < 0.001
        dash.close()


# ---------------------------------------------------------------------------
# Time filtering
# ---------------------------------------------------------------------------


class TestTimeFiltering:

    def test_only_recent_rows_included(self, db_path: Path):
        now = time.time()
        conn = sqlite3.connect(str(db_path))
        # Two recent rows (1 hour ago)
        _insert(conn, tool_name="bash", success=1, timestamp=now - 3600)
        _insert(conn, tool_name="bash", success=0, timestamp=now - 3600)
        # One old row (48 hours ago — outside default 24h window)
        _insert(conn, tool_name="bash", success=1, timestamp=now - 48 * 3600)
        conn.close()

        dash = ToolDashboard(db_path=db_path)
        stats = dash.get_stats(hours=24)
        assert stats["total_calls"] == 2
        assert stats["success_rate_by_tool"]["bash"] == 0.5
        dash.close()

    def test_wider_window_includes_older_rows(self, db_path: Path):
        now = time.time()
        conn = sqlite3.connect(str(db_path))
        _insert(conn, tool_name="bash", success=1, timestamp=now - 3600)
        _insert(conn, tool_name="bash", success=1, timestamp=now - 48 * 3600)
        conn.close()

        dash = ToolDashboard(db_path=db_path)
        stats = dash.get_stats(hours=72)
        assert stats["total_calls"] == 2
        dash.close()

    def test_narrow_window_excludes_recent(self, db_path: Path):
        now = time.time()
        conn = sqlite3.connect(str(db_path))
        # 30 minutes ago — inside a 1-hour window
        _insert(conn, tool_name="bash", success=1, timestamp=now - 1800)
        # 2 hours ago — outside a 1-hour window
        _insert(conn, tool_name="bash", success=1, timestamp=now - 7200)
        conn.close()

        dash = ToolDashboard(db_path=db_path)
        stats = dash.get_stats(hours=1)
        assert stats["total_calls"] == 1
        dash.close()
