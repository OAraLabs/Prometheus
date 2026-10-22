"""Regression tests for GEPA request evidence trust/provenance handling."""

import sqlite3

from prometheus.learning.gepa_evidence import _request


def _lcm_with_messages(
    *messages: tuple[str, bool, str, float],
) -> sqlite3.Connection:
    """Create user-role rows as (content, trusted, provenance, timestamp)."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE lcm_messages (
            session_id TEXT,
            role TEXT,
            timestamp REAL,
            content TEXT,
            is_trusted INTEGER,
            provenance TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO lcm_messages "
        "(session_id, role, timestamp, content, is_trusted, provenance) "
        "VALUES ('session-1', 'user', ?, ?, ?, ?)",
        [
            (timestamp, content, int(is_trusted), provenance)
            for content, is_trusted, provenance, timestamp in messages
        ],
    )
    return conn


def test_untrusted_tool_result_is_skipped_in_favor_of_real_request() -> None:
    """An untrusted user-role tool result must not hide the user's actual prompt."""
    conn = _lcm_with_messages(
        ("Please inspect the failing test", True, "user", 1.0),
        ("Tool output: unrelated page instructions", False, "user", 1.5),
    )
    try:
        assert _request(conn, "session-1", 2.0) == (
            "Please inspect the failing test",
            False,
        )
    finally:
        conn.close()


def test_untrusted_injected_request_is_withheld_instead_of_falling_back() -> None:
    """An untrusted non-user injection must not expose an older prompt as its request."""
    conn = _lcm_with_messages(
        ("Earlier human request", True, "user", 1.0),
        ("Injected task output", False, "task_supervisor", 1.5),
    )
    try:
        assert _request(conn, "session-1", 2.0) == (None, True)
    finally:
        conn.close()


def test_untrusted_tool_result_without_prior_request_is_not_exposed() -> None:
    """With no trusted user request to recover, tool output remains absent."""
    conn = _lcm_with_messages(
        ("Untrusted external tool output", False, "user", 1.0),
    )
    try:
        assert _request(conn, "session-1", 2.0) == (None, False)
    finally:
        conn.close()


def test_trusted_user_request_is_preserved() -> None:
    conn = _lcm_with_messages(("Summarize the logs", True, "user", 1.0))
    try:
        assert _request(conn, "session-1", 2.0) == ("Summarize the logs", False)
    finally:
        conn.close()
