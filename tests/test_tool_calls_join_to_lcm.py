"""A web-path tool_calls row must JOIN to lcm_messages (piece 5).

``tool_calls.session_id`` exists for one documented reason — the comment above
it says the column is there so a row can "join back to lcm_messages for that
context", because WHAT was called is not a trainable example without the
situation that prompted it.

On the web path that join matched nothing. The writer read
``context.session_id``, which on the shared web LoopContext is the literal
"web" — a model-routing namespace, not a conversation id — while lcm_messages
is keyed by the real ``web:<id>``.

Tested as the JOIN, deliberately, not as ``session_id != "web"``: the column
being "not-web" proves nothing about whether a reader can actually reach the
conversation from a tool-call row.
"""

from __future__ import annotations

import sqlite3

import pytest

from prometheus.telemetry.tracker import ToolCallTelemetry

WEB_NAMESPACE = "web"          # what daemon.py pins on the shared context
REAL_SESSION = "web:s-42"      # what lcm_messages is actually keyed by


def _lcm_with_a_conversation(path) -> None:
    """Stand in for the durable store, keyed the way the real one is."""
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE lcm_messages (id TEXT PRIMARY KEY, session_id TEXT NOT NULL,"
        " role TEXT, content TEXT)"
    )
    con.executemany(
        "INSERT INTO lcm_messages VALUES (?,?,?,?)",
        [("m1", REAL_SESSION, "user", "what model is this?"),
         ("m2", REAL_SESSION, "assistant", "checking…")],
    )
    con.commit(); con.close()


def _join_rows(tel_db, lcm_db) -> list[tuple]:
    """The join an analyst would actually run, across the two stores."""
    con = sqlite3.connect(tel_db)
    con.execute("ATTACH DATABASE ? AS lcm", (str(lcm_db),))
    try:
        return con.execute(
            "SELECT t.tool_name, m.role, m.content "
            "FROM tool_calls t JOIN lcm.lcm_messages m "
            "  ON m.session_id = t.session_id"
        ).fetchall()
    finally:
        con.close()


@pytest.fixture
def stores(tmp_path):
    lcm_db = tmp_path / "lcm.db"
    tel_db = tmp_path / "telemetry.db"
    _lcm_with_a_conversation(lcm_db)
    return tel_db, lcm_db


def test_a_web_path_tool_call_joins_to_its_conversation(stores):
    """THE OUTCOME. The row must reach the conversation that prompted it."""
    tel_db, lcm_db = stores
    tel = ToolCallTelemetry(db_path=str(tel_db))
    # What the fixed writer records: the per-call effective id, not the
    # namespace on the shared context.
    tel.record(model="m", tool_name="bash", success=True, session_id=REAL_SESSION)

    rows = _join_rows(tel_db, lcm_db)
    assert rows, (
        "a web-path tool_calls row does not join to lcm_messages — the column's "
        "documented purpose (reaching the conversation that prompted the call) "
        "is unmet"
    )
    assert {r[1] for r in rows} == {"user", "assistant"}
    assert any("what model is this?" in r[2] for r in rows)


def test_the_old_behaviour_would_have_joined_to_nothing(stores):
    """The defect, pinned. Recording the routing namespace yields a row that
    looks populated and reaches nothing — which is why a non-null check would
    have passed while the column was useless."""
    tel_db, lcm_db = stores
    tel = ToolCallTelemetry(db_path=str(tel_db))
    tel.record(model="m", tool_name="bash", success=True, session_id=WEB_NAMESPACE)

    stored = sqlite3.connect(tel_db).execute(
        "SELECT session_id FROM tool_calls").fetchone()[0]
    assert stored == WEB_NAMESPACE          # populated: nothing "looks missing"
    assert stored is not None
    assert _join_rows(tel_db, lcm_db) == []  # ...and joins to nothing


def test_ephemeral_still_records_no_session(stores):
    """The privacy carve-out survives the fix: ephemeral turns null the column
    on purpose, consistently with the content columns."""
    tel_db, lcm_db = stores
    tel = ToolCallTelemetry(db_path=str(tel_db))
    tel.record(model="m", tool_name="bash", success=True, session_id=None)
    stored = sqlite3.connect(tel_db).execute(
        "SELECT session_id FROM tool_calls").fetchone()[0]
    assert stored is None
    assert _join_rows(tel_db, lcm_db) == []


# ---------------------------------------------------------------------------
# The REAL path. The tests above prove the join works given the right value;
# these prove the web path now SUPPLIES it — the wiring, not the schema.
# ---------------------------------------------------------------------------


def _web_context(tmp_path, tel_db):
    """A LoopContext shaped like the daemon's SHARED web one: session_id is the
    routing namespace "web", not a conversation id."""
    from pathlib import Path

    from prometheus.__main__ import create_tool_registry
    from prometheus.engine.agent_loop import LoopContext
    from prometheus.permissions.checker import SecurityGate

    ws = Path(tmp_path) / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    gate = SecurityGate(workspace_root=[str(ws)])
    return LoopContext(
        provider=None, model="t", system_prompt="", max_tokens=512,
        tool_registry=create_tool_registry({}, gate),
        permission_checker=gate,
        telemetry=ToolCallTelemetry(db_path=str(tel_db)),
        session_id=WEB_NAMESPACE,
    ), ws


def _run_a_tool(ctx, ws, **kw):
    import asyncio

    from prometheus.engine.agent_loop import _execute_tool_call

    return asyncio.run(_execute_tool_call(
        ctx, "write_file", "t1",
        {"path": str(ws / "f.txt"), "content": "x\n"}, **kw))


def test_the_web_path_records_the_conversation_not_the_namespace(stores, tmp_path):
    """End of the thread: _run_loop's effective id reaches the tool_calls row."""
    tel_db, lcm_db = stores
    ctx, ws = _web_context(tmp_path, tel_db)

    _run_a_tool(ctx, ws, effective_session_id=REAL_SESSION)

    stored = sqlite3.connect(tel_db).execute(
        "SELECT session_id FROM tool_calls WHERE tool_name='write_file'").fetchone()[0]
    assert stored == REAL_SESSION, (
        f"the web path recorded {stored!r}; the threading is broken"
    )
    assert _join_rows(tel_db, lcm_db), "recorded id does not reach the conversation"


def test_without_the_thread_it_falls_back_to_the_namespace(stores, tmp_path):
    """Pins the pre-fix behaviour that the fallback still reproduces, so the
    difference this PR makes is visible rather than asserted."""
    tel_db, lcm_db = stores
    ctx, ws = _web_context(tmp_path, tel_db)

    _run_a_tool(ctx, ws)  # no effective_session_id — the old read

    stored = sqlite3.connect(tel_db).execute(
        "SELECT session_id FROM tool_calls WHERE tool_name='write_file'").fetchone()[0]
    assert stored == WEB_NAMESPACE
    assert _join_rows(tel_db, lcm_db) == []
