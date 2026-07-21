"""Durable session index — GET /api/sessions survives restart (feat/durable-session-index).

The route used to enumerate ``session_mgr._sessions`` only, so every daemon
restart emptied the session list even though the LCM store still held every
message. It now enumerates from ``LCMConversationStore.list_sessions()`` and
overlays the in-memory working set. The properties that matter:

  * A fresh SessionManager (= a restarted daemon) still lists every persisted
    session, with the SAME watermark cursor the messages route reports, and
    ``live: false`` marking that the in-memory working set is gone.
  * An in-memory session wins on collision (``live: true``, live field
    semantics unchanged from the pre-durable route).
  * "Forget session" now survives restart: DELETE writes a durable tombstone.
    The append-only rows stay intact — and NEWER activity revives the session,
    because gateway ids like telegram:123 are stable and a forgotten chat must
    resurface when it speaks again.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.engine.session import SessionManager  # noqa: E402
from prometheus.memory.lcm_conversation_store import LCMConversationStore  # noqa: E402
from prometheus.memory.lcm_types import MessagePart  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402


def _engine(store: LCMConversationStore):
    class _Engine:
        conversation_store = store

        def ingest_sync(self, session_id, role, content, turn_index=0, content_json=None,
                        provenance="user", is_trusted=True):
            m = MessagePart(role=role, content=content, session_id=session_id, turn_index=turn_index,
                            provenance=provenance, is_trusted=is_trusted)
            store.add_message(session_id, m)
            return m.message_id

    return _Engine()


def _seeded(tmp_path, session_ids=("telegram:123", "desktop:smoke")):
    """A store + live manager with one persisted user message per session."""
    store = LCMConversationStore(tmp_path / "lcm.db")
    engine = _engine(store)
    mgr = SessionManager()
    mgr.lcm_engine = engine
    for sid in session_ids:
        mgr.get_or_create(sid).add_user_message(f"hello from {sid}")
    return store, engine, mgr


def _restarted_client(engine):
    """Simulate a daemon restart: same durable store, EMPTY SessionManager."""
    return TestClient(create_app({}, session_mgr=SessionManager(), lcm_engine=engine))


def _row(rows, sid):
    return next((r for r in rows if r["session_id"] == sid), None)


# ---------------------------------------------------------------------------
# Restart survival
# ---------------------------------------------------------------------------


def test_sessions_survive_restart_with_true_watermarks(tmp_path):
    store, engine, _mgr = _seeded(tmp_path)
    rows = _restarted_client(engine).get("/api/sessions").json()
    for sid in ("telegram:123", "desktop:smoke"):
        row = _row(rows, sid)
        assert row is not None, f"{sid} vanished on restart"
        assert row["live"] is False  # durable-only: no in-memory working set
        assert row["watermark"] == store.max_rowid(sid)  # same cursor as /messages
        assert row["message_count"] == store.count_all(sid)
        assert row["last_active"] == store.max_timestamp(sid)
        assert row["created_at"] > 0
    assert _row(rows, "telegram:123")["gateway"] == "telegram"


def test_live_session_overlays_durable_row(tmp_path):
    store, engine, mgr = _seeded(tmp_path)
    client = TestClient(create_app({}, session_mgr=mgr, lcm_engine=engine))
    rows = client.get("/api/sessions").json()
    row = _row(rows, "telegram:123")
    assert row["live"] is True
    # Live semantics preserved: count is the in-memory list, not the row count.
    assert row["message_count"] == len(mgr.get("telegram:123").messages)
    assert row["watermark"] == store.max_rowid("telegram:123")
    # One row per session — the overlay replaces, never duplicates.
    assert sum(r["session_id"] == "telegram:123" for r in rows) == 1


def test_sessions_ordered_most_recent_first(tmp_path):
    store = LCMConversationStore(tmp_path / "lcm.db")
    now = time.time()
    store.add_message("a:old", MessagePart(role="user", content="x", session_id="a:old",
                                           timestamp=now - 100))
    store.add_message("b:new", MessagePart(role="user", content="y", session_id="b:new",
                                           timestamp=now))
    rows = _restarted_client(_engine(store)).get("/api/sessions").json()
    assert [r["session_id"] for r in rows] == ["b:new", "a:old"]


def test_no_lcm_wired_falls_back_to_in_memory_only(tmp_path):
    mgr = SessionManager()
    mgr.get_or_create("desktop:solo").add_user_message("hi")
    client = TestClient(create_app({}, session_mgr=mgr))
    rows = client.get("/api/sessions").json()
    assert [r["session_id"] for r in rows] == ["desktop:solo"]
    assert rows[0]["live"] is True
    assert rows[0]["watermark"] == 0  # pre-existing no-LCM fallback


# ---------------------------------------------------------------------------
# Forget → tombstone → revival
# ---------------------------------------------------------------------------


def test_forget_survives_restart(tmp_path):
    store, engine, mgr = _seeded(tmp_path)
    client = TestClient(create_app({}, session_mgr=mgr, lcm_engine=engine))
    assert client.delete("/api/sessions/telegram:123").json() == {"ok": True}
    # Gone from the live app...
    assert _row(client.get("/api/sessions").json(), "telegram:123") is None
    # ...and STILL gone after a restart (this is the new property).
    assert _row(_restarted_client(engine).get("/api/sessions").json(), "telegram:123") is None
    # The other session is untouched, and the forgotten rows are intact.
    assert _row(_restarted_client(engine).get("/api/sessions").json(), "desktop:smoke")
    assert store.count_all("telegram:123") > 0


def test_new_activity_revives_forgotten_session(tmp_path):
    store, engine, mgr = _seeded(tmp_path)
    client = TestClient(create_app({}, session_mgr=mgr, lcm_engine=engine))
    client.delete("/api/sessions/telegram:123")
    # The same stable chat id speaks again, strictly after the tombstone.
    # Real wall-clock ordering (tiny sleeps, no future-dating) — a
    # future-dated message would defeat the re-forget below by design.
    time.sleep(0.002)
    store.add_message("telegram:123", MessagePart(
        role="user", content="I'm back", session_id="telegram:123",
    ))
    row = _row(_restarted_client(engine).get("/api/sessions").json(), "telegram:123")
    assert row is not None, "newer activity must revive a tombstoned session"
    # Re-forgetting hides it again (the tombstone watermark refreshes).
    time.sleep(0.002)
    client2 = _restarted_client(engine)
    client2.delete("/api/sessions/telegram:123")
    assert _row(client2.get("/api/sessions").json(), "telegram:123") is None


# ---------------------------------------------------------------------------
# Store-level contract
# ---------------------------------------------------------------------------


def test_store_list_sessions_aggregates(tmp_path):
    store = LCMConversationStore(tmp_path / "lcm.db")
    now = time.time()
    for i in range(3):
        store.add_message("g:1", MessagePart(role="user", content=f"m{i}", session_id="g:1",
                                             timestamp=now + i))
    rows = store.list_sessions()
    assert len(rows) == 1
    row = rows[0]
    assert row["session_id"] == "g:1"
    assert row["message_count"] == 3
    assert row["first_timestamp"] == pytest.approx(now)
    assert row["last_timestamp"] == pytest.approx(now + 2)
    assert row["watermark"] == store.max_rowid("g:1")


def test_store_tombstone_reopen_persists(tmp_path):
    # The tombstone must live in the DB file, not in the store instance.
    db = tmp_path / "lcm.db"
    store = LCMConversationStore(db)
    store.add_message("g:1", MessagePart(role="user", content="x", session_id="g:1"))
    store.tombstone_session("g:1")
    store.close()
    reopened = LCMConversationStore(db)
    assert reopened.list_sessions() == []
