"""DELETE /api/sessions/{id} — "forget session" removal (branch feat/session-delete).

The route must REMOVE the in-memory session entry, not merely empty it: after a
DELETE, GET /api/sessions no longer lists the id and the entry is gone from
session_mgr._sessions. Durable LCM rows are append-only and intentionally left
intact — only the live in-memory handle is forgotten. This is the daemon half
that makes Beacon Desktop's "Forget session" stick (without it the next
GET /api/sessions poll would re-surface the session the desktop just dropped).
"""

from __future__ import annotations

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


def _client(tmp_path, session_ids):
    store = LCMConversationStore(tmp_path / "lcm.db")
    engine = _engine(store)
    mgr = SessionManager()
    mgr.lcm_engine = engine  # so get_or_create wires LCM persistence into each ChatSession
    for sid in session_ids:
        s = mgr.get_or_create(sid)
        s.add_user_message(f"hello from {sid}")  # persists to LCM → a rowid is assigned
    client = TestClient(create_app({}, session_mgr=mgr, lcm_engine=engine))
    return client, store, mgr


def _ids(rows):
    return {r["session_id"] for r in rows}


def test_delete_removes_session_from_listing_and_manager(tmp_path):
    client, _store, mgr = _client(tmp_path, ["desktop:s4-junk", "telegram:123"])

    # Precondition: both sessions are listed and both live in the manager.
    assert _ids(client.get("/api/sessions").json()) == {"desktop:s4-junk", "telegram:123"}
    assert "desktop:s4-junk" in mgr._sessions

    resp = client.delete("/api/sessions/desktop:s4-junk")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}

    # The forgotten session is gone from BOTH the listing and the in-memory dict...
    assert _ids(client.get("/api/sessions").json()) == {"telegram:123"}
    assert "desktop:s4-junk" not in mgr._sessions
    # ...and the survivor is untouched.
    assert "telegram:123" in mgr._sessions


def test_delete_leaves_durable_lcm_rows_intact(tmp_path):
    """Forget is in-memory only — the append-only conversation store keeps its rows."""
    client, store, _mgr = _client(tmp_path, ["desktop:s4-junk"])
    before = store.max_rowid("desktop:s4-junk")
    assert before > 0  # a row was persisted on add_user_message

    client.delete("/api/sessions/desktop:s4-junk")

    # LCM rows are deliberately NOT deleted by a forget — durable history survives.
    assert store.max_rowid("desktop:s4-junk") == before


def test_delete_unknown_session_is_safe(tmp_path):
    client, _store, mgr = _client(tmp_path, ["telegram:123"])
    resp = client.delete("/api/sessions/desktop:does-not-exist")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    # The real session is unaffected by a no-op forget.
    assert "telegram:123" in mgr._sessions
