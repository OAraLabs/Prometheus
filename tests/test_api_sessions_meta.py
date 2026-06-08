"""GET /api/sessions enrichment — watermark + gateway (branch feat/api-sessions-meta).

Side-effect asserting (not structural): the endpoint's watermark IS the session's true durable
max rowid and ADVANCES when a message is appended; gateway is the canonical id prefix.
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
    sessions = {}
    for sid in session_ids:
        s = mgr.get_or_create(sid)
        s.add_user_message(f"hello from {sid}")  # persists to LCM → a rowid is assigned
        sessions[sid] = s
    client = TestClient(create_app({}, session_mgr=mgr, lcm_engine=engine))
    return client, store, sessions


def _row(rows, sid):
    return next(r for r in rows if r["session_id"] == sid)


def test_watermark_equals_true_max_rowid(tmp_path):
    client, store, _ = _client(tmp_path, ["telegram:123", "desktop:smoke"])
    rows = client.get("/api/sessions").json()
    for sid in ("telegram:123", "desktop:smoke"):
        row = _row(rows, sid)
        # The endpoint's watermark IS the durable max rowid for that session (same cursor the
        # messages route returns) — not a parallel counter.
        assert row["watermark"] == store.max_rowid(sid)
        assert row["watermark"] > 0


def test_appending_a_message_advances_the_watermark(tmp_path):
    client, store, sessions = _client(tmp_path, ["telegram:123"])
    before = _row(client.get("/api/sessions").json(), "telegram:123")["watermark"]
    sessions["telegram:123"].add_user_message("another")  # append → new, higher rowid
    after = _row(client.get("/api/sessions").json(), "telegram:123")["watermark"]
    assert after > before
    assert after == store.max_rowid("telegram:123")


def test_gateway_is_canonical_from_id(tmp_path):
    client, _, _ = _client(tmp_path, ["telegram:8139235390", "desktop:smoke", "slack:C0123"])
    rows = client.get("/api/sessions").json()
    assert _row(rows, "telegram:8139235390")["gateway"] == "telegram"
    assert _row(rows, "desktop:smoke")["gateway"] == "desktop"
    assert _row(rows, "slack:C0123")["gateway"] == "slack"


def test_shape_is_consistent_and_keeps_count_and_created_at(tmp_path):
    client, _, _ = _client(tmp_path, ["telegram:1"])
    row = _row(client.get("/api/sessions").json(), "telegram:1")
    assert set(row.keys()) == {"session_id", "gateway", "created_at", "last_active", "message_count", "watermark"}
    assert isinstance(row["watermark"], int)
    assert isinstance(row["last_active"], (int, float))
    assert row["message_count"] == 1  # kept
    assert isinstance(row["created_at"], (int, float))  # kept
