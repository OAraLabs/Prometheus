"""Durable message identity (branch fix/durable-message-id).

The wire ``message_id`` is now the LCM rowid: durable, unique, monotonic, restart-stable —
unlike the old ``msg-{turn_index}`` (the in-memory list position, which resets on restart/
trim and repeats). These tests prove the exact properties Step 3's live smoke showed broken.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.memory.lcm_conversation_store import LCMConversationStore  # noqa: E402
from prometheus.memory.lcm_types import MessagePart  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402


def _client(store):
    class _LCM:
        conversation_store = store

    return TestClient(create_app({}, lcm_engine=_LCM()))


def test_duplicate_turn_index_get_distinct_message_ids(tmp_path):
    """The Step-3 bug exactly: two rows with the SAME turn_index (e.g. a second msg-0 after a
    restart). They must now present DISTINCT, independently-retrievable canonical ids."""
    store = LCMConversationStore(tmp_path / "lcm.db")
    a = MessagePart(role="user", content="before restart", session_id="s", turn_index=0, timestamp=100.0)
    b = MessagePart(role="user", content="after restart", session_id="s", turn_index=0, timestamp=200.0)
    store.insert_message(a)
    store.insert_message(b)
    assert a.turn_index == b.turn_index == 0  # same ordinal (the old collision)
    assert a.row_id != b.row_id  # distinct durable ids

    body = _client(store).get("/api/sessions/s/messages").json()
    ids = [m["message_id"] for m in body["messages"]]
    ordinals = [m["ordinal"] for m in body["messages"]]
    assert ordinals == [0, 0]  # ordinal still duplicates (display only — that's fine)
    assert len(set(ids)) == 2  # but message_ids are distinct
    assert ids == [a.row_id, b.row_id]
    # both rows independently retrievable via the cursor
    after_a = _client(store).get(f"/api/sessions/s/messages?since={a.row_id}").json()
    assert [m["content"] for m in after_a["messages"]] == ["after restart"]


def test_message_id_is_restart_stable(tmp_path):
    """A message's canonical id is identical before and after a daemon restart (store reopen)
    and never resets to 0 — the case Step 3's smoke missed."""
    db = tmp_path / "lcm.db"
    store1 = LCMConversationStore(db)
    m = MessagePart(role="user", content="persist me", session_id="s", turn_index=0, timestamp=100.0)
    store1.insert_message(m)
    before = m.row_id
    body_before = _client(store1).get("/api/sessions/s/messages").json()
    store1.close()

    # Simulate a daemon restart: a fresh store object on the same db file.
    store2 = LCMConversationStore(db)
    body_after = _client(store2).get("/api/sessions/s/messages").json()
    after = body_after["messages"][0]["message_id"]

    assert before == after  # SAME id across restart
    assert after != 0
    assert body_before["messages"][0]["message_id"] == after
    assert body_after["watermark"] == before


def test_shared_timestamp_turn_is_strictly_orderable_by_message_id(tmp_path):
    """A multi-message turn shares one timestamp (the Step-3 batch-ingest wrinkle); the rowid
    cursor still strictly orders and pages it."""
    store = LCMConversationStore(tmp_path / "lcm.db")
    rowids = []
    for i in range(4):
        m = MessagePart(role="assistant", content=f"chunk{i}", session_id="s", turn_index=i, timestamp=500.0)
        store.insert_message(m)
        rowids.append(m.row_id)
    assert rowids == sorted(rowids) and len(set(rowids)) == 4  # monotonic + unique

    client = _client(store)
    body = client.get("/api/sessions/s/messages").json()
    assert [m["timestamp"] for m in body["messages"]] == [500.0] * 4  # all equal
    assert [m["message_id"] for m in body["messages"]] == rowids  # strictly ordered anyway

    # paging by rowid works even though timestamps are identical
    mid = client.get(f"/api/sessions/s/messages?since={rowids[1]}").json()
    assert [m["content"] for m in mid["messages"]] == ["chunk2", "chunk3"]


def test_max_rowid_empty_session_is_zero(tmp_path):
    store = LCMConversationStore(tmp_path / "lcm.db")
    assert store.max_rowid("nonexistent") == 0
    body = _client(store).get("/api/sessions/nonexistent/messages").json()
    assert body == {"messages": [], "watermark": 0}
