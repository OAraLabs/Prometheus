"""POST /api/sessions/{id}/purge — irreversible content removal (#353).

The sibling of test_api_session_delete.py, and the contrast is the point: DELETE
tombstones and deliberately keeps the rows; purge removes them. Both exist
because "clear my list" and "that must not be on the disk" are different
requests, and only one of them can be answered with a tombstone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.engine.session import SessionManager  # noqa: E402
from prometheus.memory.lcm_conversation_store import LCMConversationStore  # noqa: E402
from prometheus.memory.lcm_types import MessagePart  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402

SECRET = "QQVXAPIPURGE9930"


def _engine(store: LCMConversationStore):
    class _Engine:
        conversation_store = store

        def ingest_sync(self, session_id, role, content, turn_index=0, content_json=None,
                        provenance="user", is_trusted=True):
            m = MessagePart(role=role, content=content, session_id=session_id,
                            turn_index=turn_index, provenance=provenance, is_trusted=is_trusted)
            store.add_message(session_id, m)
            return m.message_id

    return _Engine()


def _client(tmp_path):
    db = tmp_path / "lcm.db"
    store = LCMConversationStore(db)
    engine = _engine(store)
    mgr = SessionManager()
    mgr.lcm_engine = engine
    for sid, text in (("desktop:doomed", f"card {SECRET}"), ("telegram:123", "unrelated")):
        mgr.get_or_create(sid).add_user_message(text)
    return TestClient(create_app({}, session_mgr=mgr, lcm_engine=engine)), store, mgr, db


def _in_file(db: Path, needle: str) -> bool:
    return any(
        p.exists() and needle.encode() in p.read_bytes()
        for p in (db, Path(str(db) + "-wal"), Path(str(db) + "-shm"))
    )


def test_purge_removes_content_from_the_file(tmp_path):
    client, store, mgr, db = _client(tmp_path)
    assert _in_file(db, SECRET), "precondition: the secret is in the database file"

    r = client.post("/api/sessions/desktop:doomed/purge", json={"confirm": "desktop:doomed"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["purged"]["messages"] >= 1, body
    assert body["reversible"] is False

    assert not _in_file(db, SECRET), "the text must be gone from the file, not merely hidden"
    assert store.search(SECRET) == []
    assert "desktop:doomed" not in mgr._sessions
    assert {s["session_id"] for s in client.get("/api/sessions").json()} == {"telegram:123"}


def test_purge_requires_naming_the_session_twice(tmp_path):
    """No undo lives behind this route, so an accidental call must not work."""
    client, _store, _mgr, db = _client(tmp_path)

    for body in ({}, {"confirm": ""}, {"confirm": "telegram:123"}, {"confirm": "desktop:doome"}):
        r = client.post("/api/sessions/desktop:doomed/purge", json=body)
        assert r.status_code == 400, (body, r.text)
    # and nothing was destroyed by the refusals
    assert _in_file(db, SECRET), "a refused purge must leave the data alone"


def test_purge_reports_branches_it_did_not_follow(tmp_path):
    """A fork holds its own copies. Surviving is defensible; being silent is not."""
    client, store, _mgr, _db = _client(tmp_path)
    m = MessagePart(role="user", content="forkable")
    store.add_message("desktop:doomed", m)
    store.fork_session("desktop:doomed", m.row_id, "desktop:branch")

    body = client.post("/api/sessions/desktop:doomed/purge",
                       json={"confirm": "desktop:doomed"}).json()
    assert body["branches"] == ["desktop:branch"], body
    assert len(store.get_messages("desktop:branch")) > 0, "the branch's own copies survive"


def test_purge_without_a_store_does_not_claim_success(tmp_path):
    """The #349 failure mode: reporting ok for work that could not happen."""
    mgr = SessionManager()
    client = TestClient(create_app({}, session_mgr=mgr, lcm_engine=None))
    r = client.post("/api/sessions/x:y/purge", json={"confirm": "x:y"})
    assert r.status_code == 503, r.text
    assert "nothing was purged" in r.json()["error"]
