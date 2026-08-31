"""Session pins, daemon-side.

WHY DAEMON-SIDE. Beacon Desktop has pinned sessions for a while, but it keeps
the pin in its own local settings (``pinned:<id>`` in the renderer's store), so
the pin only ever existed on the machine that set it. Beacon iOS shipped
without pinning at all, and adding a second local implementation would have
produced two surfaces that disagree about the same conversation. A pin is a
property OF the conversation, exactly like its title — so it lives with the
conversation, in the same durable store, and every client reads the same answer.

These pin the surface end to end:

  * store: pin/unpin round-trip, ``list_sessions`` carries it, unpinning
    deletes rather than storing false;
  * REST: ``GET /api/sessions`` reports ``pinned`` as a bool for both durable
    and live sessions, ``PUT /api/sessions/{id}/pin`` sets and clears it and
    rejects a non-bool body;
  * migration: an lcm.db written before this change gains the table on open.
"""

from __future__ import annotations

import sqlite3

import pytest

from prometheus.memory.lcm_conversation_store import LCMConversationStore
from prometheus.memory.lcm_types import MessagePart


def _store(tmp_path) -> LCMConversationStore:
    return LCMConversationStore(tmp_path / "lcm.db")


def _seed(store: LCMConversationStore, sid: str) -> None:
    store.insert_message(MessagePart(role="user", content="hello", session_id=sid))


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


def test_pin_round_trip(tmp_path):
    store = _store(tmp_path)
    assert store.is_session_pinned("s1") is False
    store.set_session_pinned("s1", True)
    assert store.is_session_pinned("s1") is True
    store.set_session_pinned("s1", False)
    assert store.is_session_pinned("s1") is False


def test_pinning_twice_is_idempotent(tmp_path):
    """A double-tap, or two clients pinning at once, must not error or duplicate
    the row — session_id is the primary key and the write is INSERT OR REPLACE."""
    store = _store(tmp_path)
    store.set_session_pinned("s1", True)
    store.set_session_pinned("s1", True)
    assert store.is_session_pinned("s1") is True
    count = store._conn.execute(
        "SELECT COUNT(*) FROM session_pins WHERE session_id = 's1'"
    ).fetchone()[0]
    assert count == 1


def test_unpinning_deletes_rather_than_storing_false(tmp_path):
    """Presence IS the pin. If an unpin left a row behind, every reader would
    have to know which falsy value meant "not pinned"."""
    store = _store(tmp_path)
    store.set_session_pinned("s1", True)
    store.set_session_pinned("s1", False)
    rows = store._conn.execute(
        "SELECT * FROM session_pins WHERE session_id = 's1'"
    ).fetchall()
    assert rows == []


def test_unpinning_something_never_pinned_is_a_no_op(tmp_path):
    store = _store(tmp_path)
    store.set_session_pinned("never", False)  # must not raise
    assert store.is_session_pinned("never") is False


def test_list_sessions_carries_the_pin(tmp_path):
    store = _store(tmp_path)
    _seed(store, "telegram:1")
    _seed(store, "desktop:2")
    store.set_session_pinned("telegram:1", True)

    rows = {r["session_id"]: r for r in store.list_sessions()}
    assert rows["telegram:1"]["pinned_at"] is not None
    assert rows["desktop:2"]["pinned_at"] is None


def test_pin_survives_reopening_the_database(tmp_path):
    """The point of daemon-side: a restart must not lose the pin."""
    store = _store(tmp_path)
    _seed(store, "desktop:d")
    store.set_session_pinned("desktop:d", True)
    del store

    reopened = LCMConversationStore(tmp_path / "lcm.db")
    assert reopened.is_session_pinned("desktop:d") is True


def test_an_older_database_gains_the_table_on_open(tmp_path):
    """MIGRATION. Every deployed lcm.db predates this table. The schema block
    runs on every open with CREATE TABLE IF NOT EXISTS, so opening an old file
    is the whole migration — but only if nothing else in the schema chokes on
    a pre-existing database, which is what this actually checks."""
    db = tmp_path / "lcm.db"
    store = _store(tmp_path)
    _seed(store, "desktop:old")
    del store

    # Simulate the pre-change file: drop the table the way an older build's
    # schema would simply never have created it.
    raw = sqlite3.connect(db)
    raw.execute("DROP TABLE session_pins")
    raw.commit()
    raw.close()

    reopened = LCMConversationStore(db)
    assert reopened.is_session_pinned("desktop:old") is False
    reopened.set_session_pinned("desktop:old", True)
    assert reopened.is_session_pinned("desktop:old") is True


def test_a_pin_does_not_touch_the_message_rows(tmp_path):
    """Pins live outside the append-only log, like titles and tombstones — a
    pin or unpin must never rewrite history."""
    store = _store(tmp_path)
    _seed(store, "desktop:h")
    before = store._conn.execute("SELECT COUNT(*) FROM lcm_messages").fetchone()[0]
    store.set_session_pinned("desktop:h", True)
    store.set_session_pinned("desktop:h", False)
    after = store._conn.execute("SELECT COUNT(*) FROM lcm_messages").fetchone()[0]
    assert before == after == 1


def test_pins_and_titles_are_independent(tmp_path):
    """They share a shape, not a row. Clearing one must not clear the other."""
    store = _store(tmp_path)
    _seed(store, "desktop:i")
    store.set_session_title("desktop:i", "Ship it")
    store.set_session_pinned("desktop:i", True)

    store.set_session_title("desktop:i", "")
    assert store.is_session_pinned("desktop:i") is True

    store.set_session_pinned("desktop:i", False)
    store.set_session_title("desktop:i", "Ship it")
    assert store.get_session_title("desktop:i") == "Ship it"
    assert store.is_session_pinned("desktop:i") is False


# --------------------------------------------------------------------------- #
# REST
# --------------------------------------------------------------------------- #


@pytest.fixture()
def client_and_store(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from prometheus.web.server import create_app

    store = _store(tmp_path)

    class _Lcm:
        conversation_store = store

    app = create_app({})
    app.state.lcm_engine = _Lcm()
    return TestClient(app), store


def test_get_sessions_exposes_pinned_as_a_bool(client_and_store):
    """A client must never have to read presence out of a timestamp."""
    client, store = client_and_store
    _seed(store, "desktop:p")
    _seed(store, "desktop:q")
    store.set_session_pinned("desktop:p", True)

    rows = {r["session_id"]: r for r in client.get("/api/sessions").json()}
    assert rows["desktop:p"]["pinned"] is True
    assert rows["desktop:q"]["pinned"] is False


def test_put_pin_sets_and_clears(client_and_store):
    client, store = client_and_store
    _seed(store, "desktop:r")

    resp = client.put("/api/sessions/desktop:r/pin", json={"pinned": True})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "session_id": "desktop:r", "pinned": True}
    assert store.is_session_pinned("desktop:r") is True

    resp = client.put("/api/sessions/desktop:r/pin", json={"pinned": False})
    assert resp.status_code == 200
    assert resp.json()["pinned"] is False
    assert store.is_session_pinned("desktop:r") is False


def test_put_pin_rejects_a_non_bool(client_and_store):
    """"true" and 1 are the shapes a sloppy client sends. Neither is a bool, and
    guessing which truthiness was meant is how a pin silently inverts."""
    client, _ = client_and_store
    for bad in ("true", 1, None, [], {}):
        resp = client.put("/api/sessions/x/pin", json={"pinned": bad})
        assert resp.status_code == 400, f"{bad!r} was accepted"


def test_put_pin_rejects_an_invalid_body(client_and_store):
    client, _ = client_and_store
    resp = client.put(
        "/api/sessions/x/pin",
        content=b"not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 400


def test_a_live_session_still_reports_its_pin(tmp_path):
    """THE TRAP. GET /api/sessions builds durable rows first, then REPLACES any
    entry that is also live. The live branch rebuilds the dict from scratch, so
    a field it forgets is a field that vanishes exactly when the session is in
    use — which is when a pin matters most."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from prometheus.web.server import create_app

    store = _store(tmp_path)
    _seed(store, "desktop:live")
    _seed(store, "desktop:cold")
    store.set_session_pinned("desktop:live", True)
    store.set_session_pinned("desktop:cold", True)

    class _Session:
        created_at = 0.0
        messages: list = []

    class _Mgr:
        _sessions = {"desktop:live": _Session()}

    class _Lcm:
        conversation_store = store

    app = create_app({}, session_mgr=_Mgr())
    app.state.lcm_engine = _Lcm()
    rows = {r["session_id"]: r for r in TestClient(app).get("/api/sessions").json()}

    assert rows["desktop:live"]["live"] is True
    assert rows["desktop:live"]["pinned"] is True
    # The durable branch, unshadowed, for contrast.
    assert rows["desktop:cold"]["live"] is False
    assert rows["desktop:cold"]["pinned"] is True
