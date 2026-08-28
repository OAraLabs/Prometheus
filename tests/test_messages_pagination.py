"""GRAFT-MOBILE-BRIDGE 5 — ?limit= / ?before= on the messages route.

Acceptance, from the spec:

  1. ?limit=50 on a big session returns the newest 50 ASCENDING, has_more true.
  2. ?before= walks backwards to the start; has_more goes false exactly once.
  3. No params → today's response plus has_more (and the old internal
     10,000-row cap now SAYS so instead of truncating silently).
  4. ?since= and ?before= together → 400 — different questions.

Plus: ?since= semantics untouched, limit clamps rather than errors, garbage
cursors stay loud 400s.
"""

from __future__ import annotations

import pytest

from prometheus.memory.lcm_conversation_store import LCMConversationStore
from prometheus.memory.lcm_types import MessagePart


def _seed(store: LCMConversationStore, sid: str, n: int) -> None:
    for i in range(n):
        store.insert_message(MessagePart(
            role="user" if i % 2 == 0 else "assistant",
            content=f"m{i}", session_id=sid))


@pytest.fixture()
def client_and_store(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from prometheus.web.server import create_app

    store = LCMConversationStore(tmp_path / "lcm.db")

    class _Lcm:
        conversation_store = store

    app = create_app({})
    app.state.lcm_engine = _Lcm()
    return TestClient(app), store


def _ids(body: dict) -> list[int]:
    return [m["message_id"] for m in body["messages"]]


def test_limit_returns_the_newest_page_ascending(client_and_store):
    client, store = client_and_store
    _seed(store, "s1", 120)
    body = client.get("/api/sessions/s1/messages?limit=50").json()
    ids = _ids(body)
    assert len(ids) == 50
    assert ids == sorted(ids), "the page stays ascending like every response"
    assert ids[-1] == body["watermark"], "anchored at the newest row"
    assert body["has_more"] is True


def test_before_walks_to_the_start_and_has_more_flips_once(client_and_store):
    client, store = client_and_store
    _seed(store, "s1", 120)
    seen: list[int] = []
    cursor: int | None = None
    flips = 0
    while True:
        path = f"/api/sessions/s1/messages?limit=50" + \
               (f"&before={cursor}" if cursor is not None else "")
        body = client.get(path).json()
        ids = _ids(body)
        seen = ids + seen
        if not body["has_more"]:
            flips += 1
            break
        cursor = ids[0]
    assert flips == 1
    assert len(seen) == 120
    assert seen == sorted(seen), "pages splice into one ascending history"
    assert len(set(seen)) == 120, "no row twice, no row missed"


def test_no_params_matches_todays_shape_plus_has_more(client_and_store):
    client, store = client_and_store
    _seed(store, "s1", 5)
    body = client.get("/api/sessions/s1/messages").json()
    assert len(body["messages"]) == 5
    assert body["has_more"] is False
    assert set(body["messages"][0]) == {
        "message_id", "ordinal", "session_id", "role", "content", "content_json",
        "provenance", "is_trusted", "timestamp"}, "row shape unchanged"


def test_since_and_before_together_is_400(client_and_store):
    client, store = client_and_store
    _seed(store, "s1", 3)
    r = client.get("/api/sessions/s1/messages?since=1&before=3")
    assert r.status_code == 400


def test_since_semantics_unchanged_and_limit_bounds_it(client_and_store):
    client, store = client_and_store
    _seed(store, "s1", 10)
    all_ids = _ids(client.get("/api/sessions/s1/messages").json())
    cursor = all_ids[4]
    body = client.get(f"/api/sessions/s1/messages?since={cursor}").json()
    assert _ids(body) == all_ids[5:], "?since= alone: everything after, as before"
    assert body["has_more"] is False
    bounded = client.get(f"/api/sessions/s1/messages?since={cursor}&limit=2").json()
    assert _ids(bounded) == all_ids[5:7]
    assert bounded["has_more"] is True, "the forward read now admits a remainder"


def test_limit_clamps_and_garbage_cursors_are_400(client_and_store):
    client, store = client_and_store
    _seed(store, "s1", 3)
    assert len(_ids(client.get("/api/sessions/s1/messages?limit=99999").json())) == 3
    body = client.get("/api/sessions/s1/messages?limit=0").json()
    assert len(body["messages"]) == 1, "limit clamps to 1..500, per the spec"
    assert client.get("/api/sessions/s1/messages?limit=abc").status_code == 400
    assert client.get("/api/sessions/s1/messages?before=abc").status_code == 400
    assert client.get("/api/sessions/s1/messages?since=abc").status_code == 400


def test_pages_are_scoped_to_the_session(client_and_store):
    client, store = client_and_store
    _seed(store, "s1", 4)
    _seed(store, "s2", 4)
    body = client.get("/api/sessions/s1/messages?limit=10").json()
    assert {m["session_id"] for m in body["messages"]} == {"s1"}
    assert len(body["messages"]) == 4 and body["has_more"] is False
