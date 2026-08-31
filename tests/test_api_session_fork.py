"""Edit/branch as forked sessions (B4, design in Prometheus#340).

LCM is linear and every consumer depends on that: `?since=` pagination, context assembly, FTS,
and three gateways. So a branch is NOT a new shape — it is another session holding copies of
history up to a point. These tests pin the properties that decision buys, because if any of them
breaks, the reason to prefer forked sessions over sibling turns is gone.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.memory.lcm_conversation_store import LCMConversationStore  # noqa: E402
from prometheus.memory.lcm_types import MessagePart  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402


class _Engine:
    def __init__(self, store):
        self.conversation_store = store


@pytest.fixture
def ctx(tmp_path):
    store = LCMConversationStore(tmp_path / "lcm.db")
    rows = []
    for i, (role, text) in enumerate(
        [("user", "first"), ("assistant", "second"), ("user", "third"), ("assistant", "fourth")]
    ):
        m = MessagePart(role=role, content=text, session_id="beacon:orig", turn_index=i)
        store.insert_message(m)
        rows.append(m.message_id)
    ids = [m.row_id for m in store.messages_after_id(0, session_id="beacon:orig")]
    return TestClient(create_app({}, lcm_engine=_Engine(store))), store, ids


def test_a_fork_copies_history_up_to_the_point_and_no_further(ctx):
    c, store, ids = ctx
    body = c.post("/api/sessions/beacon:orig/fork", json={"at_rowid": ids[1]}).json()
    assert body["copied"] == 2, "two messages at or before the fork point"
    forked = store.messages_after_id(0, session_id=body["session_id"])
    assert [m.content for m in forked] == ["first", "second"]


def test_the_original_is_untouched(ctx):
    """The property that makes every other consumer safe."""
    c, store, ids = ctx
    before = store.messages_after_id(0, session_id="beacon:orig")
    c.post("/api/sessions/beacon:orig/fork", json={"at_rowid": ids[1]})
    after = store.messages_after_id(0, session_id="beacon:orig")
    assert [m.row_id for m in before] == [m.row_id for m in after], "origin rowids moved"
    assert [m.content for m in before] == [m.content for m in after]


def test_a_since_cursor_on_the_ORIGIN_never_sees_the_fork(ctx):
    """`?since=` is a per-session query; this is why forking cannot disturb a polling client."""
    c, store, ids = ctx
    c.post("/api/sessions/beacon:orig/fork", json={"at_rowid": ids[3]})
    assert store.messages_after_id(ids[3], session_id="beacon:orig") == [], "the fork leaked into the origin"


def test_the_copies_get_fresh_ids_so_summaries_stay_bound_to_the_original(ctx):
    c, store, ids = ctx
    body = c.post("/api/sessions/beacon:orig/fork", json={"at_rowid": ids[1]}).json()
    origin_ids = {m.message_id for m in store.messages_after_id(0, session_id="beacon:orig")}
    fork_ids = {m.message_id for m in store.messages_after_id(0, session_id=body["session_id"])}
    assert origin_ids.isdisjoint(fork_ids), (
        "a copy reused a message uuid — the original's summaries would now point into the fork"
    )


def test_the_fork_sorts_after_the_original(ctx):
    """Fresh rowids are higher, which is what keeps ordering coherent without a migration."""
    c, store, ids = ctx
    body = c.post("/api/sessions/beacon:orig/fork", json={"at_rowid": ids[1]}).json()
    fork_rows = [m.row_id for m in store.messages_after_id(0, session_id=body["session_id"])]
    assert min(fork_rows) > max(ids), "copies must not be interleaved with the origin's rowids"


def test_provenance_is_recorded_in_both_directions(ctx):
    c, _, ids = ctx
    new_id = c.post("/api/sessions/beacon:orig/fork", json={"at_rowid": ids[2]}).json()["session_id"]
    child = c.get(f"/api/sessions/{new_id}/fork").json()
    assert child["origin"]["origin_session"] == "beacon:orig"
    assert child["origin"]["origin_rowid"] == ids[2]
    parent = c.get("/api/sessions/beacon:orig/fork").json()
    assert [f["session_id"] for f in parent["forks"]] == [new_id], "the origin lists its forks"


def test_an_unforked_session_says_so_rather_than_erroring(ctx):
    c, _, _ = ctx
    body = c.get("/api/sessions/beacon:orig/fork").json()
    assert body["origin"] is None and body["forks"] == []


def test_the_new_id_inherits_the_gateway_prefix(ctx):
    c, _, ids = ctx
    new_id = c.post("/api/sessions/beacon:orig/fork", json={"at_rowid": ids[0]}).json()["session_id"]
    assert new_id.startswith("beacon:") and new_id != "beacon:orig"


def test_a_caller_may_name_the_fork(ctx):
    c, _, ids = ctx
    body = c.post("/api/sessions/beacon:orig/fork",
                  json={"at_rowid": ids[0], "session_id": "beacon:mine"}).json()
    assert body["session_id"] == "beacon:mine"


def test_forking_onto_the_origin_is_refused(ctx):
    """It would interleave copies with the original's own history."""
    c, _, ids = ctx
    r = c.post("/api/sessions/beacon:orig/fork",
               json={"at_rowid": ids[0], "session_id": "beacon:orig"})
    assert r.status_code == 400


def test_a_point_with_no_history_is_404_not_an_empty_session(ctx):
    """An empty fork would look like success and behave like a lost conversation."""
    c, store, _ = ctx
    r = c.post("/api/sessions/beacon:orig/fork", json={"at_rowid": 0})
    assert r.status_code == 404
    r2 = c.post("/api/sessions/beacon:nope/fork", json={"at_rowid": 999})
    assert r2.status_code == 404


def test_a_bad_rowid_is_a_400(ctx):
    c, _, _ = ctx
    assert c.post("/api/sessions/beacon:orig/fork", json={"at_rowid": "soon"}).status_code == 400
    assert c.post("/api/sessions/beacon:orig/fork", json={}).status_code == 400


def test_forking_a_fork_records_its_own_immediate_origin(ctx):
    """A chain keeps one hop per link — the child names its parent, not the root."""
    c, _, ids = ctx
    first = c.post("/api/sessions/beacon:orig/fork", json={"at_rowid": ids[1]}).json()
    rows = c.get(f"/api/sessions/{first['session_id']}/fork").json()
    assert rows["origin"]["origin_session"] == "beacon:orig"
    second = c.post(f"/api/sessions/{first['session_id']}/fork", json={"at_rowid": 999999}).json()
    assert c.get(f"/api/sessions/{second['session_id']}/fork").json()["origin"]["origin_session"] == first["session_id"]
