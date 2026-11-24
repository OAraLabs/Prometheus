"""Quiet pushes reach owner devices and the device whose session it is, nobody else (PRIVATE until released).

``turn_completed`` / ``task_completed`` / ``task_failed`` / ``coding_complete`` pushes carry a line of content (for a
turn, the first line of the reply) and the session id. They went to every registered device, so a scoped device
that is not on a live socket received the first line of replies from sessions it does not own. Now: owner devices,
plus the scoped device that owns the push's session; a push with no session goes to owners only; a store that cannot
answer sends none (fail closed). A device on a live socket still gets no quiet push, as before.
"""

from __future__ import annotations

import asyncio
import types

from prometheus.config.device_store import DeviceStore
from prometheus.push.apns import SendResult
from prometheus.push.dispatcher import PushDispatcher


class _Sender:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, **kwargs):
        self.sent.append(kwargs)
        return SendResult(outcome="ok", status=200)


def _world(tmp_path):
    store = DeviceStore(tmp_path / "devices.db")
    owner = store.mint_owner("owner phone", "ios", by="test")
    alice = store.mint("alice phone", "ios")
    bob = store.mint("bob phone", "ios")
    gone = store.mint("old phone", "ios")
    for d, tok in ((owner, "tok-owner"), (alice, "tok-alice"), (bob, "tok-bob"), (gone, "tok-gone")):
        assert store.set_push(d["id"], tok, "sandbox", "ai.test.beacon")
    store.claim_session("web:alice-1", alice["id"])
    store.claim_session("web:gone-1", gone["id"])
    store.revoke(gone["id"])
    return store, {"owner": owner, "alice": alice, "bob": bob}


def _push(store, kind, payload, live=()):
    sender = _Sender()
    bridge = types.SimpleNamespace(_ws_identity={i: types.SimpleNamespace(id=i) for i in live})
    asyncio.run(PushDispatcher(store, sender, bridge=bridge).on_signal(types.SimpleNamespace(kind=kind,
                                                                                             payload=payload)))
    return sorted(s["apns_token"] for s in sender.sent)


def test_a_turn_reaches_the_owner_and_the_device_whose_session_it_is(tmp_path):
    store, _ = _world(tmp_path)
    assert _push(store, "turn_completed", {"session_id": "web:alice-1", "summary": "the reply"}) == \
        ["tok-alice", "tok-owner"]


def test_another_devices_session_reaches_only_the_owner(tmp_path):
    store, _ = _world(tmp_path)
    assert _push(store, "turn_completed", {"session_id": "telegram:123", "summary": "a private reply"}) == \
        ["tok-owner"]


def test_a_revoked_owner_of_the_session_gets_nothing(tmp_path):
    store, _ = _world(tmp_path)
    assert _push(store, "turn_completed", {"session_id": "web:gone-1", "summary": "x"}) == ["tok-owner"]


def test_task_pushes_follow_the_same_rule_and_no_session_means_owners_only(tmp_path):
    store, _ = _world(tmp_path)
    for kind in ("task_completed", "task_failed", "coding_complete"):
        assert _push(store, kind, {"session_id": "web:alice-1", "task_id": "t1", "summary": "s"}) == \
            ["tok-alice", "tok-owner"], kind
        assert _push(store, kind, {"task_id": "t1", "summary": "s"}) == ["tok-owner"], kind


def test_a_device_on_a_live_socket_still_gets_no_quiet_push(tmp_path):
    store, ids = _world(tmp_path)
    assert _push(store, "turn_completed", {"session_id": "web:alice-1"}, live=[ids["alice"]["id"]]) == ["tok-owner"]


def test_a_store_that_cannot_answer_sends_no_quiet_push():
    class _Old:
        def push_targets(self):
            return [types.SimpleNamespace(id="d1", apns_token="tok", environment="sandbox", bundle_id="",
                                          push_failures=0)]

    assert _push(_Old(), "turn_completed", {"session_id": "web:x", "summary": "s"}) == []


def test_the_store_query(tmp_path):
    store, ids = _world(tmp_path)
    assert {t.apns_token for t in store.quiet_push_targets("web:alice-1")} == {"tok-owner", "tok-alice"}
    assert {t.apns_token for t in store.quiet_push_targets(None)} == {"tok-owner"}
