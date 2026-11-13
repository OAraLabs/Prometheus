"""An approval push goes to owner devices only.

An approval push carries the tool name and the first line of the request's description, and it used to go to
every registered device. A scoped device sees only its own sessions' requests (``GET /api/approvals``, the
``approval_pending`` frame), and it cannot approve anything but a desktop prompt, so a push to it about another
session's request told it something no other surface would, and offered it a button it could not use.

Held here:

  * the ordinary approval push reaches owner devices and nothing else: not a scoped device, not one marked for
    computer use, not a revoked owner;
  * a desktop prompt's push (no content at all: no tool, no description) reaches owner devices and the devices
    marked for computer use, because those are the ones that may answer it;
  * a store that cannot say which devices are owners sends no approval push at all (fail closed), and the other
    pushes are unchanged.
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


def _signal(kind: str, payload: dict):
    return types.SimpleNamespace(kind=kind, payload=payload, source="test")


def _world(tmp_path):
    store = DeviceStore(tmp_path / "devices.db")
    owner = store.mint_owner("owner phone", "ios", by="test")
    scoped = store.mint("scoped phone", "ios")
    marked = store.mint("marked tablet", "ios")
    gone = store.mint_owner("old owner phone", "ios", by="test")
    for d, tok in ((owner, "tok-owner"), (scoped, "tok-scoped"), (marked, "tok-marked"), (gone, "tok-gone")):
        assert store.set_push(d["id"], tok, "sandbox", "ai.test.beacon")
    assert store.set_computer(marked["id"], True, by="test")
    store.revoke(gone["id"])
    return store


def _approval(**extra) -> dict:
    return {"request_id": "r1", "tool_name": "bash", "description": "rm -rf build", "expires_at": 1.0, **extra}


def _tokens(sender: _Sender) -> list[str]:
    return sorted(s["apns_token"] for s in sender.sent)


def test_an_approval_push_reaches_owner_devices_only(tmp_path):
    sender = _Sender()
    asyncio.run(PushDispatcher(_world(tmp_path), sender, bridge=None).on_signal(
        _signal("approval_pending", _approval())))
    assert _tokens(sender) == ["tok-owner"]


def test_a_desktop_prompt_reaches_owners_and_devices_marked_for_computer_use(tmp_path):
    sender = _Sender()
    asyncio.run(PushDispatcher(_world(tmp_path), sender, bridge=None).on_signal(
        _signal("approval_pending", _approval(task_id="t1"))))
    assert _tokens(sender) == ["tok-marked", "tok-owner"]
    for s in sender.sent:
        assert "tool_name" not in s["payload"] and "rm -rf" not in str(s["payload"])


def test_the_quiet_pushes_are_unchanged(tmp_path):
    sender = _Sender()
    asyncio.run(PushDispatcher(_world(tmp_path), sender, bridge=None).on_signal(
        _signal("turn_completed", {"session_id": "web:s1"})))
    assert _tokens(sender) == ["tok-marked", "tok-owner", "tok-scoped"]


def test_a_store_that_cannot_name_its_owners_sends_no_approval_push():
    class _OldStore:
        def push_targets(self):
            return [types.SimpleNamespace(id="d1", apns_token="tok", environment="sandbox", bundle_id="",
                                          push_failures=0)]

    sender = _Sender()
    asyncio.run(PushDispatcher(_OldStore(), sender, bridge=None).on_signal(_signal("approval_pending", _approval())))
    assert sender.sent == []


def test_the_store_query_answers_owner_and_marked_devices(tmp_path):
    store = _world(tmp_path)
    names = {t.apns_token for t in store.approval_push_targets(desktop=False)}
    assert names == {"tok-owner"}
    names = {t.apns_token for t in store.approval_push_targets(desktop=True)}
    assert names == {"tok-owner", "tok-marked"}


def test_a_revoked_marked_device_gets_no_desktop_push(tmp_path):
    store = _world(tmp_path)
    marked = [d for d in store.list_devices() if d.name == "marked tablet"][0]
    store.revoke(marked.id)
    sender = _Sender()
    asyncio.run(PushDispatcher(store, sender, bridge=None).on_signal(
        _signal("approval_pending", _approval(task_id="t1"))))
    assert _tokens(sender) == ["tok-owner"]


def test_a_store_that_raises_sends_no_approval_push_and_the_bus_survives():
    class _Raising:
        def push_targets(self):
            return []

        def approval_push_targets(self, *, desktop):
            raise RuntimeError("database is locked")

    sender = _Sender()
    asyncio.run(PushDispatcher(_Raising(), sender, bridge=None).on_signal(_signal("approval_pending", _approval())))
    assert sender.sent == []
