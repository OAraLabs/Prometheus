"""GRAFT-MOBILE-BRIDGE 2 — push registration, fan-out decisions, failure law.

Everything here runs WITHOUT cryptography/h2/network: the sender is a fake
that records what would have gone to Apple. The decision table under test is
the spec's, verbatim:

  * approval_pending pushes ALWAYS — a live socket does not suppress it, the
    phone is the point. Category APPROVAL, time-sensitive, mutable-content.
  * turn/task/coding completions push only to devices with NO live WS client,
    and liveness is PER DEVICE — a desktop being open must not silently
    suppress the phone's push (the exact bug the spec forbids shipping).
  * 410 clears the registration permanently and logs once; other failures
    count up and mute the device at the threshold until a success resets —
    no retries anywhere, so no storm is possible.
  * Live Activity updates throttle to one per session per 10s; chat_done
    sends one final end and drops the token row.
"""

from __future__ import annotations

import asyncio
import types

from prometheus.config.device_store import DeviceStore
from prometheus.push.dispatcher import (
    ACTIVITY_THROTTLE_SECONDS,
    FAILURE_MUTE_THRESHOLD,
    PushDispatcher,
)


class _FakeSender:
    """Records sends; scripted outcomes per apns token."""

    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.outcomes: dict[str, str] = {}
        self.config = types.SimpleNamespace(topic="com.oaralabs.beacon")

    async def send(self, **kw):
        from prometheus.push.apns import SendResult

        self.sent.append(kw)
        outcome = self.outcomes.get(kw["apns_token"], "ok")
        status = {"ok": 200, "unregistered": 410, "failed": 500}[outcome]
        return SendResult(outcome=outcome, status=status)


class _FakeBridge:
    def __init__(self, live_ids=()):
        self._ws_identity = {
            object(): types.SimpleNamespace(id=i) for i in live_ids
        }


def _signal(kind: str, payload: dict):
    return types.SimpleNamespace(kind=kind, payload=payload, source="test")


def _store(tmp_path) -> DeviceStore:
    return DeviceStore(tmp_path / "devices.db")


def _device(store, name="phone", push_token=None):
    minted = store.mint(name, "ios")
    if push_token:
        assert store.set_push(minted["id"], push_token, "sandbox", "com.oaralabs.beacon")
    return minted


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


def test_push_columns_migrate_onto_an_existing_db(tmp_path):
    # A Piece-1 era database (no push columns) gains them on reopen.
    import sqlite3

    path = tmp_path / "devices.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE api_devices (
          id TEXT PRIMARY KEY, name TEXT NOT NULL, platform TEXT NOT NULL,
          token_sha256 TEXT NOT NULL UNIQUE, created_at REAL NOT NULL,
          last_seen_at REAL, revoked_at REAL);
        INSERT INTO api_devices VALUES ('d1', 'old', 'ios', 'x', 1.0, NULL, NULL);
    """)
    conn.commit(); conn.close()

    store = DeviceStore(path)
    assert store.set_push("d1", "tok", "production", "com.oaralabs.beacon")
    assert [t.id for t in store.push_targets()] == ["d1"]


def test_revoked_device_cannot_register_push_and_drops_from_targets(tmp_path):
    store = _store(tmp_path)
    d = _device(store, push_token="tok-a")
    store.revoke(d["id"])
    assert store.push_targets() == [], "a tombstone is not a push target"
    assert not store.set_push(d["id"], "tok-b", "sandbox", "b"), \
        "a tombstone must not be re-armable"


# --------------------------------------------------------------------------- #
# Fan-out decisions
# --------------------------------------------------------------------------- #


def test_approval_pushes_even_with_a_live_socket(tmp_path):
    store = _store(tmp_path)
    d = _device(store, push_token="tok-a")
    sender = _FakeSender()
    dispatcher = PushDispatcher(store, sender, _FakeBridge(live_ids=[d["id"]]))

    _run(dispatcher.on_signal(_signal("approval_pending", {
        "request_id": "r1", "tool_name": "write_file",
        "description": "write /tmp/x\nsecond line", "expires_at": 999.0})))

    assert len(sender.sent) == 1
    aps = sender.sent[0]["payload"]["aps"]
    assert aps["category"] == "APPROVAL"
    assert aps["interruption-level"] == "time-sensitive"
    assert aps["mutable-content"] == 1
    assert aps["alert"]["body"] == "write_file — write /tmp/x", \
        "first line only — the body is a lock screen, not a transcript"
    assert sender.sent[0]["payload"]["request_id"] == "r1"


def test_turn_completed_skips_only_the_live_device(tmp_path):
    store = _store(tmp_path)
    live = _device(store, "watching", push_token="tok-live")
    away = _device(store, "pocketed", push_token="tok-away")
    sender = _FakeSender()
    dispatcher = PushDispatcher(store, sender, _FakeBridge(live_ids=[live["id"]]))

    _run(dispatcher.on_signal(_signal("turn_completed", {"session_id": "s1"})))

    tokens = [s["apns_token"] for s in sender.sent]
    assert tokens == ["tok-away"], \
        "per-device liveness: the watching device is skipped, the pocketed one is not"
    assert sender.sent[0]["payload"]["aps"]["category"] == "TURN"


def test_global_socket_does_not_suppress_the_phone(tmp_path):
    # THE bug the spec forbids: a desktop on the global token is not a device,
    # and its live socket must not swallow the phone's push.
    store = _store(tmp_path)
    _device(store, "phone", push_token="tok-phone")
    sender = _FakeSender()
    dispatcher = PushDispatcher(store, sender, _FakeBridge(live_ids=["global"]))

    _run(dispatcher.on_signal(_signal("task_completed", {"task_id": "t9"})))

    assert [s["apns_token"] for s in sender.sent] == ["tok-phone"]
    assert sender.sent[0]["payload"]["aps"]["category"] == "TASK"


def test_unrelated_signals_push_nothing(tmp_path):
    store = _store(tmp_path)
    _device(store, push_token="tok-a")
    sender = _FakeSender()
    dispatcher = PushDispatcher(store, sender, _FakeBridge())
    _run(dispatcher.on_signal(_signal("skill_created", {"name": "x"})))
    assert sender.sent == []


# --------------------------------------------------------------------------- #
# Failure law
# --------------------------------------------------------------------------- #


def test_410_clears_the_registration_permanently(tmp_path):
    store = _store(tmp_path)
    d = _device(store, push_token="tok-dead")
    sender = _FakeSender()
    sender.outcomes["tok-dead"] = "unregistered"
    dispatcher = PushDispatcher(store, sender, _FakeBridge())

    _run(dispatcher.on_signal(_signal("turn_completed", {"session_id": "s"})))
    assert store.push_targets() == [], "410 cleared the registration"

    sender.sent.clear()
    _run(dispatcher.on_signal(_signal("turn_completed", {"session_id": "s"})))
    assert sender.sent == [], "and it stays cleared — no resurrection"
    assert store.set_push(d["id"], "tok-new", "sandbox", "b"), \
        "but a fresh registration from the device is welcome"


def test_failures_mute_at_threshold_and_success_resets(tmp_path):
    store = _store(tmp_path)
    d = _device(store, push_token="tok-flaky")
    sender = _FakeSender()
    sender.outcomes["tok-flaky"] = "failed"
    dispatcher = PushDispatcher(store, sender, _FakeBridge())

    for _ in range(FAILURE_MUTE_THRESHOLD):
        _run(dispatcher.on_signal(_signal("turn_completed", {"session_id": "s"})))
    assert len(sender.sent) == FAILURE_MUTE_THRESHOLD

    _run(dispatcher.on_signal(_signal("turn_completed", {"session_id": "s"})))
    assert len(sender.sent) == FAILURE_MUTE_THRESHOLD, "muted — no further attempts"

    # Recovery: the count resets and the next attempt goes out.
    store.reset_push_failures(d["id"])
    sender.outcomes["tok-flaky"] = "ok"
    _run(dispatcher.on_signal(_signal("turn_completed", {"session_id": "s"})))
    assert len(sender.sent) == FAILURE_MUTE_THRESHOLD + 1


# --------------------------------------------------------------------------- #
# Live Activity
# --------------------------------------------------------------------------- #


def test_activity_updates_throttle_and_end_drops_the_token(tmp_path):
    store = _store(tmp_path)
    d = _device(store, push_token="tok-a")
    store.set_activity_token(d["id"], "s1", "act-tok-1")
    sender = _FakeSender()
    clock = {"t": 1000.0}
    dispatcher = PushDispatcher(store, sender, _FakeBridge(),
                                now=lambda: clock["t"])

    _run(dispatcher.on_agent_progress("s1", {"phase": "tool", "round": 1,
                                             "elapsed_s": 3.0}))
    clock["t"] += 3
    _run(dispatcher.on_agent_progress("s1", {"phase": "tool", "round": 1,
                                             "elapsed_s": 6.0}))
    assert len(sender.sent) == 1, "3s later is inside the 10s throttle"
    assert sender.sent[0]["push_type"] == "liveactivity"
    assert sender.sent[0]["priority"] == 5
    assert sender.sent[0]["apns_token"] == "act-tok-1", \
        "liveactivity goes to the ACTIVITY token, not the alert token"
    assert sender.sent[0]["topic"].endswith(".push-type.liveactivity")

    clock["t"] += ACTIVITY_THROTTLE_SECONDS
    _run(dispatcher.on_agent_progress("s1", {"phase": "generating", "round": 2,
                                             "elapsed_s": 16.0}))
    assert len(sender.sent) == 2

    _run(dispatcher.on_chat_done("s1", {"session_id": "s1"}))
    assert sender.sent[-1]["payload"]["aps"]["event"] == "end"
    assert store.activity_targets("s1") == [], "the token row drops with the end"

    sender.sent.clear()
    _run(dispatcher.on_agent_progress("s1", {"phase": "tool"}))
    assert sender.sent == [], "no activity token, no push"
