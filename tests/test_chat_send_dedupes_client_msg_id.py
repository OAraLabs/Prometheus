"""A retried ``POST /api/chat/send`` with the same ``client_msg_id`` does not post the message twice.

A phone that sent a message and did not hear back (the radio dropped between the daemon taking it and the 200
arriving) sends it again with the same ``client_msg_id``. Until now the daemon took it as a second message: two
user rows, two turns. ``client_msg_id`` is the client's own name for the message, so the daemon treats a repeat of
it, in the same session, as the same message: the first is dispatched, the repeat is answered ``200`` with the
first one's ``run_id`` and ``"duplicate": true`` and dispatches nothing.

What counts as the same message, and what does not:

  * same session and same (trimmed) id: the same message, whatever the text says;
  * a different id, or no id: a different message, even with identical text ("two identical sends are
    legitimately two messages", ``lcm_conversation_store``);
  * the same id in another session: another message;
  * a request that was refused (bad mode, a session that is not the caller's) claims nothing, and a send whose
    dispatch failed can be retried: only a send that reached the bridge counts as sent;
  * it holds across a daemon restart, because the id is read back from the stored rows, and across two requests
    in flight at once.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.memory.lcm_conversation_store import LCMConversationStore  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402
from tests.support.device_world import GLOBAL, World  # noqa: E402


class _Bridge:
    """Records what was dispatched; can be told to fail or to wait."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str | None]] = []
        self.fail = False
        self.gate: asyncio.Event | None = None

    async def dispatch_user_message(self, session_id, content, client_msg_id=None, mode="agent",
                                    tool_choice=None, **kwargs):
        if self.gate is not None:
            await self.gate.wait()
        if self.fail:
            raise RuntimeError("dispatch failed")
        self.calls.append((session_id, content, client_msg_id))


def _client(bridge=None, **app_kwargs):
    app = create_app({}, **app_kwargs)
    app.state.ws_bridge = bridge or _Bridge()
    return TestClient(app, raise_server_exceptions=False), app.state.ws_bridge


def _send(client, *, sid="s1", text="hello", cid="c1", **extra):
    body = {"session_id": sid, "message": text, **({"client_msg_id": cid} if cid is not None else {}), **extra}
    return client.post("/api/chat/send", json=body)


# ── the same message ─────────────────────────────────────────────────────────

def test_a_repeat_of_a_client_msg_id_is_answered_with_the_first_run_and_dispatches_nothing():
    client, bridge = _client()
    first = _send(client, idempotency_key="run-1")
    again = _send(client, idempotency_key="run-2")
    assert first.status_code == 200 and first.json() == {"run_id": "run-1", "status": "sent"}
    assert again.status_code == 200
    assert again.json() == {"run_id": "run-1", "status": "sent", "duplicate": True}
    assert len(bridge.calls) == 1


def test_the_text_does_not_decide_it_the_id_does():
    client, bridge = _client()
    _send(client, text="first words")
    again = _send(client, text="other words entirely")
    assert again.json()["duplicate"] is True and bridge.calls == [("s1", "first words", "c1")]


def test_the_id_is_compared_after_trimming():
    client, bridge = _client()
    _send(client, cid="c1")
    assert _send(client, cid="  c1  ").json().get("duplicate") is True and len(bridge.calls) == 1


def test_a_first_send_does_not_say_duplicate():
    client, _ = _client()
    assert "duplicate" not in _send(client).json()


# ── not the same message ─────────────────────────────────────────────────────

def test_a_different_id_or_no_id_is_a_different_message():
    client, bridge = _client()
    _send(client, cid="c1")
    _send(client, cid="c2")
    _send(client, cid=None)
    _send(client, cid=None)
    _send(client, cid="")
    assert len(bridge.calls) == 5


def test_the_same_id_in_another_session_is_another_message():
    client, bridge = _client()
    _send(client, sid="s1", cid="c1")
    other = _send(client, sid="s2", cid="c1")
    assert "duplicate" not in other.json() and len(bridge.calls) == 2


# ── only a send that reached the bridge counts ───────────────────────────────

def test_a_refused_request_claims_nothing():
    client, bridge = _client()
    assert _send(client, mode="nonsense").status_code == 400
    assert _send(client, text="").status_code == 400
    ok = _send(client)
    assert ok.status_code == 200 and "duplicate" not in ok.json() and len(bridge.calls) == 1


def test_a_send_whose_dispatch_failed_can_be_retried():
    client, bridge = _client()
    bridge.fail = True
    assert _send(client).status_code == 500
    bridge.fail = False
    retry = _send(client)
    assert retry.status_code == 200 and "duplicate" not in retry.json() and len(bridge.calls) == 1


# ── two at once ──────────────────────────────────────────────────────────────

async def _first_then_repeat(bridge: _Bridge, *, first_fails: bool):
    """A first send held at the gate, then a repeat while it is still dispatching; the gate opens. Returns both."""
    app = create_app({})
    app.state.ws_bridge = bridge
    bridge.gate = asyncio.Event()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        body = {"session_id": "s1", "message": "hi", "client_msg_id": "c1"}
        first = asyncio.create_task(c.post("/api/chat/send", json=body))
        for _ in range(20):
            await asyncio.sleep(0)
        bridge.fail = first_fails
        second = asyncio.create_task(c.post("/api/chat/send", json=body))
        for _ in range(20):
            await asyncio.sleep(0)
        # Reviewed 2026-11-11: the repeat WAITS for the first rather than being told "sent" at once, because the
        # first can still fail, and a client told "sent" never retries.
        assert not second.done(), "the repeat was answered before the first send had finished"
        bridge.fail = False if not first_fails else True
        bridge.gate.set()
        if first_fails:
            # Only the first sees the failure; the repeat's own dispatch then succeeds.
            r1 = await asyncio.wait_for(first, timeout=2)
            bridge.fail = False
            r2 = await asyncio.wait_for(second, timeout=2)
        else:
            r1, r2 = await asyncio.wait_for(asyncio.gather(first, second), timeout=2)
    return r1, r2


@pytest.mark.asyncio
async def test_two_requests_in_flight_at_once_dispatch_once():
    bridge = _Bridge()
    first, second = await _first_then_repeat(bridge, first_fails=False)
    assert first.status_code == 200 and "duplicate" not in first.json()
    assert second.json() == {"run_id": first.json()["run_id"], "status": "sent", "duplicate": True}
    assert len(bridge.calls) == 1


@pytest.mark.asyncio
async def test_a_repeat_of_a_send_that_then_fails_is_sent_not_called_a_duplicate():
    class _FailOnce(_Bridge):
        async def dispatch_user_message(self, session_id, content, client_msg_id=None, mode="agent",
                                        tool_choice=None, **kwargs):
            if self.gate is not None:
                await self.gate.wait()
            if not getattr(self, "failed_once", False):
                self.failed_once = True
                raise RuntimeError("dispatch failed")
            self.calls.append((session_id, content, client_msg_id))

    bridge = _FailOnce()
    first, second = await _first_then_repeat(bridge, first_fails=True)
    assert first.status_code == 500
    assert second.status_code == 200 and "duplicate" not in second.json(), second.text
    assert bridge.calls == [("s1", "hi", "c1")]


# ── across a restart ─────────────────────────────────────────────────────────

def test_it_holds_across_a_restart_because_the_id_is_read_back_from_the_stored_rows(tmp_path):
    db = tmp_path / "lcm.db"
    before = LCMConversationStore(db_path=db)
    before.set_message_client_id("s1", 7, "c1")

    class _LCM:
        conversation_store = LCMConversationStore(db_path=db)

    client, bridge = _client(lcm_engine=_LCM())             # a fresh daemon: nothing in memory
    again = _send(client, cid="c1")
    assert again.json().get("duplicate") is True and bridge.calls == []
    assert "duplicate" not in _send(client, cid="c2").json() and len(bridge.calls) == 1


def test_the_store_answers_whether_an_id_was_recorded(tmp_path):
    store = LCMConversationStore(db_path=tmp_path / "lcm.db")
    assert store.has_message_client_id("s1", "c1") is False
    store.set_message_client_id("s1", 7, "c1")
    assert store.has_message_client_id("s1", "c1") is True
    assert store.has_message_client_id("s2", "c1") is False and store.has_message_client_id("s1", "c2") is False
    assert store.has_message_client_id("s1", "") is False and store.has_message_client_id("s1", None) is False


# ── the real bridge, the real store ──────────────────────────────────────────

def test_a_retried_send_leaves_one_user_row_and_one_turn(tmp_path):
    world = World(tmp_path)
    sid = world.call("a", "POST", "/api/sessions").json()["session_id"]
    body = {"session_id": sid, "message": "remember the milk", "client_msg_id": "cmid-1"}
    first = world.call("a", "POST", "/api/chat/send", json=body)
    again = world.call("a", "POST", "/api/chat/send", json=body)
    assert first.status_code == 200 and again.json().get("duplicate") is True
    rows = [m for m in world.lcm.get_all_messages(sid) if m.role == "user"]
    assert [m.content for m in rows] == ["remember the milk"]


def test_a_device_cannot_use_it_to_learn_about_another_devices_session(tmp_path):
    world = World(tmp_path)
    sid = world.device_session("a")
    world.call("a", "POST", "/api/chat/send", json={"session_id": sid, "message": "x", "client_msg_id": "secret-id"})
    probe = world.call("b", "POST", "/api/chat/send", json={"session_id": sid, "message": "y", "client_msg_id": "secret-id"})
    plain = world.call("b", "POST", "/api/chat/send", json={"session_id": sid, "message": "y", "client_msg_id": "other-id"})
    assert probe.status_code == plain.status_code and probe.json() == plain.json(), (
        "a foreign session answered differently for an id that was used there")
    assert "duplicate" not in probe.json()


def test_the_operator_can_still_send_after_a_device(tmp_path):
    world = World(tmp_path)
    sid = world.device_session("a")
    ok = world.call("op", "POST", "/api/chat/send", json={"session_id": sid, "message": "z", "client_msg_id": "op-1"})
    assert ok.status_code == 200 and GLOBAL
