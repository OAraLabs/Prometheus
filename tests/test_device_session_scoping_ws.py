"""Device scoping (WebSocket) — a device socket sees and drives only its own sessions.

The :8010 bridge is the second door onto the same conversations. Before this
change it had three ways round the REST rule:

  * ``switch_session`` replayed ANY session's history to ANY authenticated socket;
  * every turn frame for every session was broadcast to every socket (the
    ``subscribe`` filter is a preference the CLIENT sets, not a boundary);
  * ``send_message`` / ``chat_upload`` / ``interrupt`` took a session id and
    acted on it, whoever's it was.

The operator's socket (global token) is unchanged: it still receives the firehose.
See tests/test_device_session_scoping_rest.py for the ownership rule itself.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from prometheus.config.api_token import GLOBAL_IDENTITY, DeviceIdentity
from prometheus.web.ws_server import (
    _computer_frame_kinds,
    _coding_frame_kinds,
)
from tests.support.device_world import A_SECRET, GLOBAL, TG_SECRET, World


class Sock:
    """A fake WebSocket that records every frame the bridge sends it."""

    def __init__(self) -> None:
        self.frames: list[dict] = []

    async def send(self, raw: str) -> None:
        self.frames.append(json.loads(raw))

    def types(self) -> list[str]:
        return [f["type"] for f in self.frames]

    def text(self) -> str:
        return json.dumps(self.frames)

    def errors(self) -> list[dict]:
        return [f["payload"] for f in self.frames if f["type"] == "error"]


@pytest.fixture
def world(tmp_path) -> World:
    return World(tmp_path)


def attach(world: World, who: str) -> Sock:
    """Connect a fake socket the way `_authenticate` leaves a real one: in
    `_clients`, with its identity recorded in `_ws_identity`."""
    ws = Sock()
    world.bridge._clients.add(ws)
    world.bridge._ws_identity[ws] = {
        "op": GLOBAL_IDENTITY,
        "a": DeviceIdentity(id=world.a["id"], name="device-a", platform="ios"),
        "b": DeviceIdentity(id=world.b["id"], name="device-b", platform="macos"),
    }[who]
    return ws


async def cmd(world: World, ws: Sock, kind: str, **payload) -> None:
    await world.bridge._handle_client_message(ws, json.dumps({"type": kind, "payload": payload}))


def frame(kind: str, **payload) -> dict:
    return {"type": kind, "timestamp": 1.0, "payload": payload}


# --------------------------------------------------------------------------- #
# switch_session replays history
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_switch_session_does_not_replay_another_devices_history(world):
    sid_a = world.device_session("a")
    ws_b = attach(world, "b")

    await cmd(world, ws_b, "switch_session", session_id=sid_a)

    assert A_SECRET not in ws_b.text(), "B was replayed A's conversation"
    assert [e["kind"] for e in ws_b.errors()] == ["not_found"]


@pytest.mark.asyncio
async def test_switch_session_does_not_replay_an_operator_session(world):
    tg = world.operator_session()
    ws_a = attach(world, "a")
    await cmd(world, ws_a, "switch_session", session_id=tg)
    assert TG_SECRET not in ws_a.text()


@pytest.mark.asyncio
async def test_switch_session_still_replays_to_the_owner_and_the_operator(world):
    sid_a = world.device_session("a")
    ws_a, ws_op = attach(world, "a"), attach(world, "op")

    await cmd(world, ws_a, "switch_session", session_id=sid_a)
    await cmd(world, ws_op, "switch_session", session_id=sid_a)

    assert A_SECRET in ws_a.text() and not ws_a.errors()
    assert A_SECRET in ws_op.text()


@pytest.mark.asyncio
async def test_switching_to_a_brand_new_id_claims_it(world):
    ws_b = attach(world, "b")
    await cmd(world, ws_b, "switch_session", session_id="ios:fresh")
    assert not ws_b.errors()
    # The id is B's now: A cannot read it over REST.
    assert world.call("b", "GET", "/api/sessions/ios:fresh/messages").status_code == 200
    assert world.call("a", "GET", "/api/sessions/ios:fresh/messages").status_code == 404


@pytest.mark.asyncio
async def test_switching_to_a_daemon_namespace_creates_nothing(world):
    ws_a = attach(world, "a")
    await cmd(world, ws_a, "switch_session", session_id="telegram:777")
    assert [e["kind"] for e in ws_a.errors()] == ["not_found"]
    assert world.mgr.get("telegram:777") is None, "a device conjured a Telegram session"


@pytest.mark.asyncio
async def test_a_foreign_switch_and_a_missing_switch_are_told_apart_by_nothing_the_device_can_see(world):
    """Both an existing foreign id and a reserved id answer with the same frame."""
    sid_a = world.device_session("a")
    ws_b = attach(world, "b")
    await cmd(world, ws_b, "switch_session", session_id=sid_a)
    await cmd(world, ws_b, "switch_session", session_id="telegram:777")
    first, second = ws_b.errors()
    assert first["message"] == second["message"] and first["kind"] == second["kind"]


# --------------------------------------------------------------------------- #
# The broadcast
# --------------------------------------------------------------------------- #

TURN_KINDS = ("chat_message", "chat_delta", "chat_done", "command_done", "agent_progress",
              "tool_call_start", "tool_call_end", "provider_degraded", "error")


def _session_frames(sid: str) -> list[dict]:
    """One frame of every kind that carries a session id, in each shape it travels in."""
    frames = [frame(k, session_id=sid) for k in TURN_KINDS]
    # Signal-bus kinds promoted to their own frame type (flat session_id).
    frames += [frame(k, session_id=sid) for k in (*_coding_frame_kinds(), *_computer_frame_kinds())]
    frames += [frame(k, session_id=sid, request_id="r1") for k in ("approval_pending", "approval_resolved")]
    frames += [frame(k, session_id=sid) for k in ("turn_completed", "task_completed", "task_failed")]
    # The generic wrapper nests the signal payload one level down.
    frames.append({"type": "sentinel_signal", "timestamp": 1.0, "payload": {
        "kind": "idle_start", "payload": {"session_id": sid}, "source": "x"}})
    return frames


@pytest.mark.asyncio
async def test_a_device_socket_receives_only_its_own_sessions_frames(world):
    sid_a = world.device_session("a")
    sid_b = world.device_session("b", text="b's own")
    ws_a, ws_b, ws_op = attach(world, "a"), attach(world, "b"), attach(world, "op")

    for ev in _session_frames(sid_a):
        await world.bridge.broadcast(ev)

    n = len(_session_frames(sid_a))
    assert len(ws_a.frames) == n, "the owner missed some of its own frames"
    assert len(ws_op.frames) == n, "the operator socket is the firehose and must stay one"
    assert ws_b.frames == [], f"B received A's frames: {ws_b.types()}"

    ws_b.frames.clear()
    for ev in _session_frames(sid_b):
        await world.bridge.broadcast(ev)
    assert len(ws_b.frames) == n


@pytest.mark.asyncio
async def test_a_device_socket_receives_nothing_from_an_unowned_session(world):
    tg = world.operator_session()
    ws_a, ws_op = attach(world, "a"), attach(world, "op")
    for ev in _session_frames(tg):
        await world.bridge.broadcast(ev)
    assert ws_a.frames == []
    assert len(ws_op.frames) == len(_session_frames(tg))


@pytest.mark.asyncio
async def test_frames_that_belong_to_no_session_still_reach_every_socket(world):
    """dream_*, memory_updated, skill_created … are the daemon's, not a conversation's."""
    ws_a, ws_b = attach(world, "a"), attach(world, "b")
    for kind in ("dream_start", "memory_updated", "skill_created", "curator_report"):
        await world.bridge.broadcast(frame(kind, note="x"))
    assert ws_a.types() == ws_b.types() == ["dream_start", "memory_updated", "skill_created",
                                            "curator_report"]


@pytest.mark.asyncio
async def test_subscribing_cannot_widen_what_a_device_receives(world):
    sid_a = world.device_session("a")
    ws_b = attach(world, "b")

    await cmd(world, ws_b, "subscribe", sessions=[sid_a])      # asks for A's session by name
    ws_b.frames.clear()
    await world.bridge.broadcast(frame("chat_delta", session_id=sid_a, text=A_SECRET))
    assert ws_b.frames == []

    await cmd(world, ws_b, "subscribe", sessions=[])           # the firehose request
    ws_b.frames.clear()
    await world.bridge.broadcast(frame("chat_delta", session_id=sid_a, text=A_SECRET))
    assert ws_b.frames == []


@pytest.mark.asyncio
async def test_a_user_message_sent_over_rest_reaches_only_the_owners_socket(world):
    """The real path: POST /api/chat/send persists, then broadcasts through the bridge."""
    sid_a = world.call("a", "POST", "/api/sessions").json()["session_id"]
    ws_a, ws_b, ws_op = attach(world, "a"), attach(world, "b"), attach(world, "op")

    r = world.call("a", "POST", "/api/chat/send", json={"session_id": sid_a, "message": A_SECRET})
    assert r.status_code == 200

    assert A_SECRET in ws_a.text() and A_SECRET in ws_op.text()
    assert A_SECRET not in ws_b.text()


# --------------------------------------------------------------------------- #
# Commands that take a session id
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_device_cannot_send_a_message_into_another_devices_session_over_ws(world):
    sid_a = world.device_session("a")
    ws_b = attach(world, "b")
    before = world.lcm.count_all(sid_a)

    await cmd(world, ws_b, "send_message", session_id=sid_a, content="injected")

    assert world.lcm.count_all(sid_a) == before, "B wrote into A's conversation"
    assert [e["kind"] for e in ws_b.errors()] == ["not_found"]


@pytest.mark.asyncio
async def test_a_device_can_send_over_ws_to_its_own_and_to_a_brand_new_session(world):
    sid_a = world.device_session("a")
    ws_a = attach(world, "a")
    before = world.lcm.count_all(sid_a)

    await cmd(world, ws_a, "send_message", session_id=sid_a, content="again")
    await cmd(world, ws_a, "send_message", session_id="ios:made-up", content="first")

    assert world.lcm.count_all(sid_a) == before + 1
    assert world.lcm.count_all("ios:made-up") == 1
    assert not ws_a.errors()
    # …and "ios:made-up" is A's now.
    assert world.call("b", "GET", "/api/sessions/ios:made-up/messages").status_code == 404


@pytest.mark.asyncio
async def test_a_device_cannot_upload_into_another_devices_session(world, monkeypatch):
    # No disk: the "not cached" path carries the upload as a note, which is still a write.
    monkeypatch.setattr("prometheus.gateway.media_cache.cache_document_from_bytes",
                        lambda data, filename: None)
    sid_a = world.device_session("a")
    ws_b = attach(world, "b")
    before = world.lcm.count_all(sid_a)

    await cmd(world, ws_b, "chat_upload", session_id=sid_a, filename="notes.txt",
              content_base64="aGVsbG8=", mime_type="text/plain")

    assert world.lcm.count_all(sid_a) == before, "B's upload landed in A's conversation"
    assert [e["kind"] for e in ws_b.errors()] == ["not_found"]


@pytest.mark.asyncio
async def test_a_device_cannot_interrupt_another_devices_turn_over_ws(world):
    sid_a = world.device_session("a")
    stopped: list[str] = []
    world.bridge.interrupt_turn = lambda sid: stopped.append(sid) or True
    ws_b, ws_a = attach(world, "b"), attach(world, "a")

    await cmd(world, ws_b, "interrupt", session_id=sid_a)
    ack = next(f for f in ws_b.frames if f["type"] == "interrupt_ack")["payload"]
    assert ack == {"session_id": sid_a, "stopped": False}
    assert stopped == []

    await cmd(world, ws_a, "interrupt", session_id=sid_a)
    assert next(f for f in ws_a.frames if f["type"] == "interrupt_ack")["payload"]["stopped"] is True
    assert stopped == [sid_a]


@pytest.mark.asyncio
async def test_the_operator_socket_can_drive_any_session(world):
    sid_a = world.device_session("a")
    ws_op = attach(world, "op")
    before = world.lcm.count_all(sid_a)
    await cmd(world, ws_op, "send_message", session_id=sid_a, content="from the operator")
    assert world.lcm.count_all(sid_a) == before + 1
    assert not ws_op.errors()


# --------------------------------------------------------------------------- #
# End to end over a real socket: the identity must survive `_authenticate`
# --------------------------------------------------------------------------- #


async def _frames_after(ws, secs: float = 0.6) -> list[dict]:
    out: list[dict] = []
    while True:
        try:
            out.append(json.loads(await asyncio.wait_for(ws.recv(), timeout=secs)))
        except asyncio.TimeoutError:
            return out


@pytest.mark.asyncio
async def test_over_a_real_socket_a_device_token_is_scoped_and_the_global_token_is_not(world):
    websockets = pytest.importorskip("websockets")
    sid_a = world.device_session("a")
    await world.bridge.start(host="127.0.0.1", port=0)
    port = world.bridge._server.sockets[0].getsockname()[1]
    try:
        async def replay(token: str) -> list[dict]:
            async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                await ws.send(json.dumps({"type": "auth", "token": token}))
                await ws.send(json.dumps({"type": "switch_session",
                                          "payload": {"session_id": sid_a}}))
                return await _frames_after(ws)

        as_b = await replay(world.b["token"])
        as_a = await replay(world.a["token"])
        as_op = await replay(GLOBAL)
    finally:
        await world.bridge.stop()

    assert A_SECRET not in json.dumps(as_b), "B's real socket was replayed A's history"
    assert A_SECRET in json.dumps(as_a)
    assert A_SECRET in json.dumps(as_op)


@pytest.mark.asyncio
async def test_a_frame_the_scope_check_cannot_decide_is_withheld_not_raised(world):
    """broadcast() runs inside the agent turn that emitted the frame: a failing ownership
    lookup must withhold the frame from the device socket and let everyone else through."""
    sid_a = world.device_session("a")
    ws_a, ws_op = attach(world, "a"), attach(world, "op")

    def _boom(_sid):
        raise RuntimeError("devices.db went away")

    world.devices.session_owner = _boom  # type: ignore[method-assign]
    await world.bridge.broadcast(frame("chat_delta", session_id=sid_a))   # must not raise

    assert ws_a.frames == []
    assert ws_op.types() == ["chat_delta"]


@pytest.mark.asyncio
@pytest.mark.parametrize("junk", [7, ["a"], {"x": 1}, True])
async def test_a_session_id_that_is_not_a_string_is_refused_not_crashed_on(world, junk):
    ws_b = attach(world, "b")
    for kind, extra in (("switch_session", {}), ("send_message", {"content": "x"}),
                        ("chat_upload", {"content_base64": "aGk="}), ("interrupt", {}),
                        ("subscribe", {})):
        await cmd(world, ws_b, kind, session_id=junk, **extra)       # none of these may raise
    assert world.mgr._sessions == {}, "a junk id created a session"
