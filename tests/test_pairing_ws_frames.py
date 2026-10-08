"""``pairing_pending`` / ``pairing_resolved`` — what the owner's Beacon is told, and who must NOT hear it.

A pairing frame carries a source address and the match code. #692's bridge filter (``_wants``) only holds
back a frame that NAMES a session a device does not own; a pairing frame names none, so sending it with
``broadcast`` would put it in front of every scoped device on the daemon. So the frames go through a
targeted send to sockets whose identity is an operator (``identity.is_operator``: the global token or an
owner device), never through ``broadcast`` and never through the SignalBus (its tail is durable and
replayed to any authenticated client).

Real objects: the app, the ``WebSocketBridge``, a ``DeviceStore``, the pairing runtime. The only stand-in is
a fake socket that records what it is sent.
"""

from __future__ import annotations

import json

import pytest

from prometheus.config.api_token import GLOBAL_IDENTITY, DeviceIdentity
from tests.support.pairing_world import GLOBAL, T0, World


class Sock:
    """A fake WebSocket: records what is sent, and can play the client side of the auth handshake."""

    def __init__(self, token: str | None = None, *, broken: bool = False) -> None:
        self.frames: list[dict] = []
        self._token = token
        self._broken = broken
        self.closed: tuple | None = None

    async def recv(self) -> str:
        return json.dumps({"type": "auth", "token": self._token})

    async def send(self, raw: str) -> None:
        if self._broken:
            raise ConnectionError("the peer went away")
        self.frames.append(json.loads(raw))

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration                      # the client sends nothing more and disconnects

    async def close(self, code=None, reason=None) -> None:
        self.closed = (code, reason)

    def types(self) -> list[str]:
        return [f["type"] for f in self.frames]

    def text(self) -> str:
        return json.dumps(self.frames)


@pytest.fixture
def world(tmp_path) -> World:
    return World(tmp_path, bridge=True, recording=False)


def attach(world: World, who: str, **kw) -> Sock:
    """Connect a fake socket the way `_authenticate` leaves a real one: in `_clients`, identity recorded."""
    ws = Sock(**kw)
    world.bridge._clients.add(ws)
    world.bridge._ws_identity[ws] = {
        "op": GLOBAL_IDENTITY,
        "owner": DeviceIdentity(id=world.owner["id"], name="Beacon on this Mac", platform="macos", owner=True),
        "scoped": DeviceIdentity(id=world.scoped["id"], name="a scoped phone", platform="ios"),
    }[who]
    return ws


async def connect(world: World, token: str | None) -> Sock:
    """A full connection through the bridge's own handler: auth frame, welcome, the backfill, disconnect."""
    ws = Sock(token)
    await world.bridge._handler(ws)
    return ws


# ── who hears a new request ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_new_request_reaches_the_global_token_and_an_owner_device_but_not_a_scoped_one(world):
    op, owner, scoped = attach(world, "op"), attach(world, "owner"), attach(world, "scoped")
    created, _, _ = world.created()
    assert op.types() == ["pairing_pending"] and owner.types() == ["pairing_pending"]
    assert scoped.frames == [], "a pairing frame names no session, so a broadcast would have reached it"
    assert op.frames[0]["payload"]["request_id"] == created["request_id"]


@pytest.mark.asyncio
async def test_a_socket_with_no_recorded_identity_hears_nothing(world):
    stray = Sock()
    world.bridge._clients.add(stray)                    # in the set, never authenticated
    world.created()
    assert stray.frames == [], "no identity means no operator: fail closed"


@pytest.mark.asyncio
async def test_the_frame_is_what_the_contract_says(world):
    op = attach(world, "op")
    created, _, _ = world.created(source="192.0.2.42")
    (frame,) = op.frames
    assert set(frame) == {"type", "timestamp", "payload"} and isinstance(frame["timestamp"], float)
    assert frame["payload"] == {
        "request_id": created["request_id"], "device_name": "Jennifer's MacBook", "platform": "macos",
        "source_ip": "192.0.2.42", "match_code": created["match_code"],
        "created_at": int(T0), "expires_at": int(T0) + 300, "ttl_seconds": 300}


@pytest.mark.asyncio
async def test_frames_never_go_through_broadcast(world, monkeypatch):
    async def forbidden(event):
        raise AssertionError("a pairing frame went through broadcast")

    monkeypatch.setattr(world.bridge, "broadcast", forbidden)
    op = attach(world, "op")
    world.created()
    assert op.types() == ["pairing_pending"]


@pytest.mark.asyncio
async def test_no_frame_holds_a_secret(world):
    op = attach(world, "op")
    created, requester, _ = world.created()
    world.approve(created)
    token = requester.unseal(created["request_id"], world.poll(created).json()["sealed"])["token"]
    everything = op.text()
    assert created["poll_secret"] not in everything and token not in everything


# ── notified is the truth ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_notified_is_true_only_when_an_operator_socket_took_it(world):
    attach(world, "scoped")
    assert world.created(source="192.0.2.1")[0]["notified"] is False, "only a scoped device is connected"
    attach(world, "op")
    assert world.created(source="192.0.2.2")[0]["notified"] is True


# ── decisions ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_an_approval_is_announced_as_resolved_with_who_decided(world):
    op, scoped = attach(world, "op"), attach(world, "scoped")
    created, _, _ = world.created()
    world.approve(created)
    resolved = [f for f in op.frames if f["type"] == "pairing_resolved"]
    assert [f["payload"] for f in resolved] == [
        {"request_id": created["request_id"], "resolution": "approved", "by": "beacon", "resolved_at": int(T0)}]
    assert scoped.frames == []


@pytest.mark.asyncio
async def test_denial_cancellation_and_expiry_are_announced_too(world):
    op = attach(world, "op")
    one, _, _ = world.created(source="192.0.2.1")
    two, _, two_client = world.created(source="192.0.2.2")
    three, _, _ = world.created(source="192.0.2.3")
    world.as_("global", "POST", f"/api/pair/requests/{one['request_id']}/deny")
    two_client.delete(f"/api/pair/requests/{two['request_id']}", headers={"X-Pairing-Secret": two["poll_secret"]})
    world.clock.now += 301
    world.as_("global", "GET", "/api/pair/requests")           # the sweep records the third's expiry
    got = {(f["payload"]["request_id"], f["payload"]["resolution"], f["payload"]["by"])
           for f in op.frames if f["type"] == "pairing_resolved"}
    assert got == {(one["request_id"], "denied", "beacon"), (two["request_id"], "canceled", "requester"),
                   (three["request_id"], "expired", "system")}


@pytest.mark.asyncio
async def test_a_dead_operator_socket_cannot_break_a_request(world):
    attach(world, "op", broken=True)
    live = attach(world, "owner")
    response, _, _ = world.request()
    assert response.status_code == 201
    assert live.types() == ["pairing_pending"], "the healthy operator still heard"


# ── connecting later ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_an_operator_who_connects_later_is_told_what_is_still_waiting(world):
    a, _, _ = world.created(source="192.0.2.1")
    world.clock.now += 5
    b, _, _ = world.created(source="192.0.2.2")
    ws = await connect(world, GLOBAL)
    assert ws.types() == ["connected", "pairing_pending", "pairing_pending"]
    assert [f["payload"]["request_id"] for f in ws.frames[1:]] == [a["request_id"], b["request_id"]], \
        "oldest first, so the newest arrives last"


@pytest.mark.asyncio
async def test_an_owner_device_is_backfilled_but_a_scoped_device_is_not(world):
    world.created()
    owner = await connect(world, world.owner["token"])
    scoped = await connect(world, world.scoped["token"])
    assert owner.types() == ["connected", "pairing_pending"]
    assert scoped.types() == ["connected"]


@pytest.mark.asyncio
async def test_only_live_requests_are_backfilled(world):
    done, _, _ = world.created(source="192.0.2.1")
    world.approve(done)
    old, _, _ = world.created(source="192.0.2.2")
    world.clock.now += 301
    live, _, _ = world.created(source="192.0.2.3")
    ws = await connect(world, GLOBAL)
    assert [f["payload"]["request_id"] for f in ws.frames if f["type"] == "pairing_pending"] == [live["request_id"]]
    assert old and done


@pytest.mark.asyncio
async def test_a_connection_with_nothing_waiting_gets_only_the_welcome(world):
    ws = await connect(world, GLOBAL)
    assert ws.types() == ["connected"]
    assert "pair_requests" not in world.tables(), "connecting must not create the table"
