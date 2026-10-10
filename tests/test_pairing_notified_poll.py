"""``notified`` on the pending poll: is anyone who can approve reachable NOW, not only when the request began.

The 201 says ``notified`` once, as of the moment the request was created. A requester whose owner opens
Beacon a moment later kept reading "no one is connected" from its own copy of that answer. The pending poll
now carries a current ``notified``, so the requester's screen can change its mind.

It only ever goes false -> true during a request's life. Both of its reasons are REMEMBERED for that request
the moment they happen, so neither can lapse:

* **an operator was connected**: a socket whose identity is an operator (the global token or an OWNER device,
  never a scoped device) was open while the request waited. A connection that arrives later is sent what is
  waiting (the backfill), and that send is what marks each request, so it counts even if the socket is gone
  before the requester's next poll; a poll that finds one connected marks it too;
* **a channel took the event** when the request was created (Telegram sends its prompt once and does not
  re-send one later, so it cannot be probed, only remembered).

Beacon found the bug this file used to contain: it read the live connection only, so an owner who connected and
then left turned the answer back to false, and a test here asserted exactly that.

Only a PENDING answer carries it; every other status keeps its exact shape.
"""

from __future__ import annotations

import json

import pytest

from prometheus.config.api_token import GLOBAL_IDENTITY, DeviceIdentity
from tests.support.pairing_world import GLOBAL, World


class Sock:
    """A fake WebSocket. With a token it plays the client side of the handshake for the bridge's own handler."""

    def __init__(self, token: str | None = None, *, broken: bool = False, die_on_pending: int | None = None) -> None:
        self.frames: list[dict] = []
        self._token = token
        self._broken = broken
        self._die_on_pending = die_on_pending           # fail the Nth pairing_pending frame, and everything after
        self._pending_seen = 0

    async def recv(self) -> str:
        return json.dumps({"type": "auth", "token": self._token})

    async def send(self, raw: str) -> None:
        if self._broken:
            raise ConnectionError("the peer went away")
        frame = json.loads(raw)
        if self._die_on_pending is not None and frame.get("type") == "pairing_pending":
            self._pending_seen += 1
            if self._pending_seen >= self._die_on_pending:
                self._broken = True
                raise ConnectionError("the peer went away mid-backfill")
        self.frames.append(frame)

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration                      # the client sends nothing more and disconnects

    async def close(self, code=None, reason=None) -> None:
        pass


@pytest.fixture
def world(tmp_path) -> World:
    return World(tmp_path, bridge=True, recording=False)


def attach(world: World, who: str) -> Sock:
    ws = Sock()
    world.bridge._clients.add(ws)
    world.bridge._ws_identity[ws] = {
        "op": GLOBAL_IDENTITY,
        "owner": DeviceIdentity(id=world.owner["id"], name="Beacon on this Mac", platform="macos", owner=True),
        "scoped": DeviceIdentity(id=world.scoped["id"], name="a scoped phone", platform="ios"),
    }[who]
    return ws


async def connect(world: World, token: str | None, **kw) -> Sock:
    """A whole connection through the bridge's own handler: auth, welcome, the backfill, disconnect."""
    ws = Sock(token, **kw)
    await world.bridge._handler(ws)
    return ws


def poll(world: World, created: dict) -> dict:
    world.clock.now += 2                                   # the poll limiter is one a second
    response = world.poll(created)
    assert response.status_code == 200, response.text
    return response.json()


def test_the_poll_says_nobody_is_there_when_nobody_is(world):
    created, _, _ = world.created()
    assert created["notified"] is False
    assert poll(world, created) == {"status": "pending", "expires_at": int(world.clock.now - 2) + 300,
                                    "notified": False}
    assert [poll(world, created)["notified"] for _ in range(3)] == [False] * 3, "asking does not make it true"


@pytest.mark.parametrize("who", ["owner", "op"])
def test_an_operator_who_connects_after_the_request_turns_it_true(world, who):
    created, _, _ = world.created()
    assert poll(world, created)["notified"] is False
    attach(world, who)
    assert poll(world, created)["notified"] is True


def test_a_scoped_device_never_counts(world):
    created, _, _ = world.created()
    attach(world, "scoped")
    assert poll(world, created)["notified"] is False, "a scoped phone cannot approve, so it is not 'someone to ask'"


def test_once_an_operator_has_been_connected_it_stays_true_after_they_leave(world):
    """Beacon's bug: true while the owner was connected, false again the poll after they disconnected."""
    created, _, _ = world.created()
    ws = attach(world, "owner")
    assert poll(world, created)["notified"] is True
    world.bridge._clients.discard(ws)
    assert poll(world, created)["notified"] is True, "they were told; leaving does not un-tell them"


def test_it_never_goes_back_to_false_whatever_connects_and_disconnects(world):
    created, _, _ = world.created()
    seen = [poll(world, created)["notified"]]
    ws = attach(world, "owner")
    seen.append(poll(world, created)["notified"])
    world.bridge._clients.discard(ws)
    seen.append(poll(world, created)["notified"])
    ws2 = attach(world, "op")
    seen.append(poll(world, created)["notified"])
    world.bridge._clients.discard(ws2)
    seen.append(poll(world, created)["notified"])
    assert seen == [False, True, True, True, True], "false -> true only, as the contract says"


@pytest.mark.asyncio
@pytest.mark.parametrize("token_of", [lambda w: GLOBAL, lambda w: w.owner["token"]], ids=["global", "owner"])
async def test_an_operator_who_connects_and_leaves_between_two_polls_still_counts(world, token_of):
    """The whole connection happens between two polls: no poll ever sees the socket open. The backfill is what tells
    the requester someone was shown the request."""
    created, _, _ = world.created()
    assert poll(world, created)["notified"] is False
    ws = await connect(world, token_of(world))
    assert [f["type"] for f in ws.frames if f["type"] == "pairing_pending"] == ["pairing_pending"], "they were shown it"
    assert world.bridge.operator_sockets() == [], "and they are gone"
    assert poll(world, created)["notified"] is True


@pytest.mark.asyncio
async def test_a_scoped_device_that_connects_and_leaves_tells_nobody(world):
    created, _, _ = world.created()
    ws = await connect(world, world.scoped["token"])
    assert not [f for f in ws.frames if f["type"] == "pairing_pending"]
    assert poll(world, created)["notified"] is False


@pytest.mark.asyncio
async def test_a_backfill_that_could_not_be_sent_tells_nobody(world):
    created, _, _ = world.created()
    await connect(world, GLOBAL, broken=True)
    assert poll(world, created)["notified"] is False


@pytest.mark.asyncio
async def test_a_socket_that_dies_mid_backfill_counts_only_for_the_frames_it_took(world):
    first, _, _ = world.created(source="192.0.2.10")
    second, _, _ = world.created(source="192.0.2.11")
    ws = await connect(world, GLOBAL, die_on_pending=2)
    assert [f["payload"]["request_id"] for f in ws.frames if f["type"] == "pairing_pending"] == [first["request_id"]]
    assert poll(world, first)["notified"] is True, "the first reached them"
    assert poll(world, second)["notified"] is False, "the second never did"


@pytest.mark.asyncio
async def test_what_an_operator_saw_belongs_to_the_requests_that_were_waiting(world):
    first, _, _ = world.created(source="192.0.2.10")
    await connect(world, GLOBAL)                       # saw the first, then left
    second, _, _ = world.created(source="192.0.2.11")  # arrived after they left
    assert second["notified"] is False
    assert poll(world, first)["notified"] is True
    assert poll(world, second)["notified"] is False, "nobody has been shown the second"


def test_a_request_that_reached_a_channel_at_creation_stays_told(tmp_path):
    """Telegram-like: a listener took the event once, and there is nothing to probe afterwards."""
    world = World(tmp_path, bridge=True, recording=True)         # the recording listener says it took the event
    created, _, _ = world.created()
    assert created["notified"] is True
    assert poll(world, created)["notified"] is True


def test_one_requests_telling_is_not_another(tmp_path):
    world = World(tmp_path, bridge=True, recording=False)
    first, _, _ = world.created(source="192.0.2.10")
    attach(world, "owner")
    second, _, _ = world.created(source="192.0.2.11")
    assert second["notified"] is True
    world.runtime.notifier.clear()                               # the bridge stops listening
    assert poll(world, first)["notified"] is False, "the first was never told and nobody is reachable now"
    assert poll(world, second)["notified"] is True, "the second was"


def test_only_a_pending_answer_carries_it(world):
    attach(world, "owner")
    created, _, _ = world.created()
    approved = world.approve(created)
    assert approved.status_code == 200
    body = poll(world, created)
    assert body["status"] == "approved" and "notified" not in body
    denied, _, _ = world.created(source="192.0.2.12")
    world.as_("global", "POST", f"/api/pair/requests/{denied['request_id']}/deny")
    assert poll(world, denied) == {"status": "denied"}


def test_a_probe_that_fails_is_not_an_operator(world):
    def broken() -> bool:
        raise RuntimeError("the probe fell over")

    world.runtime.notifier.subscribe(lambda kind, payload: False, audience=broken)
    created, _, _ = world.created()
    assert poll(world, created)["notified"] is False             # and the poll still answers


def test_what_it_remembers_is_bounded(tmp_path):
    world = World(tmp_path, recording=False)
    runtime = world.runtime
    for number in range(runtime.TOLD_REMEMBERED + 50):
        runtime.mark_notified(f"id-{number}")
    assert runtime.is_notified("id-0") is False, "the oldest is forgotten, not kept forever"
    assert runtime.is_notified(f"id-{runtime.TOLD_REMEMBERED + 49}") is True


def test_the_notifier_audience_is_any_probe_that_says_yes():
    from prometheus.web.pairing_routes import PairingNotifier

    notifier = PairingNotifier()
    assert notifier.audience() is False
    notifier.subscribe(lambda kind, payload: True)                # a listener with no probe is not an audience
    assert notifier.audience() is False
    notifier.subscribe(lambda kind, payload: False, audience=lambda: False)
    assert notifier.audience() is False
    notifier.subscribe(lambda kind, payload: False, audience=lambda: True)
    assert notifier.audience() is True
