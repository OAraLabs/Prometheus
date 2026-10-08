"""``notified`` on the pending poll: is anyone who can approve reachable NOW, not only when the request began.

The 201 says ``notified`` once, as of the moment the request was created. A requester whose owner opens
Beacon a moment later kept reading "no one is connected" from its own copy of that answer. The pending poll
now carries a current ``notified``, so the requester's screen can change its mind.

It only ever goes false -> true during a request's life, because both of its reasons stay true:

* **the live reason**: an operator socket is connected (the global token or an OWNER device, never a scoped
  device), and a connection that arrives later is told what is waiting (the backfill), so true here means
  "will have been told", not merely "is online";
* **the sticky reason**: some channel took the event when the request was created (Telegram sends its prompt
  once and does not re-send one later, so it cannot be probed, only remembered).

Only a PENDING answer carries it; every other status keeps its exact shape.
"""

from __future__ import annotations

import json

import pytest

from prometheus.config.api_token import GLOBAL_IDENTITY, DeviceIdentity
from tests.support.pairing_world import World


class Sock:
    def __init__(self) -> None:
        self.frames: list[dict] = []

    async def send(self, raw: str) -> None:
        self.frames.append(json.loads(raw))


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


def test_it_follows_the_live_connection_when_nothing_was_sent_at_creation(world):
    created, _, _ = world.created()
    ws = attach(world, "owner")
    assert poll(world, created)["notified"] is True
    world.bridge._clients.discard(ws)
    assert poll(world, created)["notified"] is False


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
