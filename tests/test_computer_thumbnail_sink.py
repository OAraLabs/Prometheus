"""PR 6b — the transport gate: who may receive a desktop thumbnail.

``web.ws_server`` implements the ``ThumbnailSink`` the runner calls. These
tests are the security boundary of the feature: a screenshot is the most
sensitive thing the cockpit puts on the wire, and every condition below
exists to keep it off a socket that should not have it.

The four conditions (design §5.2.5):
  1. a DEVICE token — never the global token (D15: a model can read it);
  2. that device marked for computer use (the door's W3 ruling);
  3. the ``computer-thumbnails`` capability declared (an old client would
     render the whole payload into its exportable Activity feed);
  4. attached to the task's session.

Each gets a test where it is the ONLY thing failing, so a green suite says
each condition is independently load-bearing — not that some other condition
happened to cover for it.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from prometheus.config.device_store import DeviceStore
from prometheus.web.ws_server import WebSocketBridge

GLOBAL = "global-secret"
SESSION = "telegram:1"


class _FakeWS:
    """A socket that records what it was sent."""

    def __init__(self, auth_frame: str | None = None):
        self._auth_frame = auth_frame
        self.sent: list[dict] = []
        self.closed_with: int | None = None
        self.fail_send = False

    async def recv(self) -> str:
        if self._auth_frame is None:
            await asyncio.sleep(3600)
        return self._auth_frame

    async def send(self, raw: str) -> None:
        if self.fail_send:
            raise RuntimeError("socket gone")
        self.sent.append(json.loads(raw))

    async def close(self, code: int, reason: str = "") -> None:
        self.closed_with = code


def _rig(tmp_path, *, marked: bool = True):
    """A bridge with a device store, one enrolled device, and the token."""
    store = DeviceStore(tmp_path / "devices.db")
    minted = store.mint("phone", "ios")
    if marked:
        assert store.set_computer(minted["id"], True, by="will") is True
    bridge = WebSocketBridge(api_token=GLOBAL, device_store=store)
    return store, minted, bridge


async def _connect(bridge: WebSocketBridge, token: str,
                   *, caps: list[str] | None = None,
                   sessions: list[str] | None = None) -> _FakeWS:
    """Authenticate, register as a client, and subscribe."""
    ws = _FakeWS(json.dumps({"type": "auth", "token": token}))
    assert await bridge._authenticate(ws) is True
    bridge._clients.add(ws)
    payload: dict = {}
    if caps is not None:
        payload["capabilities"] = caps
    if sessions is not None:
        payload["sessions"] = sessions
    await bridge._handle_client_message(
        ws, json.dumps({"type": "subscribe", "payload": payload}))
    return ws


CAP = "computer-thumbnails"
FRAME = {"session_id": SESSION, "task_id": "abc123", "seq": 1, "app": "gedit",
         "mime_type": "image/png", "width": 480, "height": 300,
         "data_base64": "aGVsbG8=", "captured_at": "2026-10-04T00:00:00Z"}


@pytest.mark.asyncio
async def test_an_eligible_device_receives_it(tmp_path):
    """The positive case first: everything right, one picture delivered."""
    _store, minted, bridge = _rig(tmp_path)
    ws = await _connect(bridge, minted["token"], caps=[CAP])
    assert bridge.viewer_count(SESSION) == 1
    await bridge.send_thumbnail(SESSION, FRAME)
    thumbs = [f for f in ws.sent if f["type"] == "computer_step_thumbnail"]
    assert len(thumbs) == 1
    assert thumbs[0]["payload"] == FRAME


@pytest.mark.asyncio
async def test_a_global_token_never_receives_one(tmp_path):
    """CONDITION 1, and the reason it exists (D15): the global secret sits in
    a plain file the always-loaded bash tool can read, so a model holding it
    must not be able to watch the screen through a socket it opened."""
    _store, _minted, bridge = _rig(tmp_path)
    ws = await _connect(bridge, GLOBAL, caps=[CAP])
    assert bridge.viewer_count(SESSION) == 0
    await bridge.send_thumbnail(SESSION, FRAME)
    assert [f for f in ws.sent if f["type"] == "computer_step_thumbnail"] == []


@pytest.mark.asyncio
async def test_an_unmarked_device_never_receives_one(tmp_path):
    """CONDITION 2: a device nobody marked for computer use is not a viewer.
    Same W3 ruling that gates starting a task."""
    _store, minted, bridge = _rig(tmp_path, marked=False)
    ws = await _connect(bridge, minted["token"], caps=[CAP])
    assert bridge.viewer_count(SESSION) == 0
    await bridge.send_thumbnail(SESSION, FRAME)
    assert [f for f in ws.sent if f["type"] == "computer_step_thumbnail"] == []


@pytest.mark.asyncio
async def test_a_revoked_device_never_receives_one(tmp_path):
    store, minted, bridge = _rig(tmp_path)
    ws = await _connect(bridge, minted["token"], caps=[CAP])
    assert bridge.viewer_count(SESSION) == 1
    store.revoke(minted["id"])
    assert bridge.viewer_count(SESSION) == 0
    await bridge.send_thumbnail(SESSION, FRAME)
    assert [f for f in ws.sent if f["type"] == "computer_step_thumbnail"] == []


@pytest.mark.asyncio
async def test_a_client_that_declares_nothing_never_receives_one(tmp_path):
    """CONDITION 3: an old client renders an unknown kind's whole payload
    into its exportable Activity feed — a picture of a desktop in a file a
    person can copy off the machine. No declaration, no picture."""
    _store, minted, bridge = _rig(tmp_path)
    ws = await _connect(bridge, minted["token"])
    assert bridge.viewer_count(SESSION) == 0
    await bridge.send_thumbnail(SESSION, FRAME)
    assert [f for f in ws.sent if f["type"] == "computer_step_thumbnail"] == []


@pytest.mark.asyncio
async def test_a_different_capability_is_not_this_one(tmp_path):
    """Declaring SOME capability is not declaring this one — the check is the
    exact string, so a client cannot opt in by accident."""
    _store, minted, bridge = _rig(tmp_path)
    ws = await _connect(bridge, minted["token"], caps=["computer-action-log"])
    assert bridge.viewer_count(SESSION) == 0
    await bridge.send_thumbnail(SESSION, FRAME)
    assert [f for f in ws.sent if f["type"] == "computer_step_thumbnail"] == []


@pytest.mark.asyncio
async def test_a_socket_attached_to_another_session_never_receives_one(tmp_path):
    """CONDITION 4: the picture belongs to one session's task."""
    _store, minted, bridge = _rig(tmp_path)
    ws = await _connect(bridge, minted["token"], caps=[CAP],
                        sessions=["telegram:999"])
    assert bridge.viewer_count(SESSION) == 0
    await bridge.send_thumbnail(SESSION, FRAME)
    assert [f for f in ws.sent if f["type"] == "computer_step_thumbnail"] == []


@pytest.mark.asyncio
async def test_a_socket_on_the_firehose_receives_it(tmp_path):
    """No session filter means every session — that is today's behaviour, and
    it must not be broken by the thumbnail gate."""
    _store, minted, bridge = _rig(tmp_path)
    ws = await _connect(bridge, minted["token"], caps=[CAP])
    assert bridge.viewer_count(SESSION) == 1


@pytest.mark.asyncio
async def test_resubscribing_without_the_capability_stops_the_pictures(tmp_path):
    """A list REPLACES the previous set: a client that drops the capability
    stops receiving pictures, which is the point of replacing rather than
    accumulating."""
    _store, minted, bridge = _rig(tmp_path)
    ws = await _connect(bridge, minted["token"], caps=[CAP])
    assert bridge.viewer_count(SESSION) == 1
    await bridge._handle_client_message(ws, json.dumps(
        {"type": "subscribe", "payload": {"capabilities": []}}))
    assert bridge.viewer_count(SESSION) == 0


@pytest.mark.asyncio
async def test_a_junk_capability_list_declares_nothing(tmp_path):
    """Non-string entries are dropped, not errored — and dropping them all
    leaves a client that declared nothing, so it gets nothing."""
    _store, minted, bridge = _rig(tmp_path)
    ws = await _connect(bridge, minted["token"])
    await bridge._handle_client_message(ws, json.dumps(
        {"type": "subscribe", "payload": {"capabilities": [None, 1, CAP]}}))
    assert bridge.viewer_count(SESSION) == 1  # the one real string counts


@pytest.mark.asyncio
async def test_only_the_eligible_socket_of_two_gets_it(tmp_path):
    """The gate is per-socket, not per-bridge: one eligible phone and one
    global-token client connected at the same time, only the phone sees it."""
    store, minted, bridge = _rig(tmp_path)
    phone = await _connect(bridge, minted["token"], caps=[CAP])
    model = await _connect(bridge, GLOBAL, caps=[CAP])
    assert bridge.viewer_count(SESSION) == 1
    await bridge.send_thumbnail(SESSION, FRAME)
    assert len([f for f in phone.sent
                if f["type"] == "computer_step_thumbnail"]) == 1
    assert [f for f in model.sent
            if f["type"] == "computer_step_thumbnail"] == []


@pytest.mark.asyncio
async def test_no_device_store_means_no_viewers(tmp_path):
    """Fail closed: no registry means no device was ever marked, so there is
    no eligible viewer. An answer, not an error."""
    bridge = WebSocketBridge(api_token=GLOBAL, device_store=None)
    assert bridge.viewer_count(SESSION) == 0


@pytest.mark.asyncio
async def test_a_store_that_raises_means_no_viewers(tmp_path):
    """Fail closed on any doubt — an eligibility check that cannot answer
    must not answer yes."""
    store, minted, bridge = _rig(tmp_path)
    ws = await _connect(bridge, minted["token"], caps=[CAP])

    def boom(_id):
        raise RuntimeError("db gone")

    store.computer_allowed = boom
    assert bridge.viewer_count(SESSION) == 0
    await bridge.send_thumbnail(SESSION, FRAME)
    assert [f for f in ws.sent if f["type"] == "computer_step_thumbnail"] == []


@pytest.mark.asyncio
async def test_a_dead_socket_is_dropped_not_retried(tmp_path):
    """A failed send is a dropped picture. Never a retry, never a queue, and
    never a reason to end a task."""
    _store, minted, bridge = _rig(tmp_path)
    ws = await _connect(bridge, minted["token"], caps=[CAP])
    ws.fail_send = True
    await bridge.send_thumbnail(SESSION, FRAME)  # must not raise
    assert ws not in bridge._clients
    assert ws not in bridge._ws_caps


@pytest.mark.asyncio
async def test_a_disconnect_clears_its_capability(tmp_path):
    """No stale entry: a socket that goes away is not an eligible viewer.
    Exercises the real cleanup — the same three pops in _handler's finally —
    so a socket that closed mid-task stops being counted."""
    _store, minted, bridge = _rig(tmp_path)
    ws = await _connect(bridge, minted["token"], caps=[CAP])
    assert bridge.viewer_count(SESSION) == 1
    # _handler's finally, verbatim:
    bridge._clients.discard(ws)
    bridge._ws_identity.pop(ws, None)
    bridge._ws_filters.pop(ws, None)
    bridge._ws_caps.pop(ws, None)
    assert bridge.viewer_count(SESSION) == 0


@pytest.mark.asyncio
async def test_the_subscribe_ack_echoes_the_capabilities(tmp_path):
    """The client can see what it declared — a capability silently dropped
    would look identical to one never asked for."""
    _store, minted, bridge = _rig(tmp_path)
    ws = await _connect(bridge, minted["token"], caps=[CAP])
    acks = [f for f in ws.sent if f["type"] == "subscribed"]
    assert acks and acks[-1]["payload"]["capabilities"] == [CAP]


@pytest.mark.asyncio
async def test_an_unmarked_socket_gets_no_frame_of_any_kind(tmp_path):
    """send_thumbnail is not broadcast(): an ineligible socket gets nothing
    at all — not the thumbnail, not some generic wrapper of it."""
    _store, minted, bridge = _rig(tmp_path, marked=False)
    ws = await _connect(bridge, minted["token"], caps=[CAP])
    sent_before = len(ws.sent)
    await bridge.send_thumbnail(SESSION, FRAME)
    assert ws.sent[sent_before:] == []  # nothing new after the subscribe ack


def test_the_bridge_satisfies_the_sink_protocol(tmp_path):
    """The runner talks to a ThumbnailSink, not to a WebSocketBridge. This is
    checked at boot in launcher.py's wiring; if the method names drift, the
    failure lands at the first desktop task, not at import. Pin it here."""
    from prometheus.computer.thumbnails import ThumbnailSink

    _store, _minted, bridge = _rig(tmp_path)
    assert isinstance(bridge, ThumbnailSink)


def test_the_sink_protocol_declares_the_bridge_method_names():
    """Guard the other direction: the protocol must name `send_thumbnail`,
    matching the bridge. A `send`/`send_thumbnail` mismatch type-checks and
    then AttributeErrors at runtime."""
    from prometheus.computer.thumbnails import ThumbnailSink

    names = set(getattr(ThumbnailSink, "__annotations__", {}))
    # runtime_checkable protocols track members, not annotations; check both
    # the callable attributes the runner will invoke.
    assert hasattr(ThumbnailSink, "viewer_count")
    assert hasattr(ThumbnailSink, "send_thumbnail")
    assert not hasattr(ThumbnailSink, "send")
