"""One approval request by id, and the push that takes a notification back (Beacon iOS "Piece 16", b and c).

16b. A phone opening an approval notification fetched ``GET /api/approvals`` and looked for its ``request_id``.
When the request was not in the list it could not tell "answered on the desktop a second ago" from "not this
device's", so it kept Deny on a request that may be dead. ``GET /api/approvals/{request_id}`` answers:

  * 200 with the ``serialize_pending`` dict while pending (byte for byte the list's entry);
  * 410 with ``resolution`` and ``detail`` once it has ended (the approve route's ``_no_longer_active``);
  * 404 when it is not the caller's, or this daemon has no record of it at all.

Same ownership rule and order as the approve route: a scoped device hears only about its own sessions' requests,
pending or ended, and anything else is the same 404 as "never existed".

16c. A request answered elsewhere left a notification with a live Approve button on every phone until it
expired. When a request that was pushed ends (approved, denied, withdrawn, expired), the dispatcher sends one
background push (``apns-push-type: background``, priority 5, ``content-available: 1``, no alert) carrying
``approval_resolved`` and ``resolution`` to each device the request was pushed to, and to no other. Best effort:
Apple throttles background pushes and nothing depends on it.
"""

from __future__ import annotations

import asyncio
import base64
import secrets
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from prometheus.config.device_store import DeviceStore  # noqa: E402
from prometheus.permissions.approval_queue import ApprovalQueue, PendingAction  # noqa: E402
from prometheus.permissions.approver import in_process  # noqa: E402
from prometheus.push.apns import SendResult  # noqa: E402
from prometheus.push.dispatcher import PushDispatcher  # noqa: E402

GLOBAL = "by-id-global-" + secrets.token_hex(8)
PUB = base64.b64encode(bytes(range(32))).decode()


class _Gate:
    """The queue requires a gate reference; nothing here records a grant."""

    def list_grants(self):
        return []


class _Bus:
    def __init__(self) -> None:
        self.signals: list = []

    async def emit(self, signal) -> None:
        self.signals.append(signal)


class Daemon:
    def __init__(self, tmp_path) -> None:
        self.devices = DeviceStore(tmp_path / "devices.db")
        from prometheus.web.server import create_app

        self.app = create_app({"web": {"api_token": GLOBAL}}, device_store=self.devices)
        self.queue = ApprovalQueue(security_gate=_Gate(), telegram_adapter=None)
        self.queue.signal_bus = _Bus()
        self.app.state.approval_queue = self.queue
        self.owner = self.devices.mint_owner("Beacon on this Mac", "macos", by="same-mac-pairing")
        self.alice = self.devices.mint("alice's phone", "ios")
        self.bob = self.devices.mint("bob's phone", "ios")
        self.devices.claim_session("web:alice-1", self.alice["id"])
        self.client = TestClient(self.app, client=("127.0.0.1", 50123))

    def token(self, who: str) -> str:
        return {"global": GLOBAL, "owner": self.owner["token"], "alice": self.alice["token"],
                "bob": self.bob["token"]}[who]

    def get(self, who: str, path: str):
        return self.client.get(path, headers={"Authorization": f"Bearer {self.token(who)}"})

    def request(self, request_id: str, *, session: str = "web:alice-1") -> PendingAction:
        action = PendingAction(request_id=request_id, tool_name="bash", description="run ls -la",
                               session_id=session)
        self.queue._register(action)
        return action


@pytest.fixture
def d(tmp_path) -> Daemon:
    return Daemon(tmp_path)


# ── 16b: one request by id ───────────────────────────────────────────────────

@pytest.mark.parametrize("who", ["global", "owner", "alice"])
def test_a_pending_request_is_200_and_byte_for_byte_the_lists_entry(d, who):
    d.request("b0b0a001")
    one = d.get(who, "/api/approvals/b0b0a001")
    assert one.status_code == 200, one.text
    listed = [a for a in d.get(who, "/api/approvals").json() if a["request_id"] == "b0b0a001"]
    assert listed == [one.json()]
    assert one.content == d.client.get("/api/approvals", headers={
        "Authorization": f"Bearer {d.token(who)}"}).content[1:-1]


@pytest.mark.parametrize(("verb", "resolution", "says"), [
    ("approve", "approved", "already approved"),
    ("deny", "denied", "already denied"),
])
def test_an_answered_request_is_410_with_how_it_ended(d, verb, resolution, says):
    d.request("b0b0a002")
    answered = d.client.post(f"/api/approvals/b0b0a002/{verb}", headers={"Authorization": f"Bearer {GLOBAL}"},
                             json={"scope": "once"} if verb == "approve" else None)
    assert answered.status_code == 200, answered.text
    for who in ("owner", "alice"):
        r = d.get(who, "/api/approvals/b0b0a002")
        assert r.status_code == 410, r.text
        assert r.json()["resolution"] == resolution and says in r.json()["detail"]


@pytest.mark.asyncio
async def test_a_withdrawn_request_is_410_and_says_why(d):
    d.request("b0b0a003")
    await d.queue.withdraw_all(why="the daemon is shutting down")
    r = d.get("owner", "/api/approvals/b0b0a003")
    assert r.status_code == 410 and r.json()["resolution"] == "withdrawn"
    assert "shutting down" in r.json()["detail"]


@pytest.mark.parametrize("who", ["global", "owner", "alice", "bob"])
def test_a_request_this_daemon_never_raised_is_404(d, who):
    r = d.get(who, "/api/approvals/abcdef12")
    assert r.status_code == 404, r.text


@pytest.mark.asyncio
async def test_another_devices_request_is_404_pending_or_ended(d):
    d.request("b0b0a004")
    assert d.get("bob", "/api/approvals/b0b0a004").status_code == 404
    await d.queue.deny("b0b0a004", by=in_process("test"))
    assert d.get("owner", "/api/approvals/b0b0a004").status_code == 410
    r = d.get("bob", "/api/approvals/b0b0a004")
    assert r.status_code == 404, r.text
    assert r.json() == d.get("bob", "/api/approvals/abcdef12").json()


def test_the_route_needs_a_token(d):
    d.request("b0b0a005")
    assert d.client.get("/api/approvals/b0b0a005").status_code == 401


def test_the_grants_list_is_still_the_grants_list(d):
    r = d.get("global", "/api/approvals/grants")
    assert r.status_code == 200, r.text
    assert r.json() == []


@pytest.mark.parametrize("who", ["alice", "bob"])
def test_a_scoped_device_still_cannot_read_the_grants_list(d, who):
    r = d.get(who, "/api/approvals/grants")
    assert r.status_code == 403, r.text


def test_the_route_is_classified_for_scoped_devices():
    from prometheus.web import route_access

    text = open(route_access.__file__, encoding="utf-8").read()
    assert '("GET", "/api/approvals/{request_id}")' in text


# ── 16c: the push that takes the notification back ───────────────────────────

def _target(device_id: str):
    return SimpleNamespace(id=device_id, apns_token=f"tok-{device_id}", environment="sandbox",
                           bundle_id="ai.test.beacon", push_failures=0)


class _Store:
    def __init__(self, ids) -> None:
        self.ids = list(ids)

    def push_targets(self):
        return [_target(i) for i in self.ids]

    def reset_push_failures(self, _id):
        pass

    def clear_push(self, device_id):
        self.ids.remove(device_id)

    def record_push_failure(self, _id):
        return 1


class _Sender:
    config = SimpleNamespace(topic="ai.test.beacon")

    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, **kwargs):
        self.sent.append(kwargs)
        return SendResult(outcome="ok", status=200)


def _sig(kind: str, payload: dict):
    return SimpleNamespace(kind=kind, payload=payload)


def _pending(request_id: str = "c0c0a001", **extra) -> dict:
    return {"request_id": request_id, "tool_name": "bash", "description": "ls", "expires_at": 1.0, **extra}


def _cleared(sender: _Sender) -> list[dict]:
    return [s for s in sender.sent if "approval_resolved" in s["payload"]]


@pytest.mark.parametrize("resolution", ["approved", "denied", "withdrawn", "expired"])
def test_an_ended_request_sends_one_background_push_per_device_it_was_pushed_to(resolution):
    store, sender = _Store(["d1", "d2"]), _Sender()
    disp = PushDispatcher(store, sender, bridge=None, node_pub=lambda: PUB)
    asyncio.run(disp.on_signal(_sig("approval_pending", _pending())))
    store.ids.append("d3")                       # registered after the request was pushed: not told
    asyncio.run(disp.on_signal(_sig("approval_resolved", {"request_id": "c0c0a001", "resolution": resolution})))
    cleared = _cleared(sender)
    assert sorted(s["apns_token"] for s in cleared) == ["tok-d1", "tok-d2"]
    for s in cleared:
        assert s["push_type"] == "background" and s["priority"] == 5
        assert s["payload"] == {"aps": {"content-available": 1}, "approval_resolved": "c0c0a001",
                                "resolution": resolution, "node_pub": PUB}
        assert s["topic"] == "ai.test.beacon"


def test_a_desktop_prompt_is_taken_back_too():
    store, sender = _Store(["d1"]), _Sender()
    disp = PushDispatcher(store, sender, bridge=None, node_pub=lambda: PUB)
    asyncio.run(disp.on_signal(_sig("approval_pending", _pending(task_id="t1"))))
    asyncio.run(disp.on_signal(_sig("approval_resolved", {"request_id": "c0c0a001", "resolution": "approved"})))
    assert len(_cleared(sender)) == 1


def test_a_request_that_was_never_pushed_sends_nothing():
    store, sender = _Store(["d1"]), _Sender()
    disp = PushDispatcher(store, sender, bridge=None, node_pub=lambda: PUB)
    asyncio.run(disp.on_signal(_sig("approval_resolved", {"request_id": "never", "resolution": "denied"})))
    assert sender.sent == []


def test_it_is_sent_once_however_many_times_the_end_is_heard():
    store, sender = _Store(["d1"]), _Sender()
    disp = PushDispatcher(store, sender, bridge=None, node_pub=lambda: PUB)
    asyncio.run(disp.on_signal(_sig("approval_pending", _pending())))
    for _ in range(3):
        asyncio.run(disp.on_signal(_sig("approval_resolved", {"request_id": "c0c0a001", "resolution": "denied"})))
    assert len(_cleared(sender)) == 1


def test_a_device_that_unregistered_since_is_not_pushed():
    store, sender = _Store(["d1", "d2"]), _Sender()
    disp = PushDispatcher(store, sender, bridge=None, node_pub=lambda: PUB)
    asyncio.run(disp.on_signal(_sig("approval_pending", _pending())))
    store.ids.remove("d2")
    asyncio.run(disp.on_signal(_sig("approval_resolved", {"request_id": "c0c0a001", "resolution": "denied"})))
    assert [s["apns_token"] for s in _cleared(sender)] == ["tok-d1"]


def test_a_failed_approval_push_is_not_taken_back():
    store = _Store(["d1", "d2"])

    class _Half(_Sender):
        async def send(self, **kwargs):
            self.sent.append(kwargs)
            ok = kwargs["apns_token"] == "tok-d1" or kwargs["push_type"] == "background"
            return SendResult(outcome="ok" if ok else "failed", status=200 if ok else 500)

    sender = _Half()
    disp = PushDispatcher(store, sender, bridge=None, node_pub=lambda: PUB)
    asyncio.run(disp.on_signal(_sig("approval_pending", _pending())))
    asyncio.run(disp.on_signal(_sig("approval_resolved", {"request_id": "c0c0a001", "resolution": "denied"})))
    assert [s["apns_token"] for s in _cleared(sender)] == ["tok-d1"]


def test_the_record_of_what_was_pushed_is_bounded():
    store, sender = _Store(["d1"]), _Sender()
    disp = PushDispatcher(store, sender, bridge=None, node_pub=lambda: PUB)
    from prometheus.push import dispatcher as mod

    for i in range(mod.PUSHED_REQUESTS_REMEMBERED + 50):
        asyncio.run(disp.on_signal(_sig("approval_pending", _pending(f"r{i}"))))
    assert len(disp._pushed_to) == mod.PUSHED_REQUESTS_REMEMBERED


def test_the_cleared_push_carries_only_reviewed_keys():
    store, sender = _Store(["d1"]), _Sender()
    disp = PushDispatcher(store, sender, bridge=None, node_pub=lambda: PUB)
    asyncio.run(disp.on_signal(_sig("approval_pending", _pending())))
    asyncio.run(disp.on_signal(_sig("approval_resolved", {
        "request_id": "c0c0a001", "resolution": "approved", "scope": "always",
        "approved_by": {"label": "owner on telegram"}})))
    (cleared,) = _cleared(sender)
    assert set(cleared["payload"]) == {"aps", "approval_resolved", "resolution", "node_pub"}


def test_a_device_reached_after_the_request_ended_is_cleared_at_once():
    """The fan-out and the answer run in different tasks: an answer while the fan-out is still sending (review,
    2026-11-12) must not leave the devices reached after it with a live Approve."""
    store = _Store(["d1", "d2"])

    class _Slow(_Sender):
        def __init__(self) -> None:
            super().__init__()
            self.disp = None

        async def send(self, **kwargs):
            self.sent.append(kwargs)
            if kwargs["apns_token"] == "tok-d1" and kwargs["push_type"] == "alert":
                # the request is answered on another surface while d2 has still to be sent to
                await self.disp.on_signal(_sig("approval_resolved", {"request_id": "c0c0a001",
                                                                     "resolution": "approved"}))
            return SendResult(outcome="ok", status=200)

    sender = _Slow()
    disp = PushDispatcher(store, sender, bridge=None, node_pub=lambda: PUB)
    sender.disp = disp
    asyncio.run(disp.on_signal(_sig("approval_pending", _pending())))
    assert sorted(s["apns_token"] for s in _cleared(sender)) == ["tok-d1", "tok-d2"]
    assert "c0c0a001" not in disp._pushed_to


def test_refused_background_pushes_never_mute_the_approval_alerts():
    store = _Store(["d1"])
    counted: list[str] = []
    store.record_push_failure = lambda device_id: counted.append(device_id) or len(counted)

    class _Throttled(_Sender):
        async def send(self, **kwargs):
            self.sent.append(kwargs)
            ok = kwargs["push_type"] == "alert"
            return SendResult(outcome="ok" if ok else "failed", status=200 if ok else 429)

    sender = _Throttled()
    disp = PushDispatcher(store, sender, bridge=None, node_pub=lambda: PUB)
    for i in range(12):
        asyncio.run(disp.on_signal(_sig("approval_pending", _pending(f"m{i}"))))
        asyncio.run(disp.on_signal(_sig("approval_resolved", {"request_id": f"m{i}", "resolution": "denied"})))
    assert counted == []


def test_server_capabilities_lists_approval_by_id():
    """The phone turns on its one-request fetch only when the daemon says it has the route (2026-11-13)."""
    from prometheus.web.ws_server import WebSocketBridge

    bridge = WebSocketBridge.__new__(WebSocketBridge)
    bridge.config = {}
    assert "approval_by_id" in bridge._server_capabilities()
