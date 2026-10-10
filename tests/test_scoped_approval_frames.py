"""``approval_pending`` / ``approval_resolved`` frames reach a scoped socket only for ITS OWN sessions' requests.

The REST list (``GET /api/approvals``) was the visible half of the hole in an independent review; the
WebSocket was the other. The bridge treats a frame that names no session as daemon-level and sends it to
everyone, and an ordinary approval request's frame names none, so every connected scoped device received every
session's pending tool call (its tool, its description, its extents, its redacted arguments) the moment it was
raised.

The frame stays exactly as it was, because Beacon renders it. What changes is who is sent it: the bridge asks
the approval queue where the request came from (``origin_of``, a bounded log that outlives the pending entry so a
``approval_resolved`` is routed the same way) and sends it to the operator, to the device that owns the session
that raised it, and, for a desktop prompt, to a device an operator marked for computer use. A request the queue
does not know is withheld from a scoped socket: unknown is not "daemon-level".
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")

from prometheus.config.api_token import GLOBAL_IDENTITY, DeviceIdentity  # noqa: E402
from prometheus.permissions.approval_queue import ApprovalQueue, PendingAction  # noqa: E402
from prometheus.permissions.checker import PermissionMode, SecurityGate  # noqa: E402
from tests.support.device_world import World  # noqa: E402


class Sock:
    def __init__(self) -> None:
        self.frames: list[dict] = []

    async def send(self, raw: str) -> None:
        import json

        self.frames.append(json.loads(raw))

    def ids(self, kind: str) -> list[str]:
        return [f["payload"]["request_id"] for f in self.frames if f["type"] == kind]


@pytest.fixture
def w(tmp_path) -> World:
    world = World(tmp_path)
    world.queue = ApprovalQueue(security_gate=SecurityGate(mode=PermissionMode.DEFAULT, audit_logger=None))
    world.bridge.approval_queue = world.queue
    world.sid_a = world.device_session("a")
    world.sid_b = world.device_session("b")
    return world


def connect(w: World, who: str) -> Sock:
    ws = Sock()
    w.bridge._clients.add(ws)
    w.bridge._ws_identity[ws] = {
        "op": GLOBAL_IDENTITY,
        "a": DeviceIdentity(id=w.a["id"], name="device-a", platform="ios"),
        "b": DeviceIdentity(id=w.b["id"], name="device-b", platform="macos"),
    }[who]
    return ws


def raise_request(w: World, request_id: str, *, session: str | None, task: str | None = None) -> PendingAction:
    action = PendingAction(request_id=request_id, tool_name="bash", description=f"run {request_id}",
                           session_id=session, task_id=task)
    w.queue._register(action)
    return action


async def announce(w: World, action: PendingAction, kind: str = "approval_pending") -> None:
    payload = (w.queue.serialize_pending(action) if kind == "approval_pending"
               else {"request_id": action.request_id, "resolution": "approved"})
    await w.bridge.broadcast({"type": kind, "timestamp": time.time(), "payload": payload})


@pytest.mark.asyncio
async def test_a_scoped_socket_is_sent_only_its_own_sessions_requests(w):
    op, a, b = connect(w, "op"), connect(w, "a"), connect(w, "b")
    for action in (raise_request(w, "aaaa0001", session=w.sid_a),
                   raise_request(w, "bbbb0001", session=w.sid_b),
                   raise_request(w, "cccc0001", session=None),
                   raise_request(w, "dddd0001", session="telegram:42")):
        await announce(w, action)
    assert op.ids("approval_pending") == ["aaaa0001", "bbbb0001", "cccc0001", "dddd0001"]
    assert a.ids("approval_pending") == ["aaaa0001"]
    assert b.ids("approval_pending") == ["bbbb0001"]


@pytest.mark.asyncio
async def test_the_resolution_is_routed_like_the_request_even_after_it_left_the_queue(w):
    action = raise_request(w, "aaaa0001", session=w.sid_a)
    w.queue.pending.pop("aaaa0001")                               # answered: no longer pending
    a, b, op = connect(w, "a"), connect(w, "b"), connect(w, "op")
    await announce(w, action, "approval_resolved")
    assert a.ids("approval_resolved") == ["aaaa0001"]
    assert b.ids("approval_resolved") == [] and op.ids("approval_resolved") == ["aaaa0001"]


@pytest.mark.asyncio
async def test_a_request_the_queue_never_heard_of_is_withheld_from_a_scoped_socket(w):
    a, op = connect(w, "a"), connect(w, "op")
    await w.bridge.broadcast({"type": "approval_pending", "timestamp": time.time(),
                              "payload": {"request_id": "ffff0001", "tool_name": "bash"}})
    assert a.frames == [] and len(op.frames) == 1, "unknown is not 'daemon-level'"


@pytest.mark.asyncio
async def test_with_no_queue_a_scoped_socket_is_sent_none(w):
    w.bridge.approval_queue = None
    a, op = connect(w, "a"), connect(w, "op")
    await w.bridge.broadcast({"type": "approval_pending", "timestamp": time.time(),
                              "payload": {"request_id": "aaaa0001"}})
    assert a.frames == [] and len(op.frames) == 1


@pytest.mark.asyncio
async def test_a_frame_that_names_its_session_is_decided_by_that_and_needs_no_queue(w):
    """The generic session rule (#692) already routes it; the queue's log is only for frames that name nothing."""
    w.bridge.approval_queue = None
    a, b, op = connect(w, "a"), connect(w, "b"), connect(w, "op")
    await w.bridge.broadcast({"type": "approval_pending", "timestamp": time.time(),
                              "payload": {"request_id": "r1", "session_id": w.sid_a}})
    assert a.ids("approval_pending") == ["r1"] and b.frames == [] and op.ids("approval_pending") == ["r1"]


@pytest.mark.asyncio
async def test_a_desktop_prompt_goes_to_its_sessions_owner_alone_marked_or_not(w):
    """It names its session, so the generic session filter (#692) has always done this; the mark decides who may
    ANSWER a desktop prompt (REST), not who is told."""
    action = raise_request(w, "eeee0001", session=w.sid_b, task="t1")
    a, b = connect(w, "a"), connect(w, "b")
    await announce(w, action)
    assert a.ids("approval_pending") == [] and b.ids("approval_pending") == ["eeee0001"], "b owns the session"
    assert w.devices.set_computer(w.a["id"], True, by="test")
    await announce(w, action)
    assert a.ids("approval_pending") == [], "a mark does not widen who is told"
    assert b.ids("approval_pending") == ["eeee0001", "eeee0001"]


@pytest.mark.asyncio
async def test_a_mark_does_not_show_a_marked_device_ordinary_tool_calls_of_other_sessions(w):
    assert w.devices.set_computer(w.a["id"], True, by="test")
    a = connect(w, "a")
    await announce(w, raise_request(w, "bbbb0001", session=w.sid_b))
    assert a.frames == []


@pytest.mark.asyncio
async def test_other_frames_are_untouched(w):
    a = connect(w, "a")
    await w.bridge.broadcast({"type": "dream_start", "timestamp": time.time(), "payload": {"note": "daemon level"}})
    assert [f["type"] for f in a.frames] == ["dream_start"]


# ── the log that makes it possible ───────────────────────────────────────────

def test_the_queue_remembers_where_a_request_came_from_and_outlives_the_pending_entry():
    queue = ApprovalQueue(security_gate=SecurityGate(mode=PermissionMode.DEFAULT, audit_logger=None))
    queue._register(PendingAction(request_id="aaaa0001", tool_name="bash", description="x", session_id="web:s"))
    queue._register(PendingAction(request_id="bbbb0001", tool_name="computer", description="y",
                                  session_id="web:t", task_id="t1"))
    queue.pending.clear()
    assert queue.origin_of("aaaa0001") == ("web:s", False)
    assert queue.origin_of("bbbb0001") == ("web:t", True)
    assert queue.origin_of("nope") is None


def test_the_log_is_bounded():
    queue = ApprovalQueue(security_gate=SecurityGate(mode=PermissionMode.DEFAULT, audit_logger=None))
    for number in range(queue.ORIGINS_REMEMBERED + 20):
        queue._register(PendingAction(request_id=f"{number:08x}", tool_name="bash", description="x"))
    assert queue.origin_of(f"{0:08x}") is None
    assert queue.origin_of(f"{queue.ORIGINS_REMEMBERED + 19:08x}") == (None, False)


@pytest.mark.asyncio
async def test_an_ordinary_request_is_logged_with_the_run_it_was_raised_in():
    import asyncio

    from prometheus.engine import tool_context

    queue = ApprovalQueue(security_gate=SecurityGate(mode=PermissionMode.DEFAULT, audit_logger=None))
    token = tool_context.RUN_SESSION.set("web:alice-1")
    try:
        task = asyncio.ensure_future(queue.request_approval("bash", "run echo hi"))
        await asyncio.sleep(0)
    finally:
        tool_context.RUN_SESSION.reset(token)
    [pending] = queue.list_pending()
    assert queue.origin_of(pending.request_id) == ("web:alice-1", False)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_a_desktop_prompt_is_logged_by_the_desktop_channel():
    import asyncio

    from prometheus.computer.approvals import ComputerApprovalChannel

    channel = ComputerApprovalChannel(
        security_gate=SecurityGate(mode=PermissionMode.DEFAULT, audit_logger=None), people=object())
    task = asyncio.ensure_future(channel.request_for_task(
        tool_name="computer", description="click Send", extent=None, arguments=None,
        task_id="t1", session_id="web:abc"))
    await asyncio.sleep(0)
    [pending] = channel.list_pending()
    assert channel.origin_of(pending.request_id) == ("web:abc", True)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_the_composite_queue_asks_each_queue():
    from prometheus.permissions.approval_queue import ApprovalQueues

    one = ApprovalQueue(security_gate=SecurityGate(mode=PermissionMode.DEFAULT, audit_logger=None))
    two = ApprovalQueue(security_gate=SecurityGate(mode=PermissionMode.DEFAULT, audit_logger=None))
    one._register(PendingAction(request_id="aaaa0001", tool_name="bash", description="x", session_id="web:one"))
    two._register(PendingAction(request_id="bbbb0001", tool_name="bash", description="y", session_id="web:two"))
    both = ApprovalQueues(one, two)
    assert both.origin_of("aaaa0001") == ("web:one", False) and both.origin_of("bbbb0001") == ("web:two", False)
    assert both.origin_of("zzzz0001") is None

