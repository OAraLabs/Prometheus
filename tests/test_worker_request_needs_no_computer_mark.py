"""A worker's request is answered by an owner device, whichever queue holds it, and never by the API token.

SEEN (2026-11-26): a worker's write_file request, approved from the owner Beacon on the same Mac (``oara pair
--owner``), answered ``200 {"ok": false, "message": "Not approved: Only a person may do this: this device is not
marked for computer use ..."}``.

THE PATH: ``security.approval_queue.enabled`` is false (how Prometheus ships), so the gate-wide queue does not exist and
the web surface is ``ApprovalQueues(hold_lifts)`` (the hold-lift channel is always built). ``/api/workers/approvals``
filed the request in ``queues[0]``: the hold-lift channel. ``HoldLiftChannel.approve`` (and the desktop channel's)
ran the computer-use person check on EVERY request in it, a worker's included; an owner device is not marked for
computer use, so it was refused. A worker's request is not a hold lift or a desktop prompt: it is answered under its
own rules (a device token, never the API token: ``approve_action``).
"""

from __future__ import annotations

import asyncio
import secrets

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from prometheus.computer.approvals import ComputerApprovalChannel  # noqa: E402
from prometheus.computer.door import PersonCheck  # noqa: E402
from prometheus.config.device_store import DeviceStore  # noqa: E402
from prometheus.permissions.approval_queue import (  # noqa: E402
    ApprovalQueue,
    ApprovalQueues,
    ApprovalResult,
    PendingAction,
    worker_queue,
)
from prometheus.permissions.hold_lifts import HoldLiftChannel  # noqa: E402

GLOBAL = "worker-mark-" + secrets.token_hex(8)


class _Gate:
    def list_grants(self):
        return []


class World:
    def __init__(self, tmp_path, shape: str) -> None:
        self.devices = DeviceStore(tmp_path / "devices.db")
        from prometheus.web.server import create_app

        self.app = create_app({"web": {"api_token": GLOBAL}}, device_store=self.devices)
        people = PersonCheck(device_store=self.devices)
        self.primary = ApprovalQueue(security_gate=_Gate(), telegram_adapter=None)
        self.channel = ComputerApprovalChannel(security_gate=_Gate(), people=people, telegram_adapter=None)
        self.hold = HoldLiftChannel(people=people, devices=lambda: self.devices, security_gate=_Gate())
        # The shapes the daemon builds: the gate-wide queue is off by default; the hold-lift channel always exists;
        # computer use adds its channel; the daemon nests them.
        self.surface = {
            "default": ApprovalQueues(self.hold),
            "computer": ApprovalQueues(ApprovalQueues(self.channel), self.hold),
            "queue_on": ApprovalQueues(self.primary, self.hold),
            "queue_and_computer": ApprovalQueues(ApprovalQueues(self.primary, self.channel), self.hold),
        }[shape]
        self.app.state.approval_queue = self.surface
        self.owner = self.devices.mint_owner("Beacon on this Mac", "macos", by="same-mac-pairing")
        self.client = TestClient(self.app, client=("127.0.0.1", 50123))

    def ask(self):
        """File a worker's request the way ``/api/workers/approvals`` does."""
        q = worker_queue(self.surface)
        return asyncio.ensure_future(q.request_approval(
            "write_file", "Worker a053720bf asks: write outside the workspace", worker_task_id="a053720bf",
            once_only=True, session_id="beacon:abc"))

    def answer(self, verb: str, rid: str, token: str):
        return self.client.post(f"/api/approvals/{rid}/{verb}", headers={"Authorization": f"Bearer {token}"},
                                json={"scope": "once"} if verb == "approve" else None)


SHAPES = ["default", "computer", "queue_on", "queue_and_computer"]


async def _rid(w: World) -> str:
    for _ in range(300):
        pending = w.surface.pending
        if pending:
            return next(iter(pending))
        await asyncio.sleep(0.01)
    raise AssertionError("the request never became pending")


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", SHAPES)
async def test_an_owner_device_approves_a_worker_request_without_a_computer_mark(tmp_path, shape):
    w = World(tmp_path, shape)
    assert not w.devices.computer_allowed(w.owner["id"])
    task = w.ask()
    rid = await _rid(w)
    r = w.answer("approve", rid, w.owner["token"])
    assert r.status_code == 200 and r.json()["ok"] is True, r.text
    assert (await asyncio.wait_for(task, 5)) == ApprovalResult.APPROVED


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", SHAPES)
async def test_an_owner_device_denies_a_worker_request(tmp_path, shape):
    w = World(tmp_path, shape)
    task = w.ask()
    rid = await _rid(w)
    r = w.answer("deny", rid, w.owner["token"])
    assert r.status_code == 200 and r.json()["ok"] is True, r.text
    assert (await asyncio.wait_for(task, 5)) == ApprovalResult.DENIED


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", SHAPES)
async def test_the_api_token_still_cannot_approve_a_worker_request(tmp_path, shape):
    w = World(tmp_path, shape)
    task = w.ask()
    rid = await _rid(w)
    r = w.answer("approve", rid, GLOBAL)
    assert r.status_code == 401, r.text
    assert rid in w.surface.pending and not task.done()
    w.answer("deny", rid, w.owner["token"])
    await asyncio.wait_for(task, 5)


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", SHAPES)
async def test_a_chat_answer_that_is_not_the_owners_still_cannot_approve_it(tmp_path, shape):
    from prometheus.permissions import approver

    w = World(tmp_path, shape)
    task = w.ask()
    rid = await _rid(w)
    stranger = approver.Approver(kind=approver.TELEGRAM, id="999", owner=False)
    assert await w.surface.approve(rid, by=stranger) is False
    assert not task.done()
    await w.surface.withdraw_all(why="test over")
    await asyncio.wait_for(task, 5)


# ── which queue a worker's request goes to ───────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(("shape", "which"), [
    ("default", "hold"), ("computer", "channel"), ("queue_on", "primary"), ("queue_and_computer", "primary"),
])
def test_a_workers_request_goes_to_the_gate_wide_queue_when_there_is_one(tmp_path, shape, which):
    w = World(tmp_path, shape)
    assert worker_queue(w.surface) is {"hold": w.hold, "channel": w.channel, "primary": w.primary}[which]


def test_no_queue_at_all_is_none():
    assert worker_queue(None) is None
    assert worker_queue(ApprovalQueues(ApprovalQueues(ApprovalQueue(security_gate=_Gate()))).queues[0]) is not None


# ── the channels' own prompts are still a person's ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_hold_lift_still_needs_a_person_marked_for_it(tmp_path):
    from prometheus.computer.door import NotAPerson
    from prometheus.permissions import approver

    w = World(tmp_path, "default")
    action = PendingAction(request_id="1011ca7e", tool_name="device_hold", description="lift", hold_device_id="dev1",
                           once_only=True)
    w.hold._register(action)
    by = approver.Approver(kind=approver.DEVICE, id=w.owner["id"])
    with pytest.raises(NotAPerson):
        await w.hold.approve("1011ca7e", by=by)
    assert "1011ca7e" in w.hold.pending


@pytest.mark.asyncio
async def test_a_desktop_prompt_still_needs_a_person_marked_for_it(tmp_path):
    from prometheus.computer.door import NotAPerson
    from prometheus.permissions import approver

    w = World(tmp_path, "computer")
    action = PendingAction(request_id="de5c70b1", tool_name="computer_click", description="click", task_id="t1",
                           once_only=True)
    w.channel._register(action)
    by = approver.Approver(kind=approver.DEVICE, id=w.owner["id"])
    with pytest.raises(NotAPerson):
        await w.channel.approve("de5c70b1", by=by)
    assert "de5c70b1" in w.channel.pending


# ── review #2 follow-up: the skip is POSITIVE worker marking, fail closed ────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_desktop_prompt_missing_its_task_id_still_needs_a_person(tmp_path):
    """The fail-open the review found: a desktop prompt filed WITHOUT its
    task_id is not positively a worker request, so the person check must
    still run — a missing marker fails CLOSED."""
    from prometheus.computer.door import NotAPerson
    from prometheus.permissions import approver

    w = World(tmp_path, "computer")
    action = PendingAction(request_id="fa110pen", tool_name="computer_click",
                           description="click", once_only=True)  # no task_id, no worker_task_id
    w.channel._register(action)
    by = approver.Approver(kind=approver.DEVICE, id=w.owner["id"])
    with pytest.raises(NotAPerson):
        await w.channel.approve("fa110pen", by=by)
    assert "fa110pen" in w.channel.pending


@pytest.mark.asyncio
async def test_a_hold_lift_missing_its_device_marker_still_needs_a_person(tmp_path):
    from prometheus.computer.door import NotAPerson
    from prometheus.permissions import approver

    w = World(tmp_path, "default")
    action = PendingAction(request_id="fa110pen", tool_name="device_hold",
                           description="lift", once_only=True)  # no hold_device_id
    w.hold._register(action)
    by = approver.Approver(kind=approver.DEVICE, id=w.owner["id"])
    with pytest.raises(NotAPerson):
        await w.hold.approve("fa110pen", by=by)
    assert "fa110pen" in w.hold.pending


@pytest.mark.asyncio
async def test_an_unmarked_ordinary_request_in_a_channel_still_needs_a_person(tmp_path):
    """Anything not POSITIVELY a worker request keeps the check — the skip
    is a whitelist of one marker, not a blacklist of two."""
    from prometheus.computer.door import NotAPerson
    from prometheus.permissions import approver

    w = World(tmp_path, "computer")
    action = PendingAction(request_id="0rd1nary", tool_name="write_file",
                           description="write", once_only=True, session_id="beacon:abc")
    w.channel._register(action)
    by = approver.Approver(kind=approver.DEVICE, id=w.owner["id"])
    with pytest.raises(NotAPerson):
        await w.channel.approve("0rd1nary", by=by)


@pytest.mark.asyncio
async def test_worker_marker_plus_desktop_marker_keeps_the_person_check(tmp_path):
    """Review #2 hardening nit: BOTH markers → the person check stays.
    The skip is for worker-ONLY requests."""
    from prometheus.computer.door import NotAPerson
    from prometheus.permissions import approver

    w = World(tmp_path, "computer")
    action = PendingAction(request_id="b0thmark", tool_name="computer_click",
                           description="click", task_id="t1", worker_task_id="a053720bf",
                           once_only=True)
    w.channel._register(action)
    by = approver.Approver(kind=approver.DEVICE, id=w.owner["id"])
    with pytest.raises(NotAPerson):
        await w.channel.approve("b0thmark", by=by)
    assert "b0thmark" in w.channel.pending


@pytest.mark.asyncio
async def test_worker_marker_plus_hold_marker_keeps_the_person_check(tmp_path):
    from prometheus.computer.door import NotAPerson
    from prometheus.permissions import approver

    w = World(tmp_path, "default")
    action = PendingAction(request_id="b0thmark", tool_name="device_hold",
                           description="lift", hold_device_id="dev1", worker_task_id="a053720bf",
                           once_only=True)
    w.hold._register(action)
    by = approver.Approver(kind=approver.DEVICE, id=w.owner["id"])
    with pytest.raises(NotAPerson):
        await w.hold.approve("b0thmark", by=by)
    assert "b0thmark" in w.hold.pending
