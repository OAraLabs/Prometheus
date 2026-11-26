"""A request that has an answer is not pending, and a request whose asker is gone is not left behind.

SEEN (2026-11-25): a worker's request (write_file, outside the workspace, arguments cut, critical) was approved from
Beacon 13 times. GET /api/approvals kept listing the same id until its own expiry, and every approve answered.

THE CAUSE: ``request_approval`` registered the request, then awaited (the ``approval_pending`` signal, the chat
notice) BEFORE it entered the ``try`` whose ``finally`` pops it. ``_emit`` swallows ``Exception`` but not
``CancelledError``, so an asker cancelled in that window (the worker's HTTP call dropped, its process ended) left
the entry in ``pending`` with nothing waiting on it and nothing that would ever pop or withdraw it.
"""

from __future__ import annotations

import asyncio
import secrets

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from prometheus.config.device_store import DeviceStore  # noqa: E402
from prometheus.permissions.approval_queue import (  # noqa: E402
    ApprovalQueue,
    ApprovalQueues,
    ApprovalResult,
    PendingAction,
)
from prometheus.permissions.approver import in_process  # noqa: E402

GLOBAL = "finished-listed-" + secrets.token_hex(8)


class _Gate:
    def list_grants(self):
        return []


class _Bus:
    """Records every signal; can hold one kind open so a test can cancel the asker inside that await."""

    def __init__(self) -> None:
        self.signals: list = []
        self.hold: str | None = None
        self.held = asyncio.Event()

    async def emit(self, signal) -> None:
        if signal.kind == self.hold:
            self.held.set()
            await asyncio.Event().wait()
        self.signals.append(signal)

    def kinds(self) -> list[str]:
        return [s.kind for s in self.signals]


class Daemon:
    def __init__(self, tmp_path) -> None:
        self.devices = DeviceStore(tmp_path / "devices.db")
        from prometheus.web.server import create_app

        self.app = create_app({"web": {"api_token": GLOBAL}}, device_store=self.devices)
        self.queue = ApprovalQueue(security_gate=_Gate(), telegram_adapter=None)
        self.bus = _Bus()
        self.queue.signal_bus = self.bus
        self.app.state.approval_queue = self.queue
        self.owner = self.devices.mint_owner("Beacon on this Mac", "macos", by="same-mac-pairing")
        self.client = TestClient(self.app, client=("127.0.0.1", 50123))

    def call(self, verb: str, request_id: str, who: str = "owner"):
        token = self.owner["token"] if who == "owner" else GLOBAL
        return self.client.post(f"/api/approvals/{request_id}/{verb}", headers={"Authorization": f"Bearer {token}"},
                                json={"scope": "once"} if verb == "approve" else None)

    def listed(self) -> list[str]:
        r = self.client.get("/api/approvals", headers={"Authorization": f"Bearer {self.owner['token']}"})
        assert r.status_code == 200, r.text
        return [a["request_id"] for a in r.json()]

    def ask(self, **kw):
        kw.setdefault("worker_task_id", "a505dea5c")
        kw.setdefault("once_only", True)
        kw.setdefault("session_id", "beacon:abc")
        return asyncio.ensure_future(self.queue.request_approval("write_file", "Worker a505dea5c asks", **kw))


@pytest.fixture
def d(tmp_path) -> Daemon:
    return Daemon(tmp_path)


async def _the_id(d: Daemon) -> str:
    for _ in range(200):
        if d.queue.pending:
            return next(iter(d.queue.pending))
        await asyncio.sleep(0.01)
    raise AssertionError("the request never became pending")


# ── the cause: an asker cancelled before it waits ────────────────────────────

@pytest.mark.asyncio
async def test_an_asker_cancelled_before_it_waits_leaves_nothing_pending(d):
    d.bus.hold = "approval_pending"
    task = d.ask()
    rid = await _the_id(d)
    await asyncio.wait_for(d.bus.held.wait(), 5)
    d.bus.hold = None
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert d.queue.pending == {}, "the entry stayed pending with nothing waiting on it"
    assert d.queue.list_pending() == []
    assert d.listed() == []
    assert d.queue.finished(rid) == ("withdrawn", "the turn that asked ended")
    assert "approval_resolved" in d.bus.kinds()


@pytest.mark.asyncio
async def test_an_asker_that_fails_before_it_waits_leaves_nothing_pending(d, monkeypatch):
    def boom(self, action):
        raise RuntimeError("serialising failed")

    monkeypatch.setattr(ApprovalQueue, "serialize_pending", boom)
    with pytest.raises(RuntimeError):
        await d.queue.request_approval("write_file", "x", worker_task_id="a505dea5c", once_only=True)
    assert d.queue.pending == {}


@pytest.mark.asyncio
async def test_a_late_answer_to_a_withdrawn_request_is_the_finished_answer(d):
    d.bus.hold = "approval_pending"
    task = d.ask()
    rid = await _the_id(d)
    await asyncio.wait_for(d.bus.held.wait(), 5)
    d.bus.hold = None
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    for verb in ("approve", "deny"):
        r = d.call(verb, rid)
        assert r.status_code == 410, r.text
        assert r.json()["resolution"] == "withdrawn" and r.json()["ok"] is False


# ── an answered request is not pending, whether or not its waiter has popped it ───────────────────────────────────

def _stale(d: Daemon, rid: str = "c0ffee01") -> PendingAction:
    """A request that is registered and answered, with NO waiter to pop it: the state the 13 approvals met."""
    action = PendingAction(request_id=rid, tool_name="write_file", description="Worker a505dea5c asks",
                           session_id="beacon:abc", once_only=True, worker_task_id="a505dea5c")
    d.queue._register(action)
    return action


@pytest.mark.asyncio
@pytest.mark.parametrize("verb", ["approve", "deny"])
async def test_an_answered_request_is_not_listed_and_a_second_answer_is_410(d, verb):
    _stale(d)
    assert d.listed() == ["c0ffee01"]
    first = d.call(verb, "c0ffee01")
    assert first.status_code == 200 and first.json()["ok"] is True, first.text
    assert d.listed() == []
    assert d.queue.list_pending() == []
    for _ in range(3):
        again = d.call(verb, "c0ffee01")
        assert again.status_code == 410, again.text
        want = "approved" if verb == "approve" else "denied"
        assert again.json()["resolution"] == want and again.json()["ok"] is False


@pytest.mark.asyncio
async def test_list_pending_skips_an_entry_that_has_an_answer_even_if_it_is_still_in_pending(d):
    action = _stale(d)
    d.queue._remember_end(action.request_id, "approved")          # answered, entry not yet popped
    assert d.queue.list_pending() == []
    assert ApprovalQueues(d.queue).list_pending() == []


@pytest.mark.asyncio
async def test_a_timed_out_request_is_not_listed_and_is_410(d):
    d.queue._timeout = 0.05
    task = d.ask()
    rid = await _the_id(d)
    assert (await task) == ApprovalResult.TIMEOUT
    assert d.queue.pending == {} and d.listed() == []
    r = d.call("approve", rid)
    assert r.status_code == 410 and r.json()["resolution"] == "expired", r.text


@pytest.mark.asyncio
async def test_a_withdrawn_worker_request_is_not_listed_and_is_410(d):
    task = d.ask()
    rid = await _the_id(d)
    assert await d.queue.withdraw_worker("a505dea5c", why="the worker ended") == [rid]
    assert (await task) == ApprovalResult.DENIED
    assert d.queue.pending == {} and d.listed() == []
    r = d.call("deny", rid)
    assert r.status_code == 410 and r.json()["resolution"] == "withdrawn", r.text


# ── regressions: nothing else moved ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_an_unanswered_request_is_still_listed_and_a_normal_approve_wakes_the_waiter(d):
    task = d.ask()
    rid = await _the_id(d)
    assert d.listed() == [rid]
    r = d.call("approve", rid)
    assert r.status_code == 200 and r.json()["ok"] is True, r.text
    assert (await asyncio.wait_for(task, 5)) == ApprovalResult.APPROVED
    assert d.queue.pending == {} and d.listed() == []
    assert d.call("approve", rid).status_code == 410


@pytest.mark.asyncio
async def test_a_worker_request_is_still_refused_to_the_api_token_and_stays_listed(d):
    task = d.ask()
    rid = await _the_id(d)
    r = d.call("approve", rid, who="global")
    assert r.status_code == 401, r.text
    assert d.listed() == [rid]
    d.call("deny", rid)
    assert (await asyncio.wait_for(task, 5)) == ApprovalResult.DENIED


@pytest.mark.asyncio
async def test_an_answer_is_not_overwritten_by_an_asker_cancelled_after_it(d):
    task = d.ask()
    rid = await _the_id(d)
    assert await d.queue.approve(rid, by=in_process("test")) is True
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert d.queue.finished(rid) == ("approved", "")
    assert d.queue.pending == {}, "answered, then the asker ended before it could pop: nothing may be left"


@pytest.mark.asyncio
async def test_a_denied_request_whose_asker_ends_before_it_pops_is_not_left_behind(d):
    task = d.ask()
    rid = await _the_id(d)
    assert await d.queue.deny(rid, by=in_process("test")) is True
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert d.queue.finished(rid) == ("denied", "")
    assert d.queue.pending == {} and d.listed() == []


@pytest.mark.asyncio
async def test_a_bare_chat_approve_does_not_pick_an_answered_request(d):
    from prometheus.gateway.commands import _pending_actions

    action = _stale(d, "c0ffee03")
    d.queue._remember_end(action.request_id, "approved")
    assert _pending_actions(d.queue) == []
