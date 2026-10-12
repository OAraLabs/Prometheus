"""When a worker ends, for any reason, its approval requests end with it.

WHAT HAPPENED
-------------
Worker aaae770e7 was stopped at its time limit, and its bash approval stayed pending for ten more minutes: still
on the owner's screens, still answerable, with nothing left to act on the answer. Nothing connected a worker's end
to the approval queue. The task manager knows nothing of the queue, and the only thing holding the request, the
daemon's waiting route handler, never noticed the worker die.

THE RULE THESE TESTS PIN
------------------------
* The moment a worker's process ends (it finished, ran out of time, was stopped, was killed, or the daemon is going
  down), every request it raised is WITHDRAWN: out of the queue at once, ``approval_resolved`` with
  ``resolution: "withdrawn"`` and the reason (so every connected client drops the card), a Telegram notice to the
  chat that was asked, and a resolution row.
* A withdrawn request resolves as a DENIAL for anything still waiting on it. There is no new ``ApprovalResult``:
  several callers check only DENIED and TIMEOUT and treat anything else as approved, so a new value would have been
  read as a yes.
* A late approve or deny acts on nothing and says so: ``410 no_longer_active``, with why (withdrawn and the reason,
  already approved, already denied, expired, or no record because the daemon restarted). ``ok: false`` stays in
  the body. A scoped device still gets the 404 for a request that is not its own conversation's.
* A daemon going down withdraws everything still pending ("the daemon is shutting down"), because no answer given
  later can reach a process that is gone.
"""

from __future__ import annotations

import asyncio
import ast
import secrets
from pathlib import Path

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
from tests.test_worker_approvals import _app, _ask, _client, _standin_worker, _until  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class _Gate:
    """The queue requires a gate reference; nothing here records a grant."""


class _Bus:
    def __init__(self) -> None:
        self.signals: list[tuple[str, dict]] = []

    async def emit(self, signal) -> None:
        self.signals.append((signal.kind, dict(signal.payload)))

    def resolved(self) -> list[dict]:
        return [p for k, p in self.signals if k == "approval_resolved"]


class _Telegram:
    def __init__(self) -> None:
        self.sent: list[tuple[object, str]] = []

    async def send(self, chat_id, text, **_kw) -> None:
        self.sent.append((chat_id, text))


def _queue(**kw) -> tuple[ApprovalQueue, _Bus, _Telegram]:
    tg = _Telegram()
    queue = ApprovalQueue(security_gate=_Gate(), telegram_adapter=tg, **kw)
    bus = _Bus()
    queue.signal_bus = bus
    return queue, bus, tg


async def _ask_queue(queue: ApprovalQueue, *, worker: str | None = None, chat=4242, session="telegram:4242"):
    """A request raised the way the worker route raises one, left waiting. Returns (waiter, request id)."""
    before = set(queue.pending)
    waiter = asyncio.create_task(queue.request_approval(
        "bash", f"Worker {worker} asks: run ls", chat_id=chat, once_only=worker is not None,
        worker_task_id=worker, session_id=session))
    await _until(lambda: set(queue.pending) - before)
    (request_id,) = set(queue.pending) - before
    return waiter, request_id


# ── the queue ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_withdrawing_a_worker_takes_out_its_requests_and_nothing_else():
    queue, _bus, _tg = _queue()
    w1, r1 = await _ask_queue(queue, worker="aaae770e7")
    w2, r2 = await _ask_queue(queue, worker="bbbb00001")
    plain, r3 = await _ask_queue(queue)
    withdrawn = await queue.withdraw_worker("aaae770e7", why="its time limit ran out")
    assert withdrawn == [r1]
    assert r1 not in queue.pending and {r2, r3} <= set(queue.pending)
    assert await asyncio.wait_for(w1, 1) == ApprovalResult.DENIED, "a withdrawal is a no to whatever waits"
    assert not w2.done() and not plain.done()
    for waiter in (w2, plain):
        waiter.cancel()


def test_there_is_no_new_approval_result_a_caller_could_read_as_yes():
    assert {r.value for r in ApprovalResult} == {"approved", "denied", "timeout"}


@pytest.mark.asyncio
async def test_every_client_is_told_the_request_is_gone_and_why():
    queue, bus, _tg = _queue()
    _w, rid = await _ask_queue(queue, worker="aaae770e7")
    await queue.withdraw_worker("aaae770e7", why="its time limit ran out")
    assert bus.resolved() == [{"request_id": rid, "resolution": "withdrawn", "reason": "its time limit ran out"}]


@pytest.mark.asyncio
async def test_the_chat_that_was_asked_is_told_it_was_withdrawn():
    queue, _bus, tg = _queue()
    _w, rid = await _ask_queue(queue, worker="aaae770e7", chat=777)
    tg.sent.clear()
    await queue.withdraw_worker("aaae770e7", why="it was stopped")
    assert len(tg.sent) == 1
    chat, text = tg.sent[0]
    assert chat == 777 and "WITHDRAWN" in text and rid in text and "it was stopped" in text
    assert "did NOT run" in text


@pytest.mark.asyncio
async def test_a_withdrawn_request_cannot_be_answered_afterwards():
    queue, bus, _tg = _queue()
    _w, rid = await _ask_queue(queue, worker="aaae770e7")
    await queue.withdraw_worker("aaae770e7", why="it finished")
    assert queue.finished(rid) == ("withdrawn", "it finished")
    count = len(bus.signals)
    assert await queue.approve(rid, by=in_process("late")) is False
    assert await queue.deny(rid, by=in_process("late")) is False
    assert len(bus.signals) == count, "a late answer announces nothing"


@pytest.mark.asyncio
async def test_a_second_answer_resolves_nothing_even_before_the_waiter_wakes():
    queue, bus, _tg = _queue()
    action = PendingAction(request_id="0a0a0001", tool_name="bash", description="run ls")
    queue._register(action)                      # no waiter: nothing pops it, as in the instant after an answer
    assert await queue.approve("0a0a0001", by=in_process("first")) is True
    count = len(bus.signals)
    assert await queue.approve("0a0a0001", by=in_process("second")) is False
    assert await queue.deny("0a0a0001", by=in_process("second")) is False
    assert action._result == ApprovalResult.APPROVED and len(bus.signals) == count


@pytest.mark.asyncio
async def test_withdrawing_twice_or_a_worker_with_nothing_pending_is_quiet():
    queue, bus, tg = _queue()
    _w, _rid = await _ask_queue(queue, worker="aaae770e7")
    await queue.withdraw_worker("aaae770e7", why="it finished")
    count, sent = len(bus.signals), len(tg.sent)
    assert await queue.withdraw_worker("aaae770e7", why="it finished") == []
    assert await queue.withdraw_worker("never-asked", why="it finished") == []
    assert (len(bus.signals), len(tg.sent)) == (count, sent)


@pytest.mark.asyncio
async def test_the_queue_remembers_how_every_request_ended():
    queue, _bus, _tg = _queue(timeout_seconds=0.2)
    _w1, approved = await _ask_queue(queue)
    _w2, denied = await _ask_queue(queue)
    w3, expired = await _ask_queue(queue)
    await queue.approve(approved, by=in_process("p"))
    await queue.deny(denied, by=in_process("p"))
    await asyncio.wait_for(w3, 2)
    assert queue.finished(approved)[0] == "approved"
    assert queue.finished(denied)[0] == "denied"
    assert queue.finished(expired)[0] == "expired"
    assert queue.finished("0" * 8) is None


@pytest.mark.asyncio
async def test_withdraw_all_empties_the_queue_with_one_reason():
    queue, bus, _tg = _queue()
    waiters = [await _ask_queue(queue, worker="aaae770e7"), await _ask_queue(queue)]
    withdrawn = await queue.withdraw_all(why="the daemon is shutting down")
    assert sorted(withdrawn) == sorted(r for _w, r in waiters) and not queue.pending
    assert {p["reason"] for p in bus.resolved()} == {"the daemon is shutting down"}
    for waiter, _r in waiters:
        assert await asyncio.wait_for(waiter, 1) == ApprovalResult.DENIED


@pytest.mark.asyncio
async def test_both_queues_are_withdrawn_from_when_computer_use_is_on():
    main, _b1, _t1 = _queue()
    desk, _b2, _t2 = _queue()
    _w1, r1 = await _ask_queue(main, worker="aaae770e7")
    _w2, r2 = await _ask_queue(desk, worker="aaae770e7")
    both = ApprovalQueues(main, desk)
    assert sorted(await both.withdraw_worker("aaae770e7", why="it was stopped")) == sorted([r1, r2])
    assert both.finished(r2) == ("withdrawn", "it was stopped")
    assert not both.pending


# ── the task manager says when a worker's process ended ─────────────────────

@pytest.mark.asyncio
async def test_stopping_a_worker_tells_the_listeners_it_was_stopped(tmp_path, monkeypatch):
    mgr, rec, _token = await _standin_worker(tmp_path, monkeypatch)
    heard: list[tuple[str, str]] = []
    mgr.add_process_end_listener(lambda task_id, why: heard.append((task_id, why)))
    try:
        await mgr.stop_task(rec.id)
        await _until(lambda: heard)
        assert heard and heard[0][0] == rec.id and "stopped" in heard[0][1], heard
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_a_worker_out_of_time_tells_the_listeners_so(tmp_path, monkeypatch):
    from prometheus.tasks import manager as mgr_mod
    from prometheus.tasks.manager import BackgroundTaskManager

    monkeypatch.setattr(mgr_mod, "prometheus_argv", lambda *a: ["/bin/sh", "-c", "sleep 30", "standin", *a])
    mgr = BackgroundTaskManager()
    heard: list[tuple[str, str]] = []

    async def listener(task_id: str, why: str) -> None:        # async listeners are awaited
        heard.append((task_id, why))

    mgr.add_process_end_listener(listener)
    try:
        rec = await mgr.create_agent_task(prompt="p", description="slow", cwd=str(tmp_path), route="local",
                                          worker=True, session_id="beacon:abc", timeout_seconds=1)
        await _until(lambda: heard, seconds=15)
        assert heard == [(rec.id, "its time limit ran out")], heard
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_a_worker_that_finishes_tells_the_listeners_so(tmp_path, monkeypatch):
    from prometheus.tasks import manager as mgr_mod
    from prometheus.tasks.manager import BackgroundTaskManager

    monkeypatch.setattr(mgr_mod, "prometheus_argv", lambda *a: ["/bin/sh", "-c", "exit 0", "standin", *a])
    mgr = BackgroundTaskManager()
    heard: list[tuple[str, str]] = []
    mgr.add_process_end_listener(lambda task_id, why: heard.append((task_id, why)))
    try:
        rec = await mgr.create_agent_task(prompt="p", description="quick", cwd=str(tmp_path), route="local",
                                          worker=True, session_id="beacon:abc")
        await _until(lambda: heard)
        assert heard == [(rec.id, "it finished")], heard
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_a_listener_that_raises_does_not_break_the_task(tmp_path, monkeypatch):
    mgr, rec, _token = await _standin_worker(tmp_path, monkeypatch)

    def broken(_task_id: str, _why: str) -> None:
        raise RuntimeError("listener exploded")

    heard: list[str] = []
    mgr.add_process_end_listener(broken)
    mgr.add_process_end_listener(lambda task_id, _why: heard.append(task_id))
    try:
        stopped = await mgr.stop_task(rec.id)
        await _until(lambda: heard)
        assert stopped.status == "killed"
        assert heard == [rec.id], "a listener after the broken one is still told"
    finally:
        await mgr.shutdown()


# ── end to end: the route, the manager and the queue ─────────────────────────

@pytest.mark.asyncio
async def test_stopping_a_worker_withdraws_its_waiting_request_at_once(tmp_path, monkeypatch):
    mgr, rec, token = await _standin_worker(tmp_path, monkeypatch)
    queue, bus, _tg = _queue()
    try:
        async with _client(_app(queue, mgr, monkeypatch)) as c:
            req = asyncio.create_task(c.post("/api/workers/approvals", **_ask(rec.id, token)))
            pending = await _until(queue.list_pending)
            assert pending
            rid = pending[0].request_id
            await mgr.stop_task(rec.id)
            assert await _until(lambda: not queue.pending, seconds=5), "the request outlived its worker"
            resp = await asyncio.wait_for(req, 5)
        assert resp.json()["result"] == "denied"
        (resolved,) = bus.resolved()
        assert resolved["request_id"] == rid and resolved["resolution"] == "withdrawn"
        assert "stopped" in resolved["reason"]
        assert queue.finished(rid)[0] == "withdrawn"
    finally:
        await mgr.shutdown()


@pytest.mark.asyncio
async def test_a_request_from_a_worker_that_ended_while_it_was_being_read_is_never_filed(tmp_path, monkeypatch):
    """The body is read after the token check; a worker that ended meanwhile must not leave a request behind."""
    mgr, rec, token = await _standin_worker(tmp_path, monkeypatch)
    queue, _bus, _tg = _queue()
    try:
        app = _app(queue, mgr, monkeypatch)
        original = mgr.verify_worker_token
        calls = {"n": 0}

        def verify_then_end(task_id, presented):
            calls["n"] += 1
            found = original(task_id, presented)
            if calls["n"] == 1 and found is not None:   # the first check passes; the worker dies right after it
                mgr._tasks[rec.id].status = "killed"
            return found

        monkeypatch.setattr(mgr, "verify_worker_token", verify_then_end)
        async with _client(app) as c:
            resp = await asyncio.wait_for(c.post("/api/workers/approvals", **_ask(rec.id, token)), 5)
        assert resp.status_code == 401 and not queue.pending, resp.text
    finally:
        await mgr.shutdown()


# ── a late answer over REST ──────────────────────────────────────────────────

GLOBAL = "withdrawn-global-" + secrets.token_hex(8)


class Daemon:
    def __init__(self, tmp_path) -> None:
        self.devices = DeviceStore(tmp_path / "devices.db")
        from prometheus.web.server import create_app

        self.app = create_app({"web": {"api_token": GLOBAL}}, device_store=self.devices)
        self.queue, self.bus, _tg = _queue()
        self.app.state.approval_queue = self.queue
        self.owner = self.devices.mint_owner("Beacon on this Mac", "macos", by="same-mac-pairing")
        self.alice = self.devices.mint("alice's phone", "ios")
        self.bob = self.devices.mint("bob's phone", "ios")
        self.devices.claim_session("web:alice-1", self.alice["id"])
        self.client = TestClient(self.app, client=("127.0.0.1", 50123))

    def answer(self, who: str, verb: str, request_id: str):
        token = {"global": GLOBAL, "owner": self.owner["token"], "alice": self.alice["token"],
                 "bob": self.bob["token"]}[who]
        return self.client.post(f"/api/approvals/{request_id}/{verb}", headers={"Authorization": f"Bearer {token}"},
                                json={"scope": "once"} if verb == "approve" else None)

    def worker_request(self, request_id: str, worker: str = "aaae770e7") -> PendingAction:
        action = PendingAction(request_id=request_id, tool_name="bash", description="run ls",
                               session_id="web:alice-1", once_only=True, worker_task_id=worker)
        self.queue._register(action)
        return action


@pytest.fixture
def d(tmp_path) -> Daemon:
    return Daemon(tmp_path)


def _gone(response, *, says: str) -> bool:
    body = response.json()
    return (response.status_code == 410 and body.get("error") == "no_longer_active" and body.get("ok") is False
            and says in body.get("detail", ""))


@pytest.mark.asyncio
@pytest.mark.parametrize("verb", ["approve", "deny"])
async def test_a_late_answer_to_a_withdrawn_request_is_410_and_acts_on_nothing(d, verb):
    action = d.worker_request("f0f0a001")
    await d.queue.withdraw_worker("aaae770e7", why="its time limit ran out")
    count = len(d.bus.signals)
    r = d.answer("owner", verb, "f0f0a001")
    assert _gone(r, says="its time limit ran out"), r.text
    assert r.json()["resolution"] == "withdrawn"
    assert action._result == ApprovalResult.DENIED and len(d.bus.signals) == count


@pytest.mark.asyncio
async def test_the_device_whose_conversation_it_was_is_told_410_too(d):
    d.worker_request("f0f0a002")
    await d.queue.withdraw_worker("aaae770e7", why="it was stopped")
    assert _gone(d.answer("alice", "approve", "f0f0a002"), says="it was stopped")


@pytest.mark.asyncio
async def test_another_device_still_gets_the_404_it_always_did(d):
    d.worker_request("f0f0a003")
    await d.queue.withdraw_worker("aaae770e7", why="it was stopped")
    r = d.answer("bob", "approve", "f0f0a003")
    assert r.status_code == 404, r.text


def test_an_already_answered_request_is_410_and_says_how_it_ended(d):
    d.worker_request("f0f0a004")
    assert d.answer("owner", "approve", "f0f0a004").status_code == 200
    assert _gone(d.answer("owner", "approve", "f0f0a004"), says="already approved")
    d.worker_request("f0f0a005")
    assert d.answer("owner", "deny", "f0f0a005").status_code == 200
    assert _gone(d.answer("owner", "deny", "f0f0a005"), says="already denied")


def test_a_request_this_daemon_has_no_record_of_is_410_and_says_it_may_have_restarted(d):
    r = d.answer("global", "deny", "abcdef12")
    assert _gone(r, says="restarted"), r.text
    assert r.json()["resolution"] is None


def test_approve_all_is_not_a_request_id(d):
    r = d.answer("global", "approve", "all")
    assert r.status_code == 200, r.text


# ── the daemon going down ────────────────────────────────────────────────────

def test_the_daemon_withdraws_everything_pending_when_it_shuts_down():
    """Pinned in the source: the graceful shutdown asks every approval queue to withdraw before Telegram stops."""
    text = (ROOT / "src" / "prometheus" / "daemon.py").read_text()
    shutdown = text[text.index('logger.info("Shutting down...")'):]
    assert "withdraw_pending_for_shutdown(" in shutdown
    assert shutdown.index("withdraw_pending_for_shutdown(") < shutdown.index("await telegram.stop()")


@pytest.mark.asyncio
async def test_withdraw_pending_for_shutdown_reaches_every_queue_and_never_raises():
    from prometheus.permissions.approval_queue import withdraw_pending_for_shutdown

    main, bus, _tg = _queue()
    _w, rid = await _ask_queue(main, worker="aaae770e7")

    class Broken:
        async def withdraw_all(self, *, why):
            raise RuntimeError("no")

    await withdraw_pending_for_shutdown([main, None, Broken()])
    assert not main.pending and bus.resolved()[0]["reason"] == "the daemon is shutting down"
    assert main.finished(rid)[0] == "withdrawn"


def test_approve_action_still_holds_no_membership_test():
    """tests/test_api_approve_reports_what_it_did.py pins this; the 410 lives in a helper."""
    tree = ast.parse((ROOT / "src" / "prometheus" / "web" / "server.py").read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "approve_action")
    assert not any(isinstance(op, (ast.In, ast.NotIn)) for node in ast.walk(fn)
                   if isinstance(node, ast.Compare) for op in node.ops)
