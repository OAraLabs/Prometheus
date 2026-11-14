"""Approving a CRITICAL request needs the request's digest (Beacon iOS item 3, Will's suggestion, 2026-11-13).

The ``critical`` flag (#917) keeps a request off the lock screen: its push has no Approve button. The daemon now
enforces part of that itself. Every served request carries ``digest``, a sha256 over exactly what the client is
shown (request id, tool, description, arguments view, created and expiry times, the critical flag). It is served
wherever the request is served (``GET /api/approvals``, the ``approval_pending`` frame) and NEVER in a push. A
critical request is approved only with ``{"digest": <that value>}`` in the body: missing is ``409 digest_required``,
different is ``409 digest_mismatch``, and nothing is approved either way. A request that is not critical approves
as before, and denying never needs it.

What it proves, stated plainly: the client fetched this exact request. It does not prove a person read it in the
app (a notification extension that fetches holds the digest too); keeping critical requests off the lock screen
still rests on the client not offering Approve there.
"""

from __future__ import annotations

import asyncio
import secrets
import types

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from prometheus.config.device_store import DeviceStore  # noqa: E402
from prometheus.permissions.approval_queue import (  # noqa: E402
    ApprovalQueue,
    ApprovalResult,
    PendingAction,
)

GLOBAL = "digest-test-" + secrets.token_hex(8)


class _Gate:
    def add_grant(self, *a, **kw):
        return None


@pytest.fixture
def d(tmp_path):
    from prometheus.web.server import create_app

    app = create_app({"web": {"api_token": GLOBAL}}, device_store=DeviceStore(tmp_path / "devices.db"))
    queue = ApprovalQueue(security_gate=_Gate(), telegram_adapter=None)
    app.state.approval_queue = queue
    critical = PendingAction(request_id="c0c0c001", tool_name="bash", description="rm -rf build",
                             session_id="web:s1", arguments={"command": "rm -rf build"},
                             critical_matches=("command:rm -rf",))
    plain = PendingAction(request_id="c0c0c002", tool_name="bash", description="ls", session_id="web:s1",
                          arguments={"command": "ls"})
    for a in (critical, plain):
        queue._register(a)
    client = TestClient(app, client=("127.0.0.1", 50123))
    return client, queue, critical, plain


H = {"Authorization": f"Bearer {GLOBAL}"}


def _served(client, rid):
    rows = client.get("/api/approvals", headers=H).json()
    rows = rows.get("pending", rows) if isinstance(rows, dict) else rows
    return next(r for r in rows if r["request_id"] == rid)


def test_every_served_request_carries_a_digest(d):
    client, queue, critical, plain = d
    c, p = _served(client, critical.request_id), _served(client, plain.request_id)
    assert c["critical"] is True and p["critical"] is False
    for row in (c, p):
        assert isinstance(row["digest"], str) and len(row["digest"]) == 64
        int(row["digest"], 16)
    assert c["digest"] != p["digest"]


def test_a_critical_approve_without_the_digest_answers_nothing(d):
    client, queue, critical, _ = d
    r = client.post(f"/api/approvals/{critical.request_id}/approve", headers=H, json={"scope": "once"})
    assert r.status_code == 409 and r.json()["error"] == "digest_required"
    assert critical._result == ApprovalResult.TIMEOUT and critical.request_id in queue.pending


@pytest.mark.parametrize("wrong", ["", "0" * 64, "not-hex", 7, None])
def test_a_critical_approve_with_another_digest_answers_nothing(d, wrong):
    client, queue, critical, plain = d
    body = {"scope": "once", "digest": wrong}
    r = client.post(f"/api/approvals/{critical.request_id}/approve", headers=H, json=body)
    assert r.status_code == 409 and r.json()["error"] in ("digest_required", "digest_mismatch")
    assert critical._result == ApprovalResult.TIMEOUT


def test_another_requests_digest_is_not_this_ones(d):
    client, queue, critical, plain = d
    body = {"scope": "once", "digest": _served(client, plain.request_id)["digest"]}
    r = client.post(f"/api/approvals/{critical.request_id}/approve", headers=H, json=body)
    assert r.status_code == 409 and r.json()["error"] == "digest_mismatch"
    assert critical._result == ApprovalResult.TIMEOUT


def test_a_critical_approve_with_the_served_digest_approves_it(d):
    client, queue, critical, _ = d
    body = {"scope": "once", "digest": _served(client, critical.request_id)["digest"]}
    r = client.post(f"/api/approvals/{critical.request_id}/approve", headers=H, json=body)
    assert r.status_code == 200 and r.json()["ok"] is True, r.text
    assert critical._result == ApprovalResult.APPROVED


def test_a_request_that_is_not_critical_approves_as_before(d):
    client, queue, _, plain = d
    r = client.post(f"/api/approvals/{plain.request_id}/approve", headers=H, json={"scope": "once"})
    assert r.status_code == 200 and plain._result == ApprovalResult.APPROVED


def test_denying_a_critical_request_never_needs_it(d):
    client, queue, critical, _ = d
    r = client.post(f"/api/approvals/{critical.request_id}/deny", headers=H)
    assert r.status_code == 200 and critical._result == ApprovalResult.DENIED


def test_approve_all_never_approves_a_critical_request(d):
    """Until the REST route takes only single ids, `all` is refused while a critical request is pending: a
    critical request is approved one at a time, with its digest."""
    client, queue, critical, plain = d
    r = client.post("/api/approvals/all/approve", headers=H, json={"scope": "once"})
    assert r.status_code == 409 and r.json()["error"] == "digest_required"
    assert critical._result == ApprovalResult.TIMEOUT and plain._result == ApprovalResult.TIMEOUT


def test_the_digest_is_the_same_in_the_list_and_the_frame_and_never_in_a_push():
    from prometheus.push.apns import SendResult
    from prometheus.push.dispatcher import PushDispatcher

    q = ApprovalQueue(security_gate=_Gate(), telegram_adapter=None)
    signals: list = []

    class _Bus:
        async def emit(self, signal):
            signals.append(signal)

    q.signal_bus = _Bus()

    seen: dict = {}

    async def go():
        waiter = asyncio.create_task(q.request_approval("bash", "rm -rf build", arguments={"command": "rm -rf b"}))
        for _ in range(50):
            await asyncio.sleep(0)
            if signals:
                break
        action = q.pending[signals[-1].payload["request_id"]]
        seen.update(served=q.serialize_pending(action)["digest"], direct=q.digest(action))
        waiter.cancel()

    asyncio.run(go())
    frame = signals[-1].payload
    assert frame["digest"] == seen["served"] == seen["direct"]

    sent: list[dict] = []

    class _Sender:
        async def send(self, **kwargs):
            sent.append(kwargs)
            return SendResult(outcome="ok", status=200)

    class _Store:
        def push_targets(self):
            return [types.SimpleNamespace(id="d1", apns_token="tok", environment="sandbox", bundle_id="",
                                          push_failures=0)]

        def owner_device_ids(self):
            return {"d1"}

        def approval_push_targets(self, *, desktop):
            return self.push_targets()

    asyncio.run(PushDispatcher(_Store(), _Sender(), bridge=None).on_signal(
        types.SimpleNamespace(kind="approval_pending", payload=frame, source="test")))
    assert sent and all(frame["digest"] not in str(s["payload"]) for s in sent)


def test_the_digest_does_not_change_while_the_request_waits(d):
    client, queue, critical, _ = d
    first = _served(client, critical.request_id)["digest"]
    queue.critical_rules = None                                     # nothing the digest covers depends on these
    assert _served(client, critical.request_id)["digest"] == first


def test_server_capabilities_lists_approval_digest():
    from prometheus.web.ws_server import WebSocketBridge

    bridge = WebSocketBridge.__new__(WebSocketBridge)
    bridge.config = {}
    assert "approval_digest" in bridge._server_capabilities()
