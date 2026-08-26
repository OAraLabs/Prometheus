"""Approval lifecycle push — SignalBus emission + REST serializer parity (feat/approval-push).

Side-effect assertions: enqueue emits approval_pending whose payload is
BYTE-IDENTICAL to the REST serializer output (one shape, two transports —
drift impossible by construction); approve/deny/expiry each emit
approval_resolved with the right resolution. A queue without a bus still
works (pre-push behavior). The WS bridge maps both kinds to first-class
frame types.
"""

from __future__ import annotations

import asyncio

import pytest


class _Bus:
    """Stub SignalBus recording emissions."""

    def __init__(self):
        self.emitted: list[tuple[str, dict]] = []

    async def emit(self, signal):
        self.emitted.append((signal.kind, dict(signal.payload)))
        return signal


class _TG:
    async def send(self, *a, **k):
        pass


def _queue(tmp_path, bus=None):
    from prometheus.permissions.approval_queue import ApprovalQueue

    q = ApprovalQueue(telegram_adapter=_TG(), timeout_seconds=300)
    q.signal_bus = bus
    return q


def test_enqueue_emits_pending_matching_rest_shape(tmp_path):
    q = _queue(tmp_path, bus := _Bus())

    async def run():
        task = asyncio.create_task(q.request_approval("bash", "git push origin main"))
        await asyncio.sleep(0.05)  # let it enqueue and await
        # The pending emission's payload IS the REST serializer's dict.
        rid = next(iter(q.pending))
        rest_shape = q.serialize_pending(q.pending[rid])
        return task, rid, rest_shape

    task, rid, rest_shape = asyncio.run(run())
    task.cancel()

    kinds = [k for k, _ in bus.emitted]
    assert kinds[0] == "approval_pending"
    pending_payload = bus.emitted[0][1]
    assert pending_payload == rest_shape, "WS payload and REST dict must be byte-identical"
    assert pending_payload["request_id"] == rid
    assert pending_payload["tool_name"] == "bash"
    assert pending_payload["description"] == "git push origin main"
    assert "expires_at" in pending_payload and "extents" in pending_payload


def test_approve_emits_resolved(tmp_path):
    q = _queue(tmp_path, bus := _Bus())

    async def run():
        task = asyncio.create_task(q.request_approval("bash", "rm -r /tmp/x"))
        await asyncio.sleep(0.05)
        rid = next(iter(q.pending))
        ok = await q.approve(rid, scope="once")
        await asyncio.wait_for(task, timeout=2)
        return ok, task.result(), rid

    ok, result, rid = asyncio.run(run())
    assert ok is True
    from prometheus.permissions.approval_queue import ApprovalResult

    assert result == ApprovalResult.APPROVED
    resolved = [(k, p) for k, p in bus.emitted if k == "approval_resolved"]
    assert len(resolved) == 1
    assert resolved[0][1] == {"request_id": rid, "resolution": "approved", "scope": "once"}


def test_deny_emits_resolved(tmp_path):
    q = _queue(tmp_path, bus := _Bus())

    async def run():
        task = asyncio.create_task(q.request_approval("bash", "curl example.com"))
        await asyncio.sleep(0.05)
        rid = next(iter(q.pending))
        ok = await q.deny(rid)
        await asyncio.wait_for(task, timeout=2)
        return ok, rid

    ok, rid = asyncio.run(run())
    assert ok is True
    resolved = [(k, p) for k, p in bus.emitted if k == "approval_resolved"]
    assert len(resolved) == 1
    assert resolved[0][1] == {"request_id": rid, "resolution": "denied"}


def test_expiry_emits_resolved(tmp_path):
    q = _queue(tmp_path, bus := _Bus())
    q._timeout = 0.1  # expire fast

    async def run():
        result = await asyncio.wait_for(
            q.request_approval("bash", "sleep 1"), timeout=5
        )
        return result

    result = asyncio.run(run())
    from prometheus.permissions.approval_queue import ApprovalResult

    assert result == ApprovalResult.TIMEOUT
    resolved = [(k, p) for k, p in bus.emitted if k == "approval_resolved"]
    assert len(resolved) == 1
    assert resolved[0][1]["resolution"] == "expired"


def test_no_bus_is_pre_push_behavior(tmp_path):
    """signal_bus None → no emission, queue still functions (old builds)."""
    q = _queue(tmp_path, bus=None)

    async def run():
        task = asyncio.create_task(q.request_approval("bash", "ls"))
        await asyncio.sleep(0.05)
        rid = next(iter(q.pending))
        ok = await q.approve(rid)
        await asyncio.wait_for(task, timeout=2)
        return ok

    assert asyncio.run(run()) is True  # no AttributeError, works silently


def test_rest_and_ws_serializer_single_source():
    """server.py's GET handler must delegate to serialize_pending — a literal
    dict in the handler is the drift hazard this sprint removes."""
    import inspect

    from prometheus.web import server

    src = inspect.getsource(server.create_app)
    assert "serialize_pending(a) for a in queue.list_pending()" in src


def test_ws_bridge_maps_approval_kinds_to_frame_types():
    """_on_signal maps approval_pending/approval_resolved to first-class types."""
    import inspect

    from prometheus.web import ws_server

    src = inspect.getsource(ws_server.WebSocketBridge._on_signal)
    for kind in ("approval_pending", "approval_resolved"):
        assert f'signal.kind == "{kind}"' in src
        assert f'event["type"] = "{kind}"' in src
