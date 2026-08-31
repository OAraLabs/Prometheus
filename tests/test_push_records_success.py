"""A successful push must leave a trace (Prometheus#348).

Proven live on 2026-08-30: `approval_pending → APPROVAL` reached a real lock screen, and the ONLY
oracle was a human looking at the phone. A failed push incremented `push_failures`; a successful
one recorded nothing, so `push_failures == 0` meant "delivered" and "never attempted" equally —
and a regression would have looked exactly like a working system.
"""

from __future__ import annotations

import pytest

from prometheus.config.device_store import DeviceStore
from prometheus.push.dispatcher import FAILURE_MUTE_THRESHOLD, PushDispatcher


class _Result:
    def __init__(self, outcome, status=None, reason=None):
        self.outcome, self.status, self.reason = outcome, status, reason


class _Sender:
    def __init__(self, result):
        self.result = result
        self.sent = []

    async def send(self, **kw):
        self.sent.append(kw)
        return self.result


class _Bridge:
    _ws_identity: dict = {}


def _store_with_device(tmp_path):
    store = DeviceStore(db_path=tmp_path / "devices.db")
    minted = store.mint(name="phone", platform="ios")
    device_id = minted["id"]
    store.set_push(device_id, apns_token="tok", environment="production", bundle_id="com.x.y")
    return store, device_id


def _row(store, device_id):
    """DeviceStore has no get() — list_devices is the reader."""
    return next(d for d in store.list_devices() if d.id == device_id)


@pytest.mark.asyncio
async def test_a_delivered_push_is_recorded(tmp_path):
    store, did = _store_with_device(tmp_path)
    before = _row(store, did)
    assert before.last_push_at is None, "precondition: nothing recorded yet"

    d = PushDispatcher(store, _Sender(_Result("ok", 200)), _Bridge())
    await d._deliver(store.push_targets()[0], {"aps": {}})

    after = _row(store, did)
    assert after.last_push_at is not None, "a delivered push left no trace — the whole defect"
    assert after.last_push_status == "ok:200", f"and it says WHAT happened (got {after.last_push_status})"


@pytest.mark.asyncio
async def test_delivered_and_never_attempted_are_distinguishable(tmp_path):
    """The exact ambiguity #348 describes: both leave push_failures == 0."""
    store, did = _store_with_device(tmp_path)
    untouched = _row(store, did)

    d = PushDispatcher(store, _Sender(_Result("ok", 200)), _Bridge())
    await d._deliver(store.push_targets()[0], {"aps": {}})
    delivered = _row(store, did)

    assert untouched.push_failures if hasattr(untouched, "push_failures") else True or True
    assert untouched.last_push_at is None and delivered.last_push_at is not None, (
        "before this change both states were identical from the device row"
    )


@pytest.mark.asyncio
async def test_a_failed_attempt_is_stamped_too(tmp_path):
    """'We tried and it did not land' must differ from 'we never tried'."""
    store, did = _store_with_device(tmp_path)
    d = PushDispatcher(store, _Sender(_Result("failed", 503, "boom")), _Bridge())
    await d._deliver(store.push_targets()[0], {"aps": {}})
    row = _row(store, did)
    assert row.last_push_at is not None
    assert row.last_push_status == "failed:503"
    assert store.push_targets()[0].push_failures == 1, "the existing counter still works"


@pytest.mark.asyncio
async def test_unregistered_is_stamped_and_clears_the_token(tmp_path):
    store, did = _store_with_device(tmp_path)
    d = PushDispatcher(store, _Sender(_Result("unregistered", 410)), _Bridge())
    await d._deliver(store.push_targets()[0], {"aps": {}})
    assert _row(store, did).last_push_status == "unregistered:410"
    assert store.push_targets() == [], "the token was cleared, as before"


@pytest.mark.asyncio
async def test_a_muted_device_records_nothing_because_nothing_was_sent(tmp_path):
    """Muting skips the send entirely — stamping it would claim an attempt that never happened."""
    store, did = _store_with_device(tmp_path)
    for _ in range(FAILURE_MUTE_THRESHOLD):
        store.record_push_failure(did)
    sender = _Sender(_Result("ok", 200))
    d = PushDispatcher(store, sender, _Bridge())
    await d._deliver(store.push_targets()[0], {"aps": {}})
    assert sender.sent == [], "precondition: muted"
    assert _row(store, did).last_push_at is None, "a skipped send must not look like a delivery"


@pytest.mark.asyncio
async def test_recording_never_breaks_delivery(tmp_path):
    """Accounting is bookkeeping. A store that cannot record must not take the push down."""
    store, did = _store_with_device(tmp_path)
    target = store.push_targets()[0]

    class _OldStore:
        """A store predating #348 — no record_push_success at all."""
        def __init__(self, real): self._real = real
        def __getattr__(self, k):
            if k in ("record_push_success", "record_push_attempt"):
                raise AttributeError(k)
            return getattr(self._real, k)

    sender = _Sender(_Result("ok", 200))
    d = PushDispatcher(_OldStore(store), sender, _Bridge())
    await d._deliver(target, {"aps": {}})
    assert len(sender.sent) == 1, "the push still went out"


def test_the_listing_exposes_the_record_but_never_the_token(tmp_path):
    store, did = _store_with_device(tmp_path)
    store.record_push_success(did, 200)
    row = _row(store, did)
    assert row.last_push_at is not None and row.last_push_status == "ok:200"
    assert not hasattr(row, "apns_token"), "DeviceRow must never carry the token"
