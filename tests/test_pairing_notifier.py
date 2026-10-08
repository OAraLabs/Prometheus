"""``PairingNotifier`` — how the operator's channels hear about a request.

Its listeners are the Beacon sockets and Telegram's HTTP call: both asynchronous, and both must finish
BEFORE the request that caused them returns. A fire-and-forget task would be lost whenever the event loop
ended first (a short request under a test client does exactly that), and ``notified`` on the 201 could only
guess. So ``emit`` is awaited, listeners may be plain or async, every one is bounded by a timeout, and one
that fails or stalls cannot hold up or break the request or its neighbours.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from prometheus.web.pairing_routes import PairingNotifier


@pytest.mark.asyncio
async def test_an_async_listener_is_awaited_before_emit_returns():
    seen: list[str] = []

    async def listener(kind, payload):
        await asyncio.sleep(0.01)
        seen.append(kind)
        return True

    notifier = PairingNotifier()
    notifier.subscribe(listener)
    assert await notifier.emit("pending", {"a": 1}) is True
    assert seen == ["pending"], "the listener finished before emit returned"


@pytest.mark.asyncio
async def test_a_plain_function_is_a_listener_too():
    notifier = PairingNotifier()
    notifier.subscribe(lambda kind, payload: True)
    assert await notifier.emit("pending", {}) is True


@pytest.mark.asyncio
async def test_emit_is_true_only_if_some_listener_took_it():
    notifier = PairingNotifier()
    assert await notifier.emit("pending", {}) is False
    notifier.subscribe(lambda kind, payload: False)
    notifier.subscribe(lambda kind, payload: None)
    assert await notifier.emit("pending", {}) is False
    notifier.subscribe(lambda kind, payload: True)
    assert await notifier.emit("pending", {}) is True


@pytest.mark.asyncio
async def test_a_failing_listener_does_not_stop_the_others():
    taken: list[str] = []

    def boom(kind, payload):
        raise RuntimeError("boom")

    async def aboom(kind, payload):
        raise RuntimeError("async boom")

    notifier = PairingNotifier()
    notifier.subscribe(boom)
    notifier.subscribe(aboom)
    notifier.subscribe(lambda kind, payload: taken.append(kind) or True)
    assert await notifier.emit("pending", {}) is True
    assert taken == ["pending"]


@pytest.mark.asyncio
async def test_a_stalled_listener_is_cut_off_and_does_not_count(monkeypatch):
    monkeypatch.setattr(PairingNotifier, "LISTENER_TIMEOUT_SECONDS", 0.05)

    async def stalled(kind, payload):
        await asyncio.sleep(5)
        return True

    notifier = PairingNotifier()
    notifier.subscribe(stalled)
    started = time.monotonic()
    assert await notifier.emit("pending", {}) is False
    assert time.monotonic() - started < 1.0, "a stalled channel must not hold the requester's response"


@pytest.mark.asyncio
async def test_listeners_run_side_by_side_not_one_after_another():
    async def slow(kind, payload):
        await asyncio.sleep(0.2)
        return True

    notifier = PairingNotifier()
    notifier.subscribe(slow)
    notifier.subscribe(slow)
    started = time.monotonic()
    await notifier.emit("pending", {})
    assert time.monotonic() - started < 0.35


@pytest.mark.asyncio
async def test_each_listener_gets_its_own_copy_of_the_payload():
    seen: list[dict] = []

    def mutate(kind, payload):
        payload["touched"] = True
        return True

    notifier = PairingNotifier()
    notifier.subscribe(mutate)
    notifier.subscribe(lambda kind, payload: seen.append(payload) or True)
    original = {"a": 1}
    await notifier.emit("pending", original)
    assert original == {"a": 1} and seen == [{"a": 1}]
