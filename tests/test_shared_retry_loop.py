"""The one retry loop, extracted from five byte-identical copies (#295).

The extraction's contract is that behaviour is IDENTICAL afterwards. tests/test_no_replay_after_
partial_stream.py already asserts that per provider, in both directions, for all five. This file
covers what only the shared helper can be asked directly: backoff, the attempt ceiling, the
call_once factory contract, and — the one that matters most — that retry POLICY stayed
per-provider rather than being averaged into one set.
"""

from __future__ import annotations

import httpx
import pytest

from prometheus.providers.anthropic import _RETRYABLE_STATUS_CODES as ANTHROPIC_STATUSES
from prometheus.providers.retry import MAX_RETRIES, stream_with_retry
from prometheus.providers.stub import RETRYABLE_STATUS_CODES as SHARED_STATUSES


class Boom(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")

        class R:
            status_code = status

        self.response = R()


def _factory(script):
    """A fresh iterator per attempt, and a record of how many attempts were made."""
    calls = {"n": 0}

    def call_once():
        calls["n"] += 1
        behaviour = script[min(calls["n"] - 1, len(script) - 1)]

        async def gen():
            for step in behaviour:
                if isinstance(step, Exception):
                    raise step
                yield step

        return gen()

    return call_once, calls


async def _drain(call_once, statuses=SHARED_STATUSES, **kw):
    slept: list[float] = []

    async def fake_sleep(d: float) -> None:
        slept.append(d)

    seen, error = [], None
    try:
        async for ev in stream_with_retry(call_once, retryable_status=statuses,
                                          label="test", sleep=fake_sleep, **kw):
            seen.append(ev)
    except BaseException as exc:  # noqa: BLE001 — the raise is the assertion
        error = exc
    return seen, error, slept


# ── the policy that must NOT have been averaged away ─────────────────────────────────────────

def test_anthropic_still_retries_529_and_the_others_still_do_not():
    """The two sets genuinely disagree. Folding them into one constant would have silently
    changed three providers' behaviour — which is exactly what a refactor must not do."""
    assert 529 in ANTHROPIC_STATUSES, "anthropic's own overloaded code"
    assert 529 not in SHARED_STATUSES, "and it is NOT in the shared set"


@pytest.mark.asyncio
async def test_a_529_retries_under_anthropics_set_and_raises_under_the_shared_one():
    """The same exception, two callers, two outcomes — proving policy is per-call-site."""
    call_once, calls = _factory([[Boom(529)], ["recovered"]])
    seen, error, _ = await _drain(call_once, statuses=ANTHROPIC_STATUSES)
    assert calls["n"] == 2 and seen == ["recovered"] and error is None

    call_once2, calls2 = _factory([[Boom(529)], ["recovered"]])
    seen2, error2, _ = await _drain(call_once2, statuses=SHARED_STATUSES)
    assert calls2["n"] == 1, "the shared set must NOT retry 529"
    assert isinstance(error2, Boom) and seen2 == []


# ── the loop itself ──────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_fresh_iterator_per_attempt():
    """call_once is a FACTORY, not an iterator — an iterator could only be consumed once, so a
    retry would silently yield nothing."""
    call_once, calls = _factory([[Boom(503)], [Boom(503)], ["third time"]])
    seen, error, _ = await _drain(call_once)
    assert calls["n"] == 3 and seen == ["third time"] and error is None


@pytest.mark.asyncio
async def test_backoff_grows_and_is_jittered():
    call_once, _ = _factory([[Boom(503)], [Boom(503)], [Boom(503)], ["ok"]])
    _seen, _error, slept = await _drain(call_once)
    assert len(slept) == 3, "one sleep per retry, none before the first attempt"
    assert slept[0] < slept[1] < slept[2], "exponential"
    # jitter is additive up to +25%, so each delay exceeds the bare power of two
    assert slept[0] >= 1.0 and slept[1] >= 2.0 and slept[2] >= 4.0


@pytest.mark.asyncio
async def test_the_attempt_ceiling_is_respected():
    call_once, calls = _factory([[Boom(503)]])
    _seen, error, slept = await _drain(call_once)
    assert calls["n"] == MAX_RETRIES + 1, "one initial attempt plus MAX_RETRIES retries"
    assert len(slept) == MAX_RETRIES
    assert isinstance(error, Boom), "the last failure is raised, not swallowed"


@pytest.mark.asyncio
async def test_a_transport_error_retries_without_a_status():
    call_once, calls = _factory([[httpx.ConnectError("refused")], ["ok"]])
    seen, error, _ = await _drain(call_once)
    assert calls["n"] == 2 and seen == ["ok"] and error is None


@pytest.mark.asyncio
async def test_a_non_retryable_status_raises_at_once():
    call_once, calls = _factory([[Boom(401)], ["never reached"]])
    seen, error, slept = await _drain(call_once)
    assert calls["n"] == 1 and seen == [] and slept == []
    assert isinstance(error, Boom)


@pytest.mark.asyncio
async def test_output_already_yielded_stops_the_retry():
    """Issue #293's invariant, now enforced in ONE place instead of five."""
    call_once, calls = _factory([["half an answer", Boom(503)], ["a different answer"]])
    seen, error, _ = await _drain(call_once)
    assert calls["n"] == 1, "no retry once anything reached the consumer"
    assert seen == ["half an answer"]
    assert isinstance(error, Boom)
