"""The dream cycle must not freeze the daemon twice an hour.

FOUND FROM THE DAEMON'S OWN TELEMETRY, not from reading code. Its loop
watchdog had logged 328 lag warnings in seven days, and 313 of the 327 gaps
between them were 1803 seconds — a 30-minute period matching AutoDreamEngine's
default `dream_interval_minutes`. Every event was `phase=idle tool=None`
(p50 1084 ms, worst 2830 ms), so it read as "nothing the agent did".

Three of the four phases called synchronous workers on the loop: a wiki tree
walk, SQLite dedup/decay, and a telemetry aggregation.

Same harness as tests/test_event_loop_not_blocked.py: make the worker slow,
count ticks on the same loop, assert the loop kept turning.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from prometheus.sentinel.autodream import AutoDreamEngine

BLOCK_SECONDS = 0.30
TICK = 0.01
MIN_TICKS = 5


async def _run_while_ticking(coro):
    ticks = 0
    stop = False

    async def ticker():
        nonlocal ticks
        while not stop:
            ticks += 1
            await asyncio.sleep(TICK)

    task = asyncio.create_task(ticker())
    await asyncio.sleep(0)
    try:
        result = await coro
    finally:
        stop = True
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    return result, ticks


def _slow(**attrs):
    """A stub whose named methods block for BLOCK_SECONDS."""
    def make(value):
        def blocking(*a, **k):
            time.sleep(BLOCK_SECONDS)
            return value
        return blocking
    return SimpleNamespace(**{k: make(v) for k, v in attrs.items()})


def _engine() -> AutoDreamEngine:
    eng = AutoDreamEngine.__new__(AutoDreamEngine)
    eng._auto_fix_wiki = False
    eng._bus = SimpleNamespace(emit=lambda *a, **k: asyncio.sleep(0))
    return eng


@pytest.mark.asyncio
async def test_wiki_lint_phase_does_not_stall_the_loop():
    eng = _engine()
    eng._wiki_linter = _slow(
        lint=SimpleNamespace(
            issues=[], has_issues=False, error_count=0, warning_count=0,
        ),
    )
    _, ticks = await _run_while_ticking(eng._phase_wiki_lint())
    assert ticks >= MIN_TICKS, f"loop advanced only {ticks} time(s) during the wiki walk"


@pytest.mark.asyncio
async def test_memory_consolidation_phase_does_not_stall_the_loop():
    eng = _engine()
    eng._memory_consolidator = _slow(
        consolidate=SimpleNamespace(
            duplicates_merged=0, confidence_decayed=0, tombstoned=0
        ),
    )
    _, ticks = await _run_while_ticking(eng._phase_memory_consolidation())
    assert ticks >= MIN_TICKS, f"loop advanced only {ticks} time(s) during consolidation"


@pytest.mark.asyncio
async def test_telemetry_digest_phase_does_not_stall_the_loop():
    eng = _engine()
    eng._telemetry_digest = _slow(
        generate=SimpleNamespace(
            total_calls=0, anomalies=[], has_anomalies=False, summary="",
        ),
    )
    _, ticks = await _run_while_ticking(eng._phase_telemetry_digest())
    assert ticks >= MIN_TICKS, f"loop advanced only {ticks} time(s) during the digest"


@pytest.mark.asyncio
async def test_the_harness_still_detects_a_blocking_coroutine():
    """Guard the guard — without this the three above pass for free."""
    async def blocks():
        time.sleep(BLOCK_SECONDS)

    _, ticks = await _run_while_ticking(blocks())
    assert ticks < MIN_TICKS, (
        f"the harness counted {ticks} ticks through a synchronous sleep — it "
        f"cannot tell a blocked loop from a free one"
    )
