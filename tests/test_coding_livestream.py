"""feat/coding-livestream — CodingLiveStream tailer (Phase A).

Drives the tailer against a REAL telemetry.db (seeded incrementally via
ToolCallTelemetry, the production writer) and a fake SignalBus that records what
was emitted — so every assertion is on the ACTUAL emitted events, not that emit
was called. Mirrors the survey's incremental-write reality.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from prometheus.coding.livestream import CodingLiveStream
from prometheus.sentinel.signals import ActivitySignal
from prometheus.telemetry.tracker import ToolCallTelemetry
from tests.support.doubles import register_double


@register_double("coding_livestream.FakeBus", replaces="prometheus.sentinel.signal_bus.SignalBus")
class FakeBus:
    """Mirrors SignalBus.emit dispatch (kind-specific + wildcard) and records
    every emitted signal so tests assert on actual events."""

    def __init__(self) -> None:
        self.emitted: list[ActivitySignal] = []
        self._subs: dict[str, list] = defaultdict(list)

    def subscribe(self, kind: str, cb: Any) -> None:
        self._subs[kind].append(cb)

    async def emit(self, signal: ActivitySignal) -> None:
        self.emitted.append(signal)
        for cb in list(self._subs.get(signal.kind, [])) + list(self._subs.get("*", [])):
            await cb(signal)

    def kinds(self, kind: str) -> list[ActivitySignal]:
        return [s for s in self.emitted if s.kind == kind]


async def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return predicate()


def _round(tel: ToolCallTelemetry, sid: str, i: int, *, subsystem: str = "agent_loop") -> None:
    tel.record_run(
        subsystem=subsystem, operation="loop_round" if subsystem == "agent_loop" else "run",
        outcome="success", duration_ms=12.0 + i, summary={"stop_reason": "tool_calls"},
        input_tokens=100 + i, output_tokens=10 + i, round_index=i,
        session_id=sid, model="gemma4-26b", thinking=True,
    )


def _terminal(sid: str, *, status: str = "completed", exit_code: int = 0) -> ActivitySignal:
    return ActivitySignal(
        kind="task_completed" if status == "completed" else "task_failed",
        payload={"session_id": sid, "status": status, "exit_code": exit_code,
                 "return_code": exit_code, "type": "local_agent"},
        source="task_supervisor",
    )


# ---------------------------------------------------------------------------
# 1. Incremental replay — one coding_round per row, in order, no dupes, no skips
# ---------------------------------------------------------------------------

async def test_emits_one_round_per_row_in_order(tmp_path: Path) -> None:
    db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db_path=str(db))
    bus = FakeBus()
    stream = CodingLiveStream(bus, db_path=str(db), poll_interval_s=0.01)
    stream.subscribe_lifecycle()
    sid = "coding:t1"
    stream.start_tail(sid)

    for i in range(3):  # first burst
        _round(tel, sid, i)
    assert await _wait_until(lambda: len(bus.kinds("coding_round")) >= 3)
    for i in (3, 4):    # second burst — proves the tailer keeps up incrementally
        _round(tel, sid, i)
    assert await _wait_until(lambda: len(bus.kinds("coding_round")) >= 5)

    # Rows that must NOT be tailed: the terminal coding_mode row (wrong subsystem)
    # and another run's agent_loop row (wrong session_id).
    _round(tel, sid, 0, subsystem="coding_mode")
    _round(tel, "coding:other", 0)
    await asyncio.sleep(0.05)

    rounds = bus.kinds("coding_round")
    idxs = [s.payload["round_index"] for s in rounds]
    assert idxs == [0, 1, 2, 3, 4], idxs  # in order, no skips, no dupes, filtered
    p = rounds[2].payload
    assert p["session_id"] == sid
    assert p["outcome"] == "success"
    assert p["stop_reason"] == "tool_calls"
    assert p["thinking"] is True
    assert p["input_tokens"] == 102 and p["output_tokens"] == 12

    await stream.stop_all()
    tel.close()


# ---------------------------------------------------------------------------
# 2. Terminal handling — exactly one coding_complete, then no further polling
# ---------------------------------------------------------------------------

async def test_terminal_emits_one_complete_then_stops(tmp_path: Path) -> None:
    db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db_path=str(db))
    bus = FakeBus()
    stream = CodingLiveStream(bus, db_path=str(db), poll_interval_s=0.01)
    stream.subscribe_lifecycle()
    sid = "coding:t2"
    stream.start_tail(sid)
    for i in range(3):
        _round(tel, sid, i)
    assert await _wait_until(lambda: len(bus.kinds("coding_round")) >= 3)

    await bus.emit(_terminal(sid, status="completed", exit_code=0))

    completes = bus.kinds("coding_complete")
    assert len(completes) == 1
    assert completes[0].payload["outcome"] == "success"
    assert completes[0].payload["exit_code"] == 0
    assert completes[0].payload["session_id"] == sid
    assert stream.active_sessions == []  # poller torn down

    # A row committed AFTER the terminal must NOT produce a coding_round.
    before = len(bus.kinds("coding_round"))
    _round(tel, sid, 99)
    await asyncio.sleep(0.05)
    assert len(bus.kinds("coding_round")) == before

    tel.close()


# ---------------------------------------------------------------------------
# 3. No-leak — N runs start and finish → zero residual pollers
# ---------------------------------------------------------------------------

async def test_no_residual_pollers_after_runs(tmp_path: Path) -> None:
    db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db_path=str(db))
    bus = FakeBus()
    stream = CodingLiveStream(bus, db_path=str(db), poll_interval_s=0.01)
    stream.subscribe_lifecycle()

    for n in range(3):
        sid = f"coding:run{n}"
        stream.start_tail(sid)
        _round(tel, sid, 0)
    assert await _wait_until(lambda: len(bus.kinds("coding_round")) >= 3)
    assert len(stream.active_sessions) == 3

    for n in range(3):
        await bus.emit(_terminal(f"coding:run{n}"))

    assert stream.active_sessions == []          # zero residual pollers
    assert len(bus.kinds("coding_complete")) == 3
    tel.close()


async def test_start_tail_is_idempotent_and_scoped(tmp_path: Path) -> None:
    db = tmp_path / "telemetry.db"
    ToolCallTelemetry(db_path=str(db)).close()
    bus = FakeBus()
    stream = CodingLiveStream(bus, db_path=str(db), poll_interval_s=0.01)
    stream.subscribe_lifecycle()
    stream.start_tail("coding:dup")
    stream.start_tail("coding:dup")              # second call is a no-op
    assert stream.active_sessions == ["coding:dup"]
    stream.start_tail("telegram:123")            # non-coding ignored
    assert stream.active_sessions == ["coding:dup"]
    await stream.stop_all()


# ---------------------------------------------------------------------------
# 4. Error path — tail query raises → coding_stream_error, run-state untouched
# ---------------------------------------------------------------------------

async def test_tail_error_emits_stream_error(tmp_path: Path) -> None:
    # Empty db (no subsystem_runs table) → the SELECT raises → fail-loud.
    db = tmp_path / "empty.db"
    sqlite3.connect(str(db)).close()
    bus = FakeBus()
    stream = CodingLiveStream(bus, db_path=str(db), poll_interval_s=0.01)
    stream.subscribe_lifecycle()
    sid = "coding:err"
    stream.start_tail(sid)

    assert await _wait_until(lambda: len(bus.kinds("coding_stream_error")) >= 1)
    err = bus.kinds("coding_stream_error")[0]
    assert err.payload["session_id"] == sid
    assert "error" in err.payload and err.payload["error"]

    # The tail error never propagated; the run's terminal event still completes
    # cleanly (handoff to artifact review survives a dead stream).
    await bus.emit(_terminal(sid, status="failed", exit_code=1))
    completes = bus.kinds("coding_complete")
    assert len(completes) == 1
    assert completes[0].payload["outcome"] == "failed"
    assert stream.active_sessions == []
