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


# ---------------------------------------------------------------------------
# Enrichment (the closed v1 deferral): filters, seq, tools, acceptance
# ---------------------------------------------------------------------------


def _tool(tel: ToolCallTelemetry, sid: str | None, name: str, *, success: bool = True,
          error_type: str | None = None) -> None:
    tel.record(
        model="gemma4-26b", tool_name=name, success=success,
        latency_ms=42.0, error_type=error_type, session_id=sid,
    )


def _acceptance(tel: ToolCallTelemetry, sid: str, *, exit_code: int | None) -> None:
    tel.record_run(
        subsystem="coding_mode", operation="acceptance",
        outcome="success" if exit_code == 0 else "failed",
        summary={"exit_code": exit_code, "timed_out": exit_code is None,
                 "output_tail": "2 passed" if exit_code == 0 else "1 failed"},
        session_id=sid, model="gemma4-26b",
    )


async def test_non_round_agent_loop_operations_are_filtered(tmp_path: Path) -> None:
    """The latent bug the enrichment survey found: agent_loop also writes
    tool_advertisement (round_index NULL, once per episode) and microcompact
    rows under the same subsystem+session_id — both would have emitted as
    bogus coding_round frames. Unexercised in prod only because both
    instrumentations postdate the last coding run in the live DB."""
    db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db_path=str(db))
    bus = FakeBus()
    stream = CodingLiveStream(bus, db_path=str(db), poll_interval_s=0.01)
    sid = "coding:filter"
    stream.start_tail(sid)

    tel.record_run(subsystem="agent_loop", operation="tool_advertisement",
                   outcome="success", summary={}, session_id=sid)
    tel.record_run(subsystem="agent_loop", operation="microcompact",
                   outcome="success", summary={}, round_index=3, session_id=sid)
    _round(tel, sid, 0)
    assert await _wait_until(lambda: len(bus.kinds("coding_round")) >= 1)
    await asyncio.sleep(0.05)
    rounds = bus.kinds("coding_round")
    assert len(rounds) == 1
    assert rounds[0].payload["round_index"] == 0
    await stream.stop_all()


async def test_round_frames_carry_monotonic_seq(tmp_path: Path) -> None:
    """round_index restarts at 0 every episode; seq (the rowid) must not —
    it is the only per-run-unique key a client can hang a round card on."""
    db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db_path=str(db))
    bus = FakeBus()
    stream = CodingLiveStream(bus, db_path=str(db), poll_interval_s=0.01)
    sid = "coding:episodes"
    stream.start_tail(sid)

    _round(tel, sid, 0)  # episode 1
    _round(tel, sid, 1)
    _round(tel, sid, 0)  # episode 2 restarts the index
    assert await _wait_until(lambda: len(bus.kinds("coding_round")) >= 3)
    seqs = [s.payload["seq"] for s in bus.kinds("coding_round")]
    assert seqs == sorted(seqs) and len(set(seqs)) == 3, seqs
    idxs = [s.payload["round_index"] for s in bus.kinds("coding_round")]
    assert idxs == [0, 1, 0]
    await stream.stop_all()


async def test_tool_frames_attributed_to_the_last_round(tmp_path: Path) -> None:
    """Round attribution without a join: the loop commits round N's row
    before round N's tools run, so a tool row belongs to the last round the
    tail saw. Other sessions' tool rows and session-less rows are excluded."""
    db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db_path=str(db))
    bus = FakeBus()
    stream = CodingLiveStream(bus, db_path=str(db), poll_interval_s=0.01)
    sid = "coding:tools"
    stream.start_tail(sid)

    _round(tel, sid, 0)
    _tool(tel, sid, "bash")
    _tool(tel, sid, "grep", success=False, error_type="timeout")
    _tool(tel, "telegram:999", "bash")   # someone else's turn
    _tool(tel, None, "bash")             # session-less writer
    _round(tel, sid, 1)
    _tool(tel, sid, "read_file")

    assert await _wait_until(lambda: len(bus.kinds("coding_tool")) >= 3)
    await asyncio.sleep(0.05)
    tools = [(s.payload["tool_name"], s.payload["round_index"],
              s.payload["success"], s.payload["error_type"])
             for s in bus.kinds("coding_tool")]
    assert tools == [
        ("bash", 0, True, None),
        ("grep", 0, False, "timeout"),
        ("read_file", 1, True, None),
    ], tools
    await stream.stop_all()


async def test_acceptance_rows_emit_ground_truth_frames(tmp_path: Path) -> None:
    """coding_mode/acceptance rows become coding_acceptance frames; the
    terminal coding_mode/run row stays excluded as before."""
    db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db_path=str(db))
    bus = FakeBus()
    stream = CodingLiveStream(bus, db_path=str(db), poll_interval_s=0.01)
    sid = "coding:accept"
    stream.start_tail(sid)

    _round(tel, sid, 0)
    _acceptance(tel, sid, exit_code=1)
    _round(tel, sid, 1)
    _acceptance(tel, sid, exit_code=0)
    _round(tel, sid, 0, subsystem="coding_mode")  # terminal 'run' row — excluded

    assert await _wait_until(lambda: len(bus.kinds("coding_acceptance")) >= 2)
    await asyncio.sleep(0.05)
    frames = bus.kinds("coding_acceptance")
    assert [(f.payload["outcome"], f.payload["exit_code"]) for f in frames] == [
        ("failed", 1), ("success", 0),
    ]
    assert all("output_tail" in f.payload for f in frames)
    assert len(bus.kinds("coding_round")) == 2
    await stream.stop_all()


async def test_run_acceptance_persists_the_verdict(tmp_path: Path) -> None:
    """The write side: _run_acceptance records a ground-truth row per
    invocation — the fact that was previously computed and discarded."""
    from prometheus.coding.session import CodingSession, CodingTask

    class _Result:
        def __init__(self, exit_code: int | None, timed_out: bool) -> None:
            self.exit_code = exit_code
            self.timed_out = timed_out
            self.output = "collected 3 items\n1 failed, 2 passed"

    class _Sandbox:
        def __init__(self, results: list[_Result]) -> None:
            self._results = results

        async def run(self, command: str, timeout_seconds: float = 0) -> _Result:
            return self._results.pop(0)

    tel = ToolCallTelemetry(db_path=str(tmp_path / "telemetry.db"))
    session = CodingSession(
        provider=object(),
        model="scripted",
        sandbox=_Sandbox([_Result(1, False), _Result(0, False), _Result(None, True)]),
        task=CodingTask(task_id="tv", description="d", acceptance_command="pytest -q"),
        telemetry=tel,
    )
    for _ in range(3):
        await session._run_acceptance()

    con = sqlite3.connect(str(tmp_path / "telemetry.db"))
    rows = con.execute(
        "SELECT outcome, summary_json FROM subsystem_runs "
        "WHERE subsystem='coding_mode' AND operation='acceptance' "
        "AND session_id='coding:tv' ORDER BY rowid"
    ).fetchall()
    con.close()
    assert [r[0] for r in rows] == ["failed", "success", "failed"]
    import json as _json
    summaries = [_json.loads(r[1]) for r in rows]
    assert [s["exit_code"] for s in summaries] == [1, 0, None]
    assert summaries[2]["timed_out"] is True
    assert "passed" in summaries[0]["output_tail"]
