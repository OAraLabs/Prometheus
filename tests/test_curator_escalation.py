"""Sprint S4 A2 — Curator eternal-loop escalation tests.

Verifies the Sprint 1 Curator HIGH-RISK swallow at ``learning/curator.py:320``
is retrofitted:

  - run_once writes a `subsystem_runs` row at cycle completion
  - consecutive failures bump a counter
  - the counter resets on a successful run
  - crossing the threshold emits a `curator_degraded` signal exactly once
  - sleep duration follows exponential backoff, capped at the max multiplier
  - the loop iteration writes a `subsystem_runs` row even when paused

The /health command (A3) is what makes these telemetry rows user-visible.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from prometheus.learning.curator import (
    Curator,
    CuratorRun,
    _DEGRADED_THRESHOLD,
    _MAX_BACKOFF_MULTIPLIER,
    _STARTUP_GRACE_SECONDS,
)
from prometheus.learning.skill_state import SkillStateStore
from prometheus.providers.base import ApiTextDeltaEvent
from prometheus.sentinel.signals import ActivitySignal, SignalBus
from prometheus.telemetry.tracker import ToolCallTelemetry

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ScriptedProvider:
    """Yields a fixed sequence then completes (success path)."""

    def __init__(self, text: str) -> None:
        self._text = text

    async def stream_message(self, request):
        yield ApiTextDeltaEvent(text=self._text)


class _FailingProvider:
    """Raises during streaming — exercises the envelope's failure path."""

    async def stream_message(self, request):
        raise RuntimeError("synthetic provider failure")
        yield  # unreachable; keeps generator type


def _build_curator(
    tmp_path: Path,
    *,
    provider,
    telemetry: ToolCallTelemetry | None = None,
    signal_bus: SignalBus | None = None,
) -> Curator:
    auto = tmp_path / "skills" / "auto"
    auto.mkdir(parents=True)
    # Two skills so the LLM path is actually exercised (run_once short-circuits
    # the LLM call when there are no skills).
    (auto / "a.md").write_text("---\nname: a\n---\n# A\n")
    (auto / "b.md").write_text("---\nname: b\n---\n# B\n")
    return Curator(
        provider,
        auto_dir=auto,
        reports_dir=tmp_path / "curator",
        state_store=SkillStateStore(tmp_path / "_state.json"),
        interval_seconds=60,
        telemetry=telemetry,
        signal_bus=signal_bus,
    )


# ---------------------------------------------------------------------------
# Sprint S4 A2 acceptance
# ---------------------------------------------------------------------------


class TestCuratorBackoffMath:
    """Pure-function checks on `_compute_sleep`."""

    def test_no_failure_returns_base_interval(self) -> None:
        c = Curator(MagicMock(), interval_seconds=120)
        assert c._compute_sleep(0) == 120.0

    def test_exponential_growth(self) -> None:
        c = Curator(MagicMock(), interval_seconds=60)
        assert c._compute_sleep(1) == 120.0   # 2 ** 1
        assert c._compute_sleep(2) == 240.0   # 2 ** 2
        assert c._compute_sleep(3) == 480.0   # 2 ** 3 (== 8x = cap)

    def test_capped_at_max_multiplier(self) -> None:
        c = Curator(MagicMock(), interval_seconds=60)
        assert _MAX_BACKOFF_MULTIPLIER == 8
        # n=4 → 16x, but cap is 8x
        assert c._compute_sleep(4) == 60.0 * 8
        assert c._compute_sleep(99) == 60.0 * 8


class TestCuratorFailureCounter:
    """The counter increments on `run.errors`, resets on clean runs."""

    def test_clean_run_keeps_counter_at_zero(self, tmp_path: Path) -> None:
        c = _build_curator(tmp_path, provider=_ScriptedProvider(
            "```yaml\nconsolidations: []\nprunings: []\n```"
        ))
        run = asyncio.run(c.run_once())
        assert not run.errors
        asyncio.run(c._handle_run_result(run))
        assert c._consecutive_failures == 0
        assert c._degraded_signal_sent is False

    def test_run_with_errors_increments_counter(self, tmp_path: Path) -> None:
        c = _build_curator(tmp_path, provider=_FailingProvider())
        run = asyncio.run(c.run_once())
        assert run.errors  # LLM failed → recorded on run
        asyncio.run(c._handle_run_result(run))
        assert c._consecutive_failures == 1

    def test_recovery_resets_counter_and_signal_flag(self, tmp_path: Path) -> None:
        c = _build_curator(tmp_path, provider=_FailingProvider())
        # Simulate 3 failures
        for _ in range(_DEGRADED_THRESHOLD):
            run = asyncio.run(c.run_once())
            asyncio.run(c._handle_run_result(run))
        assert c._consecutive_failures == _DEGRADED_THRESHOLD
        assert c._degraded_signal_sent is True  # emitted (no bus, but flag set)

        # Now swap to a working provider and run again
        c._envelope._on_failure = "raise"  # already 'raise'
        c._provider = _ScriptedProvider(
            "```yaml\nconsolidations: []\nprunings: []\n```"
        )
        good = asyncio.run(c.run_once())
        assert not good.errors
        asyncio.run(c._handle_run_result(good))
        assert c._consecutive_failures == 0
        assert c._degraded_signal_sent is False


class TestCuratorDegradedSignal:
    """`curator_degraded` is emitted exactly once at the threshold."""

    def test_emits_on_threshold_with_bus(self, tmp_path: Path) -> None:
        bus = SignalBus()
        captured: list[ActivitySignal] = []

        async def _capture(s): captured.append(s)
        bus.subscribe("curator_degraded", _capture)

        c = _build_curator(tmp_path, provider=_FailingProvider(), signal_bus=bus)

        for _ in range(_DEGRADED_THRESHOLD):
            run = asyncio.run(c.run_once())
            asyncio.run(c._handle_run_result(run))

        assert len(captured) == 1, (
            f"expected exactly one degraded emission, got {len(captured)}"
        )
        sig = captured[0]
        assert sig.kind == "curator_degraded"
        assert sig.payload["consecutive_failures"] == _DEGRADED_THRESHOLD
        assert "synthetic provider failure" in sig.payload["last_error"]
        assert sig.payload["next_sleep_seconds"] >= c._interval * 2

    def test_does_not_re_emit_on_further_failures(self, tmp_path: Path) -> None:
        bus = SignalBus()
        captured: list[ActivitySignal] = []

        async def _capture(s): captured.append(s)
        bus.subscribe("curator_degraded", _capture)

        c = _build_curator(tmp_path, provider=_FailingProvider(), signal_bus=bus)
        # Fire 6 consecutive failures
        for _ in range(_DEGRADED_THRESHOLD + 3):
            run = asyncio.run(c.run_once())
            asyncio.run(c._handle_run_result(run))

        # Still only one emission — the flag suppresses further sends.
        assert len(captured) == 1


class TestCuratorRunRecordsTelemetry:
    """run_once writes a `subsystem_runs` row at cycle completion."""

    def test_success_writes_success_row(self, tmp_path: Path) -> None:
        tel = ToolCallTelemetry(db_path=tmp_path / "t.db")
        c = _build_curator(
            tmp_path / "rundir",
            provider=_ScriptedProvider(
                "```yaml\nconsolidations: []\nprunings: []\n```"
            ),
            telemetry=tel,
        )
        run = asyncio.run(c.run_once())
        assert not run.errors

        rows = tel.runs_since(0, subsystem="curator")
        # Envelope writes a per-call success row + run_once writes the
        # cycle row. Find the cycle one.
        cycle_rows = [r for r in rows if r["operation"] == "run_once"]
        assert len(cycle_rows) == 1
        assert cycle_rows[0]["outcome"] == "success"

    def test_failure_writes_failed_row_with_summary(self, tmp_path: Path) -> None:
        tel = ToolCallTelemetry(db_path=tmp_path / "t.db")
        c = _build_curator(
            tmp_path / "rundir",
            provider=_FailingProvider(),
            telemetry=tel,
        )
        run = asyncio.run(c.run_once())
        assert run.errors

        rows = tel.runs_since(0, subsystem="curator")
        cycle_rows = [r for r in rows if r["operation"] == "run_once"]
        assert len(cycle_rows) == 1
        # Even if some auto-transitions happened (stage 1 ran), the cycle
        # is "partial"; if nothing happened it's "failed". Our 2-skill
        # fixture has no stale/archive transitions on fresh mtimes, so
        # nothing in stage 1 either → "failed".
        assert cycle_rows[0]["outcome"] in ("failed", "partial")

    def test_partial_outcome_when_auto_transitions_but_llm_fails(
        self, tmp_path: Path
    ) -> None:
        """An auto-transition happened (deterministic stage 1) but the LLM
        review failed (stage 2). Outcome should be 'partial', not 'failed'."""
        tel = ToolCallTelemetry(db_path=tmp_path / "t.db")
        c = _build_curator(
            tmp_path / "rundir",
            provider=_FailingProvider(),
            telemetry=tel,
        )
        # Age one skill into the stale range so auto-pass marks it
        ancient = time.time() - 365 * 86400
        target = c._auto_dir / "a.md"
        os.utime(target, (ancient, ancient))

        run = asyncio.run(c.run_once())
        assert run.auto_transitions  # stage 1 did something
        assert run.errors            # stage 2 failed

        rows = tel.runs_since(0, subsystem="curator")
        cycle_rows = [r for r in rows if r["operation"] == "run_once"]
        assert cycle_rows[0]["outcome"] == "partial"

    def test_dry_run_does_not_record_cycle(self, tmp_path: Path) -> None:
        tel = ToolCallTelemetry(db_path=tmp_path / "t.db")
        c = _build_curator(
            tmp_path / "rundir",
            provider=_ScriptedProvider(
                "```yaml\nconsolidations: []\nprunings: []\n```"
            ),
            telemetry=tel,
        )
        asyncio.run(c.run_once(dry_run=True))
        # Envelope still records the LLM call, but the cycle row is gated
        # by the dry_run check.
        rows = tel.runs_since(0, subsystem="curator")
        assert not any(r["operation"] == "run_once" for r in rows)


class TestCuratorLoopOutcomeRecording:
    """The eternal loop writes a row per iteration outcome."""

    def test_paused_records_skipped(self, tmp_path: Path) -> None:
        tel = ToolCallTelemetry(db_path=tmp_path / "t.db")
        c = _build_curator(
            tmp_path / "rundir",
            provider=_ScriptedProvider("doesn't matter"),
            telemetry=tel,
        )
        c._record_loop_outcome("skipped", duration_ms=0.0)
        rows = tel.runs_since(0, subsystem="curator")
        loop_rows = [r for r in rows if r["operation"] == "_loop_iteration"]
        assert len(loop_rows) == 1
        assert loop_rows[0]["outcome"] == "skipped"


# ---------------------------------------------------------------------------
# Watermark scheduling (2026-08) — the first wait derives from the PERSISTED
# last_run_at. The prior sleep-first shape re-armed a full interval on every
# daemon restart: with the default 7-day interval and a sub-week restart
# cadence the loop fired exactly once ever (2026-07-17, after the only
# 7-day uptime streak). Same failure shape as the extractor cursor (PR #145).
# ---------------------------------------------------------------------------

_INTERVAL = 3600  # comfortably above the ctor's 60s clamp
_NOW = 1_700_000_000.0


def _watermark_curator(
    tmp_path: Path, *, last_run_at: float | None
) -> Curator:
    """Curator over a real on-disk state store; no LLM is ever touched."""
    store = SkillStateStore(tmp_path / "_state.json")
    if last_run_at is not None:
        store.update_curator(last_run_at=last_run_at)
    return Curator(
        MagicMock(),
        state_store=store,
        auto_dir=tmp_path / "skills" / "auto",
        reports_dir=tmp_path / "curator",
        interval_seconds=_INTERVAL,
    )


class TestWatermarkScheduling:
    """_initial_delay_seconds: max(grace, watermark + interval - now)."""

    def test_never_run_is_due_after_the_startup_grace(self, tmp_path: Path) -> None:
        """Watermark 0.0 counts as overdue — a brand-new install runs once
        the daemon has settled (empty skills dir → run_once is LLM-free)."""
        c = _watermark_curator(tmp_path, last_run_at=None)
        assert c._initial_delay_seconds(_NOW) == float(_STARTUP_GRACE_SECONDS)

    def test_overdue_watermark_collapses_to_the_grace(self, tmp_path: Path) -> None:
        c = _watermark_curator(tmp_path, last_run_at=_NOW - 2 * _INTERVAL)
        assert c._initial_delay_seconds(_NOW) == float(_STARTUP_GRACE_SECONDS)

    def test_midcycle_restart_waits_only_the_remainder(self, tmp_path: Path) -> None:
        """Half the interval has elapsed → wait the other half, not a fresh
        full interval. This is the sentence the old code got wrong."""
        c = _watermark_curator(tmp_path, last_run_at=_NOW - _INTERVAL / 2)
        assert c._initial_delay_seconds(_NOW) == pytest.approx(_INTERVAL / 2)

    def test_just_ran_waits_the_full_interval(self, tmp_path: Path) -> None:
        c = _watermark_curator(tmp_path, last_run_at=_NOW)
        assert c._initial_delay_seconds(_NOW) == pytest.approx(float(_INTERVAL))

    def test_restart_does_not_rearm_a_full_interval(self, tmp_path: Path) -> None:
        """THE regression: a second process over the same persisted store
        must see the same remaining wait, not start the countdown over."""
        first = _watermark_curator(tmp_path, last_run_at=_NOW - 0.75 * _INTERVAL)
        delay_first = first._initial_delay_seconds(_NOW)

        # "Restart": a fresh Curator + fresh store handle over the same file.
        second = Curator(
            MagicMock(),
            state_store=SkillStateStore(tmp_path / "_state.json"),
            auto_dir=tmp_path / "skills" / "auto",
            reports_dir=tmp_path / "curator",
            interval_seconds=_INTERVAL,
        )
        delay_second = second._initial_delay_seconds(_NOW)

        assert delay_first == pytest.approx(0.25 * _INTERVAL)
        assert delay_second == pytest.approx(delay_first)
        assert delay_second < _INTERVAL

    def test_unreadable_store_falls_back_to_overdue(self, tmp_path: Path) -> None:
        """A broken state read must not park the loop for a full interval —
        running is the recoverable direction (prunings archive, never delete)."""
        c = _watermark_curator(tmp_path, last_run_at=_NOW)
        c._state_store = MagicMock()
        c._state_store.curator.side_effect = RuntimeError("corrupt state")
        assert c._initial_delay_seconds(_NOW) == float(_STARTUP_GRACE_SECONDS)
