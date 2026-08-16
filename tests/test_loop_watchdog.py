"""The watchdog must measure lag, not manufacture it.

Rebuilt from stated constraints (see the module docstring for which). The
tests exist mostly to pin the two decisions that would silently ruin the
measurement: the monotonic clock, and the WARNING level.
"""

from __future__ import annotations

import asyncio
import logging
import time

import pytest

from prometheus.engine import loop_watchdog as W

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _clear_slot():
    W.publish_progress(None)
    yield
    W.publish_progress(None)


async def _run_one_sample(monkeypatch, block_seconds: float, interval=0.05):
    """Run exactly one watchdog iteration, blocking the loop inside the sleep."""
    real_sleep = asyncio.sleep
    calls = {"n": 0}

    async def fake_sleep(d):
        calls["n"] += 1
        if calls["n"] == 1:
            if block_seconds:
                # Block the loop the way a synchronous call would.
                time.sleep(block_seconds)
            await real_sleep(0)
            return          # let the iteration COMPUTE its lag
        raise asyncio.CancelledError   # stop on the next iteration

    monkeypatch.setattr(W.asyncio, "sleep", fake_sleep)
    with pytest.raises(asyncio.CancelledError):
        await W.watch(threshold_ms=250.0, interval=interval)


class TestItMeasuresLateness:
    def test_a_blocked_loop_is_reported_at_WARNING(self, monkeypatch, caplog):
        with caplog.at_level(logging.WARNING, logger=W.logger.name):
            asyncio.run(_run_one_sample(monkeypatch, block_seconds=0.4))
        assert "event-loop lag" in caplog.text
        recs = [r for r in caplog.records if "event-loop lag" in r.getMessage()]
        assert recs and recs[0].levelno == logging.WARNING, (
            "must be WARNING: the progress emitter logs at debug, which is "
            "exactly why 42 disconnects produced no visible evidence"
        )

    def test_an_idle_loop_reports_nothing(self, monkeypatch, caplog):
        with caplog.at_level(logging.WARNING, logger=W.logger.name):
            asyncio.run(_run_one_sample(monkeypatch, block_seconds=0.0))
        assert "event-loop lag" not in caplog.text

    def test_the_warning_names_the_phase_and_tool(self, monkeypatch, caplog):
        W.publish_progress({"phase": "tool", "tool_name": "bash"})
        with caplog.at_level(logging.WARNING, logger=W.logger.name):
            asyncio.run(_run_one_sample(monkeypatch, block_seconds=0.4))
        assert "phase=tool" in caplog.text and "tool=bash" in caplog.text

    def test_an_idle_stall_still_annotates(self, monkeypatch, caplog):
        """No turn in flight is the case a per-turn timer cannot see."""
        with caplog.at_level(logging.WARNING, logger=W.logger.name):
            asyncio.run(_run_one_sample(monkeypatch, block_seconds=0.4))
        assert "phase=idle" in caplog.text


class TestItUsesTheMonotonicClock:
    def test_lateness_comes_from_loop_time_not_wall_clock(self, monkeypatch, caplog):
        """An NTP step must not be reported as lag.

        This is the decision most likely to be "simplified" later: a
        time.time() version looks equivalent and would turn every clock
        correction into a fabricated multi-second stall — manufacturing
        exactly the evidence the watchdog exists to look for.
        """
        monkeypatch.setattr(time, "time", lambda: 1_000_000.0)  # frozen wall clock

        seen = {"loop_time": 0}
        real_sleep = asyncio.sleep

        async def one_shot(d):
            await real_sleep(0)
            raise asyncio.CancelledError

        async def go():
            loop = asyncio.get_running_loop()
            orig = loop.time

            def counting():
                seen["loop_time"] += 1
                return orig()

            monkeypatch.setattr(loop, "time", counting)
            monkeypatch.setattr(W.asyncio, "sleep", one_shot)
            with pytest.raises(asyncio.CancelledError):
                await W.watch(threshold_ms=250.0, interval=0.05)

        asyncio.run(go())
        assert seen["loop_time"] >= 2, (
            "watch() must read loop.time() before and after the sleep"
        )

    def test_a_wall_clock_jump_produces_no_warning(self, monkeypatch, caplog):
        base = [500.0]
        monkeypatch.setattr(time, "time", lambda: base[0])

        real_sleep = asyncio.sleep
        n = {"i": 0}

        async def jumpy(d):
            n["i"] += 1
            base[0] += 3600.0          # a one-hour NTP step mid-sleep
            await real_sleep(0)
            raise asyncio.CancelledError

        monkeypatch.setattr(W.asyncio, "sleep", jumpy)
        with caplog.at_level(logging.WARNING, logger=W.logger.name):
            async def go():
                with pytest.raises(asyncio.CancelledError):
                    await W.watch(threshold_ms=250.0, interval=0.05)
            asyncio.run(go())
        assert "event-loop lag" not in caplog.text, (
            "a wall-clock step was reported as loop lag — the clock is wrong"
        )


class TestTheProgressSlot:
    def test_publish_and_read_back(self):
        d = {"phase": "generating", "tool_name": None}
        W.publish_progress(d)
        assert W.current_progress() is d

    def test_cleared_slot_reports_idle(self):
        W.publish_progress(None)
        assert "phase=idle" in W._annotation()

    def test_annotation_survives_a_junk_slot(self):
        W.publish_progress("not a dict")  # type: ignore[arg-type]
        assert "phase=idle" in W._annotation(), (
            "a diagnostic that can raise inside its own logging path is worse "
            "than no diagnostic"
        )

    def test_threshold_and_interval_match_the_spec(self):
        assert W.LAG_THRESHOLD_MS == 250.0
        assert W.INTERVAL_SECONDS == 1.0


class TestItIsStartedOnceAtBoot:
    def test_daemon_starts_it_outside_any_turn(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent
               / "src/prometheus/daemon.py").read_text()
        assert "loop_watchdog.watch()" in src
        assert 'name="event-loop-watchdog"' in src

    def test_the_ws_handler_does_not_start_one_per_turn(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent
               / "src/prometheus/web/ws_server.py").read_text()
        assert "loop_watchdog.watch(" not in src, (
            "a per-turn watchdog cannot see an idle stall, which is the case "
            "Beacon's 45s terminate does not care about"
        )
        assert "publish_progress" in src
