"""The golden-trace export runs off the daemon's event loop (#705).

``run_once`` called the synchronous ``export_new_golden_traces`` directly from
an ``async def``, so the whole export (a telemetry batch read, an LCM context
read per trace, a JSONL write) ran on the daemon's only loop. On the mini an
890-trace export blocked it for 3.7 s while no turn was running: the loop
watchdog logged the spike at the same millisecond as the exporter's completion
line, and every client's 3 s progress heartbeat stalled with it.

These use a fake telemetry whose export blocks on a ``threading.Event``. An
export on the loop thread can never see the loop set that event, so each test
fails within its timeout instead of hanging if the export moves back onto the
loop.
"""

from __future__ import annotations

import asyncio
import json
import threading

import pytest

from prometheus.sentinel.golden_trace_exporter import (
    WATERMARK_FILENAME,
    GoldenTraceExporter,
)
from prometheus.telemetry.tracker import GoldenExport

WAIT_S = 5.0


class _BlockingTelemetry:
    """Export that waits for ``release`` and then reports one written batch."""

    def __init__(self, out_dir, *, last_rowid: int = 42) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.released_in_time: bool | None = None
        self.thread: threading.Thread | None = None
        self._path = out_dir / f"golden_traces_0_{last_rowid}.jsonl"
        self._last_rowid = last_rowid

    def export_new_golden_traces(self, **_kwargs) -> GoldenExport:
        self.thread = threading.current_thread()
        self.entered.set()
        self.released_in_time = self.release.wait(WAIT_S)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text("{}\n", encoding="utf-8")
        return GoldenExport(path=self._path, count=1, last_rowid=self._last_rowid)


def _exporter(tel, out_dir) -> GoldenTraceExporter:
    return GoldenTraceExporter(
        telemetry=tel, signal_bus=None, config={"output_dir": str(out_dir)},
    )


def _watermark(out_dir) -> int:
    return json.loads((out_dir / WATERMARK_FILENAME).read_text())["last_rowid"]


def test_the_loop_keeps_running_while_an_export_is_in_progress(tmp_path):
    """Another coroutine runs, and releases the export, while it is in flight."""
    out_dir = tmp_path / "trajectories"
    tel = _BlockingTelemetry(out_dir)
    exporter = _exporter(tel, out_dir)

    async def main():
        loop_thread = threading.current_thread()
        export = asyncio.create_task(exporter.run_once())
        # Only a free loop gets here while the export is still blocked.
        assert await asyncio.to_thread(tel.entered.wait, WAIT_S)
        tel.release.set()
        return loop_thread, await export

    loop_thread, path = asyncio.run(main())

    assert tel.released_in_time, "the export blocked the event loop"
    assert tel.thread is not loop_thread
    assert path == str(out_dir / "golden_traces_0_42.jsonl")
    assert _watermark(out_dir) == 42
    assert exporter.cycle_count == 1


def test_a_cancel_mid_export_still_advances_the_watermark(tmp_path):
    """Shutdown cancels the exporter task. The file and its watermark land together.

    The cancel reaches ``run_once`` at its await, but the worker thread has
    the watermark write too, so it finishes all of it. If the write ran on the
    loop after the await, the file would be on disk with the cursor behind it,
    and the next start would export the same traces into a second file.
    """
    out_dir = tmp_path / "trajectories"
    tel = _BlockingTelemetry(out_dir, last_rowid=7)
    exporter = _exporter(tel, out_dir)

    async def main():
        export = asyncio.create_task(exporter.run_once())
        assert await asyncio.to_thread(tel.entered.wait, WAIT_S)
        export.cancel()
        tel.release.set()
        with pytest.raises(asyncio.CancelledError):
            await export
        # asyncio.run joins the default executor on the way out, as the
        # daemon's own loop does, so the worker finishes before we look.

    asyncio.run(main())

    assert tel.released_in_time
    assert (out_dir / "golden_traces_0_7.jsonl").exists()
    assert _watermark(out_dir) == 7
    # The cancelled cycle never got back to the loop to count or log itself.
    assert exporter.cycle_count == 0
