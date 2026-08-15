"""GoldenTraceExporter — periodically export golden tool-call traces to JSONL.

Source: Novel code for SUNRISE Sprint Session A.
Mirrors AutoDreamEngine: interval-driven asyncio loop that calls
``ToolCallTelemetry.export_golden_traces()`` and emits a signal on completion.

The exporter does NOT subscribe to idle/activity signals — it runs on a
fixed cadence regardless of activity. The downstream consumer (e.g. GEPA)
is the one that gates on idle.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from prometheus.sentinel.signals import ActivitySignal, SignalBus

log = logging.getLogger(__name__)

# Resume point, stored beside the exports it describes. Deleting it is safe
# and self-correcting: the next cycle re-exports from the beginning rather
# than losing traces, which is the right direction for a training corpus.
WATERMARK_FILENAME = ".export_state.json"


class GoldenTraceExporter:
    """Background loop that exports golden traces on a configurable interval.

    Args:
        telemetry: ToolCallTelemetry instance (must have ``export_golden_traces``).
        signal_bus: SignalBus for emitting ``golden_traces_exported`` after a write.
        config: ``trajectory_export`` section dict from prometheus.yaml.
            Recognised keys:
              - ``enabled`` (bool, default True)
              - ``interval_seconds`` (int, default 86400 = 24h)
              - ``nightly_limit`` (int, default 1000)
              - ``output_dir`` (str path, default ``~/.prometheus/trajectories/``)
              - ``format`` (str, default ``jsonl``)
    """

    def __init__(
        self,
        telemetry: object,
        signal_bus: SignalBus | None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._telemetry = telemetry
        self._bus = signal_bus
        cfg = config or {}
        self._enabled = cfg.get("enabled", True)
        self._interval = max(60, int(cfg.get("interval_seconds", 86400)))
        self._limit = int(cfg.get("nightly_limit", 1000))
        self._output_dir = cfg.get("output_dir", "~/.prometheus/trajectories/")
        self._format = cfg.get("format", "jsonl")
        self._running = False
        self._cycle_count = 0
        self._last_path: str | None = None

    async def start(self) -> asyncio.Task | None:
        """Spawn the background loop. Returns the task (or ``None`` if disabled)."""
        if not self._enabled:
            log.info("GoldenTraceExporter: disabled by config")
            return None
        if self._running:
            log.debug("GoldenTraceExporter: already running")
            return None
        self._running = True
        task = asyncio.create_task(self._loop(), name="golden_trace_exporter")
        log.info(
            "GoldenTraceExporter: started (interval=%ds, limit=%d, dir=%s)",
            self._interval, self._limit, self._output_dir,
        )
        return task

    async def stop(self) -> None:
        """Signal the loop to exit at next interval check."""
        self._running = False

    async def _loop(self) -> None:
        """Run export → sleep until next cycle."""
        while self._running:
            try:
                await self.run_once()
            except Exception:
                log.exception("GoldenTraceExporter cycle failed")
            # Sleep in 1s slices so stop() takes effect quickly.
            for _ in range(self._interval):
                if not self._running:
                    return
                await asyncio.sleep(1)

    @property
    def _watermark_path(self) -> Path:
        return Path(self._output_dir).expanduser().resolve() / WATERMARK_FILENAME

    def _read_watermark(self) -> int:
        """Last exported rowid, or 0 when there is no usable state."""
        try:
            raw = json.loads(self._watermark_path.read_text(encoding="utf-8"))
            return max(0, int(raw["last_rowid"]))
        except FileNotFoundError:
            return 0
        except (OSError, ValueError, TypeError, KeyError):
            # Corrupt state re-exports rather than stalling: a duplicate
            # trace costs disk, a skipped one is unrecoverable.
            log.warning(
                "GoldenTraceExporter: unreadable watermark at %s — "
                "restarting the export cursor from 0.",
                self._watermark_path, exc_info=True,
            )
            return 0

    def _write_watermark(self, last_rowid: int, count: int) -> None:
        path = self._watermark_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-replace so a crash mid-write cannot leave a
            # truncated cursor that silently skips traces.
            tmp = path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps({"last_rowid": int(last_rowid), "last_count": int(count)}),
                encoding="utf-8",
            )
            tmp.replace(path)
        except OSError:
            # Non-fatal: the next cycle re-exports from the old watermark.
            # Duplicates, not gaps — the safe failure direction.
            log.warning(
                "GoldenTraceExporter: could not persist watermark to %s",
                path, exc_info=True,
            )

    async def run_once(self) -> str | None:
        """Export one batch of NEW traces. Returns the path, or None.

        Incremental by rowid watermark. The previous implementation exported
        the most recent ``limit`` rows every cycle — and the loop runs one
        cycle immediately at startup — so every daemon restart rewrote the
        same trailing traces to a fresh timestamped file. The live corpus had
        279 files carrying 18 distinct payloads.

        Exporting at startup is kept deliberately: with a watermark it costs
        nothing when there is nothing new, and it means a restart picks up
        the backlog promptly instead of waiting out a 24h interval.
        """
        since = self._read_watermark()
        result = None
        try:
            result = self._telemetry.export_new_golden_traces(
                since_rowid=since,
                limit=self._limit,
                format=self._format,
                output_dir=self._output_dir,
            )
        except Exception:
            log.exception("GoldenTraceExporter: export call failed")
            return None

        self._cycle_count += 1
        if result is None:
            log.debug("GoldenTraceExporter: no new golden traces since rowid %d", since)
            return None

        self._write_watermark(result.last_rowid, result.count)
        path_str = str(result.path)
        self._last_path = path_str
        log.info(
            "Golden traces exported: %s (%d new, rowid %d→%d)",
            path_str, result.count, since, result.last_rowid,
        )

        if self._bus is not None:
            try:
                await self._bus.emit(ActivitySignal(
                    kind="golden_traces_exported",
                    payload={
                        "path": path_str,
                        "limit": self._limit,
                        "count": result.count,
                        "last_rowid": result.last_rowid,
                    },
                    source="golden_trace_exporter",
                ))
            except Exception:
                log.debug(
                    "GoldenTraceExporter: signal emission failed",
                    exc_info=True,
                )
        return path_str

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def last_path(self) -> str | None:
        return self._last_path
