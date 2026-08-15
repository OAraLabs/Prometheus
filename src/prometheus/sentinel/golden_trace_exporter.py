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

# How many preceding messages to carry as a trace's input half. Enough for
# the call to be motivated, bounded so one long session cannot dominate the
# corpus or blow up example length.
CONTEXT_MESSAGE_LIMIT = 12

# Roles a training example's input half may contain. Tool results are kept —
# a call that follows one is usually a REACTION to it, and dropping them
# would make those examples look unmotivated.
_CONTEXT_ROLES = frozenset({"system", "user", "assistant", "tool"})


def lcm_context_resolver(store: object, *, limit: int = CONTEXT_MESSAGE_LIMIT):
    """Build a context resolver backed by the LCM conversation store.

    Recovers the conversation that PRECEDED a golden tool call, which is the
    input half of a trainable example. Telemetry deliberately stores only a
    ``session_id`` reference rather than a copy of the conversation, so this
    is where the two halves are rejoined.

    Returns a callable suitable for ``export_*_golden_traces(context_resolver=)``,
    or None when no store is available — callers then export nothing rather
    than emitting context-free rows, which is the honest failure.
    """
    if store is None:
        return None

    def resolve(trace: dict) -> list[dict[str, str]]:
        session_id = trace.get("session_id")
        ts = trace.get("timestamp")
        if not session_id or ts is None:
            # Rows written before session_id capture shipped. Unrecoverable
            # by construction — the conversation was never referenced.
            return []
        try:
            messages = store.get_messages(session_id, limit=500)
        except Exception:
            log.debug("LCM lookup failed for session %s", session_id, exc_info=True)
            return []

        out: list[dict[str, str]] = []
        for msg in messages:
            # STRICTLY BEFORE the call: a message at or after it can include
            # the tool's own result, which would leak the answer back into
            # the prompt — the exact defect this export shape replaced.
            if getattr(msg, "timestamp", 0) >= ts:
                continue
            role = getattr(msg, "role", "")
            content = getattr(msg, "content", "")
            if role not in _CONTEXT_ROLES or not content:
                continue
            out.append({"role": role, "content": content})

        # Trim trailing assistant turns. The example's TARGET is an assistant
        # turn, and the model's prose preamble for that same turn is also
        # written to LCM — so leaving these in can put the preamble both in
        # the context and in the target's `content`, and trains the model to
        # continue its own turn rather than to answer a request. Context
        # should end where the model was last asked to act.
        while out and out[-1]["role"] == "assistant":
            out.pop()
        return out[-limit:]

    return resolve


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
        conversation_store: object | None = None,
    ) -> None:
        self._telemetry = telemetry
        self._bus = signal_bus
        # Without a store the export has no input half to pair with each
        # call, so every row is skipped. Passed in rather than imported so
        # this module keeps no hard dependency on the memory layer.
        self._context_resolver = lcm_context_resolver(conversation_store)
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
                context_resolver=self._context_resolver,
            )
        except Exception:
            log.exception("GoldenTraceExporter: export call failed")
            return None

        self._cycle_count += 1
        if result is None:
            log.debug("GoldenTraceExporter: no new golden traces since rowid %d", since)
            return None

        # Advance the watermark even when nothing was written: those rows are
        # permanently untrainable (context gone), so leaving the cursor behind
        # them would re-read the same dead batch every cycle forever.
        self._write_watermark(result.last_rowid, result.count)
        if result.path is None:
            log.info(
                "GoldenTraceExporter: advanced past %d untrainable traces "
                "(rowid → %d) without writing a file",
                result.count or 0, result.last_rowid,
            )
            return None

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
