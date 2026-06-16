"""CodingLiveStream — tail telemetry.db per coding run, fan rounds over the SignalBus.

The telemetry-write-timing survey (``audits/20260616T014041Z-...``) established
that the agent loop commits one ``subsystem_runs`` row per round, immediately —
so the daemon does NOT need the coding subprocess to talk to it. It TAILS
``telemetry.db`` for ``subsystem='agent_loop' AND session_id='coding:<id>'`` and
emits a ``coding_round`` SignalBus event per new row. The shared WS bridge
(``ws_server.WebSocketBridge`` subscribes ``"*"``) already fans SignalBus events
to authed clients, so this rides the SAME framing — a new event kind, not a
parallel socket.

Lifecycle:
- **start** — :meth:`start_tail` is called from ``POST /api/code`` at launch,
  which already knows ``coding:<task_id>`` (no telemetry scanning).
- **stop** — the managed task's ``task_completed`` / ``task_failed`` SignalBus
  event (authoritative "done") triggers a final drain, one ``coding_complete``
  event with the verdict, and poller teardown.

Read-only and harm-free: the tail connection uses the shared WAL + busy_timeout
setup (``connect_telemetry_db``) and only ever SELECTs. A poll cadence of
~300ms feels instant at this round cadence (survey gaps 0.6–3.3s) at negligible
load — poll, not notify (sqlite update-hooks are unjustified here). A dead tail
NEVER harms the run: the run is a separate subprocess; the tail is pure
observation, and any tail error becomes a ``coding_stream_error`` event, not a
crash. Sprint ``feat/coding-livestream``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from prometheus.telemetry.db import connect_telemetry_db

log = logging.getLogger(__name__)

DEFAULT_DB_PATH = "~/.prometheus/telemetry.db"
DEFAULT_POLL_INTERVAL_S = 0.3
_CODING_SESSION_PREFIX = "coding:"

# The per-round columns a Live view renders. round_index/outcome/tokens/thinking/
# duration come straight from the row; stop_reason is parsed out of summary_json.
# (Per-round tool name and test pass/fail are NOT here — tool_calls has no
# session_id; that enrichment is a documented follow-up, not v1.)
_ROUND_SQL = (
    "SELECT rowid, round_index, outcome, input_tokens, output_tokens, "
    "thinking, duration_ms, summary_json, timestamp "
    "FROM subsystem_runs "
    "WHERE subsystem='agent_loop' AND session_id=? AND rowid > ? "
    "ORDER BY rowid ASC"
)


class CodingLiveStream:
    """Per-coding-run telemetry tailer → SignalBus fan-out. See module docstring."""

    def __init__(
        self,
        signal_bus: Any | None,
        *,
        db_path: str | Path = DEFAULT_DB_PATH,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    ) -> None:
        self._signal_bus = signal_bus
        self._db_path = Path(db_path).expanduser()
        self._poll_interval = poll_interval_s
        # session_id -> running poller task / its stop signal / last seen rowid.
        self._pollers: dict[str, asyncio.Task] = {}
        self._stops: dict[str, asyncio.Event] = {}
        self._last_rowid: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def subscribe_lifecycle(self) -> None:
        """Subscribe to managed-task completion so runs stop tailing on their own.

        The BackgroundTaskManager emits ``task_completed`` / ``task_failed`` with
        the task's ``session_id`` (we set it to ``coding:<id>`` at launch), so
        these are self-correlating to the tailed run.
        """
        if self._signal_bus is None:
            return
        self._signal_bus.subscribe("task_completed", self._on_task_terminal)
        self._signal_bus.subscribe("task_failed", self._on_task_terminal)

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    def start_tail(self, session_id: str) -> None:
        """Begin tailing a coding run. Idempotent (no duplicate poller per run).

        Called from ``POST /api/code`` right after launch. Requires a running
        event loop (it schedules the poller as a task); the route handler runs
        on the web server's loop, so this holds in production.
        """
        if not session_id.startswith(_CODING_SESSION_PREFIX):
            log.warning(
                "CodingLiveStream.start_tail: ignoring non-coding session_id %r",
                session_id,
            )
            return
        if session_id in self._pollers:
            return
        stop = asyncio.Event()
        self._stops[session_id] = stop
        self._last_rowid[session_id] = 0
        self._pollers[session_id] = asyncio.create_task(
            self._poll_loop(session_id, stop), name=f"coding-tail:{session_id}"
        )

    # ------------------------------------------------------------------
    # The poller
    # ------------------------------------------------------------------

    async def _poll_loop(self, session_id: str, stop: asyncio.Event) -> None:
        """Poll for new rows until stopped, then drain once more. Fail-loud."""
        conn = None
        try:
            conn = connect_telemetry_db(self._db_path)
            while not stop.is_set():
                await self._drain(session_id, conn)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=self._poll_interval)
                except asyncio.TimeoutError:
                    pass
            # Final drain after the stop signal: capture rows committed between
            # the last poll and the terminal event so none is dropped.
            await self._drain(session_id, conn)
        except Exception as exc:  # noqa: BLE001 — tail must fail loud, never freeze
            log.error(
                "CodingLiveStream: tail failed for %s — emitting coding_stream_error",
                session_id, exc_info=True,
            )
            await self._emit(
                "coding_stream_error",
                {"session_id": session_id, "error": f"{type(exc).__name__}: {exc}"},
            )
        finally:
            if conn is not None:
                conn.close()

    async def _drain(self, session_id: str, conn: Any) -> None:
        """Emit a ``coding_round`` for every new row, advancing the cursor AFTER
        each successful emit so a mid-drain failure never skips a row (the next
        read re-sees it) and never duplicates one (the cursor already passed the
        emitted rows)."""
        rows = conn.execute(
            _ROUND_SQL, (session_id, self._last_rowid.get(session_id, 0))
        ).fetchall()
        for row in rows:
            await self._emit("coding_round", self._round_payload(session_id, row))
            self._last_rowid[session_id] = row[0]  # rowid — advance only post-emit

    @staticmethod
    def _round_payload(session_id: str, row: Any) -> dict[str, Any]:
        (_rowid, round_index, outcome, in_tok, out_tok,
         thinking, duration_ms, summary_json, timestamp) = row
        stop_reason = None
        if summary_json:
            try:
                stop_reason = json.loads(summary_json).get("stop_reason")
            except Exception:
                stop_reason = None
        return {
            "session_id": session_id,
            "round_index": round_index,
            "outcome": outcome,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "thinking": None if thinking is None else bool(thinking),
            "duration_ms": duration_ms,
            "stop_reason": stop_reason,
            "timestamp": timestamp,
        }

    # ------------------------------------------------------------------
    # Stop (managed-task terminal event → final drain + coding_complete)
    # ------------------------------------------------------------------

    async def _on_task_terminal(self, signal: Any) -> None:
        """Handle task_completed / task_failed for a tailed coding run."""
        payload = getattr(signal, "payload", None) or {}
        session_id = payload.get("session_id") or ""
        if not session_id.startswith(_CODING_SESSION_PREFIX):
            return  # not a coding run — ignore other managed tasks
        if session_id not in self._pollers:
            return  # not tailing it (already torn down / never started)
        await self._stop_tail(session_id, payload)

    async def _stop_tail(self, session_id: str, terminal: dict[str, Any]) -> None:
        """Stop the poller (with a final drain), then emit exactly one complete."""
        stop = self._stops.get(session_id)
        task = self._pollers.get(session_id)
        if stop is not None:
            stop.set()
        if task is not None:
            try:
                await task  # poller does its final drain, then returns
            except Exception:  # noqa: BLE001 — already logged inside the loop
                pass
        # All rounds drained → the run is done. One terminal event with the verdict.
        await self._emit("coding_complete", {
            "session_id": session_id,
            "outcome": "success" if terminal.get("status") == "completed" else "failed",
            "task_status": terminal.get("status"),
            "exit_code": terminal.get("exit_code", terminal.get("return_code")),
            "error": terminal.get("error"),
        })
        self._pollers.pop(session_id, None)
        self._stops.pop(session_id, None)
        self._last_rowid.pop(session_id, None)

    async def stop_all(self) -> None:
        """Tear down every poller (daemon shutdown). No coding_complete emitted."""
        for stop in list(self._stops.values()):
            stop.set()
        for task in list(self._pollers.values()):
            try:
                await task
            except Exception:  # noqa: BLE001
                pass
        self._pollers.clear()
        self._stops.clear()
        self._last_rowid.clear()

    @property
    def active_sessions(self) -> list[str]:
        """Session ids currently being tailed (empty when no run is live)."""
        return list(self._pollers)

    # ------------------------------------------------------------------
    # Emit
    # ------------------------------------------------------------------

    async def _emit(self, kind: str, payload: dict[str, Any]) -> None:
        if self._signal_bus is None:
            return
        from prometheus.sentinel.signals import ActivitySignal

        await self._signal_bus.emit(
            ActivitySignal(kind=kind, payload=payload, source="coding_mode")
        )
