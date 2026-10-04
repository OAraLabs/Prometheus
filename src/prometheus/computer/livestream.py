"""The cockpit's action log — every desktop step, live, on every Beacon.

computer-use v1.1 PR 6 (design §5.2). Two layers, one emitter:

LAYER 2 — THE DURABLE LOG (SignalBus kinds)
-------------------------------------------
``computer_binding``, ``computer_task_started``, ``computer_step``,
``computer_task_ended`` and ``computer_stream_error`` go out on the SignalBus,
which persists them to ``signal_events`` (backfilled by ``GET
/api/events/recent?types=…&session_id=…&since=…``) and fans them to every
socket. :data:`COMPUTER_FRAME_KINDS` is THE declaration, and ``web.ws_server``
promotes from it — so a kind added here is a first-class frame type by
construction. An unpromoted kind ships as a generic ``sentinel_signal`` that
every client gate keyed on ``type`` silently misses: that is how
``coding_tool`` and ``task_completed`` went dark, and how iOS dropped frames.
``tests/test_computer_livestream.py`` pins the tuple to this file's
``_emit`` call sites and ws_server to the tuple.

LAYER 1 — TODAY'S CHAT TIMELINE (bridge frames)
-----------------------------------------------
So a client with no ``computer_*`` decoder still shows the task: a
``tool_call_start``/``tool_call_end`` pair for the task (``tool_name:
"computer_task"``, ``inputs.origin: "user_task"`` — it is NOT a model's
call) and one per step, ``agent_progress`` while it runs, and ``chat_done``
at the end ONLY when no chat turn is live in the session (it would clear
iOS's turn-in-flight mid-turn). Never ``turn_completed``: that is an APNs
summary push.

THE CONTENT POLICY (§5.2.2) — enforced by a test over every emitted,
persisted and backfilled payload
--------------------------------------------------------------------
Never an element token, a snapshot id, a pid or window id, a screenshot, or
the label of a row the chooser did NOT pick. The typed text is
``{"text_chars": N}``. App text that does cross (the chosen element's
description) is capped and marked ``app_text: true``. The one carve-out is
the LIVE ``approval_pending`` frame, which must show the text being approved
— its stored copy carries ``text_chars`` instead (``sentinel.signals``).

The log never harms the task: every emission is caught, and a failure is a
``computer_stream_error`` frame, never an exception in the run.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

#: Every SignalBus kind this module emits — THE declaration ws_server promotes.
COMPUTER_FRAME_KINDS = (
    "computer_binding",
    "computer_task_started",
    "computer_step",
    "computer_task_ended",
    "computer_stream_error",
)

SOURCE = "computer"
DEFAULT_KEEP_PER_SESSION = 200
_REASON_CAP = 200


def keep_per_session_from(config: Any, errors: list[str] | None = None) -> int:
    """``computer_use.action_log.keep_per_session`` (a positive int)."""
    block = ((config or {}).get("computer_use") or {}) if isinstance(config, dict) else {}
    raw = ((block.get("action_log") or {}) if isinstance(block, dict) else {}).get(
        "keep_per_session", DEFAULT_KEEP_PER_SESSION)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        if errors is not None:
            errors.append("computer_use.action_log.keep_per_session must be a "
                          f"positive integer; using {DEFAULT_KEEP_PER_SESSION}")
        return DEFAULT_KEEP_PER_SESSION
    return raw


def _cap(text: Any, cap: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= cap else text[: cap - 1] + "…"


class ComputerLiveStream:
    """The runner's listener. See the module docstring."""

    def __init__(
        self,
        bus: Any,
        *,
        bridge: Any = None,
        telemetry: Any = None,
        keep_per_session: int = DEFAULT_KEEP_PER_SESSION,
        progress_s: float = 3.0,
    ) -> None:
        self._bus = bus
        self._bridge = bridge
        self._telemetry = telemetry
        self.keep_per_session = keep_per_session
        self._progress_s = progress_s
        self._seq: dict[str, int] = {}
        self._tickers: dict[str, asyncio.Task] = {}

    # ── plumbing ────────────────────────────────────────────────────────

    async def _emit(self, kind: str, payload: dict[str, Any]) -> None:
        if self._bus is None:
            return
        try:
            from prometheus.sentinel.signals import ActivitySignal

            await self._bus.emit(ActivitySignal(kind=kind, payload=payload,
                                                source=SOURCE))
        except Exception as exc:  # noqa: BLE001 - the log never ends a task
            logger.warning("computer action log: %s not delivered: %s", kind, exc)
            if kind != "computer_stream_error":
                await self._emit("computer_stream_error", {
                    "session_id": payload.get("session_id"),
                    "task_id": payload.get("task_id"),
                    "detail": f"{kind} not delivered ({exc.__class__.__name__})",
                })

    async def _frame(self, kind: str, payload: dict[str, Any]) -> None:
        """Layer 1: a bridge frame, never persisted. Best effort."""
        bridge = self._bridge
        if bridge is None:
            return
        try:
            await bridge.broadcast({"type": kind, "timestamp": time.time(),
                                    "payload": payload})
        except Exception:  # noqa: BLE001
            logger.debug("computer layer-1 frame not sent", exc_info=True)

    def _chat_turn_live(self, session_id: str) -> bool:
        turns = getattr(self._bridge, "_turn_tasks", None) or {}
        turn = turns.get(session_id)
        return turn is not None and not turn.done()

    def _next_seq(self, task_id: str) -> int:
        self._seq[task_id] = self._seq.get(task_id, 0) + 1
        return self._seq[task_id]

    def _prune(self, session_id: str) -> None:
        tel = self._telemetry
        if tel is None or not session_id or self.keep_per_session <= 0:
            return
        try:
            tel.prune_signal_events(kinds=COMPUTER_FRAME_KINDS,
                                    session_id=session_id,
                                    keep=self.keep_per_session)
        except Exception:  # noqa: BLE001
            logger.debug("computer action log prune failed", exc_info=True)

    # ── the binding ─────────────────────────────────────────────────────

    async def binding(self, binding: Any, state: str) -> None:
        payload = {
            "session_id": binding.session_id,
            "state": state,
            "target": binding.target,
            "app": binding.app,
            "scope": binding.scope,
            "set_by": {"surface": binding.set_by.get("surface", "")},
            "expires_at": binding.expires_at,
        }
        if state == "on":
            payload["describes"] = binding.sentence()
            payload["covers"] = ["click", "press_key:tab", "press_key:escape"]
        await self._emit("computer_binding", payload)
        if state == "off":
            self._prune(binding.session_id)

    # ── a task ──────────────────────────────────────────────────────────

    async def task_started(self, task: Any, *, limits: Any, chooser: str) -> None:
        await self._emit("computer_task_started", {
            "session_id": task.session_id,
            "task_id": task.task_id,
            "target": task.target,
            "app": task.app,
            "goal": task.goal,
            "chooser": chooser,
            "started_by": {"surface": task.surface},
            "limits": {"max_steps": limits.max_steps,
                       "max_seconds": limits.max_seconds,
                       "max_approvals": limits.max_approvals},
            "text_chars": len(task.text) if task.text else 0,
        })
        await self._frame("tool_call_start", {
            "session_id": task.session_id, "call_id": task.task_id,
            "tool_name": "computer_task",
            "inputs": {"goal": task.goal, "app": task.app,
                       "origin": "user_task"},
        })
        if self._bridge is not None and self._progress_s > 0:
            self._tickers[task.task_id] = asyncio.create_task(
                self._tick(task), name=f"computer-progress-{task.task_id}")

    async def _tick(self, task: Any) -> None:
        started = time.monotonic()
        while task.status == "running":
            await asyncio.sleep(self._progress_s)
            if task.status != "running" or self._chat_turn_live(task.session_id):
                continue
            await self._frame("agent_progress", {
                "session_id": task.session_id, "phase": "tool",
                "tool_name": "computer_task", "round": 0, "chars": 0,
                "tool_calls": task.steps,
                "elapsed_s": round(time.monotonic() - started, 1),
            })

    async def step(
        self,
        task: Any,
        *,
        status: str,
        verb: str | None = None,
        description: str | None = None,
        extent: str = "",
        consent: str | None = None,
        approval_request_id: str | None = None,
        chooser: dict[str, Any] | None = None,
        verified: bool | None = None,
        after_stop: bool = False,
        candidates_offered: int = 0,
        duration_ms: int = 0,
        reason: str = "",
    ) -> None:
        seq = self._next_seq(task.task_id)
        executed = status in ("executed", "in_flight_at_stop")
        effect = (("CONFIRMED" if verified else "UNVERIFIABLE")
                  if executed else None)
        action = ({"verb": verb, "description": _cap(description, 120),
                   "app_text": True} if verb else None)
        await self._emit("computer_step", {
            "session_id": task.session_id,
            "task_id": task.task_id,
            "seq": seq,
            "status": status,
            "action": action,
            "extent": extent or None,
            "consent": consent,
            "approval_request_id": approval_request_id,
            "chooser": chooser,
            "effect": effect,
            "verified": True if verified else None,
            "after_stop": after_stop,
            "candidates_offered": candidates_offered,
            "duration_ms": duration_ms,
            "reason": _cap(reason, _REASON_CAP),
        })
        if verb and status != "awaiting_approval":
            call_id = f"{task.task_id}:{seq}"
            await self._frame("tool_call_start", {
                "session_id": task.session_id, "call_id": call_id,
                "tool_name": f"computer_{verb}",
                "inputs": {"description": _cap(description, 120),
                           "extent": extent, "origin": "user_task"},
            })
            line = {"CONFIRMED": "done — the window changed",
                    "UNVERIFIABLE": "done — no visible change"}.get(effect or "",
                                                                   _cap(reason, 120))
            if after_stop:
                line = "in flight at the stop — may have landed"
            await self._frame("tool_call_end", {
                "session_id": task.session_id, "call_id": call_id,
                "tool_name": f"computer_{verb}", "success": executed,
                "result": line,
            })

    async def task_ended(self, task: Any, *, summary: str) -> None:
        ticker = self._tickers.pop(task.task_id, None)
        if ticker is not None:
            ticker.cancel()
        duration_ms = int(((task.ended_at or time.time()) - task.started_at) * 1000)
        await self._emit("computer_task_ended", {
            "session_id": task.session_id,
            "task_id": task.task_id,
            "outcome": task.outcome,
            "reason": _cap(task.reason, _REASON_CAP),
            "steps": task.steps,
            "approvals": task.approvals,
            "in_flight_at_stop": task.in_flight_at_stop,
            "duration_ms": duration_ms,
        })
        await self._frame("tool_call_end", {
            "session_id": task.session_id, "call_id": task.task_id,
            "tool_name": "computer_task", "success": task.outcome == "done",
            "result": summary,
        })
        if not self._chat_turn_live(task.session_id):
            await self._frame("chat_done", {
                "session_id": task.session_id,
                # A HANDLE, not a rowid: iOS's ChatDonePayload requires
                # message_id, and a frame it cannot decode is dropped — the
                # very failure this module exists to prevent. No turn was
                # persisted, so there is no rowid (and no row_id key).
                "message_id": f"computer:{task.task_id}",
                "interrupted": task.outcome == "stopped",
                "origin": "user_task",
            })
        self._seq.pop(task.task_id, None)
        self._prune(task.session_id)
