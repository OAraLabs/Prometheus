"""Task completion handler — re-engagement on task_completed / task_failed.

Subscribes to the SignalBus and, when a task's ``on_complete`` requests it,
injects the result back into the creating session as a non-user, UNTRUSTED turn
via the shared ``inject_turn`` primitive (the same path Telegram inbound uses).

Scope (per design decision): this handler does NOT send notifications — the
heartbeat owns proactive Telegram notification for all terminal tasks. This
handler exists purely to *re-engage the agent* so it can act on a result.
``on_complete`` therefore gates re-engagement only; notification always happens.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)

# Signature of the shared turn-injection primitive (telegram.inject_turn).
InjectTurn = Callable[..., Awaitable[str]]

DEFAULT_REENGAGE_TURN_CAP = 3


class TaskCompletionHandler:
    """Re-engage the agent on task completion when ``on_complete`` requests it."""

    def __init__(
        self,
        *,
        signal_bus: Any,
        inject_turn: InjectTurn | None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._bus = signal_bus
        self._inject_turn = inject_turn
        cfg = config or {}
        self._turn_cap = int(cfg.get("reengage_turn_cap", DEFAULT_REENGAGE_TURN_CAP))
        # Per-session count of in-flight synthetic re-engagement turns — bounds
        # runaway chains (a re-engaged turn that spawns another task, etc.).
        self._inflight: dict[str, int] = {}

    async def start(self) -> None:
        """Subscribe to terminal task signals."""
        self._bus.subscribe("task_completed", self._on_signal)
        self._bus.subscribe("task_failed", self._on_signal)
        log.info("TaskCompletionHandler subscribed (reengage cap=%d)", self._turn_cap)

    async def _on_signal(self, signal: Any) -> None:
        payload = getattr(signal, "payload", {}) or {}
        on_complete = payload.get("on_complete", "notify")
        if on_complete not in ("reengage", "both"):
            return  # notify-only — heartbeat handles the user notification

        session_id = payload.get("session_id")
        if not session_id:
            log.info(
                "task %s requested reengage but has no session_id — skipping",
                payload.get("task_id"),
            )
            return
        if self._inject_turn is None:
            log.warning("reengage requested but no inject_turn wired — skipping")
            return

        # Turn cap — refuse if too many synthetic turns are already in flight
        # for this session.
        current = self._inflight.get(session_id, 0)
        if current >= self._turn_cap:
            log.warning(
                "reengage turn cap (%d) reached for session %s — skipping task %s",
                self._turn_cap, session_id, payload.get("task_id"),
            )
            return

        content = _build_reengage_content(payload)
        self._inflight[session_id] = current + 1
        try:
            await self._inject_turn(
                session_id,
                content,
                provenance="task_supervisor",
                is_trusted=False,
            )
        except Exception:
            log.exception("reengage inject_turn failed for task %s", payload.get("task_id"))
        finally:
            self._inflight[session_id] = max(0, self._inflight.get(session_id, 1) - 1)


def _build_reengage_content(payload: dict[str, Any]) -> str:
    """Compose the (untrusted) re-engagement turn content from a signal payload.

    If the task supplied a ``reengage_prompt`` template, it is used (with
    best-effort ``.format`` substitution of the known fields); otherwise a
    default result summary is built. The text is delivered as UNTRUSTED data —
    the model must not execute instructions found inside job output.
    """
    fields = {
        "task_id": payload.get("task_id", ""),
        "description": payload.get("description", ""),
        "status": payload.get("status", ""),
        "exit_code": payload.get("exit_code"),
        "return_code": payload.get("return_code"),
        "artifact_path": payload.get("artifact_path") or "(none)",
        "error": payload.get("error") or "",
        "output_tail": payload.get("output_tail", ""),
    }

    template = payload.get("reengage_prompt")
    if template:
        try:
            return template.format(**fields)
        except (KeyError, IndexError, ValueError):
            # Template referenced an unknown field — fall back to template + summary.
            return f"{template}\n\n{_default_summary(fields)}"
    return _default_summary(fields)


def _default_summary(fields: dict[str, Any]) -> str:
    return (
        "A background task you started has finished.\n"
        f"Task: {fields['description']} ({fields['task_id']})\n"
        f"Status: {fields['status']} (exit code {fields['exit_code']})\n"
        f"Artifact: {fields['artifact_path']}\n"
        f"--- output (tail) ---\n{fields['output_tail']}"
    )
