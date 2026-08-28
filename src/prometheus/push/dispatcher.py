"""Push fan-out decisions (GRAFT-MOBILE-BRIDGE 2).

The decision table, verbatim from the spec:

  approval_pending             → push ALWAYS, even with a live socket — the
                                 phone is the point. Category APPROVAL,
                                 time-sensitive, mutable-content.
  turn_completed               → only to devices with NO live WS client. TURN.
  task_completed / task_failed → only when no live WS client. TASK.
  coding_complete              → only when no live WS client. TASK.

"Has a live WS client" is PER DEVICE (the bridge's websocket→identity map,
which is why Piece 1 gates this piece): a desktop being open must not
silently suppress the phone's push.

Failure accounting, not retries: each notification is one attempt. A 410
clears the device's registration permanently and logs once. Other failures
increment ``push_failures``; at ``FAILURE_MUTE_THRESHOLD`` consecutive
failures the device is muted (skipped) until any success resets the count —
that is the whole backoff, and it cannot storm because nothing ever retries.

Live Activity updates ride ``agent_progress`` (bridge-emitted every 3s) but
are throttled to one push per session per ``ACTIVITY_THROTTLE_SECONDS`` —
Apple rate-limits liveactivity pushes; the 3s cadence is for a live socket.
``chat_done`` sends one final ``end`` event and drops the token row.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

FAILURE_MUTE_THRESHOLD = 8
ACTIVITY_THROTTLE_SECONDS = 10.0

# Signal kinds → notification category. approval_pending is handled apart.
_QUIET_KINDS = {
    "turn_completed": "TURN",
    "task_completed": "TASK",
    "task_failed": "TASK",
    "coding_complete": "TASK",
}


def first_line(text: str, cap: int = 120) -> str:
    line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    return line[:cap]


class PushDispatcher:
    """Decides who gets pushed what. The sender sends; the store remembers."""

    def __init__(self, store: Any, sender: Any, bridge: Any,
                 now: Any = time.time) -> None:
        self._store = store
        self._sender = sender
        self._bridge = bridge
        self._now = now
        # session_id -> wall time of the last liveactivity push.
        self._activity_pushed_at: dict[str, float] = {}
        # Devices whose 410 has been logged — "logs once", per the spec.
        self._unregistered_logged: set[str] = set()

    # ------------------------------------------------------------------
    # Bus signals
    # ------------------------------------------------------------------

    async def on_signal(self, signal: Any) -> None:
        try:
            kind = getattr(signal, "kind", "")
            payload = getattr(signal, "payload", None) or {}
            if kind == "approval_pending":
                await self._push_approval(payload)
            elif kind in _QUIET_KINDS:
                await self._push_quiet(kind, payload)
        except Exception:
            # Push is a delivery nicety; it must never take the signal bus
            # down with it.
            logger.warning("push dispatch failed", exc_info=True)

    async def _push_approval(self, payload: dict) -> None:
        body = {
            "aps": {
                "alert": {
                    "title": "Approval required",
                    "body": f"{payload.get('tool_name', 'a tool')} — "
                            f"{first_line(str(payload.get('description') or ''))}",
                },
                "category": "APPROVAL",
                "sound": "default",
                # Correct HERE and nowhere else — an approval blocks a
                # running turn.
                "interruption-level": "time-sensitive",
                "mutable-content": 1,
            },
            "request_id": payload.get("request_id"),
            "tool_name": payload.get("tool_name"),
            "expires_at": payload.get("expires_at"),
        }
        for target in self._store.push_targets():
            await self._deliver(target, body)

    async def _push_quiet(self, kind: str, payload: dict) -> None:
        live = self._live_device_ids()
        category = _QUIET_KINDS[kind]
        title = {"TURN": "Turn complete", "TASK": "Task finished"}[category]
        if kind == "task_failed":
            title = "Task failed"
        detail = first_line(str(payload.get("summary") or payload.get("title")
                                or payload.get("session_id") or ""))
        body = {
            "aps": {
                "alert": {"title": title, **({"body": detail} if detail else {})},
                "category": category,
                "sound": "default",
            },
            **({"session_id": payload["session_id"]} if payload.get("session_id") else {}),
            **({"task_id": payload["task_id"]} if payload.get("task_id") else {}),
        }
        for target in self._store.push_targets():
            # The spec's core rule: a device watching over its socket needs no
            # push; a device whose socket died (backgrounded phone) does.
            if target.id in live:
                continue
            await self._deliver(target, body)

    # ------------------------------------------------------------------
    # Live Activity (bridge-emitted events, not bus signals)
    # ------------------------------------------------------------------

    async def on_agent_progress(self, session_id: str, payload: dict) -> None:
        try:
            now = self._now()
            if now - self._activity_pushed_at.get(session_id, 0.0) < ACTIVITY_THROTTLE_SECONDS:
                return
            targets = self._store.activity_targets(session_id)
            if not targets:
                return
            self._activity_pushed_at[session_id] = now
            body = {"aps": {
                "timestamp": int(now),
                "event": "update",
                "content-state": {
                    "phase": payload.get("phase", ""),
                    "round": payload.get("round", 0),
                    "elapsedS": payload.get("elapsed_s", 0),
                    "toolCalls": payload.get("tool_calls", 0),
                    "chars": payload.get("chars", 0),
                },
            }}
            await self._deliver_activity(targets, body)
        except Exception:
            logger.warning("liveactivity push failed", exc_info=True)

    async def on_chat_done(self, session_id: str, payload: dict) -> None:
        try:
            targets = self._store.activity_targets(session_id)
            if not targets:
                return
            now = self._now()
            body = {"aps": {
                "timestamp": int(now),
                "event": "end",
                "dismissal-date": int(now),
                "content-state": {"phase": "done", "round": payload.get("round", 0),
                                  "elapsedS": payload.get("elapsed_s", 0),
                                  "toolCalls": 0, "chars": 0},
            }}
            await self._deliver_activity(targets, body)
            self._activity_pushed_at.pop(session_id, None)
            for target, _token in targets:
                self._store.clear_activity_token(target.id, session_id)
        except Exception:
            logger.warning("liveactivity end push failed", exc_info=True)

    async def _deliver_activity(self, targets: list, body: dict) -> None:
        for target, activity_token in targets:
            if target.push_failures >= FAILURE_MUTE_THRESHOLD:
                continue
            result = await self._sender.send(
                apns_token=activity_token, environment=target.environment,
                payload=body, push_type="liveactivity", priority=5,
                topic=f"{target.bundle_id or self._sender.config.topic}.push-type.liveactivity",
            )
            self._account(target, result)

    # ------------------------------------------------------------------

    def _live_device_ids(self) -> set[str]:
        identities = getattr(self._bridge, "_ws_identity", {}) or {}
        return {i.id for i in identities.values() if getattr(i, "id", "") != "global"}

    async def _deliver(self, target: Any, body: dict) -> None:
        if target.push_failures >= FAILURE_MUTE_THRESHOLD:
            return  # muted until a success resets the count — no storm
        result = await self._sender.send(
            apns_token=target.apns_token, environment=target.environment,
            payload=body, push_type="alert", priority=10,
            topic=target.bundle_id or None,
        )
        self._account(target, result)

    def _account(self, target: Any, result: Any) -> None:
        if result.outcome == "ok":
            if target.push_failures:
                self._store.reset_push_failures(target.id)
            self._unregistered_logged.discard(target.id)
        elif result.outcome == "unregistered":
            # Permanent, and logged ONCE.
            self._store.clear_push(target.id)
            if target.id not in self._unregistered_logged:
                self._unregistered_logged.add(target.id)
                logger.info("APNs says device %s is unregistered — push cleared",
                            target.id)
        else:
            count = self._store.record_push_failure(target.id)
            if count == FAILURE_MUTE_THRESHOLD:
                logger.warning("device %s muted after %d consecutive push failures",
                               target.id, count)
