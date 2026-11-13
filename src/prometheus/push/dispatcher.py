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

import base64
import binascii
import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from prometheus.push.job_activity import (
    STALE_AFTER_SECONDS,
    content_state,
    tool_detail,
    turn_end_aps,
    turn_end_phase,
)

logger = logging.getLogger(__name__)

FAILURE_MUTE_THRESHOLD = 8
ACTIVITY_THROTTLE_SECONDS = 10.0
#: How many approval requests the dispatcher remembers it pushed, and to which devices, so that the end of one can
#: take its notification back (``_push_resolved``). Requests expire in minutes; this is far more than are ever open.
PUSHED_REQUESTS_REMEMBERED = 512

# Signal kinds → notification category. approval_pending is handled apart.
_QUIET_KINDS = {
    "turn_completed": "TURN",
    "task_completed": "TASK",
    "task_failed": "TASK",
    "coding_complete": "TASK",
}


def _node_pub_now() -> str | None:
    from prometheus.config.node_identity import get_node_pubkey

    return get_node_pubkey()


def _is_node_pub(value: Any) -> bool:
    """Whether *value* is a node public key as the daemon states it everywhere (``/api/status``, the identity
    challenge): the base64 of exactly 32 raw bytes, written the way ``b64encode`` writes it."""
    if not isinstance(value, str) or len(value) != 44:
        return False
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return False
    return len(raw) == 32 and base64.b64encode(raw).decode("ascii") == value


#: APNs refuses a body over 4096 bytes (PayloadTooLarge), and an approval that never arrives blocks a turn. Every
#: caller-sized string a body carries is capped; ``test_push_carries_node_pub`` builds every body with long ones.
TOOL_NAME_CAP = 100
ID_CAP = 200


def _capped(value: Any, cap: int) -> Any:
    return value[:cap] if isinstance(value, str) else value


def first_line(text: str, cap: int = 120) -> str:
    line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    return line[:cap]


class PushDispatcher:
    """Decides who gets pushed what. The sender sends; the store remembers."""

    def __init__(self, store: Any, sender: Any, bridge: Any,
                 now: Any = time.time,
                 node_pub: Callable[[], str | None] | None = None) -> None:
        self._store = store
        self._sender = sender
        self._bridge = bridge
        self._now = now
        # Which node is pushing (see ``_named``). Injectable so a test need not own a key.
        self._node_pub = node_pub or _node_pub_now
        # session_id -> wall time of the last liveactivity push.
        self._activity_pushed_at: dict[str, float] = {}
        # session_id -> the turn's latest progress numbers (bounded, popped at its end).
        self._turn_last: OrderedDict[str, dict] = OrderedDict()
        # Devices whose 410 has been logged — "logs once", per the spec.
        self._unregistered_logged: set[str] = set()
        # request_id -> the devices its approval push reached (APNs said ok). Memory only, bounded, popped when the
        # request ends: what ``_push_resolved`` tells, and the only devices it tells.
        self._pushed_to: OrderedDict[str, set[str]] = OrderedDict()
        # request_id -> how it ended, for requests that ended while (or before) their approval push was still going
        # out: a device the fan-out reaches after the end is cleared at once instead of being left with a live Approve.
        self._ended: OrderedDict[str, str] = OrderedDict()
        # Whether a name is a registered tool (set by the launcher from the tool registry). A turn's ``detail`` is
        # a tool name only when it is one; None means nothing is: the generic word (push/job_activity.py).
        self.known_tool: Callable[[str], bool] | None = None
        # Told the moment a turn ends: (session_id, the turn's start, how it ended). The job watcher sets it
        # (push/job_activity.py), so a token registered just after the end still gets the end.
        self.turn_end_listener: Callable[[str, float | None, str], None] | None = None

    # ------------------------------------------------------------------
    # Bus signals
    # ------------------------------------------------------------------

    async def on_signal(self, signal: Any) -> None:
        try:
            kind = getattr(signal, "kind", "")
            payload = getattr(signal, "payload", None) or {}
            if kind == "approval_pending":
                await self._push_approval(payload)
            elif kind == "approval_resolved":
                await self._push_resolved(payload)
            elif kind in _QUIET_KINDS:
                await self._push_quiet(kind, payload)
        except Exception:
            # Push is a delivery nicety; it must never take the signal bus
            # down with it.
            logger.warning("push dispatch failed", exc_info=True)

    def node_pub(self) -> str | None:
        """This node's public key if it has one and it is well formed, else ``None``. Never raises."""
        try:
            value = self._node_pub()
        except Exception:
            logger.warning("push: the node key could not be read; pushing without it", exc_info=True)
            return None
        return value if _is_node_pub(value) else None

    def _named(self, body: dict) -> dict:
        """*body* with ``node_pub`` added: every push says which node sent it.

        A phone paired with several Prometheus instances gets pushes from each, and nothing else in a body says
        which. The node's public key is what it pinned when it paired (``/api/status``, the identity challenge).
        Public by nature, so it is safe in front of Apple; left out, never guessed at, when the node has no key.
        Added where each body is BUILT, not where it is sent, so the key-set review in
        ``tests/test_approval_push_fanout.py`` sees it.
        """
        pub = self.node_pub()
        return {**body, "node_pub": pub} if pub else body

    async def _push_approval(self, payload: dict) -> None:
        """Fan one pending approval out to EVERY registered device.

        ⚠ THE WIDEST READER OF ``ApprovalQueue.serialize_pending``'s dict, and
        the one its docstring omitted until 2026-09-19. The REST and WS
        surfaces answer someone already holding a token or a socket; this one
        delivers unsolicited to every phone that ever registered, through
        Apple.

        So the body below is a DELIBERATE NARROW SUBSET, not a projection that
        happens to be small. ``payload`` also carries ``arguments`` (redacted
        and truncated, but still the literal text a tool was about to type)
        and ``extents``; neither is read here and neither should be.
        ``test_approval_push_body_stays_narrow`` fails the build if this set
        changes, so widening it is a decision someone makes on purpose.

        ``node_pub`` (added on purpose; ``tests/test_push_carries_node_pub.py``) is not read from ``payload``: it
        is the node's own public key, added by ``_named`` to every body, so a phone paired with several Prometheus
        instances knows which one is pushing.
        """
        if payload.get("task_id") is not None:
            # A DESKTOP TASK'S PROMPT (computer-use v1.1 §5.2.2): no content
            # at all — not the tool, not the app, not the element, not the
            # text. And no APPROVAL category: a lock screen that cannot show
            # what is being approved must not offer to approve it. The phone
            # fetches the details over the tailnet when it is opened.
            desk = {
                "aps": {
                    "alert": {"title": "Approval required",
                              "body": "Prometheus needs a decision"},
                    "sound": "default",
                    "interruption-level": "time-sensitive",
                },
                "request_id": _capped(payload.get("request_id"), ID_CAP),
                "expires_at": payload.get("expires_at"),
            }
            desk = self._named(desk)
            for target in self._store.push_targets():
                await self._deliver_approval(target, desk, payload.get("request_id"))
            return
        body = {
            "aps": {
                "alert": {
                    "title": "Approval required",
                    "body": f"{_capped(payload.get('tool_name', 'a tool'), TOOL_NAME_CAP)} — "
                            f"{first_line(str(payload.get('description') or ''))}",
                },
                "category": "APPROVAL",
                "sound": "default",
                # Correct HERE and nowhere else — an approval blocks a
                # running turn.
                "interruption-level": "time-sensitive",
                "mutable-content": 1,
            },
            "request_id": _capped(payload.get("request_id"), ID_CAP),
            "tool_name": _capped(payload.get("tool_name"), TOOL_NAME_CAP),
            "expires_at": payload.get("expires_at"),
        }
        body = self._named(body)
        for target in self._store.push_targets():
            await self._deliver_approval(target, body, payload.get("request_id"))

    async def _deliver_approval(self, target: Any, body: dict, request_id: Any) -> None:
        """Deliver one approval push and, when APNs took it, remember that this device holds its notification."""
        if not (await self._deliver(target, body) and isinstance(request_id, str) and request_id):
            return
        if request_id in self._ended:
            # Answered elsewhere while the fan-out was still sending (the two run in different tasks).
            await self._send_resolved(target, request_id, self._ended[request_id])
            return
        self._pushed_to.setdefault(request_id, set()).add(target.id)
        self._pushed_to.move_to_end(request_id)
        while len(self._pushed_to) > PUSHED_REQUESTS_REMEMBERED:
            self._pushed_to.popitem(last=False)

    async def _push_resolved(self, payload: dict) -> None:
        """Take an approval notification back once the request has ended (Beacon iOS "Piece 16c").

        A request answered on the desktop or in Telegram, withdrawn, or expired, left a notification with a live
        Approve on every phone it reached until it expired. Tapping it got a 410: safe, and confusing. So the end
        of a request that was pushed sends one BACKGROUND push (no alert, ``content-available``, priority 5) to each
        device it reached and is still registered, and to no other; the phone removes the delivered notification.
        Best effort: Apple throttles background pushes, and nothing depends on this one arriving.

        Carries the request id and how it ended, nothing more: not who answered, not the scope.
        """
        request_id = payload.get("request_id")
        if not isinstance(request_id, str):
            return
        resolution = str(payload.get("resolution") or "")
        self._ended[request_id] = resolution
        while len(self._ended) > PUSHED_REQUESTS_REMEMBERED:
            self._ended.popitem(last=False)
        devices = self._pushed_to.pop(request_id, None)
        if not devices:
            return
        for target in self._store.push_targets():
            if target.id in devices:
                await self._send_resolved(target, request_id, resolution)

    async def _send_resolved(self, target: Any, request_id: str, resolution: str) -> None:
        """One background push taking *request_id*'s notification back from *target*.

        Not counted toward muting: Apple throttles background pushes as it likes, and a run of them refused must
        not silence the device's approval alerts. A 410 still clears the registration (that is a fact about the
        token), and every send is logged and recorded like any other.
        """
        if target.push_failures >= FAILURE_MUTE_THRESHOLD:
            return
        body = self._named({
            "aps": {"content-available": 1},
            "approval_resolved": request_id,
            "resolution": resolution,
        })
        result = await self._sender.send(
            apns_token=target.apns_token, environment=target.environment,
            payload=body, push_type="background", priority=5,
            topic=target.bundle_id or None,
        )
        outcome = getattr(result, "outcome", None)
        status = getattr(result, "status", None)
        logger.info("push approval_resolved → device %s: %s%s", target.id, outcome,
                    f" (status {status})" if status is not None else "")
        if outcome == "unregistered":
            self._account(target, result, "approval_resolved")
        elif outcome == "ok":
            self._record("record_push_success", target.id, status)
        else:
            self._record("record_push_attempt", target.id, status, "failed")

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
            **({"session_id": _capped(payload["session_id"], ID_CAP)} if payload.get("session_id") else {}),
            **({"task_id": _capped(payload["task_id"], ID_CAP)} if payload.get("task_id") else {}),
        }
        body = self._named(body)
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
            # The turn's latest numbers, kept (bounded) so its end can say them even between throttled pushes.
            self._turn_last[session_id] = dict(payload)
            self._turn_last.move_to_end(session_id)
            while len(self._turn_last) > PUSHED_REQUESTS_REMEMBERED:
                self._turn_last.popitem(last=False)
            now = self._now()
            if now - self._activity_pushed_at.get(session_id, 0.0) < ACTIVITY_THROTTLE_SECONDS:
                return
            targets = self._store.activity_targets(session_id)
            if not targets:
                return
            self._activity_pushed_at[session_id] = now
            phase = str(payload.get("phase") or "")
            detail = tool_detail(payload.get("tool_name"), self.known_tool) if phase == "tool" else ""
            body = {"aps": {
                "timestamp": int(now),
                "event": "update",
                # The job keys (push/job_activity.py, Piece 18) first, so an activity the daemon started for this
                # turn (JobActivityAttributes) decodes the same push as one the app started; then the turn's own.
                "content-state": {
                    **content_state(
                        phase=phase,
                        detail=detail,
                        elapsed_s=int(payload.get("elapsed_s") or 0),
                        steps=int(payload.get("tool_calls") or 0)),
                    "round": payload.get("round", 0),
                    "toolCalls": payload.get("tool_calls", 0),
                    "chars": payload.get("chars", 0),
                    # The app-started per-turn activity names the tool (a registered tool only, as for detail).
                    **({"toolName": detail} if self._registered_tool(payload.get("tool_name")) else {}),
                },
                # A daemon that dies mid-turn stops updating; the lock screen then shows the activity as stale.
                "stale-date": int(now) + STALE_AFTER_SECONDS,
            }}
            await self._deliver_activity(targets, self._named(body), session_id)
        except Exception:
            logger.warning("liveactivity push failed", exc_info=True)

    def _registered_tool(self, name: Any) -> bool:
        """Is *name* a tool this daemon registered? Never a model-written string: only then is it named in a push."""
        check = getattr(self, "known_tool", None)
        try:
            return isinstance(name, str) and bool(name) and check is not None and bool(check(name))
        except Exception:                               # noqa: BLE001 - doubt means not named
            return False

    async def on_chat_done(self, session_id: str, payload: dict) -> None:
        try:
            outcome = turn_end_phase(payload)
            listener = self.turn_end_listener
            if listener is not None:
                started = payload.get("turn_started_at")
                try:
                    listener(session_id, float(started) if isinstance(started, (int, float)) else None, outcome)
                except Exception:
                    logger.warning("turn end listener failed", exc_info=True)
            targets = self._store.activity_targets(session_id)
            if not targets:
                return
            now = self._now()
            # ``phase`` stays exactly "done", as it always was: the shipped phone decodes it. How the turn ended
            # (done, stopped for a user stop, paused for the round cap, failed) is ``outcome``.
            last = self._turn_last.pop(session_id, {})
            started = payload.get("turn_started_at")
            elapsed = payload.get("elapsed_s")
            if not elapsed and isinstance(started, (int, float)) and not isinstance(started, bool):
                elapsed = max(0.0, now - float(started))
            elapsed = elapsed or last.get("elapsed_s") or 0
            body = {"aps": turn_end_aps(
                now, outcome, elapsed_s=int(elapsed),
                round_=int(payload.get("round") or last.get("round") or 0),
                tool_calls=int(payload.get("tool_calls") or last.get("tool_calls") or 0),
                chars=int(payload.get("chars") or last.get("chars") or 0))}
            await self._deliver_activity(targets, self._named(body), session_id)
            self._activity_pushed_at.pop(session_id, None)
            for target, _token in targets:
                self._store.clear_activity_token(target.id, session_id)
        except Exception:
            logger.warning("liveactivity end push failed", exc_info=True)

    async def send_liveactivity(self, target: Any, activity_token: str, body: dict, *, priority: int,
                                kind: str = "liveactivity") -> str | None:
        """Send one job Live Activity push (push/job_activity.py) to *activity_token*; the outcome, or None when
        the device is muted.

        A 410 here is about the ACTIVITY token (or the push-to-start token), not the device's alert registration,
        so it is recorded and left to the caller, who drops that one token; the alert registration stays. Other
        outcomes are accounted like every push (a success resets the failure count, a failure counts toward the
        mute)."""
        if target.push_failures >= FAILURE_MUTE_THRESHOLD:
            return None
        result = await self._sender.send(
            apns_token=activity_token, environment=target.environment,
            payload=body, push_type="liveactivity", priority=priority,
            topic=f"{target.bundle_id or self._sender.config.topic}.push-type.liveactivity",
        )
        outcome = getattr(result, "outcome", None)
        if outcome == "unregistered":
            status = getattr(result, "status", None)
            logger.info("push %s → device %s: unregistered%s", kind, target.id,
                        f" (status {status})" if status is not None else "")
            self._record("record_push_attempt", target.id, status, "unregistered")
        else:
            self._account(target, result, kind)
        return outcome

    async def _deliver_activity(self, targets: list, body: dict, session_id: str) -> None:
        """A turn's activity pushes. A 410 is about the activity token, so it drops that token's row and leaves the
        device's alert registration alone (it used to clear the alert registration)."""
        for target, activity_token in targets:
            outcome = await self.send_liveactivity(target, activity_token, body, priority=5)
            if outcome == "unregistered":
                self._store.clear_activity_token(target.id, session_id)

    # ------------------------------------------------------------------

    def _live_device_ids(self) -> set[str]:
        identities = getattr(self._bridge, "_ws_identity", {}) or {}
        return {i.id for i in identities.values() if getattr(i, "id", "") != "global"}

    def _record(self, method: str, device_id: str, status: Any, outcome: str | None = None) -> None:
        """Recording is bookkeeping, never a reason to fail a delivery.

        Takes the method NAME, not a bound method: `self._store.record_push_success` is resolved
        when the ARGUMENT is built, so an older store without it raised before this function was
        ever entered and no try block here could have caught it. Looked up explicitly instead —
        a missing method is a known state, not an exception to absorb.
        """
        fn = getattr(self._store, method, None)
        if fn is None:
            return  # a store predating #348 — skip the bookkeeping, keep the push
        try:
            fn(device_id, status) if outcome is None else fn(device_id, outcome, status)
        except Exception:
            logger.warning("push accounting failed for device %s", device_id, exc_info=True)

    async def _deliver(self, target: Any, body: dict) -> bool:
        """Send one alert push; True when APNs accepted it."""
        if target.push_failures >= FAILURE_MUTE_THRESHOLD:
            return False  # muted until a success resets the count — no storm
        result = await self._sender.send(
            apns_token=target.apns_token, environment=target.environment,
            payload=body, push_type="alert", priority=10,
            topic=target.bundle_id or None,
        )
        # The category, so the log reads "push APPROVAL → device …" rather than "push push → …".
        self._account(target, result, str(((body.get("aps") or {}).get("category")) or "alert"))
        return getattr(result, "outcome", None) == "ok"

    def _account(self, target: Any, result: Any, kind: str = "push") -> None:
        # ONE line per send, whatever happens (#348). Before this, apns.py logged nothing at all
        # and only failures were recorded, so a successful push left no trace anywhere and
        # `push_failures == 0` meant "delivered" and "never attempted" equally. The device id is
        # safe to log; the APNs token is never touched here.
        logger.info(
            "push %s → device %s: %s%s",
            kind, target.id, result.outcome,
            f" (status {result.status})" if getattr(result, "status", None) is not None else "",
        )
        if result.outcome == "ok":
            if target.push_failures:
                self._store.reset_push_failures(target.id)
            self._unregistered_logged.discard(target.id)
            self._record("record_push_success", target.id, getattr(result, "status", None))
        elif result.outcome == "unregistered":
            # Permanent, and logged ONCE.
            self._store.clear_push(target.id)
            self._record("record_push_attempt", target.id, getattr(result, "status", None), "unregistered")
            if target.id not in self._unregistered_logged:
                self._unregistered_logged.add(target.id)
                logger.info("APNs says device %s is unregistered — push cleared",
                            target.id)
        else:
            self._record("record_push_attempt", target.id, getattr(result, "status", None), "failed")
            count = self._store.record_push_failure(target.id)
            if count == FAILURE_MUTE_THRESHOLD:
                logger.warning("device %s muted after %d consecutive push failures",
                               target.id, count)
