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

from prometheus.computer.thumbnails import (
    NONE,
    NO_SINK,
    ThumbnailConfig,
    build_frame,
    capture_plan,
    decide_capture,
)

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
        thumbnail_sink: Any = None,
        thumb_config: Any = None,
        thumbnail_data_dir: str | None = None,
    ) -> None:
        self._bus = bus
        self._bridge = bridge
        self._telemetry = telemetry
        self.keep_per_session = keep_per_session
        self._progress_s = progress_s
        self._seq: dict[str, int] = {}
        self._tickers: dict[str, asyncio.Task] = {}
        #: PR 6b. The sink is the WS bridge (it implements viewer_count /
        #: send_thumbnail); NO_SINK means nobody is ever watching. Config and
        #: data_dir drive the thumbnails.* keys and the opt-in persist path.
        self._thumb_sink = thumbnail_sink if thumbnail_sink is not None else NO_SINK
        self._thumb_config = thumb_config or ThumbnailConfig()
        self._thumb_data_dir = thumbnail_data_dir
        #: The latest sent frame per running task, in memory only, so a device
        #: that reconnects mid-task gets the current picture once. Discarded
        #: when the task ends — never persisted, never a cache that outlives it.
        self._last_thumbnail: dict[str, dict[str, Any]] = {}
        #: session_id → task_ids whose thumbnails were written to disk, so the
        #: opt-in files can be pruned with the action log. Only populated when
        #: ``thumbnails.persist`` is on; without it there is nothing to prune.
        self._persisted_tasks: dict[str, set[str]] = {}

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

    # ── PR 6b: the thumbnail path ────────────────────────────────────────

    def thumbnail_viewers(self, session_id: str) -> int:
        """Eligible viewers for this session, or 0.

        The runner asks this BEFORE calling the driver, so a task nobody is
        watching never captures at all. Returns 0 when the feature is off, so
        one check covers both.
        """
        if not self._thumb_config.enabled:
            return 0
        try:
            return int(self._thumb_sink.viewer_count(session_id))
        except Exception:  # noqa: BLE001 - fail closed: no viewers, no capture
            logger.debug("thumbnail viewer_count failed", exc_info=True)
            return 0

    def thumbnail_plan(self, session_id: str, *, status: str,
                       after_stop: bool) -> tuple[str, str | None]:
        """The pre-capture decision for one step: (CAPTURE|SKIP|NONE, reason).

        The runner calls this INSTEAD of writing its own status/enabled/viewer
        test, so the tri-state lives in one tested place (thumbnails.py) rather
        than being duplicated at the call site where nothing checks it.

        ``after_stop`` is answered here, before the generic plan: the call that
        was in flight at a stop is NONE, not a skip — its outcome is unknown
        and the person asked us to stop, so no thumbnail decision was made.
        """
        if after_stop:
            return NONE, None
        return capture_plan(status=status,
                            enabled=self._thumb_config.enabled,
                            viewers=self.thumbnail_viewers(session_id))

    async def _send_thumbnail(self, task_id: str,
                              frame: dict[str, Any]) -> bool:
        """Direct-only send. True when the sink accepted it.

        Never broadcast, never SignalBus: the sink decides eligibility per
        socket and a failure here is a dropped picture, not a task error.
        """
        try:
            await self._thumb_sink.send_thumbnail(frame["session_id"], frame)
            return True
        except Exception:  # noqa: BLE001 - the picture never ends a task
            logger.debug("thumbnail send failed", exc_info=True)
            return False

    def _persist_thumbnail(self, task_id: str, seq: int,
                           frame: dict[str, Any]) -> None:
        """The opt-in write. Off by default, and off is the default for a
        reason: this is the one path where a desktop screenshot touches disk.

        0600, under the thumbnails tree, pruned with the action log. Even
        when on, no route serves these in v1.1 and they never enter
        signal_events.
        """
        if not self._thumb_config.persist or not self._thumb_data_dir:
            return
        try:
            import base64
            import os

            from prometheus.computer.thumbnails import persist_path

            path = persist_path(self._thumb_data_dir, task_id, seq,
                                frame.get("mime_type") or "")
            if path is None:
                return
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = base64.b64decode(frame.get("data_base64") or "",
                                    validate=True)
            # 0600 before the write, not after: a file created with the
            # umask's mode is briefly readable by more than the owner.
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
            except Exception:
                os.close(fd)
                raise
            # Recorded only AFTER the file exists, so the prune list never
            # names a directory that is not there.
            sid = str(frame.get("session_id") or "")
            if sid:
                self._persisted_tasks.setdefault(sid, set()).add(task_id)
        except Exception:  # noqa: BLE001 - losing a picture is not a failure
            logger.debug("thumbnail persist failed", exc_info=True)

    def last_thumbnail(self, task_id: str) -> dict[str, Any] | None:
        """The latest sent frame for a running task, for a device that
        reconnects mid-task. Memory only, dropped when the task ends."""
        return self._last_thumbnail.get(task_id)

    def _chat_turn_live(self, session_id: str) -> bool:
        turns = getattr(self._bridge, "_turn_tasks", None) or {}
        turn = turns.get(session_id)
        return turn is not None and not turn.done()

    def _next_seq(self, task_id: str) -> int:
        self._seq[task_id] = self._seq.get(task_id, 0) + 1
        return self._seq[task_id]

    def _prune(self, session_id: str) -> None:
        # The opt-in FILES are pruned first and unconditionally: they are
        # deleted whether or not telemetry is wired, because a screenshot that
        # outlives the log entry that justified capturing it is a desktop
        # picture on disk with nothing tracking it.
        self._prune_thumbnails(session_id)
        tel = self._telemetry
        if tel is None or not session_id or self.keep_per_session <= 0:
            return
        try:
            tel.prune_signal_events(kinds=COMPUTER_FRAME_KINDS,
                                    session_id=session_id,
                                    keep=self.keep_per_session)
        except Exception:  # noqa: BLE001
            logger.debug("computer action log prune failed", exc_info=True)

    def _prune_thumbnails(self, session_id: str) -> None:
        """Delete the opt-in thumbnail files of one session's tasks.

        Only the task directories THIS process recorded are touched — never a
        scan of the tree, so a stale or foreign directory is not deleted by
        inference. ``persist_path`` re-validates each task id on the way, so a
        recorded id that somehow held a traversal still cannot reach outside
        the thumbnails tree.
        """
        if not self._thumb_data_dir:
            return
        tasks = self._persisted_tasks.pop(session_id, None)
        if not tasks:
            return
        try:
            import os
            import shutil

            from prometheus.computer.thumbnails import persist_path

            for task_id in tasks:
                sample = persist_path(self._thumb_data_dir, task_id, 0,
                                      "image/png")
                if sample is None:
                    continue
                shutil.rmtree(os.path.dirname(sample), ignore_errors=True)
        except Exception:  # noqa: BLE001 - losing the archive is not a failure
            logger.debug("thumbnail prune failed", exc_info=True)

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
        capture: Any = None,
        capture_app_names: Any = (),
        thumbnail_skip: str | None = None,
    ) -> None:
        seq = self._next_seq(task.task_id)
        executed = status in ("executed", "in_flight_at_stop")
        effect = (("CONFIRMED" if verified else "UNVERIFIABLE")
                  if executed else None)
        action = ({"verb": verb, "description": _cap(description, 120),
                   "app_text": True} if verb else None)

        # ── PR 6b: the thumbnail decision, BEFORE the step frame ──────────
        # The step frame reports the OUTCOME, so it is decided first. Three
        # inputs, mutually exclusive:
        #   thumbnail_skip — the runner already knew not to capture (no
        #     viewer, not executed, feature off); a reason, or None for the
        #     no-decision case;
        #   capture — a WindowCapture the runner took; decide_capture rules on
        #     it here, where the seq to send it under is known;
        #   neither — no thumbnail applies, and the step says so with null.
        thumbnail: str | None = None
        thumbnail_skip_reason: str | None = None
        if thumbnail_skip is not None:
            thumbnail, thumbnail_skip_reason = "skipped", thumbnail_skip
        elif capture is not None:
            skip, image = decide_capture(
                capture, app_names=capture_app_names)
            if skip is not None or image is None:
                # The two are set together by decide_capture; naming both
                # keeps the contract visible to the type checker rather than
                # relying on a narrowing it cannot see.
                thumbnail = "skipped"
                thumbnail_skip_reason = skip or "capture_failed"
            else:
                frame = build_frame(
                    session_id=task.session_id, task_id=task.task_id, seq=seq,
                    app=str(task.app), image=image,
                    captured_at=time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                              time.gmtime()))
                if await self._send_thumbnail(task.task_id, frame):
                    thumbnail = "sent"
                    self._last_thumbnail[task.task_id] = frame
                    self._persist_thumbnail(task.task_id, seq, frame)
                else:
                    # A viewer left between the runner's count and this send.
                    # The picture is dropped, not queued — it existed only to
                    # be watched, and nobody is watching now.
                    thumbnail, thumbnail_skip_reason = "skipped", "no_viewer"

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
            # PR 6b: the OUTCOME of the thumbnail decision, never the picture.
            # "sent" | "skipped" | None — and None means no decision was made
            # (feature off, or the step did not execute), which is not the same
            # as a skip. The image itself goes direct-only through the sink;
            # this frame is persisted and must stay content-free.
            "thumbnail": thumbnail,
            "thumbnail_skip_reason": thumbnail_skip_reason,
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
        # PR 6b: the in-memory last thumbnail is dropped with the task. It is
        # held only so a reconnecting device sees the current picture once;
        # keeping it past the end would be a cache of desktop pixels that
        # outlives the thing that justified capturing them.
        self._last_thumbnail.pop(task.task_id, None)
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
