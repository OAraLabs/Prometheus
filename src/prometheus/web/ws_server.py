"""WebSocket event bridge for Mission Control.

Runs on port 8010. Subscribes to the Prometheus SignalBus and forwards
all signals as JSON events to connected browser clients. Also accepts
client→server commands (send_message, switch_session).

Usage:
    from prometheus.web.ws_server import WebSocketBridge
    bridge = WebSocketBridge(signal_bus, session_mgr, loop_context)
    await bridge.start(host="0.0.0.0", port=8010)
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Close code for an unauthenticated / failed-auth WebSocket. 4000–4999 is the
# application-private range; 4401 mirrors HTTP 401 for "unauthorized". Beacon
# and the static UI key their auth-failure UX off this exact code.
WS_CLOSE_UNAUTHORIZED = 4401

# How long a freshly-connected socket has to send its auth frame before the
# server closes it. Short enough that an idle/probing connection can't sit
# open; long enough for a real client's first round-trip.
AUTH_FRAME_TIMEOUT_SECONDS = 5.0

# Cadence of the ``agent_progress`` liveness pulse emitted while a turn runs.
# A turn can legitimately produce NO frames for minutes (a slow local model
# thinking before its first token, or one long tool call), which is
# indistinguishable from a dead daemon to a client — that ambiguity is what
# drove Beacon's "no reply after 30s" false alarm. This is the positive signal:
# while it keeps arriving the turn is alive, so a client never has to guess
# with a bare timeout. 3s is frequent enough to feel live and is negligible
# traffic next to token deltas.
PROGRESS_INTERVAL_SECONDS = 3.0


class WebSocketBridge:
    """Bridges SignalBus events to WebSocket clients."""

    def __init__(
        self,
        signal_bus: Any | None = None,
        session_mgr: Any | None = None,
        loop_context: Any | None = None,
        agent_state_ref: Any | None = None,
        api_token: str | None = None,
        config: dict | None = None,
    ) -> None:
        self.signal_bus = signal_bus
        self.session_mgr = session_mgr
        self.loop_context = loop_context
        self.agent_state_ref = agent_state_ref
        # Full daemon config (config.web.*, etc.) — used by formatter commands
        # like /doctor and /beacon dispatched through the web slash-router.
        self.config = config or {}
        # Same secret the REST middleware uses (config.web.api_token /
        # PROMETHEUS_API_TOKEN). Empty/None => auth DISABLED, exactly like
        # the REST side, so dev/no-token setups (and the tokenless static UI)
        # keep working unchanged.
        self._api_token = api_token or ""
        self._clients: set[Any] = set()
        self._server: Any = None
        # Interrupt plumbing (feat/ws-interrupt-frame): the running turn's
        # asyncio.Task per session, registered by _run_agent itself so BOTH
        # entry paths (fire-and-forget chat + awaited Paperclip turns) are
        # stoppable. _interrupted marks a cancel as a DELIBERATE user stop —
        # any unflagged CancelledError (daemon shutdown) keeps propagating.
        self._turn_tasks: dict[str, asyncio.Task] = {}
        self._interrupted: set[str] = set()
        # Per-session turn serialization (fix: duplicate LCM rows, 2026-08-11
        # survey). The REAL lock lives on the shared SessionManager
        # (turn_lock_for) so bridge turns and telegram turns (inject_turn
        # re-engagement) on the SAME session contend on ONE lock — this local
        # map is only the fallback for bridges wired without a real manager
        # (tests construct WebSocketBridge with session_mgr=None or a stub).
        self._turn_locks: dict[str, asyncio.Lock] = {}

    @property
    def auth_required(self) -> bool:
        """True when a non-empty token is configured (parity with REST)."""
        return bool(self._api_token)

    def _token_ok(self, raw: str) -> bool:
        """Validate a first-frame auth message: {"type":"auth","token":...}.

        Constant-time token comparison (``hmac.compare_digest``) so a timing
        side-channel can't probe the secret. Any parse error, wrong type, or
        missing/incorrect token is a clean False — the caller closes 4401.
        """
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return False
        if not isinstance(msg, dict) or msg.get("type") != "auth":
            return False
        token = msg.get("token")
        if not isinstance(token, str):
            return False
        return hmac.compare_digest(token, self._api_token)

    async def start(self, host: str = "0.0.0.0", port: int = 8010) -> None:
        """Start the WebSocket server."""
        try:
            import websockets
        except ImportError:
            logger.error("websockets package not installed — pip install websockets")
            return

        # Subscribe to all SignalBus events
        if self.signal_bus:
            self.signal_bus.subscribe("*", self._on_signal)

        self._server = await websockets.serve(  # type: ignore[attr-defined]
            self._handler,
            host,
            port,
        )
        logger.info("WebSocket bridge listening on ws://%s:%d", host, port)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _handler(self, websocket: Any) -> None:
        """Handle a single WebSocket client connection.

        First-frame auth: when a token is configured, the socket must send
        ``{"type":"auth","token":...}`` within ``AUTH_FRAME_TIMEOUT_SECONDS``
        as its FIRST frame. Until that succeeds the server sends NOTHING — not
        even the ``connected`` welcome — so an unauthenticated client receives
        zero frames before the 4401 close. (See the Phase-0 survey: a browser
        client exists and cannot set an upgrade header, so auth is in-band.)
        """
        if self.auth_required and not await self._authenticate(websocket):
            return  # _authenticate already closed the socket 4401

        self._clients.add(websocket)
        logger.info("Client connected (%d total)", len(self._clients))

        # Welcome is sent only AFTER auth — never to an unauthenticated socket.
        await self._send_one(websocket, {
            "type": "connected",
            "timestamp": time.time(),
            "payload": {"version": "0.1.0"},
        })

        try:
            async for raw in websocket:
                await self._handle_client_message(websocket, raw)
        except Exception:
            pass
        finally:
            self._clients.discard(websocket)
            logger.info("Client disconnected (%d remain)", len(self._clients))

    async def _authenticate(self, websocket: Any) -> bool:
        """Await + validate the first-frame auth token. Closes 4401 on failure.

        Returns True only when a well-formed ``{"type":"auth","token":...}``
        frame with the correct token arrives within the timeout. The socket is
        NOT added to ``_clients`` and NO frame is sent until this returns True,
        so a rejected client provably receives nothing.
        """
        try:
            raw = await asyncio.wait_for(
                websocket.recv(), timeout=AUTH_FRAME_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            await self._close_unauthorized(websocket, "auth frame timeout")
            return False
        except Exception:
            # Connection dropped/closed before sending anything.
            return False

        if self._token_ok(raw):
            return True
        await self._close_unauthorized(websocket, "invalid or missing auth token")
        return False

    async def _close_unauthorized(self, websocket: Any, reason: str) -> None:
        # Generic client-facing reason (no token echo, ever); the specific
        # cause stays in the server log only.
        logger.info("WS auth rejected: %s", reason)
        try:
            await websocket.close(code=WS_CLOSE_UNAUTHORIZED, reason="unauthorized")
        except Exception:
            pass

    async def _handle_client_message(self, websocket: Any, raw: str) -> None:
        """Process a command from the browser client."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        cmd_type = msg.get("type", "")
        payload = msg.get("payload", {})

        if cmd_type == "subscribe":
            # Acknowledgement only — all events are broadcast
            await self._send_one(websocket, {
                "type": "subscribed",
                "timestamp": time.time(),
                "payload": {"channels": payload.get("channels", [])},
            })

        elif cmd_type == "send_message":
            session_id = payload.get("session_id", "")
            content = payload.get("content", "")
            mode = payload.get("mode") or "agent"  # absent → agent default (never an error)
            if mode not in ("agent", "chat"):
                await self._send_one(websocket, {
                    "type": "error",
                    "timestamp": time.time(),
                    "payload": {
                        "session_id": session_id,
                        "message": f"invalid mode {mode!r} — expected 'agent' or 'chat'",
                        "kind": "bad_request",
                    },
                })
                return
            # force-search: validate an optional per-message tool_choice against the live tool
            # registry; unknown/malformed → error frame (same discipline as malformed mode).
            # Absent → None, so `mode` resolves it downstream.
            from prometheus.api.tool_choice import normalize_tool_choice as _normalize_tc
            _reg = getattr(self.loop_context, "tool_registry", None)
            _valid = {t.name for t in _reg.list_tools()} if (_reg is not None and hasattr(_reg, "list_tools")) else None
            _tc_raw = payload.get("tool_choice")
            try:
                tool_choice = _normalize_tc(_tc_raw, _valid) if _tc_raw is not None else None
            except ValueError as exc:
                await self._send_one(websocket, {
                    "type": "error",
                    "timestamp": time.time(),
                    "payload": {
                        "session_id": session_id,
                        "message": str(exc),
                        "kind": "bad_request",
                    },
                })
                return
            if session_id and content:
                await self._handle_send_message(session_id, content, mode=mode, tool_choice=tool_choice)

        elif cmd_type == "chat_upload":
            # File upload from Beacon: { type: "chat_upload", payload: {
            #   session_id, filename, content_base64, mime_type, caption? } }
            session_id = payload.get("session_id", "")
            filename = payload.get("filename", "file")
            content_b64 = payload.get("content_base64", "")
            mime_type = payload.get("mime_type", "")
            caption = payload.get("caption", "")
            if session_id and content_b64:
                await self._handle_file_upload(
                    session_id, filename, content_b64, mime_type, caption
                )

        elif cmd_type == "switch_session":
            session_id = payload.get("session_id", "")
            if session_id and self.session_mgr:
                session = self.session_mgr.get_or_create(session_id)
                # Send existing messages for the session
                messages = session.get_messages()
                for i, m in enumerate(messages):
                    await self._send_one(websocket, {
                        "type": "chat_message",
                        "timestamp": time.time(),
                        "payload": {
                            "session_id": session_id,
                            "role": m.role,
                            "content": m.content if isinstance(m.content, str) else str(m.content),
                            "content_json": m.content_json,
                            "message_id": f"hist-{i}",
                        },
                    })

        elif cmd_type == "interrupt":
            # Stop the running turn: { type: "interrupt", payload: { session_id } }.
            # Ack goes to the REQUESTING socket only; every client learns the
            # outcome from the broadcast chat_done{interrupted:true} frame.
            session_id = payload.get("session_id", "")
            stopped = self.interrupt_turn(session_id) if session_id else False
            await self._send_one(websocket, {
                "type": "interrupt_ack",
                "timestamp": time.time(),
                "payload": {"session_id": session_id, "stopped": stopped},
            })

    async def _handle_file_upload(
        self,
        session_id: str,
        filename: str,
        content_base64: str,
        mime_type: str,
        caption: str,
    ) -> None:
        """Handle a file upload from Beacon dashboard.

        Routes by type:
          - Images → cache + vision analysis (same as Telegram photos)
          - Documents → cache + text extraction (same as Telegram documents)
        """
        import base64

        try:
            data = base64.b64decode(content_base64)
        except Exception:
            logger.warning("Invalid base64 in chat_upload")
            return

        # Size guard: 20 MB
        if len(data) > 20 * 1024 * 1024:
            await self.broadcast({
                "type": "chat_message",
                "timestamp": time.time(),
                "payload": {
                    "session_id": session_id,
                    "role": "system",
                    "content": "File too large (max 20 MB).",
                    "message_id": f"sys-{int(time.time() * 1000)}",
                },
            })
            return

        from pathlib import Path as _Path
        ext = _Path(filename).suffix.lower()

        if mime_type.startswith("image/"):
            # Image upload → cache + vision
            from prometheus.gateway.media_cache import cache_image_from_bytes, extension_from_file_path
            img_ext = ext if ext in (".png", ".jpg", ".jpeg", ".gif", ".webp") else extension_from_file_path(filename)
            cached_path = cache_image_from_bytes(data, ext=img_ext)

            desc = await self._describe_image(cached_path)
            if desc:
                user_text = f"[Image: {desc}]"
            else:
                user_text = f"[The user sent an image: {filename}]"
            if caption:
                user_text = f"{user_text}\n{caption}"
        else:
            # Document upload → cache + text extraction
            from prometheus.gateway.media_cache import cache_document_from_bytes
            from prometheus.utils.file_extract import extract_text, is_supported, unsupported_message

            if not is_supported(filename):
                await self.broadcast({
                    "type": "chat_message",
                    "timestamp": time.time(),
                    "payload": {
                        "session_id": session_id,
                        "role": "system",
                        "content": unsupported_message(filename),
                        "message_id": f"sys-{int(time.time() * 1000)}",
                    },
                })
                return

            cached_path = cache_document_from_bytes(data, filename)
            extracted = extract_text(cached_path)
            if extracted:
                user_text = f"[Content of {filename}]:\n{extracted}"
                if caption:
                    user_text = f"{caption}\n\n{user_text}"
            else:
                user_text = caption or f"[The user sent a document: {filename}]"

        # Dispatch as a user message (same as _handle_send_message)
        await self._handle_send_message(session_id, user_text)

    async def _describe_image(self, image_path: str) -> str | None:
        """Run vision analysis on a cached image file, matching Telegram gateway flow."""
        try:
            from pathlib import Path
            from prometheus.tools.base import ToolExecutionContext
            from prometheus.tools.builtin.vision import VisionInput, VisionTool
            tool = VisionTool()
            ctx = ToolExecutionContext(
                cwd=Path(self._config.workspace_root) if self._config else Path.home(),
                metadata={"source": "beacon_upload"},
            )
            result = await tool.execute(VisionInput(image_path=image_path), ctx)
            output = result.output
            if output and not output.startswith("Error"):
                return output
        except Exception as exc:
            logger.warning("Vision analysis failed for %s: %s", image_path, exc)
        return None

    async def dispatch_user_message(
        self, session_id: str, content: str, client_msg_id: str | None = None, mode: str = "agent",
        tool_choice: object | None = None
    ) -> None:
        """Public dispatch entry point — kicks off the same flow as a WS-borne
        ``send_message`` command.

        Used by the FastAPI ``POST /api/chat/send`` route so Beacon's chat
        surface (and any other HTTP caller) doesn't need to maintain its own
        WebSocket. Returns once the dispatch coroutine is queued; the actual
        agent work runs in a background task and streams back over the
        bridge's broadcast — the HTTP caller's job is to be a WS subscriber
        (which Beacon already is) to receive the response deltas.

        ``client_msg_id`` (optional) is echoed on the user ``chat_message`` frame
        alongside the canonical ``msg-{turn_index}`` id, so a client that rendered
        the message optimistically can correlate its local id to the durable one.
        """
        await self._handle_send_message(session_id, content, client_msg_id=client_msg_id, mode=mode, tool_choice=tool_choice)

    async def run_turn_awaited(self, session_id: str, content: str) -> tuple[str, Any]:
        """Run ONE agent turn in ``session_id`` and await its completion.

        Machine-caller counterpart to ``dispatch_user_message`` (which is
        fire-and-forget for chat surfaces): the Paperclip gateway needs the
        final assistant text AND the turn's token usage to report back to its
        orchestrator. The turn still goes through the exact same flow as a
        Beacon chat message — session persistence, LCM, and live WS broadcast
        — so a heartbeat renders in Beacon like any other conversation.

        No slash-command routing and no image preprocessing: content here is a
        machine-built work prompt, never a user chat message.

        Returns ``(final_text, usage)`` where usage is the last
        ``UsageSnapshot`` run_loop yielded (None if the loop produced none).
        Raises on agent failure instead of swallowing (fail loud to the
        caller, which owns external error reporting).
        """
        if not self.session_mgr or not self.loop_context:
            raise RuntimeError(
                "bridge not fully wired (session_mgr/loop_context missing) — "
                "cannot run an awaited turn"
            )
        session = self.session_mgr.get_or_create(session_id)
        turn_index = session.add_user_message(content)
        row_id = session.last_persisted_row_id()
        ts = time.time()
        await self.broadcast({
            "type": "chat_message",
            "timestamp": ts,
            "payload": {
                "session_id": session_id,
                "role": "user",
                "content": content,
                "content_json": json.dumps([{"type": "text", "text": content}]),
                "message_id": row_id,
                "ordinal": turn_index,
                "client_msg_id": None,
                "created_at": ts,
            },
        })
        return await self._run_agent(session_id, session, raise_on_error=True)

    def interrupt_turn(self, session_id: str) -> bool:
        """Stop the running agent turn in ``session_id`` (the chat Stop button).

        Flags the session as deliberately interrupted, THEN cancels the turn's
        task — order matters, so ``_run_agent``'s CancelledError handler always
        sees the flag and can distinguish a user stop from a system cancel.
        Returns False when no turn is running (idempotent — stopping a quiet
        session is not an error, the caller just gets ``stopped: false``).
        """
        task = self._turn_tasks.get(session_id)
        if task is None or task.done():
            return False
        self._interrupted.add(session_id)
        task.cancel()
        return True

    async def _handle_send_message(
        self, session_id: str, content: str, client_msg_id: str | None = None, mode: str = "agent",
        tool_choice: object | None = None
    ) -> None:
        """Process a user message — add to session and run agent loop if context available.

        A leading-slash message is a command (web parity for Telegram's
        CommandHandler): it's handled here and broadcast back, NOT added to the
        session or run through the agent. See prometheus.web.slash_router.

        If the content contains [Image: /path/to/file] references (from Beacon
        dashboard uploads), run vision analysis to describe the image before
        passing to the agent — matching the Telegram gateway's flow.
        """
        # Slash commands first — they never become conversation turns. Resolve
        # the active session (non-creating) for /steer & friends, and pass a
        # get_or_create factory for /queue (which fires on a quiet chat).
        from prometheus.web.slash_router import build_command_context, route_slash

        active_session = self.session_mgr.get(session_id) if self.session_mgr else None
        ensure_session = (
            (lambda: self.session_mgr.get_or_create(session_id))
            if self.session_mgr
            else None
        )
        outcome = await route_slash(
            content,
            build_command_context(
                self.loop_context,
                self.config,
                session=active_session,
                ensure_session=ensure_session,
            ),
        )
        if outcome.handled:
            await self._broadcast_command_reply(
                session_id, content, outcome.reply or "", client_msg_id
            )
            return

        if not self.session_mgr:
            return

        import re
        # Detect image references from Beacon: [Image: /path/to/file.ext]
        image_pattern = re.compile(r'\[Image:\s*(/[^\]]+)\]')
        matches = image_pattern.findall(content)
        if matches:
            import os
            described_parts = []
            for img_path in matches:
                if os.path.isfile(img_path):
                    desc = await self._describe_image(img_path)
                    if desc:
                        described_parts.append(f"[Image: {desc}]")
                    else:
                        described_parts.append(f"[The user sent an image: {img_path}]")
                else:
                    described_parts.append(f"[Image reference: {img_path}]")
            # Replace raw paths with descriptions
            processed = content
            for match, replacement in zip(matches, described_parts):
                processed = processed.replace(f"[Image: {match}]", replacement, 1)
            content = processed

        session = self.session_mgr.get_or_create(session_id)
        turn_index = session.add_user_message(content)
        row_id = session.last_persisted_row_id()

        # Broadcast the user message. message_id is the durable, restart-stable LCM rowid
        # — the SAME canonical id GET /api/sessions/{id}/messages reports — so a client can
        # correlate its optimistic client_msg_id to the real row. ordinal (turn_index) is a
        # NON-UNIQUE display position only.
        ts = time.time()
        await self.broadcast({
            "type": "chat_message",
            "timestamp": ts,
            "payload": {
                "session_id": session_id,
                "role": "user",
                "content": content,
                "content_json": json.dumps([{"type": "text", "text": content}]),
                "message_id": row_id,
                "ordinal": turn_index,
                "client_msg_id": client_msg_id,
                "created_at": ts,
            },
        })

        # If we have a loop context, run the agent
        if self.loop_context:
            asyncio.create_task(self._run_agent(session_id, session, mode=mode, tool_choice=tool_choice))

    async def _broadcast_command_reply(
        self,
        session_id: str,
        command_text: str,
        reply: str,
        client_msg_id: str | None = None,
    ) -> None:
        """Broadcast a slash-command exchange as transient chat messages.

        Echoes the command (role user) then its result (role assistant) so every
        connected client renders the exchange, mirroring the normal send path's
        frames — minus LCM persistence, since commands aren't conversation turns
        (parity with Telegram, which never adds command turns to the session).
        The synthetic ``cmd-*`` ids and ``transient`` flag mark them as ephemeral.
        """
        ts = time.time()
        marker = int(ts * 1000)
        await self.broadcast({
            "type": "chat_message",
            "timestamp": ts,
            "payload": {
                "session_id": session_id,
                "role": "user",
                "content": command_text,
                "content_json": json.dumps([{"type": "text", "text": command_text}]),
                "message_id": f"cmd-user-{marker}",
                "client_msg_id": client_msg_id,
                "created_at": ts,
                "transient": True,
            },
        })
        await self.broadcast({
            "type": "chat_message",
            "timestamp": ts,
            "payload": {
                "session_id": session_id,
                "role": "assistant",
                "content": reply,
                "content_json": json.dumps([{"type": "text", "text": reply}]),
                "message_id": f"cmd-asst-{marker}",
                "created_at": ts,
                "transient": True,
                "command": True,
            },
        })

    def _turn_lock_for(self, session_id: str) -> asyncio.Lock:
        """Return the per-session turn lock — shared across surfaces.

        Delegates to :meth:`SessionManager.turn_lock_for`, THE serialization
        point for turns on one session, so a live bridge turn cannot
        interleave with a telegram-side re-engagement (``inject_turn``
        targeting a ``desktop:*`` session) on the same ChatSession — two
        surface-local maps only ever serialized within a surface.

        Back-compat (mirror of the telegram shim): bridges built via
        ``__new__``, with ``session_mgr=None``, or with a stub manager keep
        the old bridge-local map, lazily initialized — the within-surface
        guarantee is unchanged.
        """
        from prometheus.engine.session import SessionManager

        mgr = getattr(self, "session_mgr", None)
        if isinstance(mgr, SessionManager):
            return mgr.turn_lock_for(session_id)
        locks = getattr(self, "_turn_locks", None)
        if locks is None:
            locks = {}
            self._turn_locks = locks
        lock = locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            locks[session_id] = lock
        return lock

    async def _run_agent(
        self, session_id: str, session: Any, mode: str = "agent", tool_choice: object | None = None,
        raise_on_error: bool = False,
    ) -> tuple[str, Any]:
        """Run the agent loop and stream results over WebSocket. `mode` ('agent'|'chat')
        is threaded as a per-call run_loop arg — NEVER stored on the shared loop_context —
        so concurrent turns can't cross-talk (Sprint B / Piece 2).

        Turns on ONE session are serialized by a per-session lock (parity
        with the telegram gateway's M6 lock; 2026-08-11 duplicate-rows fix).
        Without it, rapid-fire sends spawned N concurrent ``run_loop``s all
        appending in place to the SAME ``session.messages`` list — model
        rounds interleaved across turns, and each finishing turn re-persisted
        every other in-flight turn's tail to LCM (up to N copies per row).
        The lock is held from ``original_len`` capture through the tail
        persist, which is exactly the invariant ``persist_loop_result``
        needs. Different sessions still run concurrently.

        Returns ``(accumulated_text, last_usage)`` for awaited callers
        (``run_turn_awaited``); the fire-and-forget chat path ignores it.
        ``raise_on_error=True`` re-raises after the fail-loud logging/broadcast
        instead of swallowing (the awaited caller reports errors externally)."""
        # Update state. session_id rides every agent_state frame: with more than
        # one session alive, "thinking" is meaningless unless a client can tell
        # WHICH conversation it belongs to (Beacon could not, so it could not
        # scope a per-chat indicator).
        if self.agent_state_ref:
            self.agent_state_ref["state"] = "thinking"
        await self.broadcast({
            "type": "agent_state",
            "timestamp": time.time(),
            "payload": {"state": "thinking", "session_id": session_id},
        })

        msg_id = f"asst-{int(time.time() * 1000)}"

        # Live turn shape, mutated as events stream and sampled by the
        # heartbeat task. A plain dict (not locals) so the heartbeat sees
        # updates without threading state through every branch.
        progress = {
            "phase": "thinking",   # thinking → generating → tool → …
            "tool_name": None,
            "round": 1,            # model calls so far, 1-based (round N in flight)
            "chars": 0,            # assistant characters streamed this turn
            "tool_calls": 0,
        }
        heartbeat = asyncio.create_task(
            self._emit_progress(session_id, msg_id, progress, time.time())
        )

        _task = asyncio.current_task()

        try:
            # Serialize turns per session. The heartbeat above keeps pulsing
            # "thinking" for a queued turn — honest UX, the turn IS pending.
            # A cancel landing while we WAIT here propagates out (there is
            # nothing to persist and the task was never registered, so
            # interrupt_turn cannot have targeted it); the outer finally
            # still reaps the heartbeat.
            async with self._turn_lock_for(session_id):
                # Register THIS task as the session's running turn so
                # interrupt_turn() can cancel it. asyncio.current_task()
                # covers both entry paths (create_task from
                # _handle_send_message AND the awaited Paperclip call).
                # Registration happens INSIDE the lock: the Stop button
                # targets the turn that is actually running — a queued turn
                # is not stoppable until it starts (stop it again then).
                if _task is not None:
                    self._turn_tasks[session_id] = _task
                return await self._run_agent_locked(
                    session_id, session, mode=mode, tool_choice=tool_choice,
                    raise_on_error=raise_on_error, msg_id=msg_id,
                    progress=progress,
                )
        finally:
            # Stop the liveness pulse FIRST — a heartbeat outliving its turn
            # would tell clients a finished turn is still running. Cancel
            # without awaiting: this runs on the cancellation path too, where
            # awaiting could re-raise before the frames below are sent.
            heartbeat.cancel()
            # Unregister only if the slot still points at THIS task (a newer
            # concurrent turn may have overwritten it). Clearing the interrupt
            # flag here bounds any stale flag from a cancel that lost the race
            # with normal completion.
            if self._turn_tasks.get(session_id) is asyncio.current_task():
                self._turn_tasks.pop(session_id, None)
            self._interrupted.discard(session_id)
            if self.agent_state_ref:
                self.agent_state_ref["state"] = "idle"
            await self.broadcast({
                "type": "agent_state",
                "timestamp": time.time(),
                "payload": {"state": "idle", "session_id": session_id},
            })

    async def _run_agent_locked(
        self, session_id: str, session: Any, *, mode: str,
        tool_choice: object | None, raise_on_error: bool,
        msg_id: str, progress: dict,
    ) -> tuple[str, Any]:
        """Serialized core of :meth:`_run_agent` (holds the session's turn
        lock — same split as the telegram gateway's ``_run_agent_turn`` /
        ``_run_agent_turn_locked``). ``original_len`` capture, the in-place
        ``run_loop`` appends, and the LCM tail persist all happen under the
        lock, on BOTH the normal and the interrupted exit paths."""
        from prometheus.engine.agent_loop import run_loop

        accumulated = ""
        last_usage: Any = None
        original_len: int | None = None
        try:
            messages = session.get_messages()
            original_len = len(messages)
            async for event, _usage in run_loop(
                self.loop_context, messages, mode=mode, session_id=session_id, tool_choice=tool_choice
            ):
                if _usage is not None:
                    last_usage = _usage
                event_type = type(event).__name__

                if event_type == "AssistantTurnComplete":
                    # One model call finished; anything further is a new round.
                    progress["round"] = int(progress["round"]) + 1

                if event_type == "AssistantTextDelta":
                    accumulated += event.text
                    progress["phase"] = "generating"
                    progress["chars"] = len(accumulated)
                    await self.broadcast({
                        "type": "chat_delta",
                        "timestamp": time.time(),
                        "payload": {
                            "session_id": session_id,
                            "content": event.text,
                            "message_id": msg_id,
                        },
                    })

                elif event_type == "ToolExecutionStarted":
                    progress["phase"] = "tool"
                    progress["tool_name"] = event.tool_name
                    progress["tool_calls"] = int(progress["tool_calls"]) + 1
                    await self.broadcast({
                        "type": "tool_call_start",
                        "timestamp": time.time(),
                        "payload": {
                            "call_id": event.tool_use_id,
                            "tool_name": event.tool_name,
                            "inputs": event.tool_input,
                        },
                    })

                elif event_type == "ToolExecutionCompleted":
                    # Tool done → the model is thinking again until it emits.
                    progress["phase"] = "thinking"
                    progress["tool_name"] = None
                    await self.broadcast({
                        "type": "tool_call_end",
                        "timestamp": time.time(),
                        "payload": {
                            "call_id": event.tool_use_id,
                            "tool_name": event.tool_name,
                            "success": not event.is_error,
                            "result": event.output[:2000] if event.output else "",
                        },
                    })

            # Persist the assistant turn that run_loop appended in place onto
            # session.messages (parity with the Telegram/Slack gateways). Without
            # this the web/Beacon assistant half never reaches LCM/memory.
            session.persist_loop_result(original_len)

            # Stream done
            await self.broadcast({
                "type": "chat_done",
                "timestamp": time.time(),
                "payload": {"session_id": session_id, "message_id": msg_id},
            })
            return accumulated, last_usage

        except asyncio.CancelledError:
            # A cancel is a user interrupt ONLY when interrupt_turn() flagged
            # this session; anything else (daemon shutdown, task GC) must keep
            # propagating or teardown would hang on a swallowed cancel.
            if session_id not in self._interrupted:
                raise
            self._interrupted.discard(session_id)
            logger.info("turn interrupted by user (session=%s)", session_id)
            # Keep what the turn produced. Completed rounds are already on
            # session.messages (run_loop appends in place) — persist that
            # tail. A mid-generation stop (nothing appended yet, but text
            # already streamed to every client) is kept as a partial
            # assistant turn so the visible bubble survives a reload. A stop
            # mid-round-N>1 drops only round N's in-flight tail — the
            # completed rounds carry the substance.
            if original_len is not None:
                if len(session.messages) == original_len and accumulated:
                    from prometheus.engine.agent_loop import _make_assistant_msg
                    session.messages.append(_make_assistant_msg(accumulated))
                session.persist_loop_result(original_len)
            await self.broadcast({
                "type": "chat_done",
                "timestamp": time.time(),
                "payload": {
                    "session_id": session_id,
                    "message_id": msg_id,
                    "interrupted": True,
                },
            })
            return accumulated, last_usage

        except Exception as e:
            # FAIL LOUD (#74-adjacent follow-up): this except used to emit only a WS
            # error frame — invisible in the journal unless a client happened to
            # render it. That masked the wire-contract interface drift (a TypeError
            # from a run_loop signature change surfaced as "the turn silently
            # produced nothing") for the life of #74-on-main. Log the full traceback
            # server-side and record a silent_failure telemetry row, THEN broadcast.
            logger.exception(
                "_run_agent failed (session=%s, mode=%s): %s", session_id, mode, e
            )
            try:
                from prometheus.telemetry.tracker import get_telemetry_handle
                handle = get_telemetry_handle()
                if handle is not None and hasattr(handle, "record_silent_failure"):
                    handle.record_silent_failure(
                        "web_bridge", "_run_agent", e, context={"session_id": session_id, "mode": mode}
                    )
            except Exception:
                logger.warning("telemetry record for _run_agent failure itself failed", exc_info=True)
            # session_id lets a client render this INLINE in the right chat
            # instead of filing it in a side panel; the structured fields turn
            # "Client error '400 Bad Request'" into a cause and a next step.
            # `message` is kept as the top-level key it has always been, so
            # existing clients keep working unchanged.
            from prometheus.api.turn_errors import classify_turn_error
            detail = classify_turn_error(e)
            await self.broadcast({
                "type": "error",
                "timestamp": time.time(),
                "payload": {
                    "session_id": session_id,
                    "message": detail["message"],
                    "kind": detail["kind"],
                    "provider": detail["provider"],
                    "status": detail["status"],
                    "hint": detail["hint"],
                },
            })
            if raise_on_error:
                raise
            return accumulated, last_usage

    async def _emit_progress(
        self,
        session_id: str,
        message_id: str,
        progress: dict[str, Any],
        started_at: float,
        interval: float | None = None,
    ) -> None:
        """Pulse ``agent_progress`` every ``interval`` seconds while a turn runs.

        Started and cancelled by :meth:`_run_agent`. Samples the live
        ``progress`` dict rather than owning state, so it reports what the turn
        is ACTUALLY doing (which tool, which round, how much text so far)
        instead of a bare "still alive".

        Never raises: a broadcast failure here must not kill the turn or leave
        an unretrieved exception on a task nobody awaits.

        ``interval`` resolves from the module global at CALL time (not as a
        bound default), so the cadence stays patchable for tests.
        """
        if interval is None:
            interval = PROGRESS_INTERVAL_SECONDS
        while True:
            try:
                await asyncio.sleep(interval)
                await self.broadcast({
                    "type": "agent_progress",
                    "timestamp": time.time(),
                    "payload": {
                        "session_id": session_id,
                        "message_id": message_id,
                        "phase": progress["phase"],
                        "tool_name": progress["tool_name"],
                        "round": progress["round"],
                        "chars": progress["chars"],
                        "tool_calls": progress["tool_calls"],
                        "elapsed_s": round(time.time() - started_at, 1),
                    },
                })
            except asyncio.CancelledError:
                raise  # normal teardown — the turn ended
            except Exception:
                logger.debug("agent_progress emit failed", exc_info=True)

    async def _on_signal(self, signal: Any) -> None:
        """Forward a SignalBus event to all connected clients."""
        event = {
            "type": "sentinel_signal",
            "timestamp": signal.timestamp,
            "payload": {
                "kind": signal.kind,
                "payload": signal.payload,
                "source": signal.source,
            },
        }

        # Map specific signal kinds to dedicated event types
        if signal.kind == "dream_start":
            event["type"] = "dream_start"
            event["payload"] = signal.payload
        elif signal.kind == "dream_phase":
            event["type"] = "dream_phase"
            event["payload"] = signal.payload
        elif signal.kind == "dream_complete":
            event["type"] = "dream_complete"
            event["payload"] = signal.payload
        # Sprint S1 Stream 2: first-class event types for visible memory &
        # skill activity, matching the dream_* pattern so Beacon's
        # frontend can route them to dedicated UI panes (Sprint 3 polish).
        elif signal.kind == "skill_created":
            event["type"] = "skill_created"
            event["payload"] = signal.payload
        elif signal.kind == "skill_refined":
            event["type"] = "skill_refined"
            event["payload"] = signal.payload
        elif signal.kind == "memory_updated":
            event["type"] = "memory_updated"
            event["payload"] = signal.payload
        elif signal.kind == "curator_report":
            event["type"] = "curator_report"
            event["payload"] = signal.payload
        # Coding live-stream (feat/coding-livestream): per-round progression,
        # the terminal verdict, and a non-fatal stream-interruption marker.
        # Same first-class-type pattern so Beacon's Live view routes them; the
        # payload carries session_id for client-side run scoping (these, like
        # every signal event, broadcast to all authed clients).
        elif signal.kind in ("coding_round", "coding_complete", "coding_stream_error"):
            event["type"] = signal.kind
            event["payload"] = signal.payload

        await self.broadcast(event)

    async def broadcast(self, event: dict[str, Any]) -> None:
        """Send an event to all connected clients."""
        if not self._clients:
            return
        raw = json.dumps(event)
        dead: list[Any] = []
        for ws in self._clients:
            try:
                await ws.send(raw)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    async def _send_one(self, websocket: Any, event: dict[str, Any]) -> None:
        try:
            await websocket.send(json.dumps(event))
        except Exception:
            self._clients.discard(websocket)
