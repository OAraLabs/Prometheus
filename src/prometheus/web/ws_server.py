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


class WebSocketBridge:
    """Bridges SignalBus events to WebSocket clients."""

    def __init__(
        self,
        signal_bus: Any | None = None,
        session_mgr: Any | None = None,
        loop_context: Any | None = None,
        agent_state_ref: Any | None = None,
        api_token: str | None = None,
    ) -> None:
        self.signal_bus = signal_bus
        self.session_mgr = session_mgr
        self.loop_context = loop_context
        self.agent_state_ref = agent_state_ref
        # Same secret the REST middleware uses (config.web.api_token /
        # PROMETHEUS_API_TOKEN). Empty/None => auth DISABLED, exactly like
        # the REST side, so dev/no-token setups (and the tokenless static UI)
        # keep working unchanged.
        self._api_token = api_token or ""
        self._clients: set[Any] = set()
        self._server: Any = None

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
            if session_id and content:
                await self._handle_send_message(session_id, content)

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
            from prometheus.gateway.media_cache import cache_image_from_bytes, sniff_image_extension
            img_ext = ext if ext in (".png", ".jpg", ".jpeg", ".gif", ".webp") else sniff_image_extension(filename)
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
            from prometheus.tools.builtin.vision import VisionTool
            tool = VisionTool()
            result = await tool.arun(image_path=image_path)
            if result and not result.startswith("Error"):
                return result
        except Exception as exc:
            logger.warning("Vision analysis failed for %s: %s", image_path, exc)
        return None

    async def dispatch_user_message(
        self, session_id: str, content: str, client_msg_id: str | None = None
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
        await self._handle_send_message(session_id, content, client_msg_id=client_msg_id)

    async def _handle_send_message(
        self, session_id: str, content: str, client_msg_id: str | None = None
    ) -> None:
        """Process a user message — add to session and run agent loop if context available.

        If the content contains [Image: /path/to/file] references (from Beacon
        dashboard uploads), run vision analysis to describe the image before
        passing to the agent — matching the Telegram gateway's flow.
        """
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
            asyncio.create_task(self._run_agent(session_id, session))

    async def _run_agent(self, session_id: str, session: Any) -> None:
        """Run the agent loop and stream results over WebSocket."""
        from prometheus.engine.agent_loop import run_loop

        # Update state
        if self.agent_state_ref:
            self.agent_state_ref["state"] = "thinking"
        await self.broadcast({
            "type": "agent_state",
            "timestamp": time.time(),
            "payload": {"state": "thinking"},
        })

        msg_id = f"asst-{int(time.time() * 1000)}"
        accumulated = ""

        try:
            messages = session.get_messages()
            original_len = len(messages)
            async for event, _usage in run_loop(self.loop_context, messages):
                event_type = type(event).__name__

                if event_type == "AssistantTextDelta":
                    accumulated += event.text
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

        except Exception as e:
            await self.broadcast({
                "type": "error",
                "timestamp": time.time(),
                "payload": {"message": str(e)},
            })

        finally:
            if self.agent_state_ref:
                self.agent_state_ref["state"] = "idle"
            await self.broadcast({
                "type": "agent_state",
                "timestamp": time.time(),
                "payload": {"state": "idle"},
            })

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
