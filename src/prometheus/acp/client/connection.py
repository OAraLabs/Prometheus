"""The client half of an ACP v2 NDJSON connection.

Requests out (ids matched to responses), notifications in (dispatched to a
handler), and the AGENT's own requests in — above all
``session/request_permission`` — answered through a handler the driver
supplies. The handler is the gate's, never a rubber stamp: a connection
with no permission handler REFUSES every permission request (fail closed).

Transport is asyncio subprocess stdio: one frame per line, the framing
module's rules (v2: no batching, ids never null). The connection owns the
subprocess lifecycle; closing it closes the agent.

Source: novel code for Prometheus, 2026-11-25.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from prometheus.acp.framing import (
    INTERNAL_ERROR,
    METHOD_NOT_FOUND,
    FrameError,
    Notification,
    Request,
    Response,
    error_response,
    parse_frame,
    render,
)

log = logging.getLogger(__name__)

#: params -> result (or raise FrameError to answer with an error).
RequestHandler = Callable[[dict[str, Any] | None], Awaitable[Any]]
NotificationHandler = Callable[[str, dict[str, Any] | None], Awaitable[None]]


class ConnectionClosed(Exception):
    """The agent's stdout ended (it exited or closed the pipe)."""


class AcpClientConnection:
    """One client connection to one outside agent over stdio.

    ``permission_handler`` answers the agent's ``session/request_permission``
    requests. None means REFUSE everything — an unattended driver must not
    approve by default.
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        permission_handler: RequestHandler | None = None,
        notification_handler: NotificationHandler | None = None,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._permission_handler = permission_handler
        self._notification_handler = notification_handler
        self._next_id = 1
        self._pending: dict[int | str, asyncio.Future] = {}
        self._recv_task: asyncio.Task | None = None
        self._closed = False

    async def start(self) -> None:
        self._recv_task = asyncio.create_task(self._receive_loop())

    async def close(self) -> None:
        self._closed = True
        if self._recv_task is not None:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 - teardown
                pass
        try:
            self._writer.close()
            await self._writer.wait_closed()
        except (OSError, ConnectionError):
            pass
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(ConnectionClosed("connection closed"))
        self._pending.clear()

    async def request(self, method: str, params: dict[str, Any] | None = None,
                      *, timeout: float = 60.0) -> Any:
        """Send a request, await its response. Raises FrameError on an error
        response, ConnectionClosed if the agent goes away, TimeoutError."""
        if self._closed:
            raise ConnectionClosed("connection closed")
        rid = self._next_id
        self._next_id += 1
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        self._writer.write((render(Request(id=rid, method=method, params=params)) + "\n").encode())
        await self._writer.drain()
        try:
            return await asyncio.wait_for(fut, timeout)
        finally:
            self._pending.pop(rid, None)

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        if self._closed:
            raise ConnectionClosed("connection closed")
        self._writer.write((render(Notification(method=method, params=params)) + "\n").encode())
        await self._writer.drain()

    async def _receive_loop(self) -> None:
        try:
            while not self._closed:
                line = await self._reader.readline()
                if not line:
                    raise ConnectionClosed("agent stdout ended")
                try:
                    frame = parse_frame(line)
                except FrameError as exc:
                    # A malformed frame from the agent: answer an error if it
                    # carried a readable id, otherwise log and keep reading.
                    log.warning("acp client: malformed frame from agent: %s", exc.message)
                    if exc.request_id is not None:
                        await self._send(error_response(exc.request_id, exc.code, exc.message))
                    continue
                if isinstance(frame, Response):
                    self._resolve(frame)
                elif isinstance(frame, Request):
                    asyncio.create_task(self._answer(frame))
                elif isinstance(frame, Notification) and self._notification_handler is not None:
                    asyncio.create_task(self._dispatch_notification(frame))
        except asyncio.CancelledError:
            raise
        except ConnectionClosed:
            self._fail_pending(ConnectionClosed("agent stdout ended"))
        except (OSError, ConnectionError) as exc:
            self._fail_pending(ConnectionClosed(str(exc)))

    def _resolve(self, frame: Response) -> None:
        fut = self._pending.get(frame.id) if frame.id is not None else None
        if fut is None or fut.done():
            return
        if frame.is_error:
            err = frame.error or {}
            fut.set_exception(FrameError(int(err.get("code", INTERNAL_ERROR)),
                                         str(err.get("message", "agent error"))))
        else:
            fut.set_result(frame.result)

    def _fail_pending(self, exc: Exception) -> None:
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(exc)

    async def _answer(self, frame: Request) -> None:
        if frame.method == "session/request_permission":
            if self._permission_handler is None:
                # FAIL CLOSED: no handler, no approval. The outcome is
                # "cancelled" — the schema's not-approved shape.
                await self._send(Response(id=frame.id, result={"outcome": {"outcome": "cancelled"}}))
                return
            try:
                result = await self._permission_handler(frame.params)
            except FrameError as exc:
                await self._send(error_response(frame.id, exc.code, exc.message))
                return
            except Exception:  # noqa: BLE001 - a broken handler must not approve
                log.exception("acp client: permission handler failed — refusing")
                await self._send(Response(id=frame.id, result={"outcome": {"outcome": "cancelled"}}))
                return
            await self._send(Response(id=frame.id, result=result))
            return
        await self._send(error_response(frame.id, METHOD_NOT_FOUND,
                                        f"method not supported by this client: {frame.method}"))

    async def _dispatch_notification(self, frame: Notification) -> None:
        handler = self._notification_handler
        if handler is None:
            return
        try:
            await handler(frame.method, frame.params)
        except Exception:  # noqa: BLE001 - an observer must not kill the connection
            log.exception("acp client: notification handler failed (%s)", frame.method)

    async def _send(self, frame: Response) -> None:
        if self._closed:
            return
        try:
            self._writer.write((render(frame) + "\n").encode())
            await self._writer.drain()
        except (OSError, ConnectionError):
            self._fail_pending(ConnectionClosed("agent pipe broken"))