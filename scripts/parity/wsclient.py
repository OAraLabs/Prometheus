"""A minimal RFC 6455 client — enough to LISTEN to the daemon's event stream.

Stdlib only (the harness contract). The harness uses the socket passively:
it authenticates, then reads frames, so the turn-completion moment
(``chat_done``) is observed when the daemon emits it rather than discovered
by polling — polling would add load to the very process being timed.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import struct
import threading
import time
from typing import Any, Callable


class WSClosed(Exception):
    pass


class WSClient:
    def __init__(self, host: str, port: int, path: str = "/", timeout: float = 10.0) -> None:
        self.sock = socket.create_connection((host, port), timeout=timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (
            f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
            f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(req.encode())
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = self.sock.recv(1)
            if not chunk:
                raise WSClosed("handshake: connection closed")
            head += chunk
        status = head.split(b"\r\n", 1)[0]
        if b" 101 " not in status:
            raise WSClosed(f"handshake refused: {status!r}")
        self.sock.settimeout(None)
        self._send_lock = threading.Lock()

    def _recv_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise WSClosed("connection closed")
            buf += chunk
        return buf

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        header = bytes([0x80 | opcode])
        n = len(payload)
        if n < 126:
            header += bytes([0x80 | n])
        elif n < 65536:
            header += bytes([0x80 | 126]) + struct.pack("!H", n)
        else:
            header += bytes([0x80 | 127]) + struct.pack("!Q", n)
        mask = os.urandom(4)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        with self._send_lock:
            self.sock.sendall(header + mask + masked)

    def send_json(self, obj: Any) -> None:
        self._send_frame(0x1, json.dumps(obj).encode())

    def recv_message(self) -> str:
        parts: list[bytes] = []
        while True:
            b1, b2 = self._recv_exact(2)
            fin, opcode = b1 & 0x80, b1 & 0x0F
            n = b2 & 0x7F
            if n == 126:
                n = struct.unpack("!H", self._recv_exact(2))[0]
            elif n == 127:
                n = struct.unpack("!Q", self._recv_exact(8))[0]
            mask = self._recv_exact(4) if b2 & 0x80 else b""
            payload = self._recv_exact(n)
            if mask:
                payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
            if opcode == 0x8:
                raise WSClosed(f"server closed: {payload[:2].hex()} {payload[2:]!r}")
            if opcode == 0x9:
                self._send_frame(0xA, payload)
                continue
            if opcode == 0xA:
                continue
            parts.append(payload)
            if fin:
                return b"".join(parts).decode("utf-8", errors="replace")

    def close(self) -> None:
        try:
            self._send_frame(0x8, b"\x03\xe8")
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


class EventListener:
    """Background reader that timestamps every event on the monotonic clock."""

    def __init__(self, client: WSClient, on_event: Callable[[dict, int], None] | None = None) -> None:
        self.client = client
        self.events: list[tuple[int, dict]] = []
        self._cond = threading.Condition()
        self._on_event = on_event
        self._thread = threading.Thread(target=self._run, name="parity-ws", daemon=True)
        self.error: str | None = None

    def start(self) -> "EventListener":
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            while True:
                raw = self.client.recv_message()
                t = time.monotonic_ns()
                try:
                    ev = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                with self._cond:
                    self.events.append((t, ev))
                    self._cond.notify_all()
                if self._on_event:
                    self._on_event(ev, t)
        except (WSClosed, OSError) as exc:
            with self._cond:
                self.error = str(exc)
                self._cond.notify_all()

    def wait_for(self, pred: Callable[[dict], bool], timeout: float, start: int = 0) -> tuple[int, dict] | None:
        """First event at index >= start matching ``pred``, waiting up to ``timeout``."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                for t, ev in self.events[start:]:
                    if pred(ev):
                        return t, ev
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self.error is not None:
                    return None
                self._cond.wait(remaining)

    def mark(self) -> int:
        with self._cond:
            return len(self.events)
