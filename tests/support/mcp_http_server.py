"""A real MCP server on loopback, over streamable HTTP or SSE, for transport tests.

The SDK's own server (FastMCP) served by uvicorn in a thread, wrapped so the
test can read the ``Authorization`` header of every request that reached it.
One tool, ``whoami``, answers ``ok``.
"""

from __future__ import annotations

import socket
import threading
import time

import uvicorn
from mcp.server.fastmcp import FastMCP


class McpHttpServer:
    def __init__(self, transport: str = "streamable-http") -> None:
        self.transport = transport
        mcp = FastMCP("remote-tools", host="127.0.0.1")

        @mcp.tool()
        def echo(text: str) -> str:
            """Say it back."""
            return f"echo: {text}"

        @mcp.tool()
        def whoami() -> str:
            """Answer ok."""
            return "ok"

        inner = mcp.streamable_http_app() if transport == "streamable-http" else mcp.sse_app()
        self.seen_auth: list[str] = []

        async def app(scope, receive, send):  # records what reached the server
            if scope["type"] == "http":
                for name, value in scope["headers"]:
                    if name == b"authorization":
                        self.seen_auth.append(value.decode())
            await inner(scope, receive, send)

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.path = "/mcp" if transport == "streamable-http" else "/sse"
        self._server = uvicorn.Server(uvicorn.Config(
            app, host="127.0.0.1", port=self.port, log_level="warning", lifespan="on"))
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}{self.path}"

    def config(self, **headers: str) -> dict:
        out: dict = {"url": self.url, "transport": self.transport}
        if headers:
            out["headers"] = dict(headers)
        return out

    def __enter__(self) -> "McpHttpServer":
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started:
            assert time.monotonic() < deadline, "test MCP server did not start"
            time.sleep(0.02)
        return self

    def __exit__(self, *exc) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)
