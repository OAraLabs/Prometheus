# Provenance: Inspired by anthropics/skills frontend-design + web-artifacts-builder
# License: MIT

"""Serve an HTML dashboard on a local HTTP server."""

from __future__ import annotations

import asyncio
import socket
import subprocess
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import tempfile

from pydantic import BaseModel, Field

from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult


#: Bind addresses that reach only this machine. Used to decide whether the
#: advertised URL may name a remotely reachable address — see `_advertised_host`.
_LOOPBACK_BINDS = frozenset({"127.0.0.1", "::1", "localhost"})

#: The default, and the fix. See DashboardTool for why it is not "0.0.0.0".
DEFAULT_BIND_HOST = "127.0.0.1"


def _advertised_host(bind_host: str) -> str:
    """The host to put in the URL we hand back, given what we actually bound.

    A URL is a claim about reachability. The tool used to bind 0.0.0.0 and then
    advertise the Tailscale address, which was at least consistent. Binding
    loopback while still advertising a tailnet IP would be worse than either:
    the operator would be told a remote URL for a socket that refuses every
    remote connection, and would debug the network instead of the config.

    So the advertised host follows the bind: loopback in, loopback out.
    """
    if bind_host in _LOOPBACK_BINDS:
        return "127.0.0.1"
    if bind_host == "0.0.0.0":
        # Listening everywhere; name the address someone can actually reach.
        return _get_host_address()
    return bind_host


def _get_host_address() -> str:
    """Detect the best reachable address for the dashboard.

    Priority: Tailscale IP > LAN IP > localhost.

    ONLY CONSULTED WHEN THE OPERATOR HAS WIDENED THE BIND. On a loopback bind
    the answer would be a lie: a Tailscale address that nothing is listening
    on. See `_advertised_host`.
    """
    # Try Tailscale first
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            ip = result.stdout.strip().splitlines()[0]
            if ip:
                return ip
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fall back to the default outbound interface IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip and ip != "0.0.0.0":
            return ip
    except OSError:
        pass

    return "localhost"


class DashboardInput(BaseModel):
    """Arguments for serving an HTML dashboard."""

    html: str = Field(description="HTML content to serve")
    port: int = Field(default=8080, ge=1024, le=65535, description="Port to serve on")
    title: str | None = Field(default=None, description="Optional page title")


def _stop_server(server: HTTPServer) -> None:
    """Stop a running server AND release its port. Blocking; run off-loop.

    `shutdown()` alone is not enough and never was. It stops the serve_forever
    loop and returns, leaving the listening socket OPEN — so the port stayed
    bound for the life of the daemon, and a later call reusing that port hit
    "Address already in use" against a server this tool had itself abandoned.
    `server_close()` is the call that releases it.

    Both block: `shutdown()` waits for the serving thread to notice and exit.
    That is why callers hand this to a worker thread instead of running it on
    the event loop.
    """
    try:
        server.shutdown()
    finally:
        server.server_close()


class DashboardTool(BaseTool):
    """Serve HTML content on an HTTP server for visualization.

    ⚠ THE BIND ADDRESS IS THE OPERATOR'S CHOICE, NOT THE MODEL'S.

    This bound `("0.0.0.0", port)` unconditionally and served model-authored
    HTML on every interface of the host — no authentication, no config gate,
    on a port the MODEL picked, registered unconditionally at
    `__main__.create_tool_registry`. A single tool call published arbitrary
    markup to the LAN and the tailnet.

    THE ASYMMETRY THAT SHAPES THE FIX. A blanket loopback-only bind would be
    the simple answer and it would remove a stated use: this tool's own
    description advertises reaching the dashboard over Tailscale, and an
    operator who wants that is making a legitimate choice about their own
    machine. What is NOT legitimate is the MODEL making that choice, which is
    what "always 0.0.0.0" amounted to.

    So the interface is operator-chosen: loopback by default,
    `security.dashboard_tool_bind_host` to widen. The PORT stays
    model-chosen, and that is fine — on loopback a port is not an exposure
    decision, and capping what the model may pick would constrain the agent's
    own choices rather than where its data can flow.

    NOT FIXED HERE, DELIBERATELY: the HTML is model-authored and served with
    no Content-Security-Policy. The bundled UI got a CSP in #425; this is the
    second HTML surface and it has none. That is its own piece of work — see
    the finding recorded in tests/test_dashboard_bind_is_operator_chosen.py.
    Loopback-by-default narrows who can reach it; it does not make the markup
    safe.
    """

    name = "dashboard"
    description = (
        "Start a lightweight HTTP server that serves the provided HTML content. "
        "Returns the URL to view the dashboard. Binds to localhost by default, "
        "so the URL is reachable from this machine only; an operator can widen "
        "it with the security.dashboard_tool_bind_host config key, in which "
        "case the URL names the Tailscale or LAN address."
    )
    input_model = DashboardInput

    #: Class-level so a second DashboardTool instance still finds — and can
    #: stop — a server the first one started on the same port.
    _servers: dict[int, HTTPServer] = {}

    def __init__(self, bind_host: str = DEFAULT_BIND_HOST) -> None:
        self._bind_host = bind_host or DEFAULT_BIND_HOST

    async def execute(
        self, arguments: DashboardInput, context: ToolExecutionContext
    ) -> ToolResult:
        port = arguments.port
        html_content = arguments.html

        if arguments.title:
            if "<title>" not in html_content.lower():
                html_content = html_content.replace(
                    "<head>", f"<head><title>{arguments.title}</title>", 1
                )

        # Write HTML to a temp directory
        tmpdir = Path(tempfile.mkdtemp(prefix="prometheus-dash-"))
        index = tmpdir / "index.html"
        index.write_text(html_content, encoding="utf-8")

        # Stop any existing server on this port — OFF the event loop.
        #
        # This used to be a bare `self._servers[port].shutdown()` inside this
        # async method. `shutdown()` blocks until the serving thread notices
        # and exits, so it stalled the whole daemon's event loop from inside a
        # tool call — every other session, gateway and timer with it. It also
        # never called `server_close()`, so the port stayed bound for the life
        # of the process and the rebind below would fail against a server this
        # tool had abandoned.
        previous = self._servers.pop(port, None)
        if previous is not None:
            await asyncio.to_thread(_stop_server, previous)

        handler = partial(SimpleHTTPRequestHandler, directory=str(tmpdir))
        try:
            server = HTTPServer((self._bind_host, port), handler)
        except OSError as exc:
            return ToolResult(
                output=(
                    f"Cannot bind to {self._bind_host}:{port}: {exc}"
                ),
                is_error=True,
            )

        self._servers[port] = server
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        host = _advertised_host(self._bind_host)
        url = f"http://{host}:{port}/"
        return ToolResult(
            output=f"Dashboard serving at {url}\nHTML file: {index}",
            metadata={
                "url": url,
                "port": port,
                "html_path": str(index),
                "bind_host": self._bind_host,
            },
        )
