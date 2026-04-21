# Provenance: Inspired by anthropics/skills frontend-design + web-artifacts-builder
# License: MIT

"""Serve an HTML dashboard on a local HTTP server."""

from __future__ import annotations

import socket
import subprocess
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import tempfile

from pydantic import BaseModel, Field

from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult


def _get_host_address() -> str:
    """Detect the best reachable address for the dashboard.

    Priority: Tailscale IP > LAN IP > localhost.
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


class DashboardTool(BaseTool):
    """Serve HTML content on a local HTTP server for visualization."""

    name = "dashboard"
    description = (
        "Start a lightweight HTTP server that serves the provided HTML content. "
        "Returns the URL to view the dashboard. The URL uses the Tailscale IP "
        "if available, otherwise the LAN IP or localhost."
    )
    input_model = DashboardInput

    _servers: dict[int, HTTPServer] = {}

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

        # Stop any existing server on this port
        if port in self._servers:
            self._servers[port].shutdown()
            del self._servers[port]

        # Start HTTP server bound to all interfaces
        handler = partial(SimpleHTTPRequestHandler, directory=str(tmpdir))
        try:
            server = HTTPServer(("0.0.0.0", port), handler)
        except OSError as exc:
            return ToolResult(
                output=f"Cannot bind to port {port}: {exc}", is_error=True
            )

        self._servers[port] = server
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        host = _get_host_address()
        url = f"http://{host}:{port}/"
        return ToolResult(
            output=f"Dashboard serving at {url}\nHTML file: {index}",
            metadata={"url": url, "port": port, "html_path": str(index)},
        )
