"""MCP transport resolution — stdio vs HTTP vs SSE.

Donor: OpenClaw src/agents/mcp-transport-config.ts, mcp-stdio.ts, mcp-http.ts
License: MIT (Anthropic)

Source: Prometheus (OAra AI Lab)
License: MIT
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

DEFAULT_CONNECTION_TIMEOUT_MS = 30_000


# ---------------------------------------------------------------------------
# Resolved transport types
# ---------------------------------------------------------------------------


@dataclass
class ResolvedStdioTransport:
    """Resolved stdio transport config."""

    kind: Literal["stdio"] = "stdio"
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | None = None
    timeout_ms: int = DEFAULT_CONNECTION_TIMEOUT_MS

    @property
    def description(self) -> str:
        args_str = " ".join(self.args) if self.args else ""
        return f"{self.command} {args_str}".strip()


@dataclass
class ResolvedHttpTransport:
    """Resolved HTTP/SSE transport config."""

    kind: Literal["http"] = "http"
    transport_type: Literal["sse", "streamable-http"] = "sse"
    url: str = ""
    headers: dict[str, str] | None = None
    timeout_ms: int = DEFAULT_CONNECTION_TIMEOUT_MS

    @property
    def description(self) -> str:
        parsed = urlparse(self.url)
        if parsed.password:
            host_port = parsed.hostname or ""
            if parsed.port:
                host_port += f":{parsed.port}"
            return f"{parsed.scheme}://***@{host_port}{parsed.path}"
        return self.url


ResolvedTransport = ResolvedStdioTransport | ResolvedHttpTransport


# ---------------------------------------------------------------------------
# Config coercion helpers (from mcp-config-shared.ts)
# ---------------------------------------------------------------------------


def _coerce_string_list(value: Any) -> list[str] | None:
    """Coerce value to string list."""
    if not isinstance(value, list):
        return None
    return [str(v) for v in value if isinstance(v, (str, int, bool))]


def _coerce_string_dict(value: Any) -> dict[str, str] | None:
    """Coerce value to string dict."""
    if not isinstance(value, dict):
        return None
    return {k: str(v) for k, v in value.items() if isinstance(v, (str, int, bool))}


def _coerce_timeout(raw: dict) -> int:
    """Extract and validate timeout."""
    timeout = raw.get("connectionTimeoutMs", DEFAULT_CONNECTION_TIMEOUT_MS)
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        return DEFAULT_CONNECTION_TIMEOUT_MS
    return int(timeout)


# ---------------------------------------------------------------------------
# Transport resolvers
# ---------------------------------------------------------------------------


def resolve_stdio_config(raw: dict) -> ResolvedStdioTransport | None:
    """Resolve stdio transport config.

    From OpenClaw mcp-stdio.ts: resolveStdioMcpServerLaunchConfig
    """
    command = raw.get("command")
    if not isinstance(command, str) or not command.strip():
        return None

    cwd = raw.get("cwd") or raw.get("workingDirectory")
    if isinstance(cwd, str) and not cwd.strip():
        cwd = None

    return ResolvedStdioTransport(
        command=command.strip(),
        args=_coerce_string_list(raw.get("args")) or [],
        env=_coerce_string_dict(raw.get("env")),
        cwd=cwd if isinstance(cwd, str) else None,
        timeout_ms=_coerce_timeout(raw),
    )


def resolve_http_config(
    raw: dict,
    transport_type: str = "sse",
) -> ResolvedHttpTransport | None:
    """Resolve HTTP/SSE transport config.

    From OpenClaw mcp-http.ts: resolveHttpMcpServerLaunchConfig
    """
    url = raw.get("url")
    if not isinstance(url, str) or not url.strip():
        return None

    url = url.strip()
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            logger.warning("MCP: only http/https URLs supported, got %s", parsed.scheme)
            return None
    except Exception:
        logger.warning("MCP: invalid URL in config")
        return None

    valid_types = ("sse", "streamable-http")
    tt = transport_type if transport_type in valid_types else "sse"

    return ResolvedHttpTransport(
        transport_type=tt,
        url=url,
        headers=_coerce_string_dict(raw.get("headers")),
        timeout_ms=_coerce_timeout(raw),
    )


def resolve_transport(server_name: str, raw: dict) -> ResolvedTransport | None:
    """Resolve MCP transport from raw config dict.

    From OpenClaw mcp-transport-config.ts: resolveMcpTransportConfig

    Priority: stdio (command) > streamable-http > sse (url).
    """
    # Try stdio first
    stdio = resolve_stdio_config(raw)
    if stdio:
        return stdio

    # Check requested transport type
    requested = str(raw.get("transport", "")).strip().lower()
    if requested == "streamable-http":
        http = resolve_http_config(raw, "streamable-http")
        if http:
            return http

    # Try SSE
    http = resolve_http_config(raw, "sse")
    if http:
        return http

    logger.warning("MCP: could not resolve transport for server '%s'", server_name)
    return None
