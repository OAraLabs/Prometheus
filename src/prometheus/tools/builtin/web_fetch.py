# Provenance: HKUDS/OpenHarness (https://github.com/HKUDS/OpenHarness)
# Original: src/openharness/tools/web_fetch_tool.py
# License: Apache-2.0
# Modified: Rewritten as Prometheus BaseTool; added SSRF protection

"""Fetch a web page and return compact readable text."""

from __future__ import annotations

import ipaddress
import re
import socket

import httpx
from pydantic import BaseModel, Field

from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult


class WebFetchInput(BaseModel):
    """Arguments for fetching one web page."""

    url: str = Field(description="HTTP or HTTPS URL to fetch")
    max_chars: int = Field(
        default=12000, ge=500, le=50000, description="Maximum characters to return"
    )


class WebFetchTool(BaseTool):
    """Fetch one web page and return compact readable text."""

    name = "web_fetch"
    description = "Fetch a web page by URL and return its text content."
    input_model = WebFetchInput

    def is_read_only(self, arguments: WebFetchInput) -> bool:
        return True

    async def execute(
        self, arguments: WebFetchInput, context: ToolExecutionContext
    ) -> ToolResult:
        # SSRF protection — block private/reserved IPs
        if not _is_safe_url(arguments.url):
            return ToolResult(
                output="Blocked: URL resolves to a private or reserved IP address.",
                is_error=True,
            )

        try:
            async with httpx.AsyncClient(
                follow_redirects=True, timeout=20.0
            ) as client:
                response = await client.get(
                    arguments.url,
                    headers={"User-Agent": "Prometheus/0.1"},
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            return ToolResult(output=f"web_fetch failed: {exc}", is_error=True)

        content_type = response.headers.get("content-type", "")
        body = response.text
        if "html" in content_type:
            body = _html_to_text(body)
        body = body.strip()
        if len(body) > arguments.max_chars:
            body = body[: arguments.max_chars].rstrip() + "\n...[truncated]"

        return ToolResult(
            output=(
                f"URL: {response.url}\n"
                f"Status: {response.status_code}\n"
                f"Content-Type: {content_type or '(unknown)'}\n\n"
                f"{body}"
            )
        )


# ---------------------------------------------------------------------------
# SSRF protection
# ---------------------------------------------------------------------------

def _is_safe_url(url: str) -> bool:
    """Return False if the URL resolves to a private or reserved IP."""
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return False
        # Resolve hostname to IP addresses
        addrs = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for family, _, _, _, sockaddr in addrs:
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local:
                return False
    except (socket.gaierror, ValueError, OSError):
        return False
    return True


# ---------------------------------------------------------------------------
# HTML → text
# ---------------------------------------------------------------------------

def _html_to_text(raw: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )
    return re.sub(r"[ \t\r\f\v]+", " ", text).replace(" \n", "\n").strip()
