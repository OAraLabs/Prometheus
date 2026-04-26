# Source: Original implementation for Prometheus (WEAVE Session A)
# License: MIT
# Purpose: Stream a URL to local disk with size limit + path-traversal guard.

"""Download tool — fetches a URL and streams it to a file on disk."""

from __future__ import annotations

import ipaddress
import re
import socket
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx
from pydantic import BaseModel, Field

from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult


_DEFAULT_MAX_MB = 100
_DEFAULT_TIMEOUT = 60.0
_DOWNLOAD_DIR = Path.home() / ".prometheus" / "downloads"
_FILENAME_FALLBACK_RE = re.compile(r"[^A-Za-z0-9._-]+")


class DownloadFileInput(BaseModel):
    """Arguments for downloading a file from a URL."""

    url: str = Field(description="HTTP or HTTPS URL to download")
    destination: str | None = Field(
        default=None,
        description="Local path to save the file (default: ~/.prometheus/downloads/<filename>)",
    )
    overwrite: bool = Field(
        default=False,
        description="Overwrite the destination if it already exists",
    )


class DownloadFileTool(BaseTool):
    """Stream a URL to local disk.

    Default destination is ``~/.prometheus/downloads/``. Files larger than
    the configured size limit (default 100MB) abort the download partway and
    delete the partial file. All errors are returned as
    ``ToolResult(is_error=True)`` — never raised.
    """

    name = "download_file"
    description = (
        "Download a file from a URL to local disk. "
        "Saves to ~/.prometheus/downloads/ by default. Streams with a size limit."
    )
    input_model = DownloadFileInput
    example_call = {
        "url": "https://example.com/file.pdf",
        "destination": "~/.prometheus/downloads/file.pdf",
    }

    def __init__(self, max_mb: int = _DEFAULT_MAX_MB) -> None:
        self._max_bytes = max_mb * 1024 * 1024
        self._max_mb = max_mb

    def is_read_only(self, arguments: DownloadFileInput) -> bool:
        return False

    async def execute(
        self,
        arguments: DownloadFileInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        if not _is_safe_url(arguments.url):
            return ToolResult(
                output="Blocked: URL resolves to a private or reserved IP address.",
                is_error=True,
            )

        try:
            dest = _resolve_destination(arguments.url, arguments.destination)
        except ValueError as exc:
            return ToolResult(output=f"Error: {exc}", is_error=True)

        if dest.exists() and not arguments.overwrite:
            return ToolResult(
                output=(
                    f"Error: Destination already exists at {dest} — "
                    f"set overwrite: true to replace it."
                ),
                is_error=True,
            )

        dest.parent.mkdir(parents=True, exist_ok=True)
        bytes_written = 0
        content_type = ""
        try:
            async with httpx.AsyncClient(
                follow_redirects=True, timeout=_DEFAULT_TIMEOUT
            ) as client:
                async with client.stream(
                    "GET",
                    arguments.url,
                    headers={"User-Agent": "Prometheus/0.1"},
                ) as response:
                    if response.status_code >= 400:
                        return ToolResult(
                            output=(
                                f"Error: HTTP {response.status_code} fetching "
                                f"{arguments.url}"
                            ),
                            is_error=True,
                        )
                    content_type = response.headers.get("content-type", "")
                    declared_len = response.headers.get("content-length")
                    if declared_len is not None:
                        try:
                            if int(declared_len) > self._max_bytes:
                                return ToolResult(
                                    output=(
                                        f"Error: File exceeds {self._max_mb}MB "
                                        f"limit (server reports "
                                        f"{_format_size(int(declared_len))})"
                                    ),
                                    is_error=True,
                                )
                        except ValueError:
                            pass
                    with dest.open("wb") as fh:
                        async for chunk in response.aiter_bytes(chunk_size=65536):
                            bytes_written += len(chunk)
                            if bytes_written > self._max_bytes:
                                fh.close()
                                dest.unlink(missing_ok=True)
                                return ToolResult(
                                    output=(
                                        f"Error: Download exceeded "
                                        f"{self._max_mb}MB limit — aborted."
                                    ),
                                    is_error=True,
                                )
                            fh.write(chunk)
        except httpx.HTTPError as exc:
            dest.unlink(missing_ok=True)
            return ToolResult(output=f"Error: download failed: {exc}", is_error=True)
        except OSError as exc:
            return ToolResult(
                output=f"Error: filesystem error writing {dest}: {exc}",
                is_error=True,
            )

        return ToolResult(
            output=(
                f"Downloaded {_format_size(bytes_written)} to {dest}\n"
                f"Content-Type: {content_type or '(unknown)'}"
            ),
            metadata={
                "path": str(dest),
                "bytes": bytes_written,
                "content_type": content_type,
            },
        )


# ---------------------------------------------------------------------------
# Path resolution + traversal guard
# ---------------------------------------------------------------------------

def _resolve_destination(url: str, requested: str | None) -> Path:
    """Resolve the destination path. Apply path-traversal guard and reject
    common system paths (``/etc``, ``/sys``, ``/boot``).
    """
    if requested is None:
        candidate = _DOWNLOAD_DIR / _filename_from_url(url)
    else:
        candidate = Path(requested).expanduser()
    candidate = candidate.resolve() if candidate.is_absolute() else (
        Path.cwd() / candidate
    ).resolve()

    forbidden_prefixes = (
        Path("/etc"),
        Path("/sys"),
        Path("/boot"),
        Path("/proc"),
        Path("/dev"),
        Path.home() / ".ssh",
    )
    for forbidden in forbidden_prefixes:
        try:
            candidate.relative_to(forbidden)
        except ValueError:
            continue
        raise ValueError(f"Destination {candidate} is in a protected path")

    return candidate


def _filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = unquote(Path(parsed.path).name)
    if not name:
        host = parsed.hostname or "download"
        return _FILENAME_FALLBACK_RE.sub("_", host) + ".bin"
    return name


# ---------------------------------------------------------------------------
# SSRF protection (mirrors web_fetch._is_safe_url)
# ---------------------------------------------------------------------------

def _is_safe_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return False
        addrs = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for family, _, _, _, sockaddr in addrs:
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local:
                return False
    except (socket.gaierror, ValueError, OSError):
        return False
    return True


def _format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024
    return f"{size:.1f}TB"
