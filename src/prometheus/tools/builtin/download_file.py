# Source: Original implementation for Prometheus (WEAVE Session A)
# License: MIT
# Purpose: Stream a URL to local disk with size limit + path-traversal guard.

"""Download tool — fetches a URL and streams it to a file on disk."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx
from pydantic import BaseModel, Field

from prometheus.permissions.path_schema import DIR_FIELD, PATH_FIELD
# Both fetching tools use the ONE hook, from the security module — not one
# importing a private helper from its sibling, which is how two callers drift.
from prometheus.security.url_guard import (
    SsrfBlocked,
    guard_request_hop,
    is_safe_url,
)
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult


_DEFAULT_MAX_MB = 100
_DEFAULT_TIMEOUT = 60.0
_DOWNLOAD_DIR = Path.home() / ".prometheus" / "downloads"
_FILENAME_FALLBACK_RE = re.compile(r"[^A-Za-z0-9._-]+")


class DownloadFileInput(BaseModel):
    """Arguments for downloading a file from a URL."""

    url: str = Field(description="HTTP or HTTPS URL to download")
    destination: str | None = Field(
        json_schema_extra=PATH_FIELD,
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
        if not is_safe_url(arguments.url):
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
                follow_redirects=True,
                timeout=_DEFAULT_TIMEOUT,
                # Per-hop SSRF guard — see fetch_url_text's docstring in
                # web_fetch.py. A 302 to a loopback/tailnet address must abort
                # the chain before the body is streamed to disk, not after.
                event_hooks={"request": [guard_request_hop]},
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
        except SsrfBlocked as exc:
            # A redirect hop resolved to a non-public address. Remove the partial
            # file so nothing half-written survives, and say we refused rather
            # than reporting a transport failure the model would retry.
            dest.unlink(missing_ok=True)
            return ToolResult(
                output=f"Blocked: a redirect went to a non-public address ({exc}).",
                is_error=True,
            )
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
    derived = requested is None
    if derived:
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

    if derived:
        # THE INVARIANT THE FORBIDDEN LIST CANNOT EXPRESS: when the URL chose the
        # name, the result must still be inside the download directory. The
        # forbidden-prefix list above is a denylist of six locations — it says
        # nothing about ~/.bashrc, ~/.profile or ~/.prometheus itself, which is
        # exactly where the %2F traversal landed. A containment check is the
        # property; the denylist is only ever a partial list of places not to go.
        #
        # An explicit `destination` is NOT held to this — the operator (or the
        # model, at the operator's origin, through the gate) may name a path, and
        # that is the gate's business, not this function's.
        anchor = _DOWNLOAD_DIR.resolve()
        if not candidate.is_relative_to(anchor):
            raise ValueError(
                f"a filename derived from the URL must land under {anchor}, "
                f"not {candidate} — pass an explicit destination instead"
            )

    return candidate


def _filename_from_url(url: str) -> str:
    """Derive a SAFE filename from a URL path.

    ORDER IS THE WHOLE FIX. This used to take the last path segment first and
    unquote it afterwards::

        name = unquote(Path(parsed.path).name)

    With a percent-encoded separator that inverts the intent: ``Path(
    '/x%2F..%2F..%2F..%2F.bashrc').name`` is the single segment
    ``'x%2F..%2F..%2F..%2F.bashrc'``, and unquoting it afterwards turns the
    ``%2F``s into real separators. The caller then did
    ``_DOWNLOAD_DIR / name``, so the URL chose the write target — measured to
    resolve to ``$HOME/.bashrc``. A page the model was asked to fetch could aim
    download_file at shell profiles or systemd user units with no gate prompt
    (``destination`` is optional, and the gate has nothing to rule on when it is
    unset) and no denied_paths check, since the forbidden-prefix list never saw
    the escape.

    Unquoting BEFORE reducing to a basename means the traversal is already a real
    path when ``.name`` runs, so only the final component can survive — and a
    basename can never contain a separator. ``_resolve_destination`` then holds
    derived names inside the download directory, so containment is a property of
    the call rather than of how well this function guesses.

    This does NOT restrict which URLs may be fetched: the same URL downloads
    fine, it just lands under the download directory with its own name.
    """
    parsed = urlparse(url)
    # Unquote first, so %2F and friends are real separators before .name runs.
    decoded = unquote(parsed.path or "")
    name = Path(decoded).name

    if not name:
        host = parsed.hostname or "download"
        return _FILENAME_FALLBACK_RE.sub("_", host) + ".bin"

    # `.name` is a basename, so it cannot contain a separator and cannot escape —
    # the traversal is gone by construction. Three degenerate shapes are left, and
    # ONLY those are replaced:
    #   * the traversal dots themselves — `/..`, `//..` and `/./..` all give '..'
    #   * a NUL byte or control character from a decoded %00, which makes open()
    #     raise OSError instead of writing anything
    #
    # Spaces and punctuation are deliberately NOT sanitised. The pinned contract is
    # that `hello%20world.pdf` yields `hello world.pdf`, and a filename with a space
    # is a normal thing for a URL to carry. Stripping it would rewrite the name a
    # legitimate download asked for, to defend against a case `.name` already
    # handles. The fallback sanitiser above stays scoped to the HOST string, where
    # inventing a name anyway means there is no contract to keep.
    if name in (".", "..") or any(ord(ch) < 32 or ch == "\x7f" for ch in name):
        return "download.bin"
    return name


def _format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}" if unit != "B" else f"{int(size)}B"
        size /= 1024
    return f"{size:.1f}TB"
