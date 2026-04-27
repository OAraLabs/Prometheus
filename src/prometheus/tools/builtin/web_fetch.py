# Provenance: HKUDS/OpenHarness (https://github.com/HKUDS/OpenHarness)
# Original: src/openharness/tools/web_fetch_tool.py
# License: Apache-2.0
# Modified: Rewritten as Prometheus BaseTool; added SSRF protection;
#           replaced regex-based stripper with html.parser-based extractor
#           that preserves dotted names (e.g. asyncio.gather) and applies
#           a content-priority order (article > main > div[role=main] > body
#           > whole doc) — keeps signal, drops nav/footer noise.

"""Fetch a web page and return compact readable text."""

from __future__ import annotations

import ipaddress
import socket
from html.parser import HTMLParser
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult


class WebFetchInput(BaseModel):
    """Arguments for fetching one web page."""

    url: str = Field(description="HTTP or HTTPS URL to fetch")
    max_chars: int = Field(
        default=30000, ge=500, le=80000, description="Maximum characters to return"
    )


class WebFetchTool(BaseTool):
    """Fetch one web page and return compact readable text."""

    name = "web_fetch"
    description = (
        "Fetch any URL (web page, GitHub repo, API endpoint, raw file) and "
        "return its text content. Use this for: fetching GitHub repos, reading "
        "documentation, calling JSON APIs, reading any public URL. Handles "
        "404/403/private URLs gracefully without crashing."
    )
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
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return False
        addrs = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for _family, _, _, _, sockaddr in addrs:
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local:
                return False
    except (socket.gaierror, ValueError, OSError):
        return False
    return True


# ---------------------------------------------------------------------------
# HTML → text  (html.parser-based, no beautifulsoup)
# ---------------------------------------------------------------------------

# Inline tags emit NO separator when stripped — preserves "asyncio.gather"
# rather than splitting it into "asyncio . gather" the way a regex
# tag-to-space pass would.
_INLINE_TAGS = frozenset({
    "a", "abbr", "b", "cite", "code", "em", "i", "kbd", "mark", "q",
    "s", "samp", "small", "span", "strong", "sub", "sup", "tt", "u",
    "var", "wbr",
})

# Block tags emit a newline separator — keeps paragraph and list structure.
_BLOCK_TAGS = frozenset({
    "address", "blockquote", "br", "div", "dd", "dl", "dt", "figcaption",
    "figure", "h1", "h2", "h3", "h4", "h5", "h6", "hr", "li", "ol", "p",
    "pre", "section", "table", "tbody", "td", "tfoot", "th", "thead",
    "tr", "ul",
})

# These tags and their entire subtree are pruned before extraction.
_SKIP_TAGS = frozenset({
    "aside", "footer", "header", "nav", "noscript", "script", "style",
    "template",
})


class _TextExtractor(HTMLParser):
    """Extract readable text from an HTML stream.

    If ``target_tag`` is set, text is captured only while inside an element
    matching ``target_tag`` (and any required ``target_attrs``). When
    ``target_tag`` is None, text is captured throughout the document.
    Skip-listed tags (script, style, nav, header, footer, aside, noscript,
    template) and their subtrees are pruned regardless of target.
    """

    def __init__(
        self,
        target_tag: str | None = None,
        target_attrs: dict[str, str] | None = None,
    ) -> None:
        super().__init__(convert_charrefs=True)
        self._target_tag = target_tag
        self._target_attrs = target_attrs or {}
        self._target_depth = 0
        self._skip_depth = 0
        self._parts: list[str] = []
        # Per-open stack of "did this target_tag open match the attrs?".
        # Without this we'd decrement target_depth on every </div> instead
        # of only on the matching one, dropping out of capture early on
        # pages with many same-tag siblings.
        self._target_match_stack: list[bool] = []

    @property
    def _capturing(self) -> bool:
        if self._skip_depth > 0:
            return False
        if self._target_tag is None:
            return True
        return self._target_depth > 0

    def _push_block_separator(self) -> None:
        if self._parts and not self._parts[-1].endswith("\n"):
            self._parts.append("\n")

    def _attrs_match(
        self, attrs: list[tuple[str, str | None]]
    ) -> bool:
        if not self._target_attrs:
            return True
        attr_dict = {k: v for k, v in attrs if v is not None}
        return all(
            attr_dict.get(k) == v for k, v in self._target_attrs.items()
        )

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return

        if self._target_tag and tag == self._target_tag:
            matched = self._attrs_match(attrs)
            self._target_match_stack.append(matched)
            if matched:
                self._target_depth += 1

        if self._capturing and tag in _BLOCK_TAGS:
            self._push_block_separator()

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        # Void / self-closing tags like <br/>, <hr/>. A self-closing target
        # tag has no body to capture, so it doesn't change target_depth.
        if tag in _SKIP_TAGS:
            return
        if self._capturing and tag in _BLOCK_TAGS:
            self._push_block_separator()

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return

        if self._capturing and tag in _BLOCK_TAGS:
            self._push_block_separator()

        if self._target_tag and tag == self._target_tag and self._target_match_stack:
            was_match = self._target_match_stack.pop()
            if was_match and self._target_depth > 0:
                self._target_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._capturing:
            self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        # Per-line whitespace collapse, dedup blank lines.
        lines: list[str] = []
        prev_blank = True
        for line in raw.splitlines():
            stripped = " ".join(line.split())
            if not stripped:
                if not prev_blank:
                    lines.append("")
                prev_blank = True
            else:
                lines.append(stripped)
                prev_blank = False
        return "\n".join(lines).strip()


# Priority order: article > main > div[role=main] > body > whole doc.
# First target that yields > _MIN_USEFUL_CHARS of text wins.
_TARGET_PRIORITY: tuple[tuple[str | None, dict[str, str] | None], ...] = (
    ("article", None),
    ("main", None),
    ("div", {"role": "main"}),
    ("body", None),
    (None, None),
)
_MIN_USEFUL_CHARS = 50


def _html_to_text(raw: str) -> str:
    """Extract readable text from HTML.

    Tries content-priority targets in order: <article> first, then <main>,
    <div role="main">, <body>, whole document. Returns the text from the
    first target that produces a non-trivial amount of content (>50 chars).
    Inline tags (code/span/a/etc.) emit no separator — dotted names like
    ``asyncio.gather`` stay glued. Block tags (p/div/h1/li/etc.) emit a
    newline separator. Script/style/nav/header/footer/aside subtrees are
    dropped before extraction.
    """
    if not raw:
        return ""
    last_text = ""
    for target_tag, target_attrs in _TARGET_PRIORITY:
        extractor = _TextExtractor(
            target_tag=target_tag, target_attrs=target_attrs
        )
        try:
            extractor.feed(raw)
            extractor.close()
        except Exception:
            continue
        text = extractor.text()
        if len(text) >= _MIN_USEFUL_CHARS:
            return text
        # Remember the last non-empty fallback in case nothing crosses the
        # threshold (e.g. very short page with just a sentence).
        if text and len(text) > len(last_text):
            last_text = text
    return last_text
