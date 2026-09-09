# Provenance: NousResearch/hermes-agent (https://github.com/NousResearch/hermes-agent)
# Original: tools/browser_tool.py
# License: MIT
# Modified: Rewritten as Prometheus BaseTool; uses playwright directly, core actions only

"""Headless browser automation via Playwright."""

from __future__ import annotations

from collections import OrderedDict
from enum import Enum
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult

try:
    from playwright.async_api import (
        async_playwright,
        Browser,
        BrowserContext,
        Page,
        Playwright,
    )
except ImportError:
    async_playwright = None  # type: ignore[assignment,misc]

#: What ``navigate`` may open. A browser resolves far more than a fetcher does:
#: ``file://`` renders any readable path as a page, and ``data:``/``javascript:``
#: execute content the model composed itself. There is no legitimate case for
#: rendering a system file as a web page when ``read_file`` already exists for
#: reading files, so this is unconditional rather than config-gated.
#:
#: NOT an address-class check. Chromium follows its OWN redirects and in-page
#: navigations with no per-hop hook, so a public page that redirects to loopback
#: is still followed — closing that needs ``page.route`` interception, which
#: MUST be designed against a real playwright install and is deliberately left
#: out here (see the PR). Hop 0 pointing at the operator's own services stays
#: allowed on purpose; that is a documented legitimate use.
ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})

#: Isolated contexts kept alive at once. Each is a separate cookie jar, so this
#: bounds how much state a long-lived daemon accumulates; the least recently
#: used is closed when a new session needs one.
MAX_CONTEXTS = 4

_NOT_INSTALLED = (
    "playwright is not installed. Run: pip install playwright && "
    "playwright install chromium"
)

#: The session key used when a caller builds a ToolExecutionContext with no
#: metadata (jobs, cron). Those callers share one context with each other, and
#: nothing else — they are all daemon-internal.
_NO_SESSION = "__no_session__"


class BrowserAction(str, Enum):
    navigate = "navigate"
    snapshot = "snapshot"
    click = "click"
    type = "type"
    scroll = "scroll"
    close = "close"


class BrowserInput(BaseModel):
    """Arguments for browser automation."""

    action: BrowserAction = Field(description="Browser action to perform")
    url: str | None = Field(default=None, description="URL for navigate action")
    selector: str | None = Field(
        default=None, description="CSS selector for click/type actions"
    )
    text: str | None = Field(default=None, description="Text for type action")
    direction: str | None = Field(
        default=None, description="Scroll direction: 'up' or 'down'"
    )


class BrowserTool(BaseTool):
    """Headless browser automation with navigate, snapshot, click, type, scroll, close."""

    name = "browser"
    # The tool stays REGISTERED when playwright is absent, and says so here.
    #
    # Registering conditionally looks tidier and is what __main__'s "requires
    # optional playwright dependency" comment intended — try_register skips a
    # module that fails to import. browser.py swallowing the ImportError has
    # always defeated that, which is why /health's tool-registration section
    # cannot show the BrowserTool/"No module named 'playwright'" row that
    # tests/test_health_tool_registration_section.py synthesises as its own
    # example.
    #
    # It is still not the fix. tests/test_tool_advertisement.py asserts every
    # DEFERRED_BY_DESIGN entry is registered and drives a discoverability probe
    # per entry, so conditional registration would fail the suite on any machine
    # without playwright — CI included — and make the tool inventory depend on
    # the environment, which those guards exist to prevent. So the model is told
    # in the one place it reads before spending a call.
    description = (
        "Headless browser automation for pages that require JavaScript "
        "rendering, login flows, or interactive elements. Use web_fetch "
        "instead for simple URL fetching — only use browser when the page "
        "won't work without JavaScript."
    ) + ("" if async_playwright is not None else (
        " UNAVAILABLE on this install: playwright is not installed, so every "
        "call returns an error. Use web_fetch instead."
    ))
    input_model = BrowserInput

    def __init__(self) -> None:
        # These were CLASS attributes assigned through ``self``, so the
        # declared default and the live value lived in different places. One
        # registry is built per daemon, so in practice it was ONE page, in one
        # context, for the daemon's whole uptime: every session navigating with
        # every other session's cookies and localStorage still loaded.
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        # session id -> (context, page). Ordered so the least recently used
        # context is the one evicted.
        self._contexts: "OrderedDict[str, tuple[BrowserContext, Page]]" = OrderedDict()

    # -- lifecycle ---------------------------------------------------------

    async def _ensure_page(self, session_id: str) -> Page:
        """The page for THIS session, in a context no other session shares."""
        if async_playwright is None:
            raise RuntimeError(_NOT_INSTALLED)

        if self._browser is None or not self._browser.is_connected():
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=True)
            # Contexts belonged to a browser process that is gone; dropping the
            # handles without closing them is correct, and keeping them would
            # hand out pages that raise on first use.
            self._contexts.clear()

        entry = self._contexts.get(session_id)
        if entry is not None:
            # Reused WITHIN a session on purpose: a login flow is several tool
            # calls, and a fresh context per navigate would throw the session
            # cookie away between them.
            self._contexts.move_to_end(session_id)
            return entry[1]

        while len(self._contexts) >= MAX_CONTEXTS:
            _evicted, (old_context, _old_page) = self._contexts.popitem(last=False)
            await old_context.close()

        context = await self._browser.new_context()
        page = await context.new_page()
        self._contexts[session_id] = (context, page)
        return page

    @staticmethod
    def _session_of(context: ToolExecutionContext) -> str:
        """The isolation key, from the loop's trusted metadata."""
        session_id = (context.metadata or {}).get("session_id")
        return str(session_id) if session_id else _NO_SESSION

    async def execute(
        self, arguments: BrowserInput, context: ToolExecutionContext
    ) -> ToolResult:
        if async_playwright is None:
            return ToolResult(output=_NOT_INSTALLED, is_error=True)

        session = self._session_of(context)
        action = arguments.action
        try:
            if action == BrowserAction.navigate:
                return await self._navigate(arguments, session)
            elif action == BrowserAction.snapshot:
                return await self._snapshot(session)
            elif action == BrowserAction.click:
                return await self._click(arguments, session)
            elif action == BrowserAction.type:
                return await self._type(arguments, session)
            elif action == BrowserAction.scroll:
                return await self._scroll(arguments, session)
            elif action == BrowserAction.close:
                return await self._close(session)
            else:
                return ToolResult(output=f"Unknown action: {action}", is_error=True)
        except Exception as exc:
            return ToolResult(output=f"browser error: {exc}", is_error=True)

    async def _navigate(self, args: BrowserInput, session: str) -> ToolResult:
        if not args.url:
            return ToolResult(output="navigate requires a url", is_error=True)

        refusal = _refuse_scheme(args.url)
        if refusal:
            # Refused BEFORE the browser is touched, so a refused navigate
            # cannot be the thing that launches Chromium.
            return ToolResult(output=refusal, is_error=True)

        page = await self._ensure_page(session)
        resp = await page.goto(args.url, wait_until="networkidle", timeout=30000)
        status = resp.status if resp else "unknown"
        title = await page.title()
        return ToolResult(output=f"Navigated to {args.url} (status={status}, title={title})")

    async def _snapshot(self, session: str) -> ToolResult:
        page = await self._ensure_page(session)
        # Get accessibility tree as text representation
        text = await page.evaluate("""() => {
            function walk(node, depth) {
                let result = '';
                const indent = '  '.repeat(depth);
                if (node.nodeType === Node.TEXT_NODE) {
                    const t = node.textContent.trim();
                    if (t) result += indent + t + '\\n';
                } else if (node.nodeType === Node.ELEMENT_NODE) {
                    const tag = node.tagName.toLowerCase();
                    const role = node.getAttribute('role') || '';
                    const label = node.getAttribute('aria-label') || '';
                    const href = node.getAttribute('href') || '';
                    let meta = [tag];
                    if (role) meta.push('role=' + role);
                    if (label) meta.push('label=' + label);
                    if (href) meta.push('href=' + href);
                    if (['script','style','noscript','svg'].includes(tag)) return '';
                    result += indent + '[' + meta.join(' ') + ']\\n';
                    for (const child of node.childNodes) {
                        result += walk(child, depth + 1);
                    }
                }
                return result;
            }
            return walk(document.body, 0);
        }""")
        if len(text) > 15000:
            text = text[:15000] + "\n...[truncated]"
        title = await page.title()
        url = page.url
        return ToolResult(output=f"Page: {title} ({url})\n\n{text}")

    async def _click(self, args: BrowserInput, session: str) -> ToolResult:
        if not args.selector:
            return ToolResult(output="click requires a selector", is_error=True)
        page = await self._ensure_page(session)
        await page.click(args.selector, timeout=5000)
        return ToolResult(output=f"Clicked: {args.selector}")

    async def _type(self, args: BrowserInput, session: str) -> ToolResult:
        if not args.selector or not args.text:
            return ToolResult(output="type requires selector and text", is_error=True)
        page = await self._ensure_page(session)
        await page.fill(args.selector, args.text, timeout=5000)
        return ToolResult(output=f"Typed into {args.selector}: {args.text[:100]}")

    async def _scroll(self, args: BrowserInput, session: str) -> ToolResult:
        page = await self._ensure_page(session)
        direction = (args.direction or "down").lower()
        delta = -500 if direction == "up" else 500
        await page.mouse.wheel(0, delta)
        return ToolResult(output=f"Scrolled {direction}")

    async def _close(self, session: str) -> ToolResult:
        """Close THIS session's context; the browser only when none are left.

        Closing the shared browser on any session's ``close`` would silently
        discard another session's login. The playwright driver is stopped with
        it — the old code kept the ``start()`` handle in a local, so the driver
        subprocess could never be stopped at all and leaked on every relaunch.
        """
        entry = self._contexts.pop(session, None)
        if entry is not None:
            await entry[0].close()

        if self._contexts:
            return ToolResult(
                output=f"Browser context closed for this session "
                       f"({len(self._contexts)} other session(s) still open)."
            )

        if self._browser is not None and self._browser.is_connected():
            await self._browser.close()
        self._browser = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None
        return ToolResult(output="Browser closed.")


def _refuse_scheme(url: str) -> str | None:
    """Why *url* may not be opened, or None when it may.

    Returns a reason the model can act on rather than a bare denial, because a
    refusal it cannot interpret is a refusal it retries.
    """
    try:
        parsed = urlparse((url or "").strip())
    except ValueError:
        return "Blocked: that URL is malformed."

    scheme = (parsed.scheme or "").lower()
    if not scheme:
        return (
            "Blocked: navigate needs an absolute http:// or https:// URL "
            f"({url!r} has no scheme)."
        )
    if scheme not in ALLOWED_SCHEMES:
        if scheme == "file":
            return (
                "Blocked: the browser does not open file:// URLs. Use "
                "read_file to read a local file."
            )
        return (
            f"Blocked: {scheme}: URLs are not navigable "
            f"(http and https only)."
        )
    return None
