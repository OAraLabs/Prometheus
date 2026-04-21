# Provenance: NousResearch/hermes-agent (https://github.com/NousResearch/hermes-agent)
# Original: tools/browser_tool.py
# License: MIT
# Modified: Rewritten as Prometheus BaseTool; uses playwright directly, core actions only

"""Headless browser automation via Playwright."""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult

try:
    from playwright.async_api import async_playwright, Browser, Page
except ImportError:
    async_playwright = None  # type: ignore[assignment,misc]


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
    description = (
        "Automate a headless Chromium browser. Actions: navigate (load URL), "
        "snapshot (get page text), click (CSS selector), type (text into selector), "
        "scroll (up/down), close."
    )
    input_model = BrowserInput

    _browser: Browser | None = None
    _page: Page | None = None

    async def _ensure_browser(self) -> Page:
        if async_playwright is None:
            raise RuntimeError("playwright is not installed. Run: pip install playwright && playwright install chromium")
        if self._browser is None or not self._browser.is_connected():
            pw = await async_playwright().start()
            self._browser = await pw.chromium.launch(headless=True)
            self._page = await self._browser.new_page()
        assert self._page is not None
        return self._page

    async def execute(
        self, arguments: BrowserInput, context: ToolExecutionContext
    ) -> ToolResult:
        if async_playwright is None:
            return ToolResult(
                output="playwright is not installed. Run: pip install playwright && playwright install chromium",
                is_error=True,
            )

        action = arguments.action
        try:
            if action == BrowserAction.navigate:
                return await self._navigate(arguments)
            elif action == BrowserAction.snapshot:
                return await self._snapshot()
            elif action == BrowserAction.click:
                return await self._click(arguments)
            elif action == BrowserAction.type:
                return await self._type(arguments)
            elif action == BrowserAction.scroll:
                return await self._scroll(arguments)
            elif action == BrowserAction.close:
                return await self._close()
            else:
                return ToolResult(output=f"Unknown action: {action}", is_error=True)
        except Exception as exc:
            return ToolResult(output=f"browser error: {exc}", is_error=True)

    async def _navigate(self, args: BrowserInput) -> ToolResult:
        if not args.url:
            return ToolResult(output="navigate requires a url", is_error=True)
        page = await self._ensure_browser()
        resp = await page.goto(args.url, wait_until="networkidle", timeout=30000)
        status = resp.status if resp else "unknown"
        title = await page.title()
        return ToolResult(output=f"Navigated to {args.url} (status={status}, title={title})")

    async def _snapshot(self) -> ToolResult:
        page = await self._ensure_browser()
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

    async def _click(self, args: BrowserInput) -> ToolResult:
        if not args.selector:
            return ToolResult(output="click requires a selector", is_error=True)
        page = await self._ensure_browser()
        await page.click(args.selector, timeout=5000)
        return ToolResult(output=f"Clicked: {args.selector}")

    async def _type(self, args: BrowserInput) -> ToolResult:
        if not args.selector or not args.text:
            return ToolResult(output="type requires selector and text", is_error=True)
        page = await self._ensure_browser()
        await page.fill(args.selector, args.text, timeout=5000)
        return ToolResult(output=f"Typed into {args.selector}: {args.text[:100]}")

    async def _scroll(self, args: BrowserInput) -> ToolResult:
        page = await self._ensure_browser()
        direction = (args.direction or "down").lower()
        delta = -500 if direction == "up" else 500
        await page.mouse.wheel(0, delta)
        return ToolResult(output=f"Scrolled {direction}")

    async def _close(self) -> ToolResult:
        if self._browser and self._browser.is_connected():
            await self._browser.close()
            self._browser = None
            self._page = None
        return ToolResult(output="Browser closed.")
