# Provenance: HKUDS/OpenHarness (https://github.com/HKUDS/OpenHarness)
# Original: src/openharness/tools/web_search_tool.py
# License: Apache-2.0
# Modified: Rewritten as Prometheus BaseTool

"""Web search via DuckDuckGo HTML — no API key required."""

from __future__ import annotations

import html as html_mod
import re
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from pydantic import BaseModel, Field

from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult


class WebSearchInput(BaseModel):
    """Arguments for a web search."""

    query: str = Field(description="Search query")
    max_results: int = Field(
        default=5, ge=1, le=10, description="Maximum number of results to return"
    )


class WebSearchTool(BaseTool):
    """Search the web via DuckDuckGo and return top results."""

    name = "web_search"
    description = "Search the web and return top results with titles, URLs, and snippets."
    input_model = WebSearchInput

    def is_read_only(self, arguments: WebSearchInput) -> bool:
        return True

    async def execute(
        self,
        arguments: WebSearchInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        endpoint = "https://html.duckduckgo.com/html/"
        try:
            async with httpx.AsyncClient(
                follow_redirects=True, timeout=20.0
            ) as client:
                response = await client.get(
                    endpoint,
                    params={"q": arguments.query},
                    headers={"User-Agent": "Prometheus/0.1"},
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            return ToolResult(output=f"web_search failed: {exc}", is_error=True)

        results = _parse_search_results(response.text, limit=arguments.max_results)
        if not results:
            return ToolResult(output="No search results found.", is_error=True)

        lines = [f"Search results for: {arguments.query}"]
        for index, result in enumerate(results, start=1):
            lines.append(f"{index}. {result['title']}")
            lines.append(f"   URL: {result['url']}")
            if result["snippet"]:
                lines.append(f"   {result['snippet']}")
        return ToolResult(output="\n".join(lines))


# ---------------------------------------------------------------------------
# HTML parsing helpers
# ---------------------------------------------------------------------------

def _parse_search_results(body: str, *, limit: int) -> list[dict[str, str]]:
    snippets = [
        _clean_html(m.group("snippet"))
        for m in re.finditer(
            r'<(?:a|div|span)[^>]+class="[^"]*(?:result__snippet|result-snippet)[^"]*"[^>]*>'
            r"(?P<snippet>.*?)</(?:a|div|span)>",
            body,
            flags=re.IGNORECASE | re.DOTALL,
        )
    ]

    results: list[dict[str, str]] = []
    for idx, match in enumerate(
        re.finditer(
            r"<a(?P<attrs>[^>]+)>(?P<title>.*?)</a>",
            body,
            flags=re.IGNORECASE | re.DOTALL,
        )
    ):
        attrs = match.group("attrs")
        cls = re.search(r'class="(?P<c>[^"]+)"', attrs, re.IGNORECASE)
        if cls is None:
            continue
        names = cls.group("c")
        if "result__a" not in names and "result-link" not in names:
            continue
        href = re.search(r'href="(?P<h>[^"]+)"', attrs, re.IGNORECASE)
        if href is None:
            continue
        title = _clean_html(match.group("title"))
        url = _normalize_result_url(href.group("h"))
        snippet = snippets[idx] if idx < len(snippets) else ""
        if title and url:
            results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= limit:
            break
    return results


def _normalize_result_url(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(target) if target else raw_url
    return raw_url


def _clean_html(fragment: str) -> str:
    text = re.sub(r"(?s)<[^>]+>", " ", fragment)
    text = html_mod.unescape(text)
    return re.sub(r"\s+", " ", text).strip()
