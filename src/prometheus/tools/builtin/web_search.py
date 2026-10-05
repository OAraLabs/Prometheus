# Provenance: HKUDS/OpenHarness (https://github.com/HKUDS/OpenHarness)
# Original: src/openharness/tools/web_search_tool.py
# License: MIT
# Modified: Adapted as Prometheus BaseTool; a backend chain (SearXNG, Brave,
#           DuckDuckGo html then lite) with block detection, in
#           web_search_backends.py

"""Web search through a backend chain that ends in DuckDuckGo (no key required).

The backends, the chain and the block detection are in
:mod:`prometheus.tools.builtin.web_search_backends`. This module runs the chain,
says which backend answered and which were skipped, and writes one telemetry
row per call.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any

import httpx
from pydantic import BaseModel, Field

from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult
from prometheus.tools.builtin import web_search_backends as wsb

log = logging.getLogger(__name__)

#: The fresh-install description, byte for byte what every parity golden
#: recorded. Only the first sentence changes when other backends are active.
_DEFAULT_DESCRIPTION = (
    "Search the web via DuckDuckGo and return top results with titles, "
    "URLs, and snippets. Use this when you don't know a specific URL: "
    "finding documentation, looking up current facts (versions, prices, "
    "news), discovering libraries or repositories, comparing options, or "
    "researching unfamiliar topics. Once you have a URL from search "
    "results, use web_fetch to read its full content. No API key required."
)
_DEFAULT_LEAD = "Search the web via DuckDuckGo"


class WebSearchInput(BaseModel):
    """Arguments for a web search."""

    query: str = Field(description="Search query")
    max_results: int = Field(
        default=5, ge=1, le=10, description="Maximum number of results to return"
    )


class WebSearchTool(BaseTool):
    """Search the web through the configured backends and return top results."""

    name = "web_search"
    description = _DEFAULT_DESCRIPTION
    input_model = WebSearchInput

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """``config`` is the ``web_search`` section (None = a fresh install).
        ``transport`` is for tests; production uses httpx's own."""
        self.plan = wsb.plan_backends(config)
        self._transport = transport
        if not self.plan.is_fresh_install:
            self.description = describe(self.plan.active)
            log.info(
                "web_search: backends %s%s", " -> ".join(self.plan.active),
                "".join(f"; {n} inactive ({why})" for n, why in self.plan.inactive.items()),
            )

    def is_read_only(self, arguments: WebSearchInput) -> bool:
        return True

    async def execute(
        self,
        arguments: WebSearchInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        started = time.monotonic()
        plan = self._current_plan()
        skipped: list[dict[str, str]] = []
        backend: str | None = None
        endpoint: str | None = None
        hits: list[wsb.SearchHit] = []
        last: wsb.BackendFailure | None = None

        async with httpx.AsyncClient(
            follow_redirects=True, transport=self._transport,
        ) as client:
            for name in plan.active:
                try:
                    found, where = await self._ask(client, name, plan, arguments)
                except wsb.BackendFailure as exc:
                    last = exc
                    if name != wsb.FINAL_BACKEND:
                        skipped.append({"backend": name, "kind": exc.kind, "reason": exc.reason})
                    continue
                backend, endpoint, hits = name, where, found
                break

        blocked = backend is None and last is not None and last.kind == wsb.KIND_BLOCKED
        result = _render(arguments.query, backend, endpoint, hits, skipped, last, blocked)
        metadata = {
            "backend": backend,
            "endpoint": endpoint,
            "skipped": skipped,
            "inactive": dict(plan.inactive),
            "blocked": blocked,
        }
        _record(context, metadata, len(hits), (time.monotonic() - started) * 1000.0)
        return ToolResult(output=result[0], is_error=result[1], metadata=metadata)

    def _current_plan(self) -> wsb.BackendPlan:
        """The construction-time order, with Brave dropped if its key has gone
        since (a key added later needs a restart, like every other key)."""
        if wsb.BRAVE in self.plan.active and not wsb.resolve_brave_key():
            active = tuple(n for n in self.plan.active if n != wsb.BRAVE)
            inactive = {**self.plan.inactive,
                        wsb.BRAVE: f"no {wsb.BRAVE_KEY_ENV} in the environment or the env file"}
            return wsb.BackendPlan(active, inactive, self.plan.searxng_url)
        return self.plan

    @staticmethod
    async def _ask(
        client: httpx.AsyncClient, name: str, plan: wsb.BackendPlan,
        arguments: WebSearchInput,
    ) -> tuple[list[wsb.SearchHit], str | None]:
        if name == wsb.SEARXNG:
            return await wsb.search_searxng(
                client, plan.searxng_url, arguments.query, arguments.max_results,
            ), None
        if name == wsb.BRAVE:
            return await wsb.search_brave(
                client, wsb.resolve_brave_key(), arguments.query, arguments.max_results,
            ), None
        answer = await wsb.search_duckduckgo(client, arguments.query, arguments.max_results)
        return answer.hits, answer.endpoint


def describe(active: tuple[str, ...]) -> str:
    """The tool description for a chain. A fresh install's is the default."""
    if active == (wsb.FINAL_BACKEND,):
        return _DEFAULT_DESCRIPTION
    names = [wsb.DISPLAY_NAMES[n] for n in active]
    lead = f"Search the web via {names[0]} (falling back to {', then '.join(names[1:])})"
    return lead + _DEFAULT_DESCRIPTION[len(_DEFAULT_LEAD):]


def _render(
    query: str,
    backend: str | None,
    endpoint: str | None,
    hits: list[wsb.SearchHit],
    skipped: list[dict[str, str]],
    last: wsb.BackendFailure | None,
    blocked: bool,
) -> tuple[str, bool]:
    """(output, is_error)."""
    skipped_line = (
        "Skipped: " + "; ".join(f"{s['backend']} ({s['reason']})" for s in skipped)
        if skipped else ""
    )
    if backend is None:
        # Every backend failed. The last is DuckDuckGo, whose reason leads.
        reason = last.reason if last is not None else "no backend is active"
        if blocked:
            text = (
                f"web search is blocked right now: duckduckgo {reason}. This is not "
                f"an empty result: no backend could run the search."
            )
        else:
            text = f"web search failed: duckduckgo {reason}."
        return (text + (f" {skipped_line}." if skipped_line else ""), True)

    if not hits:
        text = f"No search results found for: {query} (via {backend})."
        return (text + (f"\n{skipped_line}" if skipped_line else ""), True)

    lines = [f"Search results for: {query} (via {backend})"]
    if skipped_line:
        lines.append(skipped_line)
    for index, hit in enumerate(hits, start=1):
        lines.append(f"{index}. {hit.title}")
        lines.append(f"   URL: {hit.url}")
        if hit.snippet:
            lines.append(f"   {hit.snippet}")
    return ("\n".join(lines), False)


def _record(
    context: ToolExecutionContext, metadata: dict[str, Any], results: int,
    duration_ms: float,
) -> None:
    """One subsystem_runs row per call: which backend answered, what was skipped.

    No query and no reason text (a reason can carry the SearXNG URL): kinds and
    backend names only. Best-effort — telemetry never costs the model its search.
    """
    from prometheus.telemetry.tracker import get_telemetry_handle

    telemetry = get_telemetry_handle()
    if telemetry is None:
        return
    meta = context.metadata or {}
    session_id = None if meta.get("ephemeral") else (
        meta.get("effective_session_id") or meta.get("session_id")
    )
    if metadata["backend"] is None:
        outcome = "failed"
    elif metadata["skipped"]:
        outcome = "partial"
    else:
        outcome = "success"
    summary = {
        "backend": metadata["backend"],
        "endpoint": metadata["endpoint"],
        "results": results,
        "skipped": [{"backend": s["backend"], "kind": s["kind"]} for s in metadata["skipped"]],
        "inactive": list(metadata["inactive"]),
        "blocked": metadata["blocked"],
    }
    try:
        telemetry.record_run(
            wsb.WEB_SEARCH_SUBSYSTEM, wsb.WEB_SEARCH_OPERATION, outcome,
            duration_ms=duration_ms, summary=summary, session_id=session_id,
        )
    except Exception:
        log.debug("web_search telemetry write failed", exc_info=True)

