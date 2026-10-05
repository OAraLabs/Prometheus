"""web_discover — find companies, people, papers or pages like a description or a URL.

Exa's neural search, as a tool of its own rather than a web_search backend: it
answers "what is LIKE this", not "what is true". Two modes, one per call:

* ``query`` — a description of what to find (``POST /search``);
* ``url``   — pages similar to that one (``POST /findSimilar``, which Exa's
  spec marks deprecated in favour of a descriptive query; it still serves it).

The tool EXISTS ONLY WITH A KEY. ``create_tool_registry`` registers it when
``EXA_API_KEY`` is in the environment or the env file (never prometheus.yaml),
so a fresh install's tool set, its advertised schemas and every parity golden
are untouched. With a key it is advertised to a config that follows the
shipped ``always_loaded`` default (``advertise_when_registered``); an operator's
pinned list is used as written.

Results are title, url and a summary cut to :data:`SUMMARY_CHARS`, at most
:data:`MAX_RESULTS` of them. A URL on a private or local address is never sent
to Exa: that would hand a third party the shape of the operator's network.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Literal
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field, model_validator

from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult

log = logging.getLogger(__name__)

EXA_KEY_ENV = "EXA_API_KEY"
EXA_BASE = "https://api.exa.ai"
MAX_RESULTS = 10
DEFAULT_RESULTS = 5
SUMMARY_CHARS = 300
#: Seconds for the one Exa request: httpx's per-phase timeout AND a wall-clock
#: limit (asyncio.wait_for), well inside the agent loop's 300 s tool timeout.
TIMEOUT = 30.0
#: Guides Exa's per-result summary toward one or two sentences.
SUMMARY_QUERY = "In one or two sentences: what is this, and what does it do or study?"


def resolve_exa_key() -> str:
    """The Exa key from the environment, then the env file; "" when neither has it."""
    from prometheus.tools.builtin.web_search_backends import read_secret

    return read_secret(EXA_KEY_ENV)


class WebDiscoverInput(BaseModel):
    """Arguments for a neural search: a description, or a URL to find pages like."""

    query: str | None = Field(
        default=None,
        description=(
            "What to find, described in words, e.g. 'startups building open-source "
            "vector databases'. Give this or url, not both."
        ),
    )
    url: str | None = Field(
        default=None,
        description="Find pages similar to this http(s) URL. Give this or query, not both.",
    )
    category: Literal["company", "people", "publication"] | None = Field(
        default=None,
        description=(
            "Optional focus: company, people, or publication (research papers)."
        ),
    )
    num_results: int = Field(
        default=DEFAULT_RESULTS, ge=1, le=MAX_RESULTS,
        description=f"How many results to return (1-{MAX_RESULTS})",
    )

    @model_validator(mode="after")
    def _one_mode(self) -> WebDiscoverInput:
        has_query = bool(self.query and self.query.strip())
        has_url = bool(self.url and self.url.strip())
        if has_query == has_url:
            raise ValueError(
                "give either query (a description) or url (to find pages like it), "
                "not both and not neither"
            )
        return self


class WebDiscoverTool(BaseTool):
    """Exa neural search: things similar to a description or a URL."""

    name = "web_discover"
    description = (
        "Find companies, people, papers or pages similar to a description or "
        "URL (neural search). Not for facts or news; use web_search for those."
    )
    input_model = WebDiscoverInput
    example_call = {"query": "startups building open-source vector databases", "num_results": 5}
    #: Registered only with a key, so being registered is the operator's choice.
    advertise_when_registered = True

    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        """``transport`` is for tests; production uses httpx's own."""
        self._transport = transport

    def is_read_only(self, arguments: BaseModel) -> bool:
        return True

    async def execute(
        self, arguments: BaseModel, context: ToolExecutionContext,
    ) -> ToolResult:
        # BaseTool's signature takes any BaseModel; the loop hands this tool its
        # own input model, and anything else is validated into one.
        if not isinstance(arguments, WebDiscoverInput):
            arguments = WebDiscoverInput.model_validate(arguments.model_dump())
        key = resolve_exa_key()
        if not key:
            return ToolResult(
                output=(
                    f"web_discover failed: no {EXA_KEY_ENV} in the environment or the "
                    f"env file any more; it was there when the tool was registered."
                ),
                is_error=True,
            )

        if arguments.url and arguments.url.strip():
            url = arguments.url.strip()
            refusal = _refuse_url(url)
            if refusal:
                return ToolResult(output=f"web_discover: {refusal}", is_error=True)
            endpoint = "/findSimilar"
            body: dict[str, object] = {"url": url, "excludeSourceDomain": True}
            header = f"Similar to: {url} (Exa neural search)"
            mode = "similar"
        else:
            query = (arguments.query or "").strip()
            endpoint = "/search"
            body = {"query": query, "type": "auto"}
            header = f"Discovered for: {query} (Exa neural search)"
            mode = "query"
        body["numResults"] = arguments.num_results
        body["contents"] = {"summary": {"query": SUMMARY_QUERY}}
        if arguments.category:
            body["category"] = arguments.category

        try:
            async with httpx.AsyncClient(transport=self._transport) as client:
                # Wall clock as well as httpx's timeout, which is per read: a
                # trickling answer would otherwise run on until the agent loop's
                # tool timeout cancelled the call.
                response = await asyncio.wait_for(client.post(
                    EXA_BASE + endpoint, json=body, timeout=TIMEOUT,
                    headers={"x-api-key": key, "Content-Type": "application/json"},
                ), timeout=TIMEOUT)
        except asyncio.TimeoutError:
            return _failed(f"Exa took longer than {TIMEOUT:.3g}s")
        except httpx.TimeoutException:
            return _failed(f"Exa timed out after {TIMEOUT:.0f}s")
        except httpx.HTTPError as exc:
            return _failed(f"could not reach Exa ({type(exc).__name__})")

        if response.status_code != 200:
            return _failed(_explain_status(response))
        try:
            data = json.loads(response.text)
        except ValueError:
            return _failed("Exa answered something that is not JSON")
        results = [
            r for r in (data.get("results") if isinstance(data, dict) else None) or []
            if isinstance(r, dict) and r.get("url")
        ][: arguments.num_results]
        if not results:
            return ToolResult(output=f"{header}\nNo results.", is_error=True,
                              metadata={"mode": mode, "results": 0})

        lines = [header]
        for index, item in enumerate(results, start=1):
            lines.append(f"{index}. {str(item.get('title') or item['url']).strip()}")
            lines.append(f"   URL: {item['url']}")
            summary = _short(str(item.get("summary") or ""))
            if summary:
                lines.append(f"   {summary}")
        return ToolResult(output="\n".join(lines), metadata={"mode": mode, "results": len(results)})


def _failed(reason: str) -> ToolResult:
    return ToolResult(output=f"web_discover failed: {reason}", is_error=True)


def _explain_status(response: httpx.Response) -> str:
    status = response.status_code
    try:
        message = str(json.loads(response.text).get("error") or "")[:200]
    except (ValueError, AttributeError):
        message = ""
    detail = f": {message}" if message else ""
    if status in (401, 403):
        return f"Exa rejected the {EXA_KEY_ENV} (HTTP {status}{detail})"
    if status == 402:
        return f"Exa says the account is out of credits or over budget (HTTP 402{detail})"
    if status == 429:
        return f"rate limited by Exa (HTTP 429{detail})"
    return f"Exa answered HTTP {status}{detail}"


def _short(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= SUMMARY_CHARS:
        return text
    return text[: SUMMARY_CHARS - 1].rstrip() + "…"


def _refuse_url(url: str) -> str:
    """Why ``url`` must not go to Exa, or "" when it may."""
    from prometheus.security.url_guard import is_ip_literal_blocked

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return f"url must be an http(s) URL with a host, got {url!r}"
    host = parsed.hostname.lower()
    if (
        host == "localhost"
        or host.endswith((".localhost", ".local", ".internal", ".lan", ".home.arpa"))
        or "." not in host.strip("[]")
        or is_ip_literal_blocked(host, block_tailnet=True)
    ):
        return (
            "refusing to send a private or local address to Exa; find-similar "
            "only works for public pages anyway"
        )
    return ""
