"""The backends behind ``web_search``: SearXNG, Brave Search and DuckDuckGo.

THE CHAIN
---------
``web_search.backends`` orders them (default ``[searxng, brave, duckduckgo]``).
A backend is ACTIVE when it can run: SearXNG needs ``web_search.searxng_url``,
Brave needs ``BRAVE_API_KEY`` in the environment or the env file, DuckDuckGo
needs nothing. The tool tries the active ones in order. A backend that errors,
times out, is rate-limited, answers with a CAPTCHA, or finds nothing falls
through to the next, and the result names each one it skipped and why.

DuckDuckGo is always last. A fresh install with no config runs it alone, which
is exactly what web_search did before there was a chain, so the tool's schema
and behaviour on a fresh install are unchanged. A config that leaves it out or
puts it earlier gets it moved to the end, with a WARNING: the fallback that
needs no setup is the one thing the chain must never lose.

KEYS
----
Read from the environment, then the env file (``config/env_file.py``), the
same order the provider registry uses. Never from prometheus.yaml: a key
written there is ignored and the WARNING says where it belongs.

A BLOCK IS NOT AN EMPTY RESULT
------------------------------
DuckDuckGo answers a client it suspects with a 202 and a CAPTCHA page
("Unfortunately, bots use DuckDuckGo too."), and parsing that page finds no
results. Reported as "No search results found", a model reads it as "nothing
exists" and moves on. Here a challenge, a 202/403/429, or a 200 with no
results and no "no results" marker is a BLOCK, raised as such; only DuckDuckGo's
own empty-result page counts as an empty result.
"""

from __future__ import annotations

import asyncio
import html as html_mod
import inspect
import json
import logging
import os
import re
import time
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar
from urllib.parse import parse_qs, unquote, urlparse

import httpx

log = logging.getLogger(__name__)

T = TypeVar("T")

SEARXNG = "searxng"
BRAVE = "brave"
DUCKDUCKGO = "duckduckgo"
KNOWN_BACKENDS: tuple[str, ...] = (SEARXNG, BRAVE, DUCKDUCKGO)
DEFAULT_ORDER: tuple[str, ...] = (SEARXNG, BRAVE, DUCKDUCKGO)
#: The fallback that needs no setup. Always active, always last.
FINAL_BACKEND = DUCKDUCKGO

DISPLAY_NAMES: dict[str, str] = {
    SEARXNG: "SearXNG",
    BRAVE: "Brave Search",
    DUCKDUCKGO: "DuckDuckGo",
}

BRAVE_KEY_ENV = "BRAVE_API_KEY"
BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
DDG_HTML_ENDPOINT = "https://html.duckduckgo.com/html/"
DDG_LITE_ENDPOINT = "https://lite.duckduckgo.com/lite/"
USER_AGENT = "Prometheus/0.1"

#: httpx timeouts per request. DuckDuckGo keeps the 20 s web_search always
#: had; the others are shorter because a slow one only delays the next backend.
#: ⚠ These are PER PHASE (connect, write, pool) and PER READ, not totals: a
#: server that trickles bytes, or a chain of redirects (each hop gets fresh
#: timeouts), never trips them. The wall-clock limits below are what bound a call.
SEARXNG_TIMEOUT = 15.0
BRAVE_TIMEOUT = 15.0
DDG_TIMEOUT = 20.0

#: Wall-clock budget for one web_search call, every backend included. Well
#: inside the agent loop's tool timeout (LoopContext.tool_timeout_seconds, 300 s,
#: engine/agent_loop.py) and the ladder's cap (TOOL_TIMEOUT_CAP_S, 120 s,
#: gym/ladder/runner.py), so the chain always ends inside the tool, with its own
#: message and its telemetry row, instead of being cancelled by the loop.
CHAIN_BUDGET_SECONDS = 60.0
#: Wall-clock limit per attempt; DuckDuckGo's is per endpoint (html, then lite).
ATTEMPT_LIMITS: dict[str, float] = {SEARXNG: 15.0, BRAVE: 15.0, DUCKDUCKGO: 20.0}
#: What the backends before DuckDuckGo must leave of the budget, so the
#: fallback that needs no setup always gets its turn: html in full, and lite.
FINAL_RESERVE_SECONDS = 30.0

#: One subsystem_runs row per call (no new table: the parity harness dumps
#: every table, so a new one would move every golden).
WEB_SEARCH_SUBSYSTEM = "web_search"
WEB_SEARCH_OPERATION = "search"

# Failure kinds. Each names why a backend was skipped.
KIND_ERROR = "error"
KIND_TIMEOUT = "timeout"
KIND_RATE_LIMITED = "rate_limited"
KIND_BLOCKED = "blocked"
KIND_JSON_DISABLED = "json_disabled"
KIND_NO_RESULTS = "no_results"

SEARXNG_JSON_DISABLED = (
    "SearXNG has JSON output disabled; enable formats: [html, json] "
    "(search.formats in its settings.yml)"
)


@dataclass(frozen=True)
class SearchHit:
    title: str
    url: str
    snippet: str = ""


class BackendFailure(Exception):
    """A backend could not answer. ``kind`` is one of the KIND_* names."""

    def __init__(self, kind: str, reason: str) -> None:
        super().__init__(f"{kind}: {reason}")
        self.kind = kind
        self.reason = reason


def attempt_limit(name: str, deadline: float, *, now: float | None = None) -> float:
    """Seconds the next attempt at ``name`` may take: its own limit, cut to what
    is left before ``deadline`` (a ``time.monotonic()`` value) and, for every
    backend but the last, to what is left after DuckDuckGo's reserve."""
    remaining = deadline - (time.monotonic() if now is None else now)
    if name != FINAL_BACKEND:
        remaining -= FINAL_RESERVE_SECONDS
    return max(0.0, min(ATTEMPT_LIMITS[name], remaining))


async def bounded(awaitable: Awaitable[T], seconds: float) -> T:
    """``awaitable``, or BackendFailure(KIND_TIMEOUT) once ``seconds`` of wall
    clock have passed, whatever the transport is doing."""
    if seconds <= 0:
        if inspect.iscoroutine(awaitable):
            awaitable.close()
        raise BackendFailure(KIND_TIMEOUT, "no time left in the search's budget")
    try:
        return await asyncio.wait_for(awaitable, timeout=seconds)
    except asyncio.TimeoutError as exc:
        raise BackendFailure(KIND_TIMEOUT, f"took longer than {seconds:.3g}s") from exc


# ---------------------------------------------------------------------------
# Which backends run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendPlan:
    """The active backends in the order they are tried, and why the rest are not."""

    active: tuple[str, ...]
    inactive: dict[str, str] = field(default_factory=dict)
    searxng_url: str = ""

    @property
    def is_fresh_install(self) -> bool:
        return self.active == (FINAL_BACKEND,)


def resolve_brave_key() -> str:
    """The Brave key from the environment, then the env file; "" when neither has it."""
    return _secret(BRAVE_KEY_ENV)


def _secret(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    from prometheus.config.env_file import parse_env_file

    try:
        return parse_env_file().get(name, "").strip()
    except OSError as exc:
        log.warning("web_search: cannot read the env file for %s (%s)", name, exc)
        return ""


def plan_backends(cfg: Mapping[str, Any] | None) -> BackendPlan:
    """Resolve ``web_search`` config (the section, or None) into a :class:`BackendPlan`."""
    cfg = cfg if isinstance(cfg, Mapping) else {}
    raw = cfg.get("backends", DEFAULT_ORDER)
    if raw is None:
        raw = DEFAULT_ORDER
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        log.warning(
            "web_search.backends must be a list, got %r; using the default order %s",
            raw, list(DEFAULT_ORDER),
        )
        raw = DEFAULT_ORDER

    order: list[str] = []
    for item in raw:
        name = str(item).strip().lower()
        if name not in KNOWN_BACKENDS:
            log.warning(
                "web_search.backends: unknown backend %r ignored (known: %s)",
                item, ", ".join(KNOWN_BACKENDS),
            )
            continue
        if name not in order:
            order.append(name)

    if FINAL_BACKEND not in order:
        if raw:
            log.warning(
                "web_search.backends leaves out %s; it is always the final fallback "
                "and has been appended", FINAL_BACKEND,
            )
        order.append(FINAL_BACKEND)
    elif order[-1] != FINAL_BACKEND:
        log.warning(
            "web_search.backends lists %s before %s; it always runs last, as the "
            "fallback that needs no setup",
            FINAL_BACKEND, ", ".join(order[order.index(FINAL_BACKEND) + 1:]),
        )
        order.remove(FINAL_BACKEND)
        order.append(FINAL_BACKEND)

    for misplaced in ("brave_api_key", "api_key"):
        if cfg.get(misplaced) or (
            isinstance(cfg.get(BRAVE), Mapping) and cfg[BRAVE].get(misplaced)
        ):
            log.warning(
                "web_search: a Brave key in prometheus.yaml is ignored; put "
                "%s=<key> in the env file (~/.config/prometheus/env) instead",
                BRAVE_KEY_ENV,
            )
            break

    searxng_url = str(cfg.get("searxng_url") or "").strip()
    active: list[str] = []
    inactive: dict[str, str] = {}
    for name in order:
        if name == SEARXNG and not searxng_url:
            inactive[name] = "web_search.searxng_url is not set"
        elif name == BRAVE and not resolve_brave_key():
            inactive[name] = f"no {BRAVE_KEY_ENV} in the environment or the env file"
        else:
            active.append(name)
    return BackendPlan(active=tuple(active), inactive=inactive, searxng_url=searxng_url)


# ---------------------------------------------------------------------------
# SearXNG — its JSON API
# ---------------------------------------------------------------------------


async def search_searxng(
    client: httpx.AsyncClient, base_url: str, query: str, limit: int,
) -> list[SearchHit]:
    data = await fetch_searxng_json(client, base_url, query)
    hits = [
        SearchHit(
            title=_clean_text(str(item.get("title") or "")),
            url=str(item.get("url") or ""),
            snippet=_clean_text(str(item.get("content") or "")),
        )
        for item in data.get("results") or []
        if isinstance(item, dict) and item.get("url")
    ][:limit]
    if not hits:
        broken = [
            f"{e[0]}: {e[1]}" if isinstance(e, (list, tuple)) and len(e) > 1 else str(e)
            for e in data.get("unresponsive_engines") or []
        ]
        if broken:
            raise BackendFailure(
                KIND_NO_RESULTS,
                "no results; unresponsive engines: " + ", ".join(broken),
            )
        raise BackendFailure(KIND_NO_RESULTS, "no results")
    return hits


async def fetch_searxng_json(
    client: httpx.AsyncClient, base_url: str, query: str, *,
    timeout: float = SEARXNG_TIMEOUT,
) -> dict[str, Any]:
    """SearXNG's JSON answer for ``query``, or BackendFailure saying why not.

    A 403 or an answer that is not JSON is KIND_JSON_DISABLED: that is what a
    stock instance (``search.formats`` without json) and a public instance with
    a browser check in front of its API both return. ``oara search setup``
    waits on this same function, so the two cannot disagree about "answers JSON".
    """
    url = base_url.rstrip("/") + "/search"
    response = await _get(
        client, url, SEARXNG,
        params={"q": query, "format": "json"},
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
        timeout=timeout,
    )
    if response.status_code == 403:
        raise BackendFailure(KIND_JSON_DISABLED, f"{SEARXNG_JSON_DISABLED} (it answered HTTP 403)")
    if response.status_code == 429:
        raise BackendFailure(KIND_RATE_LIMITED, "rate limited (HTTP 429; is its limiter on?)")
    if response.status_code >= 400:
        raise BackendFailure(KIND_ERROR, f"HTTP {response.status_code}")
    try:
        data = json.loads(response.text)
    except ValueError:
        data = None
    if not isinstance(data, dict):
        title = _page_title(response.text)
        shown = f" ({title!r})" if title else ""
        raise BackendFailure(
            KIND_JSON_DISABLED,
            f"{SEARXNG_JSON_DISABLED} (it answered "
            f"{response.headers.get('content-type', 'something').split(';')[0]}"
            f"{shown}, not JSON)",
        )
    return data


# ---------------------------------------------------------------------------
# Brave Search API
# ---------------------------------------------------------------------------


async def search_brave(
    client: httpx.AsyncClient, key: str, query: str, limit: int,
) -> list[SearchHit]:
    response = await _get(
        client, BRAVE_ENDPOINT, BRAVE,
        params={"q": query, "count": max(1, min(limit, 20))},
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": key,
            "User-Agent": USER_AGENT,
        },
        timeout=BRAVE_TIMEOUT,
        # httpx strips Authorization on a cross-origin redirect but not this
        # header, so a followed redirect would hand the key to wherever it led.
        follow_redirects=False,
    )
    if 300 <= response.status_code < 400:
        raise BackendFailure(
            KIND_ERROR,
            f"redirected (HTTP {response.status_code}); not followed, so the key "
            f"goes nowhere but Brave",
        )
    if response.status_code == 429:
        raise BackendFailure(KIND_RATE_LIMITED, "rate limited (HTTP 429)")
    if response.status_code in (401, 403):
        raise BackendFailure(
            KIND_ERROR,
            f"the {BRAVE_KEY_ENV} was rejected (HTTP {response.status_code})",
        )
    if response.status_code >= 400:
        raise BackendFailure(KIND_ERROR, f"HTTP {response.status_code}")
    try:
        data = json.loads(response.text)
    except ValueError as exc:
        raise BackendFailure(KIND_ERROR, "answered something that is not JSON") from exc
    web = data.get("web") if isinstance(data, dict) else None
    results = web.get("results") if isinstance(web, dict) else None
    hits = [
        SearchHit(
            title=_clean_text(str(item.get("title") or "")),
            url=str(item.get("url") or ""),
            snippet=_clean_text(str(item.get("description") or "")),
        )
        for item in results or []
        if isinstance(item, dict) and item.get("url")
    ][:limit]
    if not hits:
        raise BackendFailure(KIND_NO_RESULTS, "no results")
    return hits


# ---------------------------------------------------------------------------
# DuckDuckGo — html endpoint, then lite
# ---------------------------------------------------------------------------

#: What DuckDuckGo's bot challenge carries, in both endpoints' versions of it.
_DDG_CHALLENGE_MARKERS = (
    "anomaly-modal",
    "anomaly.js",
    "challenge-form",
    "bots use duckduckgo too",
)
#: DuckDuckGo's own empty-result markers (html: "No  results." with two
#: spaces; lite: "No more results.").
_DDG_EMPTY_MARKERS = ("no  results.", "no more results.", 'class="no-results"')
#: Statuses DuckDuckGo uses to turn a client away.
_DDG_REFUSAL_STATUSES = (202, 403, 429)


@dataclass(frozen=True)
class DdgAnswer:
    endpoint: str          # "html" or "lite"
    hits: list[SearchHit]


async def search_duckduckgo(
    client: httpx.AsyncClient, query: str, limit: int, *, deadline: float | None = None,
) -> DdgAnswer:
    """html, then lite. Returns hits, or an empty list on DuckDuckGo's own
    no-results page. Raises KIND_BLOCKED when it turned us away, KIND_ERROR
    (or KIND_TIMEOUT) when neither endpoint could be reached. With a
    ``deadline`` (``time.monotonic()``), each endpoint gets at most
    ``ATTEMPT_LIMITS[duckduckgo]`` of what is left."""
    failures: list[BackendFailure] = []
    for endpoint, url in (("html", DDG_HTML_ENDPOINT), ("lite", DDG_LITE_ENDPOINT)):
        seconds = (
            attempt_limit(DUCKDUCKGO, deadline) if deadline is not None
            else ATTEMPT_LIMITS[DUCKDUCKGO]
        )
        try:
            response = await bounded(_get(
                client, url, DUCKDUCKGO,
                params={"q": query},
                headers={"User-Agent": USER_AGENT},
                timeout=DDG_TIMEOUT,
            ), seconds)
            hits = classify_ddg_page(
                response.status_code, response.text, str(response.url), limit=limit,
            )
        except BackendFailure as exc:
            failures.append(BackendFailure(exc.kind, f"{endpoint}: {exc.reason}"))
            continue
        return DdgAnswer(endpoint=endpoint, hits=hits)

    reason = "; ".join(f.reason for f in failures)
    if any(f.kind == KIND_BLOCKED for f in failures):
        raise BackendFailure(KIND_BLOCKED, reason)
    if failures and all(f.kind == KIND_TIMEOUT for f in failures):
        raise BackendFailure(KIND_TIMEOUT, reason)
    raise BackendFailure(KIND_ERROR, reason)


def classify_ddg_page(status: int, body: str, final_url: str, *, limit: int) -> list[SearchHit]:
    """Results, [] for DuckDuckGo's own empty page, or BackendFailure(KIND_BLOCKED).

    Results are parsed BEFORE looking for the challenge: a search about
    DuckDuckGo's bot check returns snippets that name its markers, and the
    challenge page has no result links. Only the URL's PATH is checked for
    "anomaly"; the query string carries the user's words.
    """
    lowered = body.lower()
    if status in _DDG_REFUSAL_STATUSES:
        what = "bot challenge (CAPTCHA)" if _is_challenge(lowered, final_url) else "turned away"
        raise BackendFailure(KIND_BLOCKED, f"{what}, HTTP {status}")
    if status >= 400:
        raise BackendFailure(KIND_ERROR, f"HTTP {status}")
    hits = parse_ddg_results(body, limit=limit)
    if hits:
        return hits
    if _is_challenge(lowered, final_url):
        raise BackendFailure(KIND_BLOCKED, f"bot challenge (CAPTCHA) served as HTTP {status}")
    if any(m in lowered for m in _DDG_EMPTY_MARKERS):
        return []
    raise BackendFailure(
        KIND_BLOCKED,
        f"HTTP {status} with no results and no 'no results' marker (a soft block)",
    )


def _is_challenge(lowered_body: str, final_url: str) -> bool:
    return "anomaly" in urlparse(final_url).path.lower() or any(
        m in lowered_body for m in _DDG_CHALLENGE_MARKERS
    )


_ATTR = r"""(?P<q>["'])(?P<v>.*?)(?P=q)"""


def parse_ddg_results(body: str, *, limit: int) -> list[SearchHit]:
    """Results from either endpoint. html quotes attributes with ", lite with '."""
    snippets = [
        _clean_html(m.group("snippet"))
        for m in re.finditer(
            r"<(?:a|div|span|td)[^>]+class=[\"'][^\"']*(?:result__snippet|result-snippet)"
            r"[^\"']*[\"'][^>]*>(?P<snippet>.*?)</(?:a|div|span|td)>",
            body,
            flags=re.IGNORECASE | re.DOTALL,
        )
    ]

    hits: list[SearchHit] = []
    index = 0
    for match in re.finditer(
        r"<a(?P<attrs>[^>]+)>(?P<title>.*?)</a>", body, flags=re.IGNORECASE | re.DOTALL,
    ):
        attrs = match.group("attrs")
        cls = re.search(r"class=" + _ATTR, attrs, re.IGNORECASE)
        if cls is None:
            continue
        names = cls.group("v")
        if "result__a" not in names and "result-link" not in names:
            continue
        snippet = snippets[index] if index < len(snippets) else ""
        index += 1
        href = re.search(r"href=" + _ATTR, attrs, re.IGNORECASE)
        if href is None:
            continue
        title = _clean_html(match.group("title"))
        url = _normalize_result_url(html_mod.unescape(href.group("v")))
        if not title or not url or _is_duckduckgo_url(url):
            continue  # an ad (duckduckgo.com/y.js) or a link with nothing in it
        hits.append(SearchHit(title=title, url=url, snippet=snippet))
        if len(hits) >= limit:
            break
    return hits


def _normalize_result_url(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(target) if target else raw_url
    return raw_url


def _is_duckduckgo_url(url: str) -> bool:
    host = urlparse(url if "//" in url else "//" + url).netloc.lower()
    return host == "duckduckgo.com" or host.endswith(".duckduckgo.com")


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


async def _get(
    client: httpx.AsyncClient, url: str, backend: str, *,
    params: dict[str, Any], headers: dict[str, str], timeout: float,
    follow_redirects: bool = True,
) -> httpx.Response:
    """GET, turning transport failures into BackendFailure. Never names the
    request's headers, which is where a key travels."""
    try:
        return await client.get(
            url, params=params, headers=headers, timeout=timeout,
            follow_redirects=follow_redirects,
        )
    except httpx.TimeoutException as exc:
        raise BackendFailure(KIND_TIMEOUT, f"timed out after {timeout:.0f}s") from exc
    except httpx.HTTPError as exc:
        raise BackendFailure(
            KIND_ERROR, f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__,
        ) from exc


def _page_title(body: str) -> str:
    m = re.search(r"<title[^>]*>(?P<t>.*?)</title>", body or "", re.IGNORECASE | re.DOTALL)
    return _clean_html(m.group("t"))[:80] if m else ""


def _clean_html(fragment: str) -> str:
    text = re.sub(r"(?s)<[^>]+>", " ", fragment)
    text = html_mod.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _clean_text(text: str) -> str:
    """API text that may still carry markup (Brave marks matches with <strong>)."""
    return _clean_html(text)
