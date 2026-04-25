"""GitHub Search — REST client + agent-facing tool.

Source: Novel code for Prometheus GRAFT-SYMBIOTE Sprint Session A.
License: MIT

Wraps a small subset of the GitHub REST API:
  • GET /search/repositories
  • GET /repos/{owner}/{repo}
  • GET /repos/{owner}/{repo}/readme
  • GET /repos/{owner}/{repo}/contents/{path}

Authentication is OPTIONAL. With no token, GitHub allows ~10 search
requests/min for the unauthenticated client (lower than the 30/min for
authenticated). The token, if provided, is read from prometheus.yaml
``symbiote.github_token`` or the ``PROMETHEUS_GITHUB_TOKEN`` environment
variable. The token is NEVER logged or echoed in tool outputs.

A simple token-bucket rate limiter caps requests in-process so a misbehaving
agent can't exhaust the GitHub quota in seconds.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from typing import Any

import httpx
from pydantic import BaseModel, Field

from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult

log = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"
_DEFAULT_TIMEOUT = 30.0


class _TokenBucket:
    """Sliding-window token-bucket rate limiter.

    Keeps a deque of timestamps for the last `window_seconds` seconds and
    blocks new requests once `max_requests` have been issued in that window.
    """

    def __init__(self, max_requests: int, window_seconds: float = 60.0) -> None:
        self._max = max(1, int(max_requests))
        self._window = float(window_seconds)
        self._timestamps: deque[float] = deque()

    def wait_token(self) -> float:
        """Synchronously wait for a token. Returns the time slept (seconds)."""
        now = time.monotonic()
        self._evict(now)
        if len(self._timestamps) < self._max:
            self._timestamps.append(now)
            return 0.0
        # Sleep until the oldest token leaves the window.
        oldest = self._timestamps[0]
        sleep_for = max(0.0, self._window - (now - oldest))
        if sleep_for > 0:
            time.sleep(sleep_for)
        now = time.monotonic()
        self._evict(now)
        self._timestamps.append(now)
        return sleep_for

    def try_token(self) -> bool:
        """Non-blocking: return True if a token was available, False if rate-limited."""
        now = time.monotonic()
        self._evict(now)
        if len(self._timestamps) < self._max:
            self._timestamps.append(now)
            return True
        return False

    def _evict(self, now: float) -> None:
        while self._timestamps and (now - self._timestamps[0]) >= self._window:
            self._timestamps.popleft()


# ---------------------------------------------------------------------------
# GitHub client (private, used by ScoutEngine and the tool wrapper)
# ---------------------------------------------------------------------------


class GitHubClient:
    """Minimal authenticated HTTP client for GitHub REST."""

    def __init__(
        self,
        token: str | None = None,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        rate_limiter: _TokenBucket | None = None,
    ) -> None:
        self._token = token
        self._timeout = timeout
        if rate_limiter is None:
            rate_limiter = _TokenBucket(
                max_requests=30 if token else 10,
                window_seconds=60.0,
            )
        self._rate = rate_limiter

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> GitHubClient:
        """Build from the ``symbiote`` section of prometheus.yaml.

        Token resolution order: env var ``PROMETHEUS_GITHUB_TOKEN`` →
        ``config['github_token']``. Both fall back to None (unauthenticated).
        """
        env_token = os.environ.get("PROMETHEUS_GITHUB_TOKEN")
        cfg_token = (config or {}).get("github_token") if config else None
        token = env_token or cfg_token or None
        return cls(token=token)

    @property
    def has_token(self) -> bool:
        return bool(self._token)

    @property
    def rate_remaining_estimate(self) -> int:
        """Best-effort estimate of remaining requests in the local bucket."""
        return self._rate._max - len(self._rate._timestamps)

    async def search_repositories(
        self,
        query: str,
        *,
        sort: str = "stars",
        order: str = "desc",
        per_page: int = 10,
    ) -> dict[str, Any]:
        """Run a /search/repositories query. Raises httpx.HTTPStatusError on 4xx/5xx."""
        params = {
            "q": query,
            "sort": sort,
            "order": order,
            "per_page": max(1, min(int(per_page), 100)),
        }
        return await self._get_json("/search/repositories", params=params)

    async def get_repo(self, full_name: str) -> dict[str, Any]:
        """GET /repos/{owner}/{repo}."""
        return await self._get_json(f"/repos/{full_name}")

    async def get_readme(self, full_name: str) -> str | None:
        """GET /repos/{owner}/{repo}/readme. Returns None if no README."""
        try:
            data = await self._get_json(
                f"/repos/{full_name}/readme",
                accept="application/vnd.github.raw+json",
                expect_json=False,
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise
        if isinstance(data, str):
            return data
        # Some GitHub responses still return JSON with base64 content even
        # under the raw accept header. Decode best-effort.
        if isinstance(data, dict) and data.get("content"):
            import base64
            try:
                return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
            except Exception:
                return None
        return None

    async def get_contents(self, full_name: str, path: str) -> Any:
        """GET /repos/{owner}/{repo}/contents/{path}."""
        return await self._get_json(f"/repos/{full_name}/contents/{path.lstrip('/')}")

    # ------------------------------------------------------------------

    async def _get_json(
        self,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        accept: str = "application/vnd.github+json",
        expect_json: bool = True,
    ) -> Any:
        slept = self._rate.wait_token()
        if slept:
            log.debug("GitHubClient: throttled %.2fs before %s", slept, endpoint)

        url = f"{GITHUB_API_BASE}{endpoint}"
        headers = {
            "Accept": accept,
            "User-Agent": "prometheus-symbiote/1.0",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(url, params=params, headers=headers)
        # Don't log full URL with the params if a token might appear there.
        # Token is sent only via header, so URL is safe to log.
        log.debug("GitHubClient: %s -> %s", endpoint, resp.status_code)
        resp.raise_for_status()
        if not expect_json:
            text = resp.text
            try:
                return json.loads(text) if text.lstrip().startswith(("{", "[")) else text
            except Exception:
                return text
        return resp.json()


# ---------------------------------------------------------------------------
# Agent-facing tool
# ---------------------------------------------------------------------------


class GitHubSearchInput(BaseModel):
    """Input for the github_search tool."""

    query: str = Field(..., description="Natural language or GitHub search syntax")
    language: str | None = Field("Python", description="Filter by language")
    min_stars: int = Field(10, ge=0, description="Minimum star count")
    max_results: int = Field(5, ge=1, le=10, description="Max results to return")
    sort: str = Field("stars", description="stars | updated | relevance")
    license_filter: list[str] | None = Field(
        None,
        description="Only allow these SPDX IDs (defaults to LicenseGate.ALLOW)",
    )


class GitHubSearchTool(BaseTool):
    """Search GitHub repositories for solutions to a capability need.

    Tool name: ``github_search``
    Trust level: 2 (read-only, no side effects)
    """

    name = "github_search"
    description = (
        "Search GitHub for open-source repositories matching a capability "
        "need. Returns a JSON array of candidates with name, description, "
        "stars, language, license, and a README excerpt."
    )
    input_model = GitHubSearchInput
    example_call = {
        "query": "yaml schema validation",
        "language": "Python",
        "min_stars": 50,
        "max_results": 5,
    }

    def __init__(self, client: GitHubClient | None = None) -> None:
        self._client = client or GitHubClient()

    def is_read_only(self, arguments: BaseModel) -> bool:  # noqa: D401 - simple override
        del arguments
        return True

    async def execute(
        self,
        arguments: BaseModel,
        context: ToolExecutionContext,
    ) -> ToolResult:
        del context
        args = arguments  # type: ignore[assignment]
        assert isinstance(args, GitHubSearchInput)
        try:
            candidates = await self.search(
                query=args.query,
                language=args.language,
                min_stars=args.min_stars,
                max_results=args.max_results,
                sort=args.sort,
                license_filter=args.license_filter,
            )
        except httpx.HTTPStatusError as exc:
            return ToolResult(
                output=f"GitHub API error: {exc.response.status_code}",
                is_error=True,
            )
        except Exception as exc:
            return ToolResult(output=f"GitHub search failed: {exc}", is_error=True)
        return ToolResult(output=json.dumps(candidates, ensure_ascii=False, indent=2))

    # ------------------------------------------------------------------
    # Public methods used by ScoutEngine
    # ------------------------------------------------------------------

    @property
    def client(self) -> GitHubClient:
        return self._client

    async def search(
        self,
        *,
        query: str,
        language: str | None = "Python",
        min_stars: int = 10,
        max_results: int = 5,
        sort: str = "stars",
        license_filter: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a search and enrich each result with a README excerpt."""
        gh_query = self._build_search_query(query, language, min_stars, license_filter)
        log.info("GitHub search: %s", gh_query)
        try:
            data = await self._client.search_repositories(
                gh_query,
                sort=sort if sort != "relevance" else "best-match",
                per_page=max(1, min(int(max_results), 10)),
            )
        except httpx.HTTPStatusError as exc:
            log.warning("GitHub search HTTP %s", exc.response.status_code)
            return []

        items = data.get("items", []) if isinstance(data, dict) else []
        enriched: list[dict[str, Any]] = []
        for item in items[:max_results]:
            enriched.append(self._summarize_repo(item))
        # Best-effort README enrichment (one round per result; rate-limited).
        for entry in enriched:
            full_name = entry.get("full_name")
            if not full_name:
                continue
            try:
                readme = await self._client.get_readme(full_name)
            except httpx.HTTPStatusError as exc:
                log.debug("README fetch HTTP %s for %s", exc.response.status_code, full_name)
                readme = None
            except Exception:
                readme = None
            if readme:
                entry["readme_excerpt"] = readme[:500]
        return enriched

    @staticmethod
    def _build_search_query(
        query: str,
        language: str | None,
        min_stars: int,
        license_filter: list[str] | None,
    ) -> str:
        """Turn user inputs into a GitHub search syntax string."""
        parts = [query.strip()] if query else []
        if language:
            parts.append(f"language:{language}")
        if min_stars and min_stars > 0:
            parts.append(f"stars:>={int(min_stars)}")
        if license_filter:
            for spdx in license_filter:
                parts.append(f"license:{spdx.lower()}")
        parts.append("fork:false")
        return " ".join(p for p in parts if p)

    @staticmethod
    def _summarize_repo(item: dict[str, Any]) -> dict[str, Any]:
        """Slice a repo response down to the fields ScoutEngine cares about."""
        license_obj = item.get("license") or {}
        return {
            "full_name": item.get("full_name"),
            "description": (item.get("description") or "")[:300],
            "url": item.get("html_url"),
            "stars": int(item.get("stargazers_count", 0)),
            "language": item.get("language"),
            "license": license_obj.get("spdx_id") if isinstance(license_obj, dict) else None,
            "license_obj": license_obj if isinstance(license_obj, dict) else None,
            "last_pushed": item.get("pushed_at"),
            "topics": list(item.get("topics") or [])[:10],
            "size_kb": int(item.get("size", 0)),
            "open_issues": int(item.get("open_issues_count", 0)),
            "default_branch": item.get("default_branch", "main"),
            "readme_excerpt": "",  # filled in by search()
        }
