"""web_search's backend chain: searxng -> brave -> duckduckgo, and DuckDuckGo hardening.

Every response comes from ``tests/fixtures/web_search/`` through an
``httpx.MockTransport``; nothing here touches the network. The README in that
directory says which bodies are recorded verbatim, which are derived from a
recording and which are constructed from an API's documented shape.

What is pinned:

* the order the backends are tried in, and that a backend which errors, times
  out, is rate-limited or answers with a CAPTCHA falls through to the next;
* duckduckgo is always the final backend, whatever the config says;
* a DuckDuckGo block (the 202 bot challenge, a CAPTCHA page served as 200, an
  empty page with no "no results" marker) is an explicit
  "web search is blocked right now" error, never "no results";
* SearXNG answering 403 or HTML says its JSON output is disabled;
* the result and telemetry name the backend that answered and the ones skipped;
* keys come from the environment or the env file, never from prometheus.yaml.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest

from prometheus.tools.base import ToolExecutionContext, ToolResult
from prometheus.tools.builtin import web_search_backends as wsb
from prometheus.tools.builtin.web_search import WebSearchInput, WebSearchTool

FIXTURES = Path(__file__).parent / "fixtures" / "web_search"

DDG_HTML = "html.duckduckgo.com"
DDG_LITE = "lite.duckduckgo.com"
BRAVE = "api.search.brave.com"
SEARX = "searx.test"
SEARX_URL = f"http://{SEARX}:8888"


def _fx(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _html(name: str, status: int = 200) -> Callable[[httpx.Request], httpx.Response]:
    body = _fx(name)
    return lambda request: httpx.Response(
        status, text=body, headers={"content-type": "text/html; charset=UTF-8"},
    )


def _json(name: str, status: int = 200) -> Callable[[httpx.Request], httpx.Response]:
    body = _fx(name)
    return lambda request: httpx.Response(
        status, text=body, headers={"content-type": "application/json"},
    )


def _status(status: int, text: str = "") -> Callable[[httpx.Request], httpx.Response]:
    return lambda request: httpx.Response(status, text=text)


def _timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("timed out", request=request)


def _refused(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


class FakeWeb:
    """Answers by host; records every request in order."""

    def __init__(self, **routes: Callable[[httpx.Request], httpx.Response]) -> None:
        self.routes = {
            {"ddg_html": DDG_HTML, "ddg_lite": DDG_LITE, "brave": BRAVE,
             "searx": SEARX}[k]: v
            for k, v in routes.items()
        }
        self.calls: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        route = self.routes.get(request.url.host)
        if route is None:
            raise AssertionError(f"unexpected request to {request.url}")
        return route(request)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)

    @property
    def hosts(self) -> list[str]:
        return [c.url.host for c in self.calls]


def _run(tool: WebSearchTool, query: str = "llama.cpp grammar", **meta: Any) -> ToolResult:
    ctx = ToolExecutionContext(cwd=Path.cwd(), metadata=meta)
    return asyncio.run(tool.execute(WebSearchInput(query=query), ctx))


def _tool(web: FakeWeb, **config: Any) -> WebSearchTool:
    return WebSearchTool(config=config or None, transport=web.transport)


@pytest.fixture
def brave_key(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv(wsb.BRAVE_KEY_ENV, "test-brave-key")
    return "test-brave-key"


# ---------------------------------------------------------------------------
# The chain: which backends are active, and in what order
# ---------------------------------------------------------------------------


class TestTheChain:
    def test_a_fresh_install_runs_duckduckgo_alone(self) -> None:
        plan = wsb.plan_backends(None)
        assert plan.active == ("duckduckgo",)
        assert set(plan.inactive) == {"searxng", "brave"}
        assert "searxng_url" in plan.inactive["searxng"]
        assert wsb.BRAVE_KEY_ENV in plan.inactive["brave"]

    def test_configured_backends_run_in_the_default_order(self, brave_key: str) -> None:
        plan = wsb.plan_backends({"searxng_url": SEARX_URL})
        assert plan.active == ("searxng", "brave", "duckduckgo")
        assert plan.inactive == {}

    def test_the_order_can_be_changed(self, brave_key: str) -> None:
        plan = wsb.plan_backends(
            {"searxng_url": SEARX_URL, "backends": ["brave", "searxng", "duckduckgo"]}
        )
        assert plan.active == ("brave", "searxng", "duckduckgo")

    def test_duckduckgo_left_out_is_appended_as_the_final_fallback(
        self, brave_key: str, caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING):
            plan = wsb.plan_backends({"searxng_url": SEARX_URL, "backends": ["brave", "searxng"]})
        assert plan.active == ("brave", "searxng", "duckduckgo")
        assert any("duckduckgo" in r.getMessage() for r in caplog.records)

    def test_duckduckgo_listed_first_still_runs_last(
        self, brave_key: str, caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING):
            plan = wsb.plan_backends(
                {"searxng_url": SEARX_URL, "backends": ["duckduckgo", "searxng", "brave"]}
            )
        assert plan.active == ("searxng", "brave", "duckduckgo")
        assert any("duckduckgo" in r.getMessage() for r in caplog.records)

    def test_an_empty_list_is_still_duckduckgo(self) -> None:
        assert wsb.plan_backends({"backends": []}).active == ("duckduckgo",)

    def test_an_unknown_backend_is_named_and_ignored(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING):
            plan = wsb.plan_backends({"backends": ["bing", "duckduckgo"]})
        assert plan.active == ("duckduckgo",)
        assert any("bing" in r.getMessage() for r in caplog.records)

    def test_the_brave_key_is_read_from_the_env_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        env_file = tmp_path / "env"
        env_file.write_text(f"{wsb.BRAVE_KEY_ENV}=from-the-env-file\n")
        monkeypatch.setenv("PROMETHEUS_ENV_FILE", str(env_file))
        monkeypatch.delenv(wsb.BRAVE_KEY_ENV, raising=False)
        assert wsb.resolve_brave_key() == "from-the-env-file"
        assert wsb.plan_backends(None).active == ("brave", "duckduckgo")

    def test_a_key_written_into_the_config_is_never_used(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING):
            plan = wsb.plan_backends(
                {"brave_api_key": "a-key-in-yaml", "brave": {"api_key": "a-key-in-yaml"}}
            )
        assert plan.active == ("duckduckgo",)
        assert "brave" in plan.inactive
        assert any(wsb.BRAVE_KEY_ENV in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Fallthrough
# ---------------------------------------------------------------------------


class TestFallthrough:
    def test_the_first_backend_that_answers_is_the_only_one_asked(self, brave_key: str) -> None:
        web = FakeWeb(searx=_json("searxng_results.json"))
        result = _run(_tool(web, searxng_url=SEARX_URL))
        assert not result.is_error, result.output
        assert web.hosts == [SEARX]
        assert result.metadata["backend"] == "searxng"
        assert result.metadata["skipped"] == []
        assert "(via searxng)" in result.output.splitlines()[0]
        assert "https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md" in result.output

    def test_timeout_then_rate_limit_then_duckduckgo(self, brave_key: str) -> None:
        web = FakeWeb(
            searx=_timeout,
            brave=_status(429, '{"type":"ErrorResponse"}'),
            ddg_html=_html("ddg_html_results.html"),
        )
        result = _run(_tool(web, searxng_url=SEARX_URL))
        assert not result.is_error, result.output
        assert web.hosts == [SEARX, BRAVE, DDG_HTML]
        assert result.metadata["backend"] == "duckduckgo"
        assert [(s["backend"], s["kind"]) for s in result.metadata["skipped"]] == [
            ("searxng", "timeout"), ("brave", "rate_limited"),
        ]
        lines = result.output.splitlines()
        assert "(via duckduckgo)" in lines[0]
        assert lines[1].startswith("Skipped:")
        assert "searxng" in lines[1] and "timed out" in lines[1]
        assert "brave" in lines[1] and "429" in lines[1]

    def test_an_error_falls_through(self, brave_key: str) -> None:
        web = FakeWeb(searx=_status(500, "boom"), brave=_json("brave_results.json"))
        result = _run(_tool(web, searxng_url=SEARX_URL))
        assert not result.is_error, result.output
        assert web.hosts == [SEARX, BRAVE]
        assert result.metadata["backend"] == "brave"
        assert result.metadata["skipped"][0]["kind"] == "error"
        assert "500" in result.metadata["skipped"][0]["reason"]

    def test_a_connection_failure_falls_through(self) -> None:
        web = FakeWeb(searx=_refused, ddg_html=_html("ddg_html_results.html"))
        result = _run(_tool(web, searxng_url=SEARX_URL))
        assert not result.is_error, result.output
        assert result.metadata["skipped"][0]["backend"] == "searxng"
        assert result.metadata["skipped"][0]["kind"] == "error"

    def test_a_searxng_captcha_falls_through(self) -> None:
        web = FakeWeb(searx=_html("searxng_html_bot_check.html"),
                      ddg_html=_html("ddg_html_results.html"))
        result = _run(_tool(web, searxng_url=SEARX_URL))
        assert not result.is_error, result.output
        assert web.hosts == [SEARX, DDG_HTML]
        assert result.metadata["skipped"][0]["kind"] == "json_disabled"

    def test_a_backend_with_no_results_lets_the_next_one_try(self) -> None:
        web = FakeWeb(searx=_json("searxng_empty_unresponsive.json"),
                      ddg_html=_html("ddg_html_results.html"))
        result = _run(_tool(web, searxng_url=SEARX_URL))
        assert not result.is_error, result.output
        skipped = result.metadata["skipped"][0]
        assert skipped["backend"] == "searxng"
        # SearXNG says which of its engines failed; that is the reason.
        assert "CAPTCHA" in skipped["reason"]

    def test_brave_sends_its_key_in_the_documented_header_only(self, brave_key: str) -> None:
        web = FakeWeb(brave=_json("brave_results.json"))
        result = _run(_tool(web))
        assert not result.is_error, result.output
        sent = web.calls[0]
        assert sent.url.path == "/res/v1/web/search"
        assert sent.headers["x-subscription-token"] == brave_key
        assert brave_key not in str(sent.url)
        assert brave_key not in result.output
        assert "<strong>" not in result.output  # Brave marks matches in HTML

    def test_brave_never_follows_a_redirect_with_the_key(self, brave_key: str) -> None:
        """httpx strips Authorization on a cross-origin redirect, but not
        X-Subscription-Token: following one would hand the key to that host."""
        web = FakeWeb(
            brave=lambda r: httpx.Response(302, headers={"location": "https://evil.test/x"}),
            ddg_html=_html("ddg_html_results.html"),
        )
        result = _run(_tool(web))
        assert "evil.test" not in web.hosts
        assert web.hosts == [BRAVE, DDG_HTML]
        assert result.metadata["skipped"][0]["backend"] == "brave"
        assert "302" in result.metadata["skipped"][0]["reason"]

    def test_a_rejected_brave_key_falls_through_without_echoing_it(self, brave_key: str) -> None:
        web = FakeWeb(brave=_status(401, '{"error": "invalid token"}'),
                      ddg_html=_html("ddg_html_results.html"))
        result = _run(_tool(web))
        assert not result.is_error, result.output
        assert result.metadata["skipped"][0]["kind"] == "error"
        assert brave_key not in result.output
        assert brave_key not in json.dumps(result.metadata)


# ---------------------------------------------------------------------------
# DuckDuckGo: html, then lite; a block is said, never shown as "no results"
# ---------------------------------------------------------------------------


class TestDuckDuckGo:
    def test_the_recorded_html_page_parses(self) -> None:
        web = FakeWeb(ddg_html=_html("ddg_html_results.html"))
        result = _run(_tool(web))
        assert not result.is_error, result.output
        assert web.hosts == [DDG_HTML]
        assert result.metadata["endpoint"] == "html"
        assert "1. llama.cpp/grammars/README.md at master" in result.output
        assert "URL: https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md" in result.output

    def test_a_blocked_html_endpoint_falls_back_to_lite(self) -> None:
        web = FakeWeb(ddg_html=_html("ddg_html_anomaly_202.html", 202),
                      ddg_lite=_html("ddg_lite_results.html"))
        result = _run(_tool(web))
        assert not result.is_error, result.output
        assert web.hosts == [DDG_HTML, DDG_LITE]
        assert result.metadata["endpoint"] == "lite"

    def test_the_recorded_lite_page_parses(self) -> None:
        """lite quotes its classes with single quotes; html with double."""
        hits = wsb.parse_ddg_results(_fx("ddg_lite_results.html"), limit=10)
        assert len(hits) == 10
        assert hits[0].url == "https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md"
        assert hits[0].title == "llama.cpp/grammars/README.md at master · ggml-org/llama.cpp"
        assert hits[0].snippet.startswith("LLM inference in C/C++.")

    def test_both_endpoints_challenged_is_an_explicit_block(self) -> None:
        web = FakeWeb(ddg_html=_html("ddg_html_anomaly_202.html", 202),
                      ddg_lite=_html("ddg_lite_anomaly_202.html", 202))
        result = _run(_tool(web))
        assert result.is_error
        assert result.output.startswith("web search is blocked right now:")
        assert "No search results" not in result.output
        assert "202" in result.output
        assert result.metadata["blocked"] is True

    def test_a_challenge_served_as_200_is_still_a_block(self) -> None:
        web = FakeWeb(ddg_html=_html("ddg_html_anomaly_202.html", 200),
                      ddg_lite=_html("ddg_lite_anomaly_202.html", 200))
        result = _run(_tool(web))
        assert result.is_error
        assert result.output.startswith("web search is blocked right now:")
        assert "CAPTCHA" in result.output

    def test_an_empty_page_with_no_marker_is_a_block(self) -> None:
        web = FakeWeb(ddg_html=_html("ddg_html_empty_unmarked.html"),
                      ddg_lite=_html("ddg_html_empty_unmarked.html"))
        result = _run(_tool(web))
        assert result.is_error
        assert result.output.startswith("web search is blocked right now:")
        assert "No search results" not in result.output

    @pytest.mark.parametrize("status", [403, 429])
    def test_refusal_statuses_are_blocks(self, status: int) -> None:
        web = FakeWeb(ddg_html=_status(status), ddg_lite=_status(status))
        result = _run(_tool(web))
        assert result.output.startswith("web search is blocked right now:")
        assert str(status) in result.output

    def test_a_genuine_no_results_page_says_no_results(self) -> None:
        web = FakeWeb(ddg_html=_html("ddg_html_no_results.html"))
        result = _run(_tool(web))
        assert result.is_error
        assert result.output.startswith("No search results found")
        assert "blocked" not in result.output
        assert web.hosts == [DDG_HTML]
        assert result.metadata["blocked"] is False

    def test_results_that_mention_the_challenge_are_still_results(self) -> None:
        """A search ABOUT DuckDuckGo's bot check returns snippets that name it."""
        page = _fx("ddg_html_results.html").replace(
            "LLM inference in C/C++.",
            "Unfortunately, bots use DuckDuckGo too: the anomaly-modal and "
            "challenge-form served from anomaly.js.", 1,
        )
        assert "bots use DuckDuckGo too" in page
        web = FakeWeb(ddg_html=lambda r: httpx.Response(200, text=page))
        result = _run(_tool(web), query="duckduckgo anomaly.js challenge-form")
        assert not result.is_error, result.output
        assert result.metadata["endpoint"] == "html"

    def test_a_query_about_anomalies_with_no_results_is_not_a_captcha(self) -> None:
        """The final URL carries the query; "anomaly" in it is not a challenge."""
        web = FakeWeb(ddg_html=_html("ddg_html_no_results.html"))
        result = _run(_tool(web), query="anomaly detection qzxv")
        assert web.calls[0].url.params["q"] == "anomaly detection qzxv"
        assert result.output.startswith("No search results found")
        assert result.metadata["blocked"] is False

    def test_a_dead_html_endpoint_falls_back_to_lite(self) -> None:
        web = FakeWeb(ddg_html=_refused, ddg_lite=_html("ddg_lite_results.html"))
        result = _run(_tool(web))
        assert not result.is_error, result.output
        assert result.metadata["endpoint"] == "lite"

    def test_no_network_at_all_is_a_failure_not_a_block(self) -> None:
        web = FakeWeb(ddg_html=_refused, ddg_lite=_refused)
        result = _run(_tool(web))
        assert result.is_error
        assert result.output.startswith("web search failed:")
        assert "No search results" not in result.output

    def test_a_block_after_skipped_backends_names_them(self) -> None:
        web = FakeWeb(searx=_timeout,
                      ddg_html=_html("ddg_html_anomaly_202.html", 202),
                      ddg_lite=_html("ddg_lite_anomaly_202.html", 202))
        result = _run(_tool(web, searxng_url=SEARX_URL))
        assert result.output.startswith("web search is blocked right now:")
        assert "searxng" in result.output and "timed out" in result.output


# ---------------------------------------------------------------------------
# SearXNG: JSON API, and the JSON-disabled gotcha
# ---------------------------------------------------------------------------


JSON_DISABLED = "SearXNG has JSON output disabled; enable formats: [html, json]"


class TestSearxng:
    def test_it_asks_for_json(self) -> None:
        web = FakeWeb(searx=_json("searxng_results.json"))
        _run(_tool(web, searxng_url=SEARX_URL + "/"))
        sent = web.calls[0]
        assert sent.url.path == "/search"
        assert sent.url.params["format"] == "json"
        assert sent.url.params["q"] == "llama.cpp grammar"

    def test_403_says_json_output_is_disabled(self) -> None:
        with pytest.raises(wsb.BackendFailure) as info:
            asyncio.run(_searx_only(_html("searxng_403.html", 403)))
        assert info.value.kind == "json_disabled"
        assert JSON_DISABLED in info.value.reason

    def test_html_instead_of_json_says_json_output_is_disabled(self) -> None:
        with pytest.raises(wsb.BackendFailure) as info:
            asyncio.run(_searx_only(_html("searxng_html_bot_check.html")))
        assert info.value.kind == "json_disabled"
        assert JSON_DISABLED in info.value.reason
        # The page's own title, so a bot check is recognisable as one.
        assert "Verifying your browser" in info.value.reason

    def test_the_message_reaches_the_model_when_everything_else_fails(self) -> None:
        web = FakeWeb(searx=_html("searxng_403.html", 403),
                      ddg_html=_refused, ddg_lite=_refused)
        result = _run(_tool(web, searxng_url=SEARX_URL))
        assert result.is_error
        assert JSON_DISABLED in result.output

    def test_results_parse(self) -> None:
        hits = asyncio.run(_searx_only(_json("searxng_results.json")))
        assert [h.url for h in hits][:2] == [
            "https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md",
            "https://til.simonwillison.net/llms/llama-cpp-python-grammars",
        ]
        assert hits[0].snippet.startswith("GBNF")


async def _searx_only(route: Callable[[httpx.Request], httpx.Response]) -> list[wsb.SearchHit]:
    web = FakeWeb(searx=route)
    async with httpx.AsyncClient(transport=web.transport) as client:
        return await wsb.search_searxng(client, SEARX_URL, "llama.cpp grammar", 5)


# ---------------------------------------------------------------------------
# Telemetry: one subsystem_runs row per call, naming the backend
# ---------------------------------------------------------------------------


@pytest.fixture
def telemetry(tmp_path: Path):
    from prometheus.telemetry.tracker import (
        ToolCallTelemetry,
        get_telemetry_handle,
        set_telemetry_handle,
    )

    tel = ToolCallTelemetry(tmp_path / "telemetry.db")
    previous = get_telemetry_handle()
    set_telemetry_handle(tel)
    yield tel
    set_telemetry_handle(previous)


def _rows(tel: Any) -> list[tuple[str, str | None, dict]]:
    rows = tel._conn.execute(
        "SELECT outcome, session_id, summary_json FROM subsystem_runs "
        "WHERE subsystem = ? AND operation = ? ORDER BY rowid",
        (wsb.WEB_SEARCH_SUBSYSTEM, wsb.WEB_SEARCH_OPERATION),
    ).fetchall()
    return [(r[0], r[1], json.loads(r[2])) for r in rows]


class TestTelemetry:
    def test_each_call_records_the_backend_that_answered(self, telemetry: Any) -> None:
        web = FakeWeb(ddg_html=_html("ddg_html_results.html"))
        _run(_tool(web), session_id="web", effective_session_id="beacon:abc", ephemeral=False)
        [(outcome, session, summary)] = _rows(telemetry)
        assert outcome == "success"
        assert session == "beacon:abc"
        assert summary["backend"] == "duckduckgo"
        assert summary["endpoint"] == "html"
        assert summary["results"] == 5
        assert summary["skipped"] == []
        assert summary["inactive"] == ["searxng", "brave"]

    def test_a_fallthrough_is_partial_and_names_what_was_skipped(
        self, telemetry: Any, brave_key: str,
    ) -> None:
        web = FakeWeb(searx=_timeout, brave=_json("brave_results.json"))
        _run(_tool(web, searxng_url=SEARX_URL))
        [(outcome, _session, summary)] = _rows(telemetry)
        assert outcome == "partial"
        assert summary["backend"] == "brave"
        assert summary["skipped"] == [{"backend": "searxng", "kind": "timeout"}]

    def test_a_block_is_a_failed_row(self, telemetry: Any) -> None:
        web = FakeWeb(ddg_html=_html("ddg_html_anomaly_202.html", 202),
                      ddg_lite=_html("ddg_lite_anomaly_202.html", 202))
        _run(_tool(web))
        [(outcome, _session, summary)] = _rows(telemetry)
        assert outcome == "failed"
        assert summary["backend"] is None
        assert summary["blocked"] is True

    def test_the_row_holds_no_query_and_no_url(self, telemetry: Any) -> None:
        web = FakeWeb(searx=_timeout, ddg_html=_html("ddg_html_results.html"))
        _run(_tool(web, searxng_url=SEARX_URL), query="a private question")
        [(_o, _s, summary)] = _rows(telemetry)
        blob = json.dumps(summary)
        assert "private question" not in blob
        assert SEARX not in blob

    def test_an_ephemeral_turn_records_no_session(self, telemetry: Any) -> None:
        web = FakeWeb(ddg_html=_html("ddg_html_results.html"))
        _run(_tool(web), session_id="web", effective_session_id="beacon:abc", ephemeral=True)
        [(_o, session, _s)] = _rows(telemetry)
        assert session is None


# ---------------------------------------------------------------------------
# The advertised schema: a fresh install is byte-identical to the goldens
# ---------------------------------------------------------------------------


GOLDENS = Path(__file__).parent / "fixtures" / "parity"


def _golden_web_search_entries() -> list[dict]:
    entries: list[dict] = []
    for trace in sorted(GOLDENS.glob("*.trace.json")):
        for exchange in json.loads(trace.read_text())["exchanges"]:
            for tool in (exchange.get("request") or {}).get("tools") or []:
                if (tool.get("function") or tool).get("name") == "web_search":
                    entries.append(tool)
    return entries


class TestTheAdvertisedSchema:
    def test_the_goldens_advertise_web_search(self) -> None:
        assert len(_golden_web_search_entries()) >= 10

    @pytest.mark.parametrize("make", [
        pytest.param(lambda: WebSearchTool(), id="no-config"),
        pytest.param(lambda: WebSearchTool(config={}), id="empty-section"),
        pytest.param(
            lambda: WebSearchTool(config={"backends": ["searxng", "brave", "duckduckgo"],
                                          "searxng_url": ""}),
            id="shipped-template-section",
        ),
        pytest.param(lambda: _registry_tool(), id="create_tool_registry"),
    ])
    def test_a_fresh_install_advertises_exactly_what_the_goldens_recorded(self, make) -> None:
        tool = make()
        forms = (tool.to_openai_schema(), tool.to_api_schema())
        for entry in _golden_web_search_entries():
            assert entry in forms, (
                "web_search's advertised schema changed for a fresh install; every "
                "golden that advertises it would move"
            )

    def test_the_shipped_template_section_is_what_the_test_above_uses(self) -> None:
        import yaml

        template = yaml.safe_load(
            (Path(__file__).parents[1] / "config" / "prometheus.yaml.default").read_text()
        )
        assert template["web_search"] == {
            "backends": ["searxng", "brave", "duckduckgo"], "searxng_url": "",
        }

    def test_a_configured_chain_is_described_and_the_parameters_do_not_change(
        self, brave_key: str,
    ) -> None:
        default = WebSearchTool()
        configured = WebSearchTool(config={"searxng_url": SEARX_URL})
        assert configured.description.startswith(
            "Search the web via SearXNG (falling back to Brave Search, then DuckDuckGo)"
        )
        assert configured.to_openai_schema()["function"]["parameters"] == (
            default.to_openai_schema()["function"]["parameters"]
        )
        # The class attribute stays the fresh-install text.
        golden = (_golden_web_search_entries()[0].get("function") or {}).get("description")
        assert WebSearchTool.description == golden

    def test_the_ladder_fixture_advertises_the_fresh_install_schema(self, brave_key: str) -> None:
        from prometheus.gym.ladder.fixtures import FixtureWeb, fixture_web_tools

        _fetch, search = fixture_web_tools(FixtureWeb([]))
        assert search.to_openai_schema() == _golden_web_search_entries()[0]

    def test_the_registry_passes_the_config_section_to_the_tool(self) -> None:
        from prometheus.__main__ import create_tool_registry

        tool = create_tool_registry({}, web_search_cfg={"searxng_url": SEARX_URL}).get("web_search")
        assert tool.plan.active == ("searxng", "duckduckgo")

    def test_the_daemon_registry_passes_it_too(self) -> None:
        from prometheus.daemon import build_tool_registry

        tool = build_tool_registry({}, web_search_cfg={"searxng_url": SEARX_URL}).get("web_search")
        assert tool.plan.active == ("searxng", "duckduckgo")


def _registry_tool() -> WebSearchTool:
    from prometheus.__main__ import create_tool_registry

    tool = create_tool_registry({}).get("web_search")
    assert isinstance(tool, WebSearchTool)
    return tool


def test_the_daemon_and_the_cli_hand_their_config_section_over() -> None:
    """Pinned on the source: run_daemon and the CLI's main are not callable in a test."""
    import inspect

    import prometheus.__main__ as cli
    import prometheus.daemon as daemon

    assert 'web_search_cfg=config.get("web_search")' in inspect.getsource(daemon)
    assert 'web_search_cfg=config.get("web_search")' in inspect.getsource(cli)
