"""web_discover: Exa neural search, a tool of its own, present only with a key.

Every Exa answer comes from ``tests/fixtures/web_discover/`` through an
``httpx.MockTransport``; nothing here calls Exa. (Those fixtures are built from
Exa's published spec, not recorded: see the README there.)

What is pinned:

* with no EXA_API_KEY the tool does not exist: not registered, not advertised,
  and the advertised set is exactly the shipped default;
* with a key it is registered and advertised when the config follows the
  shipped default, and left deferred when the operator pinned their own list;
* the key comes from the environment or the env file, travels only in the
  ``x-api-key`` header, and is never echoed;
* two modes: a query (``/search``) and find-similar for a URL (``/findSimilar``);
  exactly one of them per call;
* results are title, url and a short summary, and the count is capped;
* it is a read-only tool that goes through the normal gate.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest
from pydantic import ValidationError

from prometheus.config.shipped_defaults import SHIPPED_ALWAYS_LOADED
from prometheus.tools.base import ToolExecutionContext, ToolResult
from prometheus.tools.builtin import web_discover as wd
from prometheus.tools.builtin.web_discover import WebDiscoverInput, WebDiscoverTool

FIXTURES = Path(__file__).parent / "fixtures" / "web_discover"
KEY = "test-exa-key"


def _fx(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class FakeExa:
    def __init__(self, answer: Callable[[httpx.Request], httpx.Response]) -> None:
        self.answer = answer
        self.calls: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        assert request.url.host == "api.exa.ai", f"unexpected request to {request.url}"
        return self.answer(request)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)

    def body(self, n: int = 0) -> dict[str, Any]:
        return json.loads(self.calls[n].content)


def _ok(name: str) -> Callable[[httpx.Request], httpx.Response]:
    text = _fx(name)
    return lambda request: httpx.Response(200, text=text, headers={"content-type": "application/json"})


def _status(status: int, name: str | None = None) -> Callable[[httpx.Request], httpx.Response]:
    text = _fx(name) if name else ""
    return lambda request: httpx.Response(status, text=text, headers={"content-type": "application/json"})


def _run(tool: WebDiscoverTool, **arguments: Any) -> ToolResult:
    ctx = ToolExecutionContext(cwd=Path.cwd())
    return asyncio.run(tool.execute(WebDiscoverInput(**arguments), ctx))


@pytest.fixture
def exa_key(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv(wd.EXA_KEY_ENV, KEY)
    return KEY


# ---------------------------------------------------------------------------
# No key: the tool does not exist
# ---------------------------------------------------------------------------


class TestWithoutAKey:
    def test_it_is_not_registered(self) -> None:
        from tests.support.advertisement import registered_names

        assert "web_discover" not in registered_names()

    def test_the_advertised_set_is_exactly_the_shipped_default(self) -> None:
        from tests.support.advertisement import advertised_names

        assert advertised_names() == set(SHIPPED_ALWAYS_LOADED)

    def test_with_deferral_off_the_full_catalog_has_no_web_discover(self) -> None:
        from prometheus.__main__ import create_tool_registry
        from prometheus.context.dynamic_tools import DynamicToolLoader

        loader = DynamicToolLoader(create_tool_registry({}), {"enabled": False})
        assert "web_discover" not in {s["name"] for s in loader.schemas_for_run(False)}

    def test_a_key_in_the_config_does_not_create_it(self) -> None:
        from prometheus.__main__ import create_tool_registry

        registry = create_tool_registry(
            {}, web_search_cfg={"exa_api_key": KEY, "web_discover": {"api_key": KEY}},
        )
        assert registry.get("web_discover") is None

    def test_a_key_that_vanished_after_boot_is_said(self, exa_key: str,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
        tool = WebDiscoverTool(transport=FakeExa(_ok("exa_search_results.json")).transport)
        monkeypatch.delenv(wd.EXA_KEY_ENV)
        result = _run(tool, query="vector databases")
        assert result.is_error
        assert wd.EXA_KEY_ENV in result.output


# ---------------------------------------------------------------------------
# With a key: registered and advertised
# ---------------------------------------------------------------------------


class TestWithAKey:
    def test_a_key_in_the_environment_registers_it(self, exa_key: str) -> None:
        from tests.support.advertisement import registered_names

        assert "web_discover" in registered_names()

    def test_a_key_in_the_env_file_registers_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from tests.support.advertisement import registered_names

        env_file = tmp_path / "env"
        env_file.write_text(f"{wd.EXA_KEY_ENV}={KEY}\n")
        monkeypatch.setenv("PROMETHEUS_ENV_FILE", str(env_file))
        assert "web_discover" in registered_names()

    def test_it_is_advertised_under_the_shipped_default(self, exa_key: str) -> None:
        from tests.support.advertisement import advertised_names

        assert advertised_names() == set(SHIPPED_ALWAYS_LOADED) | {"web_discover"}

    def test_it_is_advertised_when_always_loaded_is_absent(self, exa_key: str) -> None:
        from prometheus.__main__ import create_tool_registry
        from prometheus.context.dynamic_tools import DynamicToolLoader

        loader = DynamicToolLoader(create_tool_registry({}), {"enabled": True})
        assert "web_discover" in {s["name"] for s in loader.schemas_for_run(True)}

    def test_a_pinned_list_is_used_as_written(self, exa_key: str) -> None:
        from prometheus.__main__ import create_tool_registry
        from prometheus.context.dynamic_tools import DynamicToolLoader

        registry = create_tool_registry({})
        pinned = DynamicToolLoader(registry, {"enabled": True, "always_loaded": ["bash", "read_file"]})
        assert {s["name"] for s in pinned.schemas_for_run(True)} == {"bash", "read_file"}
        listed = DynamicToolLoader(
            registry, {"enabled": True, "always_loaded": ["bash", "web_discover"]},
        )
        assert {s["name"] for s in listed.schemas_for_run(True)} == {"bash", "web_discover"}

    def test_it_is_in_the_full_catalog(self, exa_key: str) -> None:
        from prometheus.__main__ import create_tool_registry
        from prometheus.context.dynamic_tools import DynamicToolLoader

        loader = DynamicToolLoader(create_tool_registry({}), {"enabled": False})
        assert "web_discover" in {s["name"] for s in loader.schemas_for_run(False)}

    def test_the_description_says_what_it_is_for_and_not_for(self) -> None:
        assert WebDiscoverTool.description == (
            "Find companies, people, papers or pages similar to a description or "
            "URL (neural search). Not for facts or news; use web_search for those."
        )

    def test_its_example_call_validates(self) -> None:
        WebDiscoverInput(**WebDiscoverTool.example_call)


# ---------------------------------------------------------------------------
# The two modes
# ---------------------------------------------------------------------------


class TestQueryMode:
    def test_it_posts_the_query_to_search_with_the_key_in_the_header(self, exa_key: str) -> None:
        exa = FakeExa(_ok("exa_search_results.json"))
        result = _run(WebDiscoverTool(transport=exa.transport),
                      query="companies building open-source vector databases")
        assert not result.is_error, result.output
        [request] = exa.calls
        assert request.method == "POST"
        assert request.url.path == "/search"
        assert request.headers["x-api-key"] == KEY
        body = exa.body()
        assert body["query"] == "companies building open-source vector databases"
        assert body["numResults"] == 5
        assert "summary" in body["contents"]
        assert "category" not in body

    def test_results_are_title_url_and_a_short_summary(self, exa_key: str) -> None:
        exa = FakeExa(_ok("exa_search_results.json"))
        result = _run(WebDiscoverTool(transport=exa.transport), query="vector databases")
        lines = result.output.splitlines()
        assert lines[0] == "Discovered for: vector databases (Exa neural search)"
        assert "1. Qdrant - Vector Database" in lines
        assert "   URL: https://qdrant.tech/" in lines
        assert any(line.strip().startswith("Qdrant is an open-source vector database")
                   for line in lines)
        milvus = next(line for line in lines if line.strip().startswith("Milvus is"))
        assert len(milvus.strip()) <= wd.SUMMARY_CHARS
        assert milvus.rstrip().endswith("…")
        assert "4. Chroma" in lines  # a result with no summary still lists

    def test_the_category_is_passed_through(self, exa_key: str) -> None:
        exa = FakeExa(_ok("exa_search_results.json"))
        _run(WebDiscoverTool(transport=exa.transport), query="Rust compiler engineers",
             category="people")
        assert exa.body()["category"] == "people"

    def test_the_count_is_capped(self, exa_key: str) -> None:
        exa = FakeExa(_ok("exa_search_results.json"))
        result = _run(WebDiscoverTool(transport=exa.transport), query="x", num_results=2)
        assert exa.body()["numResults"] == 2
        assert "3." not in result.output
        with pytest.raises(ValidationError):
            WebDiscoverInput(query="x", num_results=wd.MAX_RESULTS + 1)

    def test_the_key_is_never_in_the_url_or_the_output(self, exa_key: str) -> None:
        exa = FakeExa(_ok("exa_search_results.json"))
        result = _run(WebDiscoverTool(transport=exa.transport), query="x")
        assert KEY not in str(exa.calls[0].url)
        assert KEY not in result.output
        assert KEY not in json.dumps(result.metadata)


class TestFindSimilarMode:
    def test_a_url_goes_to_find_similar(self, exa_key: str) -> None:
        exa = FakeExa(_ok("exa_find_similar_results.json"))
        result = _run(WebDiscoverTool(transport=exa.transport),
                      url="https://arxiv.org/abs/2005.11401")
        assert not result.is_error, result.output
        [request] = exa.calls
        assert request.url.path == "/findSimilar"
        body = exa.body()
        assert body["url"] == "https://arxiv.org/abs/2005.11401"
        assert body["excludeSourceDomain"] is True
        assert "query" not in body
        assert result.output.splitlines()[0] == (
            "Similar to: https://arxiv.org/abs/2005.11401 (Exa neural search)"
        )
        assert "1. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" in result.output

    @pytest.mark.parametrize("url", [
        "http://localhost:8080/admin",
        "http://127.0.0.1/",
        "http://192.168.1.20/wiki",
        "http://10.0.0.5/",
        "http://printer.local/",
        "http://intranet/",
    ])
    def test_a_private_url_is_never_sent_to_exa(self, exa_key: str, url: str) -> None:
        exa = FakeExa(_ok("exa_find_similar_results.json"))
        result = _run(WebDiscoverTool(transport=exa.transport), url=url)
        assert result.is_error
        assert exa.calls == []
        assert "private" in result.output or "local" in result.output

    def test_a_url_that_is_not_http_is_refused(self, exa_key: str) -> None:
        exa = FakeExa(_ok("exa_find_similar_results.json"))
        result = _run(WebDiscoverTool(transport=exa.transport), url="file:///etc/passwd")
        assert result.is_error
        assert exa.calls == []


class TestInput:
    def test_query_or_url_not_both(self) -> None:
        with pytest.raises(ValidationError):
            WebDiscoverInput(query="x", url="https://example.com/")

    def test_one_of_them_is_required(self) -> None:
        with pytest.raises(ValidationError):
            WebDiscoverInput()
        with pytest.raises(ValidationError):
            WebDiscoverInput(query="   ")

    def test_every_parameter_is_described(self) -> None:
        props = WebDiscoverInput.model_json_schema()["properties"]
        assert set(props) == {"query", "url", "category", "num_results"}
        assert all(p.get("description") for p in props.values())


# ---------------------------------------------------------------------------
# Failures: said, and the key never echoed
# ---------------------------------------------------------------------------


class TestFailures:
    @pytest.mark.parametrize("status, fixture, words", [
        (401, "exa_error_401.json", "rejected"),
        (402, "exa_error_402.json", "credits"),
        (429, None, "rate limited"),
        (500, None, "HTTP 500"),
    ])
    def test_an_error_is_said(self, exa_key: str, status: int, fixture: str | None,
                              words: str) -> None:
        exa = FakeExa(_status(status, fixture))
        result = _run(WebDiscoverTool(transport=exa.transport), query="x")
        assert result.is_error
        assert result.output.startswith("web_discover failed:")
        assert words in result.output
        assert KEY not in result.output

    def test_a_timeout_is_said(self, exa_key: str) -> None:
        def slow(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("timed out", request=request)

        result = _run(WebDiscoverTool(transport=FakeExa(slow).transport), query="x")
        assert result.is_error
        assert "timed out" in result.output

    def test_no_results_is_said(self, exa_key: str) -> None:
        exa = FakeExa(lambda r: httpx.Response(200, json={"requestId": "r", "results": []}))
        result = _run(WebDiscoverTool(transport=exa.transport), query="x")
        assert result.is_error
        assert "no results" in result.output.lower()


# ---------------------------------------------------------------------------
# A read-only tool through the normal gate
# ---------------------------------------------------------------------------


class TestTheGate:
    def test_it_is_read_only(self) -> None:
        assert WebDiscoverTool().is_read_only(WebDiscoverInput(query="x")) is True

    def test_a_call_goes_through_the_security_gate(self, exa_key: str) -> None:
        from prometheus.engine.agent_loop import LoopContext, _execute_tool_call
        from prometheus.permissions.checker import SecurityGate
        from prometheus.tools.base import ToolRegistry

        seen: list[tuple[str, dict]] = []

        class RecordingGate(SecurityGate):
            def evaluate(self, tool_name, **kwargs):  # noqa: ANN001, ANN003
                seen.append((tool_name, kwargs))
                return super().evaluate(tool_name, **kwargs)

        registry = ToolRegistry()
        exa = FakeExa(_ok("exa_search_results.json"))
        registry.register(WebDiscoverTool(transport=exa.transport))
        gate = RecordingGate()
        ctx = LoopContext(provider=None, model="t", system_prompt="", max_tokens=512,
                          tool_registry=registry, permission_checker=gate)
        block = asyncio.run(_execute_tool_call(ctx, "web_discover", "t1", {"query": "x"}))
        assert not block.is_error, block.content
        assert [name for name, _ in seen] == ["web_discover"]
        assert seen[0][1]["is_read_only"] is True
        assert len(exa.calls) == 1


class TestTheCallIsBounded:
    def test_a_trickling_answer_ends_at_the_tools_own_limit(
        self, exa_key: str, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """httpx's timeout is per read; a trickle never trips it. The call has a
        wall-clock limit, so it ends with its own message instead of being
        cancelled by the agent loop's 300 s tool timeout."""
        async def drip():
            while True:
                await asyncio.sleep(0.02)
                yield b" "

        monkeypatch.setattr(wd, "TIMEOUT", 0.2)
        tool = WebDiscoverTool(
            transport=FakeExa(lambda r: httpx.Response(200, content=drip())).transport,
        )
        ctx = ToolExecutionContext(cwd=Path.cwd())
        result = asyncio.run(asyncio.wait_for(
            tool.execute(WebDiscoverInput(query="x"), ctx), timeout=5,
        ))
        assert result.is_error
        assert result.output == "web_discover failed: Exa took longer than 0.2s"

    def test_the_limit_is_inside_the_loops_tool_timeout(self) -> None:
        from dataclasses import fields

        from prometheus.engine.agent_loop import LoopContext

        loop_default = next(
            f.default for f in fields(LoopContext) if f.name == "tool_timeout_seconds"
        )
        assert WebDiscoverTool.execution_timeout_seconds is None
        assert wd.TIMEOUT < loop_default
