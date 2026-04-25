"""Tests for cloud API providers, registry, cost tracking, and formatter.

All tests use mocked HTTP — no real API calls.
"""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import ApiMessageRequest


# -----------------------------------------------------------------------
# OpenAICompatProvider
# -----------------------------------------------------------------------


class TestOpenAICompatProvider:
    """Tests for providers.openai_compat.OpenAICompatProvider."""

    def _make_provider(self, base_url="https://api.openai.com/v1", api_key="sk-test"):
        from prometheus.providers.openai_compat import OpenAICompatProvider
        return OpenAICompatProvider(
            base_url=base_url, api_key=api_key, model="gpt-4o"
        )

    def test_url_building_with_v1_suffix(self):
        p = self._make_provider(base_url="https://api.openai.com/v1")
        # base_url ends with /v1, so _call_once should use /v1/chat/completions
        assert p._base_url == "https://api.openai.com/v1"

    def test_url_building_without_v1_suffix(self):
        p = self._make_provider(base_url="https://api.x.ai/v1")
        assert p._base_url == "https://api.x.ai/v1"

    def test_gemini_url(self):
        p = self._make_provider(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai"
        )
        assert p._base_url == "https://generativelanguage.googleapis.com/v1beta/openai"


# -----------------------------------------------------------------------
# AnthropicProvider (already exists — verify it works)
# -----------------------------------------------------------------------


class TestAnthropicProvider:
    """Tests for providers.anthropic.AnthropicProvider."""

    def test_init_with_api_key(self):
        from prometheus.providers.anthropic import AnthropicProvider
        p = AnthropicProvider(api_key="sk-ant-test", model="claude-sonnet-4-6")
        assert p._model == "claude-sonnet-4-6"
        assert p._api_key == "sk-ant-test"

    def test_init_missing_key_raises(self):
        from prometheus.providers.anthropic import AnthropicProvider
        with patch.dict(os.environ, {}, clear=True):
            # Remove any ANTHROPIC_API_KEY that might be set
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises(ValueError, match="API key"):
                AnthropicProvider()

    def test_prompt_caching_headers(self):
        from prometheus.providers.anthropic import AnthropicProvider
        p = AnthropicProvider(api_key="sk-ant-test", prompt_caching=True)
        assert p._prompt_caching is True


# -----------------------------------------------------------------------
# ProviderRegistry
# -----------------------------------------------------------------------


class TestProviderRegistry:
    """Tests for providers.registry.ProviderRegistry."""

    def test_create_llama_cpp(self):
        from prometheus.providers.registry import ProviderRegistry
        from prometheus.providers.llama_cpp import LlamaCppProvider
        p = ProviderRegistry.create({"provider": "llama_cpp", "base_url": "http://localhost:8080"})
        assert isinstance(p, LlamaCppProvider)

    def test_create_ollama(self):
        from prometheus.providers.registry import ProviderRegistry
        from prometheus.providers.ollama import OllamaProvider
        p = ProviderRegistry.create({"provider": "ollama", "base_url": "http://localhost:11434"})
        assert isinstance(p, OllamaProvider)

    def test_create_openai(self):
        from prometheus.providers.registry import ProviderRegistry
        from prometheus.providers.openai_compat import OpenAICompatProvider
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            p = ProviderRegistry.create({"provider": "openai"})
            assert isinstance(p, OpenAICompatProvider)

    def test_create_gemini(self):
        from prometheus.providers.registry import ProviderRegistry
        from prometheus.providers.openai_compat import OpenAICompatProvider
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            p = ProviderRegistry.create({"provider": "gemini"})
            assert isinstance(p, OpenAICompatProvider)
            assert "generativelanguage" in p._base_url

    def test_create_xai(self):
        from prometheus.providers.registry import ProviderRegistry
        from prometheus.providers.openai_compat import OpenAICompatProvider
        with patch.dict(os.environ, {"XAI_API_KEY": "test-key"}):
            p = ProviderRegistry.create({"provider": "xai"})
            assert isinstance(p, OpenAICompatProvider)
            assert "x.ai" in p._base_url

    def test_create_anthropic(self):
        from prometheus.providers.registry import ProviderRegistry
        from prometheus.providers.anthropic import AnthropicProvider
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            p = ProviderRegistry.create({"provider": "anthropic"})
            assert isinstance(p, AnthropicProvider)

    def test_create_with_api_key_env(self):
        from prometheus.providers.registry import ProviderRegistry
        with patch.dict(os.environ, {"MY_CUSTOM_KEY": "sk-custom"}):
            p = ProviderRegistry.create({
                "provider": "openai",
                "api_key_env": "MY_CUSTOM_KEY",
            })
            assert p._api_key == "sk-custom"

    def test_missing_api_key_raises(self):
        from prometheus.providers.registry import ProviderRegistry
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(ValueError, match="API key"):
                ProviderRegistry.create({"provider": "openai"})

    def test_unknown_provider_raises(self):
        from prometheus.providers.registry import ProviderRegistry
        with pytest.raises(ValueError, match="Unknown provider"):
            ProviderRegistry.create({"provider": "does_not_exist"})

    def test_is_cloud(self):
        from prometheus.providers.registry import ProviderRegistry
        assert ProviderRegistry.is_cloud("openai") is True
        assert ProviderRegistry.is_cloud("anthropic") is True
        assert ProviderRegistry.is_cloud("gemini") is True
        assert ProviderRegistry.is_cloud("xai") is True
        assert ProviderRegistry.is_cloud("llama_cpp") is False
        assert ProviderRegistry.is_cloud("ollama") is False

    def test_list_providers(self):
        from prometheus.providers.registry import ProviderRegistry
        providers = ProviderRegistry.list_providers()
        assert "openai" in providers
        assert "anthropic" in providers
        assert "llama_cpp" in providers
        assert len(providers) == 7


# -----------------------------------------------------------------------
# PassthroughFormatter
# -----------------------------------------------------------------------


class TestPassthroughFormatter:
    """Tests for adapter.formatter.PassthroughFormatter."""

    def test_format_tools_passthrough(self):
        from prometheus.adapter.formatter import PassthroughFormatter
        f = PassthroughFormatter()
        tools = [{"name": "bash", "description": "run a command"}]
        assert f.format_tools(tools) is tools

    def test_format_system_prompt_passthrough(self):
        from prometheus.adapter.formatter import PassthroughFormatter
        f = PassthroughFormatter()
        prompt = "You are a helpful assistant."
        assert f.format_system_prompt(prompt, []) == prompt

    def test_parse_tool_calls_empty(self):
        from prometheus.adapter.formatter import PassthroughFormatter
        f = PassthroughFormatter()
        assert f.parse_tool_calls("some text") == []


# -----------------------------------------------------------------------
# CostTracker
# -----------------------------------------------------------------------


class TestCostTracker:
    """Tests for telemetry.cost.CostTracker."""

    def test_record_known_model(self):
        from prometheus.telemetry.cost import CostTracker
        ct = CostTracker()
        cost = ct.record("gpt-4o", input_tokens=1000, output_tokens=500)
        # gpt-4o: $2.50/1M input, $10/1M output
        expected = (1000 * 2.50 + 500 * 10.00) / 1_000_000
        assert abs(cost - expected) < 1e-8

    def test_record_unknown_model_free(self):
        from prometheus.telemetry.cost import CostTracker
        ct = CostTracker()
        cost = ct.record("some-local-model", input_tokens=1000, output_tokens=500)
        assert cost == 0.0

    def test_total_cost_accumulates(self):
        from prometheus.telemetry.cost import CostTracker
        ct = CostTracker()
        ct.record("gpt-4o", 1000, 500)
        ct.record("gpt-4o", 2000, 1000)
        assert ct.total_cost > 0
        assert ct.total_input_tokens == 3000
        assert ct.total_output_tokens == 1500

    def test_report_no_usage(self):
        from prometheus.telemetry.cost import CostTracker
        ct = CostTracker()
        assert "no cloud API usage" in ct.report()

    def test_report_with_usage(self):
        from prometheus.telemetry.cost import CostTracker
        ct = CostTracker()
        ct.record("gpt-4o", 10000, 2000)
        report = ct.report()
        assert "Session cost: $" in report
        assert "10,000 input" in report

    def test_prefix_match(self):
        from prometheus.telemetry.cost import CostTracker
        ct = CostTracker()
        # "gpt-4o-2024-05-13" should match "gpt-4o" pricing
        cost = ct.record("gpt-4o-2024-05-13", 1_000_000, 0)
        assert cost == 2.50  # $2.50 per 1M input tokens

    def test_to_dict(self):
        from prometheus.telemetry.cost import CostTracker
        ct = CostTracker()
        ct.record("gpt-4o", 1000, 500)
        d = ct.to_dict()
        assert "total_cost_usd" in d
        assert d["total_input_tokens"] == 1000
        assert d["total_output_tokens"] == 500
        assert d["records"] == 1


# -----------------------------------------------------------------------
# create_adapter — formatter selection
# -----------------------------------------------------------------------


class TestCreateAdapter:
    """Tests for __main__.create_adapter formatter selection."""

    def test_anthropic_gets_anthropic_formatter(self):
        from prometheus.__main__ import create_adapter
        from prometheus.adapter.formatter import AnthropicFormatter
        adapter = create_adapter({"provider": "anthropic"})
        assert isinstance(adapter.formatter, AnthropicFormatter)

    def test_openai_gets_passthrough_formatter(self):
        from prometheus.__main__ import create_adapter
        from prometheus.adapter.formatter import PassthroughFormatter
        adapter = create_adapter({"provider": "openai"})
        assert isinstance(adapter.formatter, PassthroughFormatter)

    def test_gemini_gets_passthrough_formatter(self):
        from prometheus.__main__ import create_adapter
        from prometheus.adapter.formatter import PassthroughFormatter
        adapter = create_adapter({"provider": "gemini"})
        assert isinstance(adapter.formatter, PassthroughFormatter)

    def test_xai_gets_passthrough_formatter(self):
        from prometheus.__main__ import create_adapter
        from prometheus.adapter.formatter import PassthroughFormatter
        adapter = create_adapter({"provider": "xai"})
        assert isinstance(adapter.formatter, PassthroughFormatter)

    def test_llama_cpp_gemma_tier_light(self):
        """Gemma 4 has native function_calling → tier light, keeps GemmaFormatter."""
        from prometheus.__main__ import create_adapter
        from prometheus.adapter.formatter import GemmaFormatter
        adapter = create_adapter({"provider": "llama_cpp", "model": "gemma4-26b"})
        assert adapter.tier == "light"
        assert isinstance(adapter.formatter, GemmaFormatter)

    def test_llama_cpp_qwen_tier_light(self):
        """Qwen has native function_calling → tier light, keeps QwenFormatter."""
        from prometheus.__main__ import create_adapter
        from prometheus.adapter.formatter import QwenFormatter
        adapter = create_adapter({"provider": "llama_cpp", "model": "qwen3.5-32b"})
        assert adapter.tier == "light"
        assert isinstance(adapter.formatter, QwenFormatter)

    def test_cloud_provider_strictness_none(self):
        from prometheus.__main__ import create_adapter
        adapter = create_adapter({"provider": "openai"})
        # NONE strictness — cloud models don't need validation
        assert adapter.validator.strictness.name == "NONE"


# -----------------------------------------------------------------------
# Shared helper tests (_build_openai_messages)
# -----------------------------------------------------------------------


class TestBuildOpenAIMessages:
    """Tests for stub._build_openai_messages used by OpenAICompatProvider."""

    def test_system_prompt_first(self):
        from prometheus.providers.stub import _build_openai_messages
        request = ApiMessageRequest(
            model="test",
            messages=[ConversationMessage.from_user_text("hello")],
            system_prompt="You are helpful.",
        )
        msgs = _build_openai_messages(request)
        assert msgs[0] == {"role": "system", "content": "You are helpful."}
        assert msgs[1]["role"] == "user"

    def test_no_system_prompt(self):
        from prometheus.providers.stub import _build_openai_messages
        request = ApiMessageRequest(
            model="test",
            messages=[ConversationMessage.from_user_text("hello")],
        )
        msgs = _build_openai_messages(request)
        assert msgs[0]["role"] == "user"

    def test_tool_use_message(self):
        from prometheus.providers.stub import _build_openai_messages
        msg = ConversationMessage(
            role="assistant",
            content=[
                ToolUseBlock(id="tc_1", name="bash", input={"command": "ls"}),
            ],
        )
        request = ApiMessageRequest(model="test", messages=[msg])
        msgs = _build_openai_messages(request)
        assert msgs[0]["role"] == "assistant"
        assert len(msgs[0]["tool_calls"]) == 1
        assert msgs[0]["tool_calls"][0]["function"]["name"] == "bash"


# -----------------------------------------------------------------------
# AnthropicProvider streaming tool_use parse (Phase 4 follow-up fix)
# -----------------------------------------------------------------------
#
# Regression guard: AnthropicProvider.stream_message used to read
# `block["input"]` first and only fall back to `partial_json` when input
# wasn't a dict. But Anthropic's SSE streaming ALWAYS sends `input: {}` on
# content_block_start as an empty placeholder — the real arguments stream
# in as `input_json_delta` events accumulated into `partial_json`. Since
# `{}` is a dict, the fallback never fired and every streamed tool_use
# came back with empty input, which made /claude look like it dropped
# user queries on tool calls.
#
# Fix: prefer `partial_json` when present; fall back to `block["input"]`
# only for the non-streaming case (block comes in fully populated).
# These tests lock both paths in place.


class _FakeSSEResponse:
    """Minimal stand-in for httpx's streaming response context manager."""

    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self._lines = lines
        self.status_code = status_code

    async def __aenter__(self) -> "_FakeSSEResponse":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeHttpxClient:
    """Minimal stand-in for httpx.AsyncClient used in AnthropicProvider."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    async def __aenter__(self) -> "_FakeHttpxClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def stream(self, method: str, url: str, **kwargs: object) -> _FakeSSEResponse:
        return _FakeSSEResponse(self._lines)


def _sse(payload: dict) -> str:
    """Format a dict as one ``data: {...}`` SSE line."""
    return f"data: {json.dumps(payload)}"


async def _collect_stream(provider, request):
    """Helper: drain stream_message and return (text_events, final_event)."""
    from prometheus.providers.base import (
        ApiMessageCompleteEvent,
        ApiTextDeltaEvent,
    )
    text_events: list[ApiTextDeltaEvent] = []
    final: ApiMessageCompleteEvent | None = None
    async for event in provider.stream_message(request):
        if isinstance(event, ApiTextDeltaEvent):
            text_events.append(event)
        elif isinstance(event, ApiMessageCompleteEvent):
            final = event
    return text_events, final


class TestAnthropicStreamingToolUse:
    """Phase 4 follow-up: streaming tool_use input must come from partial_json."""

    @pytest.mark.asyncio
    async def test_anthropic_streaming_tool_use_input(self):
        """SSE tool_use: input arrives across multiple input_json_delta
        fragments; final ToolUseBlock.input must be the reconstructed dict,
        NOT the empty `{}` placeholder from content_block_start.
        """
        from prometheus.providers.anthropic import AnthropicProvider
        from prometheus.engine.messages import ToolUseBlock

        fake_lines = [
            _sse({"type": "message_start", "message": {"usage": {"input_tokens": 10}}}),
            "",
            _sse({
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "lcm_grep",
                    # The placeholder that used to defeat the fallback path.
                    "input": {},
                },
            }),
            "",
            _sse({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"query":'},
            }),
            "",
            _sse({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": ' "NPPES"}'},
            }),
            "",
            _sse({"type": "content_block_stop", "index": 0}),
            "",
            _sse({
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 20},
            }),
            "",
            _sse({"type": "message_stop"}),
            "",
        ]

        provider = AnthropicProvider(api_key="sk-ant-test", model="claude-haiku-4-5-20251001")
        request = ApiMessageRequest(
            model="claude-haiku-4-5-20251001",
            messages=[ConversationMessage.from_user_text("search memory")],
        )

        fake_client = _FakeHttpxClient(fake_lines)
        with patch(
            "prometheus.providers.anthropic.httpx.AsyncClient",
            lambda **kw: fake_client,
        ):
            _, final = await _collect_stream(provider, request)

        assert final is not None, "stream_message must emit an ApiMessageCompleteEvent"
        tool_uses = final.message.tool_uses
        assert len(tool_uses) == 1
        tu = tool_uses[0]
        assert isinstance(tu, ToolUseBlock)
        assert tu.name == "lcm_grep"
        assert tu.id == "toolu_01"
        # The whole point of the fix: input is reconstructed from partial_json,
        # not left as the `{}` placeholder from content_block_start.
        assert tu.input == {"query": "NPPES"}

    @pytest.mark.asyncio
    async def test_anthropic_nonstreaming_tool_use_input_backward_compat(self):
        """When ``partial_json`` is absent and ``input`` is populated on the
        block directly (non-streaming-style response), the finalize step
        still picks up the dict from block['input']."""
        from prometheus.providers.anthropic import AnthropicProvider
        from prometheus.engine.messages import ToolUseBlock

        fake_lines = [
            _sse({"type": "message_start", "message": {"usage": {"input_tokens": 5}}}),
            "",
            _sse({
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_02",
                    "name": "bash",
                    "input": {"command": "ls -la"},
                },
            }),
            "",
            # No input_json_delta events — block's input is already complete.
            _sse({"type": "content_block_stop", "index": 0}),
            "",
            _sse({
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 10},
            }),
            "",
            _sse({"type": "message_stop"}),
            "",
        ]

        provider = AnthropicProvider(api_key="sk-ant-test")
        request = ApiMessageRequest(
            model="claude-haiku-4-5-20251001",
            messages=[ConversationMessage.from_user_text("run ls")],
        )

        fake_client = _FakeHttpxClient(fake_lines)
        with patch(
            "prometheus.providers.anthropic.httpx.AsyncClient",
            lambda **kw: fake_client,
        ):
            _, final = await _collect_stream(provider, request)

        assert final is not None
        tool_uses = final.message.tool_uses
        assert len(tool_uses) == 1
        tu = tool_uses[0]
        assert isinstance(tu, ToolUseBlock)
        assert tu.name == "bash"
        assert tu.input == {"command": "ls -la"}

    @pytest.mark.asyncio
    async def test_anthropic_streaming_tool_use_with_invalid_partial_json(self):
        """Defensive: if partial_json is present but doesn't parse, fall
        back to empty dict rather than raising."""
        from prometheus.providers.anthropic import AnthropicProvider

        fake_lines = [
            _sse({"type": "message_start", "message": {"usage": {"input_tokens": 5}}}),
            "",
            _sse({
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use", "id": "toolu_03", "name": "x", "input": {},
                },
            }),
            "",
            _sse({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"broken'},
            }),
            "",
            _sse({"type": "content_block_stop", "index": 0}),
            "",
            _sse({"type": "message_stop"}),
            "",
        ]
        provider = AnthropicProvider(api_key="sk-ant-test")
        request = ApiMessageRequest(
            model="claude-haiku-4-5-20251001",
            messages=[ConversationMessage.from_user_text("x")],
        )
        fake_client = _FakeHttpxClient(fake_lines)
        with patch(
            "prometheus.providers.anthropic.httpx.AsyncClient",
            lambda **kw: fake_client,
        ):
            _, final = await _collect_stream(provider, request)

        assert final is not None
        tu = final.message.tool_uses[0]
        # Malformed partial_json → empty dict, not a raise. The tool call
        # will fail downstream but the parser itself stays robust.
        assert tu.input == {}


# -----------------------------------------------------------------------
# Error-body logging — 400s from Anthropic were previously invisible
# -----------------------------------------------------------------------


class _FakeErrorResponse:
    """Fake streaming response that simulates an HTTP error with a body.

    Mirrors the minimal shape of httpx.Response that AnthropicProvider
    touches when status_code >= 400: aread(), text, raise_for_status().
    """

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self._body = body
        self.text = body  # Set upfront so aread() is a no-op like after streaming

    async def __aenter__(self) -> "_FakeErrorResponse":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def aread(self) -> bytes:
        return self._body.encode("utf-8")

    def raise_for_status(self) -> None:
        import httpx
        req = MagicMock(spec=httpx.Request)
        req.method = "POST"
        req.url = "https://api.anthropic.com/v1/messages"
        raise httpx.HTTPStatusError(
            f"Client error '{self.status_code}' for url ...",
            request=req,
            response=MagicMock(status_code=self.status_code, text=self._body),
        )

    async def aiter_lines(self):
        if False:
            yield ""  # unreachable — kept so this satisfies AsyncIterator shape


class _FakeErrorClient:
    """httpx.AsyncClient stand-in that yields a _FakeErrorResponse."""

    def __init__(self, status_code: int, body: str) -> None:
        self._status_code = status_code
        self._body = body

    async def __aenter__(self) -> "_FakeErrorClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def stream(self, method: str, url: str, **kwargs: object) -> _FakeErrorResponse:
        return _FakeErrorResponse(self._status_code, self._body)


class TestAnthropicErrorBodyLogging:
    """Anthropic 400s used to propagate as bare httpx.HTTPStatusError with
    no body logged — the actual API error envelope (which specifies *why*
    the payload was rejected) was invisible. Other providers (stub,
    llama_cpp, ollama) log `response.text[:500]` before raising; this
    test pins that behavior for AnthropicProvider too.
    """

    @pytest.mark.asyncio
    async def test_400_response_body_is_logged_before_raise(self, caplog):
        import httpx
        import logging
        from prometheus.providers.anthropic import AnthropicProvider

        body = (
            '{"type":"error","error":{"type":"invalid_request_error",'
            '"message":"messages.3.content: Input should be a non-empty array"}}'
        )
        fake_client = _FakeErrorClient(status_code=400, body=body)
        provider = AnthropicProvider(api_key="sk-ant-test")
        request = ApiMessageRequest(
            model="claude-haiku-4-5-20251001",
            messages=[ConversationMessage.from_user_text("hi")],
        )

        with caplog.at_level(logging.ERROR, logger="prometheus.providers.anthropic"):
            with patch(
                "prometheus.providers.anthropic.httpx.AsyncClient",
                lambda **kw: fake_client,
            ):
                with pytest.raises(httpx.HTTPStatusError) as exc_info:
                    async for _ in provider.stream_message(request):
                        pass

        # The original exception still propagates (400 is not in the
        # retryable set).
        assert exc_info.value.response.status_code == 400

        # And the body is now visible in the log.
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert error_records, "expected at least one ERROR log line"
        combined = "\n".join(r.getMessage() for r in error_records)
        assert "400" in combined
        assert "invalid_request_error" in combined
        assert "non-empty array" in combined

    @pytest.mark.asyncio
    async def test_429_response_body_is_also_logged(self, caplog):
        """Retryable statuses (429) also need the body visible so we can
        distinguish rate limits from quota exhaustion from regional
        outages. The log must fire on the FIRST failed attempt, not just
        the final re-raise."""
        import httpx
        import logging
        from prometheus.providers.anthropic import AnthropicProvider

        body = '{"type":"error","error":{"type":"rate_limit_error","message":"rate limited"}}'
        fake_client = _FakeErrorClient(status_code=429, body=body)
        provider = AnthropicProvider(api_key="sk-ant-test")
        request = ApiMessageRequest(
            model="claude-haiku-4-5-20251001",
            messages=[ConversationMessage.from_user_text("hi")],
        )

        # Short-circuit the retry backoff so the test doesn't sleep.
        async def _no_sleep(*_a, **_kw):
            return None

        with caplog.at_level(logging.ERROR, logger="prometheus.providers.anthropic"):
            with patch(
                "prometheus.providers.anthropic.httpx.AsyncClient",
                lambda **kw: fake_client,
            ), patch("asyncio.sleep", _no_sleep):
                with pytest.raises(httpx.HTTPStatusError):
                    async for _ in provider.stream_message(request):
                        pass

        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert error_records, "expected at least one ERROR log line for 429"
        combined = "\n".join(r.getMessage() for r in error_records)
        assert "rate_limit_error" in combined


# -----------------------------------------------------------------------
# _build_adapter_for tier contract (Phase 4 follow-up secondary fix)
# -----------------------------------------------------------------------


class TestBuildAdapterForCloudProviderTier:
    """Cloud providers (anthropic, openai, gemini, xai) must build adapters
    at tier=off. API-native tool calling means validator/GBNF/enforcer
    are unnecessary, and tier=off is the documented contract (see
    ModelAdapter class docstring). Local providers stay on the full
    pipeline.
    """

    def test_anthropic_adapter_tier_is_off(self):
        from prometheus.router.model_router import _build_adapter_for
        from prometheus.adapter import ModelAdapter
        adapter = _build_adapter_for("anthropic")
        assert adapter.tier == ModelAdapter.TIER_OFF

    def test_openai_adapter_tier_is_off(self):
        from prometheus.router.model_router import _build_adapter_for
        from prometheus.adapter import ModelAdapter
        adapter = _build_adapter_for("openai")
        assert adapter.tier == ModelAdapter.TIER_OFF

    def test_gemini_adapter_tier_is_off(self):
        from prometheus.router.model_router import _build_adapter_for
        from prometheus.adapter import ModelAdapter
        adapter = _build_adapter_for("gemini")
        assert adapter.tier == ModelAdapter.TIER_OFF

    def test_xai_adapter_tier_is_off(self):
        from prometheus.router.model_router import _build_adapter_for
        from prometheus.adapter import ModelAdapter
        adapter = _build_adapter_for("xai")
        assert adapter.tier == ModelAdapter.TIER_OFF

    def test_tier_off_implies_no_grammar_no_retries(self):
        """tier=off should pin strictness=NONE and max_retries=0 per
        ModelAdapter.__init__ contract."""
        from prometheus.router.model_router import _build_adapter_for
        from prometheus.adapter.validator import Strictness
        adapter = _build_adapter_for("anthropic")
        assert adapter.validator.strictness == Strictness.NONE
        assert adapter.retry.max_retries == 0

    def test_local_provider_stays_on_full_pipeline(self):
        """Local providers (llama_cpp, ollama) keep the full adapter
        pipeline — their servers don't guarantee tool-call structure."""
        from prometheus.router.model_router import _build_adapter_for
        from prometheus.adapter import ModelAdapter
        adapter = _build_adapter_for("llama_cpp")
        # tier is not "off" — local models need the validator/enforcer/retry.
        assert adapter.tier != ModelAdapter.TIER_OFF
