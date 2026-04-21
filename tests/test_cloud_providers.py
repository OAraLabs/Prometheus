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
