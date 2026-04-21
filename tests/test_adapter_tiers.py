"""Tests for three-tier adapter system."""

import pytest

from prometheus.adapter import ModelAdapter
from prometheus.adapter.formatter import (
    AnthropicFormatter,
    GemmaFormatter,
    PassthroughFormatter,
    QwenFormatter,
)
from prometheus.adapter.retry import RetryAction
from prometheus.adapter.validator import Strictness
from prometheus.providers.base import ModelProvider
from prometheus.providers.anthropic import AnthropicProvider
from prometheus.providers.llama_cpp import LlamaCppProvider
from prometheus.providers.ollama import OllamaProvider
from prometheus.providers.openai_compat import OpenAICompatProvider
from prometheus.tools.base import ToolRegistry
from prometheus.tools.builtin.bash import BashTool
from prometheus.tools.builtin.file_read import FileReadTool


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(BashTool())
    reg.register(FileReadTool())
    return reg


# ---------------------------------------------------------------------------
# Provider api_enforced_structure flag
# ---------------------------------------------------------------------------

class TestProviderFlag:
    def test_base_defaults_false(self):
        assert ModelProvider.api_enforced_structure is False

    def test_anthropic_true(self):
        assert AnthropicProvider.api_enforced_structure is True

    def test_openai_compat_true(self):
        assert OpenAICompatProvider.api_enforced_structure is True

    def test_llama_cpp_false(self):
        p = LlamaCppProvider()
        assert p.api_enforced_structure is False

    def test_ollama_false(self):
        p = OllamaProvider()
        assert p.api_enforced_structure is False


# ---------------------------------------------------------------------------
# Tier selection via create_adapter
# ---------------------------------------------------------------------------

class TestGetAdapterTier:
    def test_anthropic_tier_off(self):
        from prometheus.__main__ import _get_adapter_tier
        assert _get_adapter_tier("anthropic", "") == "off"

    def test_cloud_provider_tier_off(self):
        from prometheus.__main__ import _get_adapter_tier
        assert _get_adapter_tier("openai", "") == "off"

    def test_gemma_on_llama_cpp_tier_light(self):
        from prometheus.__main__ import _get_adapter_tier
        assert _get_adapter_tier("llama_cpp", "gemma4-26b") == "light"

    def test_qwen_on_ollama_tier_light(self):
        from prometheus.__main__ import _get_adapter_tier
        assert _get_adapter_tier("ollama", "qwen3.5-32b") == "light"

    def test_unknown_model_tier_full(self):
        from prometheus.__main__ import _get_adapter_tier
        assert _get_adapter_tier("llama_cpp", "my-custom-finetune") == "full"


# ---------------------------------------------------------------------------
# Tier behavior on ModelAdapter
# ---------------------------------------------------------------------------

class TestTierOff:
    def test_tier_off_skips_validation(self, registry):
        adapter = ModelAdapter(tier="off")
        assert adapter.tier == "off"
        # validate_and_repair should passthrough without touching input
        name, inp, repairs = adapter.validate_and_repair("bash", {"command": "ls"}, registry)
        assert name == "bash"
        assert repairs == []

    def test_tier_off_skips_extract_tool_calls(self):
        adapter = ModelAdapter(tier="off")
        result = adapter.extract_tool_calls('{"name": "bash", "arguments": {"command": "ls"}}')
        assert result == []

    def test_tier_off_grammar_returns_none(self, registry):
        adapter = ModelAdapter(tier="off")
        assert adapter.generate_grammar(registry) is None

    def test_tier_off_retry_aborts(self, registry):
        adapter = ModelAdapter(tier="off")
        action, msg = adapter.handle_retry("bash", "some error", registry)
        assert action == RetryAction.ABORT

    def test_tier_off_strictness_none(self):
        adapter = ModelAdapter(tier="off")
        assert adapter.validator.strictness == Strictness.NONE

    def test_tier_off_no_adaptive(self):
        adapter = ModelAdapter(tier="off")
        assert adapter._adaptive_strictness is False


class TestTierLight:
    def test_tier_light_strictness_none(self):
        adapter = ModelAdapter(tier="light")
        assert adapter.validator.strictness == Strictness.NONE

    def test_tier_light_max_retries_one(self):
        adapter = ModelAdapter(tier="light")
        assert adapter.retry.max_retries == 1

    def test_tier_light_adaptive_on(self):
        adapter = ModelAdapter(tier="light")
        assert adapter._adaptive_strictness is True

    def test_tier_light_grammar_works(self, registry):
        adapter = ModelAdapter(tier="light")
        grammar = adapter.generate_grammar(registry)
        assert grammar is not None
        assert "bash" in grammar

    def test_tier_light_extract_works(self):
        """Light tier: enforcer ON — model may emit tool calls as text."""
        adapter = ModelAdapter(tier="light")
        text = '```json\n{"name": "bash", "arguments": {"command": "ls"}}\n```'
        result = adapter.extract_tool_calls(text)
        assert len(result) > 0

    def test_tier_light_validates_passthrough_valid(self, registry):
        adapter = ModelAdapter(tier="light")
        name, inp, repairs = adapter.validate_and_repair("bash", {"command": "ls"}, registry)
        assert name == "bash"


class TestTierFull:
    def test_tier_full_strictness_medium(self):
        adapter = ModelAdapter(strictness="MEDIUM", tier="full")
        assert adapter.validator.strictness == Strictness.MEDIUM

    def test_tier_full_max_retries_three(self):
        adapter = ModelAdapter(tier="full")
        assert adapter.retry.max_retries == 3

    def test_tier_full_grammar_works(self, registry):
        adapter = ModelAdapter(tier="full")
        grammar = adapter.generate_grammar(registry)
        assert grammar is not None

    def test_tier_full_extract_works(self):
        adapter = ModelAdapter(tier="full")
        text = '```json\n{"name": "bash", "arguments": {"command": "ls"}}\n```'
        result = adapter.extract_tool_calls(text)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Adaptive strictness escalation: tier light → full per-tool
# ---------------------------------------------------------------------------

class TestTierEscalation:
    def test_light_tool_escalates_to_medium_on_failures(self):
        adapter = ModelAdapter(
            tier="light",
            strictness_threshold=0.8,
        )
        # 6 successes, 4 failures = 60% — below 80% threshold
        for _ in range(6):
            adapter.record_tool_call("bash", True)
        for _ in range(4):
            adapter.record_tool_call("bash", False)
        # bash should now be at MEDIUM (tier 3 behavior for this tool)
        assert adapter.get_effective_strictness("bash") == Strictness.MEDIUM

    def test_other_tools_stay_at_none(self):
        adapter = ModelAdapter(tier="light", strictness_threshold=0.8)
        for _ in range(6):
            adapter.record_tool_call("bash", True)
        for _ in range(4):
            adapter.record_tool_call("bash", False)
        # grep was never called — should still be at base (NONE)
        assert adapter.get_effective_strictness("grep") == Strictness.NONE

    def test_escalation_medium_to_strict(self):
        adapter = ModelAdapter(tier="light", strictness_threshold=0.8)
        # First bump to MEDIUM
        adapter._tool_strictness["bash"] = Strictness.MEDIUM
        # Then more failures
        for _ in range(6):
            adapter.record_tool_call("bash", True)
        for _ in range(4):
            adapter.record_tool_call("bash", False)
        assert adapter.get_effective_strictness("bash") == Strictness.STRICT


# ---------------------------------------------------------------------------
# create_adapter integration
# ---------------------------------------------------------------------------

class TestCreateAdapterTiers:
    def test_anthropic_creates_tier_off(self):
        from prometheus.__main__ import create_adapter
        adapter = create_adapter({"provider": "anthropic"})
        assert adapter.tier == "off"
        assert isinstance(adapter.formatter, AnthropicFormatter)

    def test_gemma_llama_cpp_creates_tier_light(self):
        from prometheus.__main__ import create_adapter
        adapter = create_adapter({"provider": "llama_cpp", "model": "gemma4-26b"})
        assert adapter.tier == "light"
        assert isinstance(adapter.formatter, GemmaFormatter)

    def test_qwen_ollama_creates_tier_light(self):
        from prometheus.__main__ import create_adapter
        adapter = create_adapter({"provider": "ollama", "model": "qwen3.5-32b"})
        assert adapter.tier == "light"
        assert isinstance(adapter.formatter, QwenFormatter)

    def test_unknown_model_creates_tier_full(self):
        from prometheus.__main__ import create_adapter
        adapter = create_adapter({"provider": "llama_cpp", "model": "my-custom-finetune"})
        assert adapter.tier == "full"
        assert adapter.validator.strictness == Strictness.MEDIUM

    def test_openai_cloud_creates_tier_off(self):
        from prometheus.__main__ import create_adapter
        adapter = create_adapter({"provider": "openai"})
        assert adapter.tier == "off"
        assert isinstance(adapter.formatter, PassthroughFormatter)
