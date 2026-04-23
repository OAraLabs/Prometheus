"""Tests for the Sprint 23 Model Router.

Tests classification, routing decisions, fallback, escalation,
user overrides, adapter auto-adjustment, and config loading.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prometheus.router.model_router import (
    ModelRouter,
    RouteDecision,
    RouteReason,
    RouterConfig,
    OVERRIDE_PRESETS,
    TaskClassifier,
    TaskType,
    _build_adapter_for,
    load_router_config,
)


# -- Helpers ----------------------------------------------------------------

def _make_router(**overrides) -> ModelRouter:
    """Create a router with a mock primary provider and default config."""
    primary = MagicMock()
    primary.model = "gemma4-26b"
    adapter = MagicMock()

    cfg = RouterConfig(**overrides)
    return ModelRouter(
        config=cfg,
        primary_provider=primary,
        primary_adapter=adapter,
        primary_model="gemma4-26b",
    )


# -- Classification ---------------------------------------------------------

class TestClassification:
    def test_short_greeting_is_simple(self):
        r = _make_router(smart_routing_enabled=True)
        assert r._classify_complexity("hi") == "simple"

    def test_question_is_simple(self):
        r = _make_router()
        assert r._classify_complexity("how are you") == "simple"

    def test_what_time_is_simple(self):
        r = _make_router()
        assert r._classify_complexity("what time is it") == "simple"

    def test_refactor_is_complex(self):
        r = _make_router()
        assert r._classify_complexity("refactor the auth module to use JWT") == "complex"

    def test_code_block_is_complex(self):
        r = _make_router()
        assert r._classify_complexity("fix this:\n```python\ndef foo():\n  pass\n```") == "complex"

    def test_multiline_is_complex(self):
        r = _make_router()
        assert r._classify_complexity("line one\nline two") == "complex"

    def test_over_160_chars_is_complex(self):
        r = _make_router()
        assert r._classify_complexity("a " * 100) == "complex"

    def test_over_28_words_is_complex(self):
        r = _make_router()
        assert r._classify_complexity(" ".join(["word"] * 30)) == "complex"

    def test_analyze_is_complex(self):
        r = _make_router()
        assert r._classify_complexity("analyze the performance of this query") == "complex"

    def test_implement_is_complex(self):
        r = _make_router()
        assert r._classify_complexity("implement a cache layer") == "complex"


# -- Routing decisions -------------------------------------------------------

class TestRouting:
    def test_no_override_no_smart_returns_primary(self):
        r = _make_router()
        d = r.route("hello")
        assert d.reason == RouteReason.PRIMARY
        assert d.provider is r.primary_provider

    def test_user_override_returns_override(self):
        """Phase 3.5: override is per-session, so route() must be given a session_id."""
        r = _make_router()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            r.set_override("chat_test", OVERRIDE_PRESETS["claude"])
            d = r.route("hello", context={"session_id": "chat_test"})
        assert d.reason == RouteReason.USER_OVERRIDE
        assert d.model_name == "claude-sonnet-4-6"

    def test_clear_override_returns_to_primary(self):
        """Phase 3.5: clear_override takes the session_id whose override to drop."""
        r = _make_router()
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            r.set_override("chat_test", OVERRIDE_PRESETS["claude"])
            r.clear_override("chat_test")
        d = r.route("hello", context={"session_id": "chat_test"})
        assert d.reason == RouteReason.PRIMARY

    def test_has_override_property(self):
        """Phase 3.5: has_override is True iff ANY session has an override."""
        r = _make_router()
        assert not r.has_override
        r.set_override("chat_test", {"provider": "openai", "model": "gpt-4o"})
        assert r.has_override
        r.clear_override("chat_test")
        assert not r.has_override

    def test_smart_routing_simple_goes_to_simple_provider(self):
        r = _make_router(
            smart_routing_enabled=True,
            simple_provider={"provider": "llama_cpp", "base_url": "http://localhost:8080"},
        )
        d = r.route("hi")
        assert d.reason == RouteReason.SMART_SIMPLE

    def test_smart_routing_complex_stays_on_primary(self):
        r = _make_router(
            smart_routing_enabled=True,
            simple_provider={"provider": "llama_cpp", "base_url": "http://localhost:8080"},
        )
        d = r.route("refactor the authentication module to use JWT tokens")
        assert d.reason == RouteReason.PRIMARY

    def test_smart_routing_disabled_always_primary(self):
        r = _make_router(smart_routing_enabled=False)
        d = r.route("hi")
        assert d.reason == RouteReason.PRIMARY

    def test_escalation_on_retry_exhaustion(self):
        r = _make_router(
            escalation_enabled=True,
            escalation_provider={
                "provider": "anthropic",
                "api_key": "sk-test",
                "model": "claude-sonnet-4-6",
            },
        )
        d = r.route("fix this bug", context={"retry_count": 3})
        assert d.reason == RouteReason.ESCALATION
        assert d.model_name == "claude-sonnet-4-6"

    def test_escalation_subagent_flag(self):
        r = _make_router(
            escalation_enabled=True,
            escalation_as_subagent=True,
            escalation_provider={"provider": "anthropic", "api_key": "k", "model": "m"},
        )
        d = r.route("x", context={"retry_count": 5})
        assert d.use_subagent is True

    def test_escalation_disabled_stays_primary(self):
        r = _make_router(escalation_enabled=False)
        d = r.route("x", context={"retry_count": 10})
        assert d.reason == RouteReason.PRIMARY

    def test_no_escalation_config_stays_primary(self):
        r = _make_router(escalation_enabled=True, escalation_provider=None)
        d = r.route("x", context={"retry_count": 5})
        assert d.reason == RouteReason.PRIMARY

    def test_retry_under_threshold_stays_primary(self):
        r = _make_router(
            escalation_enabled=True,
            escalation_provider={"provider": "anthropic", "api_key": "k", "model": "m"},
        )
        d = r.route("hello", context={"retry_count": 2})
        assert d.reason == RouteReason.PRIMARY


# -- Fallback ---------------------------------------------------------------

class TestFallback:
    def test_get_fallback_returns_first_available(self):
        r = _make_router(
            fallback_chain=[{"provider": "llama_cpp", "base_url": "http://localhost:8080"}]
        )
        fb = r.get_fallback()
        assert fb is not None
        assert fb.reason == RouteReason.FALLBACK

    def test_get_fallback_empty_returns_none(self):
        r = _make_router(fallback_chain=[])
        assert r.get_fallback() is None


# -- Auxiliary ---------------------------------------------------------------

class TestAuxiliary:
    def test_auxiliary_with_config_uses_configured(self):
        r = _make_router(
            auxiliary_vision={"provider": "llama_cpp", "base_url": "http://localhost:8080"},
        )
        d = r.route_auxiliary("vision")
        assert d.reason == RouteReason.AUXILIARY
        assert d.provider is not r.primary_provider

    def test_auxiliary_without_config_uses_primary(self):
        r = _make_router()
        d = r.route_auxiliary("vision")
        assert d.reason == RouteReason.AUXILIARY
        assert d.provider is r.primary_provider

    def test_auxiliary_unknown_task_uses_primary(self):
        r = _make_router()
        d = r.route_auxiliary("nonexistent")
        assert d.provider is r.primary_provider


# -- Adapter auto-adjustment ------------------------------------------------

class TestAdapterAutoAdjust:
    def test_anthropic_gets_passthrough_none(self):
        from prometheus.adapter.formatter import AnthropicFormatter
        adapter = _build_adapter_for("anthropic")
        assert isinstance(adapter.formatter, AnthropicFormatter)
        assert adapter.validator.strictness.name == "NONE"

    def test_openai_gets_passthrough_none(self):
        from prometheus.adapter.formatter import PassthroughFormatter
        adapter = _build_adapter_for("openai")
        assert isinstance(adapter.formatter, PassthroughFormatter)
        assert adapter.validator.strictness.name == "NONE"

    def test_gemini_gets_passthrough_none(self):
        from prometheus.adapter.formatter import PassthroughFormatter
        adapter = _build_adapter_for("gemini")
        assert isinstance(adapter.formatter, PassthroughFormatter)

    def test_xai_gets_passthrough_none(self):
        from prometheus.adapter.formatter import PassthroughFormatter
        adapter = _build_adapter_for("xai")
        assert isinstance(adapter.formatter, PassthroughFormatter)

    def test_llama_cpp_gets_qwen_medium(self):
        from prometheus.adapter.formatter import QwenFormatter
        adapter = _build_adapter_for("llama_cpp")
        assert isinstance(adapter.formatter, QwenFormatter)
        assert adapter.validator.strictness.name == "MEDIUM"


# -- Config loading ----------------------------------------------------------

class TestConfigLoading:
    def test_load_empty_config(self):
        cfg = load_router_config({})
        assert cfg.fallback_chain == []
        assert cfg.smart_routing_enabled is False
        assert cfg.escalation_enabled is False

    def test_load_full_config(self):
        cfg = load_router_config({
            "router": {
                "fallback": [{"provider": "ollama", "model": "auto"}],
                "smart_routing": {"enabled": True, "max_simple_chars": 200},
                "escalation": {
                    "enabled": True,
                    "provider": {"provider": "anthropic", "model": "claude-sonnet-4-6"},
                    "as_subagent": False,
                },
                "auxiliary": {
                    "vision": {"provider": "openai", "model": "gpt-4o"},
                },
            }
        })
        assert len(cfg.fallback_chain) == 1
        assert cfg.smart_routing_enabled is True
        assert cfg.max_simple_chars == 200
        assert cfg.escalation_enabled is True
        assert cfg.escalation_as_subagent is False
        assert cfg.auxiliary_vision is not None


# -- Status ------------------------------------------------------------------

class TestStatus:
    def test_status_primary_only(self):
        r = _make_router()
        st = r.status()
        assert st["primary"] == "gemma4-26b"
        assert st["override"] is None
        assert st["smart_routing"] is False

    def test_status_with_override(self):
        """Phase 3.5: status(session_id=...) reports THAT session's override."""
        r = _make_router()
        r.set_override("chat_test", {"provider": "anthropic", "model": "claude-sonnet-4-6"})
        st = r.status(session_id="chat_test")
        assert st["override"] == "claude-sonnet-4-6"
        assert st["active_override_count"] == 1

    def test_status_without_session_id_hides_override_detail(self):
        """Phase 3.5: status() with no session_id returns override=None but
        still exposes the count so diagnostic commands can see overrides exist."""
        r = _make_router()
        r.set_override("chat_a", {"provider": "anthropic", "model": "claude-sonnet-4-6"})
        r.set_override("chat_b", {"provider": "openai", "model": "gpt-4o"})
        st = r.status()
        assert st["override"] is None
        assert st["active_override_count"] == 2

    def test_status_with_escalation(self):
        r = _make_router(
            escalation_enabled=True,
            escalation_provider={"provider": "anthropic", "model": "claude-sonnet-4-6"},
        )
        st = r.status()
        assert st["escalation"] == "claude-sonnet-4-6"


# -- RetryAction.ESCALATE ---------------------------------------------------

class TestRetryEscalation:
    def test_escalate_when_router_has_escalation(self):
        from prometheus.adapter.retry import RetryEngine, RetryAction

        router = _make_router(escalation_enabled=True)
        engine = RetryEngine(max_retries=2, router=router)

        # Exhaust retries
        engine.handle_failure("bash", "error1", None)
        engine.handle_failure("bash", "error2", None)
        action, msg = engine.handle_failure("bash", "error3", None)
        assert action == RetryAction.ESCALATE

    def test_abort_when_no_router(self):
        from prometheus.adapter.retry import RetryEngine, RetryAction

        engine = RetryEngine(max_retries=2)
        engine.handle_failure("bash", "e1", None)
        engine.handle_failure("bash", "e2", None)
        action, _ = engine.handle_failure("bash", "e3", None)
        assert action == RetryAction.ABORT

    def test_abort_when_escalation_disabled(self):
        from prometheus.adapter.retry import RetryEngine, RetryAction

        router = _make_router(escalation_enabled=False)
        engine = RetryEngine(max_retries=2, router=router)
        engine.handle_failure("bash", "e1", None)
        engine.handle_failure("bash", "e2", None)
        action, _ = engine.handle_failure("bash", "e3", None)
        assert action == RetryAction.ABORT


# -- Override presets --------------------------------------------------------

class TestOverridePresets:
    def test_claude_preset(self):
        assert OVERRIDE_PRESETS["claude"]["provider"] == "anthropic"

    def test_gpt_preset(self):
        assert OVERRIDE_PRESETS["gpt"]["provider"] == "openai"

    def test_gemini_preset(self):
        assert OVERRIDE_PRESETS["gemini"]["provider"] == "gemini"

    def test_xai_preset(self):
        assert OVERRIDE_PRESETS["xai"]["provider"] == "xai"


# -- TaskClassifier (absorbed from tests/test_router.py in Phase 2) ----------
#
# These verify TaskClassifier in isolation now that it lives in
# prometheus.router.model_router. Phase 1.5's tests in test_wiring.py cover
# the classifier's INTEGRATION into route(); these cover its classification
# behavior per-category.


class TestTaskClassifier:
    """Test token-based task classification (relocated from adapter/router.py)."""

    def test_code_generation(self):
        c = TaskClassifier()
        result = c.classify("Write a Python function to parse JSON")
        assert result.task_type == TaskType.CODE_GENERATION
        assert result.confidence > 0.3
        assert "python" in result.matched_tokens or "write" in result.matched_tokens

    def test_quick_answer(self):
        c = TaskClassifier()
        result = c.classify("What is the capital of France?")
        assert result.task_type == TaskType.QUICK_ANSWER

    def test_reasoning(self):
        c = TaskClassifier()
        result = c.classify(
            "Explain the trade-offs between microservices and monoliths"
        )
        assert result.task_type == TaskType.REASONING
        assert "tradeoffs" in result.matched_tokens or "explain" in result.matched_tokens

    def test_tool_heavy(self):
        c = TaskClassifier()
        result = c.classify(
            "Search for recent news about AI", tool_mentions=["web_search"]
        )
        assert result.task_type == TaskType.TOOL_HEAVY

    def test_creative(self):
        c = TaskClassifier()
        result = c.classify("Write a short story about a robot learning to love")
        # "write" overlaps CODE_GENERATION and "story" overlaps CREATIVE
        assert result.task_type in (TaskType.CREATIVE, TaskType.CODE_GENERATION)

    def test_short_message_boosts_quick(self):
        c = TaskClassifier()
        result = c.classify("hi")
        # Short messages should boost QUICK_ANSWER
        assert result.task_type == TaskType.QUICK_ANSWER or result.confidence < 0.5

    def test_code_block_boosts_code(self):
        c = TaskClassifier()
        result = c.classify("Fix this: ```python\nprint('hello')\n```")
        assert result.task_type == TaskType.CODE_GENERATION

    def test_empty_message(self):
        c = TaskClassifier()
        result = c.classify("")
        # Empty message triggers short-message boost → QUICK_ANSWER
        assert result.task_type == TaskType.QUICK_ANSWER

    def test_classification_returns_reason(self):
        c = TaskClassifier()
        result = c.classify("Explain why Python is popular")
        assert "tokens=" in result.reason
        assert "len=" in result.reason
        assert "conf=" in result.reason
