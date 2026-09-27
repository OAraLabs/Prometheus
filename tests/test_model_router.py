"""Tests for the Sprint 23 Model Router.

Tests classification, routing decisions, fallback, escalation,
user overrides, adapter auto-adjustment, and config loading.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prometheus.providers.registry import ProviderRegistry
from prometheus.router.model_router import (
    ModelRouter,
    RouteDecision,
    RouteReason,
    RouterConfig,
    RoutingRule,
    OVERRIDE_PRESETS,
    TaskClassifier,
    TaskType,
    _build_adapter_for,
    load_router_config,
    read_router_config,
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
        # Preset default is Haiku 4.5 (Phase 4 pilot choice — fast + cheap).
        assert d.model_name == OVERRIDE_PRESETS["claude"]["model"]

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


# The live daemon's chain, by shape: a llama.cpp primary, then an Ollama box,
# then Anthropic. The hosts are placeholders.
LIVE_SHAPED_CHAIN = [
    {"provider": "ollama", "base_url": "http://ollama-box:11434", "model": "qwen3.5:9b"},
    {"provider": "anthropic", "model": "claude-sonnet-4-5"},
]


def _live_shaped_loop(monkeypatch, chain=LIVE_SHAPED_CHAIN):
    """A real primary, a real router and the loop's own context, so the name
    the router is told is the name the loop reads off a real provider."""
    from prometheus.engine.agent_loop import LoopContext

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    primary = ProviderRegistry.create({"provider": "llama_cpp", "base_url": "http://primary:8080"})
    router = ModelRouter(
        config=load_router_config({"router": {"fallback": chain}}),
        primary_provider=primary,
        primary_adapter=MagicMock(),
        primary_model="qwen3.8-27b",
    )
    ctx = LoopContext(
        provider=primary, model="qwen3.8-27b", system_prompt="test",
        max_tokens=256, model_router=router, adapter=MagicMock(),
    )
    return router, ctx


class TestFallbackNeverReturnsTheFailedProvider:
    """The circuit breaker's model switch walks router.fallback, and so does the
    hook contract's recovery from a failed pick. Each hop must leave the
    provider that just failed."""

    def test_second_hop_leaves_the_provider_that_just_failed(self, monkeypatch):
        # What the loop does on each formatting-error trip: ask, then swap in
        # whatever came back. On origin/main the second ask handed back the
        # Ollama box that had just failed, because the router ignored the name
        # and the loop called that box "llama_cpp" anyway.
        from prometheus.engine.agent_loop import _try_model_fallback
        from prometheus.providers.anthropic import AnthropicProvider
        from prometheus.providers.ollama import OllamaProvider

        _, ctx = _live_shaped_loop(monkeypatch)

        first = _try_model_fallback(ctx)
        assert isinstance(first.provider, OllamaProvider)
        ctx.provider = first.provider

        second = _try_model_fallback(ctx)
        assert second is not None
        assert second.provider is not first.provider
        assert isinstance(second.provider, AnthropicProvider)
        assert second.provider_name == "anthropic"

    def test_an_entry_of_the_failed_provider_is_skipped(self):
        r = _make_router(fallback_chain=[
            {"provider": "ollama", "base_url": "http://ollama-box:11434", "model": "qwen3.5:9b"},
            {"provider": "llama_cpp", "base_url": "http://other-box:8080"},
        ])

        d = r.get_fallback("ollama")

        assert d is not None and d.provider_name == "llama_cpp"

    def test_a_chain_that_runs_out_says_so_in_a_warning(self, monkeypatch, caplog):
        from prometheus.engine.agent_loop import _try_model_fallback

        _, ctx = _live_shaped_loop(monkeypatch, chain=LIVE_SHAPED_CHAIN[:1])
        ctx.provider = _try_model_fallback(ctx).provider

        with caplog.at_level(logging.WARNING, logger="prometheus.router.model_router"):
            assert _try_model_fallback(ctx) is None

        [message] = _router_warnings(caplog).splitlines()
        assert "no fallback left" in message
        # Names the provider that failed, and why the one entry was passed over.
        assert "after ollama failed" in message
        assert "ollama/qwen3.5:9b: skipped, ollama is what failed" in message

    def test_an_entry_that_cannot_be_built_is_named_in_the_warning(self, monkeypatch, caplog):
        monkeypatch.delenv("ROUTER_TEST_UNSET_KEY", raising=False)
        r = _make_router(fallback_chain=[
            {"provider": "anthropic", "model": "claude-sonnet-4-5",
             "api_key_env": "ROUTER_TEST_UNSET_KEY"},
        ])

        with caplog.at_level(logging.WARNING, logger="prometheus.router.model_router"):
            assert r.get_fallback("llama_cpp") is None

        text = "\n".join(rec.getMessage() for rec in caplog.records)
        assert "anthropic/claude-sonnet-4-5: could not be built" in text
        assert "ROUTER_TEST_UNSET_KEY is not set" in text

    def test_a_failed_build_never_logs_the_entrys_key(self, monkeypatch, caplog):
        # The entry is logged by provider and model. It used to be logged
        # whole, api_key included, at DEBUG.
        def refuse(cfg):
            raise ValueError("endpoint refused")

        monkeypatch.setattr(ProviderRegistry, "create", staticmethod(refuse))
        r = _make_router(fallback_chain=[
            {"provider": "openai", "api_key": "sk-router-test-secret", "model": "gpt-5.6-luna"},
        ])

        with caplog.at_level(logging.DEBUG, logger="prometheus.router.model_router"):
            assert r.get_fallback("llama_cpp") is None

        assert caplog.records
        assert not any("sk-router-test-secret" in rec.getMessage() for rec in caplog.records)

    @pytest.mark.parametrize("name", ProviderRegistry.list_providers())
    def test_every_provider_names_itself_by_the_key_it_was_built_from(self, name, monkeypatch):
        # The loop tells the router which provider failed by reading this
        # attribute (agent_loop._try_model_fallback), defaulting to "llama_cpp"
        # when it is missing. Only the OpenAI-compatible class carried it, so an
        # Anthropic or Ollama failure was reported as a llama.cpp one.
        from prometheus.providers import xai_oauth
        from prometheus.providers.registry import CLOUD_DEFAULTS

        monkeypatch.setattr(xai_oauth, "is_logged_in", lambda: False)
        env = CLOUD_DEFAULTS.get(name, {}).get("default_env")
        if env:
            monkeypatch.setenv(env, "test-key")

        provider = ProviderRegistry.create({"provider": name})

        assert provider.provider_name == name


# -- Task-type rule provider cache -------------------------------------------

CODE_MSG = "write a python function to parse json"
REASONING_MSG = "explain the tradeoffs of immutable data structures in depth"
# What YAML builds from `0x` + 4000 hex digits: the loader takes it, but
# repr() and str() refuse it (past Python's 4300-digit conversion limit).
HUGE_INT = 1 << 16000


class TestTaskRuleProviderCache:
    """Real providers, not a mocked registry: what matters is the host each
    decision's provider will actually send to."""

    @pytest.fixture(autouse=True)
    def _huge_int_stays_unprintable(self, caplog):
        # HUGE_INT tests nothing once PYTHONINTMAXSTRDIGITS=0 lifts the limit,
        # so pin the default. And keep the router's DEBUG traceback out of the
        # capture: the router's own lines name a rule by index, never by value.
        old = sys.get_int_max_str_digits()
        sys.set_int_max_str_digits(4300)
        caplog.set_level(logging.INFO, logger="prometheus.router.model_router")
        yield
        sys.set_int_max_str_digits(old)

    def test_same_model_on_two_hosts_reaches_each_host(self):
        # Two llama.cpp boxes serving one model. Keyed on "provider:model",
        # the second rule was handed the provider built for the first box.
        r = _make_router(task_rules=[
            RoutingRule(TaskType.CODE_GENERATION, "llama_cpp", "qwen3.8-27b",
                        base_url="http://gpu-a:8080"),
            RoutingRule(TaskType.REASONING, "llama_cpp", "qwen3.8-27b",
                        base_url="http://gpu-b:8080"),
        ])
        code = r.route(CODE_MSG)
        reasoning = r.route(REASONING_MSG)

        assert code.reason == reasoning.reason == RouteReason.TASK_RULE
        assert code.provider._base_url == "http://gpu-a:8080"
        assert reasoning.provider._base_url == "http://gpu-b:8080"

    def test_rules_naming_one_endpoint_share_one_provider(self):
        # The other direction: the key must not grow fields that do not
        # change the endpoint (task_type here), or the cache stops caching.
        r = _make_router(task_rules=[
            RoutingRule(TaskType.CODE_GENERATION, "llama_cpp", "qwen3.8-27b",
                        base_url="http://gpu-a:8080"),
            RoutingRule(TaskType.REASONING, "llama_cpp", "qwen3.8-27b",
                        base_url="http://gpu-a:8080"),
        ])
        code, reasoning = r.route(CODE_MSG), r.route(REASONING_MSG)

        assert code.reason == reasoning.reason == RouteReason.TASK_RULE
        assert code.provider is reasoning.provider

    def test_same_host_different_model_gets_its_own_provider(self):
        # vllm, not llama_cpp: llama.cpp ignores the model, vllm sends it.
        r = _make_router(task_rules=[
            RoutingRule(TaskType.CODE_GENERATION, "vllm", "model-a",
                        base_url="http://gpu-a:8000/v1"),
            RoutingRule(TaskType.REASONING, "vllm", "model-b",
                        base_url="http://gpu-a:8000/v1"),
        ])

        assert r.route(CODE_MSG).provider._model == "model-a"
        assert r.route(REASONING_MSG).provider._model == "model-b"

    def test_same_host_different_provider_gets_its_own_provider(self):
        from prometheus.providers.llama_cpp import LlamaCppProvider
        from prometheus.providers.openai_compat import OpenAICompatProvider

        r = _make_router(task_rules=[
            RoutingRule(TaskType.CODE_GENERATION, "llama_cpp", "qwen3.8-27b",
                        base_url="http://gpu-a:8080"),
            RoutingRule(TaskType.REASONING, "vllm", "qwen3.8-27b",
                        base_url="http://gpu-a:8080"),
        ])

        assert isinstance(r.route(CODE_MSG).provider, LlamaCppProvider)
        assert isinstance(r.route(REASONING_MSG).provider, OpenAICompatProvider)

    @pytest.mark.parametrize("base_url", [None, ""])
    def test_rule_without_base_url_reaches_the_default_host(self, base_url):
        # base_url stays out of the config when unset: passing it through
        # would fail the llama.cpp build (None) or aim it at "" instead.
        r = _make_router(task_rules=[
            RoutingRule(TaskType.CODE_GENERATION, "llama_cpp", "qwen3.8-27b",
                        base_url=base_url),
        ])
        decision = r.route(CODE_MSG)

        assert decision.reason == RouteReason.TASK_RULE
        assert decision.provider._base_url == "http://localhost:8080"

    @pytest.mark.parametrize("malformed", [
        {"base_url": ["http://gpu-a:8080"]},
        {"base_url": {"host": "gpu-a", "port": 8080}},
        {"base_url": HUGE_INT},
        {"provider": ["llama_cpp"]},
    ])
    def test_malformed_rule_still_falls_through_to_primary(self, malformed):
        # YAML can put a list, a map or an unprintable int where a string
        # belongs. Building that provider fails, and a rule whose provider
        # cannot be built falls through to the next branch. A key made from
        # the raw values raised (unhashable type: 'list') out of route() before
        # it got that far, and a repr() key made outside the try raised on
        # HUGE_INT.
        fields = {"provider": "llama_cpp", "model": "qwen3.8-27b", **malformed}
        r = _make_router(task_rules=[RoutingRule(TaskType.CODE_GENERATION, **fields)])

        decision = r.route(CODE_MSG)

        assert decision.reason == RouteReason.PRIMARY
        assert decision.provider is r.primary_provider

    @pytest.mark.parametrize("field", ["provider", "model"])
    def test_unprintable_rule_field_falls_through_instead_of_raising(self, field):
        # origin/main raised here: its f-string key printed provider and model
        # outside the try. A rule that cannot be keyed now falls through like
        # one that cannot be built. For the model, llama.cpp would otherwise
        # build fine and route a turn whose model cannot be printed.
        fields = {"provider": "llama_cpp", "model": "qwen3.8-27b", field: HUGE_INT}
        r = _make_router(task_rules=[RoutingRule(TaskType.CODE_GENERATION, **fields)])

        decision = r.route(CODE_MSG)

        assert decision.reason == RouteReason.PRIMARY
        assert decision.provider is r.primary_provider


# -- Rule and fallback fields ------------------------------------------------

class _RecordingConfig(dict):
    """A provider config that remembers every key read from it."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read: set[str] = set()

    def get(self, key, default=None):
        self.read.add(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)

    def __contains__(self, key):
        self.read.add(key)
        return super().__contains__(key)


def _router_from_yaml(router_section: dict) -> ModelRouter:
    return ModelRouter(
        config=load_router_config({"router": router_section}),
        primary_provider=MagicMock(),
        primary_adapter=MagicMock(),
        primary_model="gemma4-26b",
    )


def _router_warnings(caplog) -> str:
    return "\n".join(
        rec.getMessage() for rec in caplog.records
        if rec.name == "prometheus.router.model_router" and rec.levelno >= logging.WARNING
    )


class TestRuleAndFallbackFields:
    """Every field a rule or fallback entry carries is either honoured or
    refused at load, loudly. None is dropped."""

    @pytest.mark.parametrize("name", ProviderRegistry.list_providers())
    def test_the_declared_keys_are_exactly_the_keys_create_reads(self, name, monkeypatch):
        # Load-time refusal trusts this table, so it is pinned to the factory
        # itself: a key create() starts reading must be declared, or a config
        # using it is refused; a declared key create() stops reading would be
        # accepted and dropped.
        from prometheus.providers import xai_oauth
        from prometheus.providers.registry import CLOUD_DEFAULTS, PROVIDER_CONFIG_KEYS

        assert set(PROVIDER_CONFIG_KEYS) == set(ProviderRegistry.list_providers())
        monkeypatch.setattr(xai_oauth, "is_logged_in", lambda: False)
        env = CLOUD_DEFAULTS.get(name, {}).get("default_env")
        if env:
            monkeypatch.setenv(env, "test-key")
        cfg = _RecordingConfig(provider=name)

        ProviderRegistry.create(cfg)

        assert cfg.read == PROVIDER_CONFIG_KEYS[name]

    def test_a_rules_api_key_env_reaches_its_provider(self, monkeypatch):
        # origin/main built the rule's provider from provider, model and
        # base_url only, so it quietly used OPENAI_API_KEY.
        monkeypatch.setenv("OPENAI_API_KEY", "the-default-key")
        monkeypatch.setenv("ROUTER_TEST_RULE_KEY", "the-rules-key")
        r = _router_from_yaml({"rules": [{
            "task_type": "code_generation", "provider": "openai",
            "model": "gpt-5.6-luna", "api_key_env": "ROUTER_TEST_RULE_KEY",
        }]})

        decision = r.route(CODE_MSG)

        assert decision.reason == RouteReason.TASK_RULE
        assert decision.provider._api_key == "the-rules-key"

    def test_every_other_rule_field_reaches_its_provider(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("ROUTER_TEST_BASE_URL", "http://rule-host:9000/v1")
        r = _router_from_yaml({"rules": [{
            "task_type": "code_generation", "provider": "openai", "model": "gpt-5.6-luna",
            "base_url_env": "ROUTER_TEST_BASE_URL", "timeout": 7.5,
            "max_tokens": 1234, "vision": True,
        }]})

        provider = r.route(CODE_MSG).provider

        assert provider._base_url == "http://rule-host:9000/v1"
        assert provider._timeout == 7.5
        assert provider._default_max_tokens == 1234
        assert provider.supports_vision is True

    def test_rules_differing_only_in_key_get_their_own_providers(self, monkeypatch):
        # The cache key is the whole provider config (#569), so a newly
        # honoured field separates providers without anyone remembering to.
        monkeypatch.setenv("ROUTER_TEST_KEY_A", "key-a")
        monkeypatch.setenv("ROUTER_TEST_KEY_B", "key-b")
        r = _router_from_yaml({"rules": [
            {"task_type": "code_generation", "provider": "openai",
             "model": "gpt-5.6-luna", "api_key_env": "ROUTER_TEST_KEY_A"},
            {"task_type": "reasoning", "provider": "openai",
             "model": "gpt-5.6-luna", "api_key_env": "ROUTER_TEST_KEY_B"},
        ]})

        assert r.route(CODE_MSG).provider._api_key == "key-a"
        assert r.route(REASONING_MSG).provider._api_key == "key-b"

    def test_a_llama_cpp_rules_suppress_thinking_reaches_its_provider(self):
        r = _router_from_yaml({"rules": [{
            "task_type": "code_generation", "provider": "llama_cpp",
            "model": "qwen3.8-27b", "suppress_thinking": False,
        }]})

        assert r.route(CODE_MSG).provider._suppress_thinking is False

    @pytest.mark.parametrize("section", ["rules", "fallback"])
    def test_a_misspelt_key_is_refused_at_load(self, section, caplog):
        entry = {"provider": "openai", "model": "gpt-5.6-luna", "api_key_evn": "MY_KEY"}
        if section == "rules":
            entry["task_type"] = "code_generation"

        with caplog.at_level(logging.WARNING, logger="prometheus.router.model_router"):
            cfg = load_router_config({"router": {section: [entry]}})

        assert (cfg.task_rules if section == "rules" else cfg.fallback_chain) == []
        warnings = _router_warnings(caplog)
        assert f"router.{section}[0]" in warnings
        assert "api_key_evn" in warnings

    @pytest.mark.parametrize("section", ["rules", "fallback"])
    def test_a_key_its_provider_never_reads_is_refused_at_load(self, section, caplog):
        # api_key_env is real, but llama.cpp sends no key, so a llama.cpp
        # entry carrying one would run unauthenticated without a word.
        entry = {"provider": "llama_cpp", "model": "qwen3.8-27b", "api_key_env": "MY_KEY"}
        if section == "rules":
            entry["task_type"] = "code_generation"

        with caplog.at_level(logging.WARNING, logger="prometheus.router.model_router"):
            cfg = load_router_config({"router": {section: [entry]}})

        assert (cfg.task_rules if section == "rules" else cfg.fallback_chain) == []
        warnings = _router_warnings(caplog)
        assert "api_key_env" in warnings
        assert "llama_cpp" in warnings

    @pytest.mark.parametrize("section", ["rules", "fallback"])
    def test_an_unknown_provider_is_refused_at_load(self, section, caplog):
        entry = {"provider": "openia", "model": "gpt-5.6-luna"}
        if section == "rules":
            entry["task_type"] = "code_generation"

        with caplog.at_level(logging.WARNING, logger="prometheus.router.model_router"):
            cfg = load_router_config({"router": {section: [entry]}})

        assert (cfg.task_rules if section == "rules" else cfg.fallback_chain) == []
        assert "openia" in _router_warnings(caplog)

    def test_a_rule_without_a_provider_is_refused_for_that(self, caplog):
        # Not for its api_key_env, which is what the llama.cpp default would
        # have said.
        with caplog.at_level(logging.WARNING, logger="prometheus.router.model_router"):
            cfg = load_router_config({"router": {"rules": [
                {"task_type": "code_generation", "model": "m", "api_key_env": "MY_KEY"},
            ]}})

        assert cfg.task_rules == []
        assert _router_warnings(caplog) == "router.rules[0] ?/m refused: missing provider"

    def test_a_rule_whose_key_is_not_set_says_so_once(self, monkeypatch, caplog):
        # Honouring api_key_env means an unset variable now fails the build
        # instead of borrowing the default key. That must not be quiet either.
        monkeypatch.delenv("ROUTER_TEST_UNSET_KEY", raising=False)
        r = _router_from_yaml({"rules": [{
            "task_type": "code_generation", "provider": "openai",
            "model": "gpt-5.6-luna", "api_key_env": "ROUTER_TEST_UNSET_KEY",
        }]})

        with caplog.at_level(logging.WARNING, logger="prometheus.router.model_router"):
            first, second = r.route(CODE_MSG), r.route(CODE_MSG)

        assert first.reason == second.reason == RouteReason.PRIMARY
        warnings = _router_warnings(caplog).splitlines()
        assert len(warnings) == 1
        assert "router.rules[0] (code_generation) cannot be built" in warnings[0]
        assert "ROUTER_TEST_UNSET_KEY is not set" in warnings[0]

    def test_a_fallback_entry_that_is_not_a_mapping_is_refused_at_load(self, caplog):
        with caplog.at_level(logging.WARNING, logger="prometheus.router.model_router"):
            cfg = load_router_config({"router": {"fallback": ["anthropic"]}})

        assert cfg.fallback_chain == []
        assert "router.fallback[0]" in _router_warnings(caplog)

    def test_a_fallback_entry_without_a_provider_is_refused_at_load(self, caplog):
        # It used to build a llama.cpp provider on localhost:8080 by default.
        with caplog.at_level(logging.WARNING, logger="prometheus.router.model_router"):
            cfg = load_router_config({"router": {"fallback": [
                {"base_url": "http://ollama-box:11434", "model": "qwen3.5:9b"},
            ]}})

        assert cfg.fallback_chain == []
        assert _router_warnings(caplog) == (
            "router.fallback[0] ?/qwen3.5:9b refused: missing provider")

    def test_each_refused_entry_is_one_warning_naming_it_and_its_key(self, caplog):
        with caplog.at_level(logging.WARNING, logger="prometheus.router.model_router"):
            cfg = load_router_config({"router": {
                "fallback": [
                    LIVE_SHAPED_CHAIN[0],
                    {"provider": "llama_cpp", "model": "q", "api_key_env": "K"},
                ],
                "rules": [
                    {"task_type": "reasoning", "provider": "openai", "model": "g",
                     "api_key_evn": "K", "timout": 5},
                ],
            }})

        assert cfg.fallback_chain == [LIVE_SHAPED_CHAIN[0]]
        fallback_line, rule_line = _router_warnings(caplog).splitlines()
        assert fallback_line.startswith("router.fallback[1] llama_cpp/q refused: api_key_env is not read")
        assert rule_line.startswith("router.rules[0] openai/g refused: api_key_evn, timout are not read")
        assert cfg.problems == [fallback_line, rule_line]

    def test_read_router_config_logs_nothing(self, caplog):
        # /doctor reads the same parse; it must not re-log the boot WARNINGs.
        with caplog.at_level(logging.DEBUG, logger="prometheus.router.model_router"):
            cfg = read_router_config({"router": {"fallback": [{"provider": "x"}]}})

        assert cfg.problems == ["router.fallback[0] x/? refused: unknown provider 'x'"]
        assert not caplog.records

    @pytest.mark.parametrize("section", ["smart_routing", "escalation", "overrides"])
    def test_an_empty_router_subsection_does_not_break_boot(self, section):
        # `smart_routing:` with nothing under it is YAML null, and .get on it
        # raised out of load_router_config — which the daemon calls at boot.
        cfg = load_router_config({"router": {section: None}})

        assert cfg.problems == []

    def test_a_malformed_name_cannot_take_the_load_down(self, caplog):
        # A label that formats a 4300-digit int raises; the load must not.
        with caplog.at_level(logging.INFO, logger="prometheus.router.model_router"):
            cfg = load_router_config({"router": {"rules": [
                {"task_type": "code_generation", "provider": "llama_cpp", "model": HUGE_INT},
            ], "fallback": [{"provider": "llama_cpp", "model": HUGE_INT, "bogus": 1}]}})

        assert len(cfg.task_rules) == 1
        assert "router.fallback[0] llama_cpp/<int> refused" in _router_warnings(caplog)

    def test_a_refusal_never_logs_a_value(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="prometheus.router.model_router"):
            load_router_config({"router": {
                "rules": [{"task_type": "code_generation", "provider": "openai",
                           "model": "m", "api_key": "sk-router-test-secret", "bogus": 1}],
                "fallback": [{"provider": "openai", "model": "m",
                              "api_key": "sk-router-test-secret", "bogus": 1}],
            }})

        assert caplog.records
        assert not any("sk-router-test-secret" in rec.getMessage() for rec in caplog.records)

    def test_the_live_shaped_chain_loads_unchanged_and_quietly(self, caplog):
        with caplog.at_level(logging.WARNING, logger="prometheus.router.model_router"):
            cfg = load_router_config({"router": {"fallback": LIVE_SHAPED_CHAIN}})

        assert cfg.fallback_chain == LIVE_SHAPED_CHAIN
        assert _router_warnings(caplog) == ""

    @pytest.mark.parametrize("section", ["rules", "fallback"])
    def test_an_empty_section_loads_as_empty(self, section):
        # `fallback:` with no value is YAML null; iterating it crashed the
        # router's constructor.
        cfg = load_router_config({"router": {section: None}})
        _router_from_yaml({section: None})

        assert (cfg.task_rules if section == "rules" else cfg.fallback_chain) == []


# -- router.auxiliary (removed) ----------------------------------------------

class TestAuxiliaryRemoved:
    """route_auxiliary had no caller from the first commit on, so a provider
    configured under router.auxiliary was never used."""

    def test_a_configured_auxiliary_block_is_refused_loudly(self, caplog):
        with caplog.at_level(logging.WARNING, logger="prometheus.router.model_router"):
            load_router_config({"router": {"auxiliary": {
                "vision": {"provider": "openai", "model": "gpt-4o"},
            }}})

        assert "router.auxiliary: config key is deprecated" in _router_warnings(caplog)

    def test_the_router_has_no_auxiliary_route(self):
        assert not hasattr(ModelRouter, "route_auxiliary")
        assert "AUXILIARY" not in RouteReason.__members__


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
            }
        })
        assert len(cfg.fallback_chain) == 1
        assert cfg.smart_routing_enabled is True
        assert cfg.max_simple_chars == 200
        assert cfg.escalation_enabled is True
        assert cfg.escalation_as_subagent is False


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


# -- Per-slash-command config resolution (PR #18) ---------------------------
#
# `resolve_slash_command_target` is the new config-aware layer over
# OVERRIDE_PRESETS. Each handler call goes through it so users can pin
# specific models per slash command via prometheus.yaml.

class TestResolveSlashCommandTarget:
    def setup_method(self) -> None:
        # The fallback-warned set is module-global so we get one WARN per
        # process per command. Clear it between tests so each test sees the
        # warn behavior in isolation.
        from prometheus.router import model_router as mr
        mr._FALLBACK_WARNED.clear()

    def test_returns_user_config_when_set(self):
        """slash_commands.claude.model in config overrides the preset."""
        from prometheus.router.model_router import resolve_slash_command_target
        cfg = {
            "slash_commands": {
                "claude": {
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-5",
                    "api_key_env": "ANTHROPIC_API_KEY",
                },
            },
        }
        target = resolve_slash_command_target("claude", cfg)
        assert target["provider"] == "anthropic"
        assert target["model"] == "claude-sonnet-4-5"
        assert target["api_key_env"] == "ANTHROPIC_API_KEY"

    def test_falls_back_to_preset_when_config_missing(self, caplog):
        """No slash_commands section at all → preset default + WARN once."""
        from prometheus.router.model_router import (
            OVERRIDE_PRESETS,
            resolve_slash_command_target,
        )
        import logging

        caplog.set_level(logging.WARNING, logger="prometheus.router.model_router")
        target = resolve_slash_command_target("claude", {})
        assert target == OVERRIDE_PRESETS["claude"]
        # Exactly one WARN naming the missing section.
        warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("slash_commands.claude" in r.message for r in warn_records)

    def test_falls_back_when_specific_command_missing(self):
        """slash_commands section exists but doesn't include this command."""
        from prometheus.router.model_router import (
            OVERRIDE_PRESETS,
            resolve_slash_command_target,
        )
        cfg = {"slash_commands": {"gpt": {"model": "gpt-4o"}}}
        target = resolve_slash_command_target("claude", cfg)
        assert target == OVERRIDE_PRESETS["claude"]

    def test_partial_override_keeps_preset_defaults(self):
        """User overrides only ``model`` → preset's provider + api_key_env preserved."""
        from prometheus.router.model_router import resolve_slash_command_target
        cfg = {"slash_commands": {"claude": {"model": "claude-sonnet-4-5"}}}
        target = resolve_slash_command_target("claude", cfg)
        assert target["model"] == "claude-sonnet-4-5"
        assert target["provider"] == "anthropic"  # from preset
        assert target["api_key_env"] == "ANTHROPIC_API_KEY"  # from preset

    def test_unknown_command_returns_none(self):
        """A name not in OVERRIDE_PRESETS yields None — caller handles."""
        from prometheus.router.model_router import resolve_slash_command_target
        assert resolve_slash_command_target("definitely-not-real", {}) is None

    def test_base_url_override_is_merged(self):
        """slash_commands.<cmd>.base_url reaches the resolved target.

        Several providers front more than one host behind one config name —
        Alibaba pay-as-you-go vs its subscription-plan endpoints, Moonshot
        .ai vs .cn, GLM z.ai vs bigmodel.cn. Picking between them used to
        require editing CLOUD_DEFAULTS.
        """
        from prometheus.router.model_router import resolve_slash_command_target
        plan_url = (
            "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
        )
        cfg = {"slash_commands": {"qwen": {"base_url": plan_url}}}
        target = resolve_slash_command_target("qwen", cfg)
        assert target["base_url"] == plan_url
        assert target["provider"] == "qwen"          # from preset
        assert target["api_key_env"] == "QWEN_API_KEY"  # from preset

    def test_base_url_absent_when_not_configured(self):
        """No user base_url → the key stays absent so CLOUD_DEFAULTS wins.

        ProviderRegistry.create falls back to CLOUD_DEFAULTS only when the
        config has no base_url; emitting an empty string here would pin every
        provider to "".
        """
        from prometheus.router.model_router import resolve_slash_command_target
        target = resolve_slash_command_target("qwen", {"slash_commands": {"qwen": {}}})
        assert "base_url" not in target

    def test_warn_emitted_only_once_per_command(self, caplog):
        """First fallback emits WARN; subsequent calls for same command stay quiet."""
        from prometheus.router.model_router import resolve_slash_command_target
        import logging

        caplog.set_level(logging.WARNING, logger="prometheus.router.model_router")
        resolve_slash_command_target("claude", {})
        first_count = sum(
            1
            for r in caplog.records
            if "slash_commands.claude" in r.message and r.levelno == logging.WARNING
        )
        resolve_slash_command_target("claude", {})
        resolve_slash_command_target("claude", {})
        second_count = sum(
            1
            for r in caplog.records
            if "slash_commands.claude" in r.message and r.levelno == logging.WARNING
        )
        assert first_count == 1
        assert second_count == 1  # still 1 — not 3

    def test_resolves_each_known_command(self):
        """All four /claude /gpt /gemini /xai resolve to their provider preset by default."""
        from prometheus.router.model_router import resolve_slash_command_target
        assert resolve_slash_command_target("claude", {})["provider"] == "anthropic"
        assert resolve_slash_command_target("gpt", {})["provider"] == "openai"
        assert resolve_slash_command_target("gemini", {})["provider"] == "gemini"
        assert resolve_slash_command_target("xai", {})["provider"] == "xai"


class TestLogSlashCommandWiring:
    def setup_method(self) -> None:
        from prometheus.router import model_router as mr
        mr._FALLBACK_WARNED.clear()

    def test_logs_one_info_per_command(self, caplog):
        """Startup emits one INFO line per known slash command."""
        from prometheus.router.model_router import (
            SLASH_COMMAND_NAMES,
            log_slash_command_wiring,
        )
        import logging

        cfg = {
            "slash_commands": {
                "claude": {"model": "claude-sonnet-4-5"},
                "gpt": {"model": "gpt-4o"},
                "gemini": {"model": "gemini-2.5-pro"},
                "xai": {"model": "grok-3"},
            },
        }
        caplog.set_level(logging.INFO, logger="prometheus.router.model_router")
        log_slash_command_wiring(cfg)
        info_msgs = [
            r.message
            for r in caplog.records
            if r.levelno == logging.INFO and "slash_commands." in r.message
        ]
        # One line per known command, each showing provider / model.
        assert len(info_msgs) == len(SLASH_COMMAND_NAMES)
        joined = "\n".join(info_msgs)
        assert "claude" in joined
        assert "claude-sonnet-4-5" in joined
        assert "gpt-4o" in joined
        assert "gemini-2.5-pro" in joined
        assert "grok-3" in joined

    def test_logs_fall_back_to_preset_when_unconfigured(self, caplog):
        """No slash_commands section → INFO lines still emitted with preset defaults."""
        from prometheus.router.model_router import (
            OVERRIDE_PRESETS,
            log_slash_command_wiring,
        )
        import logging

        caplog.set_level(logging.INFO, logger="prometheus.router.model_router")
        log_slash_command_wiring({})
        info_msgs = [
            r.message
            for r in caplog.records
            if r.levelno == logging.INFO and "slash_commands." in r.message
        ]
        joined = "\n".join(info_msgs)
        assert OVERRIDE_PRESETS["claude"]["model"] in joined
        assert OVERRIDE_PRESETS["gpt"]["model"] in joined


# ---------------------------------------------------------------------------
# Per-provider model choices — the flattened catalog + `/qwen <model>` arg
# ---------------------------------------------------------------------------


class TestModelChoices:
    """resolve_model_choices / split_model_key / resolve_model_target."""

    def setup_method(self):
        from prometheus.router import model_router as mr
        mr._FALLBACK_WARNED.clear()

    def test_default_model_is_always_first(self):
        """Selecting the bare key and its default from the list must agree."""
        from prometheus.router.model_router import (
            OVERRIDE_PRESETS,
            resolve_model_choices,
        )
        # SHAPE, not value (#533): the invariant is that the list LEADS with
        # whatever the preset defaults to. Chaining a literal onto the end made
        # every model refresh edit this line while protecting nothing — the
        # literal cannot fail unless one of the two sides it sits between
        # already has.
        for key in OVERRIDE_PRESETS:
            choices = resolve_model_choices(key, {})
            assert choices and choices[0] == OVERRIDE_PRESETS[key]["model"], (
                f"resolve_model_choices({key!r})[0] is {choices[:1]!r} but the "
                f"preset defaults to {OVERRIDE_PRESETS[key]['model']!r} — "
                f"selecting the bare key and selecting its default from the "
                f"list would give different models."
            )

    def test_preset_without_choices_offers_just_its_default(self):
        """A preset with no PRESET_MODEL_CHOICES entry offers exactly one model.

        Was ``resolve_model_choices("glm", {}) == ("glm-5.2",)`` (#533). That
        spelling broke twice over: it made a data refresh a code change, and
        when glm GAINED a choices list the assertion became false for a reason
        that had nothing to do with the behaviour it was guarding. The rule is
        about the ABSENCE of an entry, so it now finds a preset that actually
        has none rather than naming one that happened to.
        """
        from prometheus.router.model_router import (
            OVERRIDE_PRESETS,
            PRESET_MODEL_CHOICES,
            resolve_model_choices,
        )
        bare = [k for k in OVERRIDE_PRESETS if k not in PRESET_MODEL_CHOICES]
        assert bare, (
            "every preset now has a choices list, so this rule is untestable — "
            "either that is intended (delete this test) or a preset lost its "
            "single-model fallback."
        )
        for key in bare:
            assert resolve_model_choices(key, {}) == (OVERRIDE_PRESETS[key]["model"],)

    def test_unknown_preset_has_no_choices(self):
        from prometheus.router.model_router import resolve_model_choices
        assert resolve_model_choices("definitely-not-real", {}) == ()

    def test_user_models_list_replaces_builtin(self):
        """A provider shipping a new model is a config edit, not a release."""
        from prometheus.router.model_router import resolve_model_choices
        from prometheus.router.model_router import (
            OVERRIDE_PRESETS,
            PRESET_MODEL_CHOICES,
        )
        default = OVERRIDE_PRESETS["qwen"]["model"]
        cfg = {"slash_commands": {"qwen": {"models": ["qwen4-max", "qwen4-flash"]}}}
        choices = resolve_model_choices("qwen", cfg)
        assert choices == (default, "qwen4-max", "qwen4-flash")
        # REPLACED, not merged — assert against the built-in list itself rather
        # than one name plucked out of it, so this keeps holding as that list moves.
        assert not (set(PRESET_MODEL_CHOICES["qwen"]) - {default}) & set(choices)

    def test_user_default_model_leads_its_own_list(self):
        from prometheus.router.model_router import resolve_model_choices
        cfg = {"slash_commands": {"qwen": {"model": "qwen4-max",
                                           "models": ["qwen4-flash"]}}}
        assert resolve_model_choices("qwen", cfg) == ("qwen4-max", "qwen4-flash")

    def test_split_bare_and_composite_keys(self):
        from prometheus.router.model_router import split_model_key
        assert split_model_key("qwen") == ("qwen", None)
        assert split_model_key("qwen:qwen3.7-plus") == ("qwen", "qwen3.7-plus")

    def test_split_uses_first_separator_only(self):
        """A model name containing a colon still round-trips."""
        from prometheus.router.model_router import split_model_key
        assert split_model_key("ollama:qwen3.5:9b") == ("ollama", "qwen3.5:9b")

    def test_target_bare_key_is_preset_default(self):
        from prometheus.router.model_router import (
            OVERRIDE_PRESETS,
            resolve_model_target,
        )
        assert (
            resolve_model_target("qwen", {})["model"]
            == OVERRIDE_PRESETS["qwen"]["model"]
        )

    def test_target_composite_key_selects_listed_model(self):
        from prometheus.router.model_router import (
            resolve_model_choices,
            resolve_model_target,
        )
        # Take a NON-DEFAULT model from the live list rather than naming one:
        # the pinned name here (qwen3.7-plus) was dropped from the catalog in
        # #533 and this test would have failed for the wrong reason.
        alternates = resolve_model_choices("qwen", {})[1:]
        assert alternates, "qwen needs >1 choice for this test to mean anything"
        model = alternates[0]
        target = resolve_model_target(f"qwen:{model}", {})
        assert target["model"] == model
        assert target["provider"] == "qwen"          # rest of preset intact
        assert target["api_key_env"] == "QWEN_API_KEY"

    def test_target_rejects_model_outside_the_list(self):
        """The vetted-list boundary: a client picks FROM a list, never names one."""
        from prometheus.router.model_router import resolve_model_target
        assert resolve_model_target("qwen:gpt-4o", {}) is None
        assert resolve_model_target("qwen:../../etc/passwd", {}) is None

    def test_target_unknown_preset_is_none(self):
        from prometheus.router.model_router import resolve_model_target
        assert resolve_model_target("nope:some-model", {}) is None

    def test_xai_stays_single_entry(self):
        """grok-3/-4/-latest are silently served as 4.3 — don't offer them."""
        from prometheus.router.model_router import resolve_model_choices
        assert resolve_model_choices("xai", {}) == ("grok-4.5",)
