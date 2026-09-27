"""The tier decision reaches the adapter, the boot line, the CLI and the probes (WP-X.28, PR 2).

Companion to tests/test_adapter_tier_resolver.py (the pure resolver). These
tests import the new module lazily, so on main they collect and each fails on
its own assertion: the silent ``full``, the header that printed the literal
text ``type(provider)``, the adapter that carries no decision, the probes that
do not exist. The mutation: revert __main__.create_adapter to its name-only
chooser and every test in TestMain goes red.
"""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import yaml

from prometheus.adapter.formatter import GemmaFormatter, PassthroughFormatter, QwenFormatter
from prometheus.adapter.validator import Strictness
from prometheus.providers.base import ModelProvider
from prometheus.providers.llama_cpp import LlamaCppProvider
from prometheus.providers.ollama import OllamaProvider

REPO = Path(__file__).resolve().parents[1]
PARITY = REPO / "tests" / "fixtures" / "parity"
REGISTRY = REPO / "config" / "model_registry.yaml"

BONSAI = "Ternary-Bonsai-2-27B-PQ2_0.gguf"          # the ladder's exact file name
ORNITH = "Ornith-1.5-9B-Q4_K_M.gguf"
PRODUCTION = "/models-root/models/Qwen3.8-27B-UD-Q4_K_XL.gguf"   # listed: qwen-3


def recorded_props() -> dict:
    trace = json.loads((PARITY / "tool_calls.trace.json").read_text())
    for exchange in trace["exchanges"]:
        if exchange.get("path") == "/props":
            return json.loads(exchange["body"])
    raise AssertionError("the tool_calls golden records no /props exchange")


def recorded_show() -> dict:
    trace = json.loads((PARITY / "repaired_tool_call.trace.json").read_text())
    for exchange in trace["exchanges"]:
        if exchange.get("path") == "/api/show":
            return json.loads(exchange["body"])
    raise AssertionError("the repaired_tool_call golden records no /api/show exchange")


def _tier():
    """The module under test, imported where used (it does not exist on main)."""
    import prometheus.adapter.tier as tier

    return tier


@pytest.fixture(scope="module")
def props() -> dict:
    return recorded_props()


@pytest.fixture(scope="module")
def show() -> dict:
    return recorded_show()


@pytest.fixture(scope="module")
def qwen_xml():
    t = _tier()
    p = recorded_props()
    return t.classify_template(chat_template=p["chat_template"], caps=p["chat_template_caps"])


@pytest.fixture(scope="module")
def not_native():
    return _tier().ToolTemplate(native=False, call_format=None, evidence="a template with no tool block")


@pytest.fixture(scope="module")
def unreadable():
    return _tier().ToolTemplate(native=None, call_format=None, evidence="nothing was readable")


# ---------------------------------------------------------------------------
# Through __main__: the registry file, the seam, the adapter, the log line
# ---------------------------------------------------------------------------

class TestMain:
    def test_bonsai_and_ornith_resolve_light_from_the_recorded_template(self, qwen_xml):
        from prometheus.__main__ import _get_adapter_tier, _resolve_adapter_tier

        for name in (BONSAI, ORNITH):
            assert _get_adapter_tier("llama_cpp", name) == "full", "unchanged without a template"
            assert _get_adapter_tier("llama_cpp", name, qwen_xml) == "light"
            d = _resolve_adapter_tier("llama_cpp", name, qwen_xml)
            assert (d.tier, d.source) == ("light", "template")

    def test_every_registered_pattern_keeps_its_tier_with_or_without_a_template(self, not_native, unreadable):
        """The registry precedes the template: a listed model's tier is what it
        was, even against a template that disagrees."""
        from prometheus.__main__ import _get_adapter_tier

        families = yaml.safe_load(REGISTRY.read_text())["models"]
        checked = 0
        for meta in families.values():
            fc = meta["capabilities"]["function_calling"]
            expected = "light" if fc.get("supported") and fc.get("requires") is None else "full"
            for pattern in meta["match_patterns"]:
                name = f"{pattern}-9B-Q4_K_M.gguf"
                assert _get_adapter_tier("llama_cpp", name) == expected, name
                assert _get_adapter_tier("llama_cpp", name, not_native) == expected, name
                assert _get_adapter_tier("llama_cpp", name, unreadable) == expected, name
                checked += 1
        assert checked > 20

    def test_create_adapter_with_the_template_builds_light_for_bonsai(self, qwen_xml, caplog):
        from prometheus.__main__ import create_adapter

        with caplog.at_level(logging.INFO):
            adapter = create_adapter({"provider": "llama_cpp", "model": BONSAI}, {}, template=qwen_xml)
        assert adapter.tier == "light"
        assert isinstance(adapter.formatter, QwenFormatter)
        assert adapter.validator.strictness == Strictness.NONE
        assert adapter.retry.max_retries == 1
        assert adapter.tier_decision.source == "template"
        lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Adapter tier:")]
        assert lines and "light" in lines[0] and "template" in lines[0] and "Bonsai" in lines[0]
        assert all(r.levelno == logging.INFO for r in caplog.records if r.getMessage().startswith("Adapter tier:"))

    def test_create_adapter_without_a_template_is_the_loud_fallback(self, caplog):
        from prometheus.__main__ import create_adapter

        with caplog.at_level(logging.INFO):
            adapter = create_adapter({"provider": "llama_cpp", "model": BONSAI})
        assert adapter.tier == "full"
        assert adapter.validator.strictness == Strictness.MEDIUM
        assert adapter.tier_decision.source == "fallback"
        warnings = [r for r in caplog.records
                    if r.levelno == logging.WARNING and r.getMessage().startswith("Adapter tier:")]
        assert warnings and "full" in warnings[0].getMessage()
        assert "model_registry.yaml" in warnings[0].getMessage()

    @pytest.mark.parametrize("cfg, tier, source, formatter", [
        ({"provider": "llama_cpp", "model": PRODUCTION}, "light", "registry", QwenFormatter),
        ({"provider": "llama_cpp", "model": "gemma4-26b"}, "light", "registry", GemmaFormatter),
        ({"provider": "ollama", "model": "qwen2.5:7b-instruct"}, "light", "registry", QwenFormatter),
        ({"provider": "openai", "model": "gpt-4o"}, "off", "provider_class", PassthroughFormatter),
    ])
    def test_todays_models_are_unchanged_and_now_say_why(self, cfg, tier, source, formatter, caplog):
        from prometheus.__main__ import create_adapter

        with caplog.at_level(logging.INFO):
            adapter = create_adapter(cfg)
        assert adapter.tier == tier
        assert isinstance(adapter.formatter, formatter)
        assert adapter.tier_decision.source == source
        lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Adapter tier:")]
        assert len(lines) == 1 and f"Adapter tier: {tier}" in lines[0]

    def test_a_disagreeing_template_warns_and_the_registry_decides(self, caplog, not_native):
        from prometheus.__main__ import create_adapter

        with caplog.at_level(logging.INFO):
            adapter = create_adapter({"provider": "llama_cpp", "model": PRODUCTION}, template=not_native)
        assert adapter.tier == "light"
        assert adapter.tier_decision.disagreement
        assert any(r.levelno == logging.WARNING and "disagrees" in r.getMessage() for r in caplog.records)

    def test_the_seam_still_forces_and_the_record_says_so(self, monkeypatch):
        """The ladder's tier sweep replaces _get_adapter_tier with a two-argument
        function (gym/ladder/tiers.py). It must keep working, and the record
        must not claim a source it was not."""
        import prometheus.__main__ as daemon

        monkeypatch.setattr(daemon, "_get_adapter_tier", lambda provider, model: "light")
        adapter = daemon.create_adapter({"provider": "llama_cpp", "model": BONSAI}, {})
        assert adapter.tier == "light"
        assert adapter.tier_decision.source == "forced"
        assert "resolver says full by fallback" in adapter.tier_decision.detail

        monkeypatch.setattr(daemon, "_get_adapter_tier", lambda provider, model: "full")
        adapter = daemon.create_adapter({"provider": "llama_cpp", "model": PRODUCTION}, {})
        assert (adapter.tier, adapter.tier_decision.source) == ("full", "forced")

    def test_a_missing_registry_warns_once_per_adapter(self, monkeypatch, caplog):
        from prometheus import __main__ as main_mod
        from prometheus.config import model_registry

        def _missing():
            raise model_registry.ModelRegistryNotFound("model_registry.yaml not found")

        monkeypatch.setattr(model_registry, "get_model_registry_path", _missing)
        with caplog.at_level(logging.WARNING):
            adapter = main_mod.create_adapter({"provider": "llama_cpp", "model": PRODUCTION})
        assert adapter.tier == "full"
        assert adapter.tier_decision.source == "fallback"
        assert "the registry file is missing" in adapter.tier_decision.detail
        not_found = [r for r in caplog.records if "model registry: NOT FOUND" in r.getMessage()]
        assert len(not_found) == 1

    def test_a_missing_registry_no_longer_decides_when_the_template_can(self, monkeypatch, qwen_xml):
        from prometheus import __main__ as main_mod
        from prometheus.config import model_registry

        def _missing():
            raise model_registry.ModelRegistryNotFound("model_registry.yaml not found")

        monkeypatch.setattr(model_registry, "get_model_registry_path", _missing)
        adapter = main_mod.create_adapter({"provider": "llama_cpp", "model": PRODUCTION}, template=qwen_xml)
        assert (adapter.tier, adapter.tier_decision.source) == ("light", "template")

    def test_the_header_no_longer_prints_the_literal_type_provider(self):
        import prometheus.__main__ as main_mod

        assert 'Provider: type(provider)' not in inspect.getsource(main_mod)

    def test_the_interactive_header_names_the_provider_class_and_the_tier(self, qwen_xml):
        from prometheus.__main__ import _interactive_header_lines, create_adapter

        adapter = create_adapter({"provider": "llama_cpp", "model": BONSAI}, template=qwen_xml)
        ctx = SimpleNamespace(model=BONSAI, provider=LlamaCppProvider(), adapter=adapter)
        lines = _interactive_header_lines(ctx)
        assert lines[0] == f"Model: {BONSAI} | Provider: LlamaCppProvider"
        assert lines[1].startswith("Adapter tier: light — ")
        assert not any("type(provider)" in line for line in lines)
        bare = _interactive_header_lines(SimpleNamespace(model="m", provider=LlamaCppProvider(), adapter=None))
        assert bare == ["Model: m | Provider: LlamaCppProvider"]


# ---------------------------------------------------------------------------
# The probes
# ---------------------------------------------------------------------------

def _resp(json_data, status_code: int = 200):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


class TestProviderProbes:
    def test_the_base_class_has_nothing_detected(self):
        assert ModelProvider.tool_template is None

    async def test_llama_cpp_reads_props(self, props):
        p = LlamaCppProvider()
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=_resp(props)):
            verdict = await p.detect_tool_template("ignored")
        assert (verdict.native, verdict.call_format) == (True, "qwen-xml")
        assert p.tool_template is verdict

    async def test_llama_cpp_unreachable_is_unknown_not_false(self):
        p = LlamaCppProvider()
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock,
                   side_effect=httpx.ConnectError("refused")):
            verdict = await p.detect_tool_template()
        assert verdict.native is None and "could not be read" in verdict.evidence

    async def test_ollama_asks_show_for_the_named_model(self, show):
        p = OllamaProvider()
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=_resp(show)) as post:
            verdict = await p.detect_tool_template("qwen2.5:7b-instruct")
        assert (verdict.native, verdict.call_format) == (True, "qwen-json")
        assert post.call_args.kwargs["json"] == {"model": "qwen2.5:7b-instruct"}
        assert p.tool_template is verdict

    async def test_ollama_without_a_model_name_does_not_guess(self):
        p = OllamaProvider()
        verdict = await p.detect_tool_template(None)
        assert verdict.native is None and "no model name" in verdict.evidence


class _FakeClient:
    """The two calls a probe makes, answered from recorded payloads."""

    def __init__(self, props=None, show=None):
        self._props, self._show = props, show

    async def get(self, url):
        if url.endswith("/props"):
            return _resp(self._props)
        if url.endswith("/v1/models"):
            return _resp({"data": [{"id": self._props.get("model_path", "m")}]})
        if url.endswith("/api/tags"):
            return _resp({"models": [{"name": "qwen2.5:7b-instruct"}]})
        if url.endswith("/api/ps"):
            return _resp({"models": []})
        raise AssertionError(url)

    async def post(self, url, json=None):
        assert url.endswith("/api/show") and json == {"model": "qwen2.5:7b-instruct"}
        return _resp(self._show)


class TestBackendProbes:
    async def test_llama_cpp_backend_carries_the_template_verdict(self, props):
        from prometheus.providers.backends import BackendSpec, BackendStatus, _probe_llama_cpp

        spec = BackendSpec(name="box", provider="llama_cpp", base_url="http://box")
        st = BackendStatus(name="box", provider="llama_cpp", base_url="http://box")
        await _probe_llama_cpp(_FakeClient(props=props), spec, st)
        assert st.ok and st.n_ctx == 32768
        assert st.extra["tool_template"]["native"] is True
        assert st.extra["tool_template"]["call_format"] == "qwen-xml"

    async def test_ollama_backend_carries_the_template_verdict(self, show):
        from prometheus.providers.backends import BackendSpec, BackendStatus, _probe_ollama

        spec = BackendSpec(name="mini", provider="ollama", base_url="http://mini", model="qwen2.5:7b-instruct")
        st = BackendStatus(name="mini", provider="ollama", base_url="http://mini")
        await _probe_ollama(_FakeClient(show=show), spec, st)
        assert st.ok and st.extra["capabilities"] == ["completion", "tools"]
        assert st.extra["tool_template"]["native"] is True
        assert st.extra["tool_template"]["call_format"] == "qwen-json"


class TestCliProbe:
    def test_llama_cpp(self, props):
        from prometheus.__main__ import _detect_tool_template_or_none

        with patch("httpx.get", return_value=_resp(props)) as get:
            verdict = _detect_tool_template_or_none({"provider": "llama_cpp", "base_url": "http://box/"})
        assert verdict.native is True and get.call_args.args[0] == "http://box/props"

    def test_ollama_with_and_without_a_model(self, show):
        from prometheus.__main__ import _detect_tool_template_or_none

        with patch("httpx.post", return_value=_resp(show)) as post:
            verdict = _detect_tool_template_or_none(
                {"provider": "ollama", "base_url": "http://mini", "model": "qwen2.5:7b-instruct"})
        assert verdict.native is True and post.call_args.kwargs["json"] == {"model": "qwen2.5:7b-instruct"}
        verdict = _detect_tool_template_or_none({"provider": "ollama", "base_url": "http://mini"})
        assert verdict.native is None

    def test_a_cloud_provider_has_no_template(self):
        from prometheus.__main__ import _detect_tool_template_or_none

        assert _detect_tool_template_or_none({"provider": "anthropic", "model": "claude"}) is None

    def test_an_unreachable_server_is_recorded_not_fatal(self, caplog):
        from prometheus.__main__ import _detect_tool_template_or_none

        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            with caplog.at_level(logging.WARNING):
                verdict = _detect_tool_template_or_none({"provider": "llama_cpp", "base_url": "http://down"})
        assert verdict.native is None and "could not be asked" in verdict.evidence


# ---------------------------------------------------------------------------
# The daemon's wiring
# ---------------------------------------------------------------------------

def test_the_daemon_probes_the_template_and_hands_it_to_create_adapter():
    """The boot path: detect beside the other probes, pass the verdict in.
    Pinned on the source because run_daemon is not callable in a test."""
    import prometheus.daemon as daemon

    source = inspect.getsource(daemon)
    assert "await provider.detect_tool_template(model_name)" in source
    assert 'create_adapter(model_config, config.get("adapter"), template=tool_template)' in source
    assert source.index("detect_tool_template(model_name)") < source.index("template=tool_template)")
