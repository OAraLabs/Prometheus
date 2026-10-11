"""``PROMETHEUS_LLAMA_CPP_URL`` and ``PROMETHEUS_OLLAMA_URL`` point the main model at another machine.

The bug this file pins: both variables wrote ``providers.llama_cpp.base_url`` / ``providers.ollama.base_url``
(config/env_override.py), a block nothing reads for those two providers. ``ProviderRegistry.create`` takes
their URL from ``model.base_url`` only, so setting either variable changed nothing at all, and the one test
on them (tests/test_config_env.py) asserted the dict it wrote, not the provider that got built. A container
user who set ``PROMETHEUS_OLLAMA_URL=http://ollama:11434`` kept talking to ``localhost:11434``.

Asserted here at the effect: the config the daemon's own loader returns is handed to the registry, and the
provider it builds is the one checked. When the main model's provider is not the variable's, nothing moves
and a WARNING names the variable, because a setting that silently does nothing is the bug itself.

Scope: the MAIN model (``model:``). Router rules and ``backends:`` entries name their own URLs and are
left alone; one variable redirecting every llama.cpp entry at once would be a different bug.
"""

from __future__ import annotations

import logging

import pytest
import yaml

from prometheus.providers.registry import ProviderRegistry

OLLAMA_VAR = "PROMETHEUS_OLLAMA_URL"
LLAMA_VAR = "PROMETHEUS_LLAMA_CPP_URL"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(OLLAMA_VAR, raising=False)
    monkeypatch.delenv(LLAMA_VAR, raising=False)


def _daemon_config(tmp_path, model: dict) -> dict:
    """What `oara daemon` would boot from: its own loader, an explicit --config."""
    from prometheus.daemon import load_config

    path = tmp_path / "prometheus.yaml"
    path.write_text(yaml.safe_dump({"model": model}), encoding="utf-8")
    return load_config(str(path))


def _built_url(config: dict) -> str:
    return ProviderRegistry.create(config["model"])._base_url


def test_the_ollama_variable_points_the_ollama_provider(tmp_path, monkeypatch):
    monkeypatch.setenv(OLLAMA_VAR, "http://ollama:11434")
    config = _daemon_config(tmp_path, {"provider": "ollama", "base_url": "http://localhost:11434", "model": "m"})
    assert _built_url(config) == "http://ollama:11434"


def test_the_llama_cpp_variable_points_the_llama_cpp_provider(tmp_path, monkeypatch):
    monkeypatch.setenv(LLAMA_VAR, "http://gpu-box:8080")
    config = _daemon_config(tmp_path, {"provider": "llama_cpp", "base_url": "http://localhost:8080"})
    assert _built_url(config) == "http://gpu-box:8080"


def test_llama_cpp_is_the_provider_when_none_is_named(tmp_path, monkeypatch):
    # ProviderRegistry.create defaults a missing provider to llama_cpp, so the variable must follow it there.
    monkeypatch.setenv(LLAMA_VAR, "http://gpu-box:8080")
    config = _daemon_config(tmp_path, {"model": "m"})
    assert _built_url(config) == "http://gpu-box:8080"


def test_the_cli_loader_agrees(tmp_path, monkeypatch):
    from prometheus.__main__ import load_config as cli_load_config

    monkeypatch.setenv(OLLAMA_VAR, "http://ollama:11434")
    path = tmp_path / "prometheus.yaml"
    path.write_text(yaml.safe_dump({"model": {"provider": "ollama", "model": "m"}}), encoding="utf-8")
    assert _built_url(cli_load_config(str(path))) == "http://ollama:11434"


@pytest.mark.parametrize(("var", "provider"), [(OLLAMA_VAR, "anthropic"), (LLAMA_VAR, "ollama"),
                                              (OLLAMA_VAR, "llama_cpp")])
def test_a_variable_for_another_provider_moves_nothing_and_says_so(var, provider, tmp_path, monkeypatch, caplog):
    monkeypatch.setenv(var, "http://elsewhere:1")
    model = {"provider": provider, "base_url": "http://configured:2", "model": "m"}
    with caplog.at_level(logging.WARNING):
        config = _daemon_config(tmp_path, model)
    assert config["model"]["base_url"] == "http://configured:2"
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(var in w and provider in w for w in warnings), warnings
    assert not any("http://elsewhere:1" in w for w in warnings), "names only, never values, on the log"


def test_no_dead_providers_block_is_written(tmp_path, monkeypatch):
    monkeypatch.setenv(OLLAMA_VAR, "http://ollama:11434")
    config = _daemon_config(tmp_path, {"provider": "ollama", "model": "m"})
    assert "ollama" not in (config.get("providers") or {}), "a key nothing reads is how this went unnoticed"
