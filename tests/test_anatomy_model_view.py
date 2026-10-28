"""The model is told what the machines can do, not where they are (Will, 2026-10-27).

A Telegram reply from a local model quoted the model server's address: the Infrastructure
section of every local prompt, ANATOMY.md and the ``anatomy`` tool name hosts, addresses,
URLs and every tailnet peer. The model needs "local llama.cpp, 32k, vision", not the
network's layout. So by default:

- hostnames, IPv4/IPv6 addresses, URLs, ``*.ts.net`` names, the tailnet peer list and a
  model file's directory are withheld from the prompt (both build paths) and from every
  answer of the ``anatomy`` tool;
- what the machines can do stays: CPU, RAM, GPU and VRAM, the model, engine, vision;
- loopback stays readable, and so do times and version numbers;
- ``anatomy.include_addresses: true`` (a literal true) restores the old text.

All addresses here are from the documentation ranges (RFC 5737, RFC 3849).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from prometheus.config.profiles import AgentProfile
from prometheus.context.prompt_assembler import build_runtime_system_prompt
from prometheus.infra.anatomy import AnatomyState
from prometheus.infra.anatomy_writer import AnatomyWriter
from prometheus.infra.model_view import (
    THIS_MACHINE,
    WITHHELD,
    for_model,
    include_addresses,
    network_names,
    without_addresses,
)
from prometheus.tools.base import ToolExecutionContext
from prometheus.tools.builtin import anatomy as anatomy_tool

SECRETS = (
    "box-one", "gpu-box", "192.0.2.10", "192.0.2.20", "192.0.2.30", "2001:db8::5",
    "peer-alpha", "peer-bravo", "box-one.example.ts.net", "/srv/someone/models", "http://",
)
CAPABILITIES = ("Example CPU 12-core", "Example GPU 24 GB", "Local Card 8 GB", "Example-27B-Q4", "llama_cpp",
                "Vision:** Enabled")


def _state() -> AnatomyState:
    return AnatomyState(
        hostname="box-one", cpu="Example CPU 12-core", ram_total_gb=16.0,
        gpu_name="Example GPU 24 GB", gpu_is_remote=True, gpu_inference_host="gpu-box",
        gpu_vram_total_mb=24000, gpu_vram_used_mb=20000, gpu_vram_free_mb=4000,
        local_gpu_name="Local Card 8 GB", local_gpu_vram_total_mb=8000,
        model_name="/srv/someone/models/Example-27B-Q4.gguf", model_quantization="Q4",
        inference_engine="llama_cpp", inference_url="http://192.0.2.20:8080", vision_enabled=True,
        tailscale_ip="192.0.2.10",
        tailscale_peers=[
            {"name": "peer-alpha", "ip": "192.0.2.30", "online": True},
            {"name": "peer-bravo", "ip": "2001:db8::5", "online": False},
            {"name": "localhost", "ip": "192.0.2.31", "online": False},
        ],
    )


def _active() -> str:
    return AnatomyWriter()._render_active(_state())


def test_the_rendered_section_keeps_capabilities_and_loses_every_address():
    text = _active() + "\n- Also at box-one.example.ts.net, see http://gpu-box:8080/props\n"
    out = without_addresses(text, {"gpu-box"}, here={"box-one"})
    for bit in SECRETS:
        assert bit not in out, bit
    for keep in CAPABILITIES:
        assert keep in out, keep
    assert "- Tailscale: connected (3 peers)" in out
    assert f"| {THIS_MACHINE} | Host |" in out
    assert "REMOTE — another machine" in out
    assert "- **Engine:** llama_cpp\n" in out, "the URL goes with its brackets"


def test_peer_names_are_withheld_wherever_they_appear():
    text = "- Tailscale: 192.0.2.10 (1 peers)\n  - peer-alpha: 192.0.2.30 (online)\nInference on peer-alpha."
    out = without_addresses(text, here=set())
    assert "peer-alpha" not in out and "Inference on another machine." in out
    assert "- Tailscale: connected (1 peer)" in out


@pytest.mark.parametrize("text", [
    "http://localhost:8080/v1", "http://127.0.0.1:8005/health", "[::1]:8010", "bind 0.0.0.0",
    "started 23:25:33", "oara-prometheus 0.9.17", "tailscale 1.104.2", "ctranslate2 4.8.3",
    "a minimum of two", "999.1.2.3",
])
def test_loopback_times_and_versions_are_left_alone(text):
    assert without_addresses(text, {"mini"}, here=set()) == text


def test_a_peer_named_localhost_does_not_mangle_loopback():
    text = "- Tailscale: 192.0.2.10 (1 peers)\n  - localhost: 192.0.2.31 (offline)\nengine http://localhost:8080"
    assert without_addresses(text, here=set()).endswith("engine http://localhost:8080")


def test_names_are_whole_words_and_sentences_keep_their_full_stop():
    out = without_addresses("Running on box-one. box-one-backup and mybox-one stay.", here={"box-one"})
    assert out == "Running on this machine. box-one-backup and mybox-one stay."


def test_addresses_with_ports_are_withheld_whole():
    out = without_addresses("at 192.0.2.20:8080 and [2001:db8::5]:443", here=set())
    assert out == f"at {WITHHELD} and {WITHHELD}"


def test_the_config_names_every_host_it_points_at():
    config = {
        "model": {"base_url": "http://gpu-box:8080", "fallback": {"base_url": "https://Remote-Box.example:9"}},
        "infrastructure": {"gpu_host": "192.0.2.20", "mini_host": "brain-box:8005"},
        "backends": {"desk": {"base_url": "http://localhost:1234"}},
        "whisper": {"url": "ws://voice-box:8001/asr"},
        "note": "not a host",
    }
    assert network_names(config) == {"gpu-box", "remote-box.example", "192.0.2.20", "brain-box", "voice-box"}


@pytest.mark.parametrize("config,on", [
    ({}, False), ({"anatomy": {}}, False), ({"anatomy": {"include_addresses": False}}, False),
    ({"anatomy": {"include_addresses": "true"}}, False), ({"anatomy": {"include_addresses": 1}}, False),
    ({"anatomy": {"include_addresses": True}}, True),
])
def test_only_a_literal_true_lets_addresses_through(config, on):
    assert include_addresses(config) is on
    text = "engine at 192.0.2.20"
    assert (for_model(text, config) == text) is on


@pytest.fixture()
def home(tmp_path, monkeypatch) -> Path:
    h = tmp_path / "prom-home"
    h.mkdir()
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(h))
    (h / "ANATOMY.md").write_text(f"# Anatomy\n\n## Active Configuration\n{_active()}\n\n## Architecture\nx\n")
    return h


@pytest.mark.parametrize("profile", [None, AgentProfile(name="full")], ids=["toggles", "profile"])
def test_the_prompt_has_no_addresses_on_either_build_path(home, tmp_path, profile):
    config = {"model": {"base_url": "http://gpu-box:8080"}}
    prompt = build_runtime_system_prompt(cwd=str(tmp_path), config=config, skills=[], profile=profile)
    for bit in ("gpu-box", "192.0.2.10", "192.0.2.20", "192.0.2.30", "2001:db8::5", "peer-alpha",
                "/srv/someone/models"):
        assert bit not in prompt, bit
    assert "Example GPU 24 GB" in prompt and "Example-27B-Q4" in prompt


def test_the_setting_puts_the_addresses_back_in_the_prompt(home, tmp_path):
    config = {"anatomy": {"include_addresses": True}}
    prompt = build_runtime_system_prompt(cwd=str(tmp_path), config=config, skills=[])
    assert "peer-alpha" in prompt and "http://192.0.2.20:8080" in prompt


class _Scanner:
    async def quick_scan(self) -> AnatomyState:
        return _state()

    async def scan(self) -> AnatomyState:
        return _state()


@pytest.fixture()
def wired(monkeypatch):
    def wire(config):
        monkeypatch.setattr(anatomy_tool, "_scanner", _Scanner())
        monkeypatch.setattr(anatomy_tool, "_writer", AnatomyWriter())
        monkeypatch.setattr(anatomy_tool, "_project_store", None)
        monkeypatch.setattr(anatomy_tool, "_config", config)
    return wire


def _ask(action: str, tmp_path: Path) -> str:
    tool = anatomy_tool.AnatomyTool()
    args = anatomy_tool.AnatomyInput(action=action)
    return asyncio.run(tool.execute(args, ToolExecutionContext(cwd=tmp_path))).output


@pytest.mark.parametrize("action", ["status", "diagram"])
def test_the_anatomy_tool_answers_without_addresses(wired, tmp_path, action, monkeypatch):
    monkeypatch.setattr("prometheus.infra.model_view.local_names", lambda: frozenset({"box-one"}))
    wired({})
    out = _ask(action, tmp_path)
    for bit in ("box-one", "gpu-box", "192.0.2.20", "peer-alpha"):
        assert bit not in out, bit


def test_the_anatomy_tool_status_keeps_the_capabilities(wired, tmp_path, monkeypatch):
    monkeypatch.setattr("prometheus.infra.model_view.local_names", lambda: frozenset({"box-one"}))
    wired(None)  # not wired: addresses are still withheld
    out = _ask("status", tmp_path)
    assert "Running on this machine" in out and "Example-27B-Q4" in out and "Vision enabled" in out


def test_the_anatomy_tool_with_the_setting_on_is_unchanged(wired, tmp_path):
    wired({"anatomy": {"include_addresses": True}})
    out = _ask("status", tmp_path)
    assert "box-one" in out and "gpu-box" in out


def test_the_daemon_wires_its_config_into_the_tool():
    import ast

    src = Path(__file__).resolve().parents[1] / "src" / "prometheus" / "daemon.py"
    calls = [n for n in ast.walk(ast.parse(src.read_text()))
             if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "set_anatomy_components"]
    assert calls and all(len(c.args) == 4 for c in calls), "scanner, writer, store, config"
