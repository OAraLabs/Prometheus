"""/anatomy over a chat relay shows what the machines can do, not where they are (Will, 2026-11-02).

Telegram, Discord and Slack keep every message on their own servers. ``/anatomy`` there
used to send this machine's name, the inference server's address and URL, and every tailnet
peer with its address. Now:

- on a chat relay (Telegram, Discord, Slack) the reply goes through infra/model_view: no
  hostnames, addresses, URLs or peer names; the peer list becomes counts;
- on a local surface (the terminal, Beacon's web slash commands) it shows everything;
- ``anatomy.include_addresses_in_chat: true`` (a literal true) shows everything on chat too.

And when no SSH user is set for a remote inference GPU, every surface says "not configured":
the scanner does not re-attempt the probe on each scan, logs it once at INFO (not a WARNING
per scan), and doctor no longer reports "No GPU detected / running on CPU".

All addresses are from the documentation ranges (RFC 5737, RFC 3849).
"""

from __future__ import annotations

import ast
import asyncio
import logging
from pathlib import Path

import pytest

from prometheus.gateway.commands import cmd_anatomy
from prometheus.infra.anatomy import AnatomyScanner, AnatomyState
from prometheus.infra.anatomy_writer import AnatomyWriter
from prometheus.infra.doctor import Doctor
from prometheus.tools.builtin import anatomy as anatomy_tool

SRC = Path(__file__).resolve().parents[1] / "src" / "prometheus"
WHERE = ("box-one", "192.0.2.10", "192.0.2.20", "192.0.2.30", "192.0.2.31", "2001:db8::5",
         "peer-alpha", "peer-bravo", "http://", "/srv/someone")


def _state(**over) -> AnatomyState:
    s = AnatomyState(
        hostname="box-one", platform="Linux", cpu="Example CPU 12-core", ram_total_gb=16.0, ram_available_gb=11.0,
        inference_engine="llama_cpp", inference_url="http://192.0.2.20:8080", vision_enabled=True,
        model_name="/srv/someone/models/Example-27B-Q4.gguf", model_quantization="Q4",
        gpu_is_remote=True, gpu_inference_host="192.0.2.20", gpu_name="Example GPU 24 GB",
        gpu_vram_total_mb=24000, gpu_vram_used_mb=20000, gpu_vram_free_mb=4000,
        local_gpu_name="Local Card 8 GB", local_gpu_vram_total_mb=8000,
        tailscale_ip="192.0.2.10",
        tailscale_peers=[
            {"name": "peer-alpha", "ip": "192.0.2.30", "online": True},
            {"name": "peer-bravo", "ip": "2001:db8::5", "online": False},
            {"name": "localhost", "ip": "192.0.2.31", "online": False},
        ],
    )
    for k, v in over.items():
        setattr(s, k, v)
    return s


class _Scanner:
    def __init__(self, state: AnatomyState) -> None:
        self.state = state

    async def scan(self) -> AnatomyState:
        return self.state


@pytest.fixture()
def wire(monkeypatch):
    monkeypatch.setattr("prometheus.infra.model_view.local_names", lambda: frozenset({"box-one"}))

    def _wire(state: AnatomyState | None = None, config: dict | None = None) -> None:
        monkeypatch.setattr(anatomy_tool, "_scanner", _Scanner(state or _state()))
        monkeypatch.setattr(anatomy_tool, "_writer", AnatomyWriter())
        monkeypatch.setattr(anatomy_tool, "_project_store", None)
        monkeypatch.setattr(anatomy_tool, "_config", config)
    return _wire


def test_on_a_chat_relay_anatomy_shows_no_address_name_or_peer(wire):
    wire(config={})
    out = asyncio.run(cmd_anatomy(chat=True))
    for bit in WHERE:
        assert bit not in out, bit
    for keep in ("Example CPU 12-core", "Example GPU 24 GB", "Local Card 8 GB", "Example-27B-Q4", "llama.cpp",
                 "Vision: enabled", "Tailscale: 3 peers (1 online)"):
        assert keep in out, keep


def test_on_a_local_surface_anatomy_shows_everything(wire):
    wire(config={})
    out = asyncio.run(cmd_anatomy())
    for bit in ("box-one", "192.0.2.20", "peer-alpha", "192.0.2.30", "http://192.0.2.20:8080"):
        assert bit in out, bit


@pytest.mark.parametrize("value,full", [(True, True), ("true", False), (1, False), (False, False)])
def test_only_a_literal_true_shows_everything_on_chat(wire, value, full):
    wire(config={"anatomy": {"include_addresses_in_chat": value}})
    out = asyncio.run(cmd_anatomy(chat=True))
    assert ("peer-alpha" in out) is full and ("192.0.2.20" in out) is full


def test_an_unwired_config_still_redacts_on_chat(wire):
    wire(config=None)
    assert "peer-alpha" not in asyncio.run(cmd_anatomy(chat=True))


def _calls(rel: str) -> list[ast.Call]:
    tree = ast.parse((SRC / rel).read_text())
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "cmd_anatomy"]


def _chat_kw(call: ast.Call) -> object:
    for kw in call.keywords:
        if kw.arg == "chat":
            return getattr(kw.value, "value", kw.value)
    return None


@pytest.mark.parametrize("rel", ["gateway/telegram.py", "gateway/discord.py", "gateway/slack.py"])
def test_every_chat_relay_asks_for_the_chat_view(rel):
    calls = _calls(rel)
    assert calls and all(_chat_kw(c) is True for c in calls), rel


def test_the_shared_table_used_by_beacon_keeps_the_full_view():
    calls = _calls("gateway/commands.py")
    assert calls and all(_chat_kw(c) is None for c in calls)


# ── Remote GPU with no SSH user: "not configured", once ────────────────────────────


def test_unset_ssh_credentials_are_not_configured_and_not_retried(monkeypatch, caplog):
    scanner = AnatomyScanner(llama_cpp_url="http://192.0.2.20:8080")

    async def no_local(state):  # noqa: ANN001
        return None

    async def no_ssh(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("ssh was attempted with no SSH user configured")

    monkeypatch.setattr(scanner, "_probe_local_gpu_as_secondary", no_local)
    monkeypatch.setattr(scanner, "_probe_remote_gpu", no_ssh)
    with caplog.at_level(logging.INFO, logger="prometheus.infra.anatomy"):
        for _ in range(3):
            state = AnatomyState()
            asyncio.run(scanner._detect_gpu(state))
            assert state.gpu_is_remote and state.gpu_probe_method == "not_configured"
            assert state.gpu_probe_error and state.gpu_probe_error.startswith("not configured")
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], "no WARNING per scan"
    assert sum("not configured" in r.getMessage() for r in caplog.records) == 1, "said once"


def _not_configured() -> AnatomyState:
    return _state(gpu_name=None, gpu_vram_total_mb=None, gpu_vram_used_mb=None, gpu_vram_free_mb=None,
                  gpu_probe_method="not_configured",
                  gpu_probe_error="not configured (set anatomy.ssh_user and anatomy.ssh_key to read it)")


def test_anatomy_says_not_configured_not_unreachable(wire):
    wire(_not_configured(), config={})
    out = asyncio.run(cmd_anatomy())
    assert "not configured" in out and "unreachable" not in out


def test_anatomy_md_says_not_configured_not_probe_failed():
    out = AnatomyWriter()._render_active(_not_configured())
    assert "not configured" in out and "not detected" not in out and "probe failed" not in out


def test_doctor_says_not_configured_not_running_on_cpu():
    check = Doctor({})._check_gpu(_not_configured())
    assert check.status == "info"
    assert "not configured" in check.message
    assert "No GPU" not in check.message and "CPU" not in (check.fix or "")
