"""Discovery is optional, and attached.

``zeroconf`` (LGPL-2.1-or-later, one dependency of its own) is the ``discovery`` extra. Two promises, and a
third about not leaving the feature unplugged:

* **Without it nothing else changes.** The daemon starts, serves, and pairs; the advertiser reports why it is
  not advertising. These tests make the library UNIMPORTABLE rather than rely on it being absent from the
  venv, so they hold on a machine that has it.
* **With it, only the extra carries it.** It is declared as an extra, locked, and not a base dependency.
* **Connected.** A correct advertiser that the launcher never starts is the "config-dark" failure this repo
  keeps a rule against, so the launcher's seams are pinned.
"""

from __future__ import annotations

import ast
import asyncio
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import prometheus.web.launcher as launcher
from prometheus.cli.doctor import check_network_discovery
from prometheus.web.discovery import Advertiser
from prometheus.web.network import HOME, NetworkSettings, NetworkState
from tests.support.pairing_world import World

REPO = Path(__file__).resolve().parents[1]


def _calls(module) -> list[ast.Call]:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)]


def _name(call: ast.Call) -> str:
    return call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, "id", "")


# ── optional ─────────────────────────────────────────────────────────────────

@pytest.fixture
def no_zeroconf(monkeypatch):
    monkeypatch.setitem(sys.modules, "zeroconf", None)          # `import zeroconf` raises ImportError
    monkeypatch.setitem(sys.modules, "zeroconf.asyncio", None)


@pytest.mark.asyncio
async def test_the_default_backend_without_the_library_is_a_status_not_a_crash(no_zeroconf):
    advertiser = Advertiser(
        state=NetworkState(mode=HOME, bind="0.0.0.0", bind_source="config", tls_enabled=False, warnings=[]),
        settings=NetworkSettings(home_network=True, allow_plaintext_lan=True), port=8005,
        hello=lambda: {"v": "1", "name": "n", "agent": "a", "fp": "", "pair": "approve", "tls": False},
        display_name=lambda: "n",
        adapters=lambda: [SimpleNamespace(name="en0", nice_name="en0",
                                          ips=[SimpleNamespace(ip="192.168.1.2", is_IPv4=True, is_IPv6=False)])])
    await advertiser.start()                                    # no backend_factory given: the real default
    assert advertiser.status.advertising is False
    assert "zeroconf is not installed" in advertiser.status.reason
    await advertiser.stop()


def test_a_daemon_without_the_library_still_pairs_a_device(tmp_path, no_zeroconf):
    world = World(tmp_path, recording=False)
    created, requester, _ = world.created()
    assert world.approve(created).status_code == 200
    polled = world.poll(created).json()
    assert polled["status"] == "approved" and requester.unseal(created["request_id"], polled["sealed"])["token"]
    assert world.as_("global", "GET", "/api/network").status_code == 200


def test_the_library_is_an_extra_and_locked_and_never_a_base_dependency():
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extras = project["optional-dependencies"]
    assert any(dep.lower().startswith("zeroconf") for dep in extras["discovery"])
    assert not any("zeroconf" in dep.lower() for dep in project["dependencies"])
    assert 'name = "zeroconf"' in (REPO / "uv.lock").read_text(encoding="utf-8")


# ── connected ────────────────────────────────────────────────────────────────

def test_the_launcher_builds_and_starts_the_advertiser():
    names = [_name(c) for c in _calls(launcher)]
    assert "Advertiser" in names, "the launcher never builds the advertiser"
    handlers = [c for c in _calls(launcher) if _name(c) == "add_event_handler"]
    started = {ast.unparse(c.args[0]) for c in handlers if c.args and ast.unparse(c.args[1]).endswith("start")}
    stopped = {ast.unparse(c.args[0]) for c in handlers if c.args and ast.unparse(c.args[1]).endswith("stop")}
    assert "'startup'" in started and "'shutdown'" in stopped, "start with the server, withdraw on shutdown"


def test_the_launcher_publishes_what_the_network_route_reads():
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    for attr in ("app.state.resolved_bind", "app.state.advertiser", "app.state.config_path"):
        assert attr in source, f"{attr} is read by GET/PUT /api/network and must be set by the launcher"


def test_the_daemon_tells_the_launcher_which_config_file_it_loaded():
    import prometheus.daemon as daemon

    calls = [c for c in _calls(daemon) if _name(c) == "launch_web"]
    assert calls and all("config_path" in {kw.arg for kw in c.keywords} for c in calls), (
        "PUT /api/network must write the SAME file the daemon read, including an explicit --config")


# ── the shipped template ─────────────────────────────────────────────────────

def test_the_template_ships_the_conservative_values():
    template = yaml.safe_load((REPO / "config" / "prometheus.yaml.default").read_text(encoding="utf-8"))
    assert template["network"] == {"home_network": False, "allow_plaintext_lan": False}
    assert template["discovery"]["mdns"] is True


# ── doctor ───────────────────────────────────────────────────────────────────

def test_no_row_for_a_machine_that_never_asked_for_home_network():
    assert check_network_discovery({}) is None
    assert check_network_discovery({"web": {"bind": "127.0.0.1"}}) is None
    assert check_network_discovery({"web": {"bind": "0.0.0.0"}}) is None


def test_asking_for_home_network_without_a_way_to_run_it_safely_is_a_warning():
    row = check_network_discovery({"web": {"bind": "0.0.0.0"}, "network": {"home_network": True}})
    assert row.status == "warning" and "allow_plaintext_lan" in row.message + (row.fix or "")


def test_home_network_over_plain_http_is_a_warning_that_says_so():
    row = check_network_discovery({"web": {"bind": "0.0.0.0"},
                                   "network": {"home_network": True, "allow_plaintext_lan": True}})
    assert row.status == "warning" and "plain HTTP" in row.message


def test_home_network_with_the_library_missing_names_the_extra(no_zeroconf):
    row = check_network_discovery({"web": {"bind": "0.0.0.0"},
                                   "network": {"home_network": True, "allow_plaintext_lan": True}})
    assert "[discovery]" in (row.fix or "") + row.message


def test_the_switch_on_a_loopback_bind_is_called_out():
    row = check_network_discovery({"web": {"bind": "127.0.0.1"}, "network": {"home_network": True}})
    assert row.status == "warning" and "127.0.0.1" in row.message


@pytest.mark.asyncio
async def test_starting_the_advertiser_never_holds_up_startup(no_zeroconf):
    advertiser = Advertiser(
        state=NetworkState(mode=HOME, bind="0.0.0.0", bind_source="config", tls_enabled=False, warnings=[]),
        settings=NetworkSettings(home_network=True, allow_plaintext_lan=True), port=8005,
        hello=lambda: {"v": "1", "name": "n", "agent": "a", "fp": "", "pair": "approve", "tls": False},
        display_name=lambda: "n", adapters=lambda: [])
    await asyncio.wait_for(advertiser.start(), timeout=5)
    await advertiser.stop()
