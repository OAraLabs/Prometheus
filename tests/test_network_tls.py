"""Home network requires TLS: #700's network mode wired to pairing PR 5's listeners (docs/design/lan-tls-pr5.md).

* Running: ``home_network`` is reported when the TLS listener set is up (``app.state.lan_tls``), with
  ``tls: {"enabled": true, "spki_sha256": <the full pin>}`` (what Beacon reads), and no plain-HTTP warning.
  Without it, the mode is not ``home_network`` unless the owner opted in to plain HTTP.
* Choosing it: ``PUT /api/network {"mode": "home_network"}`` is accepted when this daemon can serve TLS after the
  restart (an instance key it can make a certificate over), with no plain-HTTP opt-out written. With no usable
  key and no opt-out it is still ``409 tls_unavailable``.
* Announcing it: mDNS advertises the TLS REST port, with ``tls=1``.
* Listening: with no certificate the home network gets NOTHING (plain HTTP on loopback), unless
  ``network.allow_plaintext_lan`` asks for plain HTTP on the LAN.
* One reader: ``web/tls.py`` reads ``network.home_network`` through #700's ``NetworkSettings``.
"""

from __future__ import annotations

import hashlib

import pytest

pytest.importorskip("fastapi")

from prometheus.config.instance_cert import ensure_instance_certificate  # noqa: E402
from prometheus.config.instance_key import ensure_instance_key  # noqa: E402
from prometheus.web import tls as T  # noqa: E402
from tests.test_network_routes import make, saved  # noqa: E402

HOME = {"network": {"home_network": True}}


@pytest.fixture
def node(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMETHEUS_NODE_DIR", str(tmp_path / "node"))
    return tmp_path / "node"


# ── running ──────────────────────────────────────────────────────────────────

def test_running_over_tls_is_home_network_and_reports_the_full_pin(tmp_path, node):
    der = ensure_instance_key()
    ensure_instance_certificate()
    world = make(tmp_path, source="config", config=HOME, file_text="network:\n  home_network: true\n")
    world.app.state.lan_tls = T.lan_tls_plan(HOME, "0.0.0.0")
    body = world.as_("global", "GET", "/api/network").json()
    assert body["mode"] == "home_network", body
    assert body["tls"] == {"enabled": True, "spki_sha256": hashlib.sha256(der).hexdigest()}
    assert not any(w["code"] == "plain_http_on_lan" for w in body["warnings"]), body["warnings"]


def test_without_a_tls_listener_it_is_not_home_network_and_tls_is_off(tmp_path, node):
    ensure_instance_key()
    world = make(tmp_path, source="config", config=HOME, file_text="network:\n  home_network: true\n")
    body = world.as_("global", "GET", "/api/network").json()
    assert body["mode"] == "open" and body["tls"] == {"enabled": False, "spki_sha256": None}
    assert any(w["code"] == "home_network_unavailable" for w in body["warnings"])


# ── choosing it ──────────────────────────────────────────────────────────────

def test_home_network_is_accepted_when_this_daemon_can_serve_tls_after_the_restart(tmp_path, node):
    ensure_instance_key()
    world = make(tmp_path, bind="127.0.0.1", source="config", file_text='web:\n  bind: "127.0.0.1"\n')
    response = world.as_("global", "PUT", "/api/network", json={"mode": "home_network"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["applied"], body["pending_mode"]) == ("on_restart", "home_network")
    assert not any(w["code"] == "plain_http_on_lan" for w in body["warnings"]), body["warnings"]
    written = saved(world).get("network") or {}
    assert written.get("home_network") is True and not written.get("allow_plaintext_lan"), written


def test_with_no_usable_instance_key_and_no_opt_out_it_is_still_refused(tmp_path, node):
    world = make(tmp_path, bind="127.0.0.1", source="config", file_text='web:\n  bind: "127.0.0.1"\n')
    before = world.cfg_path.read_text()
    response = world.as_("global", "PUT", "/api/network", json={"mode": "home_network"})
    assert response.status_code == 409 and response.json()["error"] == "tls_unavailable"
    assert world.cfg_path.read_text() == before


# ── announcing it ────────────────────────────────────────────────────────────

def test_mdns_advertises_the_tls_port_and_says_tls(node):
    plan = T.lan_tls_plan(HOME, "0.0.0.0")
    assert T.advertised_endpoint(plan, 8005) == (plan.api_port, True)
    assert T.advertised_endpoint(None, 8005) == (8005, False)


# ── listening ────────────────────────────────────────────────────────────────

def test_with_a_certificate_the_lan_gets_tls(node):
    decision = T.decide_lan(HOME, "0.0.0.0", cert_ok=True)
    assert decision.plan is not None and not decision.loopback_only and not decision.plaintext_lan


def test_without_a_certificate_the_lan_gets_nothing(node):
    decision = T.decide_lan(HOME, "0.0.0.0", cert_ok=False)
    assert decision.plan is None and decision.loopback_only and not decision.plaintext_lan


def test_without_a_certificate_the_opt_out_puts_plain_http_on_the_lan(node):
    config = {"network": {"home_network": True, "allow_plaintext_lan": True}}
    decision = T.decide_lan(config, "0.0.0.0", cert_ok=False)
    assert decision.plan is None and not decision.loopback_only and decision.plaintext_lan


def test_with_home_network_off_nothing_changes(node):
    decision = T.decide_lan({}, "0.0.0.0", cert_ok=True)
    assert decision.plan is None and not decision.loopback_only and not decision.plaintext_lan


def test_tls_reads_the_switch_through_network_settings(node, monkeypatch):
    from prometheus.web import network

    seen = []
    real = network.NetworkSettings.from_config

    def spy(config, **kw):
        seen.append(config)
        return real(config, **kw)

    monkeypatch.setattr(network.NetworkSettings, "from_config", staticmethod(spy))
    T.lan_tls_plan(HOME, "0.0.0.0")
    assert seen, "web/tls.py reads network.home_network itself; #700's NetworkSettings is the one reader"
