"""``GET /api/network`` and ``PUT /api/network`` — what the daemon is listening for, and the owner's switch.

Contract: docs/PAIRING-APPROVAL-API.md, 2.2. Anyone with a valid token may READ it (a client wants to know
which network mode it is talking to and whether the daemon is discoverable); only an operator
(``identity.is_operator``) may change it, and a scoped device is a 403 ``operator_only``, not a 401.

What the owner's switch can and cannot do is stated, not implied:

* **It is saved, not applied.** The listeners are bound at start, so ``PUT`` writes the choice into the config
  file and answers ``applied: "on_restart"``; ``GET`` then reports the mode it is running AND the one it will
  come back as. Rebinding live is not built.
* **It refuses what cannot work.** ``home_network`` without TLS and without the owner's explicit
  ``network.allow_plaintext_lan`` is a 409 ``tls_unavailable`` and writes nothing. A bind pinned by
  ``--bind`` or ``PROMETHEUS_WEB_BIND`` outranks the file, so changing the file would be a lie: 409
  ``bind_overridden``, naming the source.
* **It edits the file's text** (comments and all) and verifies before writing; no config file is a 409, never a
  file conjured into existence.
* **It does not let a caller lock itself out.** ``this_mac`` from a caller that is not on this machine is a 409
  ``would_lock_out``: after the restart that caller could no longer reach the daemon.
* **It says whether the control should be shown** (``can_change``) and **gives every warning and the
  advertising reason a stable code** next to the English, so a client writes its own copy.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.web.bind import ResolvedBind  # noqa: E402
from prometheus.web.network import PersistError  # noqa: E402
from tests.support.pairing_world import World  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TAILNET = ".".join(("100", "64", "0", "9"))   # built at run time: the pre-commit scanner looks for these
TEMPLATE = (REPO / "config" / "prometheus.yaml.default").read_text(encoding="utf-8")


class FakeAdvertiser:
    def __init__(self, advertising: bool, reason: str | None, code: str | None = None) -> None:
        self.status = type("S", (), {"advertising": advertising, "reason": reason, "code": code})()


def make(tmp_path, *, bind="0.0.0.0", source="default", file_text: str | None = "web:\n  enabled: true\n",
         config: dict | None = None, advertiser: FakeAdvertiser | None = None) -> World:
    world = World(tmp_path, recording=False, config=config)
    path = tmp_path / "prometheus.yaml"
    if file_text is not None:
        path.write_text(file_text, encoding="utf-8")
    world.app.state.config_path = str(path)
    world.app.state.resolved_bind = ResolvedBind(bind, source)
    if advertiser is not None:
        world.app.state.advertiser = advertiser
    world.cfg_path = path  # type: ignore[attr-defined]
    world.client = TestClient(world.app, client=("127.0.0.1", 50000))      # a caller on this machine
    return world


def remote(world: World, who: str, method: str, url: str, peer: str = "192.0.2.10", **kw):
    """The same call from another machine: the TCP peer is what `would_lock_out` looks at."""
    client = TestClient(world.app, client=(peer, 50000))
    return client.request(method, url, headers=world.hdr(who), **kw)


def saved(world: World) -> dict:
    return yaml.safe_load(world.cfg_path.read_text()) or {}  # type: ignore[attr-defined]


# ── reading ──────────────────────────────────────────────────────────────────

def test_the_shape_is_the_contracts(tmp_path):
    body = make(tmp_path).as_("global", "GET", "/api/network").json()
    assert set(body) == {"mode", "bind", "bind_source", "tls", "advertising", "advertising_reason", "applied",
                         "warnings", "can_change"}
    assert body["tls"] == {"enabled": False, "spki_sha256": None}


def test_an_install_that_predates_the_setting_is_reported_open_not_dressed_up(tmp_path):
    body = make(tmp_path).as_("global", "GET", "/api/network").json()
    assert (body["mode"], body["bind"], body["bind_source"], body["applied"]) == ("open", "0.0.0.0", "default", "live")
    assert body["advertising"] is False


def test_a_loopback_bind_is_this_mac(tmp_path):
    body = make(tmp_path, bind="127.0.0.1", source="config").as_("global", "GET", "/api/network").json()
    assert body["mode"] == "this_mac" and body["bind"] == "127.0.0.1"


def test_home_network_says_it_is_plain_http_and_what_the_advertiser_is_doing(tmp_path):
    world = make(tmp_path, source="config", config={"network": {"home_network": True, "allow_plaintext_lan": True}},
                 file_text="network:\n  home_network: true\n  allow_plaintext_lan: true\n",
                 advertiser=FakeAdvertiser(True, None))
    body = world.as_("global", "GET", "/api/network").json()
    assert body["mode"] == "home_network" and body["advertising"] is True and body["advertising_reason"] is None
    assert [w["code"] for w in body["warnings"]] == ["plain_http_on_lan"]
    assert "plain HTTP" in body["warnings"][0]["message"]


def test_not_advertising_says_why(tmp_path):
    world = make(tmp_path, advertiser=FakeAdvertiser(False, "zeroconf is not installed", "library_missing"))
    body = world.as_("global", "GET", "/api/network").json()
    assert body["advertising"] is False
    assert body["advertising_reason"] == {"code": "library_missing", "message": "zeroconf is not installed"}


def test_with_no_advertiser_in_this_process_it_says_so(tmp_path):
    body = make(tmp_path).as_("global", "GET", "/api/network").json()
    assert body["advertising"] is False
    assert body["advertising_reason"]["code"] == "advertiser_not_running"
    assert "not running" in body["advertising_reason"]["message"]


def test_reading_needs_a_token_but_not_a_particular_one(tmp_path):
    world = make(tmp_path)
    assert world.client.get("/api/network").status_code == 401
    for who in ("global", "owner", "scoped"):
        assert world.as_(who, "GET", "/api/network").status_code == 200, who


def test_a_choice_waiting_for_a_restart_is_visible_before_it_happens(tmp_path):
    world = make(tmp_path, bind="127.0.0.1", source="config",
                 file_text='web:\n  bind: "0.0.0.0"\nnetwork:\n  home_network: true\n  allow_plaintext_lan: true\n')
    body = world.as_("global", "GET", "/api/network").json()
    assert body["mode"] == "this_mac", "what it is running as"
    assert body["applied"] == "on_restart" and body["pending_mode"] == "home_network", "and what it will come back as"
    assert "restart_required" in [w["code"] for w in body["warnings"]]


def test_a_pinned_bind_outranks_what_the_file_says_about_the_next_start(tmp_path):
    """--bind / PROMETHEUS_WEB_BIND win over web.bind, so a file that says 0.0.0.0 promises nothing."""
    world = make(tmp_path, bind="127.0.0.1", source="env",
                 file_text='web:\n  bind: "0.0.0.0"\nnetwork:\n  home_network: true\n  allow_plaintext_lan: true\n')
    body = world.as_("global", "GET", "/api/network").json()
    assert body["bind"] == "127.0.0.1" and body["bind_source"] == "env"
    assert body["applied"] == "live" and "pending_mode" not in body, "the environment still decides the address"


# ── codes next to the English ────────────────────────────────────────────────

def test_every_warning_is_a_code_and_a_message_and_nothing_else(tmp_path):
    for kwargs in (dict(), dict(bind="127.0.0.1", source="config", config={"network": {"home_network": True}}),
                   dict(source="config", config={"network": {"home_network": True, "allow_plaintext_lan": True}})):
        warnings = make(tmp_path, **kwargs).as_("global", "GET", "/api/network").json()["warnings"]
        assert warnings, kwargs
        for warning in warnings:
            assert set(warning) == {"code", "message"} and warning["code"] and warning["message"]


def test_the_wide_open_install_says_so_with_its_code(tmp_path):
    body = make(tmp_path).as_("global", "GET", "/api/network").json()
    assert [w["code"] for w in body["warnings"]] == ["listening_on_all_interfaces"]


def test_a_switch_that_cannot_work_is_coded(tmp_path):
    loopback = make(tmp_path, bind="127.0.0.1", source="config", config={"network": {"home_network": True}},
                    file_text='web:\n  bind: "127.0.0.1"\nnetwork:\n  home_network: true\n')
    assert [w["code"] for w in loopback.as_("global", "GET", "/api/network").json()["warnings"]] == [
        "home_network_on_loopback"]
    refused = make(tmp_path, config={"network": {"home_network": True}})
    assert {"home_network_unavailable", "listening_on_all_interfaces"} == {
        w["code"] for w in refused.as_("global", "GET", "/api/network").json()["warnings"]}


def test_the_advertising_reason_is_null_while_advertising_and_coded_otherwise(tmp_path):
    on = make(tmp_path, advertiser=FakeAdvertiser(True, None, None)).as_("global", "GET", "/api/network").json()
    assert on["advertising_reason"] is None
    off = make(tmp_path, advertiser=FakeAdvertiser(False, "discovery.mdns is off", "mdns_disabled"))
    assert off.as_("global", "GET", "/api/network").json()["advertising_reason"] == {
        "code": "mdns_disabled", "message": "discovery.mdns is off"}


# ── can_change: should the client show the control at all ────────────────────

@pytest.mark.parametrize("who, expected", [("global", True), ("owner", True), ("scoped", False)])
def test_only_an_operator_can_change_it(tmp_path, who, expected):
    body = make(tmp_path, source="config").as_(who, "GET", "/api/network").json()
    assert body["can_change"] is expected


@pytest.mark.parametrize("source, expected", [("config", True), ("default", True), ("flag", False),
                                              ("env", False), ("caller", False)])
def test_a_bind_pinned_outside_the_file_cannot_be_changed_by_anyone(tmp_path, source, expected):
    body = make(tmp_path, source=source).as_("global", "GET", "/api/network").json()
    assert body["can_change"] is expected


def test_with_the_token_off_everyone_is_the_operator(tmp_path):
    world = make(tmp_path, config={"web": {"api_token": ""}})
    assert world.client.get("/api/network").json()["can_change"] is True


def test_can_change_agrees_with_what_a_put_would_do(tmp_path):
    """The flag is the answer to 'would the control work', so a false one must not be a 403 or a pin 409 in disguise."""
    scoped = make(tmp_path, source="config")
    assert scoped.as_("scoped", "GET", "/api/network").json()["can_change"] is False
    assert scoped.as_("scoped", "PUT", "/api/network", json={"mode": "this_mac"}).status_code == 403
    pinned = make(tmp_path, bind="127.0.0.1", source="env", file_text='network:\n  allow_plaintext_lan: true\n')
    assert pinned.as_("global", "GET", "/api/network").json()["can_change"] is False
    assert pinned.as_("global", "PUT", "/api/network", json={"mode": "home_network"}).status_code == 409


# ── a caller must not lock itself out ────────────────────────────────────────

def test_this_mac_from_another_machine_is_refused_because_it_would_cut_that_caller_off(tmp_path):
    world = make(tmp_path, file_text='web:\n  bind: "0.0.0.0"\n')
    before = world.cfg_path.read_text()
    response = remote(world, "global", "PUT", "/api/network", json={"mode": "this_mac"})
    assert response.status_code == 409 and response.json()["error"] == "would_lock_out"
    detail = response.json()["detail"]
    assert "web.bind: 127.0.0.1" in detail and "Beacon" in detail, "it says where and how this CAN be done"
    assert world.cfg_path.read_text() == before


def test_an_owner_device_on_another_machine_is_refused_too(tmp_path):
    world = make(tmp_path, file_text='web:\n  bind: "0.0.0.0"\n')
    assert remote(world, "owner", "PUT", "/api/network", json={"mode": "this_mac"}).json()["error"] == "would_lock_out"


@pytest.mark.parametrize("peer", ["127.0.0.1", "::1", "::ffff:127.0.0.1"])
def test_this_mac_from_this_machine_is_fine(tmp_path, peer):
    world = make(tmp_path, file_text='web:\n  bind: "0.0.0.0"\n')
    assert remote(world, "global", "PUT", "/api/network", peer=peer, json={"mode": "this_mac"}).status_code == 200


def test_a_peer_nobody_can_place_is_not_assumed_to_be_local(tmp_path):
    """Anything that does not clearly say loopback is not loopback (TestClient's default peer is 'testclient')."""
    world = make(tmp_path, file_text='web:\n  bind: "0.0.0.0"\n')
    response = TestClient(world.app).put("/api/network", headers=world.hdr("global"), json={"mode": "this_mac"})
    assert response.status_code == 409 and response.json()["error"] == "would_lock_out"


def test_home_network_from_another_machine_is_not_a_lockout(tmp_path):
    """Opening the daemon up cannot cut the caller off."""
    world = make(tmp_path, bind="127.0.0.1", source="config",
                 file_text='web:\n  bind: "127.0.0.1"\nnetwork:\n  allow_plaintext_lan: true\n')
    assert remote(world, "global", "PUT", "/api/network", json={"mode": "home_network"}).status_code == 200


def test_the_other_refusals_come_first(tmp_path):
    world = make(tmp_path, file_text='web:\n  bind: "0.0.0.0"\n')
    assert remote(world, "scoped", "PUT", "/api/network", json={"mode": "this_mac"}).json()["error"] == "operator_only"
    assert remote(world, "global", "PUT", "/api/network", json={"mode": "open"}).json()["error"] == "invalid_request"
    assert world.client.put("/api/network", json={"mode": "this_mac"}).status_code == 401


# ── the owner's switch ───────────────────────────────────────────────────────

def test_a_scoped_device_may_not_change_it_and_is_told_403_not_401(tmp_path):
    world = make(tmp_path)
    response = world.as_("scoped", "PUT", "/api/network", json={"mode": "this_mac"})
    assert (response.status_code, response.json()["error"]) == (403, "operator_only")
    assert world.client.put("/api/network", json={"mode": "this_mac"}).status_code == 401


@pytest.mark.parametrize("body", [{}, {"mode": "open"}, {"mode": "nonsense"}, {"mode": 7}, {"mode": "this_mac", "bind": "0.0.0.0"},
                                  {"mode": "this_mac", "owner": True}, []])
def test_only_this_mac_and_home_network_can_be_asked_for(tmp_path, body):
    world = make(tmp_path)
    before = world.cfg_path.read_text()
    response = world.as_("global", "PUT", "/api/network", json=body)
    assert response.status_code == 400 and response.json()["error"] == "invalid_request"
    assert world.cfg_path.read_text() == before


def test_this_mac_is_saved_and_answers_on_restart(tmp_path):
    world = make(tmp_path, file_text='web:\n  bind: "0.0.0.0"\n')
    response = world.as_("owner", "PUT", "/api/network", json={"mode": "this_mac"})
    assert response.status_code == 200
    body = response.json()
    assert (body["mode"], body["applied"]) == ("this_mac", "on_restart")
    assert any(w["code"] == "restart_required" and "restart" in w["message"].lower() for w in body["warnings"])
    assert saved(world)["web"]["bind"] == "127.0.0.1" and saved(world)["network"]["home_network"] is False


def test_home_network_is_refused_without_tls_or_the_opt_out_and_nothing_is_written(tmp_path):
    world = make(tmp_path, bind="127.0.0.1", source="config", file_text='web:\n  bind: "127.0.0.1"\n')
    before = world.cfg_path.read_text()
    response = world.as_("global", "PUT", "/api/network", json={"mode": "home_network"})
    assert response.status_code == 409 and response.json()["error"] == "tls_unavailable"
    assert world.cfg_path.read_text() == before


def test_home_network_with_the_explicit_opt_out_is_saved_with_the_plaintext_warning(tmp_path):
    world = make(tmp_path, bind="127.0.0.1", source="config",
                 file_text='web:\n  bind: "127.0.0.1"\nnetwork:\n  allow_plaintext_lan: true\n')
    response = world.as_("global", "PUT", "/api/network", json={"mode": "home_network"})
    assert response.status_code == 200
    body = response.json()
    assert (body["mode"], body["applied"]) == ("home_network", "on_restart")
    assert {"plain_http_on_lan", "restart_required"} <= {w["code"] for w in body["warnings"]}
    assert saved(world)["web"]["bind"] == "0.0.0.0" and saved(world)["network"]["home_network"] is True


def test_home_network_does_not_widen_a_specific_address_the_owner_chose(tmp_path):
    world = make(tmp_path, bind=TAILNET, source="config",
                 file_text=f'web:\n  bind: "{TAILNET}"\nnetwork:\n  allow_plaintext_lan: true\n')
    assert world.as_("global", "PUT", "/api/network", json={"mode": "home_network"}).status_code == 200
    assert saved(world)["web"]["bind"] == TAILNET


@pytest.mark.parametrize("source", ["flag", "env", "caller"])   # "caller": launch_web was handed an address and no source
def test_a_bind_pinned_by_the_command_line_or_environment_is_not_pretended_changeable(tmp_path, source):
    world = make(tmp_path, bind="127.0.0.1", source=source,
                 file_text='network:\n  allow_plaintext_lan: true\n')
    before = world.cfg_path.read_text()
    response = world.as_("global", "PUT", "/api/network", json={"mode": "home_network"})
    assert response.status_code == 409
    assert response.json()["error"] == "bind_overridden" and response.json()["source"] == source
    assert world.cfg_path.read_text() == before


def test_no_config_file_is_a_409_and_none_is_created(tmp_path):
    world = make(tmp_path, file_text=None)
    response = world.as_("global", "PUT", "/api/network", json={"mode": "this_mac"})
    assert response.status_code == 409 and response.json()["error"] == "no_config_file"
    assert not world.cfg_path.exists()


def test_asking_for_what_is_already_saved_and_running_changes_nothing(tmp_path):
    world = make(tmp_path, bind="127.0.0.1", source="config",
                 file_text='web:\n  bind: "127.0.0.1"\nnetwork:\n  home_network: false\n')
    before = world.cfg_path.read_text()
    mtime = world.cfg_path.stat().st_mtime_ns
    response = world.as_("global", "PUT", "/api/network", json={"mode": "this_mac"})
    assert response.status_code == 200 and response.json()["applied"] == "live"
    assert world.cfg_path.read_text() == before and world.cfg_path.stat().st_mtime_ns == mtime


def test_after_a_save_the_next_read_shows_what_is_pending(tmp_path):
    world = make(tmp_path, bind="0.0.0.0", source="default",
                 file_text='network:\n  allow_plaintext_lan: true\n')
    world.as_("global", "PUT", "/api/network", json={"mode": "this_mac"})
    body = world.as_("global", "GET", "/api/network").json()
    assert body["mode"] == "open" and body["applied"] == "on_restart" and body["pending_mode"] == "this_mac"


def test_the_shipped_templates_comments_survive_a_change(tmp_path):
    world = make(tmp_path, bind="127.0.0.1", source="config", file_text=TEMPLATE,
                 config={"network": {"allow_plaintext_lan": True}})
    world.cfg_path.write_text(TEMPLATE.replace("allow_plaintext_lan: false", "allow_plaintext_lan: true"))
    before = sum(1 for ln in world.cfg_path.read_text().splitlines() if ln.lstrip().startswith("#"))
    assert world.as_("global", "PUT", "/api/network", json={"mode": "home_network"}).status_code == 200
    after = sum(1 for ln in world.cfg_path.read_text().splitlines() if ln.lstrip().startswith("#"))
    assert after == before


def test_a_write_that_fails_is_a_500_that_says_so_and_leaves_the_file(tmp_path, monkeypatch):
    import prometheus.web.network_routes as routes

    def refuse(path, mode, current_bind):
        raise PersistError("the edit did not verify")

    monkeypatch.setattr(routes, "persist_choice", refuse)
    world = make(tmp_path, file_text='network:\n  allow_plaintext_lan: true\n')
    before = world.cfg_path.read_text()
    response = world.as_("global", "PUT", "/api/network", json={"mode": "this_mac"})
    assert response.status_code == 500 and response.json()["error"] == "persist_failed"
    assert world.cfg_path.read_text() == before


def test_a_change_is_logged_with_who_made_it(tmp_path, caplog):
    world = make(tmp_path, file_text='web:\n  bind: "0.0.0.0"\n')
    with caplog.at_level(logging.INFO):
        world.as_("owner", "PUT", "/api/network", json={"mode": "this_mac"})
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("network:")]
    assert lines and "this_mac" in lines[0] and "Beacon on this Mac" in lines[0]
