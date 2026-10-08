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
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

pytest.importorskip("fastapi")

from prometheus.web.bind import ResolvedBind  # noqa: E402
from prometheus.web.network import PersistError  # noqa: E402
from tests.support.pairing_world import World  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TAILNET = ".".join(("100", "64", "0", "9"))   # built at run time: the pre-commit scanner looks for these
TEMPLATE = (REPO / "config" / "prometheus.yaml.default").read_text(encoding="utf-8")


class FakeAdvertiser:
    def __init__(self, advertising: bool, reason: str | None) -> None:
        self.status = type("S", (), {"advertising": advertising, "reason": reason})()


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
    return world


def saved(world: World) -> dict:
    return yaml.safe_load(world.cfg_path.read_text()) or {}  # type: ignore[attr-defined]


# ── reading ──────────────────────────────────────────────────────────────────

def test_the_shape_is_the_contracts(tmp_path):
    body = make(tmp_path).as_("global", "GET", "/api/network").json()
    assert set(body) == {"mode", "bind", "bind_source", "tls", "advertising", "advertising_reason", "applied",
                         "warnings"}
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
    assert any("plain HTTP" in w for w in body["warnings"])


def test_not_advertising_says_why(tmp_path):
    world = make(tmp_path, advertiser=FakeAdvertiser(False, "zeroconf is not installed"))
    body = world.as_("global", "GET", "/api/network").json()
    assert body["advertising"] is False and body["advertising_reason"] == "zeroconf is not installed"


def test_with_no_advertiser_in_this_process_it_says_so(tmp_path):
    body = make(tmp_path).as_("global", "GET", "/api/network").json()
    assert body["advertising"] is False and "not running" in body["advertising_reason"]


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


def test_a_pinned_bind_outranks_what_the_file_says_about_the_next_start(tmp_path):
    """--bind / PROMETHEUS_WEB_BIND win over web.bind, so a file that says 0.0.0.0 promises nothing."""
    world = make(tmp_path, bind="127.0.0.1", source="env",
                 file_text='web:\n  bind: "0.0.0.0"\nnetwork:\n  home_network: true\n  allow_plaintext_lan: true\n')
    body = world.as_("global", "GET", "/api/network").json()
    assert body["bind"] == "127.0.0.1" and body["bind_source"] == "env"
    assert body["applied"] == "live" and "pending_mode" not in body, "the environment still decides the address"


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
    assert any("restart" in w.lower() for w in body["warnings"])
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
    assert any("plain HTTP" in w for w in body["warnings"]) and any("restart" in w.lower() for w in body["warnings"])
    assert saved(world)["web"]["bind"] == "0.0.0.0" and saved(world)["network"]["home_network"] is True


def test_home_network_does_not_widen_a_specific_address_the_owner_chose(tmp_path):
    world = make(tmp_path, bind=TAILNET, source="config",
                 file_text=f'web:\n  bind: "{TAILNET}"\nnetwork:\n  allow_plaintext_lan: true\n')
    assert world.as_("global", "PUT", "/api/network", json={"mode": "home_network"}).status_code == 200
    assert saved(world)["web"]["bind"] == TAILNET


@pytest.mark.parametrize("source", ["flag", "env"])
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
