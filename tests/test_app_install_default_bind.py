"""An app install listens on this Mac only unless its config says otherwise — and its config CAN say otherwise.

Prometheus.app's launcher used to start the daemon with ``--bind 127.0.0.1``. A flag outranks the config file, so
the owner's network switch (``PUT /api/network``, which writes ``web.bind``) could never take effect on an app install.
The launcher no longer pins it. The lowest tier, the default, is what keeps an app install on this Mac: ``127.0.0.1``
when ``PROMETHEUS_INSTALL_KIND=app`` (set by the launcher), and ``0.0.0.0`` everywhere else as before (a Mac mini
reached over Tailscale with no ``web.bind`` must keep working). Precedence is unchanged: ``--bind`` >
``PROMETHEUS_WEB_BIND`` > ``web.bind`` > the default.
"""

from __future__ import annotations

from prometheus.web.bind import resolve_bind

APP = {"PROMETHEUS_INSTALL_KIND": "app"}


def test_an_app_install_with_no_bind_listens_on_this_mac_only():
    resolved = resolve_bind(None, env=APP)
    assert (resolved.address, resolved.source) == ("127.0.0.1", "default")
    resolved = resolve_bind({"web": {"enabled": True}}, env=APP)
    assert (resolved.address, resolved.source) == ("127.0.0.1", "default")


def test_any_other_install_keeps_every_interface_when_nothing_is_set():
    assert resolve_bind(None, env={}).address == "0.0.0.0"
    assert resolve_bind(None, env={"PROMETHEUS_INSTALL_KIND": "pip"}).address == "0.0.0.0"


def test_on_an_app_install_the_config_decides_so_the_switch_can_change_it():
    resolved = resolve_bind({"web": {"bind": "0.0.0.0"}}, env=APP)
    assert (resolved.address, resolved.source) == ("0.0.0.0", "config")


def test_the_flag_and_the_variable_still_outrank_everything():
    assert resolve_bind({"web": {"bind": "0.0.0.0"}}, flag="127.0.0.1", env=APP).source == "flag"
    env = {**APP, "PROMETHEUS_WEB_BIND": "0.0.0.0"}
    assert resolve_bind({"web": {"bind": "127.0.0.1"}}, env=env).source == "env"


def test_setup_mode_on_an_app_install_listens_on_this_mac_only(monkeypatch, tmp_path):
    from prometheus.web import setup_server

    monkeypatch.setenv("PROMETHEUS_INSTALL_KIND", "app")
    monkeypatch.delenv("PROMETHEUS_WEB_BIND", raising=False)
    monkeypatch.setenv("PROMETHEUS_ENV_FILE", str(tmp_path / "env"))
    resolved = setup_server.resolve_setup_choice()
    assert (resolved.address, resolved.source) == ("127.0.0.1", "default")


# ── plain HTTP never reaches the LAN on an app install unless the owner asked ──
#
# Independent review: with the --bind pin gone, an app install whose config names a LAN bind (by hand, or left over
# from home-network mode after the owner turned it off) would serve PLAIN HTTP to the network. On the app it now
# falls back to this Mac only unless network.allow_plaintext_lan is set. Home network on: its TLS listener takes
# the LAN (pairing PR 5) and plain HTTP stays on loopback anyway. Terminal installs keep today's behaviour.

from prometheus.web.bind import ResolvedBind  # noqa: E402

LAN = ResolvedBind("0.0.0.0", "config")


def app_install_lan_guard(*args, **kwargs):
    """Imported on use, so that before it exists each test fails on its own rather than at collection."""
    from prometheus.web.bind import app_install_lan_guard as guard

    return guard(*args, **kwargs)


def test_an_app_install_with_a_lan_bind_and_no_home_network_listens_on_this_mac_only():
    guarded = app_install_lan_guard(LAN, {"web": {"bind": "0.0.0.0"}}, env=APP)
    assert guarded.address == "127.0.0.1"
    assert guarded.source == "config", "the file still decides, so the owner's switch can still change it"


def test_the_owners_explicit_plain_http_opt_in_is_honoured():
    config = {"network": {"allow_plaintext_lan": True}}
    assert app_install_lan_guard(LAN, config, env=APP).address == "0.0.0.0"


def test_home_network_keeps_the_lan_bind_for_its_tls_listener():
    assert app_install_lan_guard(LAN, {"network": {"home_network": True}}, env=APP).address == "0.0.0.0"


def test_a_switch_that_is_not_a_boolean_is_not_an_opt_in():
    config = {"network": {"allow_plaintext_lan": "yes", "home_network": "true"}}
    assert app_install_lan_guard(LAN, config, env=APP).address == "127.0.0.1"


def test_a_variable_is_guarded_too_on_the_app():
    assert app_install_lan_guard(ResolvedBind("0.0.0.0", "env"), None, env=APP).address == "127.0.0.1"


def test_terminal_installs_keep_todays_behaviour():
    assert app_install_lan_guard(LAN, {}, env={}).address == "0.0.0.0"
    assert app_install_lan_guard(LAN, {}, env={"PROMETHEUS_INSTALL_KIND": "pip"}).address == "0.0.0.0"


def test_a_loopback_bind_is_left_alone():
    loop = ResolvedBind("127.0.0.1", "config")
    assert app_install_lan_guard(loop, {}, env=APP) == loop


def test_the_daemon_and_setup_mode_apply_it(monkeypatch, tmp_path):
    import ast
    from pathlib import Path

    import prometheus.daemon as daemon
    from prometheus.web import setup_server

    tree = ast.parse(Path(daemon.__file__).read_text(encoding="utf-8"))
    names = {(n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None))
             for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert "app_install_lan_guard" in names, "the daemon resolves its bind without the app-install guard"
    monkeypatch.setenv("PROMETHEUS_INSTALL_KIND", "app")
    monkeypatch.setenv("PROMETHEUS_WEB_BIND", "0.0.0.0")
    monkeypatch.setenv("PROMETHEUS_ENV_FILE", str(tmp_path / "env"))
    assert setup_server.resolve_setup_choice().address == "127.0.0.1"


# ── one truth: what /api/network and oara doctor say is what the app daemon binds ──
#
# The guard and the app default decide what an app install binds. GET /api/network (#700) predicts what a restart
# brings back from the file, and oara doctor says what the daemon will listen on. Both apply the same default and
# guard, or every app install is told a restart is pending that would change nothing.

def test_the_guard_reads_the_switches_through_network_settings(monkeypatch):
    from prometheus.web import network

    seen = []
    real = network.NetworkSettings.from_config

    def spy(config, **kw):
        seen.append(config)
        return real(config, **kw)

    monkeypatch.setattr(network.NetworkSettings, "from_config", staticmethod(spy))
    app_install_lan_guard(LAN, {"network": {"home_network": True}}, env=APP)
    assert seen, "the guard reads network.* itself; #700's NetworkSettings is the one reader"


def _network_on_the_app(tmp_path, monkeypatch, *, bind: str, source: str, file_text: str) -> dict:
    import pytest

    pytest.importorskip("fastapi")
    from tests.test_network_routes import make

    world = make(tmp_path, bind=bind, source=source, file_text=file_text)
    monkeypatch.setenv("PROMETHEUS_INSTALL_KIND", "app")
    return world.as_("global", "GET", "/api/network").json()


def test_an_app_install_with_no_bind_saved_has_no_restart_pending(tmp_path, monkeypatch):
    body = _network_on_the_app(tmp_path, monkeypatch, bind="127.0.0.1", source="default",
                               file_text="web:\n  enabled: true\n")
    assert (body["mode"], body["applied"]) == ("this_mac", "live"), body


def test_a_lan_bind_the_guard_holds_back_is_not_a_restart_pending(tmp_path, monkeypatch):
    body = _network_on_the_app(tmp_path, monkeypatch, bind="127.0.0.1", source="config",
                               file_text="web:\n  enabled: true\n  bind: 0.0.0.0\n")
    assert (body["mode"], body["applied"]) == ("this_mac", "live"), body


def test_doctor_says_what_the_app_daemon_will_listen_on():
    from prometheus.cli.doctor import check_web_bind

    row = check_web_bind({"web": {"enabled": True, "bind": "0.0.0.0"}}, env=APP)
    assert "127.0.0.1" in row.message and "all interfaces" not in row.message, row.message
    assert "allow_plaintext_lan" in row.message + (row.fix or ""), "say how the owner gets the LAN back"
