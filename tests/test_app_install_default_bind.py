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
