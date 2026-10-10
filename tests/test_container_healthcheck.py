"""``python -m prometheus.web.healthcheck`` — the container image's HEALTHCHECK.

A fresh container has no config, so it boots into SETUP MODE, which serves ``/api/setup/*`` and not
``/health``. Once paired and configured, the same process falls through into the real daemon, which
serves ``/health`` and not ``/api/setup/*``. One probe has to call both of those healthy, and nothing
else: a closed port, a 500, a ``/health`` that does not say ``ok``, or a setup-status page that does
not say ``setup_mode`` is an unhealthy container.

Where to probe comes from the daemon's own resolvers, never a copy of them:

* config found → ``web.api_port`` (default 8005) and :func:`prometheus.web.bind.resolve_bind`;
* no config → setup mode's :func:`resolve_setup_port` (``PROMETHEUS_WEB_API_PORT``) and
  :func:`resolve_setup_choice`;
* a wildcard bind (``0.0.0.0``, ``::``) is probed on the loopback address of the same family.

It needs no token: both endpoints are the unauthenticated ones.
"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from prometheus.web import healthcheck


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Server:
    """A tiny HTTP server whose routes the test decides."""

    def __init__(self, routes: dict[str, tuple[int, object]]):
        routes_ = routes
        seen: list[str] = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 — http.server's name
                seen.append(self.path)
                status, body = routes_.get(self.path, (404, {"detail": "Not Found"}))
                raw = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *_a):
                pass

        self.seen = seen
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("PROMETHEUS_WEB_API_PORT", raising=False)
    monkeypatch.delenv("PROMETHEUS_WEB_BIND", raising=False)


def _write_config(tmp_path, monkeypatch, web: dict) -> None:
    import yaml

    cfg_dir = tmp_path / "prom-config"
    cfg_dir.mkdir(exist_ok=True)
    (cfg_dir / "prometheus.yaml").write_text(yaml.safe_dump({"web": web}), encoding="utf-8")
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(cfg_dir))


def test_a_configured_daemon_is_healthy_when_health_says_ok(tmp_path, monkeypatch, capsys):
    with _Server({"/health": (200, {"status": "ok", "service": "prometheus"})}) as srv:
        _write_config(tmp_path, monkeypatch, {"enabled": True, "api_port": srv.port})
        assert healthcheck.main([]) == 0
    assert srv.seen == ["/health"]
    assert "ok" in capsys.readouterr().out


def test_setup_mode_is_healthy(tmp_path, monkeypatch):
    with _Server({"/api/setup/status": (200, {"setup_mode": True, "configured": False})}) as srv:
        monkeypatch.setenv("PROMETHEUS_WEB_API_PORT", str(srv.port))
        assert healthcheck.main([]) == 0
    assert "/api/setup/status" in srv.seen


def test_setup_completing_in_place_is_still_healthy(tmp_path, monkeypatch):
    # POST /api/setup/complete writes the config, then the same process starts the real daemon on the
    # config's port. Between those two moments the config exists and setup mode is still answering.
    with _Server({"/api/setup/status": (200, {"setup_mode": True, "configured": True})}) as srv:
        _write_config(tmp_path, monkeypatch, {"enabled": True, "api_port": srv.port})
        assert healthcheck.main([]) == 0


@pytest.mark.parametrize("routes", [
    {"/health": (200, {"status": "degraded"})},
    {"/health": (500, {"detail": "boom"})},
    {"/health": (200, b"not json")},
    {"/api/setup/status": (200, {"setup_mode": False})},
    {},
])
def test_anything_else_is_unhealthy(routes, tmp_path, monkeypatch, capsys):
    with _Server(routes) as srv:
        monkeypatch.setenv("PROMETHEUS_WEB_API_PORT", str(srv.port))
        _write_config(tmp_path, monkeypatch, {"enabled": True, "api_port": srv.port})
        assert healthcheck.main([]) == 1
    assert capsys.readouterr().out.strip(), "an unhealthy probe must say why (docker inspect shows it)"


def test_nothing_listening_is_unhealthy(monkeypatch, capsys):
    monkeypatch.setenv("PROMETHEUS_WEB_API_PORT", str(_free_port()))
    assert healthcheck.main([]) == 1
    assert capsys.readouterr().out.strip()


def test_the_configured_port_wins_over_the_setup_port(tmp_path, monkeypatch):
    with _Server({"/health": (200, {"status": "ok"})}) as srv:
        _write_config(tmp_path, monkeypatch, {"enabled": True, "api_port": srv.port})
        monkeypatch.setenv("PROMETHEUS_WEB_API_PORT", str(_free_port()))
        assert healthcheck.main([]) == 0


def test_the_default_port_is_8005_in_both_modes(tmp_path, monkeypatch):
    assert healthcheck.probe_target().port == 8005
    _write_config(tmp_path, monkeypatch, {"enabled": True})
    assert healthcheck.probe_target().port == 8005


@pytest.mark.parametrize(("bind", "host"), [
    ("0.0.0.0", "127.0.0.1"),
    ("::", "::1"),
    ("127.0.0.1", "127.0.0.1"),
    ("10.1.2.3", "10.1.2.3"),
])
def test_a_wildcard_bind_is_probed_on_loopback(bind, host, monkeypatch):
    monkeypatch.setenv("PROMETHEUS_WEB_BIND", bind)
    assert healthcheck.probe_target().host == host


def test_the_config_bind_is_honoured(tmp_path, monkeypatch):
    _write_config(tmp_path, monkeypatch, {"enabled": True, "bind": "127.0.0.1"})
    assert healthcheck.probe_target().host == "127.0.0.1"


def test_it_sends_no_credentials(tmp_path, monkeypatch):
    seen_headers: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            seen_headers.append(dict(self.headers))
            raw = json.dumps({"status": "ok"}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *_a):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        monkeypatch.setenv("PROMETHEUS_API_TOKEN", "must-not-be-sent")
        _write_config(tmp_path, monkeypatch, {"enabled": True, "api_port": httpd.server_address[1]})
        assert healthcheck.main([]) == 0
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert seen_headers and not any("Authorization" in h for h in seen_headers)
