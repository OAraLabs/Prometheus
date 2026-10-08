"""The routes that let Beacon, on the same Mac, pair with the one-time file secret.

* ``POST /api/setup/pair`` (setup mode, the daemon's first boot) accepts the secret in ``code``; the response
  shape is the one Beacon already parses and does not change.
* ``POST /api/pair/local`` (the running, configured daemon) accepts the same secret, so a Beacon reinstall
  is not a dead end. It is a deliberate exemption from the bearer middleware, with its own checks.

What is pinned is what keeps the exemption from being a hole: the peer AND the Host header must be
loopback, a request from a browser (an Origin header) is refused, a wrong value never touches the six-digit
code's attempt counter or lockout (and the six-digit lockout never blocks the secret), the secret is used
exactly once, nothing is logged, the feature does not exist unless this is the app install, and the
credential comes from ONE function so per-device credentials can replace it without touching the routes.
"""

from __future__ import annotations

import logging
import os

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.config import api_token as api_token_module  # noqa: E402
from prometheus.config import local_pairing as lp  # noqa: E402
from prometheus.config.api_token import TOKEN_ENV_VAR, resolve_api_token  # noqa: E402
from prometheus.web import setup_server  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402
from prometheus.web.setup_server import PairingState, create_setup_app  # noqa: E402

CODE = "042999"
DAEMON_TOKEN = "daemon-test-token-0123456789abcdef"


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    path = tmp_path / "env"
    monkeypatch.setenv("PROMETHEUS_ENV_FILE", str(path))
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)
    yield path
    os.environ.pop(TOKEN_ENV_VAR, None)


@pytest.fixture
def pairing_dir(tmp_path, monkeypatch):
    directory = tmp_path / "pairing"
    monkeypatch.setenv("PROMETHEUS_LOCAL_PAIRING_DIR", str(directory))
    return directory


@pytest.fixture
def secret(pairing_dir):
    return lp.mint_secret(pairing_dir)


def _client(app, *, peer="127.0.0.1", host="127.0.0.1:8123") -> TestClient:
    return TestClient(app, base_url=f"http://{host}", client=(peer, 50123))


def _setup_app(**pairing_kwargs):
    return create_setup_app(PairingState(code=CODE, **pairing_kwargs), api_port=8123, ws_port=8124)


def _daemon_app():
    return create_app({"web": {"api_token": DAEMON_TOKEN, "api_port": 8123, "ws_port": 8124}})


# ═══ setup mode: POST /api/setup/pair ═══════════════════════════════════════

class TestSetupModePairing:
    def test_the_secret_pairs_from_loopback_and_is_consumed(self, env_file, secret, pairing_dir):
        resp = _client(_setup_app()).post("/api/setup/pair", json={"code": secret})
        assert resp.status_code == 200, resp.text
        assert not (pairing_dir / "pair.secret").exists(), "used once, then gone"

    def test_the_response_shape_is_the_one_beacon_already_parses(self, env_file, secret):
        body = _client(_setup_app()).post("/api/setup/pair", json={"code": secret}).json()
        assert set(body) == {"token", "api_base_port", "ws_port"}
        assert body["api_base_port"] == 8123 and body["ws_port"] == 8124

    def test_the_credential_is_what_the_six_digit_path_returns_today(self, env_file, secret):
        body = _client(_setup_app()).post("/api/setup/pair", json={"code": secret}).json()
        assert body["token"] == resolve_api_token(None)[0] != ""

    def test_the_credential_comes_from_one_function(self, env_file, secret, monkeypatch):
        """Per-device credentials will replace the global token. That swap must be one function, not a
        change to every route that pairs."""
        monkeypatch.setattr(api_token_module, "issue_owner_credential", lambda config=None: "owner-device-token")
        body = _client(_setup_app()).post("/api/setup/pair", json={"code": secret}).json()
        assert body["token"] == "owner-device-token"

    def test_a_second_use_is_refused(self, env_file, secret):
        client = _client(_setup_app())
        assert client.post("/api/setup/pair", json={"code": secret}).status_code == 200
        again = client.post("/api/setup/pair", json={"code": secret})
        assert again.status_code == 401 and again.json()["error"] == "invalid_code"

    def test_a_peer_that_is_not_loopback_is_refused_and_the_secret_survives(self, env_file, secret, pairing_dir):
        resp = _client(_setup_app(), peer="192.168.1.20").post("/api/setup/pair", json={"code": secret})
        assert resp.status_code == 403 and resp.json()["error"] == "not_loopback"
        assert (pairing_dir / "pair.secret").exists()
        assert not env_file.exists(), "a refused request mints nothing"

    def test_a_foreign_host_header_is_refused_even_from_loopback(self, env_file, secret, pairing_dir):
        resp = _client(_setup_app(), host="evil.example").post("/api/setup/pair", json={"code": secret})
        assert resp.status_code == 403 and resp.json()["error"] == "not_loopback"
        assert (pairing_dir / "pair.secret").exists()

    def test_ipv6_loopback_is_loopback(self, env_file, secret):
        resp = _client(_setup_app(), peer="::1", host="[::1]:8123").post("/api/setup/pair", json={"code": secret})
        assert resp.status_code == 200

    def test_a_browser_request_is_refused(self, env_file, secret, pairing_dir):
        resp = _client(_setup_app()).post(
            "/api/setup/pair", json={"code": secret}, headers={"Origin": "https://evil.example"})
        assert resp.status_code == 400 and resp.json()["error"] == "browser_not_allowed"
        assert (pairing_dir / "pair.secret").exists()

    def test_wrong_secrets_never_burn_the_six_digit_attempts(self, env_file, secret):
        pairing = PairingState(code=CODE)
        client = _client(create_setup_app(pairing, api_port=8123, ws_port=8124))
        for _ in range(12):
            r = client.post("/api/setup/pair", json={"code": "Z" * 43})
            assert r.status_code == 401 and r.json()["error"] == "invalid_code"
        assert pairing.attempts_remaining == pairing.max_attempts, "a wrong SECRET is not a wrong CODE"
        assert client.post("/api/setup/pair", json={"code": secret}).status_code == 200, "and it never locks the secret"

    def test_a_locked_six_digit_pairing_does_not_lock_the_secret(self, env_file, secret):
        client = _client(_setup_app())
        for _ in range(5):
            client.post("/api/setup/pair", json={"code": "000000"})
        locked = client.post("/api/setup/pair", json={"code": CODE})
        assert locked.status_code == 403 and locked.json()["error"] == "pairing_locked"
        assert client.post("/api/setup/pair", json={"code": secret}).status_code == 200

    def test_an_expired_six_digit_code_does_not_expire_the_secret(self, env_file, secret):
        client = _client(_setup_app(ttl_seconds=-1))
        assert client.post("/api/setup/pair", json={"code": CODE}).json()["error"] == "pairing_expired"
        assert client.post("/api/setup/pair", json={"code": secret}).status_code == 200

    def test_the_six_digit_code_still_works_alongside(self, env_file, secret):
        resp = _client(_setup_app()).post("/api/setup/pair", json={"code": CODE})
        assert resp.status_code == 200 and set(resp.json()) == {"token", "api_base_port", "ws_port"}

    def test_the_six_digit_path_is_not_loopback_gated_as_before(self, env_file):
        """A terminal user pairing from another machine on the LAN keeps working: the loopback rule is
        for the secret only."""
        resp = _client(_setup_app(), peer="192.168.1.20", host="mac.local:8123").post(
            "/api/setup/pair", json={"code": CODE})
        assert resp.status_code == 200

    def test_without_the_app_install_a_long_value_is_just_a_wrong_code(self, env_file, tmp_path, monkeypatch):
        monkeypatch.delenv("PROMETHEUS_LOCAL_PAIRING_DIR", raising=False)
        monkeypatch.delenv("PROMETHEUS_INSTALL_KIND", raising=False)
        resp = _client(_setup_app()).post("/api/setup/pair", json={"code": "A" * 43})
        assert resp.status_code == 401 and resp.json()["error"] == "invalid_code"

    def test_a_missing_file_is_just_an_invalid_code(self, env_file, pairing_dir):
        resp = _client(_setup_app()).post("/api/setup/pair", json={"code": "A" * 43})
        assert resp.status_code == 401

    def test_no_secret_or_token_is_logged(self, env_file, secret, caplog):
        caplog.set_level(logging.DEBUG)
        body = _client(_setup_app()).post("/api/setup/pair", json={"code": secret}).json()
        joined = "\n".join(r.getMessage() for r in caplog.records)
        assert secret not in joined and body["token"] not in joined


class TestSetupModeMintsTheSecret:
    def _run(self, monkeypatch):
        async def _noop(*args, **kwargs):
            return None
        monkeypatch.setattr(setup_server, "_serve_setup_mode", _noop)
        monkeypatch.setattr(setup_server, "missing_web_stack", lambda: [])
        return setup_server.run_setup_mode()

    def test_the_app_install_gets_a_secret_at_startup(self, pairing_dir, monkeypatch, capsys):
        assert not pairing_dir.exists()
        self._run(monkeypatch)
        text = (pairing_dir / "pair.secret").read_text()
        assert len(text.strip()) == 43
        assert text.strip() not in capsys.readouterr().out, "the secret is never printed"

    def test_a_restart_before_pairing_keeps_the_same_secret(self, pairing_dir, monkeypatch):
        self._run(monkeypatch)
        first = (pairing_dir / "pair.secret").read_text()
        self._run(monkeypatch)
        assert (pairing_dir / "pair.secret").read_text() == first

    def test_other_installs_get_none(self, tmp_path, monkeypatch):
        monkeypatch.delenv("PROMETHEUS_LOCAL_PAIRING_DIR", raising=False)
        monkeypatch.delenv("PROMETHEUS_INSTALL_KIND", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        self._run(monkeypatch)
        assert not (tmp_path / "Library").exists() and not (tmp_path / ".local").exists()

    def test_terminal_users_still_get_the_six_digit_banner(self, pairing_dir, monkeypatch, capsys):
        self._run(monkeypatch)
        out = capsys.readouterr().out
        assert "pairing" in out.lower() and any(ch.isdigit() for ch in out)


# ═══ the running daemon: POST /api/pair/local ═══════════════════════════════

class TestPairLocalOnTheRunningDaemon:
    def test_it_pairs_without_a_bearer_and_returns_the_daemons_token(self, secret, pairing_dir):
        resp = _client(_daemon_app()).post("/api/pair/local", json={"code": secret})
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"token": DAEMON_TOKEN, "api_base_port": 8123, "ws_port": 8124}
        assert not (pairing_dir / "pair.secret").exists()

    def test_the_credential_comes_from_one_function(self, secret, monkeypatch):
        monkeypatch.setattr(api_token_module, "issue_owner_credential", lambda config=None: "owner-device-token")
        resp = _client(_daemon_app()).post("/api/pair/local", json={"code": secret})
        assert resp.json()["token"] == "owner-device-token"

    def test_every_other_path_still_needs_the_bearer(self, secret):
        client = _client(_daemon_app())
        assert client.get("/api/status").status_code == 401
        assert client.post("/api/pair/local/extra", json={"code": secret}).status_code == 401
        assert client.post("/api/pair/local/", json={"code": secret}).status_code in (401, 404, 307)

    def test_the_exemption_is_for_post_only(self, secret):
        assert _client(_daemon_app()).get("/api/pair/local").status_code == 401

    def test_a_second_use_is_refused(self, secret):
        client = _client(_daemon_app())
        assert client.post("/api/pair/local", json={"code": secret}).status_code == 200
        assert client.post("/api/pair/local", json={"code": secret}).status_code == 401

    def test_a_wrong_secret_is_refused_and_the_real_one_survives(self, secret, pairing_dir):
        client = _client(_daemon_app())
        for _ in range(20):
            assert client.post("/api/pair/local", json={"code": "Z" * 43}).status_code == 401
        assert client.post("/api/pair/local", json={"code": secret}).status_code == 200, "no lockout on the secret"

    def test_a_peer_that_is_not_loopback_is_refused(self, secret, pairing_dir):
        resp = _client(_daemon_app(), peer="10.0.0.9").post("/api/pair/local", json={"code": secret})
        assert resp.status_code == 403 and resp.json()["error"] == "not_loopback"
        assert (pairing_dir / "pair.secret").exists()

    def test_a_foreign_host_header_is_refused(self, secret):
        resp = _client(_daemon_app(), host="evil.example").post("/api/pair/local", json={"code": secret})
        assert resp.status_code == 403

    def test_a_browser_request_is_refused_and_carries_no_cors_headers(self, secret, pairing_dir):
        resp = _client(_daemon_app()).post(
            "/api/pair/local", json={"code": secret}, headers={"Origin": "https://evil.example"})
        assert resp.status_code == 400 and resp.json()["error"] == "browser_not_allowed"
        assert "access-control-allow-origin" not in {k.lower() for k in resp.headers}
        assert (pairing_dir / "pair.secret").exists()

    def test_it_does_not_exist_unless_this_is_the_app_install(self, monkeypatch):
        monkeypatch.delenv("PROMETHEUS_LOCAL_PAIRING_DIR", raising=False)
        monkeypatch.delenv("PROMETHEUS_INSTALL_KIND", raising=False)
        resp = _client(_daemon_app()).post("/api/pair/local", json={"code": "A" * 43})
        assert resp.status_code == 404

    def test_a_fresh_secret_from_pair_works_after_the_first_was_used(self, pairing_dir, secret):
        client = _client(_daemon_app())
        assert client.post("/api/pair/local", json={"code": secret}).status_code == 200
        fresh = lp.replace_secret(pairing_dir)                      # what `Prometheus --pair` does
        assert client.post("/api/pair/local", json={"code": fresh}).status_code == 200

    def test_a_body_that_is_not_json_is_a_400_not_a_traceback(self, secret):
        resp = _client(_daemon_app()).post("/api/pair/local", content=b"not json",
                                           headers={"Content-Type": "application/json"})
        assert resp.status_code == 400

    def test_no_secret_or_token_is_logged(self, secret, caplog):
        caplog.set_level(logging.DEBUG)
        _client(_daemon_app()).post("/api/pair/local", json={"code": secret})
        joined = "\n".join(r.getMessage() for r in caplog.records)
        assert secret not in joined and DAEMON_TOKEN not in joined
