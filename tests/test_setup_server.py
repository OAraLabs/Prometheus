"""Setup-mode pairing server (Onboarding Phase 1, item 2).

Covers the whole route surface and the pairing state machine against
the REAL app + REAL token machinery (env file isolated to tmp_path):

- pair happy path → token JSON + env-file persist
- token continuity → resolve_api_token finds the SAME token afterwards
- wrong code ×5 → attempts count down, then pairing locks until restart
- TTL expiry → rejected + locked
- one-time use → a second pair with the same code is rejected
- everything outside /api/setup/* → 403 with an honest JSON body
- find_config_file → no-mkdir config detection (the setup-mode gate)
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.config.api_token import TOKEN_ENV_VAR, resolve_api_token
from prometheus.config.env_file import parse_env_file, set_env_value
from prometheus.web.setup_server import (
    PairingState,
    create_setup_app,
    find_config_file,
    format_pairing_banner,
    mint_pairing_code,
    resolve_setup_port,
)

CODE = "042999"


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    """Isolate the env file + token env var — never the real machine.

    Yield-style with an explicit pop: a successful pair calls
    ensure_api_token, which exports the minted token into os.environ.
    monkeypatch.delenv on an ABSENT var records nothing to restore, so
    without the pop the token would leak into every later test in the
    process (found the hard way: 401s in unrelated web-API tests).
    """
    import os

    path = tmp_path / "env"
    monkeypatch.setenv("PROMETHEUS_ENV_FILE", str(path))
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)
    yield path
    os.environ.pop(TOKEN_ENV_VAR, None)


def make_client(code: str = CODE, **pairing_kwargs):
    pairing = PairingState(code=code, **pairing_kwargs)
    app = create_setup_app(pairing, api_port=8123, ws_port=8124)
    return TestClient(app), pairing


# ---------------------------------------------------------------------------
# /api/setup/status
# ---------------------------------------------------------------------------


class TestSetupStatus:
    def test_shape_when_available(self, env_file):
        client, _ = make_client()
        resp = client.get("/api/setup/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["setup_mode"] is True
        assert body["configured"] is False
        assert body["pairing"] == "available"
        assert isinstance(body["version"], str) and body["version"]

    def test_locked_after_use(self, env_file):
        client, _ = make_client()
        assert client.post("/api/setup/pair", json={"code": CODE}).status_code == 200
        assert client.get("/api/setup/status").json()["pairing"] == "locked"


# ---------------------------------------------------------------------------
# POST /api/setup/pair
# ---------------------------------------------------------------------------


class TestPairHappyPath:
    def test_returns_token_and_ports(self, env_file):
        client, _ = make_client()
        resp = client.post("/api/setup/pair", json={"code": CODE})
        assert resp.status_code == 200
        body = resp.json()
        assert body["token"] and len(body["token"]) > 20
        assert body["api_base_port"] == 8123
        assert body["ws_port"] == 8124

    def test_token_persisted_to_env_file(self, env_file):
        client, _ = make_client()
        token = client.post("/api/setup/pair", json={"code": CODE}).json()["token"]
        assert parse_env_file(env_file)[TOKEN_ENV_VAR] == token

    def test_token_continuity_into_resolve_api_token(self, env_file):
        """The pin: the real daemon later resolves the SAME token the
        paired client received — the client keeps working after setup."""
        client, _ = make_client()
        token = client.post("/api/setup/pair", json={"code": CODE}).json()["token"]
        resolved, source = resolve_api_token(None)
        assert resolved == token
        assert source in ("env", "env-file")

    def test_existing_env_file_token_is_reused_not_replaced(self, env_file):
        set_env_value(TOKEN_ENV_VAR, "pre-existing-token-value", env_file)
        client, _ = make_client()
        body = client.post("/api/setup/pair", json={"code": CODE}).json()
        assert body["token"] == "pre-existing-token-value"
        assert parse_env_file(env_file)[TOKEN_ENV_VAR] == "pre-existing-token-value"

    def test_correct_code_after_some_failures_still_pairs(self, env_file):
        client, _ = make_client()
        for _ in range(4):
            assert client.post(
                "/api/setup/pair", json={"code": "000000"}
            ).status_code == 401
        assert client.post("/api/setup/pair", json={"code": CODE}).status_code == 200


class TestPairFailures:
    def test_wrong_code_counts_down_then_locks(self, env_file):
        client, _ = make_client()
        for expected_remaining in (4, 3, 2, 1, 0):
            resp = client.post("/api/setup/pair", json={"code": "999998"})
            assert resp.status_code == 401
            body = resp.json()
            assert body["error"] == "invalid_code"
            assert body["attempts_remaining"] == expected_remaining
        # 6th attempt — even with the RIGHT code — is locked out.
        resp = client.post("/api/setup/pair", json={"code": CODE})
        assert resp.status_code == 403
        assert resp.json()["error"] == "pairing_locked"
        assert client.get("/api/setup/status").json()["pairing"] == "locked"
        # Nothing was persisted.
        assert TOKEN_ENV_VAR not in parse_env_file(env_file)

    def test_ttl_expiry_rejected_and_locked(self, env_file):
        client, _ = make_client(created_at=1_000.0, clock=lambda: 1_000.0 + 901)
        resp = client.post("/api/setup/pair", json={"code": CODE})
        assert resp.status_code == 403
        assert resp.json()["error"] == "pairing_expired"
        assert client.get("/api/setup/status").json()["pairing"] == "locked"

    def test_within_ttl_still_pairs(self, env_file):
        client, _ = make_client(created_at=1_000.0, clock=lambda: 1_000.0 + 899)
        assert client.post("/api/setup/pair", json={"code": CODE}).status_code == 200

    def test_one_time_use_reuse_rejected(self, env_file):
        client, _ = make_client()
        assert client.post("/api/setup/pair", json={"code": CODE}).status_code == 200
        resp = client.post("/api/setup/pair", json={"code": CODE})
        assert resp.status_code == 403
        assert resp.json()["error"] == "pairing_used"

    def test_missing_or_malformed_body_is_400_and_burns_no_attempt(self, env_file):
        client, pairing = make_client()
        assert client.post("/api/setup/pair", json={}).status_code == 400
        assert client.post("/api/setup/pair", json={"code": ""}).status_code == 400
        assert client.post(
            "/api/setup/pair", content=b"not json",
            headers={"content-type": "application/json"},
        ).status_code == 400
        assert pairing.attempts_remaining == 5  # only WRONG codes burn attempts
        # First actual wrong code still reports 4 remaining.
        resp = client.post("/api/setup/pair", json={"code": "000000"})
        assert resp.json()["attempts_remaining"] == 4


# ---------------------------------------------------------------------------
# Everything else → 403
# ---------------------------------------------------------------------------


class TestNonSetupRoutes403:
    @pytest.mark.parametrize("method,path", [
        ("GET", "/api/status"),
        ("GET", "/api/sessions"),
        ("POST", "/api/chat/send"),
        ("GET", "/api/files"),
        ("DELETE", "/api/sessions/abc"),
        ("GET", "/"),
        ("GET", "/docs"),
        ("GET", "/health"),
    ])
    def test_full_surface_is_unreachable(self, env_file, method, path):
        client, _ = make_client()
        resp = client.request(method, path)
        assert resp.status_code == 403
        body = resp.json()
        assert body["error"] == "setup_mode"
        assert "prometheus setup" in body["detail"]

    def test_403_even_with_a_valid_bearer_token(self, env_file):
        """Setup mode has no privileged surface — a token doesn't open it."""
        client, _ = make_client()
        token = client.post("/api/setup/pair", json={"code": CODE}).json()["token"]
        resp = client.get(
            "/api/status", headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 403
        assert resp.json()["error"] == "setup_mode"


# ---------------------------------------------------------------------------
# Pairing code + port helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_mint_pairing_code_is_six_digits(self):
        for _ in range(50):
            code = mint_pairing_code()
            assert len(code) == 6 and code.isdigit()

    def test_banner_contains_code_and_port_and_matches_house_style(self):
        banner = format_pairing_banner("123456", 8123)
        assert "123456" in banner
        assert "8123" in banner
        assert "SETUP MODE" in banner
        assert "=" * 68 in banner  # format_minted_banner's bar
        assert "prometheus setup" in banner

    def test_resolve_setup_port_env_override(self, monkeypatch):
        monkeypatch.setenv("PROMETHEUS_WEB_API_PORT", "18205")
        assert resolve_setup_port() == 18205
        monkeypatch.setenv("PROMETHEUS_WEB_API_PORT", "not-a-port")
        assert resolve_setup_port() == 8005
        monkeypatch.delenv("PROMETHEUS_WEB_API_PORT")
        assert resolve_setup_port() == 8005


# ---------------------------------------------------------------------------
# find_config_file — the setup-mode gate
# ---------------------------------------------------------------------------


class TestFindConfigFile:
    def test_explicit_path(self, tmp_path):
        cfg = tmp_path / "explicit.yaml"
        assert find_config_file(str(cfg)) is None
        cfg.write_text("web: {}\n")
        assert find_config_file(str(cfg)) == cfg

    def test_user_config_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path / "confdir"))
        monkeypatch.chdir(tmp_path)  # no repo-local config/ here
        assert find_config_file(None) is None
        # And detection MUST NOT have created the config dir (no-state rule).
        assert not (tmp_path / "confdir").exists()
        (tmp_path / "confdir").mkdir()
        (tmp_path / "confdir" / "prometheus.yaml").write_text("web: {}\n")
        assert find_config_file(None) == tmp_path / "confdir" / "prometheus.yaml"

    def test_repo_local_config_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path / "confdir"))
        monkeypatch.chdir(tmp_path)
        (tmp_path / "config").mkdir()
        repo_cfg = tmp_path / "config" / "prometheus.yaml"
        repo_cfg.write_text("web: {}\n")
        found = find_config_file(None)
        assert found is not None
        # daemon.load_config uses the same cwd-relative path — compare resolved.
        assert found.resolve() == repo_cfg.resolve()
