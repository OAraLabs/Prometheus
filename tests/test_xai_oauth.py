"""Tests for xAI SuperGrok OAuth credential source + registry wiring.

All network I/O is mocked — no real xAI calls. The token store is redirected
to a tmp dir via get_config_dir patching.
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def store_dir(tmp_path):
    """Redirect xai_oauth's token store into a tmp dir."""
    with patch("prometheus.providers.xai_oauth.get_config_dir", return_value=tmp_path):
        yield tmp_path


def _write_store(tmp_path, **overrides):
    store = {
        "access_token": "tok-fresh",
        "refresh_token": "refresh-1",
        "expires_at": time.time() + 3600,
        "token_type": "Bearer",
        "token_endpoint": "https://auth.x.ai/oauth2/token",
        "obtained_at": time.time(),
    }
    store.update(overrides)
    (tmp_path / "xai_oauth.json").write_text(json.dumps(store))
    return store


class TestTokenStore:
    def test_not_logged_in_when_no_store(self, store_dir):
        from prometheus.providers import xai_oauth
        assert xai_oauth.is_logged_in() is False
        assert xai_oauth.get_access_token() is None

    def test_valid_token_returned_without_refresh(self, store_dir):
        from prometheus.providers import xai_oauth
        _write_store(store_dir)
        with patch("prometheus.providers.xai_oauth.httpx.Client") as client:
            assert xai_oauth.get_access_token() == "tok-fresh"
            client.assert_not_called()  # fresh token => no network

    def test_logout_removes_store(self, store_dir):
        from prometheus.providers import xai_oauth
        _write_store(store_dir)
        assert xai_oauth.is_logged_in() is True
        assert xai_oauth.logout() is True
        assert xai_oauth.is_logged_in() is False
        assert xai_oauth.logout() is False  # idempotent

    def test_store_written_owner_only(self, store_dir):
        from prometheus.providers import xai_oauth
        xai_oauth._persist_tokens(
            {"access_token": "a", "refresh_token": "r", "expires_in": 3600},
            "https://auth.x.ai/oauth2/token",
        )
        mode = (store_dir / "xai_oauth.json").stat().st_mode & 0o777
        assert mode == 0o600


class TestRefresh:
    def test_expired_token_triggers_refresh(self, store_dir):
        from prometheus.providers import xai_oauth
        _write_store(store_dir, access_token="tok-old", expires_at=time.time() - 10)

        resp = MagicMock(status_code=200)
        resp.json.return_value = {
            "access_token": "tok-new",
            "refresh_token": "refresh-2",
            "expires_in": 21600,
        }
        client = MagicMock()
        client.post.return_value = resp
        with patch("prometheus.providers.xai_oauth.httpx.Client") as ctor:
            ctor.return_value.__enter__.return_value = client
            assert xai_oauth.get_access_token() == "tok-new"

        # New tokens persisted (incl. rotated refresh token).
        saved = json.loads((store_dir / "xai_oauth.json").read_text())
        assert saved["access_token"] == "tok-new"
        assert saved["refresh_token"] == "refresh-2"

    def test_refresh_failure_returns_none(self, store_dir):
        from prometheus.providers import xai_oauth
        _write_store(store_dir, expires_at=time.time() - 10)

        resp = MagicMock(status_code=400, text="bad refresh")
        client = MagicMock()
        client.post.return_value = resp
        with patch("prometheus.providers.xai_oauth.httpx.Client") as ctor:
            ctor.return_value.__enter__.return_value = client
            # Degrades to None (caller falls back to XAI_API_KEY) rather than raising.
            assert xai_oauth.get_access_token() is None


class TestRegistryPrecedence:
    def test_oauth_wins_over_api_key(self, store_dir, monkeypatch):
        from prometheus.providers.registry import _resolve_xai_credential
        _write_store(store_dir)
        monkeypatch.setenv("XAI_API_KEY", "sk-key")
        cred = _resolve_xai_credential({"provider": "xai", "api_key_env": "XAI_API_KEY"})
        assert callable(cred)          # OAuth path => callable bearer
        assert cred() == "tok-fresh"   # resolves to the OAuth token, not the key

    def test_fallback_to_key_when_not_logged_in(self, store_dir, monkeypatch):
        from prometheus.providers.registry import _resolve_xai_credential
        monkeypatch.setenv("XAI_API_KEY", "sk-key")
        cred = _resolve_xai_credential({"provider": "xai", "api_key_env": "XAI_API_KEY"})
        assert cred == "sk-key"        # static key path

    def test_callable_falls_back_to_key_on_oauth_outage(self, store_dir, monkeypatch):
        from prometheus.providers.registry import _resolve_xai_credential
        _write_store(store_dir, expires_at=time.time() - 10)
        monkeypatch.setenv("XAI_API_KEY", "sk-key")

        resp = MagicMock(status_code=400, text="bad")
        client = MagicMock()
        client.post.return_value = resp
        with patch("prometheus.providers.xai_oauth.httpx.Client") as ctor:
            ctor.return_value.__enter__.return_value = client
            cred = _resolve_xai_credential({"provider": "xai", "api_key_env": "XAI_API_KEY"})
            assert callable(cred)
            assert cred() == "sk-key"  # OAuth refresh failed -> env key


class TestStatusAndDeviceFlow:
    def test_token_status_logged_out(self, store_dir):
        from prometheus.providers import xai_oauth
        assert xai_oauth.token_status() == {"logged_in": False, "expires_at": None}

    def test_token_status_logged_in_no_network(self, store_dir):
        from prometheus.providers import xai_oauth
        exp = time.time() + 3600
        _write_store(store_dir, expires_at=exp)
        with patch("prometheus.providers.xai_oauth.httpx.Client") as client:
            st = xai_oauth.token_status()
            client.assert_not_called()  # status never hits the network
        assert st["logged_in"] is True
        assert st["expires_at"] == exp

    def test_begin_returns_code_without_waiting(self, store_dir):
        from prometheus.providers import xai_oauth
        get_resp = MagicMock(status_code=200)
        get_resp.json.return_value = {"token_endpoint": "https://auth.x.ai/oauth2/token"}
        get_resp.raise_for_status.return_value = None
        post_resp = MagicMock(status_code=200)
        post_resp.json.return_value = {
            "device_code": "dev-123", "user_code": "ABCD-EFGH",
            "verification_uri": "https://accounts.x.ai/device",
            "verification_uri_complete": "https://accounts.x.ai/device?user_code=ABCD-EFGH",
            "expires_in": 1800, "interval": 5,
        }
        client = MagicMock()
        client.get.return_value = get_resp
        client.post.return_value = post_resp
        with patch("prometheus.providers.xai_oauth.httpx.Client") as ctor:
            ctor.return_value.__enter__.return_value = client
            begin = xai_oauth.begin_device_login()
        assert begin["user_code"] == "ABCD-EFGH"
        assert begin["verification_uri"].endswith("user_code=ABCD-EFGH")
        assert begin["device_code"] == "dev-123"
        assert begin["token_endpoint"] == "https://auth.x.ai/oauth2/token"

    def test_complete_persists_on_approval(self, store_dir):
        from prometheus.providers import xai_oauth
        begin = {
            "device_code": "dev-123", "expires_in": 1800, "interval": 1,
            "token_endpoint": "https://auth.x.ai/oauth2/token",
        }
        resp = MagicMock(status_code=200)
        resp.json.return_value = {
            "access_token": "tok-live", "refresh_token": "refresh-9", "expires_in": 21600,
        }
        client = MagicMock()
        client.post.return_value = resp
        with patch("prometheus.providers.xai_oauth.httpx.Client") as ctor:
            ctor.return_value.__enter__.return_value = client
            store = xai_oauth.complete_device_login(begin)
        assert store["access_token"] == "tok-live"
        assert xai_oauth.token_status()["logged_in"] is True


class TestOAuthEndpoints:
    def _client(self):
        import pytest as _pytest
        _pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient
        from prometheus.web.server import create_app
        return TestClient(create_app({"model": {"model": "qwen3.5-32b", "provider": "local"}}))

    def test_status_logged_out(self):
        with patch("prometheus.providers.xai_oauth.token_status",
                   return_value={"logged_in": False, "expires_at": None}):
            r = self._client().get("/api/providers/xai/oauth")
        assert r.status_code == 200
        body = r.json()
        assert body["logged_in"] is False
        assert body["pending"] is False

    def test_login_returns_code_immediately(self):
        fake_begin = {
            "verification_uri": "https://accounts.x.ai/device?user_code=WXYZ-1234",
            "user_code": "WXYZ-1234", "expires_in": 1800, "interval": 5,
            "device_code": "d", "token_endpoint": "https://auth.x.ai/oauth2/token",
        }
        with patch("prometheus.providers.xai_oauth.begin_device_login", return_value=fake_begin), \
             patch("prometheus.providers.xai_oauth.complete_device_login", return_value={}):
            r = self._client().post("/api/providers/xai/oauth/login")
        assert r.status_code == 200
        body = r.json()
        assert body["pending"] is True
        assert body["user_code"] == "WXYZ-1234"

    def test_login_begin_failure_is_502(self):
        with patch("prometheus.providers.xai_oauth.begin_device_login",
                   side_effect=RuntimeError("xAI down")):
            r = self._client().post("/api/providers/xai/oauth/login")
        assert r.status_code == 502
        assert "xAI down" in r.json()["error"]

    def test_logout(self):
        with patch("prometheus.providers.xai_oauth.logout", return_value=True):
            r = self._client().request("DELETE", "/api/providers/xai/oauth")
        assert r.status_code == 200
        assert r.json()["logged_out"] is True


class TestCredentialStatus:
    """providers.credentials.credential_status — subscription vs API key,
    subscription ALWAYS preferred (mirrors registry runtime precedence)."""

    def test_subscription_wins_over_api_key(self, monkeypatch):
        import os
        from prometheus.providers.credentials import credential_status
        monkeypatch.setenv("XAI_API_KEY", "sk-x")
        with patch("prometheus.providers.xai_oauth.is_logged_in", return_value=True):
            cred = credential_status("xai", "XAI_API_KEY")
        assert cred["mode"] == "subscription"
        assert cred["detail"] == "SuperGrok subscription"
        assert cred["has_subscription"] is True
        assert cred["has_api_key"] is True

    def test_api_key_when_not_logged_in(self, monkeypatch):
        from prometheus.providers.credentials import credential_status
        monkeypatch.setenv("XAI_API_KEY", "sk-x")
        with patch("prometheus.providers.xai_oauth.is_logged_in", return_value=False):
            cred = credential_status("xai", "XAI_API_KEY")
        assert cred["mode"] == "api_key"
        assert cred["detail"] == "API key (XAI_API_KEY)"

    def test_none_when_no_credential(self, monkeypatch):
        from prometheus.providers.credentials import credential_status
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        with patch("prometheus.providers.xai_oauth.is_logged_in", return_value=False):
            cred = credential_status("xai", "XAI_API_KEY")
        assert cred["mode"] is None
        assert cred["detail"] is None
        # Hint still populated so error messages can point at the subscription.
        assert cred["subscription_label"] == "SuperGrok subscription"

    def test_provider_without_subscription_source_never_probes(self, monkeypatch):
        from prometheus.providers.credentials import credential_status
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-d")
        with patch("prometheus.providers.xai_oauth.is_logged_in") as probe:
            cred = credential_status("deepseek", "DEEPSEEK_API_KEY")
        probe.assert_not_called()
        assert cred["mode"] == "api_key"
        assert cred["subscription_label"] is None


class TestModelCatalogAvailability:
    """GET /api/models must report xai available when OAuth is logged in,
    even with no XAI_API_KEY set (PRs #99/#100), and say which credential
    each preset would use via the `auth` field."""

    def _client(self):
        import pytest as _pytest
        _pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient
        from prometheus.router.model_router import ModelRouter, RouterConfig
        from prometheus.web.server import create_app
        router = ModelRouter(
            RouterConfig(),
            primary_provider=object(),
            primary_adapter=object(),
            primary_model="qwen3.5-32b",
        )
        cfg = {"model": {"model": "qwen3.5-32b", "provider": "local"}}
        return TestClient(create_app(cfg, model_router=router))

    def _xai_entry(self, client):
        r = client.get("/api/models")
        assert r.status_code == 200
        return next(m for m in r.json()["models"] if m["key"] == "xai")

    def test_xai_available_via_oauth_without_api_key(self):
        import os
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("XAI_API_KEY", None)
            with patch("prometheus.providers.xai_oauth.is_logged_in", return_value=True):
                entry = self._xai_entry(self._client())
        assert entry["available"] is True
        assert entry["auth"] == "subscription"

    def test_xai_unavailable_when_no_key_and_logged_out(self):
        import os
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("XAI_API_KEY", None)
            with patch("prometheus.providers.xai_oauth.is_logged_in", return_value=False):
                entry = self._xai_entry(self._client())
        assert entry["available"] is False
        assert entry["auth"] is None

    def test_xai_api_key_only_reports_api_key_auth(self):
        import os
        with patch.dict(os.environ, {"XAI_API_KEY": "sk-xai-test"}, clear=False):
            with patch("prometheus.providers.xai_oauth.is_logged_in", return_value=False):
                entry = self._xai_entry(self._client())
        assert entry["available"] is True
        assert entry["auth"] == "api_key"

    def test_xai_subscription_wins_over_api_key(self):
        import os
        with patch.dict(os.environ, {"XAI_API_KEY": "sk-xai-test"}, clear=False):
            with patch("prometheus.providers.xai_oauth.is_logged_in", return_value=True):
                entry = self._xai_entry(self._client())
        assert entry["available"] is True
        assert entry["auth"] == "subscription"  # subscription preferred over key


class TestGatewaySlashGate:
    """cmd_provider_override's early key gate must honor SuperGrok OAuth for
    /xai (Telegram/Slack/Discord share this core) — a logged-in store passes
    with no XAI_API_KEY set."""

    def _agent_loop_with_router(self):
        agent_loop = MagicMock()
        router = MagicMock()
        router.config.overrides_enabled = True
        agent_loop._model_router = router
        return agent_loop

    def test_xai_passes_gate_via_oauth_and_reports_subscription(self):
        import os
        from prometheus.gateway.commands import cmd_provider_override
        agent_loop = self._agent_loop_with_router()
        with patch.dict(os.environ, {}, clear=True), \
             patch("prometheus.providers.xai_oauth.is_logged_in", return_value=True):
            text, applied = cmd_provider_override(agent_loop, {}, "telegram:1", "xai")
        assert applied is True
        assert "Switched to" in text
        assert "Auth: SuperGrok subscription" in text
        agent_loop._model_router.set_override.assert_called_once()

    def test_xai_blocked_when_no_key_and_logged_out(self):
        import os
        from prometheus.gateway.commands import cmd_provider_override
        agent_loop = self._agent_loop_with_router()
        with patch.dict(os.environ, {}, clear=True), \
             patch("prometheus.providers.xai_oauth.is_logged_in", return_value=False):
            text, applied = cmd_provider_override(agent_loop, {}, "telegram:1", "xai")
        assert applied is False
        assert "XAI_API_KEY" in text
        # The error must offer the subscription path too.
        assert "SuperGrok subscription" in text
        agent_loop._model_router.set_override.assert_not_called()

    def test_non_xai_provider_does_not_probe_oauth(self):
        import os
        from prometheus.gateway.commands import cmd_provider_override
        agent_loop = self._agent_loop_with_router()
        with patch.dict(os.environ, {}, clear=True), \
             patch("prometheus.providers.xai_oauth.is_logged_in") as probe:
            text, applied = cmd_provider_override(agent_loop, {}, "telegram:1", "deepseek")
        assert applied is False  # key missing, and no subscription source
        assert "SuperGrok" not in text  # xai's subscription isn't offered for deepseek
        probe.assert_not_called()

    def test_xai_api_key_only_reports_api_key(self):
        import os
        from prometheus.gateway.commands import cmd_provider_override
        agent_loop = self._agent_loop_with_router()
        with patch.dict(os.environ, {"XAI_API_KEY": "sk-x"}, clear=True), \
             patch("prometheus.providers.xai_oauth.is_logged_in", return_value=False):
            text, applied = cmd_provider_override(agent_loop, {}, "telegram:1", "xai")
        assert applied is True
        assert "Auth: API key (XAI_API_KEY)" in text

    def test_xai_subscription_wins_over_api_key_in_reply(self):
        import os
        from prometheus.gateway.commands import cmd_provider_override
        agent_loop = self._agent_loop_with_router()
        with patch.dict(os.environ, {"XAI_API_KEY": "sk-x"}, clear=True), \
             patch("prometheus.providers.xai_oauth.is_logged_in", return_value=True):
            text, applied = cmd_provider_override(agent_loop, {}, "telegram:1", "xai")
        assert applied is True
        assert "Auth: SuperGrok subscription" in text  # sub preferred over key

    def test_other_provider_reports_api_key_auth(self):
        import os
        from prometheus.gateway.commands import cmd_provider_override
        agent_loop = self._agent_loop_with_router()
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-d"}, clear=True):
            text, applied = cmd_provider_override(
                agent_loop, {}, "telegram:1", "deepseek")
        assert applied is True
        assert "Auth: API key (DEEPSEEK_API_KEY)" in text

    def test_route_reports_auth_for_active_override(self):
        import os
        from prometheus.gateway.commands import cmd_route
        agent_loop = MagicMock()
        router = MagicMock()
        override = MagicMock()
        override.provider_config = {
            "provider": "xai", "model": "grok-3", "api_key_env": "XAI_API_KEY",
        }
        router.get_override_for_session.return_value = override
        agent_loop._model_router = router
        with patch.dict(os.environ, {}, clear=True), \
             patch("prometheus.providers.xai_oauth.is_logged_in", return_value=True):
            text = cmd_route(agent_loop, "telegram:1", "m", "local")
        assert "Auth: SuperGrok subscription" in text


class TestCallableBearerInProvider:
    def test_provider_resolves_callable_per_request(self):
        from prometheus.providers.openai_compat import OpenAICompatProvider
        tokens = iter(["bearer-A", "bearer-B"])
        p = OpenAICompatProvider(base_url="https://api.x.ai/v1", api_key=lambda: next(tokens))
        assert p._resolve_bearer() == "bearer-A"
        assert p._resolve_bearer() == "bearer-B"  # re-resolved, not frozen

    def test_static_key_still_works(self):
        from prometheus.providers.openai_compat import OpenAICompatProvider
        p = OpenAICompatProvider(base_url="https://api.x.ai/v1", api_key="sk-static")
        assert p._resolve_bearer() == "sk-static"

    def test_empty_bearer_raises(self):
        from prometheus.providers.openai_compat import OpenAICompatProvider
        p = OpenAICompatProvider(base_url="https://api.x.ai/v1", api_key=lambda: None)
        with pytest.raises(ValueError):
            p._resolve_bearer()
