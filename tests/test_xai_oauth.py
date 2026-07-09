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
