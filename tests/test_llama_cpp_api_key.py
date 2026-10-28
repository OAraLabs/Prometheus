"""A llama.cpp server started with an API key can be used: the main model, a router entry, a named backend.

The key is optional and works like the other OpenAI-compatible providers' keys: ``api_key_env`` names the variable
it is read from, and it goes out as ``Authorization: Bearer <key>`` on every request the provider makes. A
variable that is named but not set is an error naming the variable. Without a key nothing changes: no
``Authorization`` header at all. The key is never in a log line.
"""

from __future__ import annotations

import logging

import httpx
import pytest

from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.providers.base import ApiMessageRequest
from prometheus.providers.registry import ProviderRegistry

KEY_ENV = "LLAMA_CPP_TEST_KEY"
KEY = "llama-test-" + "k3y" * 6
BASE = "http://keyed.example:8080"

PROPS = {
    "default_generation_settings": {"n_ctx": 32768},
    "modalities": {"vision": False},
    "model_path": "/models/test.gguf",
    "chat_template": "{{ messages }}",
    "total_slots": 1,
}
SSE = (
    'data: {"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
    '"usage":{"prompt_tokens":3,"completion_tokens":1}}\n\n'
    "data: [DONE]\n\n"
)


class Server:
    """A stand-in llama-server that records the Authorization header of every request, and can require a key."""

    def __init__(self, require: str | None = None) -> None:
        self.require = require
        self.seen: list[tuple[str, str | None]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("authorization")
        self.seen.append((request.url.path, auth))
        if self.require is not None and auth != f"Bearer {self.require}":
            return httpx.Response(401, json={"error": {"message": "Invalid API Key"}})
        if request.url.path == "/props":
            return httpx.Response(200, json=PROPS)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "test-model"}]})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, text=SSE, headers={"content-type": "text/event-stream"})
        return httpx.Response(404)


@pytest.fixture
def server(monkeypatch):
    stand_in = Server()
    real = httpx.AsyncClient

    class Routed(real):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(stand_in)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", Routed)
    return stand_in


def request() -> ApiMessageRequest:
    return ApiMessageRequest(
        model="test-model",
        messages=[ConversationMessage(role="user", content=[TextBlock(text="hello")])],
        max_tokens=8,
    )


async def every_call(provider) -> None:
    assert await provider.detect_context_size() == 32768
    assert await provider.detect_loaded_model() == "test-model"
    await provider.detect_vision()
    await provider.detect_tool_template()
    await provider.detect_kv_cache_types()
    async for _ in provider.stream_message(request()):
        pass


# ── the provider ─────────────────────────────────────────────────────────────

async def test_a_configured_key_goes_out_on_every_request(server, monkeypatch):
    monkeypatch.setenv(KEY_ENV, KEY)
    server.require = KEY
    provider = ProviderRegistry.create({"provider": "llama_cpp", "base_url": BASE, "api_key_env": KEY_ENV})
    await every_call(provider)
    paths = {path for path, _ in server.seen}
    assert {"/props", "/v1/models", "/v1/chat/completions"} <= paths
    assert all(auth == f"Bearer {KEY}" for _, auth in server.seen)


async def test_without_a_key_no_authorization_header_is_sent(server):
    provider = ProviderRegistry.create({"provider": "llama_cpp", "base_url": BASE})
    await every_call(provider)
    assert server.seen
    assert all(auth is None for _, auth in server.seen)


async def test_an_empty_variable_name_means_no_key(server):
    """A template or a recorded config leaves ``api_key_env: ''``: that is "no key", as it is for every provider."""
    provider = ProviderRegistry.create({"provider": "llama_cpp", "base_url": BASE, "api_key_env": "", "api_key": ""})
    await every_call(provider)
    assert all(auth is None for _, auth in server.seen)


def test_a_named_variable_that_is_not_set_is_an_error_naming_it(monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    with pytest.raises(ValueError, match=KEY_ENV):
        ProviderRegistry.create({"provider": "llama_cpp", "base_url": BASE, "api_key_env": KEY_ENV})


async def test_the_key_is_never_logged(server, monkeypatch, caplog):
    monkeypatch.setenv(KEY_ENV, KEY)
    server.require = "a-different-key"            # every request is refused, so the error paths log too
    provider = ProviderRegistry.create({"provider": "llama_cpp", "base_url": BASE, "api_key_env": KEY_ENV})
    with caplog.at_level(logging.DEBUG):
        await provider.detect_context_size()
        await provider.detect_loaded_model()
        with pytest.raises(Exception):
            async for _ in provider.stream_message(request()):
                pass
    assert caplog.records
    assert not any(KEY in record.getMessage() for record in caplog.records)
    assert KEY not in repr(provider)


# ── a router entry ───────────────────────────────────────────────────────────

def test_a_router_entry_may_name_a_key_for_llama_cpp():
    from prometheus.router.model_router import _entry_refusal

    assert _entry_refusal({"provider": "llama_cpp", "base_url": BASE, "api_key_env": KEY_ENV}, frozenset()) is None


# ── a named backend ──────────────────────────────────────────────────────────

def test_a_backend_may_name_a_key_and_its_provider_config_carries_the_name():
    from prometheus.providers.backends import BackendRegistry

    registry = BackendRegistry.from_config({
        "model": {"provider": "llama_cpp", "base_url": BASE, "api_key_env": KEY_ENV},
        "backends": {"box": {"provider": "llama_cpp", "base_url": "http://box.example:8080",
                             "api_key_env": KEY_ENV}},
    })
    assert registry.config_errors == []
    assert registry.get("box").provider_config()["api_key_env"] == KEY_ENV
    assert registry.get("local").provider_config()["api_key_env"] == KEY_ENV


def test_a_backend_without_a_key_has_none_in_its_provider_config():
    from prometheus.providers.backends import BackendRegistry

    registry = BackendRegistry.from_config({
        "model": {"provider": "llama_cpp", "base_url": BASE},
        "backends": {"box": {"provider": "llama_cpp", "base_url": "http://box.example:8080"}},
    })
    assert "api_key_env" not in registry.get("box").provider_config()
    assert "api_key_env" not in registry.get("local").provider_config()


def test_an_ollama_backend_cannot_name_a_key():
    """Ollama has no key of its own; a key it would never send is refused at load, not dropped."""
    from prometheus.providers.backends import BackendRegistry

    registry = BackendRegistry.from_config({
        "backends": {"box": {"provider": "ollama", "base_url": "http://box.example:11434",
                             "api_key_env": KEY_ENV}},
    })
    assert any("api_key_env" in error for error in registry.config_errors)


@pytest.mark.parametrize("bad", ["lower-case-name", "HAS SPACE", "1STARTS_WITH_DIGIT", "sk-" + "x" * 20])
def test_a_backend_key_variable_must_look_like_a_variable_name(bad):
    from prometheus.providers.backends import BackendRegistry

    registry = BackendRegistry.from_config({
        "backends": {"box": {"provider": "llama_cpp", "base_url": "http://box.example:8080", "api_key_env": bad}},
    })
    assert any("api_key_env" in error for error in registry.config_errors)
    if bad.startswith("sk-"):                      # a key pasted where its variable's name belongs is not echoed
        assert not any(bad in error for error in registry.config_errors)


async def test_the_backend_probe_sends_the_key(server, monkeypatch):
    from prometheus.providers.backends import BackendSpec, probe_backend

    monkeypatch.setenv(KEY_ENV, KEY)
    server.require = KEY
    spec = BackendSpec(name="box", provider="llama_cpp", base_url=BASE, api_key_env=KEY_ENV)
    status = await probe_backend(spec, timeout_s=5.0)
    assert status.ok, status.error
    assert status.model == "test-model"
    assert all(auth == f"Bearer {KEY}" for _, auth in server.seen)


async def test_a_backend_probe_with_its_variable_unset_says_so_and_asks_nothing(server, monkeypatch):
    from prometheus.providers.backends import BackendSpec, probe_backend

    monkeypatch.delenv(KEY_ENV, raising=False)
    spec = BackendSpec(name="box", provider="llama_cpp", base_url=BASE, api_key_env=KEY_ENV)
    status = await probe_backend(spec, timeout_s=5.0)
    assert not status.ok
    assert KEY_ENV in (status.error or "")
    assert server.seen == []
