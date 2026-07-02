"""Model REST (Sprint B / Piece 1) — expose the EXISTING per-session ModelRouter
override engine over bearer-authed REST. No engine change.

Asserts SIDE EFFECTS on a REAL ModelRouter (the stored override + that route()
actually picks it up), not call-counting — the proof the REST layer drives the live
engine. Catalog is key-only (no secrets). Reserved session ids surface a clean 400.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.router.model_router import (  # noqa: E402
    ModelRouter,
    OVERRIDE_PRESETS,
    RouteReason,
    RouterConfig,
)
from prometheus.web.server import create_app  # noqa: E402


def _router(primary: str = "qwen3.5-32b") -> ModelRouter:
    # A real router; primary provider/adapter are opaque stand-ins (the routes never
    # touch them, and the override path is monkeypatched where it would build clouds).
    return ModelRouter(
        RouterConfig(),
        primary_provider=object(),
        primary_adapter=object(),
        primary_model=primary,
    )


def _client(router: ModelRouter, *, token: str = "") -> TestClient:
    cfg: dict = {"model": {"model": "qwen3.5-32b", "provider": "local"}}
    if token:
        cfg["web"] = {"api_token": token}
    return TestClient(create_app(cfg, model_router=router))


def test_catalog_lists_local_plus_presets_and_leaks_no_secrets():
    c = _client(_router())
    r = c.get("/api/models")
    assert r.status_code == 200
    body = r.json()
    keys = [m["key"] for m in body["models"]]
    assert keys[0] == "local" and body["default_key"] == "local"
    assert body["models"][0]["is_default"] is True
    for k in OVERRIDE_PRESETS:
        assert k in keys
    # No secrets — not the env-var NAMES, and certainly not their values.
    blob = r.text
    for preset in OVERRIDE_PRESETS.values():
        assert preset["api_key_env"] not in blob


def test_set_preset_stores_override_and_route_picks_it_up(monkeypatch):
    router = _router()
    c = _client(router)
    r = c.post("/api/sessions/s1/model", json={"key": "claude"})
    assert r.status_code == 200 and r.json()["key"] == "claude"
    # Side effect: the REAL override is stored with the vetted preset config.
    assert router._overrides["s1"].provider_config == OVERRIDE_PRESETS["claude"]
    # Proof-it-works: the NEXT route() for this session takes the override-first path
    # and resolves to the override target — not just that set_override was called.
    import prometheus.providers.registry as _reg
    import prometheus.router.model_router as _mr

    # _route_override lazily builds a provider/adapter from the preset config; stub both
    # so the override path runs without real cloud construction (model_name comes from the
    # stored provider_config, which is what proves the switch took effect).
    from tests.support.doubles import register_double

    monkeypatch.setattr(
        _reg.ProviderRegistry, "create",
        register_double("model_rest.ProviderRegistry.create", replaces="prometheus.providers.registry.ProviderRegistry.create")(lambda *a, **k: object()),
    )
    monkeypatch.setattr(
        _mr, "_build_adapter_for",
        register_double("model_rest._build_adapter_for", replaces="prometheus.router.model_router._build_adapter_for")(lambda *a, **k: object()),
    )
    decision = router.route("hello there", {"session_id": "s1"})
    assert decision.model_name == OVERRIDE_PRESETS["claude"]["model"]
    assert decision.reason == RouteReason.USER_OVERRIDE
    # …and a session with no override still resolves to primary.
    assert router.route("hello", {"session_id": "untouched"}).model_name != OVERRIDE_PRESETS["claude"]["model"]


def test_get_reflects_override_then_clears_to_local():
    router = _router()
    c = _client(router)
    assert c.get("/api/sessions/s2/model").json()["key"] == "local"  # default before any set
    c.post("/api/sessions/s2/model", json={"key": "gpt"})
    assert c.get("/api/sessions/s2/model").json()["key"] == "gpt"


def test_post_local_key_clears_override():
    router = _router()
    c = _client(router)
    c.post("/api/sessions/s3/model", json={"key": "gemini"})
    assert "s3" in router._overrides
    r = c.post("/api/sessions/s3/model", json={"key": "local"})
    assert r.status_code == 200 and r.json()["key"] == "local"
    assert "s3" not in router._overrides  # local key → clear_override


def test_delete_clears_and_is_idempotent():
    router = _router()
    c = _client(router)
    c.post("/api/sessions/s4/model", json={"key": "xai"})
    assert c.delete("/api/sessions/s4/model").status_code == 200
    assert "s4" not in router._overrides
    # Clearing again when none set is a clean 200, not an error.
    r = c.delete("/api/sessions/s4/model")
    assert r.status_code == 200 and r.json()["key"] == "local"


def test_unknown_key_is_400():
    c = _client(_router())
    r = c.post("/api/sessions/s5/model", json={"key": "bogus-model"})
    assert r.status_code == 400


def test_reserved_session_id_is_400_not_500():
    router = _router()
    c = _client(router)
    r = c.post("/api/sessions/system/model", json={"key": "claude"})
    assert r.status_code == 400
    assert "system" not in router._overrides  # never stored


def test_routes_require_bearer_when_token_set():
    c = _client(_router(), token="secret")
    assert c.get("/api/models").status_code == 401
    assert c.get("/api/models", headers={"Authorization": "Bearer secret"}).status_code == 200
    assert c.post("/api/sessions/s6/model", json={"key": "local"}).status_code == 401


def test_preset_resolves_through_user_slash_commands_config():
    # REST must agree with the /claude slash command: a user's slash_commands.<key>
    # config is merged over the preset, so the catalog + the stored override reflect it.
    router = _router()
    cfg = {
        "model": {"model": "qwen3.5-32b", "provider": "local"},
        "slash_commands": {"claude": {"model": "claude-sonnet-4-5"}},
    }
    c = TestClient(create_app(cfg, model_router=router))
    catalog = c.get("/api/models").json()["models"]
    claude = next(m for m in catalog if m["key"] == "claude")
    assert claude["model"] == "claude-sonnet-4-5"  # user override, not the hardcoded haiku
    c.post("/api/sessions/s8/model", json={"key": "claude"})
    assert router._overrides["s8"].provider_config["model"] == "claude-sonnet-4-5"


def test_router_unavailable_is_503():
    # No router wired → routes degrade to a clean 503, never a 500.
    c = TestClient(create_app({"model": {"model": "qwen3.5-32b"}}))
    assert c.get("/api/models").status_code == 503
    assert c.get("/api/sessions/s7/model").status_code == 503
