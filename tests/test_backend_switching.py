"""PR 2 of the multi-backend plan: local backends as ordinary catalogue rows,
switched through the SAME per-session override path as /claude.

What these pin:
  * the resolvers: a configured backend resolves like a preset (target carries
    `backend`, the probe fills the served model and DETECTED vision), `local`
    is never an override target, cloud presets are untouched, ollama's vetted
    `models` list is the choice set (`mini:qwen2.5:7b-instruct`)
  * the catalogue: one row per backend (× models), `available` = reachable,
    `vision` = detected, `health` = the probe, and the local row carries both
  * POST /api/sessions/{id}/model probes BEFORE switching — 503 with the reason
    when the box is down, nothing switches; 200 sets an override whose config
    names the backend, the served model and vision
  * the context resolver: `detected:<backend>` from the registry's live window,
    only while the window's model is the one being resolved; `backend_config:`
    from the operator hint; a local backend never falls to the cloud default;
    the primary keeps the bare `detected` string
  * the compactor budgets a `/4090` turn at THAT box's window, live
  * the router: RouteDecision carries `backend`; a backend override persists
    (set + clear, sticky only, cloud never); boot restore skips dead boxes
  * /api/lcm reports the SESSION's window, not the primary's
  * the shared /<backend> command: unknown, down (refused, named), success
"""

from __future__ import annotations

import sys
import types

if "prometheus.memory" not in sys.modules:  # pragma: no cover - import plumbing
    try:
        import prometheus.memory  # noqa: F401
    except Exception:
        sys.modules["prometheus.memory"] = types.ModuleType("prometheus.memory")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.context.budget import resolve_effective_limit  # noqa: E402
from prometheus.context.compactor import ContextCompactor  # noqa: E402
from prometheus.providers import backends as backends_mod  # noqa: E402
from prometheus.providers.backends import (  # noqa: E402
    BackendRegistry,
    BackendSpec,
    BackendStatus,
    DetectedWindow,
)
from prometheus.router.model_router import (  # noqa: E402
    ModelRouter,
    RouterConfig,
    backend_command_names,
    resolve_model_choices,
    resolve_model_target,
    resolve_slash_command_target,
    restore_backend_overrides,
)
from prometheus.web.server import create_app  # noqa: E402

CFG = {
    "model": {"model": "qwen3.8-27b", "provider": "llama_cpp", "base_url": "http://a:8080"},
    "context": {"effective_limit": 72000},
    "backends": {
        "4090": {"provider": "llama_cpp", "base_url": "http://gpu-box:8080"},
        "mini": {"provider": "ollama", "base_url": "http://localhost:11434",
                 "model": "qwen2.5:14b-instruct", "models": ["qwen2.5:7b-instruct"], "context_limit": 20000},
    },
}


def _probe_factory(calls: list[str], *, down: set[str] = frozenset()):
    async def probe(spec: BackendSpec, timeout: float) -> BackendStatus:
        calls.append(spec.name)
        if spec.name in down:
            raise ConnectionError("connect timeout")
        model = {"local": "/models/Qwen3.8-27B.gguf", "4090": "/models/Qwen3.6-27B.gguf"}.get(spec.name, spec.model)
        return BackendStatus(
            name=spec.name, provider=spec.provider, base_url=spec.base_url, ok=True,
            model=model, model_path=model if spec.provider == "llama_cpp" else None,
            n_ctx={"local": 32768, "4090": 65536, "mini": 32768}[spec.name],
            vision=(spec.name == "4090"), latency_ms=5.0,
        )
    return probe


def _registry(calls: list[str] | None = None, *, down: set[str] = frozenset()) -> BackendRegistry:
    return BackendRegistry.from_config(CFG, probe=_probe_factory(calls if calls is not None else [], down=down))


@pytest.fixture
def live_registry(monkeypatch):
    """A probed registry installed as the process handle (what the daemon does)."""
    import asyncio
    reg = _registry()
    asyncio.run(reg.probe_all())
    monkeypatch.setattr(backends_mod, "_REGISTRY", reg)
    return reg


def _router(**kw) -> ModelRouter:
    return ModelRouter(RouterConfig(), primary_provider=object(), primary_adapter=object(),
                       primary_model="qwen3.8-27b", **kw)


# ── resolvers ────────────────────────────────────────────────────────────────


def test_a_backend_resolves_like_a_preset_and_local_never_does(live_registry):
    t = resolve_slash_command_target("4090", CFG)
    assert t["provider"] == "llama_cpp" and t["base_url"] == "http://gpu-box:8080" and t["backend"] == "4090"
    assert t["model"] == "/models/Qwen3.6-27B.gguf"     # filled from the probe
    assert t["vision"] is True                           # DETECTED, not declared
    m = resolve_slash_command_target("mini", CFG)
    assert m["model"] == "qwen2.5:14b-instruct" and "vision" not in m   # probe said False → absent
    assert resolve_slash_command_target("local", CFG) is None
    assert resolve_slash_command_target("nope", CFG) is None
    assert resolve_slash_command_target("claude", CFG)["provider"] == "anthropic"   # presets untouched


def test_config_only_table_when_no_daemon_registry(monkeypatch):
    monkeypatch.setattr(backends_mod, "_REGISTRY", None)
    t = resolve_slash_command_target("4090", CFG)
    assert t == {"provider": "llama_cpp", "base_url": "http://gpu-box:8080", "backend": "4090"}


def test_choices_and_composite_keys_follow_the_vetted_list(live_registry):
    assert resolve_model_choices("mini", CFG) == ("qwen2.5:14b-instruct", "qwen2.5:7b-instruct")
    assert resolve_model_choices("4090", CFG) == ("/models/Qwen3.6-27B.gguf",)
    t = resolve_model_target("mini:qwen2.5:7b-instruct", CFG)      # colon inside the model round-trips
    assert t["model"] == "qwen2.5:7b-instruct" and t["backend"] == "mini"
    assert resolve_model_target("mini:nope:1b", CFG) is None
    assert backend_command_names(CFG) == ("4090", "mini")


# ── the context resolver ─────────────────────────────────────────────────────


def _windows():
    return {
        "local": DetectedWindow("local", "qwen3.8-27b", 32768, 1.0),
        "4090": DetectedWindow("4090", "/models/Qwen3.6-27B.gguf", 65536, 1.0),
    }


def test_resolver_names_the_backend_that_detected():
    ctx = {"effective_limit": 72000, "cloud_default_limit": 200000}
    assert resolve_effective_limit(ctx, model="/models/Qwen3.6-27B.gguf", backend="4090", detected=_windows()) == (65536, "detected:4090")
    # the primary keeps the bare word so existing readers of "detected" keep working
    assert resolve_effective_limit(ctx, model="qwen3.8-27b", backend="local", detected=_windows()) == (32768, "detected")


def test_a_window_reported_for_another_model_does_not_apply():
    ctx = {"effective_limit": 72000}
    # the box restarted onto gemma; the override still names the old GGUF → the old window is stale
    windows = {"4090": DetectedWindow("4090", "/models/gemma.gguf", 8192, 2.0)}
    assert resolve_effective_limit(ctx, model="/models/Qwen3.6-27B.gguf", backend="4090", detected=windows) == (72000, "config")


def test_operator_hint_stands_in_when_the_probe_could_not_size():
    ctx = {"effective_limit": 72000, "cloud_default_limit": 200000}
    assert resolve_effective_limit(ctx, model="qwen2.5:14b", backend="mini", detected={}, backend_hint=20000) == (20000, "backend_config:mini")


def test_a_local_backend_never_falls_to_the_cloud_default():
    ctx = {"effective_limit": 72000, "cloud_default_limit": 200000}
    # model != local_model would have meant "cloud" in the single-box rules
    out = resolve_effective_limit(ctx, model="other", local_model="qwen3.8-27b", detected_limit=32768, backend="4090", detected={})
    assert out == (72000, "config")
    assert resolve_effective_limit({}, model="other", backend="4090", detected={}) == (None, "unknown")
    # and a cloud override (no backend) still gets the cloud default
    assert resolve_effective_limit(ctx, model="claude-opus-5", local_model="qwen3.8-27b", detected_limit=32768) == (200000, "cloud_default")


def test_single_box_callers_are_unchanged():
    ctx = {"effective_limit": 72000, "cloud_default_limit": 200000, "model_overrides": {"m": {"effective_limit": 1234}}}
    assert resolve_effective_limit(ctx, model="m", backend="4090", detected=_windows()) == (1234, "model_override")
    assert resolve_effective_limit(ctx, model="qwen3.8-27b", local_model="qwen3.8-27b", detected_limit=32768) == (32768, "detected")
    assert resolve_effective_limit(ctx, model="qwen3.8-27b", local_model="qwen3.8-27b") == (72000, "config")


def test_compactor_budgets_a_backend_turn_at_that_box_window_live():
    windows = _windows()
    comp = ContextCompactor(
        provider=object(), model="qwen3.8-27b", effective_limit=72000, detected_limit=32768,
        cloud_default_limit=200000, windows=lambda: windows, backend_hints={"mini": 20000},
    )
    assert comp.limit_for("qwen3.8-27b") == 32768                                   # primary, unchanged
    assert comp.limit_for("claude-opus-5") == 200000                                # cloud override, unchanged
    assert comp.limit_for("/models/Qwen3.6-27B.gguf", backend="4090") == 65536      # that box's window
    windows["4090"] = DetectedWindow("4090", "/models/Qwen3.6-27B.gguf", 16384, 3.0)  # re-probed smaller
    assert comp.limit_for("/models/Qwen3.6-27B.gguf", backend="4090") == 16384      # live, no rebuild
    assert comp.limit_for("qwen2.5:14b", backend="mini") == 20000                   # the hint
    assert comp.limit_for(None) == 32768


# ── the router ───────────────────────────────────────────────────────────────


def test_route_decision_carries_the_backend(monkeypatch):
    router = _router()
    monkeypatch.setattr("prometheus.providers.registry.ProviderRegistry.create", lambda cfg: object())
    monkeypatch.setattr("prometheus.router.model_router._build_adapter_for", lambda name: object())
    router.set_override("s1", {"provider": "llama_cpp", "base_url": "http://gpu-box:8080", "model": "m", "backend": "4090"})
    d = router._route_override("s1")
    assert d.backend == "4090" and d.provider_name == "llama_cpp"
    router.set_override("s2", {"provider": "anthropic", "model": "claude-opus-5", "vision": True})
    assert router._route_override("s2").backend is None


def test_backend_overrides_persist_and_cloud_ones_do_not():
    saved: list[tuple[str, dict | None]] = []
    router = _router(persist_override=lambda sid, cfg: saved.append((sid, cfg)))
    router.set_override("s1", {"provider": "anthropic", "model": "claude-opus-5"})
    assert saved == []                                                  # cloud: RAM-only, as before
    router.set_override("s1", {"provider": "llama_cpp", "backend": "4090", "model": "m"})
    assert saved[-1] == ("s1", {"provider": "llama_cpp", "backend": "4090", "model": "m"})
    router.clear_override("s1")
    assert saved[-1] == ("s1", None)
    router.clear_override("never-set")                                  # silent, nothing persisted
    assert len(saved) == 2


def test_one_shot_mode_does_not_persist():
    saved: list = []
    router = ModelRouter(RouterConfig(overrides_sticky=False), primary_provider=object(), primary_adapter=object(),
                         persist_override=lambda sid, cfg: saved.append((sid, cfg)))
    router.set_override("s1", {"provider": "llama_cpp", "backend": "4090"})
    assert saved == []


def test_boot_restore_applies_only_onto_boxes_found_up(live_registry):
    import asyncio
    reg = _registry(down={"mini"})
    asyncio.run(reg.probe_all())
    router = _router()
    restored, skipped = restore_backend_overrides(
        router, reg,
        {"tg:1": "4090", "tg:2": "mini", "tg:3": "gone", "tg:4": "mini:qwen2.5:7b-instruct"},
        CFG,
    )
    assert restored == 1
    assert router.get_override_for_session("tg:1").provider_config["backend"] == "4090"
    assert router.get_override_for_session("tg:2") is None
    reasons = {sid: why for sid, _k, why in skipped}
    assert "connect timeout" in reasons["tg:2"] and reasons["tg:3"] == "not configured"


def test_conversation_store_remembers_and_purges_the_binding(tmp_path):
    from prometheus.memory.lcm_conversation_store import LCMConversationStore
    store = LCMConversationStore(str(tmp_path / "lcm.db"))
    store.set_session_backend("s1", "4090")
    store.set_session_backend("s2", "mini:qwen2.5:7b-instruct", set_by="rest")
    assert store.get_session_backend("s1") == "4090"
    assert store.all_session_backends() == {"s1": "4090", "s2": "mini:qwen2.5:7b-instruct"}
    store.set_session_backend("s1", None)
    assert store.get_session_backend("s1") is None
    if hasattr(store, "purge_session"):
        store.purge_session("s2")
        assert store.get_session_backend("s2") is None


# ── the catalogue + REST switch ──────────────────────────────────────────────


def _client(reg: BackendRegistry, router: ModelRouter) -> TestClient:
    # local_model as the daemon passes it: the primary's served id. Without it a
    # cloud override cannot be told apart from the primary (the single-box rule).
    return TestClient(create_app(CFG, model_router=router, backend_registry=reg, local_model="qwen3.8-27b"))


def test_catalogue_has_one_row_per_backend_with_health(live_registry):
    c = _client(live_registry, _router())
    rows = {m["key"]: m for m in c.get("/api/models").json()["models"]}
    local = rows["local"]
    assert local["backend"] == "local" and local["health"]["ok"] is True and local["health"]["n_ctx"] == 32768
    r = rows["4090"]
    assert r["backend"] == "4090" and r["available"] is True and r["vision"] is True and r["auth"] is None
    assert r["label"] == "4090 · Qwen3.6-27B" and r["detail"] == "/models/Qwen3.6-27B.gguf"
    assert r["health"]["n_ctx"] == 65536 and r["health"]["stale"] is False
    assert rows["mini"]["model"] == "qwen2.5:14b-instruct" and rows["mini"]["vision"] is False
    assert rows["mini:qwen2.5:7b-instruct"]["backend"] == "mini"
    assert rows["claude"]["backend"] is None if "backend" in rows["claude"] else True   # cloud rows unchanged


def test_catalogue_marks_a_down_backend_unavailable(monkeypatch):
    import asyncio
    reg = _registry(down={"4090"})
    asyncio.run(reg.probe_all())
    monkeypatch.setattr(backends_mod, "_REGISTRY", reg)
    rows = {m["key"]: m for m in _client(reg, _router()).get("/api/models").json()["models"]}
    assert rows["4090"]["available"] is False and "connect timeout" in rows["4090"]["health"]["error"]
    assert rows["4090"]["vision"] is False


def test_post_probes_before_switching_and_refuses_a_down_box(monkeypatch):
    import asyncio
    calls: list[str] = []
    reg = _registry(calls, down={"mini"})
    asyncio.run(reg.probe_all())
    monkeypatch.setattr(backends_mod, "_REGISTRY", reg)
    router = _router()
    c = _client(reg, router)
    n = len(calls)
    r = c.post("/api/sessions/s1/model", json={"key": "mini"})
    assert r.status_code == 503 and "connect timeout" in r.json()["error"] and r.json()["backend"] == "mini"
    assert calls[n:] == ["mini"]                            # forced probe, this box only
    assert router.get_override_for_session("s1") is None    # nothing switched

    r = c.post("/api/sessions/s1/model", json={"key": "4090"})
    assert r.status_code == 200
    assert calls[-1] == "4090"
    body = r.json()
    assert body["key"] == "4090" and body["backend"] == "4090"
    cfg = router.get_override_for_session("s1").provider_config
    assert cfg == {"provider": "llama_cpp", "base_url": "http://gpu-box:8080", "backend": "4090",
                   "model": "/models/Qwen3.6-27B.gguf", "vision": True}
    assert c.get("/api/sessions/s1/model").json()["key"] == "4090"
    r = c.post("/api/sessions/s1/model", json={"key": "local"})
    assert r.status_code == 200 and router.get_override_for_session("s1") is None


def test_effective_model_follows_a_served_model_change(live_registry):
    router = _router()
    router.set_override("s1", {"provider": "llama_cpp", "backend": "4090", "model": "/models/old.gguf"})
    body = _client(live_registry, router).get("/api/sessions/s1/model").json()
    assert body["backend"] == "4090" and body["model"] == "/models/old.gguf" and body["label"] == "4090 · /models/old.gguf"


def test_lcm_reports_the_sessions_window_not_the_primarys(live_registry):
    router = _router()
    c = _client(live_registry, router)
    # The primary session in a bare test app: the registry's `local` window is
    # for the SERVED id (the GGUF path) while the config names a hint, so the
    # window does not apply and the configured global stands — the same rule
    # that stops a swapped box being budgeted at its old size. In the daemon,
    # local_model/detected_limit are the served id and its window, so this
    # reads "detected" there.
    assert c.get("/api/lcm/s0").json()["limit_source"] == "config"
    router.set_override("s1", {"provider": "llama_cpp", "backend": "4090", "model": "/models/Qwen3.6-27B.gguf"})
    body = c.get("/api/lcm/s1").json()
    assert (body["limit"], body["limit_source"]) == (65536, "detected:4090")
    router.set_override("s2", {"provider": "anthropic", "model": "claude-opus-5", "vision": True})
    assert c.get("/api/lcm/s2").json()["limit_source"] == "cloud_default"


# ── the shared chat command ──────────────────────────────────────────────────


class _Loop:
    def __init__(self, router):
        self._model_router = router


@pytest.mark.asyncio
async def test_cmd_backend_override_refuses_unknown_and_down_and_switches_when_up(monkeypatch):
    from prometheus.gateway.commands import cmd_backend_override, cmd_route

    calls: list[str] = []
    reg = _registry(calls, down={"mini"})
    await reg.probe_all()
    monkeypatch.setattr(backends_mod, "_REGISTRY", reg)
    router = _router()
    loop = _Loop(router)

    text, ok = await cmd_backend_override(loop, CFG, "tg:1", "nope")
    assert not ok and "Unknown backend" in text and "/4090" in text
    text, ok = await cmd_backend_override(loop, CFG, "tg:1", "local")
    assert not ok
    text, ok = await cmd_backend_override(loop, CFG, "tg:1", "mini")
    assert not ok and "mini is down" in text and "connect timeout" in text
    assert router.get_override_for_session("tg:1") is None
    text, ok = await cmd_backend_override(loop, CFG, "tg:1", "4090")
    assert ok and "Switched to 4090" in text and "64k" in text and "vision" in text
    assert router.get_override_for_session("tg:1").provider_config["backend"] == "4090"
    text, ok = await cmd_backend_override(loop, CFG, "tg:1", "mini", model="nope")
    assert not ok and "has no model" in text
    assert "Local backends" in cmd_route(loop, "tg:9", "qwen3.8-27b", "llama_cpp") and "/4090" in cmd_route(loop, "tg:9", "m", "p")


@pytest.mark.asyncio
async def test_cmd_backend_override_without_a_registry(monkeypatch):
    from prometheus.gateway.commands import cmd_backend_override
    monkeypatch.setattr(backends_mod, "_REGISTRY", None)
    text, ok = await cmd_backend_override(_Loop(_router()), CFG, "tg:1", "4090")
    assert not ok and "not initialized" in text
