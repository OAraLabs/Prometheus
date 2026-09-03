"""The backend registry (providers/backends.py) — the one table of local boxes.

Pins, without a network:
  * config validation refuses the bad entry with its reason and keeps the rest
  * the primary registers as `local`, from `model:`
  * the llama.cpp and ollama probes parse the payloads the live servers return
    (recorded 2026-09-02 from the 4090's llama-server and the mini's ollama)
  * TTL caching, force, and shared in-flight probes
  * change detection: a different served model sets `changed_at` and the window
    leaves detected_windows() until re-probed
  * a failed probe keeps the last good facts and flips `ok`
  * the snapshot / table say "never probed" and "stale" distinctly
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from prometheus.providers.backends import (
    BackendRegistry,
    BackendSpec,
    BackendStatus,
    probe_backend,
    _validate_spec,
    _reserved_names,
    BackendConfigError,
)

# ── recorded payloads ─────────────────────────────────────────────────────────

LLAMA_PROPS = {
    "default_generation_settings": {"n_ctx": 32768, "n_predict": -1},
    "total_slots": 1,
    "model_path": "/models/Qwen3.8-27B-UD-Q4_K_XL.gguf",
    "model_alias": "/models/Qwen3.8-27B-UD-Q4_K_XL.gguf",
    "modalities": {"vision": True, "video": True, "audio": False},
    "build_info": "b1-9d57ce456",
}
LLAMA_MODELS = {"object": "list", "data": [{"id": "/models/Qwen3.8-27B-UD-Q4_K_XL.gguf"}]}

OLLAMA_TAGS = {"models": [
    {"name": "ornith:9b", "size": 5_600_000_000},
    {"name": "qwen2.5:14b-instruct", "size": 9_000_000_000},
    {"name": "qwen2.5:7b-instruct", "size": 4_700_000_000},
    {"name": "qwen3:32b", "size": 20_200_000_000},
]}
OLLAMA_PS = {"models": [{"name": "qwen2.5:7b-instruct", "size_vram": 6_600_000_000}]}
OLLAMA_SHOW = {
    "capabilities": ["completion", "tools"],
    "model_info": {"general.architecture": "qwen2", "qwen2.context_length": 32768},
    "details": {"family": "qwen2"},
}


def _llama_transport(props=LLAMA_PROPS, models=LLAMA_MODELS):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/props":
            return httpx.Response(200, json=props)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=models)
        return httpx.Response(404)
    return httpx.MockTransport(handler)


def _ollama_transport(tags=OLLAMA_TAGS, ps=OLLAMA_PS, show=OLLAMA_SHOW):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json=tags)
        if request.url.path == "/api/ps":
            return httpx.Response(200, json=ps)
        if request.url.path == "/api/show":
            body = json.loads(request.content or b"{}")
            assert body.get("model"), "show must name the model"
            return httpx.Response(200, json=show)
        return httpx.Response(404)
    return httpx.MockTransport(handler)


@pytest.fixture
def patched_client(monkeypatch):
    """Route probe_backend's httpx.AsyncClient through a MockTransport chosen per test."""
    holder = {}

    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = holder["transport"]
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return holder


# ── config ────────────────────────────────────────────────────────────────────


def test_primary_registers_as_local_from_the_model_section():
    reg = BackendRegistry.from_config({"model": {"provider": "llama_cpp", "base_url": "http://gpu-box:8080/"}})
    assert reg.names() == ("local",)
    spec = reg.get("local")
    assert spec.is_primary and spec.provider == "llama_cpp"
    assert spec.base_url == "http://gpu-box:8080"  # trailing slash dropped


def test_cloud_primary_registers_no_local_backend():
    reg = BackendRegistry.from_config({"model": {"provider": "anthropic"}, "backends": {"mini": {"provider": "ollama", "base_url": "http://localhost:11434"}}})
    assert reg.names() == ("mini",)


def test_bad_entries_are_refused_one_by_one_with_reasons_and_the_rest_load():
    cfg = {
        "model": {"provider": "llama_cpp", "base_url": "http://a:8080"},
        "backends": {
            "4090": {"provider": "llama_cpp", "base_url": "http://gpu-box:8080"},
            "Bad-Name": {"provider": "llama_cpp", "base_url": "http://x:8080"},
            "claude": {"provider": "ollama", "base_url": "http://x:11434"},      # collides with a preset
            "local": {"provider": "ollama", "base_url": "http://x:11434"},       # reserved
            "cloudy": {"provider": "anthropic", "base_url": "http://x"},          # not a local provider
            "nourl": {"provider": "ollama", "base_url": "x:11434"},               # not http(s)
            "notmap": "http://x:11434",
            "badctx": {"provider": "ollama", "base_url": "http://x:11434", "context_limit": "lots"},
            "mini": {"provider": "ollama", "base_url": "http://localhost:11434", "model": "qwen2.5:14b-instruct",
                     "models": ["qwen2.5:7b-instruct"], "context_limit": "32768"},
        },
    }
    reg = BackendRegistry.from_config(cfg)
    assert reg.names() == ("local", "4090", "mini")
    joined = "\n".join(reg.config_errors)
    for needle in ("Bad-Name", "claude", "backends.local", "cloudy", "nourl", "notmap", "badctx"):
        assert needle in joined, f"{needle} missing from {joined}"
    assert len(reg.config_errors) == 7
    mini = reg.get("mini")
    # the default model is always first in the vetted list, and the hint is an int
    assert mini.models == ("qwen2.5:14b-instruct", "qwen2.5:7b-instruct")
    assert mini.context_limit == 32768


def test_reserved_names_include_every_cloud_preset():
    from prometheus.router.model_router import OVERRIDE_PRESETS
    reserved = _reserved_names()
    assert set(OVERRIDE_PRESETS) <= reserved
    assert {"local", "route", "grok"} <= reserved


def test_models_must_be_a_list():
    with pytest.raises(BackendConfigError):
        _validate_spec("m", {"provider": "ollama", "base_url": "http://x", "models": "a,b"}, frozenset())


def test_provider_config_carries_the_backend_name():
    spec = BackendSpec(name="mini", provider="ollama", base_url="http://x:11434", model="a", models=("a", "b"))
    assert spec.provider_config() == {"provider": "ollama", "base_url": "http://x:11434", "backend": "mini", "model": "a"}
    assert spec.provider_config("b")["model"] == "b"


def test_probe_settings_come_from_backend_probe_with_the_shipped_defaults():
    reg = BackendRegistry.from_config({"model": {"provider": "llama_cpp"}})
    assert (reg.ttl_s, reg.timeout_s) == (60, 5.0)
    reg2 = BackendRegistry.from_config({"model": {"provider": "llama_cpp"}, "backend_probe": {"ttl_s": 5, "timeout_s": 1}})
    assert (reg2.ttl_s, reg2.timeout_s) == (5.0, 1.0)


# ── the probes ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_llama_cpp_probe_parses_the_live_payload(patched_client):
    patched_client["transport"] = _llama_transport()
    spec = BackendSpec(name="4090", provider="llama_cpp", base_url="http://gpu-box:8080")
    st = await probe_backend(spec, 5.0)
    assert st.ok is True
    assert st.n_ctx == 32768
    assert st.vision is True
    assert st.model_path.endswith("Qwen3.8-27B-UD-Q4_K_XL.gguf")
    assert st.model == st.model_path
    assert st.latency_ms is not None


@pytest.mark.asyncio
async def test_llama_cpp_probe_leaves_vision_unknown_when_the_server_publishes_no_modalities(patched_client):
    props = {k: v for k, v in LLAMA_PROPS.items() if k != "modalities"}
    patched_client["transport"] = _llama_transport(props=props)
    st = await probe_backend(BackendSpec(name="b", provider="llama_cpp", base_url="http://x:8080"), 5.0)
    assert st.ok and st.vision is None  # unknown, not False


@pytest.mark.asyncio
async def test_ollama_probe_parses_the_live_payload(patched_client):
    patched_client["transport"] = _ollama_transport()
    spec = BackendSpec(name="mini", provider="ollama", base_url="http://localhost:11434", model="qwen2.5:14b-instruct")
    st = await probe_backend(spec, 5.0)
    assert st.ok is True
    assert st.model == "qwen2.5:14b-instruct"
    assert st.available_models == ("ornith:9b", "qwen2.5:14b-instruct", "qwen2.5:7b-instruct", "qwen3:32b")
    assert st.loaded_models == ("qwen2.5:7b-instruct",)
    assert st.n_ctx == 32768           # from model_info.<arch>.context_length
    assert st.vision is False          # capabilities present, no "vision"
    assert st.error is None


@pytest.mark.asyncio
async def test_ollama_probe_names_a_configured_model_that_is_not_pulled(patched_client):
    patched_client["transport"] = _ollama_transport()
    spec = BackendSpec(name="mini", provider="ollama", base_url="http://localhost:11434", model="nope:1b")
    st = await probe_backend(spec, 5.0)
    assert st.ok is True               # the box is up
    assert "not pulled" in st.error    # the model is not


@pytest.mark.asyncio
async def test_ollama_probe_falls_back_to_the_loaded_model_when_none_is_configured(patched_client):
    patched_client["transport"] = _ollama_transport()
    st = await probe_backend(BackendSpec(name="mini", provider="ollama", base_url="http://localhost:11434"), 5.0)
    assert st.model == "qwen2.5:7b-instruct"


# ── the registry's caching + change detection (probe injected, no HTTP) ──────


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _ok(spec: BackendSpec, model: str, n_ctx: int = 32768, vision: bool = True) -> BackendStatus:
    return BackendStatus(name=spec.name, provider=spec.provider, base_url=spec.base_url,
                         ok=True, model=model, model_path=model, n_ctx=n_ctx, vision=vision, latency_ms=3.0)


def _registry(probe, clock, ttl=60.0) -> BackendRegistry:
    specs = [
        BackendSpec(name="local", provider="llama_cpp", base_url="http://a:8080", is_primary=True),
        BackendSpec(name="4090", provider="llama_cpp", base_url="http://b:8080"),
    ]
    return BackendRegistry(specs, ttl_s=ttl, timeout_s=2.0, probe=probe, clock=clock)


@pytest.mark.asyncio
async def test_never_probed_is_distinct_from_down():
    calls = []

    async def probe(spec, timeout):
        calls.append(spec.name)
        return _ok(spec, "m1")

    reg = _registry(probe, _Clock())
    snap = reg.snapshot()["4090"]
    assert snap["probed"] is False and snap["ok"] is False and snap["stale"] is True
    assert "not probed" in reg.render_table()
    await reg.probe_all()
    assert sorted(calls) == ["4090", "local"]
    snap = reg.snapshot()["4090"]
    assert snap["probed"] is True and snap["ok"] is True and snap["stale"] is False


@pytest.mark.asyncio
async def test_ttl_caches_and_force_bypasses():
    calls = []

    async def probe(spec, timeout):
        calls.append(spec.name)
        return _ok(spec, "m1")

    clock = _Clock()
    reg = _registry(probe, clock, ttl=60)
    await reg.probe("4090")
    await reg.probe("4090")                 # within TTL → cached
    assert calls == ["4090"]
    clock.t += 61
    assert reg.is_stale("4090")
    await reg.probe("4090")                 # TTL lapsed → re-probed
    assert calls == ["4090", "4090"]
    await reg.probe("4090", force=True)     # force → probed regardless
    assert calls == ["4090", "4090", "4090"]


@pytest.mark.asyncio
async def test_concurrent_callers_share_one_probe():
    calls = []
    gate = asyncio.Event()

    async def probe(spec, timeout):
        calls.append(spec.name)
        await gate.wait()
        return _ok(spec, "m1")

    reg = _registry(probe, _Clock())
    t1 = asyncio.create_task(reg.probe("4090"))
    t2 = asyncio.create_task(reg.probe("4090"))
    await asyncio.sleep(0)
    gate.set()
    await asyncio.gather(t1, t2)
    assert calls == ["4090"]


@pytest.mark.asyncio
async def test_a_changed_served_model_is_flagged_and_its_window_invalidated_until_reprobed():
    served = {"model": "/models/Qwen3.8-27B.gguf", "n_ctx": 32768}

    async def probe(spec, timeout):
        return _ok(spec, served["model"], served["n_ctx"])

    clock = _Clock()
    reg = _registry(probe, clock)
    await reg.probe("4090", force=True)
    assert reg.status("4090").changed_at_iso is None
    assert reg.detected_windows()["4090"].n_ctx == 32768

    served.update(model="/models/gemma-4-26B.gguf", n_ctx=8192)   # llama-server restarted on another GGUF
    clock.t += 61
    st = await reg.probe("4090")
    assert st.changed_at_iso is not None
    assert reg.detected_windows()["4090"].n_ctx == 8192          # the NEW window, not the old
    assert "served model changed at" in reg.render_table()
    # a later probe with the same model keeps the change marker but does not re-flag
    clock.t += 61
    st2 = await reg.probe("4090")
    assert st2.changed_at_iso == st.changed_at_iso


@pytest.mark.asyncio
async def test_a_failed_probe_keeps_the_last_good_facts_and_flips_ok():
    fail = {"on": False}

    async def probe(spec, timeout):
        if fail["on"]:
            raise ConnectionError("connect timeout")
        return _ok(spec, "m1", 32768, True)

    clock = _Clock()
    reg = _registry(probe, clock)
    await reg.probe("4090", force=True)
    fail["on"] = True
    clock.t += 61
    st = await reg.probe("4090")
    assert st.ok is False and st.probed is True
    assert "ConnectionError" in st.error
    assert (st.model, st.n_ctx, st.vision) == ("m1", 32768, True)   # what WAS there, still visible
    assert "4090" not in reg.detected_windows()                    # but not offered as a live window
    assert "DOWN" in reg.render_table()


@pytest.mark.asyncio
async def test_a_hung_probe_is_bounded_by_the_timeout():
    async def probe(spec, timeout):
        await asyncio.sleep(10)
        return _ok(spec, "never")

    reg = BackendRegistry(
        [BackendSpec(name="slow", provider="llama_cpp", base_url="http://x:8080")],
        ttl_s=60, timeout_s=0.05, probe=probe,
    )
    st = await reg.probe("slow")
    assert st.ok is False and "timed out" in st.error


def test_render_table_lists_refused_entries_and_the_only_primary_hint():
    reg = BackendRegistry.from_config({"model": {"provider": "llama_cpp"}, "backends": {"Bad": {"provider": "ollama", "base_url": "http://x"}}})
    table = reg.render_table()
    assert "Refused config entries" in table and "Bad" in table
    reg2 = BackendRegistry.from_config({"model": {"provider": "llama_cpp"}})
    assert "only the primary" in reg2.render_table()


@pytest.mark.asyncio
async def test_snapshot_is_json_ready_and_carries_every_field_the_status_surface_needs():
    async def probe(spec, timeout):
        return _ok(spec, "/m/Q.gguf")

    reg = _registry(probe, _Clock())
    await reg.probe_all()
    row = reg.snapshot()["local"]
    json.dumps(row)  # no dataclasses / tuples leaking
    assert set(row) >= {"name", "provider", "base_url", "ok", "probed", "stale", "model", "model_path",
                        "n_ctx", "vision", "latency_ms", "probed_at", "error", "changed_at",
                        "loaded_models", "available_models"}
