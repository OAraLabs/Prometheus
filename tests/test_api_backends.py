"""The registry's surfaces: GET /api/backends, POST /api/backends/{name}/probe,
the `backends` block on /api/status, and the shared /backends chat command.

A real BackendRegistry with an injected probe (no HTTP), so what these pin is
the plumbing: the routes read the SAME registry the daemon hands to the app,
a status call never probes (cache only), a list call probes through the TTL,
`?refresh=1` forces, an unknown name is a 404, and a daemon without a registry
answers 503 rather than an empty fleet.
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

from prometheus.providers import backends as backends_mod  # noqa: E402
from prometheus.providers.backends import BackendRegistry, BackendSpec, BackendStatus  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402


class _Clock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def _registry(calls: list[str], clock: _Clock) -> BackendRegistry:
    async def probe(spec: BackendSpec, timeout: float) -> BackendStatus:
        calls.append(spec.name)
        if spec.name == "down":
            raise ConnectionError("connection refused")
        return BackendStatus(
            name=spec.name, provider=spec.provider, base_url=spec.base_url, ok=True,
            model=f"/models/{spec.name}.gguf", model_path=f"/models/{spec.name}.gguf",
            n_ctx=32768, vision=(spec.name == "local"), latency_ms=2.0,
        )

    specs = [
        BackendSpec(name="local", provider="llama_cpp", base_url="http://a:8080", is_primary=True),
        BackendSpec(name="4090", provider="llama_cpp", base_url="http://b:8080"),
        BackendSpec(name="down", provider="ollama", base_url="http://c:11434"),
    ]
    return BackendRegistry(specs, ttl_s=60, timeout_s=1.0, probe=probe, clock=clock,
                           config_errors=["backends.Bad-Name: name must match ..."])


def _client(reg: BackendRegistry | None) -> TestClient:
    cfg = {"model": {"model": "m", "provider": "llama_cpp"}}
    return TestClient(create_app(cfg, backend_registry=reg))


@pytest.fixture(autouse=True)
def _no_process_registry(monkeypatch):
    # The routes fall back to the process-wide handle; keep tests hermetic.
    monkeypatch.setattr(backends_mod, "_REGISTRY", None)


# ── /api/status: cache only, never a probe ───────────────────────────────────


def test_status_renders_the_cache_without_probing():
    calls: list[str] = []
    reg = _registry(calls, _Clock())
    c = _client(reg)
    body = c.get("/api/status").json()
    assert calls == []                                  # a status call must not touch a box
    assert set(body["backends"]) == {"local", "4090", "down"}
    assert body["backends"]["4090"]["probed"] is False
    assert body["backends"]["4090"]["stale"] is True


def test_status_backends_is_null_without_a_registry():
    c = _client(None)
    assert c.get("/api/status").json()["backends"] is None


# ── /api/backends ────────────────────────────────────────────────────────────


def test_list_probes_through_the_ttl_and_reports_config_errors():
    calls: list[str] = []
    clock = _Clock()
    reg = _registry(calls, clock)
    c = _client(reg)
    body = c.get("/api/backends").json()
    assert sorted(calls) == ["4090", "down", "local"]
    rows = {r["name"]: r for r in body["backends"]}
    assert rows["4090"]["ok"] is True and rows["4090"]["n_ctx"] == 32768 and rows["4090"]["vision"] is False
    assert rows["local"]["vision"] is True
    assert rows["down"]["ok"] is False and "ConnectionError" in rows["down"]["error"]
    assert body["config_errors"] == ["backends.Bad-Name: name must match ..."]
    assert body["ttl_s"] == 60
    # second call inside the TTL: served from cache
    c.get("/api/backends")
    assert len(calls) == 3
    # ?refresh=1 forces every backend
    c.get("/api/backends?refresh=1")
    assert len(calls) == 6
    # and the status call now shows the probed rows, still without probing
    before = len(calls)
    st = c.get("/api/status").json()["backends"]
    assert len(calls) == before and st["4090"]["probed"] is True and st["4090"]["stale"] is False


def test_probe_one_forces_and_unknown_is_404():
    calls: list[str] = []
    reg = _registry(calls, _Clock())
    c = _client(reg)
    r = c.post("/api/backends/4090/probe")
    assert r.status_code == 200 and r.json()["ok"] is True and calls == ["4090"]
    r = c.post("/api/backends/4090/probe")
    assert calls == ["4090", "4090"]                    # forced, not cached
    r = c.post("/api/backends/nope/probe")
    assert r.status_code == 404 and "known" in r.json()["error"]


def test_routes_say_503_without_a_registry():
    c = _client(None)
    assert c.get("/api/backends").status_code == 503
    assert c.post("/api/backends/x/probe").status_code == 503


def test_routes_fall_back_to_the_process_registry(monkeypatch):
    calls: list[str] = []
    reg = _registry(calls, _Clock())
    monkeypatch.setattr(backends_mod, "_REGISTRY", reg)
    c = _client(None)
    assert c.get("/api/backends").status_code == 200
    assert c.get("/api/status").json()["backends"] is not None


# ── the shared chat command ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cmd_backends_renders_the_table_and_refresh_forces(monkeypatch):
    from prometheus.gateway.commands import cmd_backends

    calls: list[str] = []
    reg = _registry(calls, _Clock())
    monkeypatch.setattr(backends_mod, "_REGISTRY", reg)
    text = await cmd_backends(None)
    assert text.startswith("Backends")
    assert "local (primary): up" in text
    assert "DOWN — ConnectionError" in text
    assert "Refused config entries" in text
    n = len(calls)
    await cmd_backends("")
    assert len(calls) == n                              # cached
    await cmd_backends("refresh")
    assert len(calls) == n + 3                          # forced
    await cmd_backends(["refresh"])                     # Telegram passes a list
    assert len(calls) == n + 6


@pytest.mark.asyncio
async def test_cmd_backends_without_a_registry_says_so(monkeypatch):
    from prometheus.gateway.commands import cmd_backends

    monkeypatch.setattr(backends_mod, "_REGISTRY", None)
    assert "not initialized" in await cmd_backends(None)
