"""PR 3 of the multi-backend plan: Anatomy and the fallback chain CONSUME the
registry instead of carrying their own idea of which backends exist.

  * AnatomyScanner with a registry fills the model fields from the registry's
    probe of `local` and carries every backend's row; without one, its own
    HTTP probe path is unchanged
  * AnatomyWriter renders the backends table in ANATOMY.md and one clause per
    backend in the chat summary
  * build_fallback_target honours `model.fallback.backend: <name>` through
    the registry; an unknown name disables the fallback rather than guessing;
    the target carries `backend`
"""

from __future__ import annotations

import asyncio

import pytest

from prometheus.engine.fallback import build_fallback_target
from prometheus.infra.anatomy import AnatomyScanner, AnatomyState
from prometheus.infra.anatomy_writer import AnatomyWriter
from prometheus.providers.backends import BackendRegistry, BackendSpec, BackendStatus

CFG = {
    "model": {"provider": "llama_cpp", "base_url": "http://gpu-box:8080"},
    "backends": {
        "4090": {"provider": "llama_cpp", "base_url": "http://gpu-box:8080"},
        "mini": {"provider": "ollama", "base_url": "http://localhost:11434", "model": "qwen2.5:14b-instruct"},
    },
}


def _probe(down: set[str] = frozenset()):
    async def probe(spec: BackendSpec, timeout: float) -> BackendStatus:
        if spec.name in down:
            raise ConnectionError("connect timeout")
        path = "/models/Qwen3.8-27B-UD-Q4_K_XL.gguf" if spec.provider == "llama_cpp" else None
        st = BackendStatus(
            name=spec.name, provider=spec.provider, base_url=spec.base_url, ok=True,
            model=path or spec.model, model_path=path,
            n_ctx=32768 if spec.name != "mini" else 16384, vision=(spec.name == "local"), latency_ms=7.0,
        )
        st.extra["total_slots"] = 1 if spec.provider == "llama_cpp" else None
        return st
    return probe


def _registry(down: set[str] = frozenset()) -> BackendRegistry:
    reg = BackendRegistry.from_config(CFG, probe=_probe(down))
    asyncio.run(reg.probe_all())
    return reg


# ── anatomy ──────────────────────────────────────────────────────────────────


def test_scanner_reads_the_registry_instead_of_probing_itself():
    reg = _registry(down={"mini"})
    scanner = AnatomyScanner(llama_cpp_url="http://gpu-box:8080", registry=reg)
    state = AnatomyState()
    asyncio.run(scanner._detect_model(state))
    assert state.model_name == "/models/Qwen3.8-27B-UD-Q4_K_XL.gguf"
    assert state.model_quantization == "Q4_K_XL"          # parsed from the served id, as before
    assert state.vision_enabled is True                    # the registry's DETECTED flag
    assert "streaming" in state.inference_features and "multi_slot" in state.inference_features
    rows = {r["name"]: r for r in state.backends}
    assert set(rows) == {"local", "4090", "mini"}
    assert rows["mini"]["ok"] is False and "connect timeout" in rows["mini"]["error"]
    assert rows["4090"]["n_ctx"] == 32768


def test_scanner_without_a_registry_keeps_its_own_probe(monkeypatch):
    """The CLI / test path: no registry → the old HTTP detection runs (here
    against a closed port, so nothing is detected — and nothing is invented)."""
    scanner = AnatomyScanner(llama_cpp_url="http://127.0.0.1:9", inference_engine="llama_cpp")
    monkeypatch.setattr(scanner, "_check_cmdline_vision", _no_cmdline_vision)
    state = AnatomyState()
    asyncio.run(scanner._detect_model(state))
    assert state.model_name is None and state.backends == []


async def _no_cmdline_vision() -> bool:
    return False


def test_writer_renders_the_backends_table_and_summary_clause():
    reg = _registry(down={"mini"})
    scanner = AnatomyScanner(registry=reg)
    state = AnatomyState(hostname="box", inference_engine="llama_cpp", inference_url="http://gpu-box:8080")
    asyncio.run(scanner._detect_model(state))
    writer = AnatomyWriter()
    md = writer._render(state, [])
    assert "### Backends (registry)" in md
    assert "| 4090 | up | llama_cpp | Qwen3.8-27B-UD-Q4_K_XL | 32k | no | 7 ms |" in md
    assert "| mini | DOWN — ConnectionError: connect timeout |" in md
    summary = writer.render_summary(state)
    assert "Backends: 4090 up (Qwen3.8-27B-UD-Q4_K_XL, 32k); mini DOWN." in summary
    # a state without backends renders neither (the CLI path)
    bare = writer._render(AnatomyState(hostname="box"), [])
    assert "Backends (registry)" not in bare


# ── fallback ─────────────────────────────────────────────────────────────────


def test_fallback_backend_key_resolves_through_the_registry(monkeypatch):
    built: list[dict] = []
    monkeypatch.setattr(
        "prometheus.providers.registry.ProviderRegistry.create",
        lambda cfg: built.append(dict(cfg)) or object(),
    )
    reg = _registry()
    target = build_fallback_target(
        {"provider": "llama_cpp", "base_url": "http://gpu-box:8080", "fallback": {"enabled": True, "backend": "mini"}},
        detected_model="/models/Qwen3.8-27B-UD-Q4_K_XL.gguf",
        registry=reg,
    )
    assert target is not None
    assert target.backend == "mini" and target.provider_name == "ollama" and target.is_local_backend
    assert target.model == "qwen2.5:14b-instruct"                     # the registry's, not the primary's
    assert built[-1]["base_url"] == "http://localhost:11434"          # built from the registry's tuple


def test_fallback_unknown_backend_disables_rather_than_guessing(caplog):
    reg = _registry()
    with caplog.at_level("WARNING"):
        target = build_fallback_target(
            {"provider": "llama_cpp", "fallback": {"backend": "gone"}}, detected_model="m", registry=reg,
        )
    assert target is None
    assert any("names no configured backend" in r.message for r in caplog.records)
    # and with no registry at all, the same honest answer
    assert build_fallback_target({"provider": "llama_cpp", "fallback": {"backend": "mini"}}, detected_model="m") is None


def test_fallback_without_a_backend_key_is_unchanged(monkeypatch):
    monkeypatch.setattr("prometheus.providers.registry.ProviderRegistry.create", lambda cfg: object())
    target = build_fallback_target({"provider": "llama_cpp", "base_url": "http://gpu-box:8080"}, detected_model="m", registry=_registry())
    assert target is not None and target.backend is None and target.model == "m"
