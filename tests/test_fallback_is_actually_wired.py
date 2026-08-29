"""The fallback must be WIRED, not merely implemented.

#296 shipped the policy, the stream wrapper and `LoopContext.fallback`, with 31 passing tests —
and `build_fallback_target()` was never called anywhere. `context.fallback` was None in every
production path, so `enabled=context.fallback is not None` was always False and every terminal
provider failure ended the turn exactly as before. The feature was inert.

The suite could not see it: every test constructed `FallbackTarget(...)` explicitly and handed it
in, which is precisely the shape that proves the consumer works while saying nothing about
whether anything produces it. It took a live 401 against the real daemon to surface.

So this file asserts the PRODUCER side, statically, at every construction site — the one thing a
behavioural test with a hand-built context structurally cannot check.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "prometheus"

# (file, callee) pairs that build a real runtime context and therefore must pass `fallback`.
SITES = [
    ("daemon.py", "AgentLoop"),
    ("daemon.py", "LoopContext"),
    ("__main__.py", "LoopContext"),
]


def _calls(path: Path, callee: str):
    tree = ast.parse(path.read_text())
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name == callee:
                out.append(node)
    return out


@pytest.mark.parametrize("filename,callee", SITES)
def test_every_runtime_construction_passes_a_fallback(filename, callee):
    path = SRC / filename
    calls = _calls(path, callee)
    assert calls, f"no {callee}(...) call found in {filename} — did the site move?"
    for call in calls:
        kwargs = {k.arg for k in call.keywords if k.arg}
        assert "fallback" in kwargs, (
            f"{filename}:{call.lineno} builds {callee} WITHOUT fallback=. The loop reads "
            f"context.fallback; if nothing sets it the whole feature is inert and every test "
            f"that hands in a FallbackTarget by hand will still pass."
        )


def test_build_fallback_target_is_actually_called_somewhere():
    """The blunt version of the same check: a producer that no one calls is dead code."""
    callers = [
        p for p in SRC.rglob("*.py")
        if "build_fallback_target(" in p.read_text() and p.name != "fallback.py"
    ]
    assert callers, "build_fallback_target() is defined and never called — the feature is inert"


def test_the_local_model_config_yields_a_usable_target():
    """The default path: no `model.fallback` section means the local model IS the target."""
    from prometheus.engine.fallback import build_fallback_target

    target = build_fallback_target({
        "provider": "llama_cpp",
        "model": "Qwen3.8-27B-UD-Q4_K_XL.gguf",
        "base_url": "http://localhost:8080",
    })
    assert target is not None, "the ordinary local config must produce a target"
    assert target.is_local_backend is True
    assert target.model == "Qwen3.8-27B-UD-Q4_K_XL.gguf"


def test_a_BLANK_config_model_still_yields_a_target():
    """The recommended configuration has `model.model: ""`.

    The template says so itself — "A HINT, not an assertion. The BACKEND is authoritative ...
    Leave blank unless you need a name for a backend that cannot be asked." Requiring the config
    field therefore returned None on the exact setup this daemon ships with, so the fallback was
    inert a SECOND time, on a live daemon, after being wired.
    """
    from prometheus.engine.fallback import build_fallback_target

    live_shape = {"provider": "llama_cpp", "model": "", "base_url": "http://gpu:8080"}
    assert build_fallback_target(live_shape) is None, "blank and undetected is genuinely nothing"

    target = build_fallback_target(live_shape, detected_model="Qwen3.8-27B.gguf")
    assert target is not None, "a blank config with a DETECTED model must still give a target"
    assert target.model == "Qwen3.8-27B.gguf"


def test_an_explicit_config_model_outranks_the_detected_one():
    """An operator naming a fallback model is saying so deliberately."""
    from prometheus.engine.fallback import build_fallback_target

    target = build_fallback_target(
        {"provider": "llama_cpp", "model": "chosen.gguf", "base_url": "http://gpu:8080"},
        detected_model="detected.gguf",
    )
    assert target.model == "chosen.gguf"


def test_every_site_passes_the_detected_model_too():
    """Passing `fallback=` is not enough — passing it a config whose model is blank yields None."""
    import ast
    for filename, callee in SITES:
        tree = ast.parse((SRC / filename).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name != callee:
                continue
            fb = next((k.value for k in node.keywords if k.arg == "fallback"), None)
            assert isinstance(fb, ast.Call), f"{filename}:{node.lineno} fallback= is not a call"
            kw = {k.arg for k in fb.keywords if k.arg}
            assert "detected_model" in kw, (
                f"{filename}:{node.lineno} calls build_fallback_target without detected_model — "
                f"on the recommended blank-model config that returns None and the feature is inert"
            )
