"""The backend names the model. The config is a hint.

THE ASYMMETRY
-------------
``context/budget.py`` already treats the backend as the source of truth for
one property: the server-reported ``n_ctx`` overrides the configured
``effective_limit`` for the local model. That mechanism was written, agreed
with, and applied — to the context WINDOW. It was never applied to the
model's IDENTITY, which stayed a config string detected once at boot.

So the same daemon believed the server about how big the window is, and the
config about what was answering. A config pin held ``model.model:
gemma4-26b`` while the rig served Qwen for SIX WEEKS. The telemetry still
carries 1,144 rows where the recorded ``model`` and the ``served_model`` the
backend returned disagree — deliberately not backfilled, because they are the
record of what happened.

WHAT THESE PIN
--------------
1. Detection logs TRANSITIONS, not every probe — it now runs on an interval,
   and a per-cycle INFO line would be noise that trains people to ignore it.
2. A model swap on the backend is a WARNING naming both sides.
3. Every OBSERVED disagreement warns, not just the first. This is the one the
   six weeks argues for: a warning that fires once at boot is a warning you
   scroll past.
4. An unreachable backend does NOT let the config quietly win — the last
   known name stands and nothing is invented.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from prometheus.daemon import _model_identity_loop
from prometheus.providers.llama_cpp import LlamaCppProvider


class _Resp:
    def __init__(self, model_id: str | None) -> None:
        self._id = model_id

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"data": ([{"id": self._id}] if self._id else [])}


class _Client:
    def __init__(self, script) -> None:
        self._script = script
        self.calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, **kw):
        out = self._script(self.calls)
        self.calls += 1
        if isinstance(out, Exception):
            raise out
        return _Resp(out)


def _patch(monkeypatch, script):
    holder = {}

    def _factory(*a, **k):
        if "c" not in holder:
            holder["c"] = _Client(script)
        return holder["c"]

    monkeypatch.setattr(
        "prometheus.providers.llama_cpp.httpx.AsyncClient", _factory
    )
    return holder


@pytest.mark.asyncio
async def test_first_detection_logs_once_at_info(monkeypatch, caplog):
    _patch(monkeypatch, lambda i: "qwen-a.gguf")
    p = LlamaCppProvider(base_url="http://x")
    with caplog.at_level(logging.DEBUG):
        assert await p.detect_loaded_model() == "qwen-a.gguf"
    info = [r for r in caplog.records if r.levelno == logging.INFO]
    assert any("Detected loaded model" in r.getMessage() for r in info)


@pytest.mark.asyncio
async def test_an_unchanged_reprobe_does_not_log_at_info(monkeypatch, caplog):
    """PROPERTY 1 — transitions only.

    Boot-only detection could afford an INFO line per call. A probe on an
    interval cannot: a line every cycle is noise, and noise is what makes a
    real transition invisible.
    """
    _patch(monkeypatch, lambda i: "qwen-a.gguf")
    p = LlamaCppProvider(base_url="http://x")
    await p.detect_loaded_model()
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        await p.detect_loaded_model()
    assert not [r for r in caplog.records if r.levelno >= logging.INFO], (
        "an unchanged re-probe logged at INFO or above; on an interval this "
        "becomes a line every cycle"
    )


@pytest.mark.asyncio
async def test_a_model_swap_warns_and_names_both_sides(monkeypatch, caplog):
    """PROPERTY 2 — the event worth a line gets one that is actionable."""
    _patch(monkeypatch, lambda i: "qwen-a.gguf" if i == 0 else "gemma-b.gguf")
    p = LlamaCppProvider(base_url="http://x")
    await p.detect_loaded_model()
    with caplog.at_level(logging.WARNING):
        await p.detect_loaded_model()
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("SERVED MODEL CHANGED" in m for m in msgs), msgs
    joined = " ".join(msgs)
    assert "qwen-a.gguf" in joined and "gemma-b.gguf" in joined, (
        "the swap warning must name BOTH sides — the old value is what "
        "anything stale is still keyed on"
    )


@pytest.mark.asyncio
async def test_every_observed_disagreement_warns_not_just_the_first(
    monkeypatch, caplog
):
    """PROPERTY 3 — the one the six weeks argues for.

    A warning that fires once at boot is a warning you scroll past. The
    config pin that held `gemma4-26b` against a Qwen rig was not undetectable
    — it was unremarked, every boot, for six weeks.
    """
    _patch(monkeypatch, lambda i: "actually-qwen.gguf")
    p = LlamaCppProvider(base_url="http://x")
    with caplog.at_level(logging.WARNING):
        task = asyncio.create_task(
            _model_identity_loop(p, "configured-gemma", 0.01)
        )
        await asyncio.sleep(0.08)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    disagreements = [
        r.getMessage() for r in caplog.records
        if "DISAGREES WITH THE BACKEND" in r.getMessage()
    ]
    assert len(disagreements) >= 2, (
        f"the disagreement warned {len(disagreements)} time(s); it must warn "
        f"on EVERY observation while it persists, or it becomes the silence "
        f"that cost six weeks"
    )
    assert "configured-gemma" in disagreements[0]
    assert "actually-qwen.gguf" in disagreements[0]


@pytest.mark.asyncio
async def test_an_unreachable_backend_does_not_let_the_config_win(
    monkeypatch, caplog
):
    """PROPERTY 4 — silence from the server is not consent for the hint.

    The failure mode being avoided: the probe fails, the loop 'falls back' to
    the configured name, and the config quietly becomes truth again — which
    is precisely the shape this PR exists to remove.
    """
    p = LlamaCppProvider(base_url="http://x")
    p.detected_model = "known-good.gguf"
    _patch(monkeypatch, lambda i: RuntimeError("connection refused"))

    with caplog.at_level(logging.WARNING):
        task = asyncio.create_task(_model_identity_loop(p, "configured", 0.01))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert p.detected_model == "known-good.gguf", (
        "an unreachable backend overwrote the last known served model"
    )
    assert not [
        r for r in caplog.records
        if "DISAGREES WITH THE BACKEND" in r.getMessage()
    ], "claimed a disagreement while the backend could not be asked"
