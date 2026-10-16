"""The compactor counts in the server's tokens, and an over-window refusal is
compacted and retried once instead of losing the turn.

Incident (2026-10-15, live 0.9.10): a Beacon session on the local Qwen3.8-27B
(llama.cpp, n_ctx 32768) reached 28,915 prompt tokens by the server's count
with no compaction. The window (32,768, detected) and the threshold (26,214,
80%) were right; the trigger compared them with a chars/4 estimate that reads
low — real/estimate was a median 1.067 over 279 live compactions, up to 1.22,
and 1.19-1.32 on that session's web-search results. ``apply()`` exited silently
every round. The next large tool result would have been refused by llama.cpp
and the turn ended with "too long" (#704).

Two halves, pinned here:

* Calibration: the loop reports the server's count for each prompt; the
  compactor scales its estimate by that ratio (never below 1.0, capped), for
  the threshold, the projected-overflow rule, the pre-flight and the
  summarizer's chunk size. Ratio 1.0 changes nothing.
* Retry: a "too long" refusal teaches the compactor the real token count and
  the stated window (an unlisted cloud model is otherwise budgeted at
  ``cloud_default_limit``), and the round runs once more, compacted. A second
  refusal ends the turn as before.

Recorded/canned providers and the fake llama-server only — no live model.
"""

from __future__ import annotations

import asyncio

import pytest

from prometheus.context.compactor import (
    CALIBRATION_CEILING,
    CALIBRATION_FLOOR,
    ContextCompactor,
)
from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.providers.base import ApiTextDeltaEvent
from tests.fixtures.fake_llama_server import FakeLlamaServer
from tests.support.doubles import register_double

MODEL = "local-27b"


@register_double(
    "calibration.summarizer",
    replaces="prometheus.providers.base.ModelProvider",
)
class _Summarizer:
    """Returns a short canned summary per call."""

    def __init__(self) -> None:
        self.calls = 0

    async def stream_message(self, request):  # noqa: ANN001
        self.calls += 1
        yield ApiTextDeltaEvent(text=f"summary {self.calls}.")


def _history(n_turns: int, filler: int) -> list[ConversationMessage]:
    msgs: list[ConversationMessage] = []
    for i in range(n_turns):
        msgs.append(ConversationMessage.from_user_text(f"Q{i:03d} " + "x" * filler))
        msgs.append(ConversationMessage(
            role="assistant", content=[TextBlock(text=f"A{i:03d} " + "y" * filler)]))
    return msgs


def _compactor(**kw) -> ContextCompactor:
    defaults = dict(
        provider=_Summarizer(), model=MODEL, effective_limit=32768,
        detected_limit=32768, protect_recent_turns=3,
    )
    defaults.update(kw)
    return ContextCompactor(**defaults)


def _events(c: ContextCompactor) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []

    class _Tel:
        def record_signal_event(self, signal_type, payload, source_subsystem):  # noqa: ANN001
            out.append((signal_type, payload))

        def __getattr__(self, name):  # noqa: ANN001
            return lambda *a, **k: None

    c._telemetry = _Tel()
    c._envelope._telemetry = None
    return out


# ---------------------------------------------------------------------------
# 1. The ratio
# ---------------------------------------------------------------------------


def test_an_observation_is_clamped_and_never_lowers_the_count():
    c = _compactor()
    assert c.calibration_for("s", MODEL) == 1.0, "nothing observed: the bare estimate"

    assert c.observe_prompt_tokens(session_id="s", model=MODEL,
                                   estimated=25_600, reported=28_915) == pytest.approx(
        28_915 / 25_600)
    # A server that counts FEWER (a cached prefix left out, as Ollama's
    # prompt_eval_count does) must not make compaction fire later than today.
    assert c.observe_prompt_tokens(session_id="s", model=MODEL,
                                   estimated=10_000, reported=4_000) == CALIBRATION_FLOOR
    # One odd count (an image, say) must not shrink the window to nothing.
    assert c.observe_prompt_tokens(session_id="s", model=MODEL,
                                   estimated=1_000, reported=9_000) == CALIBRATION_CEILING
    # Missing numbers are no observation, and leave the last one in place.
    assert c.observe_prompt_tokens(session_id="s", model=MODEL,
                                   estimated=1_000, reported=0) is None
    assert c.calibration_for("s", MODEL) == CALIBRATION_CEILING


def test_a_session_uses_its_own_ratio_then_the_models_last_then_none():
    c = _compactor()
    c.observe_prompt_tokens(session_id="beacon:a", model=MODEL, estimated=1000, reported=1300)
    c.observe_prompt_tokens(session_id="beacon:b", model=MODEL, estimated=1000, reported=1100)
    assert c.calibration_for("beacon:a", MODEL) == pytest.approx(1.3)
    assert c.calibration_for("beacon:b", MODEL) == pytest.approx(1.1)
    # A session's first round borrows the model's most recent ratio...
    assert c.calibration_for("beacon:new", MODEL) == pytest.approx(1.1)
    # ...and another model (another tokenizer) borrows nothing.
    assert c.calibration_for("beacon:a", "claude-sonnet-5") == 1.0


# ---------------------------------------------------------------------------
# 2. The incident, replayed
# ---------------------------------------------------------------------------


def _incident_history() -> list[ConversationMessage]:
    """Text-only turns whose bare estimate sits just under 26,214 (80% of 32,768)."""
    msgs = _history(30, 1700)
    c = _compactor()
    est = c.estimate_total("", msgs)
    assert 24_000 < est < 26_214, est
    return msgs


def test_uncalibrated_the_incident_does_not_compact():
    """What the live daemon did: under the line by its own count, silently."""
    c = _compactor()
    events = _events(c)
    msgs = _incident_history()
    out = asyncio.run(c.apply(msgs, session_id="beacon:9c39", model=MODEL, max_tokens=4096))
    assert out is msgs
    assert events == []


def test_once_the_server_count_is_known_the_incident_compacts():
    c = _compactor()
    events = _events(c)
    msgs = _incident_history()
    est = c.estimate_total("", msgs)
    # The loop reports what llama.cpp counted for this prompt: 1.13x the estimate.
    c.observe_prompt_tokens(session_id="beacon:9c39", model=MODEL,
                            estimated=est, reported=int(est * 28_915 / 25_600))

    out = asyncio.run(c.apply(msgs, session_id="beacon:9c39", model=MODEL, max_tokens=4096))

    assert out is not msgs and len(out) < len(msgs)
    (kind, payload), = [e for e in events if e[0] == "context_compaction"]
    assert payload["trigger"] == "threshold"
    assert payload["tokens_before"] > 26_214, "counted in the server's tokens"
    assert payload["tokens_before_estimated"] == est
    assert payload["calibration"] == pytest.approx(28_915 / 25_600, abs=1e-3)
    assert payload["converged"] and payload["tokens_after"] <= payload["ceiling"]


def test_another_sessions_ratio_does_not_compact_this_one_early_once_it_has_its_own():
    c = _compactor()
    msgs = _incident_history()
    c.observe_prompt_tokens(session_id="beacon:urls", model=MODEL, estimated=1000, reported=1400)
    c.observe_prompt_tokens(session_id="beacon:chat", model=MODEL, estimated=1000, reported=1000)
    # The model's last ratio is now 1.0 (beacon:chat); the URL-heavy one keeps its own.
    out = asyncio.run(c.apply(msgs, session_id="beacon:chat", model=MODEL, max_tokens=4096))
    assert out is msgs


# ---------------------------------------------------------------------------
# 3. A stated window
# ---------------------------------------------------------------------------


#: A cloud model with no override and no known window (context/budget.py KNOWN_CLOUD_WINDOWS).
UNLISTED = "unlisted-cloud-model"


def test_a_refusal_caps_an_unlisted_cloud_models_window():
    c = _compactor()
    assert c.limit_for(UNLISTED) == 1_000_000  # cloud_default_limit: a guess
    assert not c.is_measured(UNLISTED)

    assert c.learn_window(UNLISTED, 131_072) is True
    assert c.limit_for(UNLISTED) == 131_072
    assert c.is_measured(UNLISTED), "the provider's own number"
    assert c.threshold_for(UNLISTED)[0] == int(131_072 * 0.80)
    # Only ever lowers; the local model is untouched.
    assert c.learn_window(UNLISTED, 262_144) is False
    assert c.limit_for(UNLISTED) == 131_072
    assert c.limit_for(MODEL) == 32768


def test_the_summarizer_chunks_in_the_servers_tokens():
    c = _compactor()
    bare = c._summarizer_window()
    c.observe_prompt_tokens(session_id="", model=MODEL, estimated=1000, reported=1250)
    assert c._summarizer_window() == int(32768 / 1.25)
    assert c._summarizer_window() < bare


# ---------------------------------------------------------------------------
# 4. The loop, against a server whose tokenizer counts more than the estimate
# ---------------------------------------------------------------------------


def _loop_context(box: FakeLlamaServer, compactor: ContextCompactor | None, **kw):
    from prometheus.engine.agent_loop import LoopContext
    from prometheus.providers.registry import ProviderRegistry

    provider = ProviderRegistry.create(
        {"provider": "llama_cpp", "base_url": box.url, "model": "fake-local-27b"})
    if compactor is not None:
        compactor._provider = provider
    return LoopContext(
        provider=provider, model="fake-local-27b",
        system_prompt=kw.pop("system_prompt", "s" * 400),
        max_tokens=kw.pop("max_tokens", 256), compactor=compactor, **kw,
    )


async def _run(ctx, msgs):
    from prometheus.engine.agent_loop import run_loop
    from prometheus.engine.stream_events import AssistantTurnComplete

    return [ev async for ev, _u in run_loop(ctx, msgs, surface="cli")
            if isinstance(ev, AssistantTurnComplete)]


def _fake_compactor(n_ctx: int) -> ContextCompactor:
    return ContextCompactor(
        provider=_Summarizer(), model="fake-local-27b", effective_limit=n_ctx,
        detected_limit=n_ctx, protect_recent_turns=2,
    )


@pytest.mark.asyncio
async def test_the_loop_reports_the_servers_count_to_the_compactor():
    comp = _fake_compactor(32768)
    with FakeLlamaServer(token_ratio=1.4) as box:
        # A system prompt big enough that both sides count the same text.
        ctx = _loop_context(box, comp, system_prompt="s" * 8000)
        done = await _run(ctx, [ConversationMessage.from_user_text("hello")])
    assert done and done[-1].stop_reason not in ("context_overflow", "context_preflight")
    assert comp.calibration_for("", "fake-local-27b") == pytest.approx(1.4, abs=0.08)


@pytest.mark.asyncio
async def test_a_refused_prompt_is_compacted_and_retried_once_and_the_turn_answers():
    n_ctx = 4000
    comp = _fake_compactor(n_ctx)
    msgs = _history(18, 300) + [ConversationMessage.from_user_text("and now?")]
    # Under the line by the bare estimate, over the window by the server's count.
    assert comp.estimate_total("s" * 400, msgs) < int(n_ctx * 0.8)
    with FakeLlamaServer(n_ctx=n_ctx, token_ratio=1.5) as box:
        ctx = _loop_context(box, comp)
        done = await _run(ctx, msgs)
        rows = box.chat_rows()

    # (The fake labels every request without tools "summary": the summarizer's
    # calls and, here, the answered retry alike.)
    outcomes = [r["outcome"] for r in rows]
    assert outcomes[0] == "400", outcomes
    assert outcomes.count("400") == 1, f"refused again: {outcomes}"
    assert len(outcomes) >= 3, f"no summarizer call before the retry: {outcomes}"
    assert done, "the turn ended without an answer"
    final = done[-1]
    assert final.stop_reason not in ("context_overflow", "context_preflight"), (
        final.stop_reason, final.message.text)
    assert "too long" not in (final.message.text or "")
    # What the server said is kept for the rest of the session.
    assert comp.calibration_for("", "fake-local-27b") > 1.3


@pytest.mark.asyncio
async def test_a_second_refusal_ends_the_turn_with_the_reason():
    """Retried once, not until it fits. Here nothing is compactable (one user
    message) and the server counts 2x, past the ratio cap, so the calibrated
    retry still clears the pre-flight and is refused again."""
    comp = _fake_compactor(700)
    with FakeLlamaServer(n_ctx=700, token_ratio=2.0) as box:
        ctx = _loop_context(box, comp, system_prompt="s" * 1800)
        done = await _run(ctx, [ConversationMessage.from_user_text("hello")])
        rows = box.chat_rows()
    assert [r["outcome"] for r in rows] == ["400", "400"]
    final = done[-1]
    assert final.stop_reason == "context_overflow", final.stop_reason
    assert "too long" in (final.message.text or "") and "/reset" in (final.message.text or "")


@pytest.mark.asyncio
async def test_when_the_servers_numbers_cannot_fit_the_pre_flight_says_so_unsent():
    """The ratio within the cap: the retry's pre-flight already knows it cannot
    fit, and ends the turn without a second request."""
    comp = _fake_compactor(700)
    with FakeLlamaServer(n_ctx=700, token_ratio=1.45) as box:
        ctx = _loop_context(box, comp, system_prompt="s" * 2000)
        done = await _run(ctx, [ConversationMessage.from_user_text("hello")])
        rows = box.chat_rows()
    assert [r["outcome"] for r in rows] == ["400"]
    final = done[-1]
    assert final.stop_reason == "context_preflight_refusal", final.stop_reason
    assert "/reset" in (final.message.text or "")


@pytest.mark.asyncio
async def test_without_a_compactor_an_overflow_is_still_sent_once():
    with FakeLlamaServer(n_ctx=50) as box:
        ctx = _loop_context(box, None)
        done = await _run(ctx, [ConversationMessage.from_user_text("hello")])
        rows = box.chat_rows()
    assert done[-1].stop_reason == "context_overflow"
    assert len(rows) == 1
