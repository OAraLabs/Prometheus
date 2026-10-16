"""#773 — compaction: 80% default, per-model thresholds, projected-overflow
trigger, chunk-then-combine summarizer (nothing left out), passes that get
below the line (§6), and the dead ``context.compression_trigger`` key (§5).

Incident (2026-10-13): qwen3.8-max (DashScope) had a 1,000,000-token window
from ``cloud_default_limit``, so its compaction trigger was 746,928 tokens; it
broke down at ~241k and was never compacted. Separately, live telemetry showed
309 of 3,334 compactions LEFT MESSAGES OUT of the summarizer's input — the
worst replaced 1,685 messages (919,676 tokens) with a summary of 46 of them —
because the summarizer got one prompt capped at 18,000 characters.

Recorded/canned summarizer providers only — no live model.
"""

from __future__ import annotations

import pytest

from prometheus.context.compactor import (
    SUMMARY_MARKER_PREFIX,
    ContextCompactor,
    compaction_threshold,
)
from prometheus.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from prometheus.providers.base import ApiTextDeltaEvent
from tests.support.doubles import register_double


@register_double(
    "compaction773.summarizer",
    replaces="prometheus.providers.base.ModelProvider",
)
class _Summarizer:
    """Returns a short canned summary per call and records every request."""

    def __init__(self) -> None:
        self.requests: list = []

    async def stream_message(self, request):  # noqa: ANN001
        self.requests.append(request)
        yield ApiTextDeltaEvent(text=f"summary part {len(self.requests)}.")


def _prompt_text(request) -> str:  # noqa: ANN001
    return " ".join(m.text for m in request.messages)


def _history(n_turns: int, filler: int = 400) -> list[ConversationMessage]:
    msgs: list[ConversationMessage] = []
    for i in range(n_turns):
        msgs.append(ConversationMessage.from_user_text(f"MARK-U{i:04d} " + "x" * filler))
        msgs.append(ConversationMessage(
            role="assistant", content=[TextBlock(text=f"MARK-A{i:04d} " + "y" * filler)]))
    return msgs


def _compactor(provider=None, **kw) -> ContextCompactor:
    defaults = dict(
        provider=provider or _Summarizer(),
        model="local-27b",           # the SUMMARIZER's own model
        effective_limit=3000,        # ...and its window
        protect_recent_turns=3,
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
# 1. Threshold: 80% of the window by default; per model; no global cap
# ---------------------------------------------------------------------------


def test_default_threshold_is_80_percent_of_the_window():
    from prometheus.context.compactor import DEFAULT_THRESHOLD_PCT

    assert DEFAULT_THRESHOLD_PCT == 0.80
    # No reserve is subtracted: the projected-overflow rule covers output room.
    assert compaction_threshold(32768) == int(32768 * 0.80)
    assert compaction_threshold(1_000_000) == 800_000
    c = _compactor(effective_limit=32768)
    assert c.threshold_for() == (int(32768 * 0.80), "global pct")


def test_per_model_threshold_keys_and_the_window_is_unchanged():
    cfg = {
        "compaction": {"enabled": True},
        "context": {
            "cloud_default_limit": 1_000_000,
            "model_overrides": {
                "by-tokens": {"compaction_threshold_tokens": 200_000},
                "by-pct": {"compaction_threshold_pct": 0.15},
                "both": {"compaction_threshold_tokens": 200_000,
                         "compaction_threshold_pct": 0.15},
                "both-tokens-smaller": {"compaction_threshold_tokens": 90_000,
                                        "compaction_threshold_pct": 0.15},
            },
        },
    }
    c = ContextCompactor.from_config(cfg, provider=_Summarizer(), model="local-27b",
                                     detected_limit=32768)
    assert c.threshold_for("by-tokens") == (200_000, "model tokens")
    assert c.threshold_for("by-pct") == (150_000, "model pct")
    assert c.threshold_for("both") == (150_000, "model pct")
    assert c.threshold_for("both-tokens-smaller") == (90_000, "model tokens")
    assert c.threshold_for("no-entry") == (800_000, "global pct")
    # A threshold entry is NOT a window entry: the reported window stays real.
    assert c.limit_for("by-tokens") == 1_000_000
    assert c.limit_for("by-pct") == 1_000_000


# ---------------------------------------------------------------------------
# 2. Projected overflow: compact when the NEXT round would not fit
# ---------------------------------------------------------------------------


def _tool_round(i: int, result_chars: int) -> list[ConversationMessage]:
    return [
        ConversationMessage(role="assistant", content=[
            ToolUseBlock(id=f"c{i}", name="bash", input={"cmd": f"step {i}"})]),
        ConversationMessage(role="user", content=[
            ToolResultBlock(tool_use_id=f"c{i}", content=f"OUT{i} " + "z" * result_chars)]),
    ]


async def test_projected_overflow_triggers_compaction_under_the_threshold():
    provider = _Summarizer()
    c = _compactor(provider, effective_limit=10_000)   # threshold 8,000
    events = _events(c)
    msgs = _history(4, filler=1000)                    # ~2.1k tokens
    msgs += [ConversationMessage.from_user_text("now run the big job")]
    for i in range(3):
        msgs += _tool_round(i, result_chars=4 * 1500)  # ~1.5k tokens each
    before = c.estimate_total("", msgs)
    assert before < c.threshold_for()[0], "fixture: under the threshold"
    # The next request: current + 3,000 output + the largest recent result
    # (~1.5k) would exceed the 10,000 window.
    assert before + 3000 + 1500 > 10_000

    out = await c.apply(msgs, session_id="s", max_tokens=3000)

    assert out is not msgs and out[0].text.startswith(SUMMARY_MARKER_PREFIX)
    [(kind, payload)] = [e for e in events if e[0] == "context_compaction"]
    assert payload["trigger"] == "projected"


async def test_no_projection_no_compaction_when_everything_fits():
    provider = _Summarizer()
    c = _compactor(provider, effective_limit=10_000)
    msgs = _history(4, filler=400)
    out = await c.apply(msgs, session_id="s", max_tokens=500)
    assert out is msgs and not provider.requests


# ---------------------------------------------------------------------------
# 3. Chunk-then-combine: nothing left out, sized to the summarizer's window
# ---------------------------------------------------------------------------


async def test_a_span_far_larger_than_the_summarizer_window_is_chunked_not_cut():
    provider = _Summarizer()
    # Cloud session: 20k window (threshold 16k). Summarizer: the local 3k window.
    c = _compactor(provider, cloud_default_limit=20_000)
    events = _events(c)
    msgs = _history(90, filler=400)                    # ~19k tokens, over 16k

    out = await c.apply(msgs, session_id="cloud-s", model="cloud-model")

    assert out[0].text.startswith(SUMMARY_MARKER_PREFIX)
    assert len(provider.requests) > 2, "chunks plus a combine, not one call"
    window = c.limit_for("local-27b")
    for req in provider.requests:
        # Every summarizer request fits the summarizer's OWN window.
        assert c.estimate_total(req.system_prompt or "", req.messages) + req.max_tokens <= window
    # Nothing left out: every summarized message reached some chunk's input.
    summarized = int(out[0].text.split("–")[1].split("]")[0])
    seen = " ".join(_prompt_text(r) for r in provider.requests)
    for i in range(summarized // 2):
        assert f"MARK-U{i:04d}" in seen and f"MARK-A{i:04d}" in seen
    [(_, payload)] = [e for e in events if e[0] == "context_compaction"]
    assert payload["summarizer_input_omitted"] == 0
    assert payload["chunks"] >= 2


async def test_the_final_summary_budget_grows_with_the_span():
    small, big = _Summarizer(), _Summarizer()
    c_small = _compactor(small, cloud_default_limit=6_000)
    c_big = _compactor(big, cloud_default_limit=30_000)
    await c_small.apply(_history(14, filler=400), session_id="a", model="cloud-model")
    await c_big.apply(_history(120, filler=400), session_id="b", model="cloud-model")
    # The last request of a pass is the one that writes the final summary.
    assert big.requests[-1].max_tokens > small.requests[-1].max_tokens
    assert small.requests[-1].max_tokens >= 512


# ---------------------------------------------------------------------------
# 4. §6: a pass gets below the line, or says it cannot (once)
# ---------------------------------------------------------------------------


async def test_a_huge_current_turn_converges_by_trimming_old_tool_results():
    """The incident's shape: one user turn whose OWN rounds are most of the
    context. The protected tail cannot yield (one user turn), so before this
    the pass could never converge and re-ran every round."""
    provider = _Summarizer()
    # Eight protected turns: the protected tail alone (rounds 2-9) is far over
    # the budget, which is the shape that could never converge before.
    c = _compactor(provider, effective_limit=12_000, protect_recent_turns=8)
    events = _events(c)
    msgs = _history(4, filler=400)
    msgs.append(ConversationMessage.from_user_text("CURRENT-ASK build it"))
    for i in range(10):
        msgs += _tool_round(i, result_chars=4 * 1200)   # ~12k tokens of results

    out = await c.apply(msgs, session_id="s")

    threshold = c.threshold_for()[0]
    assert c.estimate_total("", out) <= threshold
    texts = [b.content if isinstance(b, ToolResultBlock) else getattr(b, "text", "")
             for m in out for b in m.content]
    # The question being answered and the last complete tool round are intact.
    assert any("CURRENT-ASK" in t for t in texts)
    # Every tool result in the view follows the call it answers (a result whose
    # call was summarised away is a 400 from every provider).
    seen_calls: set[str] = set()
    for m in out:
        for b in m.content:
            if isinstance(b, ToolUseBlock):
                seen_calls.add(b.id)
            if isinstance(b, ToolResultBlock):
                assert b.tool_use_id in seen_calls, b.tool_use_id
    assert any(t.startswith("OUT9 ") for t in texts)
    # Older results were trimmed with an explicit marker, never silently.
    assert any("trimmed by the compactor" in t for t in texts)
    # The originals are untouched (render view only).
    assert sum(isinstance(b, ToolResultBlock) and b.content.startswith("OUT0 ")
               for m in msgs for b in m.content) == 1
    [(_, payload)] = [e for e in events if e[0] == "context_compaction"]
    assert payload["converged"] is True
    assert payload["tool_results_trimmed"] > 0


async def test_a_pass_that_cannot_converge_is_marked_stuck_once_and_not_repeated():
    provider = _Summarizer()
    c = _compactor(provider, effective_limit=4_000)     # threshold 3,200
    events = _events(c)
    msgs = _history(6, filler=400)
    # One user message that ALONE exceeds the threshold: nothing can fix that.
    msgs.append(ConversationMessage.from_user_text("HUGE " + "q" * 4 * 4000))

    await c.apply(msgs, session_id="s")
    calls_after_first = len(provider.requests)
    await c.apply(msgs, session_id="s")                 # same size again

    stuck = [e for e in events if e[0] == "context_compaction_stuck"]
    assert len(stuck) == 1
    assert stuck[0][1]["session_id"] == "s"
    assert len(provider.requests) == calls_after_first, "no re-summarising while stuck"


# ---------------------------------------------------------------------------
# 5. §5: context.compression_trigger is retired
# ---------------------------------------------------------------------------


def test_compression_trigger_is_retired_everywhere():
    from pathlib import Path

    from prometheus.cli.migrate import HERMES_TO_PROMETHEUS_KEYS
    from prometheus.config.retired import retired_key_warnings

    root = Path(__file__).resolve().parents[1]
    template = (root / "config" / "prometheus.yaml.default").read_text(encoding="utf-8")
    assert "compression_trigger" not in template
    assert "compression_trigger" not in (root / "src/prometheus/cli/init.py").read_text(
        encoding="utf-8")
    assert HERMES_TO_PROMETHEUS_KEYS["compression.threshold"] == "compaction.threshold_pct"
    warnings = retired_key_warnings({"context": {"compression_trigger": 0.75}})
    assert len(warnings) == 1
    assert "context.compression_trigger" in warnings[0]
    assert "compaction.threshold_pct" in warnings[0]
    assert retired_key_warnings({"context": {}}) == []


def test_slash_context_reports_the_resolved_threshold_and_its_source():
    from prometheus.gateway.commands import context_threshold_line

    cfg = {"compaction": {"enabled": True},
           "context": {"model_overrides": {"m": {"compaction_threshold_tokens": 200_000}}}}
    line = context_threshold_line(1_000_000, cfg, model="m")
    assert "200,000" in line and "model tokens" in line
    line = context_threshold_line(32768, {"compaction": {"enabled": True}}, model="x")
    assert f"{int(32768 * 0.8):,}" in line and "global pct" in line


@pytest.fixture(autouse=True)
def _asyncio_mode():
    yield


async def test_a_pass_aims_at_70_percent_of_the_line_not_just_under_it():
    """§6: landing a few tokens under the line meant the next round crossed it
    again and the next pass re-summarised a longer prefix, every round."""
    c = _compactor(_Summarizer(), effective_limit=10_000, protect_recent_turns=30)
    events = _events(c)
    msgs = _history(40, filler=400)          # ~8.5k tokens, over the 8k threshold

    out = await c.apply(msgs, session_id="s", max_tokens=500)

    [(_, payload)] = [e for e in events if e[0] == "context_compaction"]
    assert payload["ceiling"] == 8000 and payload["target"] == 5600
    assert c.estimate_total("", out) <= payload["target"]


def test_the_loop_passes_the_request_max_tokens_to_the_compactor():
    """The projected-overflow rule needs this request's output room; without the
    argument it silently falls back to the fixed reserve. Since per-model output
    caps (0.9.12) that room is the model's own cap, the one _build_request sends;
    tests/test_output_caps.py checks the two values agree on a real turn."""
    import inspect

    from prometheus.engine import agent_loop

    src = inspect.getsource(agent_loop._run_loop)
    call = src[src.index("context.compactor.apply("):]
    call = call[:call.index("\n            except Exception:")]
    assert "max_tokens=_output_cap(context, context.model)" in call


def test_the_daemon_reports_retired_keys_at_startup():
    import inspect

    from prometheus import daemon

    assert "retired_key_warnings(config)" in inspect.getsource(daemon)
