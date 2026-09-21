"""The /context meter measures the conversation; compaction says when it fails.

Two defects, one shape: a number produced by one component and reported by
another, with nothing asserting the two agreed.

  * /context computed ``headroom = available - system_prompt`` and labelled
    the result "used". The system prompt is a constant, so the reply was a
    constant — 35% on turn 1 and 35% on turn 100, observed on a live session
    the compactor had just measured at 76% of the same window. Every existing
    test passed, because each one checked a figure that was individually
    right: the window, its source, the failure path. None compared the
    reported occupancy against the acted-on occupancy.

  * ContextCompactor logged every pass at INFO whether or not it reached the
    threshold. A pass that lands over re-runs next turn on the same prefix,
    summarising its own summary, and is indistinguishable in the logs from
    one that worked.
"""

from __future__ import annotations

import logging
import re

import pytest

from prometheus.context.compactor import (
    ContextCompactor,
    compaction_threshold,
    compaction_threshold_from_config,
)
from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.gateway.commands import cmd_context
from prometheus.providers.base import ApiTextDeltaEvent

LOCAL_MODEL = "Qwen3.8-27B-UD-Q4_K_XL.gguf"
SERVER_N_CTX = 32768
SYSTEM_PROMPT = "system prompt " * 100

CONFIG = {
    "model": {"model": LOCAL_MODEL, "provider": "llama_cpp"},
    "context": {"effective_limit": 72000, "reserved_output": 2000},
    "compaction": {"reserve_tokens": 4096, "threshold_pct": 0.75},
}


class _Summarizer:
    async def stream_message(self, request):  # noqa: ANN001
        yield ApiTextDeltaEvent(text="summary.")


def _history(n_turns: int = 12, filler: int = 400) -> list[ConversationMessage]:
    msgs: list[ConversationMessage] = []
    for i in range(n_turns):
        msgs.append(ConversationMessage.from_user_text(f"q{i}: " + "x" * filler))
        msgs.append(ConversationMessage(
            role="assistant", content=[TextBlock(text=f"a{i}: " + "y" * filler)]))
    return msgs


def _render(messages) -> str:
    return cmd_context(
        SYSTEM_PROMPT, LOCAL_MODEL,
        local_model=LOCAL_MODEL, detected_limit=SERVER_N_CTX,
        config=CONFIG, messages=messages,
    )


def _field(text: str, label: str) -> int:
    m = re.search(rf"{label}:\s+([\d,]+) tokens", text)
    assert m is not None, f"{label} missing from:\n{text}"
    return int(m.group(1).replace(",", ""))


def _pct(text: str) -> int:
    m = re.search(r"In use:\s+[\d,]+ tokens \((\d+)%\)", text)
    assert m is not None, text
    return int(m.group(1))


# -- the meter ---------------------------------------------------------

def test_usage_moves_with_the_conversation():
    """The regression the old meter was structurally unable to fail."""
    assert _pct(_render([])) < _pct(_render(_history(2))) < _pct(_render(_history(40)))


def test_headroom_shrinks_as_the_conversation_grows():
    def headroom(msgs):
        return _field(_render(msgs), "Headroom")

    assert headroom(_history(40)) < headroom(_history(2)) < headroom([])


def test_unthreaded_session_is_not_reported_as_zero():
    """None means nobody measured it. Printing 0 would be the original defect
    one level down: a confident figure for an unmeasured thing."""
    text = cmd_context(
        "sys", LOCAL_MODEL, local_model=LOCAL_MODEL,
        detected_limit=SERVER_N_CTX, config=CONFIG,
    )
    assert "not measured" in text
    assert "Conversation:    0 tokens" not in text


def test_meter_and_compactor_agree_to_the_token():
    """Anti-drift. Equal, not merely close — they are one estimator now."""
    messages = _history(20)
    compactor = ContextCompactor(
        provider=_Summarizer(), model=LOCAL_MODEL, effective_limit=SERVER_N_CTX,
    )
    acted_on = compactor.estimate_total(SYSTEM_PROMPT, messages)
    assert _field(_render(messages), "In use") == acted_on


def test_compaction_threshold_has_one_implementation():
    assert compaction_threshold(32768, 4096, 0.75) == 21504
    assert compaction_threshold_from_config(32768, CONFIG) == 21504
    c = ContextCompactor(
        provider=_Summarizer(), model="m", effective_limit=32768,
        reserve_tokens=4096, threshold_pct=0.75,
    )
    assert c._threshold_tokens() == compaction_threshold(32768, 4096, 0.75)


def test_meter_shows_where_compaction_fires():
    """The operator's question is 'why is it compacting when I read 35%'. The
    threshold has to be on the same card as the total it is compared against."""
    assert "Compacts above:" in _render(_history(40))


# -- the protected tail ------------------------------------------------

def _compactor(**kw) -> ContextCompactor:
    defaults = dict(
        provider=_Summarizer(), model="test-model", effective_limit=3000,
        reserve_tokens=500, threshold_pct=0.4, protect_recent_turns=3,
    )
    defaults.update(kw)
    return ContextCompactor(**defaults)


def test_tail_holds_when_it_fits():
    """Yielding is the exception. A tail inside its budget is untouched."""
    c, h = _compactor(), _history(12)
    full = c._protected_boundary(h)
    assert c._protected_boundary(h, tail_budget=10**6) == full


def test_tail_yields_when_it_alone_cannot_fit():
    c, h = _compactor(), _history(12)
    full = c._protected_boundary(h)
    tail = c._tail_tokens(h, full)
    yielded = c._protected_boundary(h, tail_budget=tail // 2)
    assert yielded > full, "protection did not yield under an impossible budget"
    assert c._tail_tokens(h, yielded) < tail


def test_tail_never_yields_past_the_current_user_turn():
    """The floor: the question being answered is never summarised away, even
    when it alone busts the budget."""
    c, h = _compactor(), _history(12)
    last_user = max(i for i, m in enumerate(h) if m.role == "user")
    assert c._protected_boundary(h, tail_budget=0) <= last_user


# -- convergence -------------------------------------------------------

@pytest.mark.asyncio
async def test_non_convergence_is_reported(caplog):
    """A pass that finishes over the line must say so. It used to log the
    same INFO as a successful one and quietly re-run every turn."""
    c = _compactor()
    h = _history(12)
    # A final user turn larger than the whole threshold: the tail cannot fit
    # at any level of protection, so no span in front of it converges.
    h.append(ConversationMessage.from_user_text("huge: " + "z" * 12000))

    with caplog.at_level(logging.WARNING, logger="prometheus.context.compactor"):
        await c.apply(h, session_id="s", system_prompt="")

    warnings = [r.message for r in caplog.records]
    assert any("did NOT converge" in m for m in warnings), warnings
    # and it must name the gap, not merely complain
    assert any("exceeds the 1000 threshold by" in m for m in warnings), warnings


@pytest.mark.asyncio
async def test_convergent_pass_stays_quiet(caplog):
    """The warning has to mean something — a healthy pass must not emit it."""
    c = _compactor()
    with caplog.at_level(logging.WARNING, logger="prometheus.context.compactor"):
        await c.apply(_history(12), session_id="s", system_prompt="")

    assert not any("did NOT converge" in r.message for r in caplog.records)
