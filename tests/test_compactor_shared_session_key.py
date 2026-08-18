"""Two conversations sharing one session_id must not see each other's summaries.

The web path pins the literal ``"web"`` on a SHARED LoopContext, so every web
session reaches ``ContextCompactor.apply()`` under the same ``session_id``.
Surveyed 2026-08-18: no leakage, and no cost — a shared key totals the same
model calls as distinct keys.

WHY it is safe is the only reason it is safe: ``_span_key`` hashes the session
id AND THE SPAN CONTENT, so two different conversations produce different keys
whether or not they share an id. The id is redundant for correctness; the
content hash is load-bearing.

That makes this a pin, not a demonstration. Narrow ``_span_key`` to the session
id alone — which is exactly what a plausible "the key is too long / just use the
session" refactor would do — and cross-conversation leakage appears immediately
and silently. Asserted by mutation in the last test here, so the pin cannot be
decoration.

The compactor's key is BEHAVIOURAL (it partitions a cache and anchors spans,
both of which change what reaches the model), which is why it deliberately did
NOT receive the ``session_id or context.session_id`` precedence that the
telemetry writers did in #258. Descriptive readers get the real id; behavioural
ones keep the key they were built with.
"""

from __future__ import annotations

import asyncio

import pytest

from prometheus.context.compactor import ContextCompactor
from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.providers.base import ApiTextDeltaEvent

SHARED_ID = "web"  # what daemon.py pins on the shared web context


class _Summarizer:
    """Returns a per-call canned summary and counts calls."""

    def __init__(self) -> None:
        self.calls = 0
        self.text = "SUMMARY"

    async def stream_message(self, request):  # noqa: ANN001
        self.calls += 1
        yield ApiTextDeltaEvent(text=self.text)


def _history(tag: str, n_turns: int = 12, filler: int = 400):
    msgs = []
    for i in range(n_turns):
        msgs.append(ConversationMessage.from_user_text(f"{tag} question {i}: " + "x" * filler))
        msgs.append(ConversationMessage(
            role="assistant", content=[TextBlock(text=f"{tag} answer {i}: " + "y" * filler)]))
    return msgs


def _compactor(provider):
    return ContextCompactor(
        provider=provider, model="test-model", effective_limit=3000,
        reserve_tokens=500, threshold_pct=0.4, protect_recent_turns=3,
    )


def _text_of(msgs) -> str:
    out = []
    for m in msgs:
        for b in (m.content if isinstance(m.content, list) else []):
            out.append(getattr(b, "text", "") or "")
    return "\n".join(out)


def test_a_shared_session_id_does_not_leak_one_summary_into_the_other(monkeypatch):
    """THE PIN. Both conversations compact under the SAME id."""
    prov = _Summarizer()
    c = _compactor(prov)

    prov.text = "SUMMARY-OF-ALPHA: alpha decisions."
    alpha = _history("ALPHA")
    asyncio.run(c.apply(alpha, session_id=SHARED_ID))

    prov.text = "SUMMARY-OF-BETA: beta topics."
    beta = _history("BETA")
    rendered_beta = _text_of(asyncio.run(c.apply(beta, session_id=SHARED_ID)))

    assert "SUMMARY-OF-ALPHA" not in rendered_beta, (
        "one conversation's summary reached another's rendered context under a "
        "shared session_id — _span_key is no longer content-addressed"
    )
    assert "ALPHA" not in rendered_beta
    assert "BETA" in rendered_beta
    assert prov.calls == 2, "each conversation must be summarised on its own"


def test_a_shared_key_costs_nothing_versus_distinct_keys():
    """Sharing the id must not silently degrade into extra model calls, which
    is the other thing a shared cache key could plausibly do."""
    shared_prov = _Summarizer()
    shared = _compactor(shared_prov)
    distinct_prov = _Summarizer()
    distinct = _compactor(distinct_prov)

    # Different lengths so the second call CLOBBERS the first's anchor index.
    alpha, beta = _history("ALPHA", 12), _history("BETA", 30)

    for ctx, ids in ((shared, (SHARED_ID, SHARED_ID)), (distinct, ("web:a", "web:b"))):
        asyncio.run(ctx.apply(alpha, session_id=ids[0]))
        asyncio.run(ctx.apply(beta, session_id=ids[1]))
        asyncio.run(ctx.apply(alpha, session_id=ids[0]))   # alpha returns

    assert shared_prov.calls == distinct_prov.calls, (
        f"shared key cost {shared_prov.calls} model calls vs "
        f"{distinct_prov.calls} for distinct keys — the anchor clobber is no "
        "longer absorbed by the content-addressed cache"
    )


def test_span_key_is_content_addressed_not_session_addressed():
    """The property the pin rests on, asserted directly."""
    a, b = _history("ALPHA"), _history("BETA")
    same_id_diff_content = (
        ContextCompactor._span_key(SHARED_ID, a),
        ContextCompactor._span_key(SHARED_ID, b),
    )
    assert same_id_diff_content[0] != same_id_diff_content[1], (
        "same session, different conversations -> same key: leakage is possible"
    )
    # And the id still participates, so distinct sessions stay partitioned.
    assert ContextCompactor._span_key("web:a", a) != ContextCompactor._span_key("web:b", a)


def test_the_pin_goes_red_when_span_key_is_narrowed_to_the_session(monkeypatch):
    """MUTATION TEST. Narrowing _span_key to the session id alone IS the defect.

    If the leakage pin above survives this mutation it is decoration, so the
    mutation is applied here and the leakage is asserted to APPEAR.
    """
    import hashlib

    @staticmethod
    def _session_only_key(session_id: str, span: list) -> str:
        return hashlib.sha256(session_id.encode("utf-8", "replace")).hexdigest()

    monkeypatch.setattr(ContextCompactor, "_span_key", _session_only_key)

    prov = _Summarizer()
    c = _compactor(prov)

    prov.text = "SUMMARY-OF-ALPHA: alpha decisions."
    asyncio.run(c.apply(_history("ALPHA"), session_id=SHARED_ID))

    prov.text = "SUMMARY-OF-BETA: beta topics."
    rendered_beta = _text_of(asyncio.run(c.apply(_history("BETA"), session_id=SHARED_ID)))

    assert "SUMMARY-OF-ALPHA" in rendered_beta, (
        "the mutation did not reproduce the defect — either the leakage path "
        "changed or this mutation no longer models it, and the pin above must "
        "be re-derived rather than trusted"
    )
    assert prov.calls == 1, "beta reused alpha's cached summary, as the defect predicts"
