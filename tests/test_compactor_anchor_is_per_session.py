"""One shared compactor, two conversations: separate anchors, separate cache.

THE DEFECT. ``daemon.py`` pins the literal ``"web"`` on the ONE ``LoopContext``
every web/Beacon/REST/WS session shares, and ``agent_loop`` used to hand that
literal to ``ContextCompactor.apply()``. So every web conversation keyed the
compactor's two pieces of per-session state — ``_session_spans`` (the anchored
span end) and the ``_cache`` partition — under the same string.

WHAT IT COST, and what it did NOT. It did not leak: ``_span_key`` hashes the
span CONTENT as well as the id, so a key can only collide when the messages are
byte-identical, and ``_session_spans`` is only ever consulted through a
content-keyed cache lookup (``anchored_key``) that a foreign anchor misses.
Nor could it drop content: ``span_end`` is bounded above by ``_select_span_end``
recomputed from THIS call's messages, so a foreign anchor can only make the span
SMALLER, never reach past a provenance barrier or the protected tail.

What it cost is model calls. The anchor exists so a GROWING conversation keeps
substituting the span it already paid to summarise; clobbered by every
interleaved session, each turn fell back to ``max_end`` — a longer prefix, a new
content key, a fresh summariser call. ``test_growth_*`` below is that cost.

WHY THIS FILE EXISTS ALONGSIDE ``test_compactor_shared_session_key.py``. That
file surveyed the same collision in #259 and concluded "no leakage, and no cost".
The leakage half is right and still pinned there. The no-cost half was an
artifact of its scenario: it replays FIXED-LENGTH histories, and a conversation
that never grows re-derives the same ``max_end`` and hits the content cache
anyway, which hides the anchor entirely. Grow the histories — what real
conversations do — and the shared key costs 2x. See that file's header note.
"""

from __future__ import annotations

import asyncio

from prometheus.context.compactor import ContextCompactor
from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.providers.base import ApiTextDeltaEvent

# What daemon.py pins on the shared web context, and what every web session
# therefore used to pass.
SHARED_ID = "web"


class _Summarizer:
    """Canned summary; counts the calls that are the whole point of the anchor."""

    def __init__(self) -> None:
        self.calls = 0
        self.text = "SUMMARY"

    async def stream_message(self, request):  # noqa: ANN001
        self.calls += 1
        yield ApiTextDeltaEvent(text=self.text)


def _history(tag: str, n_turns: int, filler: int = 400) -> list[ConversationMessage]:
    msgs: list[ConversationMessage] = []
    for i in range(n_turns):
        msgs.append(ConversationMessage.from_user_text(f"{tag} question {i}: " + "x" * filler))
        msgs.append(ConversationMessage(
            role="assistant", content=[TextBlock(text=f"{tag} answer {i}: " + "y" * filler)]))
    return msgs


def _compactor(provider: _Summarizer) -> ContextCompactor:
    return ContextCompactor(
        provider=provider, model="test-model", effective_limit=3000,
        reserve_tokens=500, threshold_pct=0.4, protect_recent_turns=3,
    )


def _interleave(compactor: ContextCompactor, ids: tuple[str, str], rounds: int = 6) -> None:
    """Two conversations taking alternating turns through ONE compactor, each
    growing by a turn as it goes — the shape every real session has."""
    alpha_turns, beta_turns = 12, 30
    for _ in range(rounds):
        asyncio.run(compactor.apply(_history("ALPHA", alpha_turns), session_id=ids[0]))
        asyncio.run(compactor.apply(_history("BETA", beta_turns), session_id=ids[1]))
        alpha_turns += 1
        beta_turns += 1


# -- the compactor's own state ------------------------------------------------

def test_distinct_ids_keep_distinct_anchors_in_session_spans() -> None:
    """ANCHOR PARTITION, asserted on ``_session_spans`` itself rather than on
    the render view — the render view can look right while the anchor is wrong,
    which is exactly how this survived a previous survey."""
    prov = _Summarizer()
    c = _compactor(prov)
    _interleave(c, ("web:alpha", "web:beta"))

    assert set(c._session_spans) == {"web:alpha", "web:beta"}, (
        f"expected one anchor per conversation, got {c._session_spans!r} — the "
        "compactor is no longer partitioning its anchor by session"
    )
    assert c._session_spans["web:alpha"] != c._session_spans["web:beta"], (
        "two conversations of different lengths anchored at the same span end; "
        "one of them is holding the other's anchor"
    )


def test_a_shared_id_collapses_both_anchors_into_one_slot() -> None:
    """The defect, stated positively, so the test above cannot pass vacuously."""
    prov = _Summarizer()
    c = _compactor(prov)
    _interleave(c, (SHARED_ID, SHARED_ID))

    assert set(c._session_spans) == {SHARED_ID}, (
        "the shared-id scenario no longer collapses the anchor — the defect "
        "this file guards has changed shape and the assertions must be re-derived"
    )


def test_distinct_ids_partition_the_summary_cache() -> None:
    """CACHE PARTITION. Every cache key must belong to exactly one of the two
    sessions: recomputing each entry's key under the other id must never match."""
    prov = _Summarizer()
    c = _compactor(prov)
    alpha, beta = _history("ALPHA", 12), _history("BETA", 30)
    asyncio.run(c.apply(alpha, session_id="web:alpha"))
    asyncio.run(c.apply(beta, session_id="web:beta"))

    assert len(c._cache) == 2
    alpha_end = c._session_spans["web:alpha"]
    alpha_key = ContextCompactor._span_key("web:alpha", alpha[:alpha_end])
    assert alpha_key in c._cache, "alpha's summary is not filed under alpha's key"
    assert ContextCompactor._span_key("web:beta", alpha[:alpha_end]) not in c._cache, (
        "alpha's cached summary is reachable under beta's session id — the "
        "session component of _span_key has stopped participating"
    )


# -- the cost the previous survey could not see -------------------------------

def test_growth_a_shared_id_pays_double_the_summariser_calls() -> None:
    """THE COST. Same interleaving, same growth, only the ids differ."""
    shared_prov = _Summarizer()
    _interleave(_compactor(shared_prov), (SHARED_ID, SHARED_ID))

    distinct_prov = _Summarizer()
    _interleave(_compactor(distinct_prov), ("web:alpha", "web:beta"))

    assert distinct_prov.calls == 6, (
        f"expected one paid call per conversation per round with a correct "
        f"anchor, got {distinct_prov.calls}"
    )
    assert shared_prov.calls > distinct_prov.calls, (
        f"shared id cost {shared_prov.calls} summariser calls, distinct ids "
        f"{distinct_prov.calls} — if these are equal the anchor has stopped "
        "being load-bearing and this file's premise is stale"
    )


def test_growth_distinct_ids_hold_their_anchor_across_a_growing_turn() -> None:
    """MECHANISM behind the count above: with its own id, a conversation that
    grows by one turn re-uses its anchored span and pays NOTHING."""
    prov = _Summarizer()
    c = _compactor(prov)

    asyncio.run(c.apply(_history("ALPHA", 12), session_id="web:alpha"))
    anchored_after_first = c._session_spans["web:alpha"]
    calls_after_first = prov.calls

    # Another session takes a turn in between — the interference that used to
    # destroy the anchor. It pays its own first call; that one is expected.
    asyncio.run(c.apply(_history("BETA", 30), session_id="web:beta"))
    calls_before_alpha_returns = prov.calls
    assert calls_before_alpha_returns == calls_after_first + 1, "beta's own first call"
    # ...then alpha comes back, one turn longer.
    asyncio.run(c.apply(_history("ALPHA", 13), session_id="web:alpha"))

    assert c._session_spans["web:alpha"] == anchored_after_first, (
        "alpha's anchor moved after an interleaved beta turn — beta's turn is "
        "reaching alpha's anchor slot"
    )
    assert prov.calls == calls_before_alpha_returns, (
        "alpha paid a summariser call for a turn its anchor already covered"
    )


# -- the wiring ---------------------------------------------------------------

def test_agent_loop_passes_the_turns_session_not_the_routing_namespace() -> None:
    """THE WIRING, pinned at the call site. ``effective_session_id`` is the
    turn's conversation; ``context.session_id`` is the literal "web". Reverting
    this argument is the defect, and nothing else in the file would notice."""
    import inspect

    from prometheus.engine import agent_loop

    src = inspect.getsource(agent_loop._run_loop)
    call = src[src.index("context.compactor.apply("):]
    call = call[:call.index(")\n")]
    assert "session_id=effective_session_id or \"\"" in call, (
        "the compactor is no longer keyed on the turn's session id; a shared "
        f"routing namespace would collapse every web conversation:\n{call}"
    )
    assert "session_id=context.session_id" not in call
