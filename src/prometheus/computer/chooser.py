"""The decision seam — and the deterministic chooser milestone 1 runs on.

SCOPE, STATED ONCE
------------------
A chooser answers exactly one question: *which of these candidate IDs*. It is
not consulted about whether an action is permitted (that is the SecurityGate),
nor about whether the turn should continue (that is the agent loop). Those are
separate questions and at least one of them would be a bad idea to delegate to
a fast decision model.

WHY THE MOCK IS THE MILESTONE, NOT A PLACEHOLDER
-------------------------------------------------
TypeSafe Jev is waitlisted early access, and Cua's own jev-use guide documents
**no** timeout, retry, or degradation behaviour for it. So the fallback path
does not exist upstream and has to be constructed here rather than inherited.
``RuleChooser`` is that construction: a deterministic local chooser that needs
no credential, no network, and no account, and which a live chooser degrades
*to* rather than a stub a live chooser replaces.

``JevChooser`` is deliberately absent. When it is written it belongs behind
this same Protocol with a timeout whose expiry returns ``abstain`` — never a
guess, and never a silently retried call in a per-step hot path.

THE LABEL MAP — RULED 2026-09-20, BEFORE ``JevChooser`` EXISTS
---------------------------------------------------------------
Written now, while the reasoning is fresh, because the constraint was measured
rather than assumed and the measurement will not be repeated.

A logit-reading classifier (SimpleJev and anything shaped like it) scores the
model's next-token distribution over *allowed answer labels*, which means every
label must extend the rendered prompt by **exactly one distinct token**. Our
candidate IDs do not:

    click-0      3 tokens   ['click', '-', '0']
    key-return   3 tokens   ['key', '-', 'return']
    reobserve    4 tokens   ['re', '##ob', '##ser', '##ve']
    abstain      2 tokens   ['abs', '##tain']

Not one is single-token, and that includes both reserved IDs. The counts above
are WordPiece and so are directional, but the conclusion is not: no subword
tokenizer emits ``click-0`` as one token, because the hyphen and the digit are
always separate pieces. So a chooser backed by such a model MUST map candidate
IDs onto a single-token alphabet for the call and map the answer back.

Four rules govern that map.

1. **Labels bind to ``snapshot_id``, exactly as element tokens do.** A label is
   meaningful only for the table it was built from. A map built against one
   snapshot and applied to another decodes to a real ID in the wrong table —
   see rule 4 for why nothing downstream catches that.

2. **The map is a per-call returned value, never instance state.** Not
   ``self._labels``, not a cache keyed by anything. A chooser that stores its
   map between calls can answer request N+1 from request N's alphabet, and the
   failure is silent. Build it, use it, return it, drop it.

3. **A table larger than the backend's alphabet ABSTAINS, with a counter.** It
   does NOT refuse, and ``max_candidates`` does NOT move to accommodate a
   model. Abstaining degrades to the deterministic path, which is the whole
   design; refusing would make a backend limitation look like a policy decision
   to the caller, and shrinking the table would let the *model* dictate which
   actions a human is offered. The counter exists because an optimisation that
   silently never runs is indistinguishable from one that runs and helps.
   ⚠ Measured on this box: a ``gnome-calculator`` window builds **37**
   candidates. A 26-letter alphabet overflows on a pocket calculator, and even
   A–Z plus 0–9 leaves 36 against a cap of 40. This is the common case, not the
   edge case.

4. **Test the DECODE against a shuffled table — not the validation.** This is
   the one failure in the whole design that produces a *wrong action* rather
   than no action, and ``validate_choice`` is structurally unable to catch it:
   a stale or off-by-one map returns an ID that genuinely IS in the table, so
   validation passes and the gate is handed a correct-looking candidate that
   the chooser never meant to pick. A test that asserts "the returned ID was
   valid" passes on a map that is wrong in every position. The assertion has to
   be that the decoded candidate is the INTENDED one, on a table whose order
   differs from the label order.

None of this belongs inside the fast chooser's own timeout/fallback handling:
abstain-on-overflow is the chooser's job, degradation on timeout belongs to the
composite that holds both choosers.
"""

from __future__ import annotations

import logging
from typing import Protocol

from prometheus.computer.types import (
    CANDIDATE_ABSTAIN,
    Choice,
    ChoiceRequest,
)

log = logging.getLogger(__name__)


class Chooser(Protocol):
    """Selects one candidate ID from a bounded table."""

    name: str

    def choose(self, request: ChoiceRequest) -> Choice:
        """Return a Choice. MUST NOT fabricate an ID outside the table.

        Implementations are not trusted to honour that — ``validate_choice``
        checks — but an implementation that does is easier to debug than one
        that relies on being caught.
        """
        ...


class RuleChooser:
    """Deterministic, local, credential-free. The milestone-1 chooser.

    Scoring is intentionally simple and intentionally *stated*: a chooser
    whose reasoning is a paragraph of heuristics is one nobody can predict,
    and the entire value of this implementation is that a test can assert
    exactly which candidate it will pick.
    """

    name = "rule"

    def __init__(self, prefer: tuple[str, ...] = ()) -> None:
        #: Substrings that raise a candidate's score, highest priority first.
        #: The caller states its goal in terms the table uses.
        self._prefer = tuple(p.lower() for p in prefer)

    def choose(self, request: ChoiceRequest) -> Choice:
        if not request.candidates:
            return Choice(CANDIDATE_ABSTAIN, confidence=1.0, source=self.name)

        goal_words = {w for w in request.goal.lower().split() if len(w) > 2}
        best, best_score = None, -1.0
        for entry in request.candidates:
            score = self._score(entry.get("description", ""), goal_words)
            if score > best_score:
                best, best_score = entry, score

        if best is None or best_score <= 0:
            # NO GUESS. An abstain that says "nothing matched" is a usable
            # signal; a lowest-scoring pick dressed as a decision is not.
            return Choice(CANDIDATE_ABSTAIN, confidence=0.0, source=self.name)
        return Choice(
            best["id"],
            confidence=min(1.0, best_score / 10.0),
            source=self.name,
        )

    def _score(self, description: str, goal_words: set[str]) -> float:
        text = description.lower()
        score = 0.0
        # Explicit preferences dominate, in the order given.
        for i, token in enumerate(self._prefer):
            if token in text:
                score += 10.0 - i
        # Then ordinary goal-word overlap.
        score += sum(1.0 for w in goal_words if w in text)
        return score


class ScriptedChooser:
    """Replays a fixed sequence of IDs. For tests that pin a whole run.

    Separate from ``RuleChooser`` on purpose: a test that wants to prove the
    VALIDATION path (an unknown ID, a stale ID) needs a chooser that will
    happily return a bad one, and giving ``RuleChooser`` a "return something
    invalid" mode would weaken the thing under test.
    """

    name = "scripted"

    def __init__(self, ids: list[str]) -> None:
        self._ids = list(ids)
        self._i = 0

    def choose(self, request: ChoiceRequest) -> Choice:
        del request
        if self._i >= len(self._ids):
            return Choice(CANDIDATE_ABSTAIN, source=self.name)
        chosen = self._ids[self._i]
        self._i += 1
        return Choice(chosen, source=self.name)
