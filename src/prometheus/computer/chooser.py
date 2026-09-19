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
