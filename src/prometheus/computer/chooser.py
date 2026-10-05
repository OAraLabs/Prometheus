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
import re
from typing import Protocol

from prometheus.computer.actions import ALLOWED_KEYS
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


#: Whole words, lowercased. Quotes and punctuation around a label do not hide it.
_WORD = re.compile(r"[a-z0-9]+")

#: Words that say nothing about WHICH row. Every click row reads "Click the
#: …", so ``the`` matched every goal that contained it and nothing ever
#: abstained (#667, #668).
_FILLER = frozenset({
    "a", "an", "the", "in", "on", "at", "of", "to", "into", "onto", "for",
    "and", "or", "with", "from", "by", "it", "its", "this", "that", "my",
    "please", "then", "now",
})

#: The action kind each row's description starts with. candidates.py writes
#: them ("Click the …", "Press tab", "Set the … to the prepared text"), so
#: the kind is read from what the chooser is already allowed to see — never
#: from the tool name (``Candidate.chooser_view``).
_KIND_OF_FIRST_WORD = {"click": "click", "press": "key", "set": "set"}

KIND_CLICK, KIND_KEY, KIND_SET = "click", "key", "set"

#: The goal's leading verb → the kind it asks for. ``press``/``hit`` are
#: special: ``press tab`` is a key, ``press send`` is a button (a click).
_PRESS_VERBS = frozenset({"press", "hit", "push"})
_CLICK_VERBS = frozenset({"click", "tap", "select", "choose", "pick"})
_SET_VERBS = frozenset({"type", "fill", "set", "write", "input"})

#: Spellings people use for the keys the table offers as "Press return" /
#: "Press escape".
_KEY_ALIASES = {"enter": "return", "esc": "escape"}


def _words(text: str) -> list[str]:
    return _WORD.findall(str(text or "").lower())


def _row_kind(description: str) -> str | None:
    first = _words(description)[:1]
    return _KIND_OF_FIRST_WORD.get(first[0]) if first else None


def goal_kind_and_words(goal: str) -> tuple[str | None, set[str]]:
    """The action kind the goal's verb asks for, and the words that pick a row.

    The kind is None when the goal starts with no recognised verb ("save",
    "tab"). The returned words never include the verb or a filler word, and
    a key named after ``press`` is spelled the way the table spells it.
    """
    words = _words(goal)
    while words and words[0] == "please":
        words = words[1:]
    if not words:
        return None, set()
    verb, rest = words[0], [w for w in words[1:] if w not in _FILLER]
    kind: str | None = None
    if verb in _PRESS_VERBS:
        obj = _KEY_ALIASES.get(rest[0], rest[0]) if rest else ""
        kind = KIND_KEY if obj in ALLOWED_KEYS else KIND_CLICK
    elif verb in _CLICK_VERBS:
        kind = KIND_CLICK
    elif verb in _SET_VERBS:
        kind = KIND_SET
    if kind is None:
        rest = [w for w in words if w not in _FILLER]
    content = {_KEY_ALIASES.get(w, w) if kind == KIND_KEY else w for w in rest}
    return kind, content


class RuleChooser:
    """Deterministic, local, credential-free. The milestone-1 chooser.

    Scoring is intentionally simple and intentionally *stated*: a chooser
    whose reasoning is a paragraph of heuristics is one nobody can predict,
    and the entire value of this implementation is that a test can assert
    exactly which candidate it will pick. The rules (#667):

    1. **The goal's verb picks the kind.** ``press <key>`` presses a key;
       ``press``/``hit`` anything else presses a button, which is a click;
       ``click``/``tap``/``select`` click; ``type``/``fill``/``set``/``write``
       set a field. Rows of another kind are not considered, and when no row
       of that kind matches, the answer is abstain — never a row of another
       kind. (The goal "press Tab in the editor" clicked the editor's "New
       tab" button five times before this.)
    2. **Words match whole, and filler words never score.** "tab" is not in
       "Table", and "the" is in every "Click the …".
    3. **No verb, no guess between kinds.** With no recognised verb, a tie at
       the top between rows of different kinds abstains; list order (clicks
       are built first) must not decide.

    ``prefer`` substrings still dominate, in the order given — but only
    among the rows the goal's kind allows.
    """

    name = "rule"

    def __init__(self, prefer: tuple[str, ...] = ()) -> None:
        #: Substrings that raise a candidate's score, highest priority first.
        #: The caller states its goal in terms the table uses.
        self._prefer = tuple(p.lower() for p in prefer)

    def choose(self, request: ChoiceRequest) -> Choice:
        if not request.candidates:
            return Choice(CANDIDATE_ABSTAIN, confidence=1.0, source=self.name)

        kind, goal_words = goal_kind_and_words(request.goal)
        scored: list[tuple[float, str | None, dict]] = []
        for entry in request.candidates:
            description = entry.get("description", "")
            row_kind = _row_kind(description)
            if kind is not None and row_kind != kind:
                continue
            scored.append((self._score(description, goal_words), row_kind, entry))

        best_score = max((s for s, _, _ in scored), default=0.0)
        if best_score <= 0:
            # NO GUESS. An abstain that says "nothing matched" is a usable
            # signal; a lowest-scoring pick dressed as a decision is not.
            return Choice(CANDIDATE_ABSTAIN, confidence=0.0, source=self.name)
        top = [(k, e) for s, k, e in scored if s == best_score]
        if len({k for k, _ in top}) > 1:
            # Rule 3: a tie between kinds is not decided by build order.
            return Choice(CANDIDATE_ABSTAIN, confidence=0.0, source=self.name)
        return Choice(
            top[0][1]["id"],
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
        # Then whole-word overlap with the goal (rule 2).
        score += len(goal_words & set(_words(description)))
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
