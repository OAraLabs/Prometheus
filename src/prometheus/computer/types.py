"""Observation and candidate types — the structured facts the gate rules on.

THE POINT OF THE CANDIDATE TABLE
---------------------------------
In the jev-use shape (Cua RFC #3931), the CLIENT builds complete bounded
driver actions and the decision model selects only an ID. Prometheus therefore
knows exactly what an action is *before* it executes, in a form it constructed
itself — which is the difference that lets consent work here at all.

Registering a driver's tools as raw MCP tools does not have this property: the
adapter's input model is ``extra="allow"`` with no fields, so the gate reads an
empty schema and every argument is invisible to it (measured:
``gate_path_for("mcp__fs__write_file", {"path": "~/.ssh/id_rsa"})`` →
``(None, None)``). A candidate is the opposite — a typed object whose schema
we wrote.

STALENESS IS A FIRST-CLASS FIELD, NOT A CONVENTION
---------------------------------------------------
Cua binds element handles to a snapshot: a new ``get_window_state`` on the
same window invalidates every earlier token, and the driver returns an
explicit stale error. That is a real fail-closed property and the reason this
module carries ``snapshot_id`` on BOTH the observation and every candidate
built from it. A candidate whose snapshot no longer matches the live one is
refused HERE, before the gate and before the driver — three independent
refusals, because the one that survives a refactor is the one that matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Reserved candidate IDs the chooser may return instead of an action.
#: They are not actions and must never be executed — ``reobserve`` asks for a
#: fresh snapshot, ``abstain`` ends the turn. Cua's own recipe reserves the
#: same two; keeping the spelling identical means a chooser written against
#: either reads correctly here.
CANDIDATE_REOBSERVE = "reobserve"
CANDIDATE_ABSTAIN = "abstain"
RESERVED_CANDIDATE_IDS: frozenset[str] = frozenset(
    {CANDIDATE_REOBSERVE, CANDIDATE_ABSTAIN}
)


@dataclass(frozen=True)
class Element:
    """One accessible element from a window snapshot."""

    element_index: int
    #: Opaque per-snapshot handle. Dies with the snapshot — see the module
    #: docstring. Never used as a consent term for that reason.
    element_token: str
    role: str
    label: str = ""
    value: str | None = None
    actions: tuple[str, ...] = ()
    editable: bool = False

    def describe(self) -> str:
        """How this element reads to a human — and to the chooser.

        Role first, then label: a chooser scanning ten candidates is looking
        for "the button" or "the text field" before it is looking for a name.
        """
        label = f" {self.label!r}" if self.label else ""
        return f"{self.role}{label}".strip()


@dataclass(frozen=True)
class Observation:
    """One window snapshot: the facts a candidate table is built from.

    ``target`` rides on the observation rather than being passed alongside it
    so that every candidate built from a snapshot inherits the machine the
    snapshot came FROM. Threading it separately would make it possible — via
    one wrong argument — to build an action for machine B out of machine A's
    element tokens, which the gate would then rule on as B.
    """

    target: str
    app: str
    pid: int
    window_id: int
    snapshot_id: str
    elements: tuple[Element, ...] = ()
    #: Set when the observation could not be trusted (no display, empty tree,
    #: a stale session). An observation carrying this must NOT be built into
    #: candidates — see ``candidates.build_candidates``.
    unusable_reason: str | None = None

    def element_by_index(self, index: int) -> Element | None:
        for el in self.elements:
            if el.element_index == index:
                return el
        return None


@dataclass(frozen=True)
class Candidate:
    """One complete, bounded action the client is willing to execute.

    ``arguments`` is the WHOLE call. Nothing is filled in later, and nothing
    the chooser returns is merged into it — the chooser returns an ID and the
    ID selects this object unchanged. That is the boundary the whole design
    rests on, and it is why ``validate_choice`` can be three lines of
    comparison rather than a sanitiser.
    """

    candidate_id: str
    tool_name: str
    arguments: dict[str, Any]
    description: str
    snapshot_id: str
    #: Kept for the audit trail and the prompt — which element this targets,
    #: in human terms, since the token itself is unreadable.
    target_description: str = ""

    def chooser_view(self) -> dict[str, str]:
        """What the decision model is allowed to see.

        ⚠ ID AND DESCRIPTION ONLY. Not the tool name, not the arguments, not
        the element token, not the app. Cua's guide states the same boundary
        for jev-use — the request *"cannot contain Driver tool names or
        arguments, screenshot bytes, or environment data"* — and it is worth
        keeping even against a local chooser: the decision layer stays LESS
        privileged than the gate, which is the right way round and the
        opposite of a model driving raw tools.
        """
        return {"id": self.candidate_id, "description": self.description}


@dataclass
class ChoiceRequest:
    """What is sent to a chooser. Constructed from candidates, never raw."""

    goal: str
    snapshot_id: str
    candidates: list[dict[str, str]] = field(default_factory=list)
    history: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Choice:
    """What a chooser returns: an ID and nothing that could become an action."""

    candidate_id: str
    confidence: float | None = None
    #: Which chooser produced this, for telemetry and for the audit row.
    source: str = "unknown"
