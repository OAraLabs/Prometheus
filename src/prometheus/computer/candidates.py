"""Building the candidate table, and validating what a chooser returns.

THE TABLE IS THE SECURITY BOUNDARY
-----------------------------------
The client builds complete bounded actions; the chooser returns an ID; the
client validates the ID against the table it built. Nothing the chooser says
is merged into an action — which is why ``validate_choice`` compares rather
than sanitises, and why an unknown ID is an error rather than a fallback.

``build_candidates`` REFUSES an unusable observation instead of returning an
empty table. Those are different states and conflating them is the failure the
precondition check exists to prevent: an empty table from a dead display reads
exactly like an empty table from a window with nothing actionable in it, and a
loop that shrugs at "no candidates" would report success having done nothing.
"""

from __future__ import annotations

from typing import Any

from prometheus.computer.actions import ALLOWED_KEYS
from prometheus.computer.types import (
    RESERVED_CANDIDATE_IDS,
    Candidate,
    ChoiceRequest,
    Observation,
)
from prometheus.permissions.computer_schema import DELIVERY_BACKGROUND


class UnusableObservation(RuntimeError):
    """The observation cannot be built into candidates, and why."""


class InvalidChoice(RuntimeError):
    """The chooser returned something outside the table. Always fatal."""


#: Roles a click is offered for. An enumeration, deliberately: offering a
#: click on every node in the tree makes a table nobody can review and hands a
#: chooser a hundred ways to hit something unintended.
_CLICKABLE_ROLES: frozenset[str] = frozenset({
    "push button", "button", "toggle button", "check box", "radio button",
    "menu item", "link", "list item", "tab",
})

_EDITABLE_ROLES: frozenset[str] = frozenset({
    "text", "entry", "password text", "paragraph", "document text",
})


def build_candidates(
    observation: Observation,
    *,
    text_to_type: str | None = None,
    max_candidates: int = 40,
) -> list[Candidate]:
    """Every bounded action worth offering for this snapshot.

    ``text_to_type`` is supplied by the CALLER, never by the chooser — the
    chooser picks which field to type a known string into, it does not get to
    say what the string is. That asymmetry is deliberate and it is what keeps
    the payload out of the decision layer entirely.
    """
    if observation.unusable_reason:
        raise UnusableObservation(observation.unusable_reason)

    base = {
        "app": observation.app,
        "pid": observation.pid,
        "window_id": observation.window_id,
        "snapshot_id": observation.snapshot_id,
        "delivery_mode": DELIVERY_BACKGROUND,
    }
    out: list[Candidate] = []

    for el in observation.elements:
        if len(out) >= max_candidates:
            break
        role = el.role.lower()
        if role in _CLICKABLE_ROLES:
            out.append(Candidate(
                candidate_id=f"click-{el.element_index}",
                tool_name="computer_click",
                arguments={**base, "element_token": el.element_token},
                description=f"Click the {el.describe()}",
                snapshot_id=observation.snapshot_id,
                target_description=el.describe(),
            ))
        if text_to_type is not None and (el.editable or role in _EDITABLE_ROLES):
            out.append(Candidate(
                candidate_id=f"type-{el.element_index}",
                tool_name="computer_type_text",
                arguments={
                    **base,
                    "element_token": el.element_token,
                    "text": text_to_type,
                },
                # The text is NAMED in the description so a human reviewing
                # the table sees it. The chooser sees this description too —
                # which is fine: it is the caller's own string, already known
                # to the caller. What the chooser never sees is the arguments.
                description=f"Type the prepared text into the {el.describe()}",
                snapshot_id=observation.snapshot_id,
                target_description=el.describe(),
            ))

    for key in ("return", "tab", "escape"):
        if key in ALLOWED_KEYS and len(out) < max_candidates:
            out.append(Candidate(
                candidate_id=f"key-{key}",
                tool_name="computer_press_key",
                arguments={**base, "key": key},
                description=f"Press {key}",
                snapshot_id=observation.snapshot_id,
                target_description=f"key {key}",
            ))

    _assert_unique_ids(out)
    return out


def _assert_unique_ids(candidates: list[Candidate]) -> None:
    """Duplicate IDs would make validation ambiguous. Fail at BUILD time.

    A table with two ``click-3`` entries turns ``validate_choice`` into a
    coin flip that looks like it succeeded. Cheaper to refuse the table.
    """
    seen: set[str] = set()
    for c in candidates:
        if c.candidate_id in seen:
            raise ValueError(f"duplicate candidate id {c.candidate_id!r}")
        if c.candidate_id in RESERVED_CANDIDATE_IDS:
            raise ValueError(
                f"candidate id {c.candidate_id!r} collides with a reserved id"
            )
        seen.add(c.candidate_id)


def build_choice_request(
    goal: str,
    observation: Observation,
    candidates: list[Candidate],
    history: list[str] | None = None,
) -> ChoiceRequest:
    """What the chooser is allowed to see. IDs and descriptions only."""
    return ChoiceRequest(
        goal=goal,
        snapshot_id=observation.snapshot_id,
        candidates=[c.chooser_view() for c in candidates],
        history=list(history or []),
    )


def validate_choice(
    candidate_id: str,
    candidates: list[Candidate],
    observation: Observation,
) -> Candidate | None:
    """Resolve a chosen ID to the action WE built, or fail closed.

    Returns None for the reserved IDs (``reobserve``/``abstain``) — those are
    legitimate answers that are not actions. Raises ``InvalidChoice`` for
    anything else that does not resolve, including a stale one.

    THREE CHECKS, and the order matters. Unknown before stale, because an
    unknown ID is a protocol violation (the chooser invented something) while
    a stale one is an ordinary race, and the two deserve different words in
    the log.
    """
    if not isinstance(candidate_id, str) or not candidate_id:
        raise InvalidChoice("chooser returned no candidate id")
    if candidate_id in RESERVED_CANDIDATE_IDS:
        return None

    match = next((c for c in candidates if c.candidate_id == candidate_id), None)
    if match is None:
        raise InvalidChoice(
            f"chooser returned {candidate_id!r}, which is not in the "
            f"{len(candidates)}-candidate table it was given"
        )
    if match.snapshot_id != observation.snapshot_id:
        raise InvalidChoice(
            f"candidate {candidate_id!r} was built from snapshot "
            f"{match.snapshot_id!r} but the live snapshot is "
            f"{observation.snapshot_id!r} — re-observe and rebuild"
        )
    return match


def action_arguments(candidate: Candidate) -> dict[str, Any]:
    """The driver call, unchanged from the moment it was built.

    A copy, so a caller mutating the returned dict cannot retroactively edit
    the thing that was validated and gated.
    """
    return dict(candidate.arguments)
