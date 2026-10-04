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

import re
import sys
from typing import Any

from prometheus.computer.actions import ALLOWED_KEYS
from prometheus.computer.types import (
    RESERVED_CANDIDATE_IDS,
    Candidate,
    ChoiceRequest,
    Element,
    Observation,
)
from prometheus.permissions.computer_schema import (
    DELIVERY_BACKGROUND,
    SITE_NONE,
    SITE_UNKNOWN,
)


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

#: Words in an app's name that mark it as a browser, an Electron app or a
#: WebView host. A FLOOR, NOT A CONFIG KEY (computer-use v1.1 §5.4.3): on
#: Linux the web-content flag only ever arrives as true, so a browser that has
#: not exposed its page shows only unflagged chrome and would otherwise pass as
#: a plain app. Matched per WORD of the name, so "Google Chrome" and
#: "chromium-browser" both hit. Over-matching costs a prompt; under-matching
#: would let a page pass as "no web content".
_WEB_HOST_WORDS: frozenset[str] = frozenset({
    # browsers
    "browser", "web", "firefox", "librewolf", "waterfox", "floorp",
    "chromium", "chrome", "brave", "edge", "msedge", "opera", "vivaldi",
    "epiphany", "falkon", "konqueror", "qutebrowser", "midori", "safari",
    # Electron and other embedded-web apps
    "electron", "code", "vscode", "vscodium", "codium", "cursor", "slack",
    "discord", "obsidian", "signal", "teams", "notion", "figma", "postman",
    "spotify", "1password", "bitwarden", "whatsapp", "element", "joplin",
    "logseq", "mattermost", "skype", "zoom",
    # WebView hosts
    "webkit", "webview", "evolution", "geary", "thunderbird", "yelp",
    "steam",
})

#: Platforms whose accessibility path can flag web content at all. Elsewhere
#: (the upstream Windows MSAA fallback, for one) the absence of a flag proves
#: nothing, so nothing there is ever "no web content".
_PLATFORMS_THAT_FLAG_WEB: tuple[str, ...] = ("linux",)


def site_of(observation: Observation, *, platform: str | None = None) -> str:
    """The ``site`` term for every action built from *observation*.

    ``-`` (positively no web content) ONLY on positive evidence; UNKNOWN
    otherwise. v1.1 never yields an origin — no provider reads one yet — so
    the answer is ``-`` or UNKNOWN. All of these must hold for ``-``:

    1. the walk is complete by OUR OWN evidence — not degraded, not
       truncated, and every node the driver counted was returned. Not the
       driver's ``elements_complete``: 0.28.2 hard-codes it false on Linux,
       and relying on it would make every site UNKNOWN;
    2. no node of the WHOLE walk is web content or document-family
       (``Observation.web_content_seen is False`` — None is no evidence);
    3. the app is not a browser, Electron or WebView host;
    4. the platform can flag web content at all.
    """
    plat = sys.platform if platform is None else platform
    if not plat.startswith(_PLATFORMS_THAT_FLAG_WEB):
        return SITE_UNKNOWN
    if observation.unusable_reason or observation.degraded:
        return SITE_UNKNOWN
    if observation.truncated:
        return SITE_UNKNOWN
    total = observation.total_element_count
    returned = observation.returned_element_count
    if total is None or returned is None or total != returned:
        return SITE_UNKNOWN
    if observation.web_content_seen is not False:
        return SITE_UNKNOWN
    if any(_is_web_or_document(el) for el in observation.elements):
        return SITE_UNKNOWN
    if _is_web_host(observation.app):
        return SITE_UNKNOWN
    return SITE_NONE


def _is_web_host(app: str) -> bool:
    words = {w for w in re.split(r"[^a-z0-9]+", app.lower()) if w}
    return bool(words & _WEB_HOST_WORDS)


def _is_document_role(role: str) -> bool:
    role = role.lower()
    return "document" in role or role == "embedded"


def _is_web_or_document(el: Element) -> bool:
    return el.in_web_content is True or _is_document_role(el.role)


def _inside_web_content(el: Element, by_index: dict[int, Element]) -> bool:
    """Flagged as web content, or under a document-family node we can see.

    Cua RFC 4268's rule: web content is an untrusted source. v1.1 offers none
    of it — page-authored labels never reach the chooser, and no page
    element can be clicked on the strength of an app-level answer.
    """
    if el.in_web_content is True:
        return True
    seen: set[int] = set()
    parent = el.parent_index
    while parent is not None and parent not in seen:
        seen.add(parent)
        node = by_index.get(parent)
        if node is None:
            return False
        if _is_document_role(node.role):
            return True
        parent = node.parent_index
    return False


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
    if not observation.elements:
        # ⚠ NOT "NOTHING TO DO". The key rows below are appended whatever the
        # tree holds, so an empty observation used to become a 3-row table —
        # Return, Tab and Escape aimed at a window nobody could see — and the
        # loop's "abstained" branch never fired (computer-use v1.1, D10).
        raise UnusableObservation(
            "the observation holds no elements, so there is nothing to build "
            "a bounded action from — an empty tree is not an idle window"
        )

    base = {
        "target": observation.target,
        "app": observation.app,
        # The WINDOW's site, on every row — keys and text go to focus, so a
        # key press is no narrower than the window it lands in.
        "site": site_of(observation),
        "pid": observation.pid,
        "window_id": observation.window_id,
        "snapshot_id": observation.snapshot_id,
        "delivery_mode": DELIVERY_BACKGROUND,
    }
    out: list[Candidate] = []
    by_index = {el.element_index: el for el in observation.elements}

    for el in observation.elements:
        if len(out) >= max_candidates:
            break
        if _inside_web_content(el, by_index):
            continue  # v1.1 offers no web content (§5.4.3)
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
