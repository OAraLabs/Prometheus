"""The computer-use extent the gate rules on, assembled from a tool's schema.

Companion to ``tool_paths.py``: that module answers "which absolute path does
this call target"; this one answers "which application, which verb, delivered
how". Both return a TWO-CHANNEL result — a value, or a reason it is unknown —
and both treat unknown as *prompt*, never as *allowed*. That contract is the
whole lesson of ``tool_paths``' docstring and it is reproduced here rather
than reinvented.

WHAT A PERSON IS ASKED TO REFUSE
--------------------------------
Will's requirement, 2026-09-19: *"write it so the approval prompt can render
it in a sentence a person can refuse."* ``describe()`` is that sentence, and
it is deliberately blunt about width::

    Prometheus may click anything in Firefox, in the background
    (without raising the window), until you revoke it.

Not "grant computer_click on firefox:click:background". If the honest sentence
reads too wide to accept, the answer is to refuse it and approve once — which
is precisely the judgement the sentence exists to enable. The extent value
(``firefox:click:background``) is the machine half; ``describe()`` is the
half a human rules on, and they are produced from the same object so they
cannot drift (Standing-Principles §17).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from prometheus.permissions.computer_schema import (
    DELIVERY_BACKGROUND,
    DELIVERY_MODES,
    declared_app_param,
    declared_computer_verb,
    declared_delivery_param,
    declared_payload_params,
)

#: Grant kind. Parallel to "path_prefix" / "command_prefix".
COMPUTER_ACTION_KIND = "computer_action"


@dataclass(frozen=True)
class ComputerExtent:
    """One computer-use action, in the terms consent is granted in."""

    app: str
    verb: str
    delivery: str
    #: Arguments the extent cannot describe. Non-empty => not rememberable.
    payload_params: tuple[str, ...] = ()

    @property
    def value(self) -> str:
        """The grant value: ``app:verb:delivery``."""
        return f"{self.app}:{self.verb}:{self.delivery}"

    @property
    def rememberable(self) -> bool:
        """Whether a lasting grant may be offered for this call.

        False whenever the call carries a payload the extent cannot name. See
        ``computer_schema``'s docstring: the alternative is a grant meaning
        "type anything into this app, forever" minted from a prompt that
        showed one string.
        """
        return not self.payload_params

    def describe(self) -> str:
        """The refusable sentence. Wide grants must READ wide."""
        where = (
            "in the background (without raising the window)"
            if self.delivery == DELIVERY_BACKGROUND
            else "in the foreground (raising the window, taking focus)"
        )
        return (
            f"Prometheus may {_verb_phrase(self.verb)} in {self.app}, "
            f"{where}"
        )

    def why_not_rememberable(self) -> str:
        """Operator-facing reason a lasting grant is not on offer."""
        args = ", ".join(self.payload_params)
        return (
            f"no lasting grant is offered: the value of {args} cannot be part "
            f"of a remembered grant, so remembering this would mean "
            f"'{_verb_phrase(self.verb)} in {self.app}' — approve it once "
            f"instead, each time"
        )


#: How each verb reads in the refusable sentence. A verb absent here still
#: works — it falls back to the raw verb — but it reads worse, and a new verb
#: should be added deliberately rather than inheriting a generic phrasing.
_VERB_PHRASES: dict[str, str] = {
    "click": "click anything",
    "scroll": "scroll anything",
    "press_key": "press keys",
    "type_text": "type any text",
    "set_value": "set any field value",
    "invoke_menu": "use any menu item",
    "observe": "read window contents",
}


def _verb_phrase(verb: str) -> str:
    return _VERB_PHRASES.get(verb, f"perform {verb}")


def computer_extent_for(
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    schema: dict[str, Any] | None = None,
) -> tuple[ComputerExtent | None, str | None]:
    """The extent this call targets, for the SecurityGate.

    Returns ``(extent, unknown_reason)``:

    * ``(extent, None)``  — a real extent the gate can rule on.
    * ``(None, None)``    — not a computer action at all (the common case).
    * ``(None, reason)``  — this IS a computer action and its extent could not
      be assembled. The caller MUST treat this as requiring approval; it must
      never fall through to "allowed".

    The third case is the one that matters, and it is why this returns a
    reason rather than just None. A computer tool whose author forgot to
    declare the app param would otherwise resolve to "not a computer action"
    and land in ``evaluate``'s auto-allow tail — reinstating the exact defect
    this change closes, in a tool nobody thought to check. Unmapped is LOUD.
    """
    verb = declared_computer_verb(schema)
    if verb is None:
        return None, None  # an ordinary tool; nothing to say

    app_param = declared_app_param(schema)
    if app_param is None:
        return None, (
            f"{tool_name} declares the computer verb {verb!r} but no argument "
            f"declaring the target application — the security gate cannot "
            f"rule on it"
        )

    raw_app = tool_input.get(app_param)
    app = str(raw_app).strip() if raw_app is not None else ""
    if not app:
        return None, (
            f"{tool_name} did not name a target application (argument "
            f"{app_param!r} is empty) — the security gate cannot rule on it"
        )

    # Delivery defaults to background when the tool declares no selector: the
    # SAFER of the two (no focus steal), and stated rather than assumed. A
    # tool that CAN go foreground must declare the param, which is what makes
    # the two extents distinguishable.
    delivery = DELIVERY_BACKGROUND
    delivery_param = declared_delivery_param(schema)
    if delivery_param is not None:
        raw = tool_input.get(delivery_param)
        if raw is not None:
            delivery = str(raw).strip().lower()
    if delivery not in DELIVERY_MODES:
        return None, (
            f"{tool_name} requested an unrecognised delivery mode "
            f"{delivery!r} — the security gate cannot rule on it"
        )

    return (
        ComputerExtent(
            app=_normalise_app(app),
            verb=verb,
            delivery=delivery,
            payload_params=declared_payload_params(schema),
        ),
        None,
    )


def _normalise_app(app: str) -> str:
    """Fold an app identifier to one spelling.

    ``Firefox``, ``firefox`` and ``FireFox`` must not become three separate
    grants — an operator who granted one would be asked again for the next and
    would reasonably read the second prompt as a bug. Colons are stripped
    because the grant value is colon-delimited and an app name containing one
    would forge a different extent (``evil:click:background`` inside the app
    field). That is a small thing that would be a real one later.
    """
    return app.replace(":", "_").strip().lower()
