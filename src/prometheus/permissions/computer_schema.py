"""How a computer-use tool DECLARES the extent the gate must rule on.

WHY THIS EXISTS — a file path is not the only kind of target
------------------------------------------------------------
``path_schema.py`` records three failures where a control decided "is this a
path?" from a parameter NAME and got it wrong. The fix was for the schema to
carry the fact. This module applies the same bargain to a second kind of
target, because computer use introduced one the gate had no vocabulary for.

``SecurityGate.evaluate`` rules on three terms: ``tool_name``, ``file_path``,
``command``. A click has none of them. Measured on ``origin/main`` before this
change, with the gate's own code and nothing stubbed::

    mcp__cua__click     -> PROMPT   (the ``mcp__`` prefix rule, checker.py:955)
    computer_click      -> ALLOW    (reason string '')
    computer_type_text  -> ALLOW
    computer_kill_app   -> ALLOW

⚠ READ THAT SECOND COLUMN. The only thing putting a third-party desktop action
in front of a human was the ``mcp__`` NAME PREFIX, and wrapping a driver as a
first-party toolset — the change that lets the gate see arguments at all —
DELETES that prefix. A wrap that lands without this module and without
``checker``'s computer rule is a regression: it would auto-approve clicks and
keystrokes with an empty reason. The wrap and the rule are one change.

THE EXTENT, AND WHAT IT DELIBERATELY CANNOT SAY
------------------------------------------------
Ruled by Will 2026-09-19: the unit of remembered consent is

    app : verb : delivery_mode          e.g. ``firefox:click:background``

Hermes keys the same idea as ``cua:<action>:<background|foreground>``. The
APP term is the difference and it is the whole argument: "allow clicks" and
"allow clicks in Mail" are different grants and only one of them is something
a person would give.

Two terms are ABSENT ON PURPOSE, and their absence is load-bearing:

* **The element.** Cua element tokens are snapshot-bound — a new
  ``get_window_state`` on the same window invalidates every prior token, and
  the driver returns an explicit stale error. An extent keyed on one would be
  unrememberable by construction: correct for safety, useless as consent.
* **The payload.** A grant to "type in Firefox" must not become a grant to
  type ANYTHING in Firefox. There is no stable term for "this text, into that
  field", so the extent cannot describe a payload-bearing call —

  **and therefore a payload-bearing call is NOT REMEMBERABLE AT ALL.**

  That is what ``COMPUTER_PAYLOAD_KEY`` marks. A tool that declares a payload
  param gets ``rememberable=False``, ``derive_grant`` returns None, and the
  operator is asked every time. This is the same honest answer SPRINT-CONSENT
  rule 4 already gives when extent cannot be determined, for the same reason:
  consent that cannot be described cannot be informed. Without it, a prompt
  reading *type "hello" into Search* would mint a grant meaning *type anything
  into Firefox, forever* — the exact narrow-prompt/wide-grant inversion that
  ``approval_queue.derive_grant`` was rewritten to stop (approving ONE file in
  $HOME once granted write_file across ALL of $HOME).

USAGE
-----
The verb sits on the model, the other terms sit next to the fields::

    class ClickInput(BaseModel):
        model_config = ConfigDict(json_schema_extra=computer_verb("click"))
        app: str = Field(..., json_schema_extra=COMPUTER_APP_FIELD)
        delivery_mode: str = Field("background",
                                   json_schema_extra=COMPUTER_DELIVERY_FIELD)

    class TypeTextInput(BaseModel):
        model_config = ConfigDict(json_schema_extra=computer_verb("type_text"))
        app: str = Field(..., json_schema_extra=COMPUTER_APP_FIELD)
        text: str = Field(..., json_schema_extra=COMPUTER_PAYLOAD_FIELD)
"""

from __future__ import annotations

from typing import Any

#: Schema key carrying the action verb. Sits on the schema ROOT (the model),
#: not on a field: the verb is a property of the tool, not of an argument.
#: Namespaced and ``x-`` prefixed exactly like ``x-prometheus-path``.
COMPUTER_VERB_KEY = "x-prometheus-computer-verb"

#: Field keys. Which argument names the target app, and which selects the
#: delivery mode. Declared per-field for the reason path_schema.py records at
#: length: a name pattern would be the fourth enumeration to get this wrong.
COMPUTER_APP_KEY = "x-prometheus-computer-app"
COMPUTER_DELIVERY_KEY = "x-prometheus-computer-delivery"

#: Field key marking an argument the extent CANNOT describe. Any tool with one
#: is never rememberable. See the module docstring — this is the safety device,
#: not a hint.
COMPUTER_PAYLOAD_KEY = "x-prometheus-computer-payload"

#: Drop-in ``Field(json_schema_extra=...)`` values.
COMPUTER_APP_FIELD: dict[str, Any] = {COMPUTER_APP_KEY: True}
COMPUTER_DELIVERY_FIELD: dict[str, Any] = {COMPUTER_DELIVERY_KEY: True}
COMPUTER_PAYLOAD_FIELD: dict[str, Any] = {COMPUTER_PAYLOAD_KEY: True}

#: The delivery modes Cua documents. ``background`` injects without raising
#: the target; ``foreground`` fronts it, acts, and restores. They are SEPARATE
#: extents and a background grant never covers the foreground variant — the
#: one piece of Hermes's key shape worth keeping verbatim, because the visible
#: variant steals focus from whatever the operator is actually doing.
DELIVERY_BACKGROUND = "background"
DELIVERY_FOREGROUND = "foreground"
DELIVERY_MODES: frozenset[str] = frozenset({DELIVERY_BACKGROUND, DELIVERY_FOREGROUND})


def computer_verb(verb: str) -> dict[str, Any]:
    """Model-level ``json_schema_extra`` declaring this tool's action verb."""
    return {COMPUTER_VERB_KEY: verb}


def _properties(schema: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {}
    props = schema.get("properties")
    return props if isinstance(props, dict) else {}


def declared_computer_verb(schema: dict[str, Any] | None) -> str | None:
    """The action verb this schema declares, or None if it declares none.

    A None here is what makes ``evaluate`` treat the call as an ordinary tool.
    It is deliberately the ONLY way in: a tool is a computer action because its
    author said so on the model, never because its name starts with
    ``computer_``. That is the naming-convention mistake this whole module
    exists to avoid repeating one level up.
    """
    if not isinstance(schema, dict):
        return None
    verb = schema.get(COMPUTER_VERB_KEY)
    return verb if isinstance(verb, str) and verb else None


def _field_declaring(schema: dict[str, Any] | None, key: str) -> str | None:
    """Name of the single field declaring *key*, or None."""
    for name, spec in _properties(schema).items():
        if isinstance(spec, dict) and spec.get(key) is True:
            return name
    return None


def declared_app_param(schema: dict[str, Any] | None) -> str | None:
    """Which argument names the target application."""
    return _field_declaring(schema, COMPUTER_APP_KEY)


def declared_delivery_param(schema: dict[str, Any] | None) -> str | None:
    """Which argument selects background/foreground delivery."""
    return _field_declaring(schema, COMPUTER_DELIVERY_KEY)


def declared_payload_params(schema: dict[str, Any] | None) -> tuple[str, ...]:
    """Every argument carrying data the extent cannot describe.

    Non-empty means NOT REMEMBERABLE. Returned as a tuple rather than a bool
    so the refusal can name the arguments that caused it — an operator told
    "this cannot be remembered" deserves to know which part of the call is the
    reason.
    """
    return tuple(
        name for name, spec in _properties(schema).items()
        if isinstance(spec, dict) and spec.get(COMPUTER_PAYLOAD_KEY) is True
    )
