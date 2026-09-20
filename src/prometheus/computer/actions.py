"""The wrapped computer-use tools — typed schemas the security gate can read.

WHY THESE ARE DECLARED MODELS AND NOT A DYNAMIC PASSTHROUGH
------------------------------------------------------------
This is the entire reason the toolset is wrapped rather than registered as a
third-party MCP server. Measured on ``origin/main`` against the real gate:

    RAW MCP (``_McpDynamicInput``, extra="allow", no fields)
      schema                                   {'properties': {}, ...}
      gate_path_for(clipboard_write, ~/.ssh/id_rsa)   -> (None, None)

    WRAPPED (these models)
      declared paths                           {'file_path': 'file'}
      gate verdict on ~/.ssh/id_rsa            -> DENY

A declared schema is what turns "the gate cannot see this" into "the floor
already covers it". ``clipboard_write``'s path, ``browser_download``'s
destination and ``browser_set_input_files``' file list are ordinary
filesystem writes and reads wearing a UI-automation costume, and
``x-prometheus-path`` is exactly the right control for them.

⚠ AND THE OTHER HALF: declaring the schema is NOT sufficient on its own.
Wrapping removes the ``mcp__`` name prefix, which was the only thing keeping
these calls in front of a human — a wrapped ``computer_click`` with no path
and no command auto-allowed with an empty reason. Every model here therefore
also declares ``x-prometheus-computer-verb``, which is what routes it to the
gate's computer rule. The two declarations ship together or not at all.

WHAT IS IN v1, AND WHAT IS DELIBERATELY NOT
--------------------------------------------
In: observe, click, scroll, press_key, type_text, invoke_menu, verify.
Nine tools, all background-delivered, all snapshot-bound.

OUT, each for a reason rather than for scope:

* the whole ``browser_*`` subtree, ``page`` above all — ``execute_javascript``
  is arbitrary JS in the operator's logged-in browser and
  ``browser_set_input_files`` uploads named local files into a web form. Both
  are exfiltration primitives; Prometheus already has a browser tool.
* ``clipboard_read`` — returns whatever the operator last copied, which is
  routinely a password.
* ``kill_app`` (documented as ``kill -9``), ``launch_app``, ``install_ffmpeg``,
  ``set_config``, and the recording/replay family.
* ``foreground`` delivery. Background only in v1 — the extent keeps the two
  distinguishable so foreground can be added later WITHOUT inheriting the
  grants background already earned.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from prometheus.permissions.computer_schema import (
    COMPUTER_APP_FIELD,
    COMPUTER_DELIVERY_FIELD,
    COMPUTER_PAYLOAD_FIELD,
    COMPUTER_WINDOW_FIELD,
    COMPUTER_TARGET_FIELD,
    DELIVERY_BACKGROUND,
    computer_verb,
)

# ---------------------------------------------------------------------------
# Shared fields
#
# `target` and `app` are consent terms, so both are REQUIRED on every action:
# an action that could not name its machine or its app resolves to "extent
# unknown" and prompts forever. `pid`/`window_id`/`snapshot_id`/
# `element_token` are driver plumbing.
#
# WHY `target` IS ON THE ACTION AND NOT AMBIENT STATE. A per-session "current
# machine" would make the gate's subject depend on when the call happened
# rather than on what the call says — the same defect as resolving a relative
# path against the process's working directory, which permissions/ has now
# rejected twice. The action names its own machine.
# ---------------------------------------------------------------------------


class _ActionBase(BaseModel):
    """Fields every desktop action carries."""

    target: str = Field(
        ...,
        description=(
            "Target machine: a logical name declared in config, never a "
            "hostname or address."
        ),
        json_schema_extra=COMPUTER_TARGET_FIELD,
    )
    app: str = Field(
        ...,
        description="Target application (the consent term).",
        json_schema_extra=COMPUTER_APP_FIELD,
    )
    pid: int = Field(
        ..., description="Target process id.",
        json_schema_extra=COMPUTER_WINDOW_FIELD,
    )
    window_id: int = Field(
        ..., description="Target window id.",
        json_schema_extra=COMPUTER_WINDOW_FIELD,
    )
    snapshot_id: str = Field(
        ...,
        description=(
            "The observation this action was built from. Refused if the live "
            "snapshot has moved on."
        ),
    )
    delivery_mode: str = Field(
        DELIVERY_BACKGROUND,
        description="background (no focus steal) or foreground.",
        json_schema_extra=COMPUTER_DELIVERY_FIELD,
    )


class ObserveInput(BaseModel):
    """Read a window's accessibility tree. The loop's first step."""

    model_config = ConfigDict(json_schema_extra=computer_verb("observe"))

    target: str = Field(..., json_schema_extra=COMPUTER_TARGET_FIELD)
    app: str = Field(..., json_schema_extra=COMPUTER_APP_FIELD)
    pid: int = Field(..., json_schema_extra=COMPUTER_WINDOW_FIELD)
    window_id: int = Field(..., json_schema_extra=COMPUTER_WINDOW_FIELD)


class ClickInput(_ActionBase):
    model_config = ConfigDict(json_schema_extra=computer_verb("click"))

    element_token: str = Field(
        ...,
        description="Opaque per-snapshot element handle from the observation.",
    )


class ScrollInput(_ActionBase):
    model_config = ConfigDict(json_schema_extra=computer_verb("scroll"))

    direction: str = Field(..., description="up, down, left or right.")
    amount: int = Field(1, ge=1, le=50)


class PressKeyInput(_ActionBase):
    """A single named key.

    NOT a payload: the key set is closed and enumerated (``_ALLOWED_KEYS``),
    so the extent CAN describe it — "press keys in Firefox" is a statement
    whose reach an operator can picture. Free-form text is a different
    question and lives in ``TypeTextInput``.
    """

    model_config = ConfigDict(json_schema_extra=computer_verb("press_key"))

    key: str = Field(..., description="return, tab, escape, up/down/left/right, …")


class TypeTextInput(_ActionBase):
    """Insert text.

    ⚠ ``text`` IS DECLARED AS A PAYLOAD, and that declaration is the whole
    safety property of this tool. It makes the call NOT REMEMBERABLE: the
    extent ``target:app:verb:delivery`` has no term for a string, so a
    remembered grant would mean "type ANY text into this app on this machine,
    forever" — minted from a prompt that showed one string. See
    permissions/computer_schema.py.
    """

    model_config = ConfigDict(json_schema_extra=computer_verb("type_text"))

    text: str = Field(
        ...,
        description="The text to insert.",
        json_schema_extra=COMPUTER_PAYLOAD_FIELD,
    )
    element_token: str = Field(..., description="Target element from the observation.")


class InvokeMenuInput(_ActionBase):
    """Invoke a menu path.

    ``path`` is a payload for the same reason ``text`` is: "use any menu item
    in Firefox" and "use File > Save" are different grants, and the extent can
    only express the first. Cua documents that a missing, ambiguous, disabled
    or structurally mismatched segment fails closed on the driver side too.
    """

    model_config = ConfigDict(json_schema_extra=computer_verb("invoke_menu"))

    path: list[str] = Field(
        ...,
        min_length=1,
        description="Menu path, e.g. ['File', 'Save'].",
        json_schema_extra=COMPUTER_PAYLOAD_FIELD,
    )


class VerifyInput(BaseModel):
    """Check an expectation against fresh state. Read-only, post-action."""

    model_config = ConfigDict(json_schema_extra=computer_verb("observe"))

    target: str = Field(..., json_schema_extra=COMPUTER_TARGET_FIELD)
    app: str = Field(..., json_schema_extra=COMPUTER_APP_FIELD)
    pid: int = Field(..., json_schema_extra=COMPUTER_WINDOW_FIELD)
    window_id: int = Field(..., json_schema_extra=COMPUTER_WINDOW_FIELD)
    expect_role: str | None = None
    expect_label: str | None = None
    expect_value: str | None = None


#: Keys ``press_key`` will dispatch. A CLOSED set, so the extent's claim that
#: "press keys" is describable stays true. Anything not here is refused before
#: the gate — adding one is a deliberate act, which is the point.
ALLOWED_KEYS: frozenset[str] = frozenset({
    "return", "enter", "tab", "escape", "space", "backspace", "delete",
    "up", "down", "left", "right", "home", "end", "pageup", "pagedown",
})

#: Verbs this package implements, mapped to their input model. The registry
#: derives from THIS — never a hand-kept second list.
ACTION_MODELS: dict[str, type[BaseModel]] = {
    "observe": ObserveInput,
    "click": ClickInput,
    "scroll": ScrollInput,
    "press_key": PressKeyInput,
    "type_text": TypeTextInput,
    "invoke_menu": InvokeMenuInput,
    "verify": VerifyInput,
}


def schema_for(verb: str) -> dict[str, Any]:
    """The JSON schema the gate reads for *verb*."""
    model = ACTION_MODELS.get(verb)
    if model is None:
        raise KeyError(f"no computer action model for verb {verb!r}")
    return model.model_json_schema()
