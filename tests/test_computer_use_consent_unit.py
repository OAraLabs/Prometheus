"""The unit of remembered consent for a desktop action, and its hard limit.

RULED BY WILL 2026-09-19, amended the same day to add the TARGET term:

    target:app:verb:delivery_mode

Hermes keys the same idea as ``cua:<action>:<background|foreground>``. Each of
the two extra terms earns its place the same way: "allow clicks", "allow
clicks in Mail", and "allow clicks in Mail on that machine" are three
different grants, and an operator would give exactly one. A key missing either
term cannot tell them apart — and a grant that silently spans machines is the
widening shape.

The target term could not have been added later. The moment a second machine
exists, a stored three-term value is ambiguous: it means either "where it was
granted" (provenance the record does not carry) or "anywhere" (the widening).
``Grant.from_config_dict`` therefore REFUSES a value that does not carry it,
rather than padding one in.

THE LIMIT, ESTABLISHED BEFORE IT COULD BE DISCOVERED AT A PROMPT
-----------------------------------------------------------------
The extent **cannot** express "type THIS text into THAT field", and no amount
of adding terms fixes it:

* the field — the only stable handle a driver offers is a snapshot-bound
  element token, which dies at the next observation. An extent keyed on one
  would be unrememberable by construction.
* the text — there is no term for a payload, and inventing one would mean
  storing the string in the grant, so "remember this" would mean "remember
  this exact sentence" and the next keystroke would prompt anyway.

So a payload-bearing action is NOT REMEMBERABLE AT ALL. That is the honest
answer and it is the same one SPRINT-CONSENT rule 4 already gives: consent
that cannot be described cannot be informed. The alternative — offering
``box:firefox:type_text:background`` off a prompt that showed one string — is
the narrow-prompt/wide-grant inversion that ``derive_grant`` was rewritten to
refuse (approving ONE file in $HOME once granted write_file across ALL of it).
"""

from __future__ import annotations

import pytest

from prometheus.computer.actions import (
    ClickInput, InvokeMenuInput, TypeTextInput,
)
from prometheus.permissions.approval_queue import (
    PendingAction, derive_grant, prospective_extents,
)
from prometheus.permissions.checker import Grant
from prometheus.permissions.computer_extent import (
    COMPUTER_ACTION_KIND, ComputerExtent, computer_extent_for,
)

CLICK_ARGS = {
    "target": "box", "app": "Firefox", "pid": 1, "window_id": 2,
    "snapshot_id": "s",
    "element_token": "t", "delivery_mode": "background",
}


def _extent(model, args) -> ComputerExtent:
    extent, unknown = computer_extent_for(
        "computer_x", args, schema=model.model_json_schema()
    )
    assert unknown is None, unknown
    assert extent is not None
    return extent


def _pending(extent, tool="computer_click") -> PendingAction:
    return PendingAction(
        request_id="r1", tool_name=tool, description="d",
        grant_computer_action=extent,
    )


# ── THE EXTENT ──────────────────────────────────────────────────────────────

def test_the_extent_is_target_app_verb_delivery():
    extent = _extent(ClickInput, CLICK_ARGS)
    assert extent.value == "box:firefox:click:background"


def test_the_app_is_case_folded_so_one_grant_covers_one_app():
    a = _extent(ClickInput, {**CLICK_ARGS, "app": "Firefox"})
    b = _extent(ClickInput, {**CLICK_ARGS, "app": "firefox"})
    assert a.value == b.value, (
        "two spellings of one app produced two extents — the operator would "
        "grant one and be asked again for the other"
    )


def test_a_colon_in_an_app_name_cannot_forge_an_extent():
    """`app` is operator-influenced; the value is colon-delimited."""
    extent = _extent(ClickInput, {**CLICK_ARGS, "app": "evil:click:background"})
    assert extent.value.count(":") == 3, (
        f"an app name containing colons forged a different extent: "
        f"{extent.value!r}"
    )
    assert extent.value == "box:evil_click_background:click:background"


# ── THE LIMIT: A PAYLOAD IS NOT REMEMBERABLE ────────────────────────────────

def test_a_click_is_rememberable():
    assert _extent(ClickInput, CLICK_ARGS).rememberable


def test_typing_is_NOT_rememberable():
    """The answer to 'can the extent express type THIS into THAT'. It cannot."""
    extent = _extent(TypeTextInput, {**CLICK_ARGS, "text": "hello"})
    assert extent.payload_params == ("text",)
    assert not extent.rememberable


def test_a_menu_path_is_NOT_rememberable():
    extent = _extent(InvokeMenuInput, {**CLICK_ARGS, "path": ["File", "Save"]})
    assert not extent.rememberable, (
        "'use any menu item in Firefox' and 'use File > Save' are different "
        "grants and the extent can only express the first"
    )


def test_a_payload_action_offers_no_lasting_scope_on_any_surface():
    """Both surfaces render from prospective_extents, which renders from
    derive_grant — so this is one check, not three."""
    extent = _extent(TypeTextInput, {**CLICK_ARGS, "text": "transfer 500"})
    action = _pending(extent, tool="computer_type_text")
    assert derive_grant(action, verb="always") is None
    assert prospective_extents(action) == {}, (
        "a lasting scope was offered for a payload-bearing action — the "
        "operator would be consenting to 'type ANY text into Firefox' off a "
        "prompt that showed one string"
    )


def test_a_click_DOES_offer_a_lasting_scope():
    """The refusal above must be specific, not a blanket 'never remember'."""
    action = _pending(_extent(ClickInput, CLICK_ARGS))
    grant = derive_grant(action, verb="always")
    assert grant is not None
    assert grant.kind == COMPUTER_ACTION_KIND
    assert grant.value == "box:firefox:click:background"
    assert prospective_extents(action), "no scope was offered for a click"


def test_a_desktop_grant_is_never_widened():
    """`… here` widens a path to its parent. An extent has no parent."""
    action = _pending(_extent(ClickInput, CLICK_ARGS))
    for verb in ("always", "always here", "until-restart", "until-restart here"):
        grant = derive_grant(action, verb=verb)
        if grant is not None:
            assert grant.widened is False, f"{verb} widened a desktop extent"


# ── THE SENTENCE A PERSON REFUSES ───────────────────────────────────────────

def test_the_description_reads_as_wide_as_the_grant_actually_is():
    """Will's requirement: renderable in a sentence a person can refuse.

    A grant covering every element in an app must SAY so. A description that
    sounded narrower than the grant is consent under a false description —
    the defect the whole consent sprint exists to remove.
    """
    grant = Grant(
        kind=COMPUTER_ACTION_KIND, value="box:firefox:click:background",
        tool_name="computer_click", scope="persistent",
    )
    text = grant.describe()
    assert "anything" in text, (
        f"the description does not convey that the grant covers every "
        f"element in the app: {text!r}"
    )
    assert "firefox" in text
    assert "background" in text and "raising" in text
    assert "until revoked" in text


def test_foreground_and_background_describe_differently():
    bg = Grant(kind=COMPUTER_ACTION_KIND, value="box:firefox:click:background",
               tool_name="computer_click").describe()
    fg = Grant(kind=COMPUTER_ACTION_KIND, value="box:firefox:click:foreground",
               tool_name="computer_click").describe()
    assert bg != fg
    assert "focus" in fg, f"the foreground sentence does not mention focus: {fg!r}"


def test_the_refusal_explains_why_nothing_can_be_remembered():
    extent = _extent(TypeTextInput, {**CLICK_ARGS, "text": "x"})
    why = extent.why_not_rememberable()
    assert "text" in why
    assert "once" in why, (
        f"the operator is told nothing is remembered but not what to do "
        f"instead: {why!r}"
    )


# ── MATCHING IS EXACT ───────────────────────────────────────────────────────

@pytest.mark.parametrize("stored,candidate,should_match", [
    ("box:firefox:click:background", "box:firefox:click:background", True),
    ("box:firefox:click:background", "box:firefox:click:foreground", False),
    ("box:firefox:click:background", "box:mail:click:background", False),
    ("box:firefox:click:background", "box:firefox:type_text:background", False),
    ("box:firefox:click:", "box:firefox:click:background", False),
    ("box:firefox:click:background", None, False),
    # THE AMENDMENT: the same app, verb and delivery on ANOTHER machine.
    ("box:firefox:click:background", "laptop:firefox:click:background", False),
    # And a pre-target value must never match anything.
    ("firefox:click:background", "box:firefox:click:background", False),
])
def test_grant_matching_is_exact_not_prefix(stored, candidate, should_match):
    """`firefox:click:` must not match every delivery mode.

    Every other grant kind is a prefix because paths and commands nest. An
    `app:verb:delivery` triple does not nest, and a prefix rule would silently
    fold the foreground variant into a background grant — collapsing the
    distinction the delivery term exists to keep.
    """
    grant = Grant(kind=COMPUTER_ACTION_KIND, value=stored,
                  tool_name="computer_click")
    assert grant.matches("computer_click", None, None, candidate) is should_match


def test_a_stored_desktop_grant_survives_a_config_round_trip():
    grant = Grant(kind=COMPUTER_ACTION_KIND, value="box:firefox:click:background",
                  tool_name="computer_click")
    back = Grant.from_config_dict(grant.to_config_dict())
    assert back is not None, (
        "a computer_action grant did not survive from_config_dict — it would "
        "be silently dropped on restart and the operator re-asked forever"
    )
    assert back.kind == COMPUTER_ACTION_KIND
    assert back.matches("computer_click", None, None,
                        "box:firefox:click:background")


# ── UNKNOWN IS LOUD ─────────────────────────────────────────────────────────

def test_an_action_that_names_no_app_is_unknown_not_ignorable():
    extent, unknown = computer_extent_for(
        "computer_click", {**CLICK_ARGS, "app": ""},
        schema=ClickInput.model_json_schema(),
    )
    assert extent is None
    assert unknown, (
        "an action with no app resolved to 'not a computer action' — it would "
        "fall through to evaluate's auto-allow tail"
    )


def test_an_unrecognised_delivery_mode_is_unknown():
    extent, unknown = computer_extent_for(
        "computer_click", {**CLICK_ARGS, "delivery_mode": "sneaky"},
        schema=ClickInput.model_json_schema(),
    )
    assert extent is None and unknown


def test_an_ordinary_tool_is_not_a_computer_action():
    """The declaration is the ONLY way in. Never the tool's name."""
    from prometheus.tools.base import BaseTool  # noqa: F401
    from pydantic import BaseModel

    class Ordinary(BaseModel):
        app: str

    extent, unknown = computer_extent_for(
        "computer_looks_like_one", {"app": "firefox"},
        schema=Ordinary.model_json_schema(),
    )
    assert extent is None and unknown is None, (
        "a model with no declared verb was treated as a computer action "
        "because of its NAME — the exact mistake this design avoids"
    )
