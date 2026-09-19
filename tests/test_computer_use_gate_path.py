"""A wrapped desktop action must NOT auto-allow. Through the real dispatch path.

THE DEFECT THIS FILE PINS
--------------------------
Registering a desktop driver's tools as raw MCP tools inherits the MCP consent
gap: the adapter's input model is ``extra="allow"`` with no fields, so the gate
reads an empty schema and every argument is invisible to it. The obvious fix
— wrap the driver as a first-party toolset with declared schemas — is right,
and on its own it is a REGRESSION. Measured on ``origin/main`` before this
change, against the real gate:

    mcp__cua__click      -> PROMPT    (checker.py's ``mcp__`` prefix rule)
    computer_click       -> ALLOW     reason ''
    computer_type_text   -> ALLOW
    computer_kill_app    -> ALLOW

A click carries no ``file_path`` and no ``command``, so every tier in
``evaluate`` is skipped and it falls through to "Auto-allowed" at the tail.
**The only thing holding a third-party desktop action in front of a human was
the ``mcp__`` NAME PREFIX** — and wrapping deletes that prefix. The protection
lived in the name, and the refactor credited with improving it removed it.

WHY THESE TESTS GO THROUGH ``_execute_tool_call``
--------------------------------------------------
``tests/test_gate_sees_the_path.py`` records why at length: every test of the
old path-defect invoked the gate directly and handed it a path BY HAND —
supplying an argument the caller never supplied — and a 7-of-7 verification
passed an hour before the defect was found. So every behavioural test here
reaches the gate only through the real dispatch path, with the tool's real
schema and the tool's real parameter names.

(That sibling guard scans test sources as TEXT, so spelling the old pattern
out literally here would register this file as an offender on the strength of
its own prose. Described rather than quoted, deliberately.)
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from prometheus.computer.driver import FixtureDriver
from prometheus.computer.tools import TOOL_CLASSES, build_computer_tools
from prometheus.computer.types import Element, Observation
from prometheus.engine.agent_loop import LoopContext, _execute_tool_call
from prometheus.permissions.checker import PermissionMode, SecurityGate
from prometheus.tools.base import ToolRegistry

SNAP = "snap-1"


def _observation() -> Observation:
    return Observation(
        target="box", app="scratchapp", pid=4242, window_id=7, snapshot_id=SNAP,
        elements=(
            Element(0, "tok-send", "push button", "Send"),
            Element(1, "tok-field", "text", "Search", editable=True),
        ),
    )


def _ctx(prompted: list, *, approve: bool = True, mode=PermissionMode.DEFAULT):
    """A loop context wired to a REAL gate and the REAL wrapped tools.

    The prompt answers YES by default: if a control ever fails open, the
    action lands and the assertion on ``driver.dispatched`` fails loudly,
    rather than the test passing on a refusal that came from elsewhere.
    """
    driver = FixtureDriver([_observation(), _observation()])
    registry = ToolRegistry()
    for tool in build_computer_tools(driver):
        registry.register(tool)
    gate = SecurityGate(mode=mode, audit_logger=None)

    async def prompt(tool_name, reason, arguments=None):
        prompted.append({"tool": tool_name, "reason": reason, "args": arguments})
        return approve

    ctx = LoopContext(
        provider=None, model="t", system_prompt="", max_tokens=512,
        tool_registry=registry, permission_checker=gate,
        permission_prompt=prompt,
    )
    return ctx, driver, gate


def _call(ctx, tool: str, args: dict):
    return asyncio.run(_execute_tool_call(ctx, tool, "t1", args))


def _click_args(**over):
    args = {
        "target": "box", "app": "scratchapp", "pid": 4242, "window_id": 7,
        "snapshot_id": SNAP, "element_token": "tok-send",
        "delivery_mode": "background",
    }
    args.update(over)
    return args


# ── THE REGRESSION: a click must reach a human ──────────────────────────────

@pytest.mark.parametrize("tool,args", [
    ("computer_click", _click_args()),
    ("computer_scroll", {**_click_args(), "direction": "down", "amount": 3}),
    ("computer_press_key", {**_click_args(), "key": "return"}),
    ("computer_type_text", {**_click_args(), "text": "hello"}),
])
def test_a_desktop_action_is_never_auto_allowed(tool, args):
    """The measured defect, pinned. Each of these ALLOWed before the rule."""
    args.pop("element_token", None)
    if tool in ("computer_click", "computer_type_text"):
        args["element_token"] = "tok-send"
    prompted: list = []
    ctx, driver, _ = _ctx(prompted)
    _call(ctx, tool, args)
    assert prompted, (
        f"{tool} was NOT put in front of a human — the gate auto-allowed a "
        f"desktop action. This is the exact regression the computer rule in "
        f"checker.evaluate exists to prevent."
    )


def test_a_refused_action_never_reaches_the_driver():
    """Refusal must stop the ACTION, not merely return an error string."""
    prompted: list = []
    ctx, driver, _ = _ctx(prompted, approve=False)
    result = _call(ctx, "computer_click", _click_args())
    assert result.is_error, "a declined approval did not produce an error"
    assert driver.dispatched == [], (
        f"the action reached the driver despite being declined: "
        f"{driver.dispatched}"
    )


def test_an_approved_action_does_reach_the_driver():
    """The control is not merely 'refuse everything'. Effect, not prose."""
    prompted: list = []
    ctx, driver, _ = _ctx(prompted, approve=True)
    _call(ctx, "computer_click", _click_args())
    assert len(driver.dispatched) == 1, (
        f"an approved click did not reach the driver: {driver.dispatched}"
    )
    verb, args = driver.dispatched[0]
    assert verb == "click"
    assert args["element_token"] == "tok-send"


# ── THE ARGUMENTS ARE PART OF THE ASK ───────────────────────────────────────

def test_the_operator_is_shown_what_would_be_typed():
    """For type_text the arguments ARE the decision.

    An approval prompt naming only the tool is not consent to anything in
    particular — the whole reason 'arguments in the prompt' was promoted from
    a backlog item to a prerequisite for this work.
    """
    prompted: list = []
    ctx, _, _ = _ctx(prompted)
    _call(ctx, "computer_type_text",
          {**_click_args(), "element_token": "tok-field", "text": "transfer 500"})
    assert prompted, "no approval was raised at all"
    args = prompted[0]["args"]
    assert args is not None, (
        "the approval prompt received no arguments — the operator would be "
        "asked to approve a keystroke without seeing it"
    )
    assert "transfer 500" in str(args), (
        f"the text being typed was not in the approval payload: {args}"
    )


def test_the_reason_names_the_app_and_the_verb():
    prompted: list = []
    ctx, _, _ = _ctx(prompted)
    _call(ctx, "computer_click", _click_args())
    reason = prompted[0]["reason"]
    assert "scratchapp" in reason, f"the app is not in the reason: {reason!r}"
    assert "click" in reason.lower(), f"the verb is not in the reason: {reason!r}"
    assert "background" in reason.lower(), (
        f"the delivery mode is not in the reason: {reason!r} — background and "
        f"foreground are different grants and the prompt must say which"
    )


# ── A REMEMBERED GRANT SILENCES THE PROMPT, AND ONLY THE RIGHT ONE ──────────

def test_a_matching_grant_silences_the_prompt():
    from prometheus.permissions.checker import Grant
    from prometheus.permissions.computer_extent import COMPUTER_ACTION_KIND

    prompted: list = []
    ctx, driver, gate = _ctx(prompted)
    gate.add_grant(Grant(
        kind=COMPUTER_ACTION_KIND, value="box:scratchapp:click:background",
        tool_name="computer_click",
    ))
    _call(ctx, "computer_click", _click_args())
    assert not prompted, (
        "a matching computer_action grant did not silence the prompt"
    )
    assert len(driver.dispatched) == 1, "the granted action did not execute"


def test_a_grant_for_another_app_does_not_carry_over():
    from prometheus.permissions.checker import Grant
    from prometheus.permissions.computer_extent import COMPUTER_ACTION_KIND

    prompted: list = []
    ctx, _, gate = _ctx(prompted)
    gate.add_grant(Grant(
        kind=COMPUTER_ACTION_KIND, value="box:mail:click:background",
        tool_name="computer_click",
    ))
    _call(ctx, "computer_click", _click_args())
    assert prompted, (
        "a grant for a DIFFERENT app silenced this one — 'allow clicks in "
        "Mail' must not mean 'allow clicks in scratchapp'. That distinction "
        "is the reason the app term is in the extent at all."
    )


def test_a_background_grant_does_not_cover_foreground():
    from prometheus.permissions.checker import Grant
    from prometheus.permissions.computer_extent import COMPUTER_ACTION_KIND

    prompted: list = []
    ctx, _, gate = _ctx(prompted)
    gate.add_grant(Grant(
        kind=COMPUTER_ACTION_KIND, value="box:scratchapp:click:background",
        tool_name="computer_click",
    ))
    _call(ctx, "computer_click", _click_args(delivery_mode="foreground"))
    assert prompted, (
        "a background grant covered a FOREGROUND action — the visible variant "
        "steals focus from whatever the operator is doing and is a separate "
        "consent"
    )


# ── STRUCTURAL: the declarations cannot be dropped ──────────────────────────

def test_every_wrapped_tool_declares_a_computer_verb():
    """Without the verb, the tool routes to evaluate's auto-allow tail.

    This is the structural half of the regression above: a NEW computer tool
    added without the declaration would not be covered by any test that
    enumerates today's tools, so the requirement is asserted over the registry
    rather than over a list.
    """
    from prometheus.permissions.computer_schema import (
        declared_app_param, declared_computer_verb,
    )

    for tool in build_computer_tools(None):
        schema = tool.input_model.model_json_schema()
        verb = declared_computer_verb(schema)
        assert verb, (
            f"{tool.name} declares no x-prometheus-computer-verb — the gate "
            f"would treat it as an ordinary tool and auto-allow it"
        )
        assert declared_app_param(schema), (
            f"{tool.name} declares no app parameter — its extent could not be "
            f"assembled and it would prompt forever with nothing rememberable"
        )


def test_no_test_here_calls_evaluate_by_hand():
    """Guard the guard. See this module's docstring.

    The behavioural tests must reach the gate through ``_execute_tool_call``.
    A test that constructs the decision itself proves the gate's logic and
    nothing about whether the caller ever supplies the subject — which is
    precisely how the four-month path defect survived a green suite.
    """
    import ast

    tree = ast.parse(Path(__file__).read_text())
    # Parsed, not grepped. A substring check over the source trips on this
    # file's own prose about the defect — the guard would be measuring its
    # own docstring. The AST sees calls and nothing else.
    offenders = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "evaluate"
    ]
    assert not offenders, (
        f"a test in this file calls .evaluate() directly (line(s) "
        f"{offenders}); drive _execute_tool_call instead"
    )
    assert "_execute_tool_call(" in Path(__file__).read_text()


# ── THE TARGET TERM: A GRANT NEVER SPANS MACHINES ───────────────────────────

def test_a_grant_for_another_machine_does_not_carry_over():
    """Will's amendment, 2026-09-19, through the real dispatch path.

    "Allow clicks in Mail on this box" and "allow clicks in Mail on my laptop"
    are different grants and an operator would give exactly one. Same app,
    same verb, same delivery — only the machine differs.
    """
    from prometheus.permissions.checker import Grant
    from prometheus.permissions.computer_extent import COMPUTER_ACTION_KIND

    prompted: list = []
    ctx, driver, gate = _ctx(prompted)
    gate.add_grant(Grant(
        kind=COMPUTER_ACTION_KIND, value="laptop:scratchapp:click:background",
        tool_name="computer_click",
    ))
    _call(ctx, "computer_click", _click_args())   # target="box"
    assert prompted, (
        "a grant for ANOTHER MACHINE silenced this one — a grant that spans "
        "machines is the widening shape the target term exists to prevent"
    )


def test_a_pre_target_grant_matches_nothing():
    """A three-term value cannot be rescued by padding; it must not match.

    Nothing should ever construct one — from_config_dict refuses it on load.
    This pins the in-memory half: even if one is injected directly, the
    machine it was granted on is unknowable and it authorises nothing.
    """
    from prometheus.permissions.checker import Grant
    from prometheus.permissions.computer_extent import COMPUTER_ACTION_KIND

    prompted: list = []
    ctx, _, gate = _ctx(prompted)
    gate.add_grant(Grant(
        kind=COMPUTER_ACTION_KIND, value="scratchapp:click:background",
        tool_name="computer_click",
    ))
    _call(ctx, "computer_click", _click_args())
    assert prompted, (
        "a pre-target grant authorised a call — its machine is unknowable, so "
        "it must authorise nothing rather than everything"
    )


def test_the_reason_names_the_machine():
    prompted: list = []
    ctx, _, _ = _ctx(prompted)
    _call(ctx, "computer_click", _click_args())
    reason = prompted[0]["reason"]
    assert "box" in reason, (
        f"the target machine is not in the approval reason: {reason!r} — the "
        f"operator cannot tell which machine they are approving an action on"
    )


def test_an_action_naming_no_machine_is_refused_not_defaulted():
    """An empty target must be UNKNOWN, never the empty string.

    A blank term still renders four segments (':scratchapp:click:background')
    and would match any other call that also failed to name a machine — a
    cross-machine grant by accident, which is the thing the term prevents.
    """
    prompted: list = []
    ctx, driver, _ = _ctx(prompted, approve=False)
    result = _call(ctx, "computer_click", _click_args(target=""))
    assert result.is_error, "an action with no target machine was not refused"
    assert driver.dispatched == []
