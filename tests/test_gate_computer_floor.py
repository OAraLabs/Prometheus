"""`/gate off` does not waive desktop consent — and nothing else changes.

THE DEFECT (computer-use v1.1, D1 — verified by Will at checker.py:1026-1037)
------------------------------------------------------------------------------
In ``PermissionMode.AUTONOMOUS`` ``SecurityGate.evaluate`` returned ALLOW
before it reached the computer rule. Measured on origin/main 856ebb8: a known
extent and an unknown one both came back ``allowed=True,
requires_confirmation=False`` under ``/gate off``, where DEFAULT prompts for
both. ``ComputerUseLoop.step`` had no override, so a door built on today's
loop would click — and TYPE — unprompted, and the payload rule ("typed text is
never remembered") would not even be consulted. Fixing it is a hard
precondition for the door.

THE RULE NOW
------------
A computer action's consent is not a mode. In every mode it is decided the
same way: a stored ``computer_action`` grant matching the extent EXACTLY
allows it; otherwise it prompts. What ``/gate off`` waives — the approval
tiers for ordinary tools — it still waives, and the first test below proves
that byte for byte against a golden table captured on origin/main.

AND D18
-------
A legacy ``tool`` grant naming a computer tool (``security.grants`` still
loads ``kind: tool`` rows) matched EVERY call of that tool — any app, and
even an extent the gate could not assemble — because grants are checked
before the computer rule. Only an exact ``computer_action`` grant may silence
a computer action now.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
from pathlib import Path

import pytest

from prometheus.computer.actions import schema_for
from prometheus.computer.candidates import action_arguments, build_candidates
from prometheus.computer.chooser import RuleChooser, ScriptedChooser
from prometheus.computer.driver import FixtureDriver
from prometheus.computer.loop import ComputerUseLoop
from prometheus.computer.types import Element, Observation
from prometheus.permissions.checker import (
    Grant,
    PermissionMode,
    SecurityGate,
    TrustLevel,
)
from prometheus.permissions.computer_extent import (
    COMPUTER_ACTION_KIND,
    computer_extent_for,
)
from tests.support.gate_matrix import matrix

GOLDEN = Path(__file__).parent / "fixtures" / "gate_noncomputer_golden.json"

MODES = (PermissionMode.DEFAULT, PermissionMode.STRICT,
         PermissionMode.AUTONOMOUS)
ORIGINS = ("user", "system")


# ── 1. NOTHING ELSE CHANGES ─────────────────────────────────────────────────

def test_every_non_computer_decision_is_exactly_what_main_gave():
    """840 decisions — 3 modes x 2 origins x 7 grant sets x 20 calls — each
    compared field by field (allowed, requires_confirmation, action, trust
    level, reason) with the table captured on origin/main 856ebb8 BEFORE the
    floor existed. Not an expectation table: both sides are live runs of the
    real gate (``tests/support/gate_matrix.py``)."""
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    now = json.loads(json.dumps(matrix()))
    assert len(now) == len(golden) == 840
    changed = [(g, n) for g, n in zip(golden, now) if g != n]
    assert not changed, f"{len(changed)} non-computer decisions changed: {changed[:3]}"


def test_autonomous_still_waives_the_ordinary_approval_tiers():
    """The floor is narrow: `/gate off` keeps meaning what it meant for every
    tool that is not a desktop action."""
    gate = SecurityGate(mode=PermissionMode.AUTONOMOUS, audit_logger=None,
                        workspace_root="/opt/prom-gate-matrix/ws")
    for tool, kw in (("write_file", {"file_path": "/opt/elsewhere/x",
                                     "path_is_write": True}),
                     ("bash", {"command": "git push origin main"}),
                     ("mcp__fs__write_file", {})):
        d = gate.evaluate(tool, origin="system", **kw)
        assert d.allowed and d.trust_level == TrustLevel.AUTONOMOUS, tool


# ── 2. A COMPUTER ACTION'S CONSENT IS NOT A MODE ────────────────────────────

def _extent(verb: str, **over):
    # `site` is carried for the five-term extent (computer-use v1.1 PR 3b);
    # where the schema does not declare it yet, it is ignored. Grant values
    # below are DERIVED from the extent rather than written out, so these
    # tests hold whichever order the v1.1 PRs land in.
    args = {"target": "box", "app": "scratchapp", "pid": 1, "window_id": 2,
            "snapshot_id": "s1", "delivery_mode": "background",
            "element_token": "tok", "site": "-"}
    args.update(over)
    return computer_extent_for(f"computer_{verb}", args, schema=schema_for(verb))


def _click_grant() -> Grant:
    return Grant(kind=COMPUTER_ACTION_KIND, value=_extent("click")[0].value,
                 tool_name="computer_click")


def _evaluate(gate: SecurityGate, verb: str, origin: str = "user", **over):
    extent, unknown = _extent(verb, **over)
    return gate.evaluate(f"computer_{verb}", is_read_only=False, origin=origin,
                         computer_action=extent, computer_unknown=unknown)


@pytest.mark.parametrize("origin", ORIGINS)
@pytest.mark.parametrize("verb,over", [
    ("click", {}),
    ("press_key", {"key": "return"}),
    ("type_text", {"text": "hello"}),
    ("invoke_menu", {"path": ["File", "Save"]}),
])
def test_autonomous_prompts_for_a_known_computer_extent(verb, over, origin):
    """THE DEFECT. Each of these was ALLOWed under `/gate off`."""
    gate = SecurityGate(mode=PermissionMode.AUTONOMOUS, audit_logger=None)
    d = _evaluate(gate, verb, origin, **over)
    assert not d.allowed
    assert d.requires_confirmation
    assert "acts on the desktop" in d.reason


@pytest.mark.parametrize("origin", ORIGINS)
def test_autonomous_prompts_for_an_unknown_computer_extent(origin):
    gate = SecurityGate(mode=PermissionMode.AUTONOMOUS, audit_logger=None)
    d = _evaluate(gate, "click", origin, app="")
    assert not d.allowed and d.requires_confirmation
    assert "did not name a target application" in d.reason


def test_a_typed_payload_is_never_rememberable_under_autonomous():
    gate = SecurityGate(mode=PermissionMode.AUTONOMOUS, audit_logger=None)
    d = _evaluate(gate, "type_text", text="hello")
    assert "no lasting grant is offered" in d.reason


@pytest.mark.parametrize("origin", ORIGINS)
@pytest.mark.parametrize("verb,over", [
    ("click", {}), ("type_text", {"text": "x"}), ("click", {"app": ""}),
])
def test_computer_decisions_are_identical_in_every_mode(verb, over, origin):
    """'The floor is not a mode' (checker.py's own words), for desktop
    consent: DEFAULT, STRICT and AUTONOMOUS decide a computer action alike."""
    seen = {
        mode: _evaluate(SecurityGate(mode=mode, audit_logger=None),
                        verb, origin, **over)
        for mode in MODES
    }
    shapes = {(d.allowed, d.requires_confirmation, d.action, d.reason)
              for d in seen.values()}
    assert len(shapes) == 1, seen


def test_a_stored_computer_grant_still_allows_under_autonomous():
    """`/gate off` neither waives NOR tightens: a remembered exact grant
    allows, as it does in DEFAULT — at the grant's level, not AUTONOMOUS."""
    grant = _click_grant()
    for mode in MODES:
        gate = SecurityGate(mode=mode, audit_logger=None, grants=[grant])
        d = _evaluate(gate, "click")
        assert d.allowed, mode
        assert d.trust_level == TrustLevel.AUTO, mode


def test_the_approve_target_is_recorded_so_approve_always_works():
    """The prompt under AUTONOMOUS carries the same structured target as in
    DEFAULT, so `/approve always` derives the same exact grant."""
    gate = SecurityGate(mode=PermissionMode.AUTONOMOUS, audit_logger=None)
    d = _evaluate(gate, "click")
    target = gate.approve_target_for(d.reason)
    assert target["computer_action"] is not None
    assert target["computer_action"].value == _extent("click")[0].value


# ── 3. D18: A LEGACY `tool` GRANT DOES NOT COVER A DESKTOP ACTION ───────────

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("over", [{}, {"app": ""}, {"app": "bank"}])
def test_a_tool_grant_naming_a_computer_tool_allows_nothing(mode, over):
    """It matched any app, and even an extent that could not be assembled."""
    grant = Grant(kind="tool", value="", tool_name="computer_click")
    gate = SecurityGate(mode=mode, audit_logger=None, grants=[grant])
    d = _evaluate(gate, "click", **over)
    assert not d.allowed
    assert d.requires_confirmation


# ── 4. THE LOOP: AUTONOMOUS REACHES THE APPROVER, OR REFUSES ────────────────

#: Completeness evidence, passed where ``Observation`` has the fields (PR 1
#: and PR 3b add them): with it, the site term is "no web content" and a
#: stored grant can match; without it the site would be UNKNOWN and no grant
#: could. Filtered, so this file runs unchanged on either side of those PRs.
_EVIDENCE = {"total_element_count": 1, "returned_element_count": 1,
             "web_content_seen": False}


def _obs(snapshot: str) -> Observation:
    have = {f.name for f in dataclasses.fields(Observation)}
    return Observation(
        target="box", app="scratchapp", pid=1, window_id=2,
        snapshot_id=snapshot,
        elements=(Element(0, f"tok-send-{snapshot}", "push button", "Send"),),
        **{k: v for k, v in _EVIDENCE.items() if k in have},
    )


def _loop_grant() -> Grant:
    """A stored grant for exactly the extent the loop will compute."""
    cand = build_candidates(_obs("s1"))[0]
    extent, _ = computer_extent_for(cand.tool_name, action_arguments(cand),
                                    schema=schema_for("click"))
    assert extent is not None and extent.rememberable, extent
    return Grant(kind=COMPUTER_ACTION_KIND, value=extent.value,
                 tool_name="computer_click")


@pytest.fixture(autouse=True)
def _a_platform_that_can_flag_web_content(monkeypatch):
    """The site evidence rule (PR 3b) only ever yields "no web content" on a
    platform whose accessibility path can flag web content. Pinned so these
    tests mean the same on CI's macOS leg."""
    monkeypatch.setattr(sys, "platform", "linux")


def _loop(mode, *, approve=True, with_approver=True, grants=(),
          before_act=None):
    prompted: list = []
    driver = FixtureDriver([_obs("s1"), _obs("s2")])

    async def approver(tool_name, reason, arguments=None):
        prompted.append(reason)
        return approve

    kw = {}
    if before_act is not None:
        kw["before_act"] = before_act
    loop = ComputerUseLoop(
        driver=driver, chooser=RuleChooser(prefer=("send",)),
        gate=SecurityGate(mode=mode, audit_logger=None, grants=list(grants)),
        approve=approver if with_approver else None,
        skip_preconditions=True, **kw,
    )
    return loop, driver, prompted


def _step(loop):
    return asyncio.run(loop.step("press send", "box", "scratchapp", 1, 2))


def test_under_autonomous_the_loop_reaches_the_approver():
    loop, driver, prompted = _loop(PermissionMode.AUTONOMOUS)
    result = _step(loop)
    assert prompted, "a desktop action executed under /gate off unprompted"
    assert result.ok and driver.dispatched


def test_under_autonomous_a_declined_prompt_dispatches_nothing():
    loop, driver, prompted = _loop(PermissionMode.AUTONOMOUS, approve=False)
    result = _step(loop)
    assert prompted and result.status == "refused"
    assert driver.dispatched == []


def test_under_autonomous_with_no_approver_the_action_is_refused():
    """Nobody to ask and `/gate off` must not combine into "go ahead"."""
    loop, driver, _ = _loop(PermissionMode.AUTONOMOUS, with_approver=False)
    result = _step(loop)
    assert result.status == "refused"
    assert driver.dispatched == []


def test_under_autonomous_a_stored_grant_still_acts_without_a_prompt():
    loop, driver, prompted = _loop(PermissionMode.AUTONOMOUS,
                                   grants=[_loop_grant()])
    result = _step(loop)
    assert result.ok and driver.dispatched
    assert not prompted


# ── 5. THE before_act SEAM ──────────────────────────────────────────────────

@pytest.mark.parametrize("path", ["prompt", "grant"])
def test_before_act_runs_on_every_path_that_dispatches(path):
    """Per-action checks (stop, high-consequence labels, ceilings) must hold
    even when a stored grant allows and no approver is ever called."""
    calls: list = []

    def before_act(candidate, extent, decision):
        calls.append((candidate.candidate_id, extent.value, decision.allowed))
        return True

    grants = [_loop_grant()] if path == "grant" else []
    loop, driver, _ = _loop(PermissionMode.DEFAULT, grants=grants,
                            before_act=before_act)
    result = _step(loop)
    assert result.ok and driver.dispatched
    assert calls == [("click-0", _loop_grant().value, path == "grant")]


def test_before_act_saying_no_dispatches_nothing():
    loop, driver, _ = _loop(
        PermissionMode.DEFAULT, before_act=lambda c, e, d: False)
    result = _step(loop)
    assert result.status == "refused"
    assert "before_act" in result.reason
    assert driver.dispatched == []


def test_before_act_is_not_consulted_when_nothing_would_dispatch():
    calls: list = []
    loop, driver, _ = _loop(
        PermissionMode.DEFAULT, approve=False,
        before_act=lambda c, e, d: calls.append(1) or True)
    assert _step(loop).status == "refused"
    assert calls == []


def test_a_before_act_that_raises_dispatches_nothing():
    """A broken check is a refusal, never a pass."""
    def boom(candidate, extent, decision):
        raise RuntimeError("check broke")

    loop, driver, _ = _loop(PermissionMode.DEFAULT, before_act=boom)
    result = _step(loop)
    assert result.status == "refused"
    assert driver.dispatched == []


def test_before_act_is_synchronous():
    """No await may sit between the check and the dispatch, so the seam is
    a plain callable — an async one is refused at construction."""
    async def async_check(candidate, extent, decision):
        return True

    with pytest.raises(TypeError, match="synchronous"):
        ComputerUseLoop(
            driver=FixtureDriver([_obs("s1")]), chooser=ScriptedChooser([]),
            gate=SecurityGate(audit_logger=None), before_act=async_check)
