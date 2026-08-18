"""THE INVARIANT: the SHIPPED_DENIED_PATHS floor holds above every grant kind.

WHY THIS FILE EXISTS, AND WHY IT IS WRITTEN FIRST
-------------------------------------------------
`derive_grant` can widen a single-file approval to a whole parent directory,
and its rule-4 fallback produces `kind="tool", value=""` — a grant whose
`matches()` never looks at a path at all. The only thing standing between that
and a real compromise is the ORDERING in `SecurityGate.evaluate`:
`_check_denied_path` returns DENY *before* grants are consulted, so — in the
words of the comment that states the design — *"a grant can silence a prompt,
never resurrect a block."*

That ordering was verified by outcome during the CONSENT Phase 0 survey and
had NO test. It held by accident: nothing would have caught a refactor that
moved the grant check above the floor check. This sprint refactors the grant
path, so the invariant is pinned FIRST and kept green throughout.

BOTH DIRECTIONS (§2c)
---------------------
A uniform-DENY regression — a bug that refuses everything — would satisfy any
suite that only checks that credential paths are refused, and would look like
the control working. So every case here asserts the ADMISSION direction too:
an ordinary file must flip `APPROVE → ALLOW` when a grant covers it. If the
floor column is all DENY *and* the ordinary column is all ALLOW, the matrix is
measuring something. If the ordinary column goes DENY, the suite is blind and
says so.
"""

from __future__ import annotations

import pytest

from prometheus.config.shipped_defaults import SHIPPED_DENIED_PATHS
from prometheus.permissions.checker import Grant, SecurityGate

# Paths on the shipped floor. Each must be refused no matter what is granted.
FLOOR_TARGETS = [
    pytest.param("/home/will/.ssh/authorized_keys", id="ssh"),
    pytest.param("/home/will/.gnupg/secring.gpg", id="gnupg"),
    pytest.param("/home/will/.config/prometheus/env", id="config-env"),
    pytest.param("/root/.ssh/id_rsa", id="ssh-other-home"),
    pytest.param("/etc/shadow", id="etc"),
]

# Not on the floor. Used as the admission control in every case.
ORDINARY = "/home/will/ordinary-file.txt"


# THE THIRD AXIS (gap 2, 2026-08-18). The floor is not a mode: it must hold
# under `/gate off` exactly as it does under default. It did NOT — the
# AUTONOMOUS branch of `evaluate` returned ALLOW *before* `_check_denied_path`,
# so write_file/read_file on ~/.ssh were permitted while `/gate off`'s own reply
# text told the operator that denied paths were still enforced. Measured before
# the fix: default DENY, strict DENY, autonomous ALLOW.
#
# Adding mode here rather than writing a parallel check is deliberate — the
# existing source-order test pins grant-vs-floor ordering, which cannot see a
# branch that returns above BOTH.
MODES = [
    pytest.param("default", id="mode-default"),
    pytest.param("strict", id="mode-strict"),
    pytest.param("autonomous", id="mode-autonomous-GATE-OFF"),
]


def _gate(grants: list[Grant] | None = None, mode: str = "default") -> SecurityGate:
    """A gate carrying the SHIPPED floor, with a workspace root set.

    ``workspace_root`` is NOT decoration here. Without it there is no
    APPROVE tier at all — every non-floor path returns ALLOW and the
    admission direction below silently stops discriminating. The first
    draft of this file omitted it and two admission cases failed, which
    is the fixture being unfaithful to production rather than the
    assertions being wrong: the live gate always carries one.
    """
    return SecurityGate(
        denied_paths=list(SHIPPED_DENIED_PATHS),
        workspace_root=["/srv/workspace-for-this-test"],
        grants=grants or [],
        mode=mode,
    )


def _grant(kind: str, value: str, tool: str = "write_file") -> Grant:
    """Build a Grant without depending on this sprint's constructor changes.

    Positional-free so added fields (grant_id, created_at, request_id) cannot
    break this file — the invariant must survive the record change.
    """
    return Grant(kind=kind, value=value, tool_name=tool)


# The four kinds, widest-first where a widest exists.
GRANT_CASES = [
    pytest.param(None, id="no-grants-baseline"),
    pytest.param(("path_prefix", "/home/will"), id="path_prefix-home"),
    pytest.param(("path_prefix", "/"), id="path_prefix-ROOT-widest-path"),
    pytest.param(("tool", ""), id="tool-empty-WIDEST-OF-ALL"),
    pytest.param(("path_prefix", "/home/will/.ssh"), id="path_prefix-AIMED-AT-FLOOR"),
]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("grant_spec", GRANT_CASES)
@pytest.mark.parametrize("target", FLOOR_TARGETS)
def test_floor_holds_above_every_grant_kind(grant_spec, target, mode):
    """BREACH direction: no grant of any kind, IN ANY MODE, reaches the floor.

    Includes a grant aimed DIRECTLY at the floor (`path_prefix /home/will/.ssh`)
    — the case an attacker or a careless `/approve always` would produce — and
    now every case again under `/gate off`."""
    grants = [_grant(*grant_spec)] if grant_spec else []
    decision = _gate(grants, mode=mode).evaluate("write_file", file_path=target)

    assert decision.action == "DENY", (
        f"FLOOR BREACHED in mode={mode!r}: grant={grant_spec!r} reached "
        f"{target!r} (got {decision.action}). Either the grant check moved "
        f"above _check_denied_path, or a mode branch returned before the "
        f"floor was consulted."
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("target", FLOOR_TARGETS)
@pytest.mark.parametrize("tool", ["write_file", "edit_file", "read_file"])
def test_floor_holds_for_every_path_declaring_tool(tool, target, mode):
    """The floor is a property of the PATH, not of the tool that names it.

    read_file is the case the arc missed: it is not in _APPROVE_TOOLS, so it
    never reaches the approval tier and its only floor is _check_denied_path.
    """
    decision = _gate(mode=mode).evaluate(
        tool, file_path=target, is_read_only=(tool == "read_file"))
    assert decision.action == "DENY", (
        f"FLOOR BREACHED in mode={mode!r}: {tool} reached {target!r} "
        f"(got {decision.action})"
    )


def test_bash_is_NOT_covered_by_this_invariant():
    """Stated so the suite cannot be read as claiming more than it proves.

    `gate.evaluate("bash", ...)` is handed a command STRING and no file_path,
    so _check_denied_path is never reached — in ANY mode, at EITHER origin.
    That floor is enforced below the tool layer by the kernel profile
    (security.bash_confinement / deploy/apparmor/prometheus-bash), verified
    separately by outcome. If this assertion ever flips, the gate grew a path
    view for bash and this file should grow bash cases to match.
    """
    for mode in ("default", "strict", "autonomous"):
        decision = _gate(mode=mode).evaluate(
            "bash", command=f"cat {FLOOR_TARGETS[0].values[0]}", origin="user")
        assert decision.action == "ALLOW", (
            f"bash in mode={mode!r} now returns {decision.action} for a floor "
            "path — the gate gained a path view; add bash to the matrix above"
        )


@pytest.mark.parametrize("grant_spec", GRANT_CASES)
def test_admission_direction_still_works(grant_spec):
    """ADMISSION direction: without this, a uniform-DENY bug passes as success.

    No grant  -> APPROVE (a prompt is raised).
    Any grant covering the target -> ALLOW (the prompt is silenced).
    A grant NOT covering the target -> APPROVE (still prompts)."""
    grants = [_grant(*grant_spec)] if grant_spec else []
    decision = _gate(grants).evaluate("write_file", file_path=ORDINARY)

    if grant_spec is None:
        assert decision.action == "APPROVE", (
            "an out-of-workspace write with no grant must raise a prompt; "
            f"got {decision.action}"
        )
        return

    kind, value = grant_spec
    covers = kind == "tool" or ORDINARY.startswith(value.rstrip("/") + "/") or value == "/"
    expected = "ALLOW" if covers else "APPROVE"
    assert decision.action == expected, (
        f"grant={grant_spec!r} on an ORDINARY path: expected {expected}, "
        f"got {decision.action}. If this is DENY, the suite has become "
        f"blind — a control that refuses everything looks identical to one "
        f"that works."
    )


def test_the_matrix_is_not_uniform():
    """Guard identity: prove the two directions actually differ.

    If every cell in the matrix returned the same verdict, both tests above
    would pass while measuring nothing. This asserts the matrix discriminates:
    the same grant that is REFUSED on a floor path is ALLOWED on an ordinary
    one."""
    g = [_grant("path_prefix", "/home/will")]
    floor = _gate(g).evaluate("write_file", file_path=FLOOR_TARGETS[0].values[0])
    ordinary = _gate(g).evaluate("write_file", file_path=ORDINARY)

    assert floor.action == "DENY"
    assert ordinary.action == "ALLOW"
    assert floor.action != ordinary.action, (
        "one grant produced the same verdict on a floor path and an ordinary "
        "path — the matrix cannot distinguish containment from a blanket refusal"
    )


def test_denied_path_check_precedes_grant_check_in_source():
    """Total invariant, cheaper than enumerating states.

    The behavioural tests above prove the ordering for the cases they cover.
    This pins the ordering itself, so a refactor that reorders the blocks is
    caught even for a grant kind nobody thought to parametrise."""
    import inspect

    src = inspect.getsource(SecurityGate.evaluate)
    floor_at = src.find("_check_denied_path")
    grants_at = src.find("self._grants")

    assert floor_at != -1, "_check_denied_path call vanished from evaluate()"
    assert grants_at != -1, "grant check vanished from evaluate()"
    assert floor_at < grants_at, (
        "THE GRANT CHECK NOW PRECEDES THE FLOOR CHECK in evaluate(). "
        "A grant can now resurrect a blocked call. This is the single "
        "invariant SPRINT-CONSENT exists to preserve."
    )
