"""The ``computer`` block on /api/status — the first surface this answer has.

Until this block, ``check_preconditions``' result reached NOWHERE: not a log
line, not a metric, not an endpoint. It appeared only as a blocked
``StepResult.reason``, and only when a step was actually attempted. So "can
computer use act right now?" could only be answered by trying it — the worst
way to learn the answer is no, because the failing case is the one that looks
like success.

THE THREE PROPERTIES THESE TESTS HOLD
--------------------------------------
1. **Both halves independently.** Input is XTEST over X11; observation is
   AT-SPI over D-Bus. They fail on different axes, the session bus outlives a
   dead graphical session, and the mixed state is reachable. A single boolean
   cannot say "input would dispatch and observation would return an empty
   tree" — which is exactly the state that reports success having done
   nothing.
2. **``unknown`` is a third answer and never reads as healthy.** Same ruling
   the tracking-ref axis applies one block over.
3. **It connects.** ``DISPLAY`` being set is not evidence of a display: on the
   reference deployment it is set, inherited from a lingering ``systemd
   --user`` manager, and may point at a session that ended.
"""

from __future__ import annotations

import inspect

import pytest

from prometheus.computer import status as cstatus
from prometheus.computer.driver import (
    HALF_OK,
    HALF_UNAVAILABLE,
    HALF_UNKNOWN,
    STATE_ACT_ONLY,
    STATE_OBSERVE_ONLY,
    STATE_READY,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
    check_preconditions,
)

LIVE = {"DISPLAY": ":0", "XDG_RUNTIME_DIR": "/run/user/1000"}


def _sub(env):
    return cstatus.render(check_preconditions(env))


# ── 1. BOTH HALVES, INDEPENDENTLY ───────────────────────────────────────────

def test_a_dead_display_still_reports_the_observe_half():
    """THE REGRESSION. The old code returned as soon as the X half failed.

    A box with no display never learned whether its accessibility bus was up,
    which made two controls on two transports look like one control with one
    answer.
    """
    block = _sub({"DISPLAY": ":99", "XDG_RUNTIME_DIR": "/run/user/1000"})
    assert block["act"]["state"] == HALF_UNAVAILABLE
    assert block["observe"]["state"] != HALF_UNKNOWN, (
        "the observe half was not evaluated because the act half failed "
        "first — the halves are not independent"
    )


def test_no_display_at_all_still_reports_the_observe_half():
    block = _sub({"XDG_RUNTIME_DIR": "/run/user/1000"})
    assert block["act"]["state"] == HALF_UNAVAILABLE
    assert block["observe"]["state"] == HALF_OK


def test_the_mixed_state_has_its_own_name():
    """`act_only` is the field to alert on: input dispatches, tree is empty."""
    block = _sub({"DISPLAY": ":0", "XDG_RUNTIME_DIR": "/nonexistent-rt"})
    assert block["act"]["state"] == HALF_OK
    assert block["observe"]["state"] == HALF_UNAVAILABLE
    assert block["state"] == STATE_ACT_ONLY, (
        f"the dangerous mixed state rendered as {block['state']!r} — it must "
        f"be nameable, not hidden inside a generic failure"
    )


def test_the_other_mixed_state_is_distinguishable():
    block = _sub({"DISPLAY": ":99", "XDG_RUNTIME_DIR": "/run/user/1000"})
    assert block["state"] == STATE_OBSERVE_ONLY
    assert block["state"] != STATE_ACT_ONLY


def test_both_down_is_unavailable_not_a_mixed_state():
    block = _sub({"XDG_RUNTIME_DIR": "/nonexistent-rt"})
    assert block["state"] == STATE_UNAVAILABLE


# ── 2. UNKNOWN IS A THIRD ANSWER ────────────────────────────────────────────

def test_unknown_outranks_a_known_good_half():
    """An unknown half means the rollup cannot be trusted.

    Reporting `act_only` when the observe half merely could not be REACHED
    would assert something unestablished — the failure this vocabulary exists
    to prevent, one level up.
    """
    block = _sub({"DISPLAY": "remotehost:0", "XDG_RUNTIME_DIR": "/run/user/1000"})
    assert block["act"]["state"] == HALF_UNKNOWN
    assert block["observe"]["state"] == HALF_OK
    assert block["state"] == STATE_UNKNOWN, (
        "an unknown half collapsed into a confident rollup"
    )


def test_unknown_never_renders_as_the_healthy_value():
    for env in (
        {"DISPLAY": "remotehost:0", "XDG_RUNTIME_DIR": "/run/user/1000"},
        {"DISPLAY": ":0"},  # no XDG_RUNTIME_DIR -> bus not locatable
    ):
        block = _sub(env)
        assert block["state"] != STATE_READY, (
            f"{env} rendered as ready despite an unestablished half"
        )


def test_a_broken_probe_degrades_to_unknown_not_unavailable():
    """A probe that broke tells us nothing about the substrate.

    Claiming it is DOWN is an assertion we did not establish — the same
    discipline the halves themselves follow.
    """
    block = cstatus._unknown_substrate("probe failed: RuntimeError")
    assert block["state"] == STATE_UNKNOWN
    assert block["act"]["state"] == HALF_UNKNOWN
    assert block["observe"]["state"] == HALF_UNKNOWN


def test_substrate_block_never_raises(monkeypatch):
    def boom(_env=None):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(cstatus, "check_preconditions", boom)
    block = cstatus.substrate_block()
    assert block["state"] == STATE_UNKNOWN, (
        "a broken probe propagated instead of degrading — it would take "
        "/api/status down with it"
    )


# ── 3. IT CONNECTS ──────────────────────────────────────────────────────────

def test_display_set_but_dead_is_not_ok():
    """`DISPLAY` being SET is not evidence of a display. That is the mistake."""
    block = _sub({"DISPLAY": ":99", "XDG_RUNTIME_DIR": "/run/user/1000"})
    assert block["act"]["state"] != HALF_OK


def test_the_act_probe_actually_opens_a_socket():
    """Pinned structurally: a future 'optimisation' to read the variable
    instead would pass every behavioural test above on this box, because
    :0 happens to be live here."""
    from prometheus.computer import driver

    src = inspect.getsource(driver._check_act_half)
    assert "socket.socket" in src and "connect(" in src, (
        "the act probe no longer connects — it would report a stale "
        "inherited DISPLAY as healthy"
    )


# ── NO LOCATIONS ON THE WIRE ────────────────────────────────────────────────

@pytest.mark.parametrize("env", [
    LIVE,
    {"DISPLAY": ":99", "XDG_RUNTIME_DIR": "/run/user/1000"},
    {"DISPLAY": ":0", "XDG_RUNTIME_DIR": "/nonexistent-rt"},
    {"XDG_RUNTIME_DIR": "/run/user/1000"},
])
def test_the_block_leaks_no_path_or_display_number(env):
    """`component` NAMES the thing; it does not locate it.

    The locations involved are a uid-bearing runtime path and a display
    number, and /api/status's whole audience is someone already worried.
    """
    blob = str(_sub(env))
    for leak in ("/run/user/", "/tmp/.X11-unix", "at-spi/bus", ":99", ":0"):
        assert leak not in blob, (
            f"the status block leaked {leak!r}: {blob}"
        )


# ── THE SHAPE AGREES WITH ITS NEIGHBOUR ─────────────────────────────────────

def test_axes_are_strings_not_booleans():
    """Same convention as the `deployment` block: per-axis STRINGS plus a
    rollup, so an operator does not learn a second vocabulary halfway down
    one payload."""
    block = _sub(LIVE)
    for axis in ("act", "observe"):
        assert isinstance(block[axis]["state"], str)
        assert not isinstance(block[axis]["state"], bool)
    assert isinstance(block["state"], str)


def test_every_rollup_state_is_reachable():
    """A vocabulary with an unreachable member is a vocabulary nobody can
    trust the meaning of."""
    seen = {
        _sub(LIVE)["state"],
        _sub({"DISPLAY": ":0", "XDG_RUNTIME_DIR": "/nonexistent-rt"})["state"],
        _sub({"DISPLAY": ":99", "XDG_RUNTIME_DIR": "/run/user/1000"})["state"],
        _sub({"XDG_RUNTIME_DIR": "/nonexistent-rt"})["state"],
        _sub({"DISPLAY": "h:0", "XDG_RUNTIME_DIR": "/run/user/1000"})["state"],
    }
    assert seen == {STATE_READY, STATE_ACT_ONLY, STATE_OBSERVE_ONLY,
                    STATE_UNAVAILABLE, STATE_UNKNOWN}


# ── REGISTERED IS THE OTHER HALF OF THE HONEST ANSWER ───────────────────────

def test_no_registry_reports_none_not_zero():
    """`None` ("nothing to ask") and `0` ("asked; none") are different facts."""
    assert cstatus._registered_count(None) is None
    assert cstatus._targets(None) is None


def test_an_empty_registry_reports_zero():
    from prometheus.tools.base import ToolRegistry

    assert cstatus._registered_count(ToolRegistry()) == 0


def test_registered_counts_only_wrapped_desktop_tools():
    from prometheus.computer.tools import build_computer_tools
    from prometheus.tools.base import ToolRegistry

    reg = ToolRegistry()
    for t in build_computer_tools(None):
        reg.register(t)
    assert cstatus._registered_count(reg) == len(build_computer_tools(None))


def test_the_daemon_registers_none_today():
    """A healthy substrate must not read as "a model can click".

    `register_computer_tools` has no call site by design; this pins that the
    status block would say so rather than implying otherwise.
    """
    import prometheus.computer.tools as tools
    from pathlib import Path

    root = Path(tools.__file__).resolve().parents[1]
    hits = [
        f"{p}:{i}"
        for p in root.rglob("*.py")
        for i, line in enumerate(p.read_text().splitlines(), 1)
        if "register_computer_tools(" in line and "def " not in line
    ]
    assert not hits, (
        f"register_computer_tools now has call site(s) {hits} — a model can "
        f"click. That is a deliberate decision; update this test and say so."
    )


# ── TARGETS EXPOSE NAMES, NEVER CONNECTIONS ─────────────────────────────────

def test_targets_report_binding_without_exposing_connection_settings():
    from prometheus.computer.driver import FixtureDriver
    from prometheus.computer.targets import Target, TargetRegistry
    from prometheus.computer.types import Element, Observation

    reg = TargetRegistry()
    reg.declare(Target(name="box", kind="local"),
                FixtureDriver([Observation(
                    target="box", app="a", pid=1, window_id=2, snapshot_id="s",
                    elements=(Element(0, "t", "push button", "Go"),))]))
    reg.declare(Target(name="other", kind="remote",
                       connection={"endpoint": "CANARY-CONNECTION"}))

    out = cstatus._targets(reg)
    assert {t["name"] for t in out} == {"box", "other"}
    assert [t for t in out if t["name"] == "box"][0]["driver_bound"] is True
    assert [t for t in out if t["name"] == "other"][0]["driver_bound"] is False
    assert "CANARY-CONNECTION" not in str(out), (
        "a target's connection settings reached the status payload; they are "
        "opaque by construction and must not leave the process"
    )
