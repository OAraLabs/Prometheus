"""The candidate table is the boundary: nothing a chooser says becomes an action.

The client builds complete bounded actions; the chooser returns an ID; the
client validates that ID against the table it built. An unknown ID, a stale
ID, or a reserved one each resolve WITHOUT an action, and none of them is
coerced into the nearest plausible candidate.

These tests use ``ScriptedChooser`` precisely because it will happily return a
bad ID. A chooser that could not misbehave would prove nothing about the
validation path.
"""

from __future__ import annotations

import asyncio

import pytest

from prometheus.computer.candidates import (
    InvalidChoice, UnusableObservation, build_candidates,
    build_choice_request, validate_choice,
)
from prometheus.computer.chooser import RuleChooser, ScriptedChooser
from prometheus.computer.driver import FixtureDriver, StaleSnapshot
from prometheus.computer.loop import ComputerUseLoop
from prometheus.computer.types import (
    CANDIDATE_ABSTAIN, CANDIDATE_REOBSERVE, Element, Observation,
)
from prometheus.permissions.checker import PermissionMode, SecurityGate


def _obs(snapshot="s1") -> Observation:
    return Observation(
        target="box", app="scratchapp", pid=1, window_id=2,
        snapshot_id=snapshot,
        elements=(
            Element(0, f"tok-send-{snapshot}", "push button", "Send"),
            Element(1, f"tok-cancel-{snapshot}", "push button", "Cancel"),
            Element(2, f"tok-field-{snapshot}", "text", "Search", editable=True),
        ),
    )


# ── BUILDING ────────────────────────────────────────────────────────────────

def test_candidates_are_complete_bounded_actions():
    cands = build_candidates(_obs())
    assert cands
    for c in cands:
        assert c.arguments.get("app") == "scratchapp"
        assert c.arguments.get("snapshot_id") == "s1"
        assert c.arguments.get("delivery_mode") == "background", (
            "a candidate defaulted to something other than background — the "
            "safer delivery mode must be the one you get without asking"
        )


def test_the_chooser_never_sees_arguments_or_tool_names():
    """Cua states the same boundary for jev-use, and it is worth keeping even
    for a local chooser: the decision layer stays LESS privileged than the
    gate."""
    cands = build_candidates(_obs(), text_to_type="secret-value")
    request = build_choice_request("press send", _obs(), cands)
    blob = str(request.candidates)
    assert "tok-send" not in blob, f"an element token leaked to the chooser: {blob}"
    assert "computer_click" not in blob, f"a tool name leaked: {blob}"
    assert "secret-value" not in blob, f"a payload leaked to the chooser: {blob}"
    for entry in request.candidates:
        assert set(entry) == {"id", "description"}


def test_an_unusable_observation_refuses_rather_than_returning_nothing():
    """Empty-because-dead and empty-because-idle must not look the same."""
    dead = Observation(
        target="box", app="a", pid=1, window_id=2, snapshot_id="s",
        unusable_reason="no AT-SPI bus; the tree would be empty",
    )
    with pytest.raises(UnusableObservation):
        build_candidates(dead)


def test_duplicate_candidate_ids_are_refused_at_build_time():
    from prometheus.computer.candidates import _assert_unique_ids
    from prometheus.computer.types import Candidate

    dupe = [
        Candidate("click-1", "computer_click", {}, "a", "s"),
        Candidate("click-1", "computer_click", {}, "b", "s"),
    ]
    with pytest.raises(ValueError):
        _assert_unique_ids(dupe)


def test_a_candidate_may_not_take_a_reserved_id():
    from prometheus.computer.candidates import _assert_unique_ids
    from prometheus.computer.types import Candidate

    with pytest.raises(ValueError):
        _assert_unique_ids([
            Candidate(CANDIDATE_ABSTAIN, "computer_click", {}, "a", "s"),
        ])


# ── VALIDATING ──────────────────────────────────────────────────────────────

def test_an_unknown_id_is_refused_not_coerced():
    obs = _obs()
    cands = build_candidates(obs)
    with pytest.raises(InvalidChoice) as exc:
        validate_choice("click-9999", cands, obs)
    assert "not in the" in str(exc.value)


def test_a_stale_candidate_is_refused():
    """The snapshot invariant, enforced before the gate and before the driver."""
    old = _obs("s1")
    cands = build_candidates(old)
    fresh = _obs("s2")
    with pytest.raises(InvalidChoice) as exc:
        validate_choice(cands[0].candidate_id, cands, fresh)
    assert "snapshot" in str(exc.value)


@pytest.mark.parametrize("reserved", [CANDIDATE_ABSTAIN, CANDIDATE_REOBSERVE])
def test_reserved_ids_resolve_to_no_action(reserved):
    obs = _obs()
    assert validate_choice(reserved, build_candidates(obs), obs) is None


def test_an_empty_answer_is_refused():
    obs = _obs()
    with pytest.raises(InvalidChoice):
        validate_choice("", build_candidates(obs), obs)


# ── THE FULL PATH ───────────────────────────────────────────────────────────

def _loop(chooser, *, approve=True, driver=None):
    prompted: list = []
    driver = driver or FixtureDriver([_obs("s1"), _obs("s2")])

    async def approver(tool_name, reason, arguments=None):
        prompted.append({"tool": tool_name, "reason": reason, "args": arguments})
        return approve

    loop = ComputerUseLoop(
        driver=driver, chooser=chooser,
        gate=SecurityGate(mode=PermissionMode.DEFAULT, audit_logger=None),
        approve=approver, skip_preconditions=True,
    )
    return loop, driver, prompted


def _run(loop, **kw):
    return asyncio.run(loop.step(
        goal=kw.pop("goal", "press send"), target="box", app="scratchapp",
        pid=1, window_id=2, **kw,
    ))


def test_one_action_goes_through_the_whole_path():
    """observe -> build -> select -> validate -> GATE -> execute -> verify."""
    loop, driver, prompted = _loop(RuleChooser(prefer=("send",)))
    result = _run(loop)
    assert result.ok, f"the step did not execute: {result.status} {result.reason}"
    assert result.candidate is not None
    assert "Send" in result.candidate.description
    assert prompted, "the action executed without ever reaching the gate"
    assert result.extent == "box:scratchapp:click:background"
    assert driver.dispatched, "nothing reached the driver"
    verb, args = driver.dispatched[0]
    assert verb == "click"
    assert args["element_token"] == "tok-send-s1", (
        "the executed action is not the one that was validated and gated"
    )


def test_a_declined_approval_stops_the_action():
    loop, driver, _ = _loop(RuleChooser(prefer=("send",)), approve=False)
    result = _run(loop)
    assert result.status == "refused"
    assert driver.dispatched == [], (
        f"a declined action still reached the driver: {driver.dispatched}"
    )


def test_an_invented_id_stops_the_step_and_executes_nothing():
    loop, driver, prompted = _loop(ScriptedChooser(["click-does-not-exist"]))
    result = _run(loop)
    assert result.status == "refused"
    assert driver.dispatched == []
    assert not prompted, (
        "an invalid choice reached the gate — validation must fail closed "
        "BEFORE the gate, not rely on it"
    )


def test_abstain_executes_nothing_and_is_not_an_error():
    loop, driver, _ = _loop(ScriptedChooser([CANDIDATE_ABSTAIN]))
    result = _run(loop)
    assert result.status == "abstained"
    assert driver.dispatched == []


def test_reobserve_is_distinguishable_from_abstain():
    loop, _, _ = _loop(ScriptedChooser([CANDIDATE_REOBSERVE]))
    assert _run(loop).status == "reobserve"


def test_a_chooser_that_matches_nothing_abstains_rather_than_guessing():
    loop, driver, _ = _loop(RuleChooser(prefer=("nonexistent-thing",)))
    result = _run(loop, goal="zzz")
    assert result.status == "abstained", (
        f"the chooser guessed instead of abstaining: {result.status}"
    )
    assert driver.dispatched == []


def test_the_driver_refuses_a_stale_snapshot_even_if_everything_else_passes():
    """The third independent refusal. Belt, braces, and the driver's own rule."""
    driver = FixtureDriver([_obs("s1")])
    driver.observe("box", "scratchapp", 1, 2)
    with pytest.raises(StaleSnapshot):
        driver.act("click", {"snapshot_id": "s-old", "element_token": "t"})


# ── THE SUBSTRATE CHECK ─────────────────────────────────────────────────────
#
# ⚠ THESE WERE HOST-DEPENDENT AND PASSED BY ACCIDENT. The first version of
# `test_a_missing_display_blocks_the_step` called the real
# `check_preconditions()` against the REAL environment and asserted "blocked".
# It passed in CI and in a plain shell — because neither has a DISPLAY — and
# FAILED the moment the suite ran from a graphical terminal.
#
# That is worse than flaky. This is the test for the cold-path refusal, the
# single most important safety property of the module, and it was proving it
# only because the machine happened to have no display. Had the refusal broken
# entirely, every headless runner — which is every CI runner — would still
# have gone green.
#
# So the environment is now BUILT, never read: `DISPLAY` is removed from the
# process env and `X11_SOCKET_DIR` points at an empty tmp dir, so the answer
# is the same on a graphical workstation and a headless runner.


@pytest.fixture
def no_display(tmp_path, monkeypatch):
    """A process that genuinely has no reachable display, on any host."""
    from prometheus.computer import driver as _driver

    empty = tmp_path / "X11-unix"
    empty.mkdir()
    monkeypatch.setattr(_driver, "X11_SOCKET_DIR", str(empty))
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "rt"))
    (tmp_path / "rt" / "at-spi").mkdir(parents=True)
    (tmp_path / "rt" / "at-spi" / "bus").write_text("")
    return tmp_path


def test_a_missing_display_blocks_the_step_rather_than_doing_nothing(no_display):
    """The measured failure mode on the reference deployment: the a11y bus
    stays up with the lingering user manager while the X server is gone, so
    observation answers and input lands nowhere. Refuse; do not report
    success."""
    loop, driver, _ = _loop(RuleChooser(prefer=("send",)))
    loop._skip_preconditions = False
    result = _run(loop)
    assert result.status == "blocked", (
        f"a step ran without verifying the display substrate: {result.status}"
    )
    assert "DISPLAY" in result.reason
    assert driver.dispatched == []


def test_display_set_but_dead_is_still_refused(tmp_path, monkeypatch):
    """DISPLAY being SET is not evidence of a display — that is the mistake."""
    from prometheus.computer import driver as _driver
    from prometheus.computer.driver import check_preconditions

    empty = tmp_path / "X11-unix"
    empty.mkdir()
    monkeypatch.setattr(_driver, "X11_SOCKET_DIR", str(empty))
    result = check_preconditions({
        "DISPLAY": ":0", "XDG_RUNTIME_DIR": str(tmp_path),
    })
    assert not result.ok
    assert "stale" in result.reason or "does not exist" in result.reason


def test_a_live_display_with_no_a11y_bus_is_refused(tmp_path, monkeypatch):
    """The half-up case, named explicitly: input would dispatch, observation
    would return nothing, and the step would look successful.

    The display here is a REAL listening socket, so this exercises the real
    connect() rather than relying on the host having one.
    """
    import socket

    from prometheus.computer import driver as _driver
    from prometheus.computer.driver import STATE_ACT_ONLY, check_preconditions

    x11 = tmp_path / "X11-unix"
    x11.mkdir()
    monkeypatch.setattr(_driver, "X11_SOCKET_DIR", str(x11))
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(str(x11 / "X7"))
    sock.listen(1)
    try:
        result = check_preconditions({
            "DISPLAY": ":7", "XDG_RUNTIME_DIR": str(tmp_path / "no-such-rt"),
        })
        assert not result.ok
        assert result.act.ok, "the live display half should have answered"
        assert result.state == STATE_ACT_ONLY, (
            "the dangerous mixed state was not named"
        )
    finally:
        sock.close()
