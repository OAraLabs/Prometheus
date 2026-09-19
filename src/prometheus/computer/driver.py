"""The driver seam, and the precondition check that must pass before it is used.

TWO THINGS LIVE HERE AND THEY ARE RELATED
------------------------------------------
``Driver`` is the protocol the loop executes against. ``FixtureDriver``
implements it from recorded observations, which is what makes milestone 1
provable with no display, no Cua install, and no credential.

``check_preconditions`` is the other half, and it exists because of a
measured property of the deployment box rather than a hypothetical.

THE PRECONDITION FINDING (measured on the reference deployment, 2026-09-19)
----------------------------------------------------------------------------
Cua's Linux backend takes its accessibility tree from **AT-SPI over D-Bus**
and synthesises input with **XTEST over X11**. Those are two different
transports and on this box they fail independently:

* ``systemd --user`` runs with ``Linger=yes``, so it starts at boot — BEFORE
  any graphical login — and the daemon (``WantedBy=default.target``) starts
  with it. A cold boot with nobody logging in gives the daemon NO ``DISPLAY``.
  A later graphical login never reaches the already-running process, because
  a process environment is fixed at exec.
* ``systemctl --user show-environment`` holds ``DISPLAY`` / ``XAUTHORITY``
  imported by whichever GNOME session ran most recently. A session that dies
  after the daemon started leaves the daemon holding a STALE display and an
  ``XAUTHORITY`` pointing at a randomly-suffixed mutter file that a new
  session will not recreate.
* Through all of it the session bus at ``$XDG_RUNTIME_DIR/bus`` stays up with
  the lingering user manager. **So the observe half can answer while the act
  half has no server.**

That last line is the whole reason this function exists. An empty
accessibility tree and a working one are the same shape; a candidate table
built from an empty observation is empty, and a loop that treats "no
candidates" as "nothing to do" would report success having done nothing. The
answer is not to trust the driver to be loud — that was not established from
its documentation — but to check both halves here, before the first action,
and refuse with a reason naming which half failed.

⚠ ``DISPLAY`` BEING SET IS NOT EVIDENCE OF A DISPLAY. Checking the variable
is exactly the mistake: on this box it is set, inherited, and may point at
nothing. The check CONNECTS.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from typing import Any, Protocol

from prometheus.computer.types import Observation


#: Where X11 puts its unix sockets. A named constant rather than a literal
#: because it is an ENVIRONMENT ASSUMPTION, and a test that wants to exercise
#: the real ``connect()`` needs somewhere it can actually create a socket.
#: Tests point this at a tmp dir and bind a real listener there — which is the
#: only way to test the connect path without depending on the host having a
#: display, and depending on the host is how a test ends up measuring the box
#: instead of the code.
X11_SOCKET_DIR = "/tmp/.X11-unix"

#: The three answers a half can give. ``unknown`` is a THIRD ANSWER and never
#: collapses into ``ok`` — a check that cannot see a problem must say so
#: rather than report clean. Same ruling #518 applied to the tracking ref.
HALF_OK = "ok"
HALF_UNAVAILABLE = "unavailable"
HALF_UNKNOWN = "unknown"

#: Rollup states. ``act_only`` is the one this whole module exists for.
STATE_READY = "ready"
STATE_ACT_ONLY = "act_only"
STATE_OBSERVE_ONLY = "observe_only"
STATE_UNAVAILABLE = "unavailable"
STATE_UNKNOWN = "unknown"


@dataclass(frozen=True)
class HalfResult:
    """One half of the substrate: what it is, and whether it answered.

    ``component`` NAMES the thing rather than locating it. The status payload
    renders these verbatim, and the locations involved are a uid-bearing
    runtime path and a display number — neither belongs on an endpoint whose
    whole job is to be read by someone who is worried. The paths stay in the
    logs, where the person reading them is already on the box.
    """

    state: str
    component: str
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.state == HALF_OK


@dataclass(frozen=True)
class PreconditionResult:
    """Both halves, reported INDEPENDENTLY, plus a rollup.

    ⚠ THE HALVES ARE EVALUATED SEPARATELY AND NEITHER SHORT-CIRCUITS THE
    OTHER. An earlier version returned as soon as the X half failed, so a box
    with no display never learned whether its accessibility bus was up. That
    made the two halves look like one control with one answer — and the whole
    reason this module exists is that they fail on DIFFERENT AXES and the
    dangerous case is the MIXED one, which a single boolean cannot express.
    """

    act: HalfResult
    observe: HalfResult
    display: str = ""

    @property
    def ok(self) -> bool:
        """Both halves answered. Anything else refuses."""
        return self.act.ok and self.observe.ok

    @property
    def state(self) -> str:
        """The rollup, by remedy urgency — and UNKNOWN OUTRANKS EVERYTHING.

        An unknown half means the rollup cannot be trusted, so it is reported
        as unknown rather than as the worse of the two known answers. Saying
        ``act_only`` when the observe half merely could not be reached would
        be asserting something we did not establish — which is the failure
        this vocabulary exists to prevent, one level up.
        """
        if HALF_UNKNOWN in (self.act.state, self.observe.state):
            return STATE_UNKNOWN
        if self.act.ok and self.observe.ok:
            return STATE_READY
        if self.act.ok:
            # ⚠ THE DANGEROUS ONE. Input dispatches; the tree comes back
            # empty. An empty tree is shaped exactly like a working one, so a
            # loop that shrugs at "no candidates" reports success having done
            # nothing. Named separately so it can never hide inside a generic
            # failure.
            return STATE_ACT_ONLY
        if self.observe.ok:
            return STATE_OBSERVE_ONLY
        return STATE_UNAVAILABLE

    @property
    def reason(self) -> str:
        """Why a step is refused. Empty when both halves answered."""
        parts = [h.detail for h in (self.act, self.observe)
                 if h.state != HALF_OK and h.detail]
        return " / ".join(parts)

    def __bool__(self) -> bool:
        return self.ok


def check_preconditions(env: dict[str, str] | None = None) -> PreconditionResult:
    """Verify BOTH halves of the desktop substrate. Refuse loudly, not quietly.

    Deliberately does not import Cua, start anything, or take a screenshot —
    it answers "could an action possibly land?" and nothing more. Returning a
    result rather than raising keeps it usable as a diagnostic (``oara
    doctor``, ``GET /api/status``) as well as a gate.
    """
    env = dict(os.environ if env is None else env)
    display = env.get("DISPLAY", "")
    # Evaluated independently — see PreconditionResult's docstring.
    return PreconditionResult(
        act=_check_act_half(display),
        observe=_check_observe_half(env),
        display=display,
    )


def _check_act_half(display: str) -> HalfResult:
    """XTEST over X11: could an input event land anywhere?

    ⚠ CONNECTS. It does not read the variable and believe it. On the
    reference deployment ``DISPLAY`` is set, inherited from a lingering
    ``systemd --user`` manager, and may point at a session that has ended —
    so the variable being present is not evidence of a display, and checking
    it is exactly the mistake.
    """
    component = "x11-display"
    if not display:
        return HalfResult(
            HALF_UNAVAILABLE, component,
            "no DISPLAY in this process's environment — the daemon was "
            "started without a graphical session. A lingering systemd --user "
            "manager starts at boot, before any login, and a later login does "
            "not reach an already-running process. Restart the daemon from "
            "inside a graphical session.",
        )
    sock_path = _x11_socket_path(display)
    if sock_path is None:
        # A remote display might well work; this check cannot tell, and
        # saying "unavailable" would assert something unestablished.
        return HalfResult(
            HALF_UNKNOWN, component,
            "DISPLAY is not a local socket display; this check cannot "
            "establish whether a remote X server would accept input.",
        )
    if not os.path.exists(sock_path):
        return HalfResult(
            HALF_UNAVAILABLE, component,
            "DISPLAY is set but its socket does not exist — a stale display "
            "inherited from a session that has ended. Nothing would be typed "
            "or clicked; actions would dispatch into a dead server.",
        )
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect(sock_path)
    except OSError as exc:
        return HalfResult(
            HALF_UNAVAILABLE, component,
            f"the display socket exists but refused a connection "
            f"({exc.__class__.__name__}) — the X server is not accepting "
            f"clients.",
        )
    except Exception as exc:  # noqa: BLE001 - a probe must not raise upward
        return HalfResult(
            HALF_UNKNOWN, component,
            f"the display could not be probed ({exc.__class__.__name__}).",
        )
    return HalfResult(HALF_OK, component, "")


def _check_observe_half(env: dict[str, str]) -> HalfResult:
    """AT-SPI over D-Bus: would an observation return a real tree?

    Separate transport from the act half, and that is the point: the session
    bus outlives a dead graphical session, so this half can answer while the
    other has no server at all.
    """
    component = "at-spi-bus"
    runtime = env.get("XDG_RUNTIME_DIR", "")
    if not runtime:
        return HalfResult(
            HALF_UNKNOWN, component,
            "no XDG_RUNTIME_DIR — the accessibility bus cannot be located, "
            "so whether it would answer is not established.",
        )
    try:
        present = os.path.exists(os.path.join(runtime, "at-spi", "bus"))
    except Exception as exc:  # noqa: BLE001
        return HalfResult(
            HALF_UNKNOWN, component,
            f"the accessibility bus could not be probed "
            f"({exc.__class__.__name__}).",
        )
    if not present:
        return HalfResult(
            HALF_UNAVAILABLE, component,
            "the accessibility bus socket is absent — observation would "
            "return an empty tree while input still dispatched. That "
            "combination reports success and does nothing.",
        )
    return HalfResult(HALF_OK, component, "")


def _x11_socket_path(display: str) -> str | None:
    """``:0`` / ``:0.1`` -> ``<X11_SOCKET_DIR>/X0``. None for remote displays."""
    if not display.startswith(":"):
        return None
    number = display[1:].split(".", 1)[0]
    if not number.isdigit():
        return None
    return os.path.join(X11_SOCKET_DIR, f"X{number}")


class Driver(Protocol):
    """What the loop needs from a desktop driver.

    Narrow on purpose. The real Cua SDK surface is ~50 tools; this is the
    subset v1 executes, and the narrowness is the security boundary as much as
    the abstraction — a driver implementation cannot offer the loop
    ``execute_javascript`` because the loop has nowhere to call it.
    """

    def observe(
        self, target: str, app: str, pid: int, window_id: int
    ) -> Observation:
        """Fresh snapshot. Every call supersedes the previous one's tokens.

        ``target`` is passed even though a bound driver already knows which
        machine it reaches: the Observation it returns carries the target into
        every candidate built from it, and a driver silently stamping its own
        idea of the name would let a registry mis-binding produce actions
        labelled for one machine and executed on another.
        """
        ...

    def act(self, verb: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatch one complete bounded action. Raises ``StaleSnapshot``."""
        ...


class StaleSnapshot(RuntimeError):
    """The action's snapshot has been superseded. Never retried silently."""


class DriverUnavailable(RuntimeError):
    """The driver cannot act — preconditions failed, or it is not installed."""


class FixtureDriver:
    """A Driver backed by recorded observations. No display, no Cua, no network.

    This is the milestone-1 substrate. It is not a mock in the "returns
    whatever you assert" sense — it enforces the real invariants:

    * every ``observe`` mints a NEW snapshot id and invalidates the previous
      one, exactly as Cua documents;
    * ``act`` raises ``StaleSnapshot`` when handed a superseded snapshot,
      rather than succeeding;
    * dispatched actions are recorded, so a test asserts on the EFFECT (what
      reached the driver) rather than on a return string.

    That last property is the one that matters: a test asserting "the tool
    said ok" measures prose. ``dispatched`` is what actually happened.
    """

    def __init__(self, observations: list[Observation]) -> None:
        if not observations:
            raise ValueError("FixtureDriver needs at least one observation")
        self._observations = list(observations)
        self._cursor = 0
        # Primed to the FIRST observation rather than None. A FixtureDriver is
        # constructed *from* recorded observations, so "the window as last
        # seen" is the first of them — a driver that treated its own starting
        # state as stale would refuse every action until something called
        # observe(), which is an artefact of the fixture rather than a
        # property of the real invariant. The invariant itself is unchanged
        # and still tested: once observe() advances to a later snapshot, an
        # action carrying the earlier one is refused.
        self._current_snapshot: str | None = observations[0].snapshot_id
        #: Every action that reached the driver, in order. Assert on THIS.
        self.dispatched: list[tuple[str, dict[str, Any]]] = []

    def observe(
        self, target: str, app: str, pid: int, window_id: int
    ) -> Observation:
        obs = self._observations[min(self._cursor, len(self._observations) - 1)]
        if obs.target != target:
            # A fixture labelled for another machine is a test-authoring
            # mistake, but it is the same mistake a mis-bound registry would
            # make in production, so it is refused rather than accommodated.
            raise DriverUnavailable(
                f"fixture observation is for target {obs.target!r}, not "
                f"{target!r}"
            )
        self._cursor += 1
        self._current_snapshot = obs.snapshot_id
        return obs

    def act(self, verb: str, arguments: dict[str, Any]) -> dict[str, Any]:
        snapshot = arguments.get("snapshot_id")
        if snapshot and snapshot != self._current_snapshot:
            raise StaleSnapshot(
                f"snapshot {snapshot!r} has been superseded by "
                f"{self._current_snapshot!r} — re-observe before acting"
            )
        self.dispatched.append((verb, dict(arguments)))
        return {"ok": True, "verb": verb}
