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


@dataclass(frozen=True)
class PreconditionResult:
    """Whether the desktop is usable, and which half failed if not."""

    ok: bool
    display: str = ""
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


def check_preconditions(env: dict[str, str] | None = None) -> PreconditionResult:
    """Verify BOTH halves of the desktop substrate. Refuse loudly, not quietly.

    Deliberately does not import Cua, start anything, or take a screenshot —
    it answers "could an action possibly land?" and nothing more. Returning a
    reason rather than raising keeps it usable as a diagnostic (``oara
    doctor``) as well as a gate.
    """
    env = dict(os.environ if env is None else env)

    display = env.get("DISPLAY", "")
    if not display:
        return PreconditionResult(
            ok=False,
            reason=(
                "no DISPLAY in this process's environment — the daemon was "
                "started without a graphical session (a lingering systemd "
                "--user manager starts at boot, before any login, and a later "
                "login does not reach an already-running process). Restart "
                "the daemon from inside a graphical session."
            ),
        )

    # CONNECT, do not trust the variable. A stale DISPLAY from a dead session
    # is indistinguishable from a live one until something opens the socket.
    sock_path = _x11_socket_path(display)
    if sock_path is None:
        return PreconditionResult(
            ok=False, display=display,
            reason=(
                f"DISPLAY={display!r} is not a local socket display; remote X "
                f"is not supported by this precondition check"
            ),
        )
    if not os.path.exists(sock_path):
        return PreconditionResult(
            ok=False, display=display,
            reason=(
                f"DISPLAY={display!r} is set but {sock_path} does not exist — "
                f"a stale display inherited from a session that has ended. "
                f"Nothing would be typed or clicked; actions would be "
                f"dispatched into a dead server."
            ),
        )
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect(sock_path)
    except OSError as exc:
        return PreconditionResult(
            ok=False, display=display,
            reason=(
                f"DISPLAY={display!r} exists but refused a connection "
                f"({exc.__class__.__name__}: {exc}) — the X server is not "
                f"accepting clients."
            ),
        )

    # The OTHER half. An accessibility bus that is absent means observation
    # returns nothing, and "nothing" must not read as "no work to do".
    runtime = env.get("XDG_RUNTIME_DIR", "")
    a11y = os.path.join(runtime, "at-spi", "bus") if runtime else ""
    if not runtime:
        return PreconditionResult(
            ok=False, display=display,
            reason="no XDG_RUNTIME_DIR — the accessibility bus cannot be located",
        )
    if not os.path.exists(a11y):
        return PreconditionResult(
            ok=False, display=display,
            reason=(
                f"the X display is live but the AT-SPI bus socket ({a11y}) is "
                f"absent — observation would return an empty tree while input "
                f"still dispatched. That combination reports success and does "
                f"nothing, so it is refused here."
            ),
        )

    return PreconditionResult(ok=True, display=display)


def _x11_socket_path(display: str) -> str | None:
    """``:0`` / ``:0.1`` -> ``/tmp/.X11-unix/X0``. None for remote displays."""
    if not display.startswith(":"):
        return None
    number = display[1:].split(".", 1)[0]
    if not number.isdigit():
        return None
    return f"/tmp/.X11-unix/X{number}"


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
