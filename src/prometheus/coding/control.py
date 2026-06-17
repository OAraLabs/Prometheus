"""Mid-run control channel for coding runs (Loop Manager Sprint 2, strong-Supervised).

Fork-1 design (Phase 0 finding): a coding run is a SEPARATE subprocess, so the daemon and
the run communicate through a single JSON control file the run polls at its EPISODE SEAM
(``coding/session.py``, top of the ``while True``). The daemon is the SOLE writer — a
``paused`` flag plus an append-only injection queue; the run is the only reader and tracks
which injections it has applied IN MEMORY, so the two processes never contend on the file.

FAIL-SAFE, NOT FAIL-OPEN (this is the safety-critical path): a missing / unreadable /
malformed control file reads as "not paused, nothing to inject" — a broken channel can never
wedge or corrupt a run. The channel is DORMANT unless a ``control_dir`` is supplied, so an
un-supervised run is byte-identical to today (default-unchanged guarantee).

Episode-level granularity is deliberate (Phase 0 decision): a human steers AFTER the model
completes a coherent chunk and stops — the only well-formed injection point (mid-episode
would corrupt run_loop's tool-call/tool-result pairing).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

CONTROL_FILENAME = "control.json"


@dataclass(frozen=True)
class Injection:
    """One human correction queued for the run. ``id`` lets the run apply each exactly once."""

    id: str
    text: str


@dataclass(frozen=True)
class ControlState:
    """The control file's content, normalized. The safe default is dormant (no pause, no inject)."""

    paused: bool = False
    injections: tuple[Injection, ...] = ()


EMPTY = ControlState()


def control_path(control_dir: str | Path) -> Path:
    """The control file path for a run, given its per-run control dir. Daemon and run both
    derive this identically from the same dir, so no side-channel is needed."""
    return Path(control_dir).expanduser() / CONTROL_FILENAME


def parse_control(raw: str | None) -> ControlState:
    """Parse the control file content — FAIL-SAFE. Anything unparseable (None, bad JSON,
    wrong shape, malformed entries) degrades to ``EMPTY`` (dormant), never raises."""
    if not raw:
        return EMPTY
    try:
        obj = json.loads(raw)
    except Exception:
        return EMPTY
    if not isinstance(obj, dict):
        return EMPTY
    paused = bool(obj.get("paused", False))
    injections: list[Injection] = []
    raw_injs = obj.get("injections", [])
    if isinstance(raw_injs, list):
        for item in raw_injs:
            if not isinstance(item, dict):
                continue
            iid = item.get("id")
            text = item.get("text")
            # A blank text or non-string id/text is dropped — never inject garbage.
            if isinstance(iid, str) and iid and isinstance(text, str) and text.strip():
                injections.append(Injection(id=iid, text=text))
    return ControlState(paused=paused, injections=tuple(injections))


def serialize_control(state: ControlState) -> str:
    return json.dumps(
        {
            "paused": state.paused,
            "injections": [{"id": i.id, "text": i.text} for i in state.injections],
        },
        indent=2,
    )


# ── Pure state transitions (the daemon endpoints' logic — testable without IO) ──

def with_paused(state: ControlState, paused: bool) -> ControlState:
    return ControlState(paused=paused, injections=state.injections)


def with_injection(state: ControlState, injection: Injection) -> ControlState:
    """Append an injection (queue grows; the run consumes by id, so order/idempotency hold)."""
    return ControlState(paused=state.paused, injections=state.injections + (injection,))


# ── The RUN's reader side (inside the coding subprocess) ──

class RunControl:
    """Polled at the episode seam by ``CodingSession``. Disabled (a complete no-op) when no
    ``control_dir`` is given — that is the default-unchanged path. Applied injections are
    tracked in memory so each lands exactly once and the run never writes the control file."""

    def __init__(self, control_dir: str | Path | None) -> None:
        self._path: Path | None = control_path(control_dir) if control_dir else None
        self._applied: set[str] = set()

    @property
    def enabled(self) -> bool:
        return self._path is not None

    @property
    def path(self) -> Path | None:
        return self._path

    def read(self) -> ControlState:
        """FAIL-SAFE read. Disabled / missing / unreadable / malformed → ``EMPTY`` (dormant)."""
        if self._path is None:
            return EMPTY
        try:
            raw = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return EMPTY
        except Exception:  # noqa: BLE001 — OSError, UnicodeDecodeError (garbage bytes), anything
            log.warning(
                "coding control: unreadable control file %s — treating as no-control",
                self._path, exc_info=True,
            )
            return EMPTY
        return parse_control(raw)

    def take_new_injections(self, state: ControlState) -> list[Injection]:
        """Return injections not yet applied this run, marking them applied. Idempotent:
        the same queued injection is returned once even if it stays in the file."""
        fresh = [i for i in state.injections if i.id not in self._applied]
        for i in fresh:
            self._applied.add(i.id)
        return fresh


# ── The DAEMON's writer side (the endpoint handlers; sole writer of the file) ──

def read_state_for_write(path: Path) -> ControlState:
    """Read current state to read-modify-write. Missing/bad → EMPTY (a fresh queue)."""
    try:
        return parse_control(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return EMPTY
    except OSError:
        return EMPTY


def write_state(path: Path, state: ControlState) -> None:
    """Persist control state (daemon side). Creates the control dir if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_control(state), encoding="utf-8")
