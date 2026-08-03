"""Per-session ephemeral mode — "Prometheus won't remember this".

A chat flagged ephemeral does not reach the durable memory pipeline: no
``lcm_messages`` row, so no FTS entry, no summary, no MemoryExtractor input,
no ``memory.db`` fact and no wiki page. Three further write paths are
suppressed at their own call sites (auto-skill creation, the content columns
on ``telemetry.tool_calls``, repair-pair capture) because they are reached
from the agent loop rather than from the conversation store.

What this is NOT: it is not "nothing is recorded". The permission audit log
still records every tool decision (deliberately — an agent running ``bash``
with no trail is a worse hole than the one this closes), ``subsystem_runs``
still records token counts and round indices, cached attachment bytes still
land under ``~/.prometheus/cache/``, and anything the agent WRITES while
doing the work is an ordinary file that no retention mechanism tracks. The
user-facing wording must stay "Prometheus won't remember this" and must never
promise "this isn't recorded" — see ``gateway.commands.cmd_ephemeral``.

Storage shape follows the per-chat voice-mode override
(``gateway/commands.py``): a small JSON file in the config dir, re-read on
every access, so the flag survives a daemon restart. Deliberately NOT the
``ModelRouter._overrides`` shape — that dict is in-memory only, and a
privacy flag that silently evaporates on restart is worse than no flag,
because the user has already been told it is on.

Placement: this lives under ``config/`` rather than ``gateway/`` because the
engine (``engine.session``, ``engine.agent_loop``) must read it, and
gateway→engine is the existing dependency direction. Importing the gateway
from the engine would invert it.

Failure posture (CROSS-CUTTING §8 — say which direction a control fails):

* **Read** (:func:`is_session_ephemeral`) fails toward RETENTION and logs at
  ERROR. Failing the other way would mean an unreadable file silently
  disables the entire memory pipeline for every session at once — a much
  larger, much quieter loss than the one it would prevent.
* **Write** (:func:`set_session_ephemeral`) fails LOUDLY: it raises, and it
  re-reads the file to confirm the flag actually landed before returning.
  The command surface must report that failure instead of confirming. A
  ``/ephemeral on`` that silently did not persist is the one outcome that
  turns this feature into a lie, so it is the one the code refuses to allow.
  (The voice-mode precedent swallows its OSError — correct there, where a
  lost setting is cosmetic. This deliberately diverges.)
"""

from __future__ import annotations

import json
import logging

from prometheus.config.paths import get_config_dir

log = logging.getLogger(__name__)

_EPHEMERAL_FILE = "ephemeral_sessions.json"


class EphemeralFlagWriteError(RuntimeError):
    """The ephemeral flag could not be persisted (or did not read back).

    Raised by :func:`set_session_ephemeral` so the caller reports the failure
    to the user rather than confirming a mode that is not in effect.
    """


def ephemeral_path() -> str:
    """Path to the per-session ephemeral flag file."""
    return str(get_config_dir() / _EPHEMERAL_FILE)


def load_ephemeral_sessions() -> dict[str, bool]:
    """Read the flag map ``{session_id: True}``.

    Only truthy entries are meaningful; ``off`` removes the key rather than
    storing ``False``, so the file stays a small allowlist of ephemeral chats
    instead of accumulating one row per chat that ever toggled.

    Returns ``{}`` on any read failure — the fail-toward-retention direction
    documented in the module docstring. The failure is logged at ERROR, not
    swallowed, because "the file is unreadable" and "no chat is ephemeral"
    must not look the same in the logs.
    """
    path = ephemeral_path()
    try:
        with open(path) as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return {}
    except (ValueError, OSError) as exc:
        log.error(
            "Ephemeral flag file %s is unreadable (%s) — treating every session "
            "as NON-ephemeral (retention ON). Sessions the user set to "
            "ephemeral are being persisted right now.",
            path, exc,
        )
        return {}
    if not isinstance(data, dict):
        log.error(
            "Ephemeral flag file %s is not a JSON object (%s) — treating every "
            "session as NON-ephemeral (retention ON).", path, type(data).__name__,
        )
        return {}
    return {str(k): bool(v) for k, v in data.items() if v}


def is_session_ephemeral(session_id: str | None) -> bool:
    """True iff *session_id* is currently flagged ephemeral.

    Re-reads the file on every call (voice-mode semantics), so a toggle takes
    effect on the very next turn without a restart and without any cache to
    invalidate. ``None`` / empty ids — the reserved system flows — are never
    ephemeral.
    """
    if not session_id:
        return False
    return bool(load_ephemeral_sessions().get(str(session_id)))


def set_session_ephemeral(session_id: str, on: bool) -> None:
    """Turn ephemeral mode on or off for *session_id*, durably.

    Raises :class:`EphemeralFlagWriteError` when the flag could not be written
    OR did not read back with the requested value. The read-back is the point:
    a write that appeared to succeed but left the file unchanged would leave
    the user believing a privacy mode is active when it is not. Verify against
    the artifact, not the return code.
    """
    if not session_id:
        raise EphemeralFlagWriteError("refusing to set an ephemeral flag on an empty session id")

    sessions = load_ephemeral_sessions()
    if on:
        sessions[str(session_id)] = True
    else:
        sessions.pop(str(session_id), None)

    path = ephemeral_path()
    try:
        with open(path, "w") as fh:
            json.dump(sessions, fh, indent=2, sort_keys=True)
    except OSError as exc:
        raise EphemeralFlagWriteError(
            f"could not write the ephemeral flag file {path}: {exc}"
        ) from exc

    # Read back through the SAME resolver the rest of the system uses, so this
    # confirms the state that will actually be consulted — not the state we
    # believe we wrote.
    if is_session_ephemeral(session_id) != bool(on):
        raise EphemeralFlagWriteError(
            f"the ephemeral flag for {session_id} did not read back as "
            f"{'on' if on else 'off'} after writing {path}"
        )
