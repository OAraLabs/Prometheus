"""SkillStateStore — persisted per-skill metadata (pin, state, curator bookkeeping).

# Pattern adapted from Hermes Agent (NousResearch/hermes-agent)
# Original: hermes_agent/agent/curator.py (state-file persistence pattern)
# License: MIT
# Adaptation notes:
#   - Hermes persists state in a state file written via tempfile + fsync + os.replace;
#     we adopt the same atomic-write pattern.
#   - Hermes reads ``pinned`` / ``state`` / ``last_activity_at`` from a
#     ``tools/skill_usage`` module that is NOT in the public Hermes tree.
#     Prometheus owns the storage choice here — a single JSON file at
#     ``~/.prometheus/skills/auto/_state.json``, leading-underscore keeps it
#     out of skill discovery (skills/loader.py only globs ``*.md``).
#   - Mtime-based ``last_used_at`` v1 (no telemetry instrumentation) per Sprint 1
#     scope decision — usage_count stays 0 until a follow-up sprint wires a
#     counter from the skill-loading tool. See Sprint 1 reporting-back.
#   - Curator bookkeeping (``last_run_at``, ``last_report_path``, ``paused``)
#     lives in the same file under a ``curator`` key. Single file keeps the
#     atomic-write story simple; cost is a slightly larger payload on every
#     pin operation. Acceptable at <100KB total expected size.

The file format:

    {
      "version": 1,
      "skills": {
        "<skill-name>": {
          "pinned": false,
          "state": "active",           # active | stale | archived
          "first_seen_at": 1737401200.0,
          "notes": ""
        },
        ...
      },
      "curator": {
        "last_run_at": 1737000000.0,
        "last_report_path": "/home/will/.prometheus/curator/20260520-090000/REPORT.md",
        "paused": false,
        "run_count": 0
      }
    }

`last_used_at` and `usage_count` are intentionally absent — they are derived
fields (mtime + future telemetry), not persisted on the state file.

Differences from Hermes
-----------------------
Hermes splits per-skill telemetry (`tools/skill_usage`) from Curator state
(`logs/curator/`). Prometheus colocates them because the storage burden is
tiny for a solo-operator skill library (typically < 200 skills) and a single
file simplifies the atomic-write contract. If multi-user scenarios ship later
this should split.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prometheus.config.paths import get_config_dir

log = logging.getLogger(__name__)

# State values (subset of Hermes's STATE_ACTIVE / STATE_STALE / STATE_ARCHIVED).
SKILL_STATE_ACTIVE = "active"
SKILL_STATE_STALE = "stale"
SKILL_STATE_ARCHIVED = "archived"
_VALID_STATES = frozenset({SKILL_STATE_ACTIVE, SKILL_STATE_STALE, SKILL_STATE_ARCHIVED})

_STATE_VERSION = 1


def _default_state_path() -> Path:
    """Return ``~/.prometheus/skills/auto/_state.json``.

    Leading underscore intentionally keeps this file outside the
    ``skills/loader.py`` glob (which matches ``*.md`` only). Inside the
    ``auto/`` namespace because that's the set of skills Curator manages.
    """
    return get_config_dir() / "skills" / "auto" / "_state.json"


@dataclass
class SkillRecord:
    """Per-skill persisted metadata."""

    pinned: bool = False
    state: str = SKILL_STATE_ACTIVE
    first_seen_at: float = field(default_factory=time.time)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "pinned": bool(self.pinned),
            "state": self.state,
            "first_seen_at": float(self.first_seen_at),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillRecord:
        state = data.get("state", SKILL_STATE_ACTIVE)
        if state not in _VALID_STATES:
            state = SKILL_STATE_ACTIVE
        return cls(
            pinned=bool(data.get("pinned", False)),
            state=state,
            first_seen_at=float(data.get("first_seen_at", time.time())),
            notes=str(data.get("notes", "")),
        )


@dataclass
class CuratorState:
    """Curator bookkeeping (last run, last report, pause flag)."""

    last_run_at: float = 0.0
    last_report_path: str = ""
    paused: bool = False
    run_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_run_at": float(self.last_run_at),
            "last_report_path": self.last_report_path,
            "paused": bool(self.paused),
            "run_count": int(self.run_count),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CuratorState:
        return cls(
            last_run_at=float(data.get("last_run_at", 0.0)),
            last_report_path=str(data.get("last_report_path", "")),
            paused=bool(data.get("paused", False)),
            run_count=int(data.get("run_count", 0)),
        )


class SkillStateStore:
    """Persisted per-skill state + Curator bookkeeping.

    Read-modify-write with atomic file replacement. Cheap enough for the
    sub-second operations we do (pin a skill, mark stale, record a run).

    Concurrency model: the daemon is the single writer. Reads from CLI
    commands are point-in-time; concurrent writes from two daemon
    threads would race but the asyncio loop serialises them in practice.
    If a future sprint needs strict cross-process safety, swap to fcntl
    advisory lock around the read-modify-write block.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = Path(path) if path else _default_state_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------- Read ----------------------------

    def load(self) -> dict[str, Any]:
        """Return the full state dict. Initialises an empty doc if missing."""
        if not self._path.exists():
            return self._empty()
        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except Exception:
            log.exception(
                "SkillStateStore: failed to read %s, returning empty state",
                self._path,
            )
            return self._empty()
        # Tolerate older / partial docs.
        if not isinstance(data, dict) or data.get("version") != _STATE_VERSION:
            log.warning(
                "SkillStateStore: schema version mismatch at %s (got %r, "
                "expected %d) — resetting",
                self._path, data.get("version"), _STATE_VERSION,
            )
            return self._empty()
        return data

    def get_skill(self, name: str) -> SkillRecord:
        """Return the record for *name* (default record if not present)."""
        state = self.load()
        rec = state.get("skills", {}).get(name)
        if rec is None:
            return SkillRecord()
        return SkillRecord.from_dict(rec)

    def list_skills(self) -> dict[str, SkillRecord]:
        """Return all known records, keyed by skill name."""
        state = self.load()
        return {
            name: SkillRecord.from_dict(rec)
            for name, rec in state.get("skills", {}).items()
        }

    def curator(self) -> CuratorState:
        """Return curator bookkeeping."""
        state = self.load()
        return CuratorState.from_dict(state.get("curator", {}))

    # ---------------------------- Write ----------------------------

    def upsert_skill(self, name: str, record: SkillRecord) -> None:
        """Insert or replace the record for *name*."""
        state = self.load()
        state.setdefault("skills", {})[name] = record.to_dict()
        self._atomic_write(state)

    def set_pinned(self, name: str, pinned: bool) -> SkillRecord:
        """Toggle the ``pinned`` flag for *name*. Returns the new record."""
        rec = self.get_skill(name)
        rec.pinned = bool(pinned)
        self.upsert_skill(name, rec)
        return rec

    def set_state(self, name: str, state_value: str) -> SkillRecord:
        """Update the lifecycle state. ``state_value`` must be a valid state."""
        if state_value not in _VALID_STATES:
            raise ValueError(
                f"invalid state {state_value!r}; expected one of "
                f"{sorted(_VALID_STATES)}"
            )
        rec = self.get_skill(name)
        rec.state = state_value
        self.upsert_skill(name, rec)
        return rec

    def remove_skill(self, name: str) -> bool:
        """Drop a skill's record entirely. Returns True if it was present."""
        state = self.load()
        skills = state.setdefault("skills", {})
        if name not in skills:
            return False
        del skills[name]
        self._atomic_write(state)
        return True

    def update_curator(self, **fields: Any) -> CuratorState:
        """Apply partial updates to the Curator bookkeeping block."""
        state = self.load()
        cur = state.setdefault("curator", {})
        cur.update({k: v for k, v in fields.items() if v is not None})
        # Increment run count when ``last_run_at`` advances.
        if "last_run_at" in fields:
            cur["run_count"] = int(cur.get("run_count", 0)) + 1
        self._atomic_write(state)
        return CuratorState.from_dict(cur)

    # ---------------------------- Internals ----------------------------

    def _empty(self) -> dict[str, Any]:
        return {
            "version": _STATE_VERSION,
            "skills": {},
            "curator": CuratorState().to_dict(),
        }

    def _atomic_write(self, data: dict[str, Any]) -> None:
        """Write *data* atomically (tempfile → fsync → os.replace).

        Adapted from Hermes ``agent/curator.py`` state-write pattern.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(data, indent=2, sort_keys=True).encode("utf-8")
        # Tempfile in the same directory so os.replace is atomic on POSIX.
        fd, tmp_path = tempfile.mkstemp(
            prefix="_state.", suffix=".json.tmp", dir=str(self._path.parent)
        )
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, self._path)
        except Exception:
            # Best-effort cleanup of the temp file on failure.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
