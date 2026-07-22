"""SkillDraftStore — the human-review tier of the two-tier trust policy.

Two-tier trust (record-a-skill): DOM-recorded traces are deterministic
ground truth, so the live recorder may auto-persist them into
``skills/auto/`` once the quality gate passes. Vision-derived skills
(video ingestion, screen-capture inference) are *model interpretations*
of what happened — they never take the auto path. Instead they land here
as drafts: ``<draft_id>.md`` (the full SKILL.md content, Prometheus
frontmatter) plus a ``<draft_id>.json`` sidecar (source, title,
description, counts, provenance). A human reviews the draft in Beacon —
optionally redlining the markdown — and only an explicit ACCEPT routes
the content through ``SkillCreator.persist_skill_content()`` (the same
validated write path every machine-generated skill uses).

Curator convention applies: machine-managed skill files are archived,
NEVER deleted. Rejected drafts move to ``drafts/.rejected/``; accepted
drafts move to ``drafts/.accepted/`` as an audit trail after the persist
succeeds.

Model-free, stdlib only.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prometheus.config.paths import get_config_dir

log = logging.getLogger(__name__)

# Path-traversal defense: every draft_id crossing the API boundary must
# match this exactly (same paranoia as server.py's _resolve helpers).
_DRAFT_ID_RE = re.compile(r"^draft-[0-9]+-[0-9a-f]{4}$")

_REJECTED_DIR_NAME = ".rejected"
_ACCEPTED_DIR_NAME = ".accepted"

_NUMBERED_LINE_RE = re.compile(r"^\s*\d+[.)]\s")
_BULLET_LINE_RE = re.compile(r"^\s*[-*]\s")


class SkillDraftStore:
    """Disk-backed store for skill drafts awaiting human review.

    Args:
        drafts_dir: Override the drafts directory
            (default ``~/.prometheus/skills/drafts``).
    """

    def __init__(self, drafts_dir: Path | None = None) -> None:
        self._drafts_dir = drafts_dir or (get_config_dir() / "skills" / "drafts")
        self._drafts_dir.mkdir(parents=True, exist_ok=True)

    @property
    def drafts_dir(self) -> Path:
        return self._drafts_dir

    # ── id validation ──────────────────────────────────────────────

    @staticmethod
    def _validate_id(draft_id: str) -> str:
        """Raise ``ValueError`` unless ``draft_id`` matches the strict format."""
        if not isinstance(draft_id, str) or not _DRAFT_ID_RE.match(draft_id):
            raise ValueError(f"invalid draft id: {draft_id!r}")
        return draft_id

    # ── CRUD ───────────────────────────────────────────────────────

    def create(
        self,
        content: str,
        *,
        source: str,
        provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store a new draft and return its sidecar dict.

        ``title`` / ``description`` / ``name`` are parsed out of the
        markdown (frontmatter + body). ``step_count`` /
        ``parameter_count`` come from ``provenance`` when present (the
        producing pipeline knows best), otherwise they are counted from
        the ``## Steps`` / ``## Parameters`` sections.
        """
        provenance = dict(provenance or {})
        draft_id = self._new_id()

        name = _extract_frontmatter_field(content, "name")
        title = _extract_title(content) or name or draft_id
        description = (
            _extract_frontmatter_field(content, "description")
            or _first_body_line(content)
        )

        step_count = provenance.get("step_count")
        if not isinstance(step_count, int):
            step_count = _count_section_lines(content, "Steps", _NUMBERED_LINE_RE)
        parameter_count = provenance.get("parameter_count")
        if not isinstance(parameter_count, int):
            parameter_count = _count_section_lines(content, "Parameters", _BULLET_LINE_RE)

        sidecar: dict[str, Any] = {
            "draft_id": draft_id,
            "source": source,
            "name": name,
            "title": title,
            "description": description,
            "step_count": step_count,
            "parameter_count": parameter_count,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "provenance": provenance,
        }

        (self._drafts_dir / f"{draft_id}.md").write_text(content, encoding="utf-8")
        (self._drafts_dir / f"{draft_id}.json").write_text(
            json.dumps(sidecar, indent=2, default=str), encoding="utf-8"
        )
        log.info("Skill drafts: created %s (source=%s, title=%r)", draft_id, source, title)
        return sidecar

    def list(self) -> list[dict[str, Any]]:
        """All pending draft sidecars, newest first."""
        sidecars: list[dict[str, Any]] = []
        for path in self._drafts_dir.glob("*.json"):
            if not _DRAFT_ID_RE.match(path.stem):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                log.warning("Skill drafts: unreadable sidecar %s — skipping", path)
                continue
            if isinstance(data, dict):
                sidecars.append(data)
        sidecars.sort(key=lambda d: str(d.get("created_at", "")), reverse=True)
        return sidecars

    def get(self, draft_id: str) -> tuple[str, dict[str, Any]]:
        """Return ``(content, sidecar)`` for a pending draft.

        Raises ``ValueError`` on an invalid id, ``KeyError`` when the
        draft does not exist (or has already been accepted/rejected).
        """
        self._validate_id(draft_id)
        md_path = self._drafts_dir / f"{draft_id}.md"
        sidecar_path = self._drafts_dir / f"{draft_id}.json"
        if not md_path.is_file() or not sidecar_path.is_file():
            raise KeyError(draft_id)
        content = md_path.read_text(encoding="utf-8")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if not isinstance(sidecar, dict):
            raise KeyError(draft_id)
        return content, sidecar

    # ── lifecycle (archive, never delete) ──────────────────────────

    def resolve_rejected_dir(self) -> Path:
        """The ``drafts/.rejected/`` archive dir (created on demand)."""
        path = self._drafts_dir / _REJECTED_DIR_NAME
        path.mkdir(parents=True, exist_ok=True)
        return path

    def resolve_accepted_dir(self) -> Path:
        """The ``drafts/.accepted/`` audit-trail dir (created on demand)."""
        path = self._drafts_dir / _ACCEPTED_DIR_NAME
        path.mkdir(parents=True, exist_ok=True)
        return path

    def reject(self, draft_id: str) -> None:
        """Move both draft files into ``drafts/.rejected/`` (never delete)."""
        self._move_pair(draft_id, self.resolve_rejected_dir())
        log.info("Skill drafts: rejected %s (archived to .rejected/)", draft_id)

    def remove_accepted(self, draft_id: str) -> None:
        """Move both draft files into ``drafts/.accepted/`` after a successful persist."""
        self._move_pair(draft_id, self.resolve_accepted_dir())
        log.info("Skill drafts: accepted %s (archived to .accepted/)", draft_id)

    # ── internals ──────────────────────────────────────────────────

    def _new_id(self) -> str:
        """Generate ``draft-<unixtime>-<4 hex>`` — retry on the freak collision."""
        for _ in range(8):
            draft_id = f"draft-{int(time.time())}-{uuid.uuid4().hex[:4]}"
            if not (self._drafts_dir / f"{draft_id}.md").exists():
                return draft_id
        raise RuntimeError("could not allocate a unique draft id")

    def _move_pair(self, draft_id: str, dest_dir: Path) -> None:
        self._validate_id(draft_id)
        md_path = self._drafts_dir / f"{draft_id}.md"
        sidecar_path = self._drafts_dir / f"{draft_id}.json"
        if not md_path.is_file() and not sidecar_path.is_file():
            raise KeyError(draft_id)
        for src in (md_path, sidecar_path):
            if src.is_file():
                src.replace(dest_dir / src.name)


# ── model-free markdown parsing helpers ────────────────────────────


def _extract_frontmatter_field(content: str, field: str) -> str | None:
    """Pull ``<field>:`` from YAML frontmatter. ``None`` when absent or empty."""
    in_fm = False
    prefix = f"{field}:"
    for raw in content.splitlines():
        line = raw.strip()
        if line == "---":
            if in_fm:
                break
            in_fm = True
            continue
        if in_fm and line.startswith(prefix):
            value = line.split(":", 1)[1].strip().strip("'\"")
            return value or None
    return None


def _extract_title(content: str) -> str | None:
    """First ``# `` heading in the body (outside frontmatter)."""
    in_fm = False
    for raw in content.splitlines():
        line = raw.strip()
        if line == "---":
            in_fm = not in_fm
            continue
        if not in_fm and line.startswith("# "):
            return line[2:].strip() or None
    return None


def _first_body_line(content: str) -> str:
    """First non-heading, non-empty body line (description fallback)."""
    in_fm = False
    for raw in content.splitlines():
        line = raw.strip()
        if line == "---":
            in_fm = not in_fm
            continue
        if not in_fm and line and not line.startswith("#"):
            return line
    return ""


def _count_section_lines(content: str, section: str, pattern: re.Pattern[str]) -> int:
    """Count lines matching ``pattern`` inside the ``## <section>`` block."""
    in_section = False
    count = 0
    for raw in content.splitlines():
        stripped = raw.strip()
        if stripped.startswith("## "):
            in_section = stripped[3:].strip().lower() == section.lower()
            continue
        if in_section and pattern.match(raw):
            count += 1
    return count
