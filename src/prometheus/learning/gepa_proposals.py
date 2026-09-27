"""GEPA proposals — the staging area, and the only way one reaches ``skills/auto/``.

GEPA never changes a live skill. A variant that beat the live skill on the
same evidence is written here as a PROPOSAL, and it changes nothing until a
person promotes it. This module is the whole review surface: list, show,
promote, reject. ``oara gepa`` only calls it, so a chat command or a Beacon
panel added later reuses these functions rather than a second copy of the
rules.

Layout, under ``~/.prometheus/skills/proposals/``::

    <id>.json   the sidecar: target skill, the live version it was scored
                against (sha256), the scores, the aggregate evidence summary,
                who judged and who generated
    <id>.md     the variant, exactly as it was judged
    <id>.diff   live → variant, redacted
    .promoted/  proposals a person promoted  (moved here, never deleted)
    .rejected/  proposals a person rejected  (moved here, never deleted)

Not ``skills/drafts/``: Beacon's drafts panel ACCEPTs a draft by writing it
beside the live skill under a suffixed name, with neither the scanner nor an
archive — the wrong operation for a replacement. And not ``skills/auto/``: the
loader serves every ``*.md`` there.

Promotion refuses unless the live skill is byte-for-byte the version the
variant was scored against and the staged variant is byte-for-byte what was
scored: a score is evidence about those two texts and no others. It then
runs the DangerousCodeScanner gate, archives the live version to
``skills/auto/archive/`` and writes the variant in its place.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prometheus.config.paths import config_dir_path
from prometheus.security.log_redaction import redact_secrets

log = logging.getLogger(__name__)

# Path-traversal defense: an id crossing a surface boundary must match this
# exactly before it names a file.
PROPOSAL_ID_RE = re.compile(r"^gepa-[0-9]+-[0-9a-f]{4}$")

# A sidecar's skill_file is a bare file name in skills/auto/, never a path.
_SKILL_FILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.md$")

PROMOTED_DIR_NAME = ".promoted"
REJECTED_DIR_NAME = ".rejected"
ARCHIVE_DIR_NAME = "archive"

STATUS_PENDING = "pending"
STATUS_PROMOTED = "promoted"
STATUS_REJECTED = "rejected"


def default_proposals_dir() -> Path:
    return config_dir_path() / "skills" / "proposals"


def default_auto_dir() -> Path:
    return config_dir_path() / "skills" / "auto"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def frontmatter_closed(text: str) -> bool:
    """True when *text* opens with a ``---`` frontmatter block that closes.

    The loader reads a skill's name and description only from a block that
    starts on the first line, so a variant without one would be served under
    its file stem with a made-up description.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return False
    return any(line.strip() == "---" for line in lines[1:])


def skill_identity(stem: str, text: str) -> tuple[str, str]:
    """``(served name, description)`` the loader would read from *text*."""
    from prometheus.skills.loader import _parse_skill_markdown

    return _parse_skill_markdown(stem, text)


class ProposalError(Exception):
    """A proposal operation refused. ``code`` is stable for any surface to branch on.

    Codes: ``invalid_id``, ``not_found``, ``not_pending``, ``invalid``,
    ``live_missing``, ``stale``, ``modified``, ``identity``, ``unsafe``,
    ``scanner_failed``, ``io``.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Proposal:
    """One proposal as stored: its sidecar, the variant and the diff."""

    id: str
    status: str
    sidecar: dict[str, Any]
    variant: str
    diff: str


@dataclass(frozen=True)
class PromotionResult:
    proposal_id: str
    skill: str
    skill_path: Path
    archive_path: Path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temp file in the same directory, then rename over *path*.

    The temp name does not end in ``.md``: the loader globs ``*.md``, and a
    half-written skill must never be served.
    """
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class ProposalStore:
    """Disk-backed staging area for GEPA proposals.

    Creates nothing until a proposal is written: listing an empty store, or a
    dry run, leaves no directory behind.
    """

    def __init__(
        self,
        proposals_dir: Path | None = None,
        skills_auto_dir: Path | None = None,
    ) -> None:
        self._dir = Path(proposals_dir) if proposals_dir else default_proposals_dir()
        self._auto_dir = Path(skills_auto_dir) if skills_auto_dir else default_auto_dir()

    @property
    def proposals_dir(self) -> Path:
        return self._dir

    @property
    def skills_auto_dir(self) -> Path:
        return self._auto_dir

    @property
    def archive_dir(self) -> Path:
        return self._auto_dir / ARCHIVE_DIR_NAME

    # ── ids and paths ──────────────────────────────────────────────

    @staticmethod
    def validate_id(proposal_id: str) -> str:
        if not isinstance(proposal_id, str) or not PROPOSAL_ID_RE.match(proposal_id):
            raise ProposalError("invalid_id", f"not a proposal id: {proposal_id!r}")
        return proposal_id

    def _new_id(self) -> str:
        for _ in range(8):
            pid = f"gepa-{int(time.time())}-{uuid.uuid4().hex[:4]}"
            if not any(
                (d / f"{pid}.json").exists()
                for d in (self._dir, self._dir / PROMOTED_DIR_NAME, self._dir / REJECTED_DIR_NAME)
            ):
                return pid
        raise RuntimeError("could not allocate a unique proposal id")

    def _files(self, directory: Path, pid: str) -> tuple[Path, Path, Path]:
        return (directory / f"{pid}.json", directory / f"{pid}.md", directory / f"{pid}.diff")

    def _target(self, sidecar: dict[str, Any]) -> Path:
        name = sidecar.get("skill_file")
        if not isinstance(name, str) or not _SKILL_FILE_RE.match(name) or ".bak-" in name:
            raise ProposalError("invalid", f"the proposal names no valid skill file: {name!r}")
        return self._auto_dir / name

    # ── writing (GEPA) ─────────────────────────────────────────────

    def create(
        self,
        *,
        skill_file: str,
        skill_name: str,
        live_text: str,
        variant_text: str,
        scores: dict[str, Any],
        rule: dict[str, Any],
        evidence: dict[str, Any],
        judge: dict[str, Any] | None,
        generator: dict[str, Any],
        variants_judged: int,
    ) -> dict[str, Any]:
        """Stage a proposal and return its sidecar. Writes nothing in ``skills/auto/``."""
        pid = self._new_id()
        self._dir.mkdir(parents=True, exist_ok=True)
        sidecar_path, variant_path, diff_path = self._files(self._dir, pid)
        diff = "".join(difflib.unified_diff(
            live_text.splitlines(keepends=True),
            variant_text.splitlines(keepends=True),
            fromfile=f"skills/auto/{skill_file} (live)",
            tofile=f"proposal {pid}",
        ))
        sidecar: dict[str, Any] = {
            "id": pid,
            "status": STATUS_PENDING,
            "created_at": _now_iso(),
            "skill": skill_name,
            "skill_file": skill_file,
            "live_sha256": sha256_text(live_text),
            "variant_sha256": sha256_text(variant_text),
            "scores": scores,
            "rule": rule,
            "evidence": evidence,
            "judge": judge,
            "generator": generator,
            "variants_judged": variants_judged,
        }
        variant_path.write_text(variant_text, encoding="utf-8")
        diff_path.write_text(redact_secrets(diff), encoding="utf-8")
        # The sidecar last: a proposal is pending exactly when its sidecar is
        # there, so a crash mid-write leaves no half proposal in the list.
        _atomic_write(sidecar_path, json.dumps(sidecar, indent=2, default=str))
        log.info("GEPA: staged proposal %s for %s", pid, skill_file)
        return sidecar

    def has_pending(self, skill_file: str, live_sha256: str) -> bool:
        """A proposal for this exact live version is already waiting for a person."""
        return any(
            s.get("skill_file") == skill_file and s.get("live_sha256") == live_sha256
            for s in self._sidecars(self._dir)
        )

    # ── reading ────────────────────────────────────────────────────

    def _sidecars(self, directory: Path) -> list[dict[str, Any]]:
        if not directory.is_dir():
            return []
        out: list[dict[str, Any]] = []
        for path in directory.glob("gepa-*.json"):
            if not PROPOSAL_ID_RE.match(path.stem):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                log.warning("GEPA proposals: unreadable sidecar %s — skipping", path.name)
                continue
            if isinstance(data, dict):
                out.append(data)
        out.sort(key=lambda d: str(d.get("created_at", "")), reverse=True)
        return out

    def stale_reason(self, sidecar: dict[str, Any]) -> str | None:
        """Why this proposal can no longer be promoted as scored, or None if it can."""
        try:
            target = self._target(sidecar)
        except ProposalError as exc:
            return exc.message
        if not target.is_file():
            return "the live skill no longer exists"
        try:
            live = target.read_text(encoding="utf-8")
        except OSError:
            return "the live skill cannot be read"
        if sha256_text(live) != sidecar.get("live_sha256"):
            return "the live skill changed after this proposal was scored"
        return None

    def entries(self, status: str = STATUS_PENDING) -> list[dict[str, Any]]:
        """Sidecars with that status, newest first. Pending ones carry ``stale`` (a reason or None)."""
        directory = {
            STATUS_PENDING: self._dir,
            STATUS_PROMOTED: self._dir / PROMOTED_DIR_NAME,
            STATUS_REJECTED: self._dir / REJECTED_DIR_NAME,
        }.get(status)
        if directory is None:
            raise ValueError(f"unknown status: {status!r}")
        sidecars = self._sidecars(directory)
        if status == STATUS_PENDING:
            for s in sidecars:
                s["stale"] = self.stale_reason(s)
        return sidecars

    def get(self, proposal_id: str) -> Proposal:
        """A proposal in any state. ``ProposalError`` (``invalid_id`` / ``not_found``) otherwise."""
        pid = self.validate_id(proposal_id)
        for status, directory in (
            (STATUS_PENDING, self._dir),
            (STATUS_PROMOTED, self._dir / PROMOTED_DIR_NAME),
            (STATUS_REJECTED, self._dir / REJECTED_DIR_NAME),
        ):
            sidecar_path, variant_path, diff_path = self._files(directory, pid)
            if not sidecar_path.is_file():
                continue
            try:
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                variant = variant_path.read_text(encoding="utf-8")
                diff = diff_path.read_text(encoding="utf-8") if diff_path.is_file() else ""
            except (OSError, ValueError) as exc:
                raise ProposalError("io", f"proposal {pid} cannot be read: {exc}") from exc
            if not isinstance(sidecar, dict):
                raise ProposalError("io", f"proposal {pid} has a malformed sidecar")
            if status == STATUS_PENDING:
                sidecar["stale"] = self.stale_reason(sidecar)
            return Proposal(id=pid, status=status, sidecar=sidecar, variant=variant, diff=diff)
        raise ProposalError("not_found", f"no proposal {pid}")

    # ── the human actions ──────────────────────────────────────────

    def _pending(self, proposal_id: str) -> Proposal:
        proposal = self.get(proposal_id)
        if proposal.status != STATUS_PENDING:
            raise ProposalError(
                "not_pending", f"proposal {proposal.id} was already {proposal.status}"
            )
        return proposal

    def _close(self, proposal: Proposal, dest_name: str, updates: dict[str, Any]) -> None:
        """Move a pending proposal's files into *dest_name* with its sidecar updated."""
        dest = self._dir / dest_name
        dest.mkdir(parents=True, exist_ok=True)
        sidecar = {k: v for k, v in proposal.sidecar.items() if k != "stale"}
        sidecar.update(updates)
        src_json, src_md, src_diff = self._files(self._dir, proposal.id)
        dst_json, dst_md, dst_diff = self._files(dest, proposal.id)
        for src, dst in ((src_md, dst_md), (src_diff, dst_diff)):
            if src.is_file():
                os.replace(src, dst)
        _atomic_write(dst_json, json.dumps(sidecar, indent=2, default=str))
        src_json.unlink(missing_ok=True)

    def reject(self, proposal_id: str, *, reason: str | None = None, actor: str = "cli") -> dict[str, Any]:
        """Close a pending proposal without touching any skill."""
        proposal = self._pending(proposal_id)
        updates = {
            "status": STATUS_REJECTED,
            "rejected_at": _now_iso(),
            "rejected_by": actor,
            "reason": reason or "",
        }
        self._close(proposal, REJECTED_DIR_NAME, updates)
        log.info("GEPA: proposal %s rejected (%s)", proposal.id, actor)
        return {**{k: v for k, v in proposal.sidecar.items() if k != "stale"}, **updates}

    def promote(self, proposal_id: str, *, actor: str = "cli") -> PromotionResult:
        """Replace the live skill with the proposal's variant — the explicit human action.

        Refuses (``ProposalError``) unless every check holds; nothing is
        written until they all have.
        """
        proposal = self._pending(proposal_id)
        sidecar, variant = proposal.sidecar, proposal.variant
        target = self._target(sidecar)
        if not target.is_file():
            raise ProposalError("live_missing", f"{target.name} no longer exists in skills/auto/")
        live = target.read_text(encoding="utf-8")
        if sha256_text(live) != sidecar.get("live_sha256"):
            raise ProposalError(
                "stale",
                f"{target.name} changed after proposal {proposal.id} was scored; its score is "
                "about a version that is gone. Reject it; GEPA scores the new version next cycle.",
            )
        if sha256_text(variant) != sidecar.get("variant_sha256"):
            raise ProposalError(
                "modified",
                f"the staged variant of {proposal.id} is not the text that was scored",
            )
        if not frontmatter_closed(variant) or (
            skill_identity(target.stem, variant) != skill_identity(target.stem, live)
        ):
            raise ProposalError(
                "identity",
                "the variant does not keep the live skill's frontmatter name and description",
            )

        # The scanner gate. A failure to scan refuses too: nothing unscanned
        # is ever written into skills/auto/.
        try:
            from prometheus.security.code_scanner import DangerousCodeScanner

            scan = DangerousCodeScanner().scan_markdown_content(variant, file_path=str(target))
        except Exception as exc:
            log.exception("GEPA: scanner failed on proposal %s", proposal.id)
            raise ProposalError("scanner_failed", f"the code scanner failed: {exc}") from exc
        if scan.is_dangerous:
            findings = "; ".join(f"{f.rule}: {f.detail}" for f in scan.findings)
            raise ProposalError("unsafe", f"the variant contains dangerous code — {findings}")

        archive_path = self._archive(target, live)
        # The narrowest window we can leave: the live file is checked again
        # right before it is replaced.
        if sha256_text(target.read_text(encoding="utf-8")) != sidecar.get("live_sha256"):
            raise ProposalError(
                "stale",
                f"{target.name} changed while proposal {proposal.id} was being promoted; "
                f"nothing was replaced (the copy archived as {archive_path.name} stays)",
            )
        try:
            _atomic_write(target, variant)
        except OSError as exc:
            raise ProposalError("io", f"could not write {target.name}: {exc}") from exc

        updates = {
            "status": STATUS_PROMOTED,
            "promoted_at": _now_iso(),
            "promoted_by": actor,
            "archive": f"{ARCHIVE_DIR_NAME}/{archive_path.name}",
        }
        try:
            self._close(proposal, PROMOTED_DIR_NAME, updates)
        except OSError:
            # The skill IS promoted; only the bookkeeping move failed. A second
            # promote refuses as stale, so this cannot apply twice.
            log.warning("GEPA: promoted %s but could not file proposal %s",
                        target.name, proposal.id, exc_info=True)
        log.info("GEPA: promoted proposal %s into %s (%s; previous version %s)",
                 proposal.id, target.name, actor, archive_path.name)
        return PromotionResult(
            proposal_id=proposal.id,
            skill=str(sidecar.get("skill") or target.stem),
            skill_path=target,
            archive_path=archive_path,
        )

    def _archive(self, target: Path, live: str) -> Path:
        """Copy the live version into ``auto/archive/`` under a name no copy holds yet."""
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time())
        for n in range(1, 100):
            suffix = "" if n == 1 else f"-{n}"
            path = self.archive_dir / f"{target.stem}_{stamp}{suffix}.md"
            try:
                # Exclusive: an archive is never overwritten (two promotions in
                # one second must not share a name — the #594 collision).
                with path.open("x", encoding="utf-8") as fh:
                    fh.write(live)
                return path
            except FileExistsError:
                continue
            except OSError as exc:
                raise ProposalError("io", f"could not archive {target.name}: {exc}") from exc
        raise ProposalError("io", f"could not find a free archive name for {target.name}")


# ── text for any surface ─────────────────────────────────────────────


def _fmt(x: Any) -> str:
    return f"{x:.2f}" if isinstance(x, (int, float)) else "?"


def _judge_label(judge: Any) -> str:
    if not isinstance(judge, dict) or not judge.get("model"):
        return "unknown"
    return f"{judge['model']} ({'pinned' if judge.get('pinned') else 'not pinned'})"


def _generator_label(gen: Any) -> str:
    if not isinstance(gen, dict):
        return "unknown"
    where = "hosted" if gen.get("hosted") else "local"
    return f"{gen.get('provider') or '?'} ({where})"


def render_list(sidecars: list[dict[str, Any]], *, status: str = STATUS_PENDING) -> str:
    """One line per proposal. The same text for the CLI and any chat surface."""
    if not sidecars:
        return f"No {status} GEPA proposals."
    head = f"{status.capitalize()} GEPA proposals ({len(sidecars)})"
    if status == STATUS_PENDING:
        head += " — no skill changes until one is promoted"
    lines = [head + ":"]
    for s in sidecars:
        sc = s.get("scores") or {}
        line = (
            f"  {s.get('id')}  {s.get('skill')}  "
            f"{_fmt(sc.get('live_mean'))} → {_fmt(sc.get('variant_mean'))} "
            f"(+{_fmt(sc.get('gain'))}, {sc.get('runs', '?')} runs)  {str(s.get('created_at', ''))[:10]}"
        )
        if s.get("stale"):
            line += f"  STALE: {s['stale']}"
        lines.append(line)
    if status == STATUS_PENDING:
        lines.append("")
        lines.append("Review: oara gepa show <id> · promote: oara gepa promote <id> · "
                     "reject: oara gepa reject <id>")
    return "\n".join(lines)


def render_proposal(proposal: Proposal) -> str:
    """Scores, the aggregate evidence and the diff. No request, call or reply text."""
    s = proposal.sidecar
    sc = s.get("scores") or {}
    rule = s.get("rule") or {}
    ev = s.get("evidence") or {}
    lines = [
        f"Proposal {proposal.id} ({proposal.status})",
        f"Skill: {s.get('skill')} (skills/auto/{s.get('skill_file')})",
        f"Created: {s.get('created_at')}",
        f"Judge: {_judge_label(s.get('judge'))}   Generator: {_generator_label(s.get('generator'))}",
    ]
    if s.get("stale"):
        lines.append(f"STALE: {s['stale']} — promote will refuse; reject it.")
    live = sc.get("live") or []
    variant = sc.get("variant") or []
    lines.append(f"Scores on the same {sc.get('runs', len(live))} runs (live → variant):")
    for i, (a, b) in enumerate(zip(live, variant), 1):
        lines.append(f"  run {i}: {_fmt(a)} → {_fmt(b)}")
    lines.append(
        f"  mean:  {_fmt(sc.get('live_mean'))} → {_fmt(sc.get('variant_mean'))} "
        f"(+{_fmt(sc.get('gain'))}; needed ≥ +{_fmt(rule.get('min_margin'))}, "
        f"a mean ≥ {_fmt(rule.get('threshold'))}, worse on no run)"
    )
    outcomes = ev.get("outcomes") or {}
    tools = ev.get("tools") or {}
    lines.append(
        f"Evidence (counts only): {ev.get('runs', 0)} runs in {ev.get('sessions', 0)} sessions, "
        f"loads {ev.get('first_load') or '?'} → {ev.get('last_load') or '?'}; "
        f"{ev.get('calls_after_load', 0)} calls after the load"
        + (f" ({', '.join(f'{k} {v}' for k, v in outcomes.items())})" if outcomes else "")
        + f"; {ev.get('runs_with_reply', 0)} ended with a reply, "
        f"{ev.get('runs_without_reply', 0)} without, {ev.get('runs_end_unknown', 0)} unknown"
    )
    if tools:
        lines.append("  tools after the load: " + ", ".join(f"{k} ×{v}" for k, v in tools.items()))
    if proposal.status == STATUS_PROMOTED:
        lines.append(f"Promoted {s.get('promoted_at')} by {s.get('promoted_by')}; "
                     f"the previous version is skills/auto/{s.get('archive')}")
    elif proposal.status == STATUS_REJECTED:
        why = f": {s.get('reason')}" if s.get("reason") else ""
        lines.append(f"Rejected {s.get('rejected_at')} by {s.get('rejected_by')}{why}")
    lines.append("")
    lines.append("Diff (live → proposal):")
    lines.append(proposal.diff.rstrip() or "(no textual difference)")
    return "\n".join(lines)


def render_promotion(result: PromotionResult) -> str:
    return (
        f"Promoted {result.proposal_id}: {result.skill_path.name} now holds the variant.\n"
        f"The previous version is {ARCHIVE_DIR_NAME}/{result.archive_path.name} in skills/auto/; "
        "the next load of the skill serves the new text."
    )
