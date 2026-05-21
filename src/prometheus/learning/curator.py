"""Curator — scheduled consolidation pass over auto-generated skills.

# Pattern adapted from Hermes Agent (NousResearch/hermes-agent)
# Original: hermes_agent/agent/curator.py (state machine + two-stage grading + YAML output)
# License: MIT
# Adaptation notes:
#   - Schedule: Prometheus uses a wall-clock asyncio loop (GoldenTraceExporter
#     pattern, see ``sentinel/golden_trace_exporter.py``). Hermes triggers
#     ``maybe_run_curator`` from a session-start hook with an idle gate; the
#     Prometheus daemon is long-running so a periodic loop with an in-line
#     idle check is the better fit. ``min_idle_seconds`` config knob preserves
#     the spirit of the upstream gate.
#   - Two-stage pipeline matches upstream:
#       1. Deterministic auto-pass flips lifecycle state (active / stale /
#          archived) based on file mtime — no LLM, no file moves.
#       2. LLM review pass emits a fenced YAML block with ``consolidations``
#          and ``prunings`` lists. We parse with PyYAML, no grammar / JSON
#          schema — same prompt shape Hermes uses.
#   - Acts on LLM prunings only (move skill file to ``auto/.archive/``).
#     Hermes lets the model invoke ``skill_manage action=delete`` to perform
#     the actual move; Prometheus has no in-loop tool surface for the
#     Curator, so we apply prunings ourselves. Consolidations are RECORDED
#     in the report as suggestions but NOT auto-applied — autonomous content
#     merging would conflict with the sprint constraint "pruned skills go to
#     archive, never delete", and a consolidation deletes the source skill.
#   - Pinned skills are excluded from BOTH the auto-archive transition AND
#     any LLM pruning suggestion. The pin check happens on the way out of
#     the LLM pass, defensively, so the model can't accidentally prune a
#     pinned skill by name even if asked.
#   - Report layout matches Hermes: ``~/.prometheus/curator/{stamp}/`` with
#     ``run.json`` (machine-readable) and ``REPORT.md`` (human-readable).
#     ``stamp = YYYYMMDD-HHMMSS``; same-second collisions get a numeric
#     suffix.
#   - State persistence is the SkillStateStore (single JSON, atomic
#     tempfile-fsync-rename). Hermes splits state across multiple files
#     under ``logs/curator/``; we colocate.
#
# Differences from Hermes
# -----------------------
# 1. Wall-clock cron instead of session-start hook (see above).
# 2. We apply prunings directly (file system move) instead of leaning on a
#    skill_manage tool; we don't apply consolidations (report-only).
# 3. ``REFINE`` doesn't exist as a Curator action — SkillRefiner is the
#    refinement subsystem and runs on a different signal (post-task hook).
#    The sprint spec's PIN/KEEP/REFINE/CONSOLIDATE/PRUNE vocabulary
#    collapses to: state lifecycle (auto) + consolidation suggestions (LLM,
#    report-only) + prunings (LLM, applied). KEEP is implicit. PIN is a user
#    action via ``/skills pin``.
# 4. Usage signal: mtime-based v1 (no telemetry counter yet) — see
#    ``skill_state.py`` adaptation notes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prometheus.config.paths import get_config_dir
from prometheus.learning.skill_state import (
    SKILL_STATE_ACTIVE,
    SKILL_STATE_ARCHIVED,
    SKILL_STATE_STALE,
    SkillRecord,
    SkillStateStore,
)

if TYPE_CHECKING:
    from prometheus.providers.base import ModelProvider
    from prometheus.sentinel.signals import SignalBus

log = logging.getLogger(__name__)

# Module-level singleton — set by daemon.py after Curator is constructed
# so /curator run / show / status commands can reach it. Matches the
# tools/builtin/sentinel_status.py set_sentinel_components pattern.
_curator_instance: "Curator | None" = None


def set_curator(curator: "Curator | None") -> None:
    """Register the running Curator instance for command access."""
    global _curator_instance
    _curator_instance = curator


def get_curator() -> "Curator | None":
    """Return the registered Curator instance (None if not wired)."""
    return _curator_instance

# Defaults match Hermes constants.
_DEFAULT_INTERVAL_SECONDS = 7 * 24 * 3600       # 7 days
_DEFAULT_STALE_AFTER_DAYS = 30
_DEFAULT_ARCHIVE_AFTER_DAYS = 90
_DEFAULT_MIN_IDLE_SECONDS = 0                   # 0 = always run on tick
_DEFAULT_MAX_PRUNINGS_PER_RUN = 10              # safety cap

_REVIEW_PROMPT = """\
You are the skill library curator. Review the auto-generated skills below
and recommend consolidations and prunings. Output ONLY a fenced YAML block.

Each skill is listed with:
  - name: filename stem
  - state: lifecycle (active / stale / archived)
  - pinned: protected from any change
  - last_used_days_ago: integer
  - first_line: the file's first non-frontmatter line (description)

Rules:
  - DO NOT prune pinned skills. DO NOT consolidate pinned skills.
  - Prefer suggesting consolidation when two or more skills clearly cover
    the same task (e.g. share a prefix, same first-line meaning).
  - Recommend pruning only when a skill is stale or archived AND has no
    consolidation target AND is clearly low-signal (one-shot debug,
    superseded approach, etc.).
  - At most {max_prunings} prunings per run.
  - When in doubt, leave the skill alone (empty lists are valid output).

Output format (must be valid YAML inside the fence):
```yaml
consolidations:
  - from: skill-a
    into: umbrella-skill
    reason: short reason
prunings:
  - name: skill-x
    reason: short reason
```

Skill library:
{library}
"""


@dataclass
class CuratorRun:
    """One Curator pass result."""

    started_at: float
    ended_at: float = 0.0
    stamp: str = ""
    auto_transitions: list[dict[str, Any]] = field(default_factory=list)
    consolidations: list[dict[str, Any]] = field(default_factory=list)
    prunings: list[dict[str, Any]] = field(default_factory=list)
    skipped_pinned: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    report_path: str = ""
    json_path: str = ""
    skills_reviewed: int = 0
    llm_raw_output: str = ""

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.ended_at - self.started_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "stamp": self.stamp,
            "skills_reviewed": self.skills_reviewed,
            "auto_transitions": self.auto_transitions,
            "consolidations": self.consolidations,
            "prunings": self.prunings,
            "skipped_pinned": self.skipped_pinned,
            "errors": self.errors,
            "report_path": self.report_path,
            "json_path": self.json_path,
        }


def _default_auto_dir() -> Path:
    return get_config_dir() / "skills" / "auto"


def _default_reports_dir() -> Path:
    return get_config_dir() / "curator"


class Curator:
    """Scheduled consolidation pass over auto-generated skills.

    Stage 1 (auto): walks ``~/.prometheus/skills/auto/*.md``; for each skill
    flips its persisted lifecycle state based on file mtime vs ``stale``/
    ``archive`` cutoffs. No file moves. No LLM calls.

    Stage 2 (LLM): feeds the model the list of (name, state, pinned,
    last_used_days_ago, first_line) and parses a fenced YAML block with
    ``consolidations`` (report-only) and ``prunings`` (applied — file is
    moved to ``auto/.archive/``).

    Reports land under ``~/.prometheus/curator/{YYYYMMDD-HHMMSS}/`` as
    ``REPORT.md`` (human-readable) and ``run.json`` (full record).

    Args:
        provider: ModelProvider for the LLM review pass.
        model: Model name to call.
        signal_bus: Optional SignalBus for emitting ``curator_report``.
        state_store: SkillStateStore (defaults to module-level path).
        auto_dir: Path to the auto-skills directory.
        reports_dir: Path to write Curator run reports.
        interval_seconds: How often the background loop fires (default 7d).
        stale_after_days: Lifecycle threshold for active → stale (mtime).
        archive_after_days: Lifecycle threshold for stale → archived (mtime).
        min_idle_seconds: Skip a tick if the daemon was active in the last N seconds.
            ``0`` disables the gate (default). Reserved for future use; the
            current implementation always runs on tick.
        max_prunings_per_run: Safety cap on how many skills may be archived
            per run. The LLM is told this cap in the prompt.
    """

    def __init__(
        self,
        provider: ModelProvider,
        *,
        model: str = "default",
        signal_bus: SignalBus | None = None,
        state_store: SkillStateStore | None = None,
        auto_dir: Path | None = None,
        reports_dir: Path | None = None,
        interval_seconds: int = _DEFAULT_INTERVAL_SECONDS,
        stale_after_days: int = _DEFAULT_STALE_AFTER_DAYS,
        archive_after_days: int = _DEFAULT_ARCHIVE_AFTER_DAYS,
        min_idle_seconds: int = _DEFAULT_MIN_IDLE_SECONDS,
        max_prunings_per_run: int = _DEFAULT_MAX_PRUNINGS_PER_RUN,
    ) -> None:
        self._provider = provider
        self._model = model
        self._signal_bus = signal_bus
        self._state_store = state_store or SkillStateStore()
        self._auto_dir = Path(auto_dir or _default_auto_dir())
        self._reports_dir = Path(reports_dir or _default_reports_dir())
        self._interval = max(60, int(interval_seconds))
        self._stale_after_days = max(1, int(stale_after_days))
        self._archive_after_days = max(1, int(archive_after_days))
        self._min_idle_seconds = max(0, int(min_idle_seconds))
        self._max_prunings = max(0, int(max_prunings_per_run))
        self._running = False
        self._task: asyncio.Task | None = None

    # ---------------------- Construction helpers ----------------------

    @classmethod
    def from_config(
        cls,
        provider: ModelProvider,
        *,
        model: str = "default",
        signal_bus: SignalBus | None = None,
        config: dict[str, Any] | None = None,
    ) -> Curator | None:
        """Build a Curator from a ``learning`` config dict.

        Returns ``None`` if ``curator_enabled`` is False so callers can skip
        wiring the scheduled task entirely.
        """
        cfg = (config or {}).get("learning", config or {})
        if not cfg.get("curator_enabled", True):
            return None
        return cls(
            provider,
            model=model,
            signal_bus=signal_bus,
            interval_seconds=int(cfg.get("curator_interval_seconds", _DEFAULT_INTERVAL_SECONDS)),
            stale_after_days=int(cfg.get("curator_stale_after_days", _DEFAULT_STALE_AFTER_DAYS)),
            archive_after_days=int(cfg.get("curator_archive_after_days", _DEFAULT_ARCHIVE_AFTER_DAYS)),
            min_idle_seconds=int(cfg.get("curator_min_idle_seconds", _DEFAULT_MIN_IDLE_SECONDS)),
            max_prunings_per_run=int(cfg.get("curator_max_prunings_per_run", _DEFAULT_MAX_PRUNINGS_PER_RUN)),
        )

    # ---------------------- Lifecycle ----------------------

    async def start(self) -> asyncio.Task | None:
        """Spawn the background loop. Returns the task, or ``None`` if already running."""
        if self._running:
            return None
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="curator_loop")
        log.info(
            "Curator: started (interval=%ds, stale=%dd, archive=%dd, auto_dir=%s)",
            self._interval,
            self._stale_after_days,
            self._archive_after_days,
            self._auto_dir,
        )
        return self._task

    async def stop(self) -> None:
        """Signal the loop to exit at next check."""
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    async def _loop(self) -> None:
        """Background loop: run_once → sleep interval."""
        # Defer the first run by one interval to avoid clobbering on every restart.
        # Hermes does the same for first-run safety.
        try:
            await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            return
        while self._running:
            try:
                if self._state_store.curator().paused:
                    log.debug("Curator: paused, skipping tick")
                else:
                    await self.run_once()
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("Curator: run_once failed")
            try:
                await asyncio.sleep(self._interval)
            except asyncio.CancelledError:
                break

    # ---------------------- Manual entry point ----------------------

    async def run_once(self, *, dry_run: bool = False) -> CuratorRun:
        """Execute one full Curator pass. Returns the run record.

        ``dry_run=True`` skips file moves and state-store writes but still
        invokes the LLM and produces a report (useful for ``/curator run --dry-run``).
        """
        started = time.time()
        run = CuratorRun(started_at=started, stamp=self._make_stamp(started))

        skills = self._discover_skills()
        run.skills_reviewed = len(skills)

        # Stage 1: auto-transitions (deterministic, no LLM).
        run.auto_transitions = self._apply_auto_transitions(skills, dry_run=dry_run)

        # Stage 2: LLM review pass.
        if skills:
            try:
                consolidations, prunings, raw = await self._review_via_llm(skills)
                run.consolidations = consolidations
                run.prunings = prunings
                run.llm_raw_output = raw[:8000]  # cap raw output in record
            except Exception as exc:
                log.exception("Curator: LLM review failed")
                run.errors.append(f"LLM review failed: {exc!s}")

        # Apply prunings (move to .archive/, never delete).
        if run.prunings and not dry_run:
            run.skipped_pinned, applied = self._apply_prunings(run.prunings)
            # Keep only the records we actually moved.
            run.prunings = applied

        run.ended_at = time.time()

        # Persist the report + state.
        try:
            run.report_path, run.json_path = self._write_report(run)
        except Exception as exc:
            log.exception("Curator: report write failed")
            run.errors.append(f"report write failed: {exc!s}")

        if not dry_run:
            self._state_store.update_curator(
                last_run_at=run.ended_at,
                last_report_path=run.report_path,
            )

        # Emit signal (best-effort).
        await self._emit_report_signal(run)

        log.info(
            "Curator: run %s — reviewed=%d, auto=%d, consolidations=%d, prunings=%d, errors=%d",
            run.stamp,
            run.skills_reviewed,
            len(run.auto_transitions),
            len(run.consolidations),
            len(run.prunings),
            len(run.errors),
        )
        return run

    # ---------------------- Internals: discovery + auto-pass ----------------------

    def _discover_skills(self) -> list[dict[str, Any]]:
        """Walk ``auto_dir`` for ``*.md`` skill files (excluding ``.archive``)."""
        if not self._auto_dir.is_dir():
            return []
        out: list[dict[str, Any]] = []
        for path in sorted(self._auto_dir.glob("*.md")):
            try:
                stat = path.stat()
            except OSError:
                continue
            name = path.stem
            rec = self._state_store.get_skill(name)
            days_ago = int((time.time() - stat.st_mtime) / 86400)
            out.append({
                "name": name,
                "path": str(path),
                "mtime": stat.st_mtime,
                "size": stat.st_size,
                "days_ago": days_ago,
                "first_line": self._first_meaningful_line(path),
                "state": rec.state,
                "pinned": rec.pinned,
            })
        return out

    def _apply_auto_transitions(
        self, skills: list[dict[str, Any]], *, dry_run: bool
    ) -> list[dict[str, Any]]:
        """Flip lifecycle state based on mtime cutoffs.

        Pinned skills are skipped. State transitions are applied immediately
        to the SkillStateStore unless ``dry_run`` is True.

        Returns a list of ``{name, from_state, to_state, days_ago}`` records.
        """
        transitions: list[dict[str, Any]] = []
        for skill in skills:
            if skill["pinned"]:
                continue
            current = skill["state"]
            target = self._target_state(skill["days_ago"])
            if target == current:
                continue
            transitions.append({
                "name": skill["name"],
                "from_state": current,
                "to_state": target,
                "days_ago": skill["days_ago"],
            })
            if not dry_run:
                try:
                    self._state_store.set_state(skill["name"], target)
                    skill["state"] = target
                except Exception:
                    log.exception(
                        "Curator: failed to persist state transition for %s",
                        skill["name"],
                    )
        return transitions

    def _target_state(self, days_ago: int) -> str:
        if days_ago >= self._archive_after_days:
            return SKILL_STATE_ARCHIVED
        if days_ago >= self._stale_after_days:
            return SKILL_STATE_STALE
        return SKILL_STATE_ACTIVE

    # ---------------------- Internals: LLM review ----------------------

    async def _review_via_llm(
        self, skills: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
        """Call the model. Parse fenced YAML. Return (consolidations, prunings, raw)."""
        library = self._format_library_for_prompt(skills)
        prompt = _REVIEW_PROMPT.format(
            library=library,
            max_prunings=self._max_prunings,
        )

        raw = await self._call_model(prompt)
        parsed = self._parse_yaml_block(raw)
        if not isinstance(parsed, dict):
            return [], [], raw
        consolidations = self._normalise_list(parsed.get("consolidations", []), keys=("from", "into", "reason"))
        prunings = self._normalise_list(parsed.get("prunings", []), keys=("name", "reason"))
        return consolidations, prunings, raw

    async def _call_model(self, prompt: str) -> str:
        from prometheus.engine.messages import ConversationMessage, TextBlock
        from prometheus.providers.base import ApiMessageRequest, ApiTextDeltaEvent

        # ConversationMessage.content is list[ContentBlock]; the existing
        # SkillCreator / SkillRefiner / MemoryExtractor callers pass a raw
        # string which fails pydantic validation. Curator uses the correct
        # list-of-blocks shape; the three pre-existing sites are flagged in
        # Sprint 1 reporting-back.
        request = ApiMessageRequest(
            model=self._model,
            messages=[ConversationMessage(role="user", content=[TextBlock(text=prompt)])],
            max_tokens=2048,
        )
        parts: list[str] = []
        async for event in self._provider.stream_message(request):
            if isinstance(event, ApiTextDeltaEvent):
                parts.append(event.text)
        return "".join(parts)

    @staticmethod
    def _format_library_for_prompt(skills: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for s in skills:
            pinned_tag = " [PINNED]" if s["pinned"] else ""
            lines.append(
                f"- name: {s['name']}{pinned_tag}\n"
                f"  state: {s['state']}\n"
                f"  last_used_days_ago: {s['days_ago']}\n"
                f"  first_line: {s['first_line'][:120]}"
            )
        return "\n".join(lines) if lines else "(no skills in auto/ directory)"

    @staticmethod
    def _parse_yaml_block(raw: str) -> dict[str, Any] | None:
        """Extract a fenced ```yaml ... ``` block. Falls back to whole text."""
        try:
            import yaml
        except Exception:
            log.warning("Curator: PyYAML missing; falling back to JSON parse")
            yaml = None  # type: ignore[assignment]

        block = raw
        fence_start = raw.find("```")
        if fence_start != -1:
            # Find body between the first and second fence.
            after = raw[fence_start + 3:]
            # Strip the optional language tag on the first line.
            nl = after.find("\n")
            if nl != -1:
                after_body = after[nl + 1:]
            else:
                after_body = after
            fence_end = after_body.find("```")
            if fence_end != -1:
                block = after_body[:fence_end]

        if yaml is not None:
            try:
                return yaml.safe_load(block) or {}
            except Exception:
                log.exception("Curator: YAML parse failed; raw block:\n%s", block[:500])
                return None

        # Fallback: try JSON.
        try:
            return json.loads(block)
        except Exception:
            return None

    @staticmethod
    def _normalise_list(items: Any, *, keys: tuple[str, ...]) -> list[dict[str, Any]]:
        if not isinstance(items, list):
            return []
        out: list[dict[str, Any]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            row = {k: str(it.get(k, "")).strip() for k in keys}
            if not row.get(keys[0]):
                continue
            out.append(row)
        return out

    # ---------------------- Internals: apply prunings ----------------------

    def _apply_prunings(
        self, prunings: list[dict[str, Any]]
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Move pruned skill files to ``auto/.archive/``. Skips pinned skills.

        Returns ``(skipped_pinned_names, applied_records)``.
        """
        applied: list[dict[str, Any]] = []
        skipped: list[str] = []
        capped = prunings[: self._max_prunings] if self._max_prunings else prunings
        archive_dir = self._auto_dir / ".archive"
        archive_dir.mkdir(parents=True, exist_ok=True)

        for entry in capped:
            name = entry.get("name", "").strip()
            if not name:
                continue
            rec = self._state_store.get_skill(name)
            if rec.pinned:
                skipped.append(name)
                continue
            src = self._auto_dir / f"{name}.md"
            if not src.exists():
                log.warning("Curator: pruning target %s does not exist on disk", src)
                continue
            # Same-name collisions in archive get a timestamp suffix.
            dest = archive_dir / src.name
            if dest.exists():
                dest = archive_dir / f"{src.stem}-{int(time.time())}{src.suffix}"
            try:
                shutil.move(str(src), str(dest))
            except Exception as exc:
                log.exception("Curator: failed to archive %s", src)
                applied.append({**entry, "archived_to": "", "error": str(exc)})
                continue
            try:
                self._state_store.set_state(name, SKILL_STATE_ARCHIVED)
            except Exception:
                log.exception(
                    "Curator: archived %s but failed to update state_store", name
                )
            applied.append({**entry, "archived_to": str(dest)})

        return skipped, applied

    # ---------------------- Internals: report writing ----------------------

    def _make_stamp(self, when: float) -> str:
        base = datetime.fromtimestamp(when).strftime("%Y%m%d-%H%M%S")
        # If a same-stamp directory already exists (sub-second collisions),
        # add a numeric suffix.
        candidate = self._reports_dir / base
        if not candidate.exists():
            return base
        for i in range(1, 100):
            if not (self._reports_dir / f"{base}-{i}").exists():
                return f"{base}-{i}"
        return f"{base}-overflow"

    def _write_report(self, run: CuratorRun) -> tuple[str, str]:
        """Write ``REPORT.md`` and ``run.json``. Returns the two paths."""
        run_dir = self._reports_dir / run.stamp
        run_dir.mkdir(parents=True, exist_ok=True)

        report_path = run_dir / "REPORT.md"
        json_path = run_dir / "run.json"

        report_path.write_text(self._render_report_md(run), encoding="utf-8")
        json_path.write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")
        return str(report_path), str(json_path)

    @staticmethod
    def _render_report_md(run: CuratorRun) -> str:
        when = datetime.fromtimestamp(run.started_at).strftime("%Y-%m-%d %H:%M:%S")
        lines: list[str] = [
            f"# Curator Report — {run.stamp}",
            "",
            f"_started: {when} • duration: {run.duration_seconds:.1f}s • "
            f"skills reviewed: {run.skills_reviewed}_",
            "",
        ]
        if run.errors:
            lines += ["## Errors"]
            for e in run.errors:
                lines.append(f"- {e}")
            lines.append("")
        lines += ["## Auto transitions"]
        if run.auto_transitions:
            for t in run.auto_transitions:
                lines.append(
                    f"- `{t['name']}`: {t['from_state']} → {t['to_state']} "
                    f"({t['days_ago']}d since last use)"
                )
        else:
            lines.append("_none_")
        lines.append("")

        lines += ["## Consolidation suggestions (report-only — not applied)"]
        if run.consolidations:
            for c in run.consolidations:
                lines.append(
                    f"- merge `{c.get('from','?')}` → `{c.get('into','?')}` — "
                    f"{c.get('reason','')}"
                )
        else:
            lines.append("_none_")
        lines.append("")

        lines += ["## Prunings (archived to `auto/.archive/`)"]
        if run.prunings:
            for p in run.prunings:
                dest = p.get("archived_to", "?")
                err = f" — ERROR: {p['error']}" if p.get("error") else ""
                lines.append(
                    f"- `{p.get('name','?')}` → `{dest}` — {p.get('reason','')}{err}"
                )
        else:
            lines.append("_none_")
        lines.append("")

        if run.skipped_pinned:
            lines += [
                "## Skipped (pinned)",
                "",
                ", ".join(f"`{n}`" for n in run.skipped_pinned),
                "",
            ]
        return "\n".join(lines)

    @staticmethod
    def _first_meaningful_line(path: Path) -> str:
        try:
            with path.open("r", encoding="utf-8") as fh:
                in_frontmatter = False
                for raw in fh:
                    line = raw.strip()
                    if line == "---":
                        in_frontmatter = not in_frontmatter
                        continue
                    if in_frontmatter:
                        continue
                    if not line or line.startswith("#"):
                        continue
                    return line
        except OSError:
            pass
        return ""

    # ---------------------- Signal emission ----------------------

    async def _emit_report_signal(self, run: CuratorRun) -> None:
        if self._signal_bus is None:
            return
        try:
            from prometheus.sentinel.signals import ActivitySignal

            await self._signal_bus.emit(ActivitySignal(
                kind="curator_report",
                payload={
                    "stamp": run.stamp,
                    "skills_reviewed": run.skills_reviewed,
                    "auto_transitions": len(run.auto_transitions),
                    "consolidations": len(run.consolidations),
                    "prunings": len(run.prunings),
                    "skipped_pinned": len(run.skipped_pinned),
                    "duration_seconds": run.duration_seconds,
                    "report_path": run.report_path,
                    "errors": run.errors,
                },
                source="curator",
            ))
        except Exception:
            log.debug("Curator: signal emission failed", exc_info=True)
