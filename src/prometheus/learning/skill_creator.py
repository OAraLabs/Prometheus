"""SkillCreator — auto-generate SKILL.md files from successful tool-call traces.

PostTaskHook: after a task completes, decide whether the turn contains a
reusable procedure and, if so, produce a skill file under
~/.prometheus/skills/auto/.

The decision is a two-stage gate (2026-08 quality-gate sprint; survey at
audits/20260803T215431Z-skillcreator-quality-gate-survey.md — 50% of the
library was one-offs, failed lookups, and duplicates when tool COUNT was
the only condition):

- Stage 0, deterministic and pre-LLM: 3–50 tool calls, none failed.
- Stage 1, inside the single generation call: the model sees the final
  reply text, per-call error flags, and the existing skill list, and may
  answer ``SKIP: <reason>`` instead of a skill. This is the only layer
  that catches turns whose calls all succeeded mechanically but whose
  ANSWER was negative (the failed-lookup class).

The write path then gates what gets saved (skill-usage audit, option C1):

- a description that is literally ``name: …`` — a frontmatter line the
  loader's tolerant scan keeps verbatim — is rejected, for every writer;
- on the auto path (``on_collision="skip"``), a skill whose
  ``name + description`` is within cosine ``dedupe_threshold`` (default
  ``similarity.DEFAULT_THRESHOLD`` = 0.80) of an existing served skill is
  rejected as a near-duplicate. Deliberate writers keep their content.

Before any of that, every writer's content passes ``DangerousCodeScanner`` —
the same ``scan_markdown_content`` call SkillRefiner and GEPA make before they
write (WP-X.40). A DANGEROUS verdict writes nothing.

Usage:
    creator = SkillCreator(provider)
    skill_path = await creator.maybe_create(task_record, tool_trace, final_text)
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prometheus.config.paths import get_config_dir
from prometheus.learning.skill_files import archive_copy, atomic_write, create_exclusive
from prometheus.learning.trace_format import format_trace
from prometheus.skills.loader import _parse_skill_markdown, load_skill_registry
from prometheus.skills.similarity import DEFAULT_THRESHOLD, skill_text

if TYPE_CHECKING:
    from prometheus.providers.base import ModelProvider

log = logging.getLogger(__name__)

_MIN_TOOL_CALLS = 3
# Five-check upper bound: past this a turn is a saga, not a procedure.
_MAX_TOOL_CALLS = 50
_AUTO_SKILLS_DIR_NAME = "skills/auto"

# The SKIP opt-out below is load-bearing, not decorative: traces whose calls
# all exited 0 but whose ANSWER was negative ("that directory doesn't exist")
# have no deterministic tell — the generation model is the only judge that
# sees the outcome. tests/test_skill_creator.py::TestStage1LLMOptOut fails
# if the opt-out is removed from this prompt.
_GENERATION_PROMPT = """\
You are a skill generator. Given a sequence of tool calls that accomplished a task,
produce a SKILL.md file that codifies the approach for reuse.

Not every turn deserves a skill. Output exactly:

SKIP: <one-line reason>

instead of a skill file when ANY of these hold:
- the request was a question, lookup, or capability test, not a procedure to repeat
- the outcome was a failure or a negative result (nothing found, nonexistent target, task incomplete)
- the trace only read or inspected things to produce an answer; nothing was built, changed, or configured
- an existing skill in the list below already covers this approach

Otherwise, format:
---
name: <short-kebab-case-name>
description: <one-line description of what the skill does>
---

# <Skill Name>

## When to use
<one sentence>

## Steps
1. <step>
2. <step>
...

## Notes
- <any caveats or variations>

Task description: {task_description}

Final response to the user (how the task actually ended):
{final_text}

Tool call trace, one call per line as tool(input) → result, both cut short
(failed calls are marked [ERROR]):
{trace}

Existing skills (do NOT re-create these):
{existing_skills}

Output ONLY the SKILL.md content or the SKIP line. No commentary.
"""


# A description that is literally the frontmatter's ``name:`` line: two of the
# 57 auto skills the mini wrote (docs/audits/SKILL-USAGE.md §5).
_MALFORMED_DESCRIPTION = re.compile(r"^\s*name\s*:", re.IGNORECASE)


def served_skill_catalog() -> list[tuple[str, str]]:
    """``[(name, skill_text), …]`` for every skill the registry serves now."""
    return [(s.name, skill_text(s.name, s.description))
            for s in load_skill_registry().list_skills()]


def _get_auto_skills_dir() -> Path:
    path = get_config_dir() / _AUTO_SKILLS_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _slugify(text: str) -> str:
    """Convert text to a kebab-case filename slug.

    Bounded at 64 chars. Order matters: strip leading/trailing dashes from
    the substitution result, *then* truncate, *then* rstrip any dash that
    a mid-word truncation left behind. The pre-PR-#20 implementation
    (``slug.strip("-")[:60]``) stripped before truncating, leaving a
    trailing ``-`` whenever the 60th char was in the middle of a token
    — that's how filenames like ``you-can-take-down-…-before-.md``
    landed in ``~/.prometheus/skills/auto/``.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower().strip())
    return slug.strip("-")[:64].rstrip("-")


_COLLISION_MODES = frozenset({"suffix", "skip", "refuse", "replace"})


class SkillNameTaken(Exception):
    """``on_collision="refuse"``: a skill with this name is already served.

    ``files`` are the auto skill files holding it; ``served`` the skills the
    registry serves under it from elsewhere (``(source, path)`` — a package
    builtin, the user's own ``skills/``), which an auto skill of that name
    would silently take over from. A surface names them all (a skill-draft
    ACCEPT answers 409 with them).
    """

    def __init__(
        self,
        name: str,
        files: list[Path],
        served: list[tuple[str, Path]] | None = None,
    ) -> None:
        self.name = name
        self.files = list(files)
        self.served = list(served or [])
        super().__init__(
            f"a skill named {name!r} is already served: " + ", ".join(self.where()))

    def where(self) -> list[str]:
        """``auto:<file>`` for each auto file, ``<source>:<file>`` for the rest."""
        return ([f"auto:{f.name}" for f in self.files]
                + [f"{source}:{path.name}" for source, path in self.served])


@dataclass(frozen=True)
class DivertedToDraft:
    """A machine-written skill whose name was already served, staged for a person.

    What :meth:`SkillCreator.persist_or_divert` returns instead of a path: the
    skill sits in ``skills/drafts/`` and the accept flow (409, replace or
    rename) decides what happens to it. Nothing machine-written changes a
    live skill without a person — the same rule as GEPA.
    """

    draft_id: str
    skill_name: str
    served_by: list[str]


class SkillNameExtractionError(ValueError):
    """Raised internally when the LLM's response lacks a usable ``name:``.

    Never propagated to callers — caught at the SkillCreator boundary and
    surfaced via ``telemetry.record_silent_failure`` so missing-name
    failures are observable in /health verbose. Carries a descriptive
    message for the silent_failures row.
    """


class SkillCreator:
    """Generate SKILL.md files from successful task tool-call traces.

    Args:
        provider: ModelProvider for generating skill content.
        model: Model name for the generation call.
        min_tool_calls: Minimum tool calls to trigger skill creation.
        auto_dir: Override the auto-skills output directory.
    """

    def __init__(
        self,
        provider: ModelProvider,
        *,
        model: str = "default",
        min_tool_calls: int = _MIN_TOOL_CALLS,
        max_tool_calls: int = _MAX_TOOL_CALLS,
        auto_dir: Path | None = None,
        signal_bus: object | None = None,
        telemetry: object | None = None,
        similarity: object | None = None,
        dedupe_threshold: float | None = None,
        catalog: Any = None,
        drafts: object | None = None,
    ) -> None:
        from prometheus.learning.llm_envelope import LLMCallEnvelope

        # Where persist_or_divert stages a skill whose name is already served
        # (default: the configured skills/drafts/, built on first use).
        self._drafts = drafts
        # The near-duplicate gate: a checker with ``available``,
        # ``unavailable_reason`` and ``nearest(text, catalog)`` (default: the
        # process-wide encoder, built on first use), the cosine at or above
        # which a skill is rejected, and what it is compared against.
        self._similarity = similarity
        self._dedupe_threshold = (
            DEFAULT_THRESHOLD if dedupe_threshold is None else float(dedupe_threshold)
        )
        self._catalog = catalog or served_skill_catalog
        self._warned_unavailable = False
        self._provider = provider
        self._model = model
        self._min_tool_calls = min_tool_calls
        self._max_tool_calls = max_tool_calls
        self._auto_dir = auto_dir or _get_auto_skills_dir()
        # Sprint S1: SignalBus is wired by daemon.py inside the SENTINEL
        # block (after SignalBus exists, since SkillCreator construction
        # happens earlier in the daemon startup).
        self._signal_bus = signal_bus
        # Sprint S4 A1: LLMCallEnvelope replaces the per-subsystem _call_model
        # exception-swallow pattern from ed8f1a6. on_failure="return_none"
        # preserves the legacy maybe_create contract (returns None on failure)
        # while making every failure visible in telemetry.silent_failures.
        self._telemetry = telemetry
        self._envelope = LLMCallEnvelope(
            subsystem="skill_creator",
            telemetry=telemetry,
            on_failure="return_none",
        )

    @property
    def signal_bus(self) -> object | None:
        return self._signal_bus

    @signal_bus.setter
    def signal_bus(self, bus: object) -> None:
        self._signal_bus = bus

    @classmethod
    def from_config(
        cls,
        provider: ModelProvider,
        config_path: str | None = None,
        *,
        telemetry: object | None = None,
    ) -> SkillCreator:
        """Build from prometheus.yaml learning section."""
        import yaml

        if config_path is None:
            from prometheus.config.defaults import resolve_config_path
            config_path = str(resolve_config_path())

        # Narrow the catch to genuine I/O + YAML-parse errors so any other
        # exception (e.g., a future config-schema upgrade that introduces
        # validation) propagates instead of silently demoting the
        # subsystem to defaults. See docs/audits/SILENT-FAILURE-AUDIT.md
        # Tier-1 hotfix and the PR #1 / ed8f1a6 incident.
        try:
            with open(Path(config_path).expanduser()) as fh:
                data = yaml.safe_load(fh) or {}
            learning = data.get("learning", {}) or {}
            min_calls = learning.get("skill_min_tool_calls", _MIN_TOOL_CALLS)
            threshold = learning.get("skill_dedupe_threshold", DEFAULT_THRESHOLD)
        except (OSError, yaml.YAMLError) as exc:
            log.warning(
                "SkillCreator.from_config: failed to load %s (%s: %s); "
                "using default skill_min_tool_calls=%d",
                config_path, type(exc).__name__, exc, _MIN_TOOL_CALLS,
            )
            min_calls = _MIN_TOOL_CALLS
            threshold = DEFAULT_THRESHOLD

        return cls(provider, min_tool_calls=min_calls, telemetry=telemetry,
                   dedupe_threshold=threshold)

    async def maybe_create(
        self,
        task_description: str,
        tool_trace: list[dict[str, Any]],
        final_text: str = "",
    ) -> Path | None:
        """Create a skill if the trace passes the two-stage quality gate.

        Stage 0 (deterministic, pre-LLM): 3–50 calls and zero failed calls —
        a turn that stumbled is not a procedure worth codifying, and the
        skips here cost nothing.

        Stage 1 (the generation call itself): the model sees the final reply
        text, the error-flagged trace, and the existing skill list, and may
        answer ``SKIP: <reason>`` for question/test/failed/already-covered
        turns. Traces whose calls all succeeded but whose ANSWER was negative
        are caught only here.

        Args:
            task_description: The raw user message that started the turn.
            tool_trace: List of dicts with keys: tool_name, arguments,
                result, is_error.
            final_text: The assistant's final reply for the turn — the only
                place the semantic outcome lives when every call succeeded
                mechanically.

        Returns:
            Path to the created SKILL.md, or None if skipped.
        """
        n_calls = len(tool_trace)
        if n_calls < self._min_tool_calls:
            log.debug(
                "SkillCreator: only %d tool calls (need %d), skipping",
                n_calls,
                self._min_tool_calls,
            )
            return None
        if n_calls > self._max_tool_calls:
            log.info(
                "SkillCreator: %d tool calls (max %d), skipping",
                n_calls,
                self._max_tool_calls,
            )
            return None
        errored = sum(1 for call in tool_trace if call.get("is_error"))
        if errored:
            log.info(
                "SkillCreator: trace has %d failed call(s), skipping",
                errored,
            )
            return None

        trace_text = self._format_trace(tool_trace)
        prompt = _GENERATION_PROMPT.format(
            task_description=task_description,
            final_text=(final_text or "").strip()[:1500] or "(not captured)",
            trace=trace_text,
            existing_skills=self._existing_skills_listing() or "(none)",
        )

        # Envelope returns None on failure (see __init__ on_failure mode); the
        # surrounding try/except is no longer needed because the envelope wrote
        # to telemetry.silent_failures with the full traceback.
        content = await self._call_model(prompt)
        if content is None or not content.strip():
            return None

        skip_reason = self._parse_skip(content)
        if skip_reason is not None:
            log.info(
                "SkillCreator: model declined (%s) — task: %.80s",
                skip_reason or "no reason given",
                task_description,
            )
            return None

        return await self.persist_skill_content(
            content, trigger=task_description, on_collision="skip",
        )

    async def persist_skill_content(
        self,
        content: str,
        *,
        trigger: str,
        on_collision: str = "suffix",
        replaced: list[tuple[Path, Path]] | None = None,
    ) -> Path | None:
        """Validate and write skill markdown through the standard auto-skill path.

        This is THE write path for machine-generated skills — used by
        :meth:`maybe_create`, teacher escalation (``escalation/teacher.py``),
        record-a-skill and an ACCEPTed skill draft (both through
        ``LiveRecorderService.persist_content``), so every writer gets the
        same validation: the DangerousCodeScanner gate, frontmatter-``name:``
        extraction with no fallback, slug confinement to ``[a-z0-9-]``
        inside the auto dir (a hostile ``name:`` cannot traverse out), the
        no-overwrite policy, and the ``skill_created`` signal. Returns the
        written path, or ``None`` when validation rejected the content (the
        refusal recorded in telemetry).

        ``trigger`` is the originating task/request description — used for
        telemetry context and the emitted signal, never for the filename.

        ``on_collision`` decides what an existing ``<slug>.md`` means:

        - ``"suffix"`` (default) writes ``<slug>-<unixtime>.md``, and
          ``-<unixtime>-2``, ``-3``… when that is taken too — right for
          deliberate writers (teacher escalation, record-a-skill) that must
          not lose content;
        - ``"skip"`` treats the collision as near-duplicate evidence and
          writes nothing — the auto path uses this; the timestamp-suffix
          behaviour there is how three ``debug-cron-job-failure*`` copies
          accumulated;
        - ``"refuse"`` raises :class:`SkillNameTaken` when a skill with this
          name is already served — an auto file (its own, or another serving
          the same name) or a skill served from elsewhere (a builtin, the
          user's ``skills/``) it would take over from. An accepted draft uses
          this, and :meth:`persist_or_divert` turns it into a draft;
        - ``"replace"`` archives every such file into ``auto/archive/`` (the
          way a GEPA promotion does), then writes ``<slug>.md``. The
          ``(live file, archive copy)`` pairs are appended to *replaced*.

        Every write is exclusive (``O_CREAT | O_EXCL``) or, for a replace, an
        atomic rename: no write ever lands over a file it did not mean to
        replace — two suffixed writes in one second used to share a name, and
        the second erased the first.
        """
        if on_collision not in _COLLISION_MODES:
            raise ValueError(f"on_collision must be one of {sorted(_COLLISION_MODES)}, "
                             f"not {on_collision!r}")
        # WP-X.40: the scanner gate, first, so nothing it refuses reaches any
        # later step. SkillRefiner and GEPA scan what they write; until this
        # the four writers that come through here did not.
        if not self._passes_code_scan(content, trigger=trigger):
            return None

        # PR #20: derive the slug from the LLM's frontmatter ``name:``, not
        # from the raw user message. The pre-PR-#20 path slugified
        # ``task_description``, which produced filenames like
        # ``<long-run-on-user-message-truncated-mid-word>-.md`` (trailing
        # dash from the strip-before-truncate bug in _slugify) even though
        # the LLM was correctly emitting a clean ``name: <kebab-case>``
        # in the frontmatter.
        #
        # If the LLM output lacks a usable ``name:``, we DO NOT fall back to
        # slugifying ``trigger`` — that's the bug. Skip the write and record
        # the failure so it surfaces in /health verbose.
        name = self._extract_name(content)
        if not name:
            self._record_name_failure(
                content=content,
                task_description=trigger,
                reason="LLM output missing or empty 'name:' frontmatter field",
            )
            return None

        slug = _slugify(name)
        if not slug:
            # A name like ``"!!!"`` or ``"🚀"`` slugifies to empty. Same skip
            # path — don't write junk to disk.
            self._record_name_failure(
                content=content,
                task_description=trigger,
                reason=f"LLM 'name:' field {name!r} slugified to empty",
                name_raw=name,
            )
            return None

        base = self._auto_dir / f"{slug}.md"

        # Don't overwrite existing skills. The cheap early answer for the auto
        # path; the exclusive write below is the one that decides.
        if on_collision == "skip" and base.exists():
            log.info(
                "SkillCreator: %r already exists — skipping duplicate "
                "(near-duplicate gate)",
                base.name,
            )
            return None

        # The quality gate reads the skill the way the registry will serve it.
        _, description = _parse_skill_markdown(slug, content.strip())
        if _MALFORMED_DESCRIPTION.match(description):
            self._record_gate({"reason": "malformed_description"})
            return None
        if on_collision == "skip" and self._near_duplicate(name, description):
            return None

        text = content.strip() + "\n"
        if on_collision == "replace":
            path = self._replace(slug, base, text, replaced)
        else:
            if on_collision == "refuse":
                clash = self._clashes(slug, base)
                served = self._served_elsewhere(slug)
                if clash or served:
                    raise SkillNameTaken(name, clash, served)
            path = self._write_new(slug, base, text, suffix=on_collision == "suffix")
            if path is None:
                # Taken between the check and the write: another writer won.
                if on_collision == "refuse":
                    raise SkillNameTaken(name, [base])
                log.info("SkillCreator: %r appeared before it was written — skipping",
                         base.name)
                return None
        log.info("SkillCreator: created skill at %s", path)

        # Sprint S1 Stream 2: emit skill_created so the Telegram gateway,
        # Beacon WebSocket, and any future subscribers see the event.
        await self._emit_created_signal(
            skill_path=path,
            task_description=trigger,
            content=content,
        )
        return path

    def _near_duplicate(self, name: str, description: str) -> bool:
        """True (and recorded) when an existing served skill is this close."""
        checker = self._similarity
        if checker is None:
            from prometheus.skills.similarity import default_checker

            checker = self._similarity = default_checker()
        try:
            if not checker.available:  # type: ignore[attr-defined]
                if not self._warned_unavailable:
                    self._warned_unavailable = True
                    log.warning(
                        "SkillCreator: near-duplicate check skipped — %s",
                        checker.unavailable_reason,  # type: ignore[attr-defined]
                    )
                return False
            hit = checker.nearest(  # type: ignore[attr-defined]
                skill_text(name, description), self._catalog())
        except Exception:
            log.warning("SkillCreator: near-duplicate check failed — skill kept",
                        exc_info=True)
            return False
        if hit is None or hit[0] < self._dedupe_threshold:
            return False
        score, nearest = hit
        self._record_gate({"reason": "near_duplicate", "nearest": nearest,
                           "score": round(float(score), 4),
                           "threshold": self._dedupe_threshold})
        return True

    def _write_new(self, slug: str, base: Path, text: str, *, suffix: bool) -> Path | None:
        """Create *base*, or (``suffix``) the first free ``<slug>-<unixtime>[-N].md``.

        Exclusive at every name: returns None only when *base* is taken and no
        suffix was asked for.
        """
        try:
            create_exclusive(base, text)
            return base
        except FileExistsError:
            if not suffix:
                return None
        stamp = int(time.time())
        for n in range(1, 1000):
            path = self._auto_dir / f"{slug}-{stamp}{'' if n == 1 else f'-{n}'}.md"
            try:
                create_exclusive(path, text)
                return path
            except FileExistsError:
                continue
        raise FileExistsError(f"no free name for {slug} in {self._auto_dir}")

    def _clashes(self, slug: str, base: Path) -> list[Path]:
        """The live auto skill files a skill with this slug would sit beside.

        The file at its own path, and any other auto skill whose served name
        slugifies the same: the registry serves one of them and hides the
        rest, so a second file under the same name is either dead on arrival
        or shadows the first. SkillRefiner's ``.bak-`` backups are history,
        not live skills.
        """
        out = [base] if base.exists() else []
        for path in sorted(self._auto_dir.glob("*.md")):
            if path == base or ".bak-" in path.name:
                continue
            try:
                served, _ = _parse_skill_markdown(path.stem, path.read_text(encoding="utf-8"))
            except OSError:
                continue
            if _slugify(served) == slug:
                out.append(path)
        return out

    def _served_elsewhere(self, slug: str) -> list[tuple[str, Path]]:
        """Skills the registry serves from outside this auto dir under the same name.

        A package builtin or one of the user's own ``skills/``: the registry
        registers auto skills after them, so an auto skill of that name would
        silently take its place.
        """
        try:
            registry = load_skill_registry()
        except Exception:
            log.warning("SkillCreator: skill registry unreadable — served names unchecked",
                        exc_info=True)
            return []
        auto = self._auto_dir.resolve()
        out: list[tuple[str, Path]] = []
        for served in registry.list_skills():
            if not served.path or _slugify(served.name) != slug:
                continue
            path = Path(served.path)
            if path.resolve().parent == auto:
                continue  # this auto dir's own files are _clashes' business
            out.append((served.source or "unknown", path))
        return out

    async def persist_or_divert(
        self,
        content: str,
        *,
        trigger: str,
        source: str,
    ) -> Path | DivertedToDraft | None:
        """Write the skill, or stage it for a person when its name is already served.

        For the deliberate writers that used to add a suffixed copy — teacher
        escalation, record-a-skill. A copy under a taken name was hidden (the
        registry serves one file per name), or took over from a builtin or a
        user's skill; neither is a machine's call. It is never a replace: the
        skill goes to ``skills/drafts/`` and the accept flow (409, replace or
        rename) applies. Validation refusals (the scanner, a bad ``name:``)
        still return None and stage nothing.
        """
        try:
            return await self.persist_skill_content(content, trigger=trigger, on_collision="refuse")
        except SkillNameTaken as taken:
            return self._divert(content, trigger=trigger, source=source, taken=taken)

    def _divert(
        self,
        content: str,
        *,
        trigger: str,
        source: str,
        taken: SkillNameTaken,
    ) -> DivertedToDraft | None:
        served_by = taken.where()
        what = (trigger or "")[:200]
        try:
            store = self._drafts
            if store is None:
                from prometheus.learning.skill_drafts import SkillDraftStore

                store = self._drafts = SkillDraftStore()
            sidecar = store.create(  # type: ignore[attr-defined]
                content, source=source,
                provenance={"reason": "name_already_served", "served_by": served_by,
                            "trigger": what},
            )
        except Exception as exc:
            log.exception("SkillCreator: could not stage %r as a draft — nothing was written",
                          taken.name)
            self._record_diversion("failed", {
                "skill": taken.name, "source": source, "served_by": served_by,
                "trigger": what, "error": f"{type(exc).__name__}: {exc}"[:300],
            })
            return None
        draft_id = str(sidecar["draft_id"])
        log.info("SkillCreator: a skill named %r is already served (%s) — staged as draft %s "
                 "for review instead of written", taken.name, ", ".join(served_by), draft_id)
        self._record_diversion("success", {
            "skill": taken.name, "draft_id": draft_id, "source": source,
            "served_by": served_by, "trigger": what,
        })
        return DivertedToDraft(draft_id=draft_id, skill_name=taken.name, served_by=served_by)

    def _record_diversion(self, outcome: str, summary: dict[str, Any]) -> None:
        if self._telemetry is None:
            return
        try:
            self._telemetry.record_run(  # type: ignore[attr-defined]
                "skill_creator", "divert_to_draft", outcome, summary=summary)
        except Exception:
            log.debug("SkillCreator: diversion telemetry failed", exc_info=True)

    def _replace(
        self,
        slug: str,
        base: Path,
        text: str,
        replaced: list[tuple[Path, Path]] | None,
    ) -> Path:
        """Archive every clashing live file, then write *base* — a promotion's order.

        Each file is copied into ``auto/archive/`` first (exclusive names), so
        nothing is lost if a later step fails; *base* is then replaced in one
        rename, and the other clashing files leave ``auto/`` (their copies
        stay in the archive), so exactly one file serves the name.
        """
        archive_dir = self._auto_dir / "archive"
        pairs = [(live, archive_copy(live, archive_dir)) for live in self._clashes(slug, base)]
        atomic_write(base, text)
        for live, _ in pairs:
            if live != base:
                live.unlink(missing_ok=True)
        for live, copy in pairs:
            log.info("SkillCreator: replaced %s (the previous version is archive/%s)",
                     live.name, copy.name)
        if replaced is not None:
            replaced.extend(pairs)
        return base

    def _passes_code_scan(self, content: str, *, trigger: str) -> bool:
        """True when ``DangerousCodeScanner`` lets *content* through.

        The same call SkillRefiner and GEPA make: ``scan_markdown_content``,
        which reads the Python code blocks. SUSPICIOUS findings pass, as they
        do there. A DANGEROUS verdict is refused with a WARNING naming the
        trigger and the scanner's reasons, and a ``subsystem_runs`` row
        (``skill_creator``/``code_scan``). A scanner that fails refuses too
        (fail safe, as in SkillRefiner and GEPA): unscanned content is never
        written.
        """
        what = (trigger or "")[:200]
        try:
            from prometheus.security.code_scanner import DangerousCodeScanner

            scan = DangerousCodeScanner().scan_markdown_content(content)
        except Exception as exc:
            log.exception(
                "SkillCreator: DangerousCodeScanner failed — refusing to write "
                "the skill for %r", what,
            )
            self._record_scan_refusal("failed", {
                "reason": "scanner_failed", "trigger": what,
                "error": f"{type(exc).__name__}: {exc}"[:300],
            })
            return False
        if not scan.is_dangerous:
            return True
        findings = [f"{f.rule} (line {f.line}): {f.detail}"
                    for f in scan.findings if f.severity == "dangerous"][:10]
        log.warning(
            "SkillCreator: refusing to write the skill for %r — it contains "
            "dangerous code: %s", what, "; ".join(findings),
        )
        self._record_scan_refusal("skipped", {
            "reason": "dangerous_code", "trigger": what, "findings": findings,
        })
        return False

    def _record_scan_refusal(self, outcome: str, summary: dict[str, Any]) -> None:
        if self._telemetry is None:
            return
        try:
            self._telemetry.record_run(  # type: ignore[attr-defined]
                "skill_creator", "code_scan", outcome, summary=summary)
        except Exception:
            log.debug("SkillCreator: code-scan telemetry failed", exc_info=True)

    def _record_gate(self, summary: dict[str, Any]) -> None:
        log.info("SkillCreator: skill rejected by the quality gate — %s", summary)
        if self._telemetry is None:
            return
        try:
            self._telemetry.record_run(  # type: ignore[attr-defined]
                "skill_creator", "quality_gate", "skipped", summary=summary)
        except Exception:
            log.debug("SkillCreator: quality-gate telemetry failed", exc_info=True)

    async def _emit_created_signal(
        self,
        *,
        skill_path: Path,
        task_description: str,
        content: str,
    ) -> None:
        if self._signal_bus is None:
            return
        try:
            from prometheus.sentinel.signals import ActivitySignal

            summary = self._extract_description(content) or skill_path.stem
            await self._signal_bus.emit(ActivitySignal(
                kind="skill_created",
                payload={
                    "skill_name": skill_path.stem,
                    "skill_path": str(skill_path),
                    "trigger_task": task_description[:200],
                    "summary": summary[:200],
                },
                source="skill_creator",
            ))
        except Exception:
            log.debug("SkillCreator: signal emission failed", exc_info=True)

    @staticmethod
    def _parse_skip(content: str) -> str | None:
        """Return the model's decline reason, or ``None`` when content is a skill.

        Only the first non-empty line is consulted, so a skill that merely
        MENTIONS "skip" in its body never trips this. Lenient on the exact
        shape ("SKIP", "SKIP: reason", "SKIPPING: reason") — a decline that
        fell through would only die later in ``name:`` extraction as a
        silent_failure, so leniency converts noise into clean skips.
        """
        stripped = content.strip()
        if not stripped:
            return None
        first = stripped.splitlines()[0].strip()
        if first.upper().startswith("SKIP"):
            return first.split(":", 1)[1].strip() if ":" in first else ""
        return None

    def _existing_skills_listing(self, *, cap: int = 100) -> str:
        """One ``- name: description`` line per existing auto-skill.

        Auto-dir only, deliberately: the duplication pressure this feeds
        (Stage 1's already-covered check) is auto-vs-auto — three
        near-identical release-check skills landed in two minutes on
        2026-08-03 — while builtin/user skills are human-curated. Failure
        here must never block generation; worst case the model just does
        not see the list.
        """
        lines: list[str] = []
        try:
            for path in sorted(self._auto_dir.glob("*.md"))[:cap]:
                try:
                    text = path.read_text(encoding="utf-8")
                except OSError:
                    continue
                name = self._extract_name(text) or path.stem
                desc = self._extract_description(text)
                lines.append(f"- {name}: {desc[:90]}" if desc else f"- {name}")
        except Exception:
            log.debug("SkillCreator: existing-skill listing failed", exc_info=True)
            return ""
        return "\n".join(lines)

    @staticmethod
    def _extract_description(content: str) -> str:
        """Pull `description:` from frontmatter, falling back to first body line."""
        in_fm = False
        first_body_line = ""
        for raw in content.splitlines():
            line = raw.strip()
            if line == "---":
                in_fm = not in_fm
                continue
            if in_fm:
                if line.startswith("description:"):
                    return line.split(":", 1)[1].strip().strip("'\"")
            else:
                if line and not line.startswith("#") and not first_body_line:
                    first_body_line = line
        return first_body_line

    @staticmethod
    def _extract_name(content: str) -> str | None:
        """Pull ``name:`` from YAML frontmatter. Returns ``None`` when absent or empty.

        Unlike :meth:`_extract_description`, this method has NO fallback —
        a missing or empty ``name`` is a hard failure (the LLM produced
        an unusable response) and the caller should skip writing rather
        than guess a name from elsewhere.
        """
        in_fm = False
        for raw in content.splitlines():
            line = raw.strip()
            if line == "---":
                in_fm = not in_fm
                continue
            if in_fm and line.startswith("name:"):
                value = line.split(":", 1)[1].strip().strip("'\"")
                return value or None
        return None

    def _record_name_failure(
        self,
        *,
        content: str,
        task_description: str,
        reason: str,
        name_raw: str | None = None,
    ) -> None:
        """Surface a missing-or-unusable ``name:`` to telemetry + logs.

        Constructs a :class:`SkillNameExtractionError` (never raised — only
        passed to ``telemetry.record_silent_failure`` for the ``exc=`` field)
        so the failure mode is queryable by exception type.
        """
        log.warning("SkillCreator: %s — skipping skill write", reason)
        if self._telemetry is None:
            return
        try:
            ctx: dict[str, Any] = {
                "content_preview": content[:200],
                "task_description": task_description[:200],
            }
            if name_raw is not None:
                ctx["name_raw"] = name_raw
            self._telemetry.record_silent_failure(
                subsystem="skill_creator",
                operation="extract_name",
                exc=SkillNameExtractionError(reason),
                context=ctx,
            )
        except Exception:
            log.warning(
                "SkillCreator: failed to record silent_failure for name "
                "extraction (best-effort)",
                exc_info=True,
            )

    async def _call_model(self, prompt: str) -> str | None:
        """Invoke the model via LLMCallEnvelope. Returns None on failure.

        Thin wrapper around the shared envelope so future _call_model
        bugs (ed8f1a6-shaped or otherwise) surface in
        telemetry.silent_failures instead of being silently swallowed.
        """
        return await self._envelope.call(
            provider=self._provider,
            model=self._model,
            prompt=prompt,
            max_tokens=1024,
            operation="generate_skill",
        )

    @staticmethod
    def _format_trace(trace: list[dict[str, Any]]) -> str:
        """Format a tool trace into readable text, marking failed calls.

        Each call shows what it was given (``learning/trace_format``):
        redacted, then cut. Stage 0 skips any-error traces outright, so
        ``[ERROR]`` only reaches a prompt if that check is ever relaxed —
        the marker keeps Stage 1 honest independently of Stage 0's
        configuration.
        """
        return format_trace(trace, mark_errors=True)
