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

Usage:
    creator = SkillCreator(provider)
    skill_path = await creator.maybe_create(task_record, tool_trace, final_text)
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prometheus.config.paths import get_config_dir

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

Tool call trace (failed calls are marked [ERROR]):
{trace}

Existing skills (do NOT re-create these):
{existing_skills}

Output ONLY the SKILL.md content or the SKIP line. No commentary.
"""


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
    ) -> None:
        from prometheus.learning.llm_envelope import LLMCallEnvelope

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
            from prometheus.config.defaults import DEFAULTS_PATH
            config_path = str(DEFAULTS_PATH)

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
        except (OSError, yaml.YAMLError) as exc:
            log.warning(
                "SkillCreator.from_config: failed to load %s (%s: %s); "
                "using default skill_min_tool_calls=%d",
                config_path, type(exc).__name__, exc, _MIN_TOOL_CALLS,
            )
            min_calls = _MIN_TOOL_CALLS

        return cls(provider, min_tool_calls=min_calls, telemetry=telemetry)

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
    ) -> Path | None:
        """Validate and write skill markdown through the standard auto-skill path.

        This is THE write path for machine-generated skills — used by
        :meth:`maybe_create` and by teacher escalation
        (``escalation/teacher.py``), so every writer gets the same
        validation: frontmatter-``name:`` extraction with no fallback,
        slug confinement to ``[a-z0-9-]`` inside the auto dir (a hostile
        ``name:`` cannot traverse out), the no-overwrite policy, and the
        ``skill_created`` signal. Returns the written path, or ``None``
        when validation rejected the content (failure recorded in
        ``telemetry.silent_failures``).

        ``trigger`` is the originating task/request description — used for
        telemetry context and the emitted signal, never for the filename.

        ``on_collision`` decides what an existing ``<slug>.md`` means:
        ``"suffix"`` (default) writes ``<slug>-<unixtime>.md`` — right for
        deliberate writers (teacher escalation, record-a-skill, an ACCEPTed
        draft) that must not lose content. ``"skip"`` treats the collision
        as near-duplicate evidence and writes nothing — the auto path uses
        this; the timestamp-suffix behaviour there is how three
        ``debug-cron-job-failure*`` copies accumulated.
        """
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

        path = self._auto_dir / f"{slug}.md"

        # Don't overwrite existing skills
        if path.exists():
            if on_collision == "skip":
                log.info(
                    "SkillCreator: %r already exists — skipping duplicate "
                    "(near-duplicate gate)",
                    path.name,
                )
                return None
            path = self._auto_dir / f"{slug}-{int(time.time())}.md"

        path.write_text(content.strip() + "\n", encoding="utf-8")
        log.info("SkillCreator: created skill at %s", path)

        # Sprint S1 Stream 2: emit skill_created so the Telegram gateway,
        # Beacon WebSocket, and any future subscribers see the event.
        await self._emit_created_signal(
            skill_path=path,
            task_description=trigger,
            content=content,
        )
        return path

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

        Stage 0 skips any-error traces outright, so ``[ERROR]`` only
        reaches a prompt if that check is ever relaxed — the marker keeps
        Stage 1 honest independently of Stage 0's configuration.
        """
        lines: list[str] = []
        for i, call in enumerate(trace, 1):
            tool = call.get("tool_name", "unknown")
            args = call.get("arguments", {})
            result = str(call.get("result", ""))[:200]
            flag = " [ERROR]" if call.get("is_error") else ""
            lines.append(f"{i}. {tool}({args}) → {result}{flag}")
        return "\n".join(lines)
