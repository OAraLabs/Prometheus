"""SkillRefiner — compare actual tool traces to skill steps and refine.

After a task uses a skill, compare what actually happened to what the
skill prescribed. If the deviation led to a better outcome, update the skill.

"Uses" means the task LOADED it: the post-task hook reads the task's own
trace for a successful ``skill`` call and refines only that auto skill, and
only when the task completed. The hook it replaces (``maybe_refine_recent``)
took the most recently modified auto skill after ANY task with 3+ tool calls
— 208 refinement calls on the mini, none after a real load
(docs/audits/SKILL-USAGE.md §6).

Usage (direct):
    refiner = SkillRefiner(provider)
    updated = await refiner.maybe_refine(skill_path, tool_trace, outcome)

Usage (post-task hook on AgentLoop):
    refiner = SkillRefiner.from_config(provider)
    agent_loop.add_post_task_hook(refiner.maybe_refine_loaded)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prometheus.config.paths import get_config_dir

if TYPE_CHECKING:
    from prometheus.providers.base import ModelProvider

log = logging.getLogger(__name__)

_AUTO_SKILLS_DIR_NAME = "skills/auto"


def _get_auto_skills_dir() -> Path:
    path = get_config_dir() / _AUTO_SKILLS_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path

_REFINEMENT_PROMPT = """\
You are a skill refinement engine. A skill was used to guide a task, but the
actual execution deviated from the prescribed steps. Analyze whether the
deviation improved the outcome and, if so, update the skill.

Current skill content:
```
{skill_content}
```

Actual tool trace (what happened):
{trace}

Outcome: {outcome}

Rules:
- If the deviation was beneficial, update the skill steps to match.
- If the deviation was neutral or harmful, keep the original steps.
- Preserve the YAML frontmatter (name, description).
- Keep the same markdown structure.
- Output the FULL updated SKILL.md content, or output "NO_CHANGE" if no update needed.
"""


class SkillRefiner:
    """Refine skills based on actual execution traces.

    Args:
        provider: ModelProvider for refinement analysis.
        model: Model name for the refinement call.
        auto_dir: Override the auto-skills directory (used by ``maybe_refine_recent``).
        min_tool_calls: Minimum tool calls in the trace before considering refinement.
    """

    def __init__(
        self,
        provider: ModelProvider,
        *,
        model: str = "default",
        auto_dir: Path | None = None,
        min_tool_calls: int = 3,
        signal_bus: object | None = None,
        telemetry: object | None = None,
    ) -> None:
        from prometheus.learning.llm_envelope import LLMCallEnvelope

        self._provider = provider
        self._model = model
        self._auto_dir = auto_dir or _get_auto_skills_dir()
        self._min_tool_calls = min_tool_calls
        # Sprint S1: SignalBus wired by daemon.py after construction.
        self._signal_bus = signal_bus
        # Sprint S4 A1: shared LLMCallEnvelope. on_failure="return_none" so
        # `maybe_refine`'s existing "returns False on failure" contract still
        # holds — the caller checks `if response is None`.
        self._telemetry = telemetry
        self._envelope = LLMCallEnvelope(
            subsystem="skill_refiner",
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
    ) -> SkillRefiner | None:
        """Build from prometheus.yaml learning section.

        Returns ``None`` if ``learning.skill_refinement_enabled`` is False
        (so callers can skip wiring the hook entirely).
        """
        import yaml

        if config_path is None:
            from prometheus.config.defaults import resolve_config_path
            config_path = str(resolve_config_path())

        # Narrow the catch — see SkillCreator.from_config for the rationale
        # (Tier-1 hotfix from docs/audits/SILENT-FAILURE-AUDIT.md). Any
        # exception type other than I/O or YAML-parse should propagate.
        try:
            with open(Path(config_path).expanduser()) as fh:
                data = yaml.safe_load(fh) or {}
            learning = data.get("learning", {}) or {}
        except (OSError, yaml.YAMLError) as exc:
            log.warning(
                "SkillRefiner.from_config: failed to load %s (%s: %s); "
                "treating learning config as empty",
                config_path, type(exc).__name__, exc,
            )
            learning = {}

        if not learning.get("skill_refinement_enabled", False):
            return None

        model = learning.get("skill_refiner_model", "default")
        return cls(provider, model=model, telemetry=telemetry)

    async def maybe_refine_loaded(
        self,
        task_description: str,
        tool_trace: list[dict[str, Any]],
        final_text: str = "",
    ) -> bool:
        """Post-task-hook entry point: refine each auto skill this task loaded.

        A skill is refined only when all of these hold:

        - the trace has a successful ``skill`` call that loaded it, and the
          skill resolves to a file in the auto dir (user and builtin skills
          are curated, never rewritten here);
        - the task completed: no tool call failed after that load, and the
          turn ended with a reply;
        - the trace has at least ``min_tool_calls`` calls (the old floor).

        ``tool_trace`` entries carry ``tool_input`` (AgentLoop.run_async).
        Each loaded skill is refined at most once per task. Returns True when
        any skill was updated.
        """
        if len(tool_trace) < self._min_tool_calls:
            return False
        if not (final_text or "").strip():
            return False
        loaded: list[tuple[int, str]] = []
        for i, call in enumerate(tool_trace):
            if call.get("tool_name") != "skill" or call.get("is_error"):
                continue
            name = (call.get("tool_input") or {}).get("name")
            if isinstance(name, str) and name.strip():
                loaded.append((i, name.strip()))
        refined = False
        seen: set[Path] = set()
        for i, name in loaded:
            if any(later.get("is_error") for later in tool_trace[i + 1:]):
                log.debug("SkillRefiner: %s was loaded but the task failed after it", name)
                continue
            path = self._auto_skill_path(name)
            if path is None or path in seen:
                continue
            seen.add(path)
            try:
                refined = await self.maybe_refine(path, tool_trace, outcome=task_description) or refined
            except Exception:
                log.exception("SkillRefiner: maybe_refine failed for %s", path)
        return refined

    def _auto_skill_path(self, name: str) -> Path | None:
        """The auto-dir file serving *name*, matched the way the registry matches.

        The registry keys on the frontmatter ``name`` (falling back to the file
        stem) and ``get()`` also tries lower-case and title-case, so a load of
        "Release-Check" is the file named ``release-check``. Only the auto dir is
        searched: that is the only place this refiner may write.
        """
        if not self._auto_dir.is_dir():
            return None
        from prometheus.skills.loader import _parse_skill_markdown

        wanted = name.lower()
        for path in sorted(self._auto_dir.glob("*.md")):
            if ".bak-" in path.name:
                continue
            try:
                parsed, _ = _parse_skill_markdown(path.stem, path.read_text(encoding="utf-8"))
            except OSError:
                continue
            if parsed.lower() == wanted or path.stem.lower() == wanted:
                return path
        return None

    async def maybe_refine(
        self,
        skill_path: Path,
        tool_trace: list[dict[str, Any]],
        outcome: str,
    ) -> bool:
        """Refine a skill if the execution deviated beneficially.

        Args:
            skill_path: Path to the SKILL.md file.
            tool_trace: Actual tool calls executed.
            outcome: Description of the task outcome (success/failure + details).

        Returns:
            True if the skill was updated, False otherwise.
        """
        if not skill_path.exists():
            log.warning("SkillRefiner: skill not found at %s", skill_path)
            return False

        skill_content = skill_path.read_text(encoding="utf-8")
        trace_text = self._format_trace(tool_trace)

        prompt = _REFINEMENT_PROMPT.format(
            skill_content=skill_content,
            trace=trace_text,
            outcome=outcome,
        )

        # Envelope returns None on failure (telemetry written). The outer
        # try/except this replaces was the second-tier instance of the
        # ed8f1a6 pattern flagged HIGH-RISK in PR #2's audit.
        response = await self._call_model(prompt)
        if response is None:
            return False
        response = response.strip()
        if not response or response == "NO_CHANGE":
            log.debug("SkillRefiner: no changes needed for %s", skill_path.name)
            return False

        # Validate the response looks like a skill file
        if not response.startswith("---"):
            log.warning("SkillRefiner: response doesn't look like SKILL.md, skipping")
            return False

        # TRUST-CONTEXT: scan AI-generated content before overwriting an
        # existing skill. A self-improving loop that can rewrite skills
        # needs the scanner — see PROMETHEUS.md Security Philosophy.
        try:
            from prometheus.security.code_scanner import DangerousCodeScanner
            scanner = DangerousCodeScanner()
            scan = scanner.scan_markdown_content(
                response, file_path=str(skill_path)
            )
            if scan.is_dangerous:
                log.warning(
                    "SkillRefiner: refusing to update %s — refined content "
                    "contains dangerous code: %s",
                    skill_path.name,
                    "; ".join(f"{f.rule}:{f.detail}" for f in scan.findings),
                )
                return False
        except Exception:
            # Scanner unavailable — fail SAFE (skip the refine).
            log.exception(
                "SkillRefiner: DangerousCodeScanner failed for %s — "
                "skipping refine",
                skill_path.name,
            )
            return False

        # Back up the original
        backup = skill_path.with_suffix(f".bak-{int(time.time())}.md")
        backup.write_text(skill_content, encoding="utf-8")

        # Write the refined version
        skill_path.write_text(response + "\n", encoding="utf-8")
        log.info("SkillRefiner: updated %s (backup at %s)", skill_path.name, backup.name)

        # Sprint S1 Stream 2: surface the refinement to gateways + Beacon.
        await self._emit_refined_signal(
            skill_path=skill_path,
            outcome=outcome,
            backup=backup,
        )
        return True

    async def _emit_refined_signal(
        self,
        *,
        skill_path: Path,
        outcome: str,
        backup: Path,
    ) -> None:
        if self._signal_bus is None:
            return
        try:
            from prometheus.sentinel.signals import ActivitySignal

            await self._signal_bus.emit(ActivitySignal(
                kind="skill_refined",
                payload={
                    "skill_name": skill_path.stem,
                    "skill_path": str(skill_path),
                    "trigger_task": outcome[:200],
                    "summary": f"Refined from execution trace; backup at {backup.name}",
                    "backup_path": str(backup),
                },
                source="skill_refiner",
            ))
        except Exception:
            log.debug("SkillRefiner: signal emission failed", exc_info=True)

    async def _call_model(self, prompt: str) -> str | None:
        """Invoke the model via LLMCallEnvelope. Returns None on failure."""
        return await self._envelope.call(
            provider=self._provider,
            model=self._model,
            prompt=prompt,
            max_tokens=2048,
            operation="refine_skill",
        )

    @staticmethod
    def _format_trace(trace: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for i, call in enumerate(trace, 1):
            tool = call.get("tool_name", "unknown")
            args = call.get("arguments", {})
            result = str(call.get("result", ""))[:200]
            lines.append(f"{i}. {tool}({args}) → {result}")
        return "\n".join(lines)
