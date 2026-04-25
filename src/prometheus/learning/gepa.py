"""GEPA — Generalized Evolutionary Prompt Architecture.

Reads golden tool-call traces, generates improved skill variants, evaluates
them against the local LLM judge with constrained-decoding JSON output, and
promotes winners back to ``~/.prometheus/skills/auto/``.

Design — clean-room build using Prometheus's own provider, telemetry, and
``evals.judge.PrometheusJudge`` pattern. No external dependencies.

Boundaries:
  • Operates ONLY on auto-generated skills under ``~/.prometheus/skills/auto/``.
    Never touches manually-created skills in the parent skills/ directory.
  • Reads JSONL traces from ``~/.prometheus/trajectories/`` (written by
    ``GoldenTraceExporter``).
  • Archives the previous version before promotion to ``auto/archive/``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prometheus.config.paths import get_config_dir

if TYPE_CHECKING:
    from prometheus.providers.base import ModelProvider

log = logging.getLogger(__name__)


_VARIANT_PROMPT = """\
You are a skill improvement engine. The skill below is auto-generated and was
used in real tool-call traces. Produce a single improved version of the skill
that:
  • Preserves the YAML frontmatter (name, description) verbatim.
  • Tightens the When-to-use sentence so the trigger conditions are clearer.
  • Refines the Steps to match what *actually worked* in the traces.
  • Adds at most one Notes bullet capturing a real pitfall observed.
  • Stays under 1500 characters total.

Output ONLY the new SKILL.md content. No commentary, no markdown fences.

CURRENT SKILL:
{current_skill}

RECENT TRACES (most recent {n_traces}):
{traces}
"""


_JUDGE_SYSTEM = """\
You are a strict evaluator of agent skill documents. You will receive a skill
markdown file and a set of recent tool-call traces where this skill was used.
Score the skill's fitness on a 0.0-1.0 scale:
  1.0 — accurate, specific, would clearly help the agent reproduce the traces
  0.7 — mostly accurate, minor gaps or vague steps
  0.5 — partially accurate or generic
  0.3 — misleading or off-topic
  0.0 — doesn't match the traces at all

Respond with a JSON object: {"score": <float>, "reasoning": "<brief>"}.
"""


_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["score", "reasoning"],
    "additionalProperties": False,
}


@dataclass
class GEPAReport:
    """Result of one optimization cycle."""

    timestamp: float
    skills_evaluated: int = 0
    skills_promoted: int = 0
    skills_unchanged: int = 0
    promotions: list[dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0
    notes: str = ""

    def to_telegram_summary(self) -> str:
        """Plain-text summary suitable for ``parse_mode=None`` Telegram send."""
        if self.skills_evaluated == 0:
            base = "GEPA: no candidate skills found this cycle."
            if self.notes:
                base += f" ({self.notes})"
            return base
        lines = [
            f"GEPA cycle complete ({self.duration_seconds:.0f}s)",
            f"Evaluated: {self.skills_evaluated}, "
            f"Promoted: {self.skills_promoted}, "
            f"Unchanged: {self.skills_unchanged}",
        ]
        for promo in self.promotions[:5]:
            lines.append(
                f"  • {promo['skill']}: "
                f"{promo['old_score']:.2f} → {promo['new_score']:.2f}"
            )
        return "\n".join(lines)


class GEPAOptimizer:
    """Read golden traces, generate skill variants, evaluate, promote winners.

    Args:
        provider: ModelProvider — used to generate variants.
        judge: Optional PrometheusJudge-like object with an ``evaluate`` method.
            If None, a default ``PrometheusJudge`` is created lazily from
            ``judge_base_url``.
        judge_base_url: Used to lazily build the default judge.
        telemetry: ToolCallTelemetry instance (kept for future cycle-stat logging).
        config: ``learning`` section dict from prometheus.yaml. Recognised keys:
            ``gepa_enabled`` (bool, default False)
            ``gepa_max_skills_per_cycle`` (int, default 3)
            ``gepa_variants_per_skill`` (int, default 3)
            ``gepa_min_traces_required`` (int, default 10)
            ``gepa_judge_threshold`` (float, default 0.7)
            ``gepa_model`` (str | None, default None — falls back to provider default)
        trajectories_dir: Override for golden trace directory.
        skills_auto_dir: Override for auto skills directory.
    """

    def __init__(
        self,
        provider: ModelProvider,
        *,
        judge: object | None = None,
        judge_base_url: str | None = None,
        telemetry: object | None = None,
        config: dict[str, Any] | None = None,
        trajectories_dir: Path | None = None,
        skills_auto_dir: Path | None = None,
    ) -> None:
        self._provider = provider
        self._judge = judge
        self._judge_base_url = judge_base_url
        self._telemetry = telemetry
        cfg = config or {}
        self._enabled = bool(cfg.get("gepa_enabled", False))
        self._max_skills = int(cfg.get("gepa_max_skills_per_cycle", 3))
        self._variants = int(cfg.get("gepa_variants_per_skill", 3))
        self._min_traces = int(cfg.get("gepa_min_traces_required", 10))
        self._threshold = float(cfg.get("gepa_judge_threshold", 0.7))
        self._model = cfg.get("gepa_model") or "default"

        default_traj = get_config_dir() / "trajectories"
        default_skills = get_config_dir() / "skills" / "auto"
        self._trajectories_dir = trajectories_dir or default_traj
        self._skills_auto_dir = skills_auto_dir or default_skills
        self._archive_dir = self._skills_auto_dir / "archive"

    @classmethod
    def from_config(
        cls,
        provider: ModelProvider,
        *,
        telemetry: object | None = None,
        judge_base_url: str | None = None,
        config_path: str | None = None,
    ) -> GEPAOptimizer | None:
        """Build from prometheus.yaml. Returns None if disabled."""
        import yaml

        if config_path is None:
            from prometheus.config.defaults import DEFAULTS_PATH
            config_path = str(DEFAULTS_PATH)

        try:
            with open(Path(config_path).expanduser()) as fh:
                data = yaml.safe_load(fh) or {}
        except (OSError, Exception):
            data = {}

        learning = data.get("learning", {}) or {}
        if not learning.get("gepa_enabled", False):
            return None

        if judge_base_url is None:
            evals_cfg = data.get("evals", {}) or {}
            judge_base_url = evals_cfg.get("judge_base_url")

        return cls(
            provider,
            telemetry=telemetry,
            judge_base_url=judge_base_url,
            config=learning,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_optimization_cycle(self) -> GEPAReport:
        """One full pass: load traces → find candidates → variants → judge → promote.

        Always returns a GEPAReport. Never raises on individual skill failures
        (they're recorded in ``notes`` and the cycle moves on).
        """
        start = time.time()
        report = GEPAReport(timestamp=start)

        if not self._enabled:
            report.notes = "disabled"
            report.duration_seconds = time.time() - start
            return report

        # Lazy-create the auto-skills dir so existence-check below is meaningful.
        if not self._skills_auto_dir.exists():
            try:
                self._skills_auto_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
            report.notes = "auto/ dir empty"
            report.duration_seconds = time.time() - start
            return report

        traces = await self._load_recent_traces(limit=200)
        if len(traces) < self._min_traces:
            report.notes = f"only {len(traces)} traces (< min {self._min_traces})"
            report.duration_seconds = time.time() - start
            return report

        candidates = await self._find_candidate_skills(traces)
        if not candidates:
            report.notes = "no auto-skill matches in traces"
            report.duration_seconds = time.time() - start
            return report

        candidates = candidates[: self._max_skills]
        for skill_path in candidates:
            try:
                await self._optimize_one(skill_path, traces, report)
            except Exception:
                log.exception("GEPA: optimize_one failed for %s", skill_path)

        report.duration_seconds = time.time() - start
        return report

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _optimize_one(
        self,
        skill_path: Path,
        traces: list[dict[str, Any]],
        report: GEPAReport,
    ) -> None:
        """Generate variants, evaluate, promote if a winner beats the threshold."""
        report.skills_evaluated += 1
        current = skill_path.read_text(encoding="utf-8")
        current_score = await self._evaluate_variant(current, traces)

        variants = await self._generate_variants(skill_path, traces)
        if not variants:
            report.skills_unchanged += 1
            return

        scored: list[tuple[str, float]] = []
        for variant in variants:
            score = await self._evaluate_variant(variant, traces)
            scored.append((variant, score))

        best_variant, best_score = max(scored, key=lambda item: item[1])
        if best_score > current_score and best_score >= self._threshold:
            await self._promote_winner(skill_path, best_variant, best_score)
            report.skills_promoted += 1
            report.promotions.append({
                "skill": skill_path.stem,
                "old_score": current_score,
                "new_score": best_score,
            })
        else:
            report.skills_unchanged += 1

    async def _load_recent_traces(self, limit: int = 200) -> list[dict[str, Any]]:
        """Read the last N JSONL records from the most recent trajectories file."""
        if not self._trajectories_dir.exists():
            return []
        files = sorted(self._trajectories_dir.glob("golden_traces_*.jsonl"))
        if not files:
            return []
        try:
            text = files[-1].read_text(encoding="utf-8")
        except OSError:
            return []
        out: list[dict[str, Any]] = []
        for line in text.strip().splitlines()[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    async def _find_candidate_skills(
        self,
        traces: list[dict[str, Any]],
    ) -> list[Path]:
        """Return auto-skill files referenced as Skill-tool invocations.

        ``parsed_tool_call`` is echoed in the user content of each JSONL line
        as ``"Reference parsed call: {...json...}"``. Parse it back out and
        match the ``input.skill`` field against files in the auto/ directory.

        Reads the auto/ directory directly via ``glob`` — does NOT consult
        SkillRegistry.find_by_path() (no such method exists; SkillRegistry
        only exposes ``get(name)`` and ``list_skills()``).
        """
        skill_names: set[str] = set()
        for trace in traces:
            meta = trace.get("_meta", {}) or {}
            if meta.get("tool_name") != "Skill":
                continue
            user_content = ""
            for msg in trace.get("messages", []) or []:
                if msg.get("role") == "user":
                    user_content = msg.get("content", "") or ""
                    break
            marker = "Reference parsed call: "
            idx = user_content.find(marker)
            if idx < 0:
                continue
            payload = user_content[idx + len(marker):].strip()
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                continue
            skill_name = (parsed.get("input") or {}).get("skill")
            if skill_name:
                skill_names.add(str(skill_name))

        out: list[Path] = []
        for name in sorted(skill_names):
            candidate = self._skills_auto_dir / f"{name}.md"
            if candidate.exists():
                out.append(candidate)
        return out

    async def _generate_variants(
        self,
        skill_path: Path,
        traces: list[dict[str, Any]],
    ) -> list[str]:
        """Ask the LLM for ``self._variants`` improved versions of the skill."""
        current = skill_path.read_text(encoding="utf-8")
        relevant = [
            t for t in traces if skill_path.stem in json.dumps(t)
        ][-5:]
        if not relevant:
            relevant = traces[-5:]

        traces_text = self._format_traces(relevant)
        prompt = _VARIANT_PROMPT.format(
            current_skill=current,
            traces=traces_text,
            n_traces=len(relevant),
        )

        results: list[str] = []
        for _ in range(self._variants):
            try:
                text = await self._call_provider(prompt)
            except Exception:
                log.debug("GEPA: variant generation failed", exc_info=True)
                continue
            text = text.strip()
            if text and text.startswith("---"):
                results.append(text)
        return results

    async def _evaluate_variant(
        self,
        variant: str,
        traces: list[dict[str, Any]],
    ) -> float:
        """Score a variant via the LLM judge. Returns 0.0 on failure."""
        judge = self._get_or_build_judge()
        if judge is None:
            log.debug("GEPA: no judge available, returning 0.0")
            return 0.0
        traces_text = self._format_traces(traces[-5:])
        try:
            verdict = await judge.evaluate(
                task_input="Evaluate the skill below.",
                agent_output=variant,
                expected_behavior=(
                    "The skill must accurately describe the steps the agent "
                    "actually took in these traces and give clear guidance "
                    "for reproducing them."
                ),
                tool_trace=[{"tool_name": "trace_summary", "summary": traces_text}],
            )
        except Exception:
            log.debug("GEPA: judge evaluation failed", exc_info=True)
            return 0.0
        return float(verdict.score)

    async def _promote_winner(
        self,
        skill_path: Path,
        winner_content: str,
        score: float,
    ) -> None:
        """Archive the current version, write the winner with a provenance header."""
        self._archive_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        archive_path = self._archive_dir / f"{skill_path.stem}_{ts}.md"
        try:
            archive_path.write_text(
                skill_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        except OSError:
            log.exception("GEPA: failed to archive %s", skill_path)
            return

        header = (
            f"<!-- GEPA-optimized: {ts}, score: {score:.2f} -->\n"
            f"<!-- Previous: {archive_path.name} -->\n\n"
        )
        skill_path.write_text(header + winner_content, encoding="utf-8")
        log.info(
            "GEPA: promoted %s (score=%.2f, prev archived as %s)",
            skill_path.name, score, archive_path.name,
        )

    def _format_traces(self, traces: list[dict[str, Any]]) -> str:
        """Render traces as a compact text block for prompts."""
        lines: list[str] = []
        for i, trace in enumerate(traces, 1):
            meta = trace.get("_meta", {}) or {}
            tool = meta.get("tool_name", "?")
            user_msg = ""
            assistant_msg = ""
            for msg in trace.get("messages", []) or []:
                if msg.get("role") == "user" and not user_msg:
                    user_msg = (msg.get("content", "") or "")[:200]
                elif msg.get("role") == "assistant" and not assistant_msg:
                    assistant_msg = (msg.get("content", "") or "")[:200]
            lines.append(f"#{i} [{tool}] user={user_msg!r} assistant={assistant_msg!r}")
        return "\n".join(lines)

    def _get_or_build_judge(self) -> object | None:
        """Lazily construct the default ``PrometheusJudge`` if none was supplied."""
        if self._judge is not None:
            return self._judge
        if not self._judge_base_url:
            return None
        try:
            from prometheus.evals.judge import PrometheusJudge
            self._judge = PrometheusJudge(base_url=self._judge_base_url)
        except Exception:
            log.exception("GEPA: failed to build PrometheusJudge")
            return None
        return self._judge

    async def _call_provider(self, prompt: str) -> str:
        """Stream a single completion and return concatenated text."""
        from prometheus.engine.messages import ConversationMessage
        from prometheus.providers.base import (
            ApiMessageRequest,
            ApiTextDeltaEvent,
        )

        request = ApiMessageRequest(
            model=self._model,
            messages=[ConversationMessage.from_user_text(prompt)],
            max_tokens=2048,
        )
        text_parts: list[str] = []
        async for event in self._provider.stream_message(request):
            if isinstance(event, ApiTextDeltaEvent):
                text_parts.append(event.text)
        return "".join(text_parts)
