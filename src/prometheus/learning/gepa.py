"""GEPA — measured, human-approved improvement of the skills Prometheus writes itself.

One cycle:

1. **Candidates come from real use.** An auto skill (``skills/auto/``) is a
   candidate when the model LOADED it at least ``gepa_min_loads`` times, per
   the load counter (one ``subsystem_runs`` row per successful ``skill`` call,
   #591).
2. **Evidence is the runs around those loads** — the request, the calls after
   the load and how they went, whether the run ended with a reply
   (:mod:`prometheus.learning.gepa_evidence`). The latest
   :data:`~prometheus.learning.gepa_evidence.MAX_EVIDENCE_RUNS` runs are used.
3. **The local model writes variants** from that evidence. A hosted provider
   is used only when ``learning.gepa_allow_hosted`` is ``true``; everything
   sent is redacted first.
4. **The judge scores the live skill and every variant on each run** — the
   same evidence for all. A variant is PROPOSED only if every verdict parsed,
   its mean beats the live skill's by at least ``gepa_min_margin``, the mean
   clears ``gepa_judge_threshold``, and it scores below the live skill on no
   run. An unparseable verdict never wins. The default margin, 0.1, is the
   smallest at which no formatting-only rewrite of a skill passed this rule
   against the configured judge (``docs/audits/gepa-margin/judge_noise.py``:
   0 of 30 at 0.1, 3 of 30 at 0.05), and every deliberate degradation fell
   by at least that much.
5. **A proposal is staged, never applied** (:mod:`prometheus.learning.gepa_proposals`).
   GEPA does not write ``skills/auto/``; a person promotes with
   ``oara gepa promote``.

Every cycle records one ``subsystem_runs`` row (``gepa``/``cycle``) whose
summary carries ``candidates``, ``variants``, ``judged``, ``unparseable`` and
``proposed``. ``oara gepa dry-run`` runs steps 1–2 and the gates, and reports
counts only.

History: until WP-X.33 GEPA read golden-trace exports and looked for a
``Skill`` tool, an ``input.skill`` field and a ``Reference parsed call:``
marker — none of which exist (the tool is ``skill``, its field ``name``, and
the export shape changed in #586). It found nothing, and would have written
straight into ``skills/auto/`` if it had.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, Any

from prometheus.config.paths import config_dir_path, lcm_db_path
from prometheus.learning import gepa_evidence as ev
from prometheus.learning.gepa_proposals import (
    ProposalStore,
    frontmatter_closed,
    sha256_text,
    skill_identity,
)
from prometheus.security.log_redaction import redact_secrets

if TYPE_CHECKING:
    from prometheus.providers.base import ModelProvider

log = logging.getLogger(__name__)

GEPA_SUBSYSTEM = "gepa"
GEPA_CYCLE_OPERATION = "cycle"

# PrometheusJudge.evaluate reads the first 3,000 characters of the document it
# scores (``agent_output[:3000]``). A longer document would be judged on a
# prefix, so live skills and variants longer than this are not compared.
JUDGE_DOC_CHARS = 3000

# Rounding room for the margin test: 0.7 - 0.6 is 0.0999… in floating point.
_EPS = 1e-9


_VARIANT_PROMPT = """\
You improve a skill document that an AI agent loads before doing a task. The
agent loaded the skill below in the real runs listed after it. Write one
improved version of the skill.

Rules:
- Keep the YAML frontmatter (the block between the two --- lines) exactly as it is.
- Make the steps match what worked in these runs; fix or drop steps that led to failed calls.
- Add at most one Notes bullet, for a pitfall these runs actually hit.
- Keep it general: no names, paths, data or details specific to these runs.
- Stay under 1500 characters.

Output ONLY the new skill document, starting with its --- line. No commentary, no code fences.

CURRENT SKILL:
{current_skill}

RUNS THAT LOADED IT ({n_runs}):
{runs}
"""

# The judge's own system prompt asks for a 0.0–1.0 score as JSON; these are
# its task and expectation fields. The document scored goes in agent_output.
_JUDGE_TASK = """\
An AI agent loads the skill document below before starting a task, and follows it.
Rate how well the document would guide the agent to do THIS task correctly.

The task, from a real run that loaded this skill:
{run}"""

_JUDGE_EXPECTED = (
    "A good skill document gives steps that match what worked in the run, warns "
    "about what went wrong in it, invents no steps or tools, and stays general. "
    "1.0: following it leads straight to a correct result. 0.0: it would mislead."
)


def _json_candidates(raw: str) -> list[str]:
    text = raw.strip()
    fenced = re.sub(r"```(?:json)?\s*\n?", "", text).strip()
    out = [text]
    if fenced != text:
        out.append(fenced)
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        out.append(text[start:end + 1])
    return out


def parsed_score(verdict: object) -> float | None:
    """The score the judge actually gave, or None when its answer held none.

    ``PrometheusJudge`` turns an empty answer into 0.0 and an unparseable one
    into the first number in its text, and returns both as ordinary verdicts.
    So the raw answer is read again here, strictly: a JSON object (bare, fenced
    or embedded) whose ``score`` is a number from 0 to 1. Anything else — no
    object, no ``score``, a string, a boolean, out of range — is None, and None
    never wins a comparison.

    WP-X.22 is making the judge report whether it parsed a score; once it
    lands, its verdict status replaces this re-read.
    """
    raw = getattr(verdict, "raw_response", None)
    if not isinstance(raw, str) or not raw.strip():
        return None
    for candidate in _json_candidates(raw):
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if not isinstance(parsed, dict):
            return None
        score = parsed.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            return None
        value = float(score)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            return None
        return value
    return None


def _is_local_provider(name: str | None) -> bool:
    """A provider Prometheus knows and does not class as a cloud API.

    An unknown name is NOT local: the gate below fails closed.
    """
    if not name:
        return False
    from prometheus.providers.registry import ProviderRegistry

    return name in ProviderRegistry.list_providers() and not ProviderRegistry.is_cloud(name)


def _normalized(text: str) -> str:
    return " ".join(text.split())


# ── reports ──────────────────────────────────────────────────────────


@dataclass
class Candidate:
    """An auto skill loaded often enough, with the runs it will be judged on."""

    skill: ev.AutoSkill
    loads: int
    runs: list[ev.LoadRun]
    live_sha256: str


@dataclass
class CyclePlan:
    """What a cycle would work on, before any model is called."""

    auto_skills: int = 0
    load_rows: int = 0
    auto_loads: int = 0
    skills_loaded: int = 0
    below_min_loads: int = 0
    candidates: list[Candidate] = field(default_factory=list)
    evidence_short: int = 0
    pending: int = 0
    too_long: int = 0
    ready: list[Candidate] = field(default_factory=list)
    counter_error: str = ""


@dataclass
class GEPAReport:
    """Result of one cycle. The counts are the ones its ``subsystem_runs`` row carries."""

    timestamp: float
    candidates: int = 0
    variants: int = 0
    judged: int = 0
    unparseable: int = 0
    proposed: int = 0
    live_unparseable: int = 0
    variants_rejected: int = 0
    unsafe: int = 0
    errors: int = 0
    proposals: list[dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0
    notes: str = ""

    def to_telegram_summary(self) -> str:
        """Plain-text summary suitable for ``parse_mode=None`` Telegram send."""
        if self.judged == 0:
            base = "GEPA: nothing to judge this cycle."
            if self.notes:
                base += f" ({self.notes})"
            return base
        lines = [
            f"GEPA cycle complete ({self.duration_seconds:.0f}s)",
            f"Candidates: {self.candidates}, variants: {self.variants}, "
            f"judged: {self.judged} ({self.unparseable} unparseable), "
            f"proposed: {self.proposed}",
        ]
        for promo in self.proposals[:5]:
            lines.append(
                f"  • {promo['skill']}: {promo['live_mean']:.2f} → "
                f"{promo['variant_mean']:.2f} ({promo['id']})"
            )
        if self.proposed:
            lines.append("No skill changed. Review with `oara gepa proposals`.")
        return "\n".join(lines)


@dataclass
class DryRunReport:
    """What a cycle would do now, as counts. Nothing is generated, judged or written."""

    enabled: bool
    generator: str
    generator_allowed: bool
    generator_reason: str
    judge_configured: bool
    judge_pinned: bool
    min_loads: int
    max_skills: int
    variants_per_skill: int
    min_margin: float
    threshold: float
    pending_proposals: int
    plan: CyclePlan

    @property
    def would_optimise(self) -> list[Candidate]:
        if not (self.generator_allowed and self.judge_configured) or self.plan.counter_error:
            return []
        return self.plan.ready[: self.max_skills]

    @property
    def generation_calls(self) -> int:
        return len(self.would_optimise) * self.variants_per_skill

    @property
    def judge_calls_at_most(self) -> int:
        return sum(len(c.runs) * (1 + self.variants_per_skill) for c in self.would_optimise)

    def to_dict(self) -> dict[str, Any]:
        p = self.plan
        return {
            "enabled": self.enabled,
            "generator": self.generator,
            "generator_allowed": self.generator_allowed,
            "judge_configured": self.judge_configured,
            "judge_pinned": self.judge_pinned,
            "auto_skills": p.auto_skills,
            "load_rows": p.load_rows,
            "auto_loads": p.auto_loads,
            "skills_loaded": p.skills_loaded,
            "below_min_loads": p.below_min_loads,
            "candidates": len(p.candidates),
            "evidence_short": p.evidence_short,
            "pending_for_live_version": p.pending,
            "too_long_for_judge": p.too_long,
            "ready": len(p.ready),
            "would_optimise": len(self.would_optimise),
            "generation_calls": self.generation_calls,
            "judge_calls_at_most": self.judge_calls_at_most,
            "pending_proposals": self.pending_proposals,
            "min_loads": self.min_loads,
            "counter_error": p.counter_error,
        }

    def to_text(self) -> str:
        p = self.plan
        yes = {True: "yes", False: "no"}
        gen = f"{self.generator} — " + ("allowed" if self.generator_allowed
                                         else f"REFUSED: {self.generator_reason}")
        judge = ("configured, " + ("pinned" if self.judge_pinned else "NOT pinned")
                 if self.judge_configured else "not configured (evals.judge_base_url)")
        lines = [
            "GEPA dry run — nothing is generated, judged or written.",
            f"  enabled (learning.gepa_enabled): {yes[self.enabled]}",
            f"  variant generator: {gen}",
            f"  judge: {judge}",
            f"  auto skills: {p.auto_skills}",
        ]
        if p.counter_error:
            lines.append(f"  load counter: UNREADABLE — {p.counter_error}")
        else:
            lines += [
                f"  load counter: {p.load_rows} successful loads, {p.auto_loads} of them from auto skills",
                f"  auto skills loaded at least once: {p.skills_loaded} "
                f"({p.below_min_loads} below the minimum of {self.min_loads} loads)",
                f"  candidates (≥{self.min_loads} loads): {len(p.candidates)}",
                f"    fewer than {self.min_loads} runs with readable evidence: {p.evidence_short}",
                f"    a proposal for the live version is already pending: {p.pending}",
                f"    live skill longer than the judge reads ({JUDGE_DOC_CHARS} chars): {p.too_long}",
                f"  ready: {len(p.ready)}; would optimise this cycle: {len(self.would_optimise)} "
                f"(at most {self.max_skills})",
                f"  would make: {self.generation_calls} generation calls, "
                f"at most {self.judge_calls_at_most} judge calls",
            ]
        lines.append(f"  proposals already pending review: {self.pending_proposals}")
        lines.append(
            f"  rule: propose a variant only if every verdict parses, its mean beats the live "
            f"skill's by ≥{self.min_margin:.2f}, reaches ≥{self.threshold:.2f}, and it is worse "
            "on no run"
        )
        return "\n".join(lines)


# ── the optimizer ────────────────────────────────────────────────────


class GEPAOptimizer:
    """Find skills in real use, generate variants, judge them, stage the winners.

    Args:
        provider: ModelProvider that generates variants.
        provider_name: its registry name (``model.provider``). Decides whether
            the generator is local; an unknown name is treated as not local.
        judge: Optional PrometheusJudge-like object with an ``evaluate`` method.
            If None, a default ``PrometheusJudge`` is created lazily from
            ``judge_base_url`` / ``judge_model``.
        telemetry: ToolCallTelemetry. Its database holds the load counter and
            the calls; the cycle's ``subsystem_runs`` row is written through it.
        config: ``learning`` section dict from prometheus.yaml. Recognised keys:
            ``gepa_enabled`` (bool, default False)
            ``gepa_max_skills_per_cycle`` (int, default 3)
            ``gepa_variants_per_skill`` (int, default 3)
            ``gepa_min_loads`` (int, default 3)
            ``gepa_min_margin`` (float, default 0.1)
            ``gepa_judge_threshold`` (float, default 0.7)
            ``gepa_allow_hosted`` (bool, default False)
            ``gepa_model`` (str, default ``default`` — the provider's model)
        skills_auto_dir / proposals_dir / telemetry_db / lcm_db: overrides.
    """

    def __init__(
        self,
        provider: ModelProvider | None,
        *,
        provider_name: str | None = None,
        judge: object | None = None,
        judge_base_url: str | None = None,
        judge_model: str | None = None,
        telemetry: object | None = None,
        config: dict[str, Any] | None = None,
        skills_auto_dir: Path | None = None,
        proposals_dir: Path | None = None,
        telemetry_db: Path | None = None,
        lcm_db: Path | None = None,
    ) -> None:
        self._provider = provider
        self._provider_name = provider_name
        self._judge = judge
        self._judge_base_url = judge_base_url
        self._judge_model = judge_model
        self._telemetry = telemetry
        cfg = config or {}
        self._enabled = bool(cfg.get("gepa_enabled", False))
        self._max_skills = max(1, int(cfg.get("gepa_max_skills_per_cycle", 3)))
        self._variants = max(1, int(cfg.get("gepa_variants_per_skill", 3)))
        self._min_loads = max(1, int(cfg.get("gepa_min_loads", 3)))
        self._margin = min(1.0, max(0.0, float(cfg.get("gepa_min_margin", 0.1))))
        self._threshold = float(cfg.get("gepa_judge_threshold", 0.7))
        # Only a real YAML true opts in: the string "false" is truthy.
        self._allow_hosted = cfg.get("gepa_allow_hosted", False) is True
        self._model = cfg.get("gepa_model") or "default"

        base = config_dir_path()
        self._skills_auto_dir = Path(skills_auto_dir) if skills_auto_dir else base / "skills" / "auto"
        self._store = ProposalStore(proposals_dir=proposals_dir, skills_auto_dir=self._skills_auto_dir)
        if telemetry_db is None:
            telemetry_db = getattr(telemetry, "db_path", None) or base / "telemetry.db"
        self._telemetry_db = Path(telemetry_db)
        # Where the stores are, not a request to create them: a dry run on a
        # box with no conversation store yet must leave nothing behind.
        self._lcm_db = Path(lcm_db) if lcm_db else lcm_db_path()

    @classmethod
    def from_config(
        cls,
        provider: ModelProvider,
        *,
        telemetry: object | None = None,
        judge_base_url: str | None = None,
        judge_model: str | None = None,
        config_path: str | None = None,
    ) -> GEPAOptimizer | None:
        """Build from prometheus.yaml. Returns None if disabled."""
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
        except (OSError, yaml.YAMLError) as exc:
            log.warning(
                "GEPAOptimizer.from_config: failed to load %s (%s: %s); "
                "treating config as empty",
                config_path, type(exc).__name__, exc,
            )
            data = {}

        learning = data.get("learning", {}) or {}
        if not learning.get("gepa_enabled", False):
            return None
        return cls.from_config_dict(
            data, provider=provider, telemetry=telemetry,
            judge_base_url=judge_base_url, judge_model=judge_model,
        )

    @classmethod
    def from_config_dict(
        cls,
        data: dict[str, Any],
        *,
        provider: ModelProvider | None = None,
        telemetry: object | None = None,
        judge_base_url: str | None = None,
        judge_model: str | None = None,
        skills_auto_dir: Path | None = None,
        proposals_dir: Path | None = None,
        telemetry_db: Path | None = None,
        lcm_db: Path | None = None,
    ) -> GEPAOptimizer:
        """Build from a loaded config dict, enabled or not (the dry run needs both)."""
        learning = data.get("learning", {}) or {}
        evals_cfg = data.get("evals", {}) or {}
        model_cfg = data.get("model", {}) or {}
        if judge_base_url is None:
            judge_base_url = evals_cfg.get("judge_base_url")
        # evals.judge_model was declared in config but never read here, so the
        # judge fell through to _detect_model() and graded with whatever the
        # judge endpoint happened to have loaded. That is the non-determinism
        # the pin exists to remove — and when judge_base_url coincides with the
        # main model's base_url it is outright self-judging.
        if judge_model is None:
            judge_model = evals_cfg.get("judge_model")
        return cls(
            provider,
            # ProviderRegistry.create's own default, so a config without the
            # key names the provider the daemon would actually build.
            provider_name=model_cfg.get("provider", "llama_cpp"),
            telemetry=telemetry,
            judge_base_url=judge_base_url,
            judge_model=judge_model,
            config=learning,
            skills_auto_dir=skills_auto_dir,
            proposals_dir=proposals_dir,
            telemetry_db=telemetry_db,
            lcm_db=lcm_db,
        )

    # ------------------------------------------------------------------
    # Gates
    # ------------------------------------------------------------------

    def generator_status(self) -> tuple[bool, str, str]:
        """``(allowed, label, reason)`` for the provider that would write variants.

        Local by default: variants are written from real requests and calls,
        so a hosted provider is used only when ``learning.gepa_allow_hosted``
        is explicitly true. An unknown provider name counts as hosted.
        """
        name = self._provider_name
        local = _is_local_provider(name)
        label = f"{name or 'unknown'} ({'local' if local else 'hosted'})"
        if local or self._allow_hosted:
            return True, label, ""
        return False, label, (
            f"{name or 'the provider'} is not a local provider; GEPA sends skills and "
            "evidence to a hosted model only when learning.gepa_allow_hosted is true"
        )

    def _refusal(self) -> str:
        """Why a cycle must not call any model, or ``""`` when it may."""
        allowed, _, reason = self.generator_status()
        if not allowed:
            return f"variant generator refused: {reason}"
        if self._provider is None:
            return "no provider to generate variants"
        if self._get_or_build_judge() is None:
            return "no judge configured (evals.judge_base_url)"
        return ""

    # ------------------------------------------------------------------
    # Planning (no model calls, no writes)
    # ------------------------------------------------------------------

    def plan(self) -> CyclePlan:
        """Candidates, their evidence and the skips — read-only."""
        plan = CyclePlan()
        skills = ev.auto_skills(self._skills_auto_dir)
        plan.auto_skills = len(skills)
        try:
            tel = ev.open_readonly(self._telemetry_db)
        except Exception as exc:
            plan.counter_error = f"telemetry database not readable ({type(exc).__name__})"
            return plan
        lcm = None
        try:
            try:
                events = ev.load_events(tel)
            except Exception as exc:
                plan.counter_error = f"load counter not readable ({type(exc).__name__})"
                return plan
            plan.load_rows = len(events)
            plan.auto_loads = sum(1 for e in events if e.source == "auto")
            try:
                lcm = ev.open_readonly(self._lcm_db)
            except Exception:
                log.warning("GEPA: conversation store not readable — requests and "
                            "replies will be unknown", exc_info=True)
            loaded: list[tuple[ev.AutoSkill, list[ev.LoadEvent]]] = []
            for skill in skills:
                loads = ev.loads_of(skill, events)
                if not loads:
                    continue
                plan.skills_loaded += 1
                if len(loads) < self._min_loads:
                    plan.below_min_loads += 1
                    continue
                loaded.append((skill, loads))
            # Most-used first, so the per-cycle cap spends itself on the
            # skills that steer the most runs.
            loaded.sort(key=lambda item: (len(item[1]), item[1][-1].timestamp), reverse=True)
            for skill, loads in loaded:
                live_sha = sha256_text(skill.text)
                cand = Candidate(skill=skill, loads=len(loads), runs=[], live_sha256=live_sha)
                plan.candidates.append(cand)
                if len(skill.text) > JUDGE_DOC_CHARS:
                    plan.too_long += 1
                    continue
                if self._store.has_pending(skill.path.name, live_sha):
                    plan.pending += 1
                    continue
                for event in reversed(loads):
                    if len(cand.runs) >= ev.MAX_EVIDENCE_RUNS:
                        break
                    try:
                        run = ev.run_for_load(tel, lcm, event)
                    except Exception:
                        log.warning("GEPA: could not read the run of one load of %s",
                                    skill.stem, exc_info=True)
                        continue
                    if run is not None:
                        cand.runs.append(run)
                if len(cand.runs) < self._min_loads:
                    plan.evidence_short += 1
                    continue
                plan.ready.append(cand)
        finally:
            tel.close()
            if lcm is not None:
                lcm.close()
        return plan

    def dry_run(self) -> DryRunReport:
        """What a cycle would do now, from the stores as they are. Counts only."""
        allowed, label, reason = self.generator_status()
        # Building the judge makes no call; provenance() says whether it is pinned.
        judge_record = self._judge_provenance()
        return DryRunReport(
            enabled=self._enabled,
            generator=label,
            generator_allowed=allowed,
            generator_reason=reason,
            judge_configured=self._get_or_build_judge() is not None,
            judge_pinned=bool(judge_record and judge_record.get("pinned")),
            min_loads=self._min_loads,
            max_skills=self._max_skills,
            variants_per_skill=self._variants,
            min_margin=self._margin,
            threshold=self._threshold,
            pending_proposals=len(self._store.entries()),
            plan=self.plan(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_optimization_cycle(self) -> GEPAReport:
        """One cycle: plan → gates → variants → judge → stage the winners.

        Always returns a GEPAReport and never raises. Every cycle that runs
        writes one ``subsystem_runs`` row; a disabled optimizer runs nothing
        and writes none.
        """
        start = time.time()
        report = GEPAReport(timestamp=start)

        if not self._enabled:
            report.notes = "disabled"
            report.duration_seconds = time.time() - start
            return report

        outcome = "failed"
        plan: CyclePlan | None = None
        try:
            plan = await asyncio.to_thread(self.plan)
            report.candidates = len(plan.candidates)
            refusal = plan.counter_error or self._refusal()
            if refusal:
                outcome, report.notes = "skipped", refusal
            elif not plan.ready:
                outcome, report.notes = "skipped", self._nothing_ready(plan)
            else:
                for cand in plan.ready[: self._max_skills]:
                    try:
                        await self._optimize_one(cand, report)
                    except Exception:
                        report.errors += 1
                        log.exception("GEPA: optimising %s failed", cand.skill.path.name)
                outcome = "partial" if report.errors else "success"
        except Exception:
            log.exception("GEPA: cycle failed")
            report.notes = "cycle failed — see the log"
        report.duration_seconds = time.time() - start
        self._record_cycle(report, plan, outcome)
        return report

    def _nothing_ready(self, plan: CyclePlan) -> str:
        if not plan.candidates:
            return (f"no auto skill loaded at least {self._min_loads} times "
                    f"({plan.skills_loaded} loaded at all)")
        return (f"{len(plan.candidates)} candidates, none ready: "
                f"{plan.evidence_short} short of evidence, {plan.pending} already proposed, "
                f"{plan.too_long} too long for the judge")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _optimize_one(self, cand: Candidate, report: GEPAReport) -> None:
        """Judge the live skill, generate and judge variants, stage the winner if any."""
        judge = self._get_or_build_judge()
        runs = cand.runs
        live_doc = redact_secrets(cand.skill.text)

        live_scores: list[float] = []
        for run in runs:
            score = await self._score(judge, live_doc, run, report)
            if score is None:
                # Nothing measurable to beat: a variant cannot be shown better
                # than a score that does not exist.
                report.live_unparseable += 1
                return
            live_scores.append(score)

        variants = await self._generate_variants(cand, live_doc, report)
        report.variants += len(variants)

        best: tuple[str, list[float]] | None = None
        for variant in variants:
            scores: list[float] = []
            for run in runs:
                score = await self._score(judge, variant, run, report)
                if score is None:
                    break  # an unparseable verdict never wins
                scores.append(score)
            if len(scores) != len(runs) or not self._beats(scores, live_scores):
                continue
            if best is None or fmean(scores) > fmean(best[1]):
                best = (variant, scores)
        if best is None:
            return

        variant, scores = best
        try:
            from prometheus.security.code_scanner import DangerousCodeScanner

            scan = DangerousCodeScanner().scan_markdown_content(
                variant, file_path=str(cand.skill.path))
            unsafe = scan.is_dangerous
        except Exception:
            log.exception("GEPA: scanner failed on a variant of %s — not proposing",
                          cand.skill.path.name)
            unsafe = True
        if unsafe:
            report.unsafe += 1
            return

        live_mean, variant_mean = fmean(live_scores), fmean(scores)
        judge_record = self._judge_provenance()
        sidecar = self._store.create(
            skill_file=cand.skill.path.name,
            skill_name=cand.skill.served_name,
            live_text=cand.skill.text,
            variant_text=variant,
            scores={
                "runs": len(runs),
                "live": live_scores,
                "variant": scores,
                "live_mean": round(live_mean, 4),
                "variant_mean": round(variant_mean, 4),
                "gain": round(variant_mean - live_mean, 4),
            },
            rule={
                "min_loads": self._min_loads,
                "min_margin": self._margin,
                "threshold": self._threshold,
                "worse_on_no_run": True,
            },
            evidence=ev.summarize(runs, loads_counted=cand.loads).to_dict(),
            # WHO GRADED these numbers, next to them. Until 2026-08-02 GEPA
            # ran an UNPINNED judge while the nightly script ran a pinned one,
            # and the records were indistinguishable. Never inferred: absent
            # means unknown.
            judge=judge_record,
            generator={
                "provider": self._provider_name,
                "hosted": not _is_local_provider(self._provider_name),
                "model": self._model,
                # The K/V cache quantisation of the backend that produced the
                # variants — recorded, never set and never inferred.
                "kv_cache": self._kv_cache_provenance(),
            },
            variants_judged=len(variants),
        )
        report.proposed += 1
        report.proposals.append({
            "id": sidecar["id"],
            "skill": cand.skill.served_name,
            "live_mean": live_mean,
            "variant_mean": variant_mean,
        })

    def _beats(self, variant: list[float], live: list[float]) -> bool:
        """The proposal rule: better on the mean by the margin, good enough, worse on no run."""
        v_mean, l_mean = fmean(variant), fmean(live)
        gain = v_mean - l_mean
        return (
            gain > 0
            and gain >= self._margin - _EPS
            and v_mean >= self._threshold - _EPS
            and all(v >= base - _EPS for v, base in zip(variant, live))
        )

    async def _score(
        self,
        judge: Any,
        doc: str,
        run: ev.LoadRun,
        report: GEPAReport,
    ) -> float | None:
        """One verdict on one document for one run; None when it holds no score."""
        report.judged += 1
        try:
            verdict = await judge.evaluate(
                task_input=redact_secrets(_JUDGE_TASK.format(run=run.render())),
                agent_output=redact_secrets(doc),
                expected_behavior=_JUDGE_EXPECTED,
            )
        except Exception:
            log.debug("GEPA: judge call failed", exc_info=True)
            report.unparseable += 1
            return None
        score = parsed_score(verdict)
        if score is None:
            report.unparseable += 1
        return score

    async def _generate_variants(
        self,
        cand: Candidate,
        live_doc: str,
        report: GEPAReport,
    ) -> list[str]:
        """Ask the generator for ``gepa_variants_per_skill`` variants; keep the valid, distinct ones."""
        prompt = _VARIANT_PROMPT.format(
            current_skill=live_doc,
            n_runs=len(cand.runs),
            runs="\n\n".join(f"Run {i}:\n{run.render()}" for i, run in enumerate(cand.runs, 1)),
        )
        seen = {_normalized(live_doc)}
        out: list[str] = []
        for _ in range(self._variants):
            try:
                text = await self._call_provider(prompt)
            except Exception:
                log.debug("GEPA: variant generation failed", exc_info=True)
                report.variants_rejected += 1
                continue
            variant = self._clean_variant(text, cand.skill)
            if variant is None or _normalized(variant) in seen:
                report.variants_rejected += 1
                continue
            seen.add(_normalized(variant))
            out.append(variant)
        return out

    @staticmethod
    def _clean_variant(text: str, skill: ev.AutoSkill) -> str | None:
        """The variant as it would be written, or None when it cannot replace *skill*.

        It must open with a closed frontmatter block that keeps the live
        skill's name and description (the registry and the prompt's skill list
        know it by those), and fit what the judge reads.
        """
        text = (text or "").strip()
        fence = re.fullmatch(r"```[A-Za-z]*\n(.*)\n```", text, flags=re.S)
        if fence:
            text = fence.group(1).strip()
        text = redact_secrets(text) + "\n"
        if not frontmatter_closed(text):
            return None
        if skill_identity(skill.stem, text) != (skill.served_name, skill.description):
            return None
        if len(text) > JUDGE_DOC_CHARS:
            return None
        return text

    def _record_cycle(self, report: GEPAReport, plan: CyclePlan | None, outcome: str) -> None:
        """The cycle's one ``subsystem_runs`` row. A telemetry failure is logged, never raised."""
        tel = self._telemetry
        record = getattr(tel, "record_run", None)
        if record is None:
            return
        judge = self._judge_provenance() or {}
        summary: dict[str, Any] = {
            "candidates": report.candidates,
            "variants": report.variants,
            "judged": report.judged,
            "unparseable": report.unparseable,
            "proposed": report.proposed,
            "live_unparseable": report.live_unparseable,
            "variants_rejected": report.variants_rejected,
            "unsafe": report.unsafe,
            "errors": report.errors,
            "min_loads": self._min_loads,
            "min_margin": self._margin,
            "threshold": self._threshold,
            "generator": self.generator_status()[1],
            "judge_model": judge.get("model"),
            "judge_pinned": judge.get("pinned"),
            "note": report.notes,
        }
        if plan is not None:
            summary.update({
                "auto_skills": plan.auto_skills,
                "auto_loads": plan.auto_loads,
                "below_min_loads": plan.below_min_loads,
                "evidence_short": plan.evidence_short,
                "pending": plan.pending,
                "too_long": plan.too_long,
                "ready": len(plan.ready),
            })
        try:
            record(
                GEPA_SUBSYSTEM, GEPA_CYCLE_OPERATION, outcome,
                duration_ms=report.duration_seconds * 1000.0,
                summary=summary,
            )
        except Exception:
            log.warning("GEPA: could not record the cycle's telemetry row", exc_info=True)

    def _judge_provenance(self) -> dict[str, object] | None:
        """Who graded — or None if no judge is available.

        Reads through the same lazy accessor the scoring path uses, so the
        recorded judge is necessarily the one that produced the numbers rather
        than a separately-constructed lookalike.
        """
        judge = self._get_or_build_judge()
        prov = getattr(judge, "provenance", None)
        return prov() if callable(prov) else None

    def _kv_cache_provenance(self) -> dict[str, object] | None:
        """The backend's K/V cache types, if the provider was probed for them.

        Read off the provider's cached probe result rather than re-probing:
        the value recorded must be the one in force while these completions
        were produced, not a fresh reading taken after the fact. ``None``
        means never probed — distinct from a probe that came back
        ``source="unreported"``, which means the server was asked and does
        not publish it.
        """
        return getattr(self._provider, "server_kv_cache", None)

    def _get_or_build_judge(self) -> object | None:
        """Lazily construct the default ``PrometheusJudge`` if none was supplied."""
        if self._judge is not None:
            return self._judge
        if not self._judge_base_url:
            return None
        try:
            from prometheus.evals.judge import PrometheusJudge
            self._judge = PrometheusJudge(
                base_url=self._judge_base_url, model=self._judge_model
            )
        except Exception:
            log.exception("GEPA: failed to build PrometheusJudge")
            return None
        return self._judge

    async def _call_provider(self, prompt: str) -> str:
        """Stream a single completion and return concatenated text.

        The prompt is built from skills and the runs they were used in, so
        token shapes are redacted before it is sent (X.37).
        """
        from prometheus.engine.messages import ConversationMessage
        from prometheus.providers.base import (
            ApiMessageRequest,
            ApiTextDeltaEvent,
        )

        provider = self._provider
        if provider is None:
            raise RuntimeError("GEPA has no provider to generate variants")
        request = ApiMessageRequest(
            model=self._model,
            messages=[ConversationMessage.from_user_text(redact_secrets(prompt))],
            max_tokens=2048,
        )
        text_parts: list[str] = []
        async for event in provider.stream_message(request):
            if isinstance(event, ApiTextDeltaEvent):
                text_parts.append(event.text)
        return "".join(text_parts)
