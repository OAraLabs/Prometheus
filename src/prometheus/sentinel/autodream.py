"""AutoDreamEngine — idle-time background intelligence.

Source: Novel code for Prometheus Sprint 9.
When the user stops talking to Prometheus, it starts thinking. Runs 4 phases:
wiki lint, memory consolidation, telemetry digest, knowledge synthesis.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prometheus.config.paths import get_config_dir
from prometheus.sentinel.signals import ActivitySignal, SignalBus

if TYPE_CHECKING:
    from prometheus.sentinel.knowledge_synth import KnowledgeSynthesizer
    from prometheus.sentinel.memory_consolidator import MemoryConsolidator
    from prometheus.sentinel.telemetry_digest import TelemetryDigest
    from prometheus.sentinel.wiki_lint import WikiLinter

log = logging.getLogger(__name__)


@dataclass
class DreamResult:
    """Result of one dream phase."""

    phase: str
    duration_seconds: float
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class AutoDreamEngine:
    """Idle-triggered background intelligence.

    Subscribes to ``idle_start`` / ``idle_end`` signals. When idle, runs
    dream cycles: wiki lint, memory consolidation, telemetry digest,
    and (budget-capped) knowledge synthesis.
    """

    def __init__(
        self,
        bus: SignalBus,
        *,
        wiki_linter: WikiLinter | None = None,
        memory_consolidator: MemoryConsolidator | None = None,
        telemetry_digest: TelemetryDigest | None = None,
        knowledge_synth: KnowledgeSynthesizer | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._bus = bus
        self._wiki_linter = wiki_linter
        self._memory_consolidator = memory_consolidator
        self._telemetry_digest = telemetry_digest
        self._knowledge_synth = knowledge_synth

        cfg = config or {}
        self._dream_interval = cfg.get("dream_interval_minutes", 30) * 60
        self._budget_tokens = cfg.get("dream_budget_tokens", 2000)
        self._auto_fix_wiki = cfg.get("auto_fix_wiki", True)
        self._synthesis_enabled = cfg.get("synthesis_enabled", True)

        self._dreaming = False
        self._last_cycle: float = 0.0
        self._cycle_count: int = 0
        self._last_results: list[DreamResult] = []
        self._log_path = get_config_dir() / "sentinel" / "dream_log.md"

    async def start(self) -> None:
        """Subscribe to idle signals on the bus."""
        self._bus.subscribe("idle_start", self._on_idle_start)
        self._bus.subscribe("idle_end", self._on_idle_end)
        log.info("AutoDreamEngine: ready (interval=%ds)", self._dream_interval)

    async def _on_idle_start(self, signal: ActivitySignal) -> None:
        """Begin dreaming when idle detected."""
        if self._dreaming:
            return
        self._dreaming = True
        log.info("AutoDreamEngine: idle detected, starting dream loop")
        asyncio.create_task(self._dream_loop())

    async def _on_idle_end(self, signal: ActivitySignal) -> None:
        """Stop dreaming when activity resumes."""
        if self._dreaming:
            self._dreaming = False
            log.info("AutoDreamEngine: activity resumed, stopping dreams")

    async def _dream_loop(self) -> None:
        """Run dream cycles while idle."""
        while self._dreaming:
            await self.run_cycle()
            # Wait for next cycle or until stopped
            for _ in range(int(self._dream_interval)):
                if not self._dreaming:
                    return
                await asyncio.sleep(1)

    async def run_cycle(self) -> list[DreamResult]:
        """Execute one dream cycle — all 4 phases in order.

        Can be called directly (e.g. from tests) or via the dream loop.
        Checks ``_dreaming`` between phases so the cycle can abort early
        if ``idle_end`` fires, but only when running inside the loop.
        """
        results: list[DreamResult] = []
        was_dreaming = self._dreaming  # track if loop-driven
        self._cycle_count += 1
        log.info("AutoDreamEngine: dream cycle #%d started", self._cycle_count)

        phases: list[tuple[str, object | None, object]] = [
            ("wiki_lint", self._wiki_linter, self._phase_wiki_lint),
            ("memory_consolidation", self._memory_consolidator, self._phase_memory_consolidation),
            ("telemetry_digest", self._telemetry_digest, self._phase_telemetry_digest),
        ]
        if self._synthesis_enabled:
            phases.append(("knowledge_synthesis", self._knowledge_synth, self._phase_knowledge_synthesis))

        for name, component, fn in phases:
            if component is None:
                continue
            # If we were loop-driven and got stopped, abort
            if was_dreaming and not self._dreaming:
                break
            result = await self._run_phase(name, fn)
            results.append(result)

        self._last_cycle = time.time()
        self._last_results = results

        # Emit dream_complete signal
        await self._bus.emit(ActivitySignal(
            kind="dream_complete",
            payload={
                "cycle": self._cycle_count,
                "phases": [r.phase for r in results],
                "errors": [r.phase for r in results if r.error],
            },
            source="autodream",
        ))

        self._write_log(results)
        return results

    async def _run_phase(self, name: str, fn) -> DreamResult:
        """Run a single phase with timing and error handling."""
        start = time.time()
        try:
            summary = await fn()
            return DreamResult(
                phase=name,
                duration_seconds=time.time() - start,
                summary=summary or {},
            )
        except Exception as exc:
            log.exception("AutoDreamEngine: phase %s failed", name)
            return DreamResult(
                phase=name,
                duration_seconds=time.time() - start,
                error=str(exc),
            )

    async def _phase_wiki_lint(self) -> dict[str, Any]:
        """Phase 1: Wiki lint + optional auto-fix."""
        result = self._wiki_linter.lint()  # type: ignore[union-attr]
        summary: dict[str, Any] = {
            "issues": len(result.issues),
            "errors": result.error_count,
            "warnings": result.warning_count,
        }
        if result.has_issues and self._auto_fix_wiki:
            fixed = self._wiki_linter.auto_fix(result)  # type: ignore[union-attr]
            summary["auto_fixed"] = fixed
        return summary

    async def _phase_memory_consolidation(self) -> dict[str, Any]:
        """Phase 2: Memory dedup, decay, tombstone."""
        result = self._memory_consolidator.consolidate()  # type: ignore[union-attr]
        return {
            "duplicates_merged": result.duplicates_merged,
            "confidence_decayed": result.confidence_decayed,
            "tombstoned": result.tombstoned,
        }

    async def _phase_telemetry_digest(self) -> dict[str, Any]:
        """Phase 3: Telemetry health check."""
        result = self._telemetry_digest.generate()  # type: ignore[union-attr]
        summary = {
            "total_calls": result.total_calls,
            "anomalies": len(result.anomalies),
        }
        if result.has_anomalies:
            await self._bus.emit(ActivitySignal(
                kind="dream_insight",
                payload={"digest": result.summary},
                source="autodream",
            ))
        return summary

    async def _phase_knowledge_synthesis(self) -> dict[str, Any]:
        """Phase 4: LLM-powered cross-entity insight generation."""
        insights = await self._knowledge_synth.synthesize(  # type: ignore[union-attr]
            budget_tokens=self._budget_tokens,
        )
        return {
            "insights_generated": len(insights),
            "tokens_used": sum(i.tokens_used for i in insights),
        }

    def _write_log(self, results: list[DreamResult]) -> None:
        """Append cycle results to dream_log.md."""
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d %H:%M", time.localtime())
        lines = [f"\n## Dream Cycle #{self._cycle_count} — {timestamp}\n"]
        for r in results:
            status = "OK" if not r.error else f"FAILED: {r.error}"
            lines.append(f"- **{r.phase}**: {status} ({r.duration_seconds:.1f}s)")
            if r.summary:
                for k, v in r.summary.items():
                    lines.append(f"  - {k}: {v}")
        lines.append("")

        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

    # ------------------------------------------------------------------
    # Status (used by SentinelStatusTool)
    # ------------------------------------------------------------------

    @property
    def dreaming(self) -> bool:
        return self._dreaming

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def last_cycle_time(self) -> float:
        return self._last_cycle

    @property
    def last_results(self) -> list[DreamResult]:
        return self._last_results
