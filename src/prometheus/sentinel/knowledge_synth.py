"""KnowledgeSynthesizer — cross-entity pattern detection via LLM.

Source: Novel code for Prometheus Sprint 9.
The only AutoDream component that uses the LLM. Budget-capped to avoid
burning GPU time during idle cycles.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from prometheus.config.paths import get_config_dir

if TYPE_CHECKING:
    from prometheus.memory.store import MemoryStore
    from prometheus.providers.base import ModelProvider

log = logging.getLogger(__name__)

_SYNTHESIS_PROMPT = """\
You are analyzing a knowledge base about entities that frequently co-occur.
Given the facts below about a cluster of related entities, identify non-obvious
patterns, connections, or insights that link them together.

Entities in cluster: {entities}

Facts:
{facts}

Respond with a brief insight (2-4 sentences) about what connects these entities
or what patterns emerge from the data. Focus on actionable or surprising connections.
"""


@dataclass
class SynthInsight:
    """One synthesized insight from entity co-occurrence."""

    entities: list[str]
    insight: str
    tokens_used: int = 0


class KnowledgeSynthesizer:
    """Find cross-entity patterns and generate insight pages.

    Budget-capped: tracks cumulative token usage and stops when budget
    is exceeded.
    """

    def __init__(
        self,
        store: MemoryStore,
        provider: ModelProvider,
        *,
        model: str = "default",
        budget_tokens: int = 2000,
        min_cluster_size: int = 2,
        wiki_root: Path | None = None,
        telemetry: object | None = None,
    ) -> None:
        from prometheus.learning.llm_envelope import LLMCallEnvelope

        self._store = store
        self._provider = provider
        self._model = model
        self._budget_tokens = budget_tokens
        self._min_cluster_size = min_cluster_size
        self._wiki_root = Path(wiki_root) if wiki_root else get_config_dir() / "wiki"
        self._telemetry = telemetry
        # Route the synthesis call through the shared envelope (like
        # skill_creator / memory_extractor). The envelope constructs the
        # request with the canonical ``content=[TextBlock(text=prompt)]``
        # shape — this method previously built ``content=prompt`` (a str),
        # raising pydantic ``list_type`` on EVERY dream cycle (the phase had
        # never worked) — and surfaces any call failure to
        # ``telemetry.silent_failures`` instead of only an ERROR log.
        # on_failure="return_none" keeps the "no insight on failure" contract.
        self._envelope = LLMCallEnvelope(
            subsystem="knowledge_synth",
            telemetry=telemetry,
            on_failure="return_none",
        )

    async def synthesize(self, budget_tokens: int | None = None) -> list[SynthInsight]:
        """Find entity clusters and generate insights. Budget-capped."""
        budget = budget_tokens or self._budget_tokens
        clusters = self._build_entity_clusters()

        if not clusters:
            log.debug("KnowledgeSynthesizer: no clusters found")
            return []

        insights: list[SynthInsight] = []
        tokens_spent = 0

        for cluster in clusters:
            if tokens_spent >= budget:
                break

            insight = await self._generate_insight(cluster)
            if insight:
                tokens_spent += insight.tokens_used
                insights.append(insight)
                self._write_insight_page(insight)

        if insights:
            log.info(
                "KnowledgeSynthesizer: generated %d insight(s), %d tokens used",
                len(insights),
                tokens_spent,
            )
        return insights

    def _build_entity_clusters(self) -> list[list[str]]:
        """Find entities that co-occur in facts via shared source events."""
        memories = self._store.get_all_memories(min_confidence=0.3, limit=2000)

        # Build co-occurrence: entities sharing source_event_ids
        entity_events: dict[str, set[str]] = defaultdict(set)
        for mem in memories:
            for eid in mem.get("source_event_ids", []):
                entity_events[mem["entity_name"]].add(eid)

        # Find entities with overlapping event sets
        entities = list(entity_events.keys())
        adjacency: dict[str, set[str]] = defaultdict(set)

        for i, a in enumerate(entities):
            for b in entities[i + 1:]:
                overlap = entity_events[a] & entity_events[b]
                if len(overlap) >= 2:  # At least 2 shared events
                    adjacency[a].add(b)
                    adjacency[b].add(a)

        # Find connected components
        visited: set[str] = set()
        clusters: list[list[str]] = []

        for entity in entities:
            if entity in visited or entity not in adjacency:
                continue
            cluster: list[str] = []
            stack = [entity]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                cluster.append(node)
                stack.extend(adjacency.get(node, set()) - visited)
            if len(cluster) >= self._min_cluster_size:
                clusters.append(sorted(cluster))

        # Sort by size descending (highest-signal first)
        clusters.sort(key=len, reverse=True)
        return clusters

    async def _generate_insight(self, cluster: list[str]) -> SynthInsight | None:
        """Generate insight for one entity cluster via the shared LLM envelope.

        The envelope builds the request with ``content=[TextBlock(text=prompt)]``
        (canonical ContentBlock list), so the ed8f1a6 string-vs-list crash that
        previously broke this phase on every dream cycle is structurally
        impossible, and any failure lands in telemetry.silent_failures.
        """
        # Gather facts about these entities
        facts_lines: list[str] = []
        for entity in cluster[:10]:  # Cap to avoid huge prompts
            mems = self._store.search_memories(entity=entity, limit=10)
            for mem in mems:
                facts_lines.append(
                    f"  [{entity}] {mem['fact']} (confidence: {mem['confidence']:.2f})"
                )

        if len(facts_lines) < 3:
            return None

        prompt = _SYNTHESIS_PROMPT.format(
            entities=", ".join(cluster[:10]),
            facts="\n".join(facts_lines[:30]),  # Cap facts
        )

        # on_failure="return_none": a failed/empty call returns None (recorded
        # in telemetry by the envelope) → no insight, same as the old contract.
        insight_text = await self._envelope.call(
            provider=self._provider,
            model=self._model,
            prompt=prompt,
            max_tokens=min(self._budget_tokens, 512),
            operation="generate_insight",
        )
        if not insight_text or not insight_text.strip():
            return None
        insight_text = insight_text.strip()

        return SynthInsight(
            entities=cluster,
            insight=insight_text,
            # The envelope abstracts the token stream; estimate output tokens
            # from text length (~chars/4) for the wiki frontmatter (it is
            # informational only).
            tokens_used=max(1, len(insight_text) // 4),
        )

    def _write_insight_page(self, insight: SynthInsight) -> None:
        """Write insight to wiki/queries/insight-{date}-{topic}.md."""
        queries_dir = self._wiki_root / "queries"
        queries_dir.mkdir(parents=True, exist_ok=True)

        date_str = time.strftime("%Y%m%d", time.localtime())
        topic = "-".join(insight.entities[:3]).lower().replace(" ", "-")[:40]
        filename = f"insight-{date_str}-{topic}.md"
        path = queries_dir / filename

        content = (
            f"---\ntype: insight\ngenerated: {time.strftime('%Y-%m-%d %H:%M')}\n"
            f"entities: {insight.entities}\ntokens_used: {insight.tokens_used}\n---\n\n"
            f"# Insight: {', '.join(insight.entities[:5])}\n\n"
            f"{insight.insight}\n\n"
            f"## Related Entities\n"
        )
        for entity in insight.entities:
            content += f"- [[{entity}]]\n"

        path.write_text(content, encoding="utf-8")
        log.debug("KnowledgeSynthesizer: wrote %s", path)
