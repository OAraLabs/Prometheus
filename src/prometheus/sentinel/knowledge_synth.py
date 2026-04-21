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
    ) -> None:
        self._store = store
        self._provider = provider
        self._model = model
        self._budget_tokens = budget_tokens
        self._min_cluster_size = min_cluster_size
        self._wiki_root = Path(wiki_root) if wiki_root else get_config_dir() / "wiki"

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
        """Generate insight for one entity cluster via LLM."""
        from prometheus.engine.messages import ConversationMessage
        from prometheus.providers.base import (
            ApiMessageCompleteEvent,
            ApiMessageRequest,
            ApiTextDeltaEvent,
        )

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

        request = ApiMessageRequest(
            model=self._model,
            messages=[ConversationMessage(role="user", content=prompt)],
            max_tokens=min(self._budget_tokens, 512),
        )

        text_parts: list[str] = []
        tokens_used = 0
        try:
            async for event in self._provider.stream_message(request):
                if isinstance(event, ApiTextDeltaEvent):
                    text_parts.append(event.text)
                elif isinstance(event, ApiMessageCompleteEvent):
                    if event.usage:
                        tokens_used = getattr(event.usage, "output_tokens", 0)
        except Exception:
            log.exception("KnowledgeSynthesizer: LLM call failed for cluster %s", cluster)
            return None

        insight_text = "".join(text_parts).strip()
        if not insight_text:
            return None

        return SynthInsight(
            entities=cluster,
            insight=insight_text,
            tokens_used=tokens_used,
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
