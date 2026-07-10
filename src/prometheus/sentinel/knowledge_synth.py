"""KnowledgeSynthesizer — cross-entity pattern detection via LLM.

Source: Novel code for Prometheus Sprint 9.
The only AutoDream component that uses the LLM. Budget-capped to avoid
burning GPU time during idle cycles.
"""

from __future__ import annotations

import hashlib
import logging
import re
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

# Facts fed to the prompt AND fingerprinted per cluster. One cap for both:
# the skip check must see exactly the fact set the insight was synthesized
# from, or the two drift and unchanged clusters re-synthesize anyway.
_MAX_CLUSTER_FACTS = 30

# ``facts_hash`` line in an insight page's frontmatter — the fingerprint of
# the facts the page was last synthesized from.
_FACTS_HASH_RE = re.compile(r"^facts_hash: ([0-9a-f]{64})$", re.MULTILINE)


def _norm_text(text: str) -> str:
    """Lowercase + collapse whitespace, for fingerprint stability."""
    return re.sub(r"\s+", " ", text.strip().lower())


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
        skipped_unchanged = 0

        for cluster in clusters:
            if tokens_spent >= budget:
                break

            facts = self._gather_cluster_facts(cluster)
            if len(facts) < 3:
                continue

            # Dedup gate: if this cluster's insight page already reflects
            # exactly these facts, re-synthesizing would mint a near-identical
            # page (the live vault accumulated the same slug 20×) and burn
            # budget. Skip BEFORE the LLM call.
            fingerprint = self._facts_fingerprint(cluster, facts)
            if self._read_page_fingerprint(self._insight_page_path(cluster)) == fingerprint:
                skipped_unchanged += 1
                continue

            insight = await self._generate_insight(cluster, facts)
            if insight:
                tokens_spent += insight.tokens_used
                insights.append(insight)
                self._write_insight_page(insight, fingerprint=fingerprint)

        if skipped_unchanged:
            log.debug(
                "KnowledgeSynthesizer: skipped %d cluster(s) whose facts are "
                "unchanged since their insight page was written",
                skipped_unchanged,
            )
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

    def _gather_cluster_facts(
        self, cluster: list[str]
    ) -> list[tuple[str, str, float]]:
        """Collect ``(entity, fact, confidence)`` rows for a cluster, capped."""
        facts: list[tuple[str, str, float]] = []
        for entity in cluster[:10]:  # Cap to avoid huge prompts
            mems = self._store.search_memories(entity=entity, limit=10)
            for mem in mems:
                facts.append((entity, mem["fact"], mem["confidence"]))
        return facts[:_MAX_CLUSTER_FACTS]

    def _facts_fingerprint(
        self, cluster: list[str], facts: list[tuple[str, str, float]]
    ) -> str:
        """Stable digest of a cluster's underlying facts.

        Confidence is excluded on purpose: a dedup merge bumps confidence and
        mention_count without changing what is known, and must not trigger a
        re-synthesis.
        """
        lines = sorted(
            _norm_text(entity) + "\t" + _norm_text(fact)
            for entity, fact, _confidence in facts
        )
        payload = "\n".join([",".join(_norm_text(e) for e in cluster), *lines])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _insight_page_path(self, entities: list[str]) -> Path:
        """Stable page path for a cluster: wiki/queries/insight-{topic}.md.

        Entities are model-extracted free text and can contain path
        separators / "$vars" / dots — the slug must stay a single safe
        path component inside queries_dir (same whitelist idiom as
        skill_creator._slugify).
        """
        topic = re.sub(r"[^a-z0-9]+", "-", "-".join(entities[:3]).lower())
        topic = topic.strip("-")[:40].rstrip("-") or "cluster"
        return self._wiki_root / "queries" / f"insight-{topic}.md"

    @staticmethod
    def _read_page_fingerprint(path: Path) -> str | None:
        """``facts_hash`` recorded in an existing insight page, or ``None``."""
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return None
        m = _FACTS_HASH_RE.search(text)
        return m.group(1) if m else None

    async def _generate_insight(
        self, cluster: list[str], facts: list[tuple[str, str, float]]
    ) -> SynthInsight | None:
        """Generate insight for one entity cluster via the shared LLM envelope.

        The envelope builds the request with ``content=[TextBlock(text=prompt)]``
        (canonical ContentBlock list), so the ed8f1a6 string-vs-list crash that
        previously broke this phase on every dream cycle is structurally
        impossible, and any failure lands in telemetry.silent_failures.
        """
        facts_lines = [
            f"  [{entity}] {fact} (confidence: {confidence:.2f})"
            for entity, fact, confidence in facts
        ]

        prompt = _SYNTHESIS_PROMPT.format(
            entities=", ".join(cluster[:10]),
            facts="\n".join(facts_lines),
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

    def _write_insight_page(
        self, insight: SynthInsight, *, fingerprint: str | None = None
    ) -> None:
        """Write/update wiki/queries/insight-{topic}.md — one stable page per
        cluster, updated in place when the underlying facts change.

        Pre-dedup, the name carried a date stamp, so every idle cycle minted a
        near-identical new page for an unchanged cluster — and queries/ is the
        only non-regenerable vault content, so the noise accumulated durably.
        Legacy dated pages are left in place (the vault is never edited from
        here beyond this cluster's own page).
        """
        path = self._insight_page_path(insight.entities)
        path.parent.mkdir(parents=True, exist_ok=True)

        content = (
            f"---\ntype: insight\ngenerated: {time.strftime('%Y-%m-%d %H:%M')}\n"
            f"entities: {insight.entities}\ntokens_used: {insight.tokens_used}\n"
        )
        if fingerprint:
            content += f"facts_hash: {fingerprint}\n"
        content += (
            f"---\n\n"
            f"# Insight: {', '.join(insight.entities[:5])}\n\n"
            f"{insight.insight}\n\n"
            f"## Related Entities\n"
        )
        for entity in insight.entities:
            content += f"- [[{entity}]]\n"

        path.write_text(content, encoding="utf-8")
        log.debug("KnowledgeSynthesizer: wrote %s", path)
