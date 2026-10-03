"""MemoryConsolidator — dedup, decay, and clean the MemoryStore.

Source: Novel code for Prometheus Sprint 9.
Runs during AutoDream idle cycles. No LLM needed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prometheus.memory.store import MemoryStore

log = logging.getLogger(__name__)

#: The ``store_marks`` entry holding how far decay has been charged
#: (wall-clock seconds). See ``MemoryConsolidator._decay_confidence``.
DECAY_MARK = "decay_charged_through"


@dataclass
class ConsolidationResult:
    """Summary of one consolidation pass."""

    duplicates_merged: int = 0
    confidence_decayed: int = 0
    mentions_refreshed: int = 0
    tombstoned: int = 0

    @property
    def total_actions(self) -> int:
        return (
            self.duplicates_merged
            + self.confidence_decayed
            + self.mentions_refreshed
            + self.tombstoned
        )


class MemoryConsolidator:
    """Keeps the MemoryStore clean over time."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        decay_rate: float = 0.05,
        min_confidence: float = 0.1,
        stale_days: int = 90,
        similarity_threshold: float = 0.80,
    ) -> None:
        self._store = store
        self._decay_rate = decay_rate
        self._min_confidence = min_confidence
        self._stale_days = stale_days
        self._similarity_threshold = similarity_threshold

    def consolidate(self) -> ConsolidationResult:
        """Run all consolidation passes. No LLM needed."""
        result = ConsolidationResult()
        result.duplicates_merged = self._merge_duplicates()
        result.confidence_decayed = self._decay_confidence()
        result.tombstoned = self._tombstone()

        if result.total_actions:
            log.info(
                "MemoryConsolidator: merged=%d decayed=%d tombstoned=%d",
                result.duplicates_merged,
                result.confidence_decayed,
                result.tombstoned,
            )
        return result

    def _merge_duplicates(self) -> int:
        """Find facts with >80% text similarity about the same entity, merge."""
        memories = self._store.get_all_memories(limit=2000)

        # Group by entity_name
        by_entity: dict[str, list[dict]] = {}
        for mem in memories:
            by_entity.setdefault(mem["entity_name"], []).append(mem)

        merged = 0
        for entity_mems in by_entity.values():
            if len(entity_mems) < 2:
                continue
            # Compare pairs
            to_delete: set[str] = set()
            for i, a in enumerate(entity_mems):
                if a["id"] in to_delete:
                    continue
                for b in entity_mems[i + 1:]:
                    if b["id"] in to_delete:
                        continue
                    ratio = SequenceMatcher(
                        None, a["fact"].lower(), b["fact"].lower()
                    ).ratio()
                    if ratio >= self._similarity_threshold:
                        # Keep the one with higher confidence
                        keep, remove = (a, b) if a["confidence"] >= b["confidence"] else (b, a)
                        new_count = keep.get("mention_count", 1) + remove.get("mention_count", 1)
                        self._store.update_memory(
                            keep["id"], mention_count=new_count
                        )
                        to_delete.add(remove["id"])

            for mid in to_delete:
                self._store.delete_memory(mid)
                merged += 1

        return merged

    def _decay_confidence(self) -> int:
        """Charge confidence decay for the stale time since the last pass.

        A fact is stale once ``stale_days`` pass without a mention. It loses
        ``decay_rate`` for every 30 days it spends stale, never going below
        ``min_confidence``. Each pass charges only the stale time since the
        store's ``decay_charged_through`` mark, then moves the mark to now.
        So the total does not depend on how often passes run, and a restart
        (a new consolidator on the same store) never charges a window twice.
        With no mark yet (a new store, or the first pass after this change)
        a fact is charged its whole stale time, once.

        This used to subtract ``decay_rate x periods overdue`` from the
        CURRENT confidence on every pass, so how often AutoDream ran set the
        speed: at 48 passes a day a fact 90+ days unmentioned lost 0.15 per
        pass and hit the floor within hours. Live on 2026-10-03, all 2,121
        stale facts of 2,173 sat at exactly the floor.
        """
        now = time.time()
        charged_through = self._store.get_mark(DECAY_MARK)
        stale_seconds = self._stale_days * 86400
        memories = self._store.get_all_memories(limit=5000)
        decayed = 0

        for mem in memories:
            if mem["confidence"] <= self._min_confidence:
                continue
            last = mem.get("last_mentioned", mem.get("timestamp", 0))
            stale_since = last + stale_seconds
            start = stale_since if charged_through is None else max(stale_since, charged_through)
            if now <= start:
                continue
            periods = (now - start) / (30 * 86400)
            new_conf = max(
                self._min_confidence,
                mem["confidence"] - (self._decay_rate * periods),
            )
            if new_conf < mem["confidence"]:
                self._store.update_memory(mem["id"], confidence=new_conf)
                decayed += 1

        # Never rewound: a wall clock stepping back must not re-open a window.
        if charged_through is None or now > charged_through:
            self._store.set_mark(DECAY_MARK, now)
        return decayed

    def _tombstone(self) -> int:
        """Delete memories below minimum confidence threshold."""
        memories = self._store.get_all_memories(
            min_confidence=0.0, limit=5000
        )
        removed = 0
        for mem in memories:
            if mem["confidence"] < self._min_confidence:
                self._store.delete_memory(mem["id"])
                removed += 1
        return removed
