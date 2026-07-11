"""KnowledgeSynthesizer dedup — one stable page per cluster, skip unchanged.

Pre-fix, ``_write_insight_page`` minted a date-stamped file per cycle, so an
unchanged cluster produced a near-identical page on every idle dream (the live
vault held 100 insight files with the top slug repeated 20×) — and queries/ is
the only non-regenerable vault content, so the noise accumulated durably.

Now the page name is stable per cluster, the facts it was synthesized from are
fingerprinted into ``facts_hash`` frontmatter, and an unchanged cluster is
skipped BEFORE the LLM call (no budget burned, no page rewritten). Facts
changed → the same page is updated in place, never a second file.
"""

from __future__ import annotations

import re
from pathlib import Path

from prometheus.providers.base import ApiTextDeltaEvent
from prometheus.sentinel.knowledge_synth import KnowledgeSynthesizer


class _ClusterStore:
    """Two entities sharing ≥2 source events → exactly one cluster.

    ``facts`` is mutable so tests can change what the store knows between
    synthesize() calls; ``confidence`` likewise, to prove confidence-only
    drift does not retrigger synthesis.
    """

    def __init__(self) -> None:
        self.facts: dict[str, list[str]] = {
            "EntA": ["EntA ships with EntB", "EntA depends on the EntB parser"],
            "EntB": ["EntB is maintained by the EntA team", "EntB releases track EntA"],
        }
        self.confidence = 0.9

    def get_all_memories(self, min_confidence: float = 0.0, limit: int = 1000):
        return [
            {
                "entity_name": name,
                "fact": fact,
                "confidence": self.confidence,
                "source_event_ids": ["e1", "e2"],
            }
            for name, facts in self.facts.items()
            for fact in facts
        ]

    def search_memories(self, entity: str, limit: int):
        return [
            {"fact": f, "confidence": self.confidence}
            for f in self.facts[entity][:limit]
        ]


class _CountingProvider:
    """Streams a distinct canned insight per call; counts LLM invocations."""

    def __init__(self) -> None:
        self.calls = 0

    async def stream_message(self, request):  # noqa: ANN001 — provider duck type
        self.calls += 1
        yield ApiTextDeltaEvent(text=f"Synthesized insight #{self.calls}.")


def _make(tmp_path: Path) -> tuple[_ClusterStore, _CountingProvider, KnowledgeSynthesizer]:
    store, provider = _ClusterStore(), _CountingProvider()
    ks = KnowledgeSynthesizer(
        store=store, provider=provider, model="m", wiki_root=tmp_path / "wiki",
    )
    return store, provider, ks


def _pages(tmp_path: Path) -> list[Path]:
    return sorted((tmp_path / "wiki" / "queries").glob("*.md"))


async def test_unchanged_cluster_synthesized_once(tmp_path):
    _store, provider, ks = _make(tmp_path)

    first = await ks.synthesize()
    assert len(first) == 1
    pages = _pages(tmp_path)
    assert len(pages) == 1
    # Stable, date-free name — repeated cycles cannot mint dated duplicates.
    assert re.fullmatch(r"insight-[a-z0-9-]+\.md", pages[0].name)
    body = pages[0].read_text(encoding="utf-8")
    assert re.search(r"^facts_hash: [0-9a-f]{64}$", body, re.MULTILINE)

    # Second idle cycle, same facts: skipped BEFORE the LLM call.
    second = await ks.synthesize()
    assert second == []
    assert provider.calls == 1, "unchanged cluster must not burn LLM budget"
    assert _pages(tmp_path) == pages
    assert pages[0].read_text(encoding="utf-8") == body


async def test_changed_facts_update_page_in_place(tmp_path):
    store, provider, ks = _make(tmp_path)
    await ks.synthesize()
    old_body = _pages(tmp_path)[0].read_text(encoding="utf-8")

    store.facts["EntA"].append("EntA v2 broke EntB compatibility")
    updated = await ks.synthesize()

    assert len(updated) == 1
    assert provider.calls == 2
    pages = _pages(tmp_path)
    assert len(pages) == 1, "changed facts must update in place, not add a file"
    new_body = pages[0].read_text(encoding="utf-8")
    assert new_body != old_body
    assert "Synthesized insight #2." in new_body


async def test_confidence_only_change_does_not_resynthesize(tmp_path):
    store, provider, ks = _make(tmp_path)
    await ks.synthesize()

    # A dedup merge bumps confidence without changing what is known.
    store.confidence = 0.95
    assert await ks.synthesize() == []
    assert provider.calls == 1
