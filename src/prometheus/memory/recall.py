# Provenance: NEW for the MEMORY-3 follow-up "passive recall" sprint — closes
#             the "memory/wiki is write-only to the model" gap from
#             MEMORY-AUDIT-20260617 (§3: extraction→wiki pipeline never feeds
#             back into the agent's per-turn context).
"""Passive memory recall — surface stored facts relevant to the current turn.

The extraction pipeline (``MemoryExtractor`` → ``memory.db`` →
``WikiCompiler``) has always been write-only from the model's point of view:
facts flow OUT of conversations but never back IN unless the model explicitly
calls ``wiki_query``. ``MemoryRecall`` closes that loop: at the start of each
``run_loop`` invocation the latest user message is matched (FTS5, any-token)
against the ``memories`` table and the best few facts ride that run's system
prompt as a ``# Recalled memory`` section.

Design constraints (mirroring the steer / empty-retry-nudge channel in
``engine.agent_loop``):

- **Request-only.** The block is appended to the per-run system-prompt copy,
  never to ``messages`` — recalled facts must not enter durable history, or
  the extractor would re-extract its own output on the next cycle (a
  self-reinforcing feedback loop).
- **Fail-open.** Any store error degrades to "no recall"; a broken
  ``memory.db`` must never block a turn.
- **Bounded.** Hard caps on fact count and characters keep the local model's
  context cost predictable.
- **Manual-first.** ``/note`` facts (``manual=1``, max-trust) outrank ambient
  extracted facts, same as ``wiki_query``'s ranking priority.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Words that carry no recall signal. Deliberately small: over-stripping hurts
# more than under-stripping because FTS5 bm25 already down-ranks common terms.
_STOPWORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has
    have how i if in into is it its just like me my no not of on or our out
    over please should so than that the their them then there these they this
    to us up was we were what when where which who why will with would you
    your yours
    """.split()
)

# Word tokens worth searching on: 3+ chars, letters/digits/underscore. Keeps
# identifiers like "memory_db" whole; drops "ok"/"hi"-class noise.
_KEYWORD_RE = re.compile(r"[A-Za-z0-9_]{3,}")


@dataclass
class RecallConfig:
    """Knobs for passive recall, read from the ``memory.recall`` config block."""

    enabled: bool = True
    max_facts: int = 6        # hard cap on facts injected per turn
    max_chars: int = 900      # hard cap on the whole rendered block
    min_confidence: float = 0.6
    max_keywords: int = 12    # cap on search terms taken from the message
    per_entity_cap: int = 2   # diversity guard: max facts from one entity

    @classmethod
    def from_config(cls, config: dict | None) -> "RecallConfig":
        section = ((config or {}).get("memory") or {}).get("recall") or {}
        defaults = cls()
        return cls(
            enabled=bool(section.get("enabled", defaults.enabled)),
            max_facts=int(section.get("max_facts", defaults.max_facts)),
            max_chars=int(section.get("max_chars", defaults.max_chars)),
            min_confidence=float(
                section.get("min_confidence", defaults.min_confidence)
            ),
            max_keywords=int(section.get("max_keywords", defaults.max_keywords)),
            per_entity_cap=int(
                section.get("per_entity_cap", defaults.per_entity_cap)
            ),
        )


_HEADER = "# Recalled memory"
_PREAMBLE = (
    "Facts recalled from long-term memory because they may relate to the "
    "current request. Treat them as background you already know: apply them "
    "when relevant, silently ignore them when not, and do not narrate that "
    "they were retrieved or cite \"my memory\"."
)


class MemoryRecall:
    """Match the latest user message against stored facts and render a block.

    Stateless between calls; safe to share across sessions. The store's own
    read-lock wrappers make concurrent use from the event loop safe, and the
    single FTS query per turn is a few milliseconds against a multi-MB db —
    negligible next to a model call.
    """

    def __init__(self, store, config: RecallConfig | None = None) -> None:
        self._store = store
        self._config = config or RecallConfig()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def recall(self, query: str) -> str:
        """Return a rendered ``# Recalled memory`` block, or ``""``.

        Empty string means "inject nothing" — blank/stopword-only queries,
        no matching facts, and store errors all land here (fail-open).
        """
        cfg = self._config
        if not cfg.enabled or not query:
            return ""
        keywords = self._extract_keywords(query)
        if not keywords:
            return ""
        try:
            candidates = self._store.search_memories(
                query=" ".join(keywords),
                min_confidence=cfg.min_confidence,
                limit=cfg.max_facts * 4,  # headroom for dedup + entity cap
                match_any=True,
            )
        except Exception:
            log.warning(
                "MemoryRecall: search failed — continuing without recall",
                exc_info=True,
            )
            return ""
        selected = self._select(candidates)
        if not selected:
            return ""
        return self._render(selected)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _extract_keywords(self, query: str) -> list[str]:
        seen: set[str] = set()
        keywords: list[str] = []
        for token in _KEYWORD_RE.findall(query.lower()):
            if token in _STOPWORDS or token in seen:
                continue
            seen.add(token)
            keywords.append(token)
            if len(keywords) >= self._config.max_keywords:
                break
        return keywords

    def _select(self, candidates: list[dict]) -> list[dict]:
        """Manual-first, deduped, entity-capped shortlist within budget."""
        # Stable sort: manual (/note) facts to the front, FTS relevance order
        # preserved within each group.
        ordered = sorted(
            candidates, key=lambda m: 0 if m.get("manual") else 1
        )
        from prometheus.memory.store import _normalize_for_dedup

        seen_facts: set[str] = set()
        per_entity: dict[str, int] = {}
        selected: list[dict] = []
        for mem in ordered:
            fact = (mem.get("fact") or "").strip()
            if not fact:
                continue
            norm = _normalize_for_dedup(fact)
            if norm in seen_facts:
                continue
            entity_key = (mem.get("entity_name") or "").strip().lower()
            if per_entity.get(entity_key, 0) >= self._config.per_entity_cap:
                continue
            seen_facts.add(norm)
            per_entity[entity_key] = per_entity.get(entity_key, 0) + 1
            selected.append(mem)
            if len(selected) >= self._config.max_facts:
                break
        return selected

    def _render(self, selected: list[dict]) -> str:
        lines = [_HEADER, "", _PREAMBLE, ""]
        block_len = sum(len(line) + 1 for line in lines)
        rendered_any = False
        for mem in selected:
            entity = (mem.get("entity_name") or "").strip()
            fact = (mem.get("fact") or "").strip()
            marker = "[user note] " if mem.get("manual") else ""
            line = f"- {marker}{entity}: {fact}" if entity else f"- {marker}{fact}"
            if block_len + len(line) + 1 > self._config.max_chars:
                break
            lines.append(line)
            block_len += len(line) + 1
            rendered_any = True
        if not rendered_any:
            return ""
        return "\n".join(lines)
