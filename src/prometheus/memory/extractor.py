"""Async memory extractor — batch-processes conversation messages into structured facts.

Adapted from OpenClaw's production memory_extractor (battle-tested, 30-min cadence).
Changes from original:
  - Reads from Prometheus MemoryStore (messages table) instead of Archive SQLite
  - Calls ModelProvider instead of Claude API directly
  - Retains identical extraction prompt, entity categories, confidence scoring,
    deduplication logic, and batch size (10-20 events per call)
  - Dual-writes to SQLite memories table + optional Obsidian vault
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from prometheus.memory.store import MemoryStore

if TYPE_CHECKING:
    from prometheus.providers.base import ModelProvider

log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Extraction prompt (kept verbatim from OpenClaw production)
# ------------------------------------------------------------------

_EXTRACTION_PROMPT = """\
You are a memory extraction system. Analyze the following conversation messages
and extract structured facts about entities mentioned.

For each fact, output a JSON object on its own line with these fields:
  entity_type: one of person, place, organization, task, tool, concept, preference
  entity_name: the specific name of the entity
  relationship: how this entity relates to the user (e.g. "colleague", "uses daily", "works at")
  fact: a single concrete, specific statement about the entity
  confidence: float 0.0-1.0 based on how explicitly stated the fact is
  tags: list of relevant keyword strings

Rules:
- Only extract facts that are clearly stated, not inferred.
- One fact per JSON object. Multiple objects for multiple facts.
- Skip generic statements ("the user said hello").
- Confidence >= 0.8: explicitly stated. 0.5-0.8: implied. < 0.5: uncertain.
- Output ONLY JSON objects, one per line. No prose, no markdown.

Messages:
{messages}
"""

_BATCH_SIZE = 15  # messages per extraction call
_DEFAULT_CADENCE_SECONDS = 1800  # 30 minutes


class MemoryExtractor:
    """Extract structured entity facts from conversation history.

    Usage (standalone):
        extractor = MemoryExtractor(store, provider)
        await extractor.run_once()

    Usage (background loop):
        await extractor.run_forever(interval=1800)
    """

    def __init__(
        self,
        store: MemoryStore,
        provider: ModelProvider,
        *,
        model: str = "default",
        obsidian_writer: ObsidianWriter | None = None,
        batch_size: int = _BATCH_SIZE,
        post_extract_callback: Callable[[list[dict]], None] | None = None,
        signal_bus: object | None = None,
    ) -> None:
        self._store = store
        self._provider = provider
        self._model = model
        self._obsidian = obsidian_writer
        self._batch_size = batch_size
        self._post_extract_callback = post_extract_callback
        self._signal_bus = signal_bus
        self._last_run: float = 0.0
        self._last_processed_ts: float = 0.0

    @property
    def signal_bus(self) -> object | None:
        return self._signal_bus

    @signal_bus.setter
    def signal_bus(self, bus: object) -> None:
        """Set signal bus after construction (used by daemon.py wiring)."""
        self._signal_bus = bus

    async def run_once(self, session_id: str | None = None) -> tuple[int, list[dict]]:
        """Run one extraction pass.

        Returns ``(count_persisted, list_of_fact_dicts)`` so callers
        (e.g. WikiCompiler) can act on the freshly-extracted facts.
        """
        since = self._last_processed_ts
        if session_id:
            messages = self._store.get_messages(
                session_id, since=since or None, compressed=False, limit=500
            )
        else:
            messages = self._store._conn.execute(
                "SELECT * FROM messages WHERE compressed = 0"
                + (" AND timestamp > ?" if since else "")
                + " ORDER BY timestamp ASC LIMIT 500",
                (since,) if since else (),
            ).fetchall()
            messages = [dict(m) for m in messages]

        if not messages:
            log.debug("MemoryExtractor: no new messages to process")
            return 0, []

        total_persisted = 0
        all_facts: list[dict] = []
        for i in range(0, len(messages), self._batch_size):
            batch = messages[i : i + self._batch_size]
            persisted, facts = await self._process_batch(batch)
            total_persisted += persisted
            all_facts.extend(facts)

        if messages:
            self._last_processed_ts = max(float(m["timestamp"]) for m in messages)
        self._last_run = time.time()
        log.info("MemoryExtractor: persisted %d memories from %d messages", total_persisted, len(messages))

        # SENTINEL: emit extraction_complete signal (Sprint 9)
        if self._signal_bus and all_facts:
            try:
                from prometheus.sentinel.signals import ActivitySignal
                await self._signal_bus.emit(ActivitySignal(
                    kind="extraction_complete",
                    payload={"count": total_persisted, "facts": len(all_facts)},
                    source="memory_extractor",
                ))
            except Exception:
                log.debug("MemoryExtractor: signal emission failed (SENTINEL not available)")

        return total_persisted, all_facts

    async def run_forever(
        self,
        interval: float = _DEFAULT_CADENCE_SECONDS,
        session_id: str | None = None,
    ) -> None:
        """Run extraction on a repeating interval (default 30 minutes)."""
        log.info("MemoryExtractor: starting background loop every %.0fs", interval)
        while True:
            try:
                _count, facts = await self.run_once(session_id=session_id)
                if facts and self._post_extract_callback:
                    try:
                        self._post_extract_callback(facts)
                    except Exception:
                        log.exception("MemoryExtractor: post-extract callback failed")
            except Exception:
                log.exception("MemoryExtractor: extraction pass failed")
            await asyncio.sleep(interval)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _process_batch(self, messages: list[dict]) -> tuple[int, list[dict]]:
        """Send one batch to the LLM and persist extracted facts.

        Returns ``(count_persisted, list_of_persisted_fact_dicts)``.
        """
        formatted = self._format_messages(messages)
        prompt = _EXTRACTION_PROMPT.format(messages=formatted)

        try:
            raw = await self._call_model(prompt)
        except Exception:
            log.exception("MemoryExtractor: model call failed for batch of %d", len(messages))
            return 0, []

        facts = self._parse_facts(raw)
        source_ids = [m["id"] for m in messages]
        persisted = 0
        persisted_facts: list[dict] = []
        for fact in facts:
            try:
                self._store.persist_memory(
                    entity_type=fact.get("entity_type", "concept"),
                    entity_name=fact["entity_name"],
                    fact=fact["fact"],
                    confidence=float(fact.get("confidence", 0.5)),
                    relationship=fact.get("relationship"),
                    source_event_ids=source_ids,
                    tags=fact.get("tags", []),
                )
                if self._obsidian:
                    self._obsidian.write_fact(fact)
                persisted += 1
                persisted_facts.append(fact)
            except Exception:
                log.exception("MemoryExtractor: failed to persist fact: %s", fact)
        return persisted, persisted_facts

    async def _call_model(self, prompt: str) -> str:
        """Call the ModelProvider and return the full text response."""
        from prometheus.engine.messages import ConversationMessage
        from prometheus.providers.base import ApiMessageRequest

        request = ApiMessageRequest(
            model=self._model,
            messages=[ConversationMessage(role="user", content=prompt)],
            max_tokens=2048,
        )
        text_parts: list[str] = []
        async for event in self._provider.stream_message(request):
            from prometheus.providers.base import ApiTextDeltaEvent
            if isinstance(event, ApiTextDeltaEvent):
                text_parts.append(event.text)
        return "".join(text_parts)

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        lines: list[str] = []
        for m in messages:
            role = m.get("role", "unknown")
            content = m.get("content", "")
            lines.append(f"[{role}]: {content}")
        return "\n".join(lines)

    @staticmethod
    def _parse_facts(raw: str) -> list[dict]:
        """Parse newline-delimited JSON objects from model output."""
        facts: list[dict] = []
        required = {"entity_name", "fact"}
        for line in raw.splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
                if required.issubset(obj.keys()):
                    facts.append(obj)
            except json.JSONDecodeError:
                continue
        return facts


# ------------------------------------------------------------------
# ObsidianWriter — optional dual-write to Obsidian vault
# ------------------------------------------------------------------


class ObsidianWriter:
    """Write extracted memory facts to an Obsidian vault for human readability.

    Each entity gets its own markdown note under ``vault_path/Memory/<entity_name>.md``.
    """

    def __init__(self, vault_path: str | Path) -> None:
        self._vault = Path(vault_path).expanduser()
        self._memory_dir = self._vault / "Memory"
        self._memory_dir.mkdir(parents=True, exist_ok=True)

    def write_fact(self, fact: dict) -> None:
        """Append a fact to the entity's note in the vault."""
        entity_name = fact.get("entity_name", "Unknown")
        safe_name = re.sub(r'[<>:"/\\|?*]', "_", entity_name)
        note_path = self._memory_dir / f"{safe_name}.md"

        timestamp = time.strftime("%Y-%m-%d %H:%M", time.localtime())
        confidence = fact.get("confidence", 0.0)
        relationship = fact.get("relationship", "")
        fact_text = fact.get("fact", "")
        tags = fact.get("tags", [])

        entry_lines = [
            f"\n## {timestamp}",
            f"- **Relationship**: {relationship}" if relationship else None,
            f"- **Fact**: {fact_text}",
            f"- **Confidence**: {confidence:.2f}",
            f"- **Tags**: {', '.join(tags)}" if tags else None,
        ]
        entry = "\n".join(line for line in entry_lines if line is not None)

        if not note_path.exists():
            header = (
                f"# {entity_name}\n\n"
                f"**Entity Type**: {fact.get('entity_type', 'unknown')}\n"
            )
            note_path.write_text(header, encoding="utf-8")

        with note_path.open("a", encoding="utf-8") as fh:
            fh.write(entry + "\n")
