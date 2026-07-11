"""Async memory extractor — batch-processes conversation messages into structured facts.

Adapted from OpenClaw's production memory_extractor (battle-tested, 30-min cadence).
Changes from original:
  - Reads from Prometheus MemoryStore (messages table) instead of Archive SQLite
  - Calls ModelProvider instead of Claude API directly
  - Retains identical extraction prompt, entity categories, confidence scoring,
    deduplication logic, and batch size (10-20 events per call)
  - 2026-07 wiki-quality audit: machine-harness sessions (bakeoff/eval,
    coding mode, gym, smoke, "system") are excluded from mining, and a fact
    that paraphrases an existing one folds into that row before persist
  - Writes facts to the SQLite memories table only; the human-facing wiki
    under ``~/.prometheus/wiki/`` is a pure projection rendered from that
    store by the WikiCompiler (there is no second markdown writer here)

TRUST-CONTEXT: this extractor is an autonomous, model-driven write path
— there is no human in the loop to sanction each fact. Its write surface
is SQLite only (``MemoryStore``); the wiki is regenerated from the store,
never written to directly from here.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import TYPE_CHECKING, Callable

from prometheus.memory.entity_validation import classify_entity, quarantine
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

# ------------------------------------------------------------------
# Extraction hygiene (2026-07 wiki-quality audit)
# ------------------------------------------------------------------

# Machine-harness session families: eval/bakeoff runs, coding mode, gym
# harvests, smoke scripts, and the reserved "system" id. Their chatter is
# fixture/eval material, not conversation with the user — mining it filed the
# eval library "marshmallow" as a client organization and dozens of
# path-trivia "facts" about the user. Module constants are the per-install
# tuning surface (same idiom as entity_validation).
_MACHINE_SESSION_PREFIXES: tuple[str, ...] = (
    "bakeoff:",
    "coding:",
    "eval:",
    "gym:",
    "smoke:",
)
_MACHINE_SESSION_IDS = frozenset({"system"})


def _is_machine_session(session_id: str | None) -> bool:
    """True iff *session_id* belongs to a machine harness, not a user chat."""
    sid = (session_id or "").strip()
    return sid in _MACHINE_SESSION_IDS or sid.startswith(_MACHINE_SESSION_PREFIXES)


# Near-duplicate folding: the model re-states the same fact in slightly
# different words on every pass (people/will.md accumulated ~80 path-trivia
# paraphrases). ``persist_memory`` dedups exact normalized matches only, so
# paraphrases pile up as new rows. A new fact whose token set overlaps an
# existing fact of the same entity at or above this threshold is folded into
# that row instead. Deterministic — no embeddings, no network.
_NEAR_DUP_THRESHOLD = 0.75
_NEAR_DUP_SCAN_LIMIT = 200  # store rows scanned per entity when folding

# Scaffolding the extraction model wraps every fact in ("The user's ...",
# "... appears to be ..."). Excluded from the similarity token set so
# paraphrases differing only in scaffolding fold together.
_FACT_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for",
    "from", "has", "have", "her", "his", "in", "is", "it", "its", "of", "on",
    "or", "s", "that", "the", "their", "this", "to", "was", "were", "with",
    "user", "users",
    "also", "appear", "appears", "based", "called", "indicates", "indicating",
    "likely", "named", "seems", "suggesting", "suggests",
})
_FACT_TOKEN_RE = re.compile(r"[a-z0-9.]+")


def _fact_token_set(fact: str) -> frozenset[str]:
    """Content tokens of a fact: lowercased, stopwords dropped, edge dots
    stripped (keeps IPs/versions whole while shedding sentence periods)."""
    tokens = (t.strip(".") for t in _FACT_TOKEN_RE.findall((fact or "").lower()))
    return frozenset(t for t in tokens if t and t not in _FACT_STOPWORDS)


def _near_dup_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    """max(Jaccard, containment) over two token sets.

    Containment catches "same fact plus extra inferred fluff", which pure
    Jaccard under-scores.
    """
    if not a or not b:
        return 0.0
    overlap = len(a & b)
    return max(overlap / len(a | b), overlap / min(len(a), len(b)))


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
        batch_size: int = _BATCH_SIZE,
        post_extract_callback: Callable[[list[dict]], None] | None = None,
        signal_bus: object | None = None,
        telemetry: object | None = None,
        lcm_conversation_store: object | None = None,
    ) -> None:
        from prometheus.learning.llm_envelope import LLMCallEnvelope

        self._store = store  # facts store (.persist_memory + .search_memories)
        self._provider = provider
        self._model = model
        self._batch_size = batch_size
        self._post_extract_callback = post_extract_callback
        self._signal_bus = signal_bus
        self._last_run: float = 0.0
        self._last_processed_ts: float = 0.0
        # PR fix/memory-lcm-full-rewire (2026-05-26): conversation reads
        # now come from LCM, not MemoryStore.messages (which was unwired
        # — nothing produced to it). If ``lcm_conversation_store`` is
        # None at construction time (e.g. CLI / unit tests where LCM
        # isn't set up), ``run_once`` lazily looks it up via
        # ``LCMEngine.conversation_store`` if a wired engine is later
        # set on the daemon. The injection point keeps unit tests simple.
        self._lcm_conv_store = lcm_conversation_store
        # Sprint S4 A1: shared LLMCallEnvelope. on_failure="return_none" so
        # _process_batch preserves its "returns (0, []) on failure" contract
        # without the redundant try/except. Failures still land in
        # telemetry.silent_failures with full traceback.
        self._telemetry = telemetry
        self._envelope = LLMCallEnvelope(
            subsystem="memory_extractor",
            telemetry=telemetry,
            on_failure="return_none",
        )

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

        PR fix/memory-lcm-full-rewire (2026-05-26): read path is now
        LCMConversationStore.messages_since(self._last_processed_ts).
        The watermark semantics — strictly greater than, global across
        sessions (when session_id is None), excludes compacted — match
        the pre-PR MemoryStore.messages query exactly.
        """
        since = self._last_processed_ts
        conv_store = self._resolve_lcm_conv_store()
        if conv_store is None:
            log.debug(
                "MemoryExtractor: LCM conversation store unavailable, "
                "skipping pass"
            )
            return 0, []

        # LCM read — returns list[MessagePart]. Convert to the dict shape
        # _process_batch / _format_messages expect (matching the legacy
        # MemoryStore.messages row dict: id, session_id, role, content,
        # timestamp). Token counts come from the MessagePart for free
        # but aren't used downstream.
        parts = conv_store.messages_since(
            since, limit=500, session_id=session_id
        )

        # Watermark advances over EVERY row read this pass — including the
        # non-user rows skipped below — so untrusted turns aren't re-scanned on
        # each cadence. Computed before filtering for exactly that reason.
        max_ts_seen = max((part.timestamp for part in parts), default=None)

        # TRUST-CONTEXT: only mine genuine conversation. Injected non-"user"
        # provenance turns (task_supervisor job output now; cron / orchestrator
        # later) are UNTRUSTED data, not statements by the user — they must
        # never be extracted as user facts. Filter on the PERSISTED provenance
        # column (durable as of the trust-tag plumbing); never re-derive trust
        # from message text.
        user_parts = [
            part for part in parts
            if (getattr(part, "provenance", "user") or "user") == "user"
        ]
        skipped = len(parts) - len(user_parts)
        if skipped:
            log.debug(
                "MemoryExtractor: skipped %d non-user-provenance message(s) — "
                "not mined into memory", skipped,
            )

        # HYGIENE: drop machine-harness sessions (evals/bakeoff, coding mode,
        # gym, smoke, "system"). Their fixture chatter is not knowledge about
        # the user — see _MACHINE_SESSION_PREFIXES.
        conversational_parts = [
            part for part in user_parts
            if not _is_machine_session(part.session_id)
        ]
        machine_skipped = len(user_parts) - len(conversational_parts)
        if machine_skipped:
            log.debug(
                "MemoryExtractor: skipped %d machine-session message(s) — "
                "not mined into memory", machine_skipped,
            )

        messages = [
            {
                "id": part.message_id,
                "session_id": part.session_id,
                "role": part.role,
                "content": part.content,
                "timestamp": part.timestamp,
            }
            for part in conversational_parts
        ]

        if not messages:
            # Still advance past any skipped non-user rows so they aren't re-read.
            if max_ts_seen is not None:
                self._last_processed_ts = max(self._last_processed_ts, max_ts_seen)
            log.debug("MemoryExtractor: no new user-provenance messages to process")
            return 0, []

        total_persisted = 0
        all_facts: list[dict] = []
        for i in range(0, len(messages), self._batch_size):
            batch = messages[i : i + self._batch_size]
            persisted, facts = await self._process_batch(batch)
            total_persisted += persisted
            all_facts.extend(facts)

        # Advance over all rows seen this pass (mined + skipped), so a trailing
        # run of skipped task turns isn't re-read on the next pass.
        if max_ts_seen is not None:
            self._last_processed_ts = max(self._last_processed_ts, max_ts_seen)
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

        # Envelope returns None on failure (writes telemetry.silent_failures).
        # Drops the redundant outer try/except per Sprint 4 audit.
        raw = await self._call_model(prompt)
        if raw is None:
            return 0, []

        facts = self._parse_facts(raw)
        source_ids = [m["id"] for m in messages]
        persisted = 0
        persisted_facts: list[dict] = []
        for fact in facts:
            # Structural entity gate: never persist junk (paths, code
            # identifiers, shell syntax, over-long task strings) as entities.
            # Rejections are quarantined for inspection, not silently dropped.
            entity_name = fact.get("entity_name", "")
            reason = classify_entity(entity_name)
            if reason is not None:
                quarantine(str(entity_name), reason, context="extractor")
                continue
            # Near-duplicate folding: rewrite a paraphrase to the stored
            # canonical text so persist_memory's exact-normalized dedup merges
            # it into the existing row (mention_count++, sources unioned)
            # instead of minting a new one.
            canonical = self._fold_near_duplicate(entity_name, fact["fact"])
            if canonical is not None:
                fact["fact"] = canonical
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
                # Carry provenance into the dict handed to the WikiCompiler
                # callback. The source ids are persisted to memory.db above,
                # but _parse_facts() produced this dict without them, so the
                # wiki rendered "source: unknown". Re-attach the real ids.
                fact["source_event_ids"] = source_ids
                persisted += 1
                persisted_facts.append(fact)
            except Exception:
                log.exception("MemoryExtractor: failed to persist fact: %s", fact)
        return persisted, persisted_facts

    def _fold_near_duplicate(self, entity_name: str, fact: object) -> str | None:
        """Return the stored fact text *fact* is a near-duplicate of, or ``None``.

        Best-effort — any lookup failure means "no fold", never a failed pass.
        """
        if not isinstance(fact, str):
            return None
        new_tokens = _fact_token_set(fact)
        if not new_tokens:
            return None
        try:
            candidates = self._store.search_memories(
                entity=entity_name, limit=_NEAR_DUP_SCAN_LIMIT
            )
            target = entity_name.strip().lower()
            for row in candidates:
                # search_memories(entity=) is a LIKE %...% match — hold
                # folding to exact (case-insensitive) entity identity.
                if (row.get("entity_name") or "").strip().lower() != target:
                    continue
                existing = row.get("fact") or ""
                sim = _near_dup_similarity(new_tokens, _fact_token_set(existing))
                if sim >= _NEAR_DUP_THRESHOLD:
                    if existing != fact:
                        log.debug(
                            "MemoryExtractor: folding near-duplicate fact for %r"
                            " (similarity %.2f): %r -> %r",
                            entity_name, sim, fact, existing,
                        )
                    return existing
        except Exception:
            log.debug(
                "MemoryExtractor: near-duplicate scan failed for %r — "
                "persisting without folding", entity_name, exc_info=True,
            )
        return None

    async def _call_model(self, prompt: str) -> str | None:
        """Invoke the model via LLMCallEnvelope. Returns None on failure."""
        return await self._envelope.call(
            provider=self._provider,
            model=self._model,
            prompt=prompt,
            max_tokens=2048,
            operation="extract_memory_batch",
        )

    def _resolve_lcm_conv_store(self) -> object | None:
        """Return the LCMConversationStore handle, or ``None`` if unavailable.

        Resolution order:
          1. Explicitly-injected ``lcm_conversation_store`` from ``__init__``
             (used by unit tests for direct control)
          2. The conversation store on the module-level LCM engine, if a
             daemon has wired one via ``LCMEngine`` initialisation
             (production path)

        Returns ``None`` if neither path yields a store — in that case
        ``run_once`` skips the pass without raising.
        """
        if self._lcm_conv_store is not None:
            return self._lcm_conv_store
        try:
            from prometheus.tools.builtin import lcm_grep
        except Exception:
            return None
        engine = getattr(lcm_grep, "_engine", None)
        if engine is None:
            return None
        try:
            return engine.conversation_store
        except Exception:
            return None

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
