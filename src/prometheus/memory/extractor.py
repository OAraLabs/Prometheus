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
  - 2026-08 cursor fix: extraction progress is a DURABLE per-scope ROWID
    cursor in ``memory.db`` (``extractor_cursors``). It replaced a single
    in-memory ``_last_processed_ts`` float that served two scopes at once and
    so failed in BOTH directions from one root cause: it reset to 0.0 on every
    restart (re-mining the 500 oldest uncompacted rows across ALL sessions —
    and because ``persist_memory`` increments ``mention_count`` on a dedup hit
    while ``mention_count >= 2`` is the wiki page-worthiness threshold, a
    re-mine could promote a one-off mention to a wiki page), while a
    per-session pre-compaction flush advanced the same global value and
    stranded every other session's older rows below it permanently

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
from prometheus.memory.store import EXTRACTOR_GLOBAL_SCOPE, MemoryStore

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
# Promoted to memory/session_kind.py — retention, /api/status and the clients need the same
# answer, and a second copy of this list is how one of them drifts. Re-exported under the old
# private names so this module's existing callers and tests keep working.
from prometheus.memory.session_kind import (  # noqa: E402
    MACHINE_SESSION_IDS as _MACHINE_SESSION_IDS,
    MACHINE_SESSION_PREFIXES as _MACHINE_SESSION_PREFIXES,
    is_machine_session as _is_machine_session,
)


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
        # Progress is now a DURABLE, PER-SCOPE ROWID cursor in ``memory.db``
        # (``extractor_cursors``), not an in-memory timestamp. The float this
        # replaced was a single value serving two scopes, and it failed in both
        # directions at once — see :meth:`_cursor_for` and the module history.
        self._seeded = False
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
        # Seed the migration floor AT CONSTRUCTION, not lazily on the first
        # pass. Doing it lazily looked equivalent and was not: the first pass is
        # 30 minutes after boot (or whenever a compaction flush fires), so
        # everything said in between would have been seeded PAST and never
        # mined — the under-extraction failure this whole change exists to cure,
        # reintroduced by its own migration. Caught by six pre-existing tests
        # going red, not by reading the code.
        #
        # It is a one-time event (INSERT OR IGNORE on an absent cursor), so the
        # only rows it can skip are those already in the store when the daemon
        # first boots after this lands. In ``daemon.py`` the gateways start at
        # ~541 and this runs at ~862, so a message arriving inside that startup
        # window on that single boot is the residual, bounded loss.
        self._seed_cursor_once(self._resolve_lcm_conv_store())

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

        Progress is a DURABLE ROWID cursor, per scope
        (:meth:`_cursor_for`). ``session_id=None`` is the cross-session sweep
        and reads against the reserved global floor; a ``session_id`` (the
        pre-compaction flush) reads against that session's own cursor and
        advances only it, so one session's flush can no longer strand another
        session's rows.
        """
        conv_store = self._resolve_lcm_conv_store()
        if conv_store is None:
            log.debug(
                "MemoryExtractor: LCM conversation store unavailable, "
                "skipping pass"
            )
            return 0, []

        self._seed_cursor_once(conv_store)
        since_row_id = self._cursor_for(session_id)

        # LCM read — returns list[MessagePart]. Convert to the dict shape
        # _process_batch / _format_messages expect (matching the legacy
        # MemoryStore.messages row dict: id, session_id, role, content,
        # timestamp). Token counts come from the MessagePart for free
        # but aren't used downstream.
        #
        # ``include_compacted=False`` preserves the previous read's semantics —
        # the extractor must not re-process summaries-of-summaries. Note the
        # consequence, unchanged from the timestamp cursor but worth stating: a
        # row compacted before extraction reaches it is skipped AND the cursor
        # moves past it. Compaction is therefore still the deadline it always
        # was; this fix is about restarts, not about that race.
        parts = conv_store.messages_after_id(
            since_row_id, limit=500, session_id=session_id,
            include_compacted=False,
        )

        # The cursor advances over EVERY row READ this pass — including rows
        # dropped just below as already-consumed, and the non-user /
        # machine-session rows filtered out further down — so skipped turns
        # aren't re-scanned on each cadence. Computed from the unfiltered read
        # for exactly that reason, and per session, because that is the scope
        # the cursor is keyed on.
        max_row_by_scope = self._max_row_by_scope(parts, session_id)

        # A sweep's single global-floor query cannot express each session's own
        # cursor; this applies it per row. See _drop_already_consumed.
        parts = self._drop_already_consumed(parts, session_id)

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
            self._advance_cursors(max_row_by_scope)
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
        self._advance_cursors(max_row_by_scope)
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
    # Cursors
    #
    # THE INVARIANT: a per-session cursor is AUTHORITATIVE when present; the
    # reserved global scope is what an ABSENT one resolves to. Everything below
    # exists to keep that true, because it is what makes each row mined at most
    # once no matter which entry point reaches it first:
    #
    #   sweep mines to rowid N  → cursor[each touched session] = its max,
    #                             cursor['*'] = N
    #   flush(X) reads > cursor[X]                → never re-reads the sweep's rows
    #   flush(X) mines to M     → cursor[X] = M    (global untouched)
    #   next sweep reads X from cursor[X] = M      → never re-reads the flush's rows
    #   brand-new session Y     → no cursor row, resolves to '*' = N,
    #                             and every row of Y is > N anyway
    #
    # The version this replaced had ONE float for both scopes, so the third and
    # fourth lines collided: a flush advanced the value the sweep read from.
    # ------------------------------------------------------------------

    def _cursor_for(self, session_id: str | None) -> int:
        """The rowid floor for the QUERY.

        ``max`` of the global floor and the session's own cursor — never
        either alone. A session cursor ahead of the global floor means a flush
        already consumed those rows; a global floor ahead of the session cursor
        means a sweep did. Taking the max is the only choice that cannot
        re-expose rows either one has passed.
        """
        global_floor = self._store.get_extractor_cursor(EXTRACTOR_GLOBAL_SCOPE)
        if session_id is None:
            return global_floor
        return max(global_floor, self._store.get_extractor_cursor(session_id))

    def _drop_already_consumed(self, parts: list, session_id: str | None) -> list:
        """Per-row floor for the cross-session sweep.

        THE GAP THIS CLOSES, found by its own test rather than by reading the
        code: the sweep issues ONE query against the global floor, so it cannot
        express "and also respect each session's own cursor". A flush that ran
        ahead of the global floor left its rows selectable by the next sweep,
        which re-mined them — the same double-mine the whole fix exists to
        prevent, just moved one layer down.

        Single-session passes need nothing here: :meth:`_cursor_for` already
        folded that session's cursor into the query floor.
        """
        if session_id is not None:
            return parts
        floors: dict[str, int] = {}
        kept = []
        for part in parts:
            sid = part.session_id or ""
            if sid not in floors:
                floors[sid] = self._store.get_extractor_cursor(sid) if sid else 0
            if int(getattr(part, "row_id", 0) or 0) > floors[sid]:
                kept.append(part)
        dropped = len(parts) - len(kept)
        if dropped:
            log.debug(
                "MemoryExtractor: %d row(s) already consumed by a per-session "
                "flush — not re-mined", dropped,
            )
        return kept

    @staticmethod
    def _max_row_by_scope(parts: list, session_id: str | None) -> dict[str, int]:
        """``{scope: max rowid seen}`` for the rows this pass read.

        A sweep touches many sessions, so it advances each session's own cursor
        plus the global floor. A single-session pass advances only that session
        — deliberately NOT the global floor, which is the whole fix: a
        pre-compaction flush must not move the value every other session is
        read against.
        """
        out: dict[str, int] = {}
        for part in parts:
            row_id = int(getattr(part, "row_id", 0) or 0)
            if not row_id:
                continue
            sid = part.session_id
            if sid:
                out[sid] = max(out.get(sid, 0), row_id)
            if session_id is None:
                out[EXTRACTOR_GLOBAL_SCOPE] = max(
                    out.get(EXTRACTOR_GLOBAL_SCOPE, 0), row_id
                )
        return out

    def _advance_cursors(self, max_row_by_scope: dict[str, int]) -> None:
        """Persist the pass's progress. Best-effort per scope.

        A cursor that fails to advance means rows are re-read next pass — noisy
        but not lossy — so a write failure must not abort the pass. It IS
        logged: silently failing to advance is how a cursor becomes decorative.
        """
        for scope, row_id in max_row_by_scope.items():
            try:
                self._store.set_extractor_cursor(scope, row_id)
            except Exception:
                log.warning(
                    "MemoryExtractor: failed to advance cursor for scope %r to "
                    "rowid %d — those rows will be re-read next pass",
                    scope, row_id, exc_info=True,
                )

    def _seed_cursor_once(self, conv_store: object | None) -> None:
        """Seed the global floor from the store's CURRENT max rowid, once.

        See :meth:`MemoryStore.seed_extractor_cursor_if_absent` for the trade.
        Called from ``__init__``; retried on the first pass only if the store
        was not yet resolvable then (late-wired CLI paths). A no-op the moment
        a cursor row exists, so it can never rewind a live cursor.
        """
        if self._seeded or conv_store is None:
            return
        self._seeded = True
        try:
            existing = self._store.get_extractor_cursor(EXTRACTOR_GLOBAL_SCOPE)
            if existing:
                return
            # The global floor comes from the TABLE's max rowid, never from
            # list_sessions(): that aggregate excludes tombstoned sessions
            # ("forget session"), so a floor derived from it sits BELOW a
            # forgotten chat's rows and the first pass re-mines them. The live
            # box has 5 tombstones and would have been correct only by
            # coincidence (its highest row happens not to be tombstoned).
            high = int(conv_store.max_rowid_all())  # type: ignore[attr-defined]
            # Per-session AUDIT rows still come from list_sessions(); a
            # tombstoned session correctly gets none and falls back to the
            # global floor, which is now guaranteed to be at or above its rows.
            per_session: dict[str, int] = {}
            for row in conv_store.list_sessions():  # type: ignore[attr-defined]
                sid = row.get("session_id")
                if sid:
                    per_session[str(sid)] = int(row.get("watermark") or 0)
            if self._store.seed_extractor_cursor_if_absent(high, per_session):
                log.info(
                    "MemoryExtractor: seeded the extraction cursor at rowid %d "
                    "across %d session(s) (existing history treated as already "
                    "consumed — see seed_extractor_cursor_if_absent for the "
                    "trade)", high, len(per_session),
                )
        except Exception:
            log.warning(
                "MemoryExtractor: cursor seeding failed; this pass falls back "
                "to whatever cursor exists", exc_info=True,
            )

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
