"""Retract what the assistant learned from a chat the owner is purging.

``LCMConversationStore.purge_session`` removes a chat's messages and summaries. Memory, the wiki and
the insight pages are DERIVED from chats, and a purge that stops at the conversation database leaves
a pasted secret alive as a fact, a wiki line and an insight. This module undoes the derivation.

THE RULE (the owner's)
----------------------
* A fact learned ONLY from the purged chat is deleted.
* A fact a surviving chat also supports stays, and the purged chat stops being one of its sources.
* A note the owner saved with ``/note`` is KEPT by default, listed in the preview as such, and
  forgotten only when the purge asks for it (``forget_notes``). Then it is judged like any fact.
* The owner sees what will go before confirming (:meth:`RetractionPlan.to_preview`).

A note records no chat (its one source is :data:`~prometheus.memory.store.MANUAL_SOURCE`), so a
purge can only link a note to a chat through dedup: a ``/note`` that matched a fact the extractor had
learned there was merged into that fact. Those merged notes are what ``forget_notes`` is about.

WHAT "ALSO SUPPORTED" HAS TO MEAN
---------------------------------
A fact's ``source_event_ids`` is batch-level, not exact. The extractor sweeps new messages across
every chat in rowid order, cuts them into batches, and every fact from a batch cites every message
in it (``MemoryExtractor._process_batch``). One batch can mix chats, so a surviving id may be a
batch-mate of the purged message and say nothing about who taught the fact. Read literally, "still
has a surviving source" would keep a fact learned only from the purged chat, which is the one
outcome a purge must not allow.

So a surviving source counts as support only if that message actually mentions the fact's entity
(:func:`_supports`). The cost is deliberate and stated: a fact that a surviving chat expressed
without naming the entity (a pronoun, an alias) can be removed although that chat supports it. A
purge exists so that the purged content does not survive, and the preview lists every removal with
its reason, so the owner sees it first. The way out is to record the source chat per fact at write
time; until then this is the honest reading of the data.

ORDER, AND WHY
--------------
The chat's message ids come from the LCM store and the id-to-chat mapping disappears with the
purge, so the plan is made, and applied, BEFORE the conversation rows go; a sweep with the same ids
afterwards catches a fact the extractor wrote from those messages while the purge was running. Each
step is idempotent. The one step that is not atomic with the rest is the wiki rebuild (files), so
the memory transaction records a mark and the next apply finishes the rebuild if it never ran.

Source: novel code for Prometheus, 2026-10-23.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from prometheus.memory.wiki_compiler import WikiCompiler, entity_page_relpath

if TYPE_CHECKING:  # pragma: no cover
    from prometheus.memory.lcm_conversation_store import LCMConversationStore
    from prometheus.memory.store import MemoryStore

log = logging.getLogger(__name__)

#: Set in the same transaction as the fact changes and cleared once the wiki is rebuilt, so a rebuild
#: that failed after the facts were committed is finished by the next apply even though that apply's
#: own plan finds nothing left to retract.
WIKI_REGEN_PENDING_MARK = "purge_wiki_regen_pending"

#: The preview lists at most this many entries per list; ``totals`` always carries the real counts.
PREVIEW_LIST_CAP = 200

_ALL = 1_000_000
#: Entity names shorter than this are not searched for in filed answers: "AI" would match everything.
_MIN_REVIEW_NAME = 3

#: Said to the owner with every preview and purge, so the promise is not wider than the code.
NOT_COVERED = (
    "Notes the assistant wrote into its own MEMORY.md or USER.md files.",
    "Skills or skill drafts the learning loop wrote from this chat.",
    "Telemetry rows (tool-call records carry the session id).",
    "Forks of this chat; they are listed as branches and are purged separately.",
    "Text in other chats that quotes this one.",
    "Notes you saved with /note that were never merged into a fact learned from this chat: a note "
    "records no chat, so there is nothing to link it to.",
)


def _supports(content: str, entity_name: str) -> bool:
    """Does a surviving message actually mention the fact's entity? See the module docstring."""
    needle = (entity_name or "").strip().casefold()
    return bool(needle) and needle in (content or "").casefold()


def _mentions_any(text: str, names: Iterable[str]) -> bool:
    folded = text.casefold()
    for name in names:
        if len(name) >= _MIN_REVIEW_NAME and re.search(
            rf"(?<!\w){re.escape(name)}(?!\w)", folded
        ):
            return True
    return False


def _frontmatter(text: str) -> dict[str, Any]:
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    try:
        data = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def _unlink_inside(root: Path, relpath: str) -> None:
    """Delete ``root/relpath`` if it is inside ``root``; refuse anything that resolves outside."""
    target = (root / relpath).resolve()
    if root.resolve() not in target.parents:
        raise ValueError(f"refusing to delete {relpath!r}: it resolves outside the wiki")
    target.unlink(missing_ok=True)


@dataclass(frozen=True)
class RetractionPlan:
    """What retracting one chat would do. Built read-only; applied by :func:`apply_retraction`."""

    session_id: str
    purged_ids: frozenset[str]
    #: Facts to delete: the fact row plus ``reason``.
    removed: tuple[dict, ...] = ()
    #: Facts that stay and lose the purged chat as a source.
    kept: tuple[dict, ...] = ()
    #: ``/note`` facts that stay because the owner did not ask to forget notes. They lose the purged
    #: chat as a source too, and are listed apart from ``kept`` so the owner sees them as notes.
    notes_kept: tuple[dict, ...] = ()
    #: The choice this plan was made under, so the sweep after the purge makes the same one.
    forget_notes: bool = False
    legacy_summary_ids: tuple[str, ...] = ()
    legacy_message_count: int = 0
    has_cursor: bool = False
    pages_removed: tuple[str, ...] = ()
    pages_rebuilt: tuple[str, ...] = ()
    insight_pages_removed: tuple[str, ...] = ()
    queries_to_review: tuple[str, ...] = ()

    @property
    def empty(self) -> bool:
        return not (
            self.removed or self.kept or self.notes_kept or self.legacy_summary_ids
            or self.legacy_message_count
            or self.has_cursor or self.insight_pages_removed
        )

    def to_preview(self, *, limit: int = PREVIEW_LIST_CAP) -> dict[str, Any]:
        """The owner-facing description of the plan: what goes, what stays, what is rebuilt."""
        removed = [
            {k: f[k] for k in ("id", "entity_name", "entity_type", "fact", "manual", "reason")}
            for f in self.removed
        ]
        kept = [
            {k: f[k] for k in ("id", "entity_name", "fact", "manual", "sources_dropped",
                               "sources_remaining")}
            for f in self.kept
        ]
        notes = [
            {k: f[k] for k in ("id", "entity_name", "fact", "sources_dropped")}
            for f in self.notes_kept
        ]
        lists: dict[str, list[Any]] = {
            "facts_removed": removed,
            "facts_kept": kept,
            "notes_kept": notes,
            "wiki_pages_removed": list(self.pages_removed),
            "wiki_pages_rebuilt": list(self.pages_rebuilt),
            "insight_pages_removed": list(self.insight_pages_removed),
            "queries_to_review": list(self.queries_to_review),
        }
        totals = {name: len(items) for name, items in lists.items()}
        return {
            "available": True,
            "forget_notes": self.forget_notes,
            **{name: items[:limit] for name, items in lists.items()},
            "totals": totals,
            "truncated": any(n > limit for n in totals.values()),
        }


def plan_retraction(
    session_id: str,
    *,
    conversation_store: LCMConversationStore,
    memory_store: MemoryStore,
    wiki_root: Path,
    purged_ids: Iterable[str] | None = None,
    forget_notes: bool = False,
) -> RetractionPlan:
    """Work out what retracting *session_id* would do. Reads only; changes nothing.

    ``purged_ids`` is for the sweep after the purge, when the chat's rows are gone from the
    conversation store and its ids must come from the earlier plan. ``forget_notes`` is the owner's
    explicit choice to treat ``/note`` facts like any other; by default they stay.
    """
    purged = (
        frozenset(purged_ids) if purged_ids is not None
        else frozenset(conversation_store.message_ids_for_session(session_id))
    )
    memories = memory_store.get_all_memories(min_confidence=-1.0, limit=_ALL)
    cited = [m for m in memories if purged.intersection(m["source_event_ids"])]

    others = {i for m in cited for i in m["source_event_ids"] if i not in purged}
    known = conversation_store.sessions_and_text_for_message_ids(others)

    removed: list[dict] = []
    kept: list[dict] = []
    notes_kept: list[dict] = []
    for m in cited:
        sources = list(m["source_event_ids"])
        rest = [i for i in sources if i not in purged]
        is_note = bool(m.get("manual"))
        if is_note and not forget_notes:
            notes_kept.append({
                "id": m["id"], "entity_name": m["entity_name"], "fact": m["fact"],
                "sources_dropped": len(sources) - len(rest), "sources_remaining": len(rest),
            })
            continue
        support = [
            i for i in rest
            if i in known and known[i][0] != session_id and _supports(known[i][1], m["entity_name"])
        ]
        if support:
            kept.append({
                "id": m["id"], "entity_name": m["entity_name"], "fact": m["fact"],
                "manual": is_note,
                "sources_dropped": len(sources) - len(rest), "sources_remaining": len(rest),
            })
            continue
        if is_note:
            reason = "a note you saved, merged into a fact learned in this chat; you chose to forget notes"
        elif not rest:
            reason = "its only source is this chat"
        elif not any(i in known for i in rest):
            reason = "its other sources are not in any chat"
        else:
            reason = f"its other sources do not mention {m['entity_name']!r}"
        removed.append({
            "id": m["id"], "entity_name": m["entity_name"], "entity_type": m["entity_type"],
            "fact": m["fact"], "manual": bool(m.get("manual")), "reason": reason,
        })

    # What the wiki would look like afterwards, computed without touching a file.
    removed_ids = {f["id"] for f in removed}
    kept_by_id = {k["id"]: k for k in (*kept, *notes_kept)}
    after: list[dict] = []
    for m in memories:
        if m["id"] in removed_ids:
            continue
        entry = kept_by_id.get(m["id"])
        if entry is not None:
            total = entry["sources_dropped"] + entry["sources_remaining"]
            m = {**m, "mention_count": max(1, round(
                (m.get("mention_count") or 1) * entry["sources_remaining"] / total))}
        after.append(m)

    def _pages(rows: Iterable[dict]) -> dict[str, str]:
        return {
            entity_page_relpath(etype, name): name
            for name, etype in WikiCompiler.page_entities_for(rows).items()
        }

    before_pages, after_pages = _pages(memories), _pages(after)
    touched = {f["entity_name"].casefold() for f in removed} | {
        f["entity_name"].casefold() for f in (*kept, *notes_kept)}
    pages_removed = sorted(set(before_pages) - set(after_pages))
    pages_rebuilt = sorted(
        p for p, name in after_pages.items() if p in before_pages and name.casefold() in touched)

    insight_removed, to_review = _scan_queries(Path(wiki_root), {f["entity_name"].casefold() for f in removed})

    legacy = memory_store.legacy_rows_for_session(session_id, purged)
    return RetractionPlan(
        session_id=session_id,
        purged_ids=purged,
        removed=tuple(removed),
        kept=tuple(kept),
        notes_kept=tuple(notes_kept),
        forget_notes=forget_notes,
        legacy_summary_ids=tuple(legacy["summary_ids"]),
        legacy_message_count=len(legacy["message_ids"]),
        has_cursor=memory_store.get_extractor_cursor(session_id) > 0,
        pages_removed=tuple(pages_removed),
        pages_rebuilt=tuple(pages_rebuilt),
        insight_pages_removed=tuple(insight_removed),
        queries_to_review=tuple(to_review),
    )


def _scan_queries(wiki_root: Path, removed_entities: set[str]) -> tuple[list[str], list[str]]:
    """Split ``queries/`` into insight pages to delete and filed answers to leave for review.

    An insight page records the entities it was synthesized from. One that includes a removed fact's
    entity may carry that fact, so it goes; the next synthesis cycle rebuilds it from what is left.
    A filed answer records no source at all, so it can only be listed, never judged.
    """
    queries = wiki_root / "queries"
    if not removed_entities or not queries.is_dir():
        return [], []
    insight_removed: list[str] = []
    to_review: list[str] = []
    for page in sorted(queries.glob("*.md")):
        text = page.read_text(encoding="utf-8", errors="replace")
        rel = f"queries/{page.name}"
        fm = _frontmatter(text)
        if fm.get("type") == "insight":
            entities = fm.get("entities")
            if isinstance(entities, list) and {str(e).casefold() for e in entities} & removed_entities:
                insight_removed.append(rel)
                continue
        if _mentions_any(text, removed_entities):
            to_review.append(rel)
    return insight_removed, to_review


def apply_retraction(
    plan: RetractionPlan, *, memory_store: MemoryStore, wiki_compiler: WikiCompiler
) -> dict[str, Any]:
    """Carry out *plan*: insight pages, then one memory transaction, then the wiki rebuild.

    Raises on any failure (the loud-failure law); callers must not treat a raise as "nothing
    happened". Safe to repeat: a second call finds nothing to do, and finishes a wiki rebuild that an
    earlier call committed the facts for but never completed.
    """
    pending = bool(memory_store.get_mark(WIKI_REGEN_PENDING_MARK))
    summary = {
        "facts_removed": len(plan.removed), "facts_kept": len(plan.kept),
        "notes_kept": len(plan.notes_kept), "forget_notes": plan.forget_notes,
        "wiki_pages_removed": len(plan.pages_removed), "wiki_pages_rebuilt": len(plan.pages_rebuilt),
        "insight_pages_removed": len(plan.insight_pages_removed),
        "queries_to_review": list(plan.queries_to_review),
    }
    if plan.empty and not pending:
        return summary

    if not plan.empty:
        for rel in plan.insight_pages_removed:
            _unlink_inside(wiki_compiler.wiki_root, rel)
        counts = memory_store.retract(
            delete_ids=[f["id"] for f in plan.removed],
            strip_fact_ids=[f["id"] for f in (*plan.kept, *plan.notes_kept)],
            strip_message_ids=plan.purged_ids,
            legacy_session_id=plan.session_id,
            legacy_summary_ids=plan.legacy_summary_ids,
            cursor_scope=plan.session_id,
            marks={WIKI_REGEN_PENDING_MARK: 1.0},
        )
        log.info("purge retraction for %s: %s", plan.session_id, counts)

    wiki_compiler.regenerate_all()
    memory_store.set_mark(WIKI_REGEN_PENDING_MARK, 0.0)
    return summary


def retract_before_purge(
    session_id: str,
    *,
    conversation_store: LCMConversationStore,
    memory_store: MemoryStore,
    wiki_compiler: WikiCompiler,
    forget_notes: bool = False,
) -> tuple[RetractionPlan, dict[str, Any]]:
    """Plan and apply, in the one order that works: before the conversation rows are deleted."""
    plan = plan_retraction(
        session_id, conversation_store=conversation_store, memory_store=memory_store,
        wiki_root=wiki_compiler.wiki_root, forget_notes=forget_notes)
    return plan, apply_retraction(plan, memory_store=memory_store, wiki_compiler=wiki_compiler)


def sweep_after_purge(
    plan: RetractionPlan,
    *,
    conversation_store: LCMConversationStore,
    memory_store: MemoryStore,
    wiki_compiler: WikiCompiler,
) -> dict[str, Any]:
    """Catch a fact the extractor wrote from the purged messages while the purge was running."""
    late = plan_retraction(
        plan.session_id, conversation_store=conversation_store, memory_store=memory_store,
        wiki_root=wiki_compiler.wiki_root, purged_ids=plan.purged_ids,
        forget_notes=plan.forget_notes)
    return apply_retraction(late, memory_store=memory_store, wiki_compiler=wiki_compiler)
