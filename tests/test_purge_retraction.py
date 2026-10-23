"""Purging a chat retracts what the assistant learned only from it.

``purge_session`` removes a chat's messages and summaries from the conversation database. Before
this, it left the memory facts the extractor had mined from that chat, the wiki pages compiled from
those facts, and the insight pages synthesized from them. A secret pasted into a chat and then
purged survived as a fact and a wiki page.

The rule (decided by the owner):

* a fact learned ONLY from the purged chat is deleted;
* a fact a surviving chat also supports stays, but the purged chat stops being one of its sources;
* the owner is shown what will go before confirming.

One thing in the data shapes all of this. A fact's ``source_event_ids`` is BATCH-level: the
extractor cuts a mixed sweep of messages into batches and every fact from a batch cites every
message in it, so a surviving id can be a batch-mate of the purged message and say nothing about
who taught the fact. A surviving source therefore only counts as support if that message actually
mentions the fact's entity. Tests marked "batch-mate" pin that.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from prometheus.memory.lcm_conversation_store import LCMConversationStore
from prometheus.memory.lcm_types import MessagePart
from prometheus.memory.purge_derived import (
    apply_retraction,
    plan_retraction,
    retract_before_purge,
    sweep_after_purge,
)
from prometheus.memory.retention import DAY_SECONDS, apply_retention, plan_retention
from prometheus.memory.store import MemoryStore
from prometheus.memory.wiki_compiler import WikiCompiler
from prometheus.sentinel.knowledge_synth import KnowledgeSynthesizer, SynthInsight

DOOMED = "desktop:doomed"
KEEPER = "telegram:keeper"

#: Lowercase on purpose: facts pass through secret redaction on the way in, and a marker shaped like
#: a token would be scrubbed before the test could look for it.
SECRET = "zorblattxylo"


def _say(lcm: LCMConversationStore, session: str, text: str, mid: str | None = None) -> str:
    """A user message. Pass ``mid`` for a fixed id: wiki pages cite the SMALLEST source id of a
    fact, so a test that reads the page back needs ids whose order it controls."""
    part = MessagePart(role="user", content=text)
    if mid:
        part.message_id = mid
    return lcm.add_message(session, part)


def _fact(mem: MemoryStore, entity: str, text: str, sources: list[str], *, count: int = 1,
          manual: bool = False, kind: str = "concept") -> str:
    mid = mem.persist_memory(kind, entity, text, 0.9, source_event_ids=sources, manual=manual)
    if count != 1:
        mem.update_memory(mid, mention_count=count)
    return mid


@pytest.fixture()
def world(tmp_path: Path) -> SimpleNamespace:
    """Two chats, six facts, a wiki, and two insight pages: every case the rule has to tell apart."""
    lcm = LCMConversationStore(tmp_path / "lcm.db")
    mem = MemoryStore(db_path=tmp_path / "memory.db")
    wiki = tmp_path / "wiki"

    d1 = _say(lcm, DOOMED, f"Zephyr Corp card is {SECRET}. Marlowe lives in Lisbon.", "a0000001")
    d2 = _say(lcm, DOOMED, "Quillfeather likes tea. Also Zephyr Corp again.", "a0000002")
    d3 = _say(lcm, DOOMED, "Orphaned Thing and Note Thing are things I said here.", "a0000003")
    k1 = _say(lcm, KEEPER, "Marlowe moved to Lisbon last spring.", "b0000001")
    k2 = _say(lcm, KEEPER, "The Gardens need water in summer.", "b0000002")

    ids = SimpleNamespace(
        # learned only from the purged chat
        only=_fact(mem, "Zephyr Corp", f"Zephyr Corp card number is {SECRET}", [d1, d2], count=2,
                   kind="organization"),
        # also supported: k1 really mentions Marlowe
        shared=_fact(mem, "Marlowe", "Marlowe lives in Lisbon", [d1, k1], count=4, kind="person"),
        # batch-mate: k2 does NOT mention Quillfeather, it only rode in the same extraction batch
        mate=_fact(mem, "Quillfeather", "Quillfeather likes tea", [d2, k2]),
        # nothing to do with the purged chat
        bystander=_fact(mem, "Gardens", "Gardens need water", [k2], count=2),
        # a source that exists in no chat is not support
        orphan=_fact(mem, "Orphaned Thing", "Orphaned Thing was mentioned", [d3, "ghost-id-0001"]),
        # a human-asserted note whose only source is the purged chat
        note=_fact(mem, "Note Thing", "Note Thing is a thing", [d3], manual=True),
    )
    # Two entities at 2+ mentions (and the manual one) get pages.
    compiler = WikiCompiler(store=mem, wiki_root=wiki)
    compiler.regenerate_all()

    synth = KnowledgeSynthesizer(mem, object(), wiki_root=wiki)
    synth._write_insight_page(
        SynthInsight(entities=["Zephyr Corp", "Marlowe"], insight="Zephyr and Marlowe overlap."),
        fingerprint="0" * 64)
    synth._write_insight_page(
        SynthInsight(entities=["Gardens", "Water"], insight="Gardens need water."),
        fingerprint="1" * 64)
    (wiki / "queries" / "my-filed-answer.md").write_text(
        "# What did I say about Zephyr Corp?\n\nA filed answer.\n", encoding="utf-8")

    return SimpleNamespace(
        lcm=lcm, mem=mem, wiki=wiki, compiler=compiler, ids=ids,
        d=[d1, d2, d3], k=[k1, k2], tmp=tmp_path)


def _plan(w, **kw):
    return plan_retraction(DOOMED, conversation_store=w.lcm, memory_store=w.mem,
                           wiki_root=w.wiki, **kw)


def _snapshot(w):
    rows = [tuple(r) for r in w.mem._conn.execute("SELECT * FROM memories ORDER BY id").fetchall()]
    files = {str(p.relative_to(w.wiki)): p.read_bytes() for p in sorted(w.wiki.rglob("*")) if p.is_file()}
    return rows, files


def _removed_ids(plan) -> set[str]:
    return {f["id"] for f in plan.removed}


def _file_contains(db: Path, needle: str) -> bool:
    return any(p.exists() and needle.encode() in p.read_bytes()
               for p in (db, Path(str(db) + "-wal"), Path(str(db) + "-shm")))


def _index_terms(mem: MemoryStore, prefix: str) -> list[str]:
    mem._conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS _vocab_probe USING fts5vocab(memories_fts, 'row')")
    return [r[0] for r in mem._conn.execute(
        "SELECT term FROM _vocab_probe WHERE term LIKE ?", (prefix.lower() + "%",)).fetchall()]


# ── the rule ───────────────────────────────────────────────────────────

def test_a_fact_learned_only_from_the_purged_chat_is_removed(world):
    plan = _plan(world)
    assert world.ids.only in _removed_ids(plan)

    apply_retraction(plan, memory_store=world.mem, wiki_compiler=world.compiler)

    assert world.mem.get_memory(world.ids.only) is None
    assert world.mem.search_memories(entity="Zephyr Corp", limit=10) == []


def test_a_fact_a_surviving_chat_also_supports_stays_and_loses_the_purged_source(world):
    plan = _plan(world)
    assert world.ids.shared not in _removed_ids(plan)

    apply_retraction(plan, memory_store=world.mem, wiki_compiler=world.compiler)

    kept = world.mem.get_memory(world.ids.shared)
    assert kept is not None
    assert kept["source_event_ids"] == world.k[:1], "only the surviving chat's message remains"
    assert kept["fact"] == "Marlowe lives in Lisbon"
    # 4 mentions across 2 sources, one of them purged: half the mentions go, floor 1.
    assert kept["mention_count"] == 2


def test_mention_count_never_drops_below_one(world):
    sole = _fact(world.mem, "Marlowe", "Marlowe has a cat", [world.d[0], world.k[0]], count=1)
    plan = _plan(world)
    apply_retraction(plan, memory_store=world.mem, wiki_compiler=world.compiler)
    assert world.mem.get_memory(sole)["mention_count"] >= 1


def test_batch_mate_a_surviving_source_that_never_mentions_the_entity_is_not_support(world):
    """k2 sat in the same extraction batch as d2; it does not say anything about Quillfeather."""
    plan = _plan(world)
    assert world.ids.mate in _removed_ids(plan)
    reason = next(f["reason"] for f in plan.removed if f["id"] == world.ids.mate)
    assert "Quillfeather" in reason, "the preview must say WHY a fact with a surviving source goes"

    apply_retraction(plan, memory_store=world.mem, wiki_compiler=world.compiler)
    assert world.mem.get_memory(world.ids.mate) is None


def test_ids_that_exist_in_no_chat_are_not_support(world):
    plan = _plan(world)
    assert world.ids.orphan in _removed_ids(plan)


def test_a_manual_note_is_treated_like_any_other_fact_and_flagged(world):
    plan = _plan(world)
    entry = next(f for f in plan.removed if f["id"] == world.ids.note)
    assert entry["manual"] is True
    apply_retraction(plan, memory_store=world.mem, wiki_compiler=world.compiler)
    assert world.mem.get_memory(world.ids.note) is None


def test_facts_that_cite_no_purged_message_are_left_exactly_as_they_were(world):
    before = world.mem.get_memory(world.ids.bystander)
    plan = _plan(world)
    assert world.ids.bystander not in _removed_ids(plan)
    assert world.ids.bystander not in {f["id"] for f in plan.kept}

    apply_retraction(plan, memory_store=world.mem, wiki_compiler=world.compiler)
    assert world.mem.get_memory(world.ids.bystander) == before


# ── plan and apply ─────────────────────────────────────────────────────

def test_planning_changes_nothing(world):
    before = _snapshot(world)
    _plan(world)
    _plan(world)
    assert _snapshot(world) == before


def test_apply_is_idempotent(world):
    plan = _plan(world)
    apply_retraction(plan, memory_store=world.mem, wiki_compiler=world.compiler)
    after_first = _snapshot(world)

    again = _plan(world)
    assert again.removed == () and again.kept == (), "a second plan finds nothing left to do"
    apply_retraction(again, memory_store=world.mem, wiki_compiler=world.compiler)
    rows, _files = _snapshot(world)
    assert rows == after_first[0]


def test_the_preview_and_the_apply_agree(world):
    """What the owner is shown is what is removed."""
    plan = _plan(world)
    shown = {f["id"] for f in plan.to_preview()["facts_removed"]}
    before = {m["id"] for m in world.mem.get_all_memories(limit=1000)}

    apply_retraction(plan, memory_store=world.mem, wiki_compiler=world.compiler)

    after = {m["id"] for m in world.mem.get_all_memories(limit=1000)}
    assert before - after == shown


def test_a_fact_written_from_the_purged_messages_during_the_purge_is_caught_by_the_sweep(world):
    plan, _ = retract_before_purge(DOOMED, conversation_store=world.lcm,
                                   memory_store=world.mem, wiki_compiler=world.compiler)
    world.lcm.purge_session(DOOMED)
    # the extractor was mid-batch: it writes a fact citing a message that is now gone
    late = _fact(world.mem, "Latecomer", f"Latecomer owes {SECRET}", [sorted(plan.purged_ids)[0]])

    summary = sweep_after_purge(plan, conversation_store=world.lcm, memory_store=world.mem,
                                wiki_compiler=world.compiler)

    assert world.mem.get_memory(late) is None
    assert summary["facts_removed"] == 1


def test_retraction_can_be_planned_from_ids_after_the_chat_is_gone(world):
    ids = set(world.lcm.message_ids_for_session(DOOMED))
    world.lcm.purge_session(DOOMED)
    plan = plan_retraction(DOOMED, conversation_store=world.lcm, memory_store=world.mem,
                           wiki_root=world.wiki, purged_ids=ids)
    assert world.ids.only in _removed_ids(plan)


# ── the disk ───────────────────────────────────────────────────────────

def test_the_removed_text_is_not_in_the_memory_file_its_wal_or_its_index(world):
    db = world.tmp / "memory.db"
    assert _file_contains(db, SECRET), "precondition: the secret is in the file"
    assert _index_terms(world.mem, SECRET) == [SECRET], "precondition: the index holds the term"

    apply_retraction(_plan(world), memory_store=world.mem, wiki_compiler=world.compiler)

    assert _index_terms(world.mem, SECRET) == [], "the FTS index must forget the term, not just the row"
    assert not _file_contains(db, SECRET), "the text must be gone from the file, not merely hidden"


# ── the wiki ───────────────────────────────────────────────────────────

def test_a_page_that_existed_only_on_the_purged_chat_is_gone(world):
    assert (world.wiki / "clients" / "Zephyr Corp.md").exists(), "precondition"
    apply_retraction(_plan(world), memory_store=world.mem, wiki_compiler=world.compiler)

    assert not (world.wiki / "clients" / "Zephyr Corp.md").exists()
    # The filed answer is left in place on purpose (it is listed for review), and the index lists
    # it by title, so the assertion is about the entity's PAGE, not the name anywhere.
    assert "clients/Zephyr Corp.md" not in (world.wiki / "index.md").read_text(encoding="utf-8")
    assert SECRET not in "".join(
        p.read_text(encoding="utf-8", errors="ignore") for p in world.wiki.rglob("*.md"))


def test_a_kept_entitys_page_is_rebuilt_without_the_purged_source(world):
    page = world.wiki / "people" / "Marlowe.md"
    assert world.d[0][:8] in page.read_text(encoding="utf-8"), "precondition: the page cites the source"

    apply_retraction(_plan(world), memory_store=world.mem, wiki_compiler=world.compiler)

    text = page.read_text(encoding="utf-8") if page.exists() else ""
    assert world.d[0][:8] not in text
    assert not text or "Marlowe lives in Lisbon" in text


def test_insight_pages_that_used_a_removed_entity_go_and_unrelated_ones_stay(world):
    plan = _plan(world)
    assert plan.insight_pages_removed == ("queries/insight-zephyr-corp-marlowe.md",)
    apply_retraction(plan, memory_store=world.mem, wiki_compiler=world.compiler)

    assert not (world.wiki / "queries" / "insight-zephyr-corp-marlowe.md").exists()
    assert (world.wiki / "queries" / "insight-gardens-water.md").exists()


def test_a_filed_answer_that_mentions_a_removed_entity_is_listed_not_touched(world):
    plan = _plan(world)
    assert "queries/my-filed-answer.md" in plan.queries_to_review

    before = (world.wiki / "queries" / "my-filed-answer.md").read_bytes()
    apply_retraction(plan, memory_store=world.mem, wiki_compiler=world.compiler)
    assert (world.wiki / "queries" / "my-filed-answer.md").read_bytes() == before


def test_the_preview_names_the_pages_that_will_go_or_change(world):
    preview = _plan(world).to_preview()
    assert "clients/Zephyr Corp.md" in preview["wiki_pages_removed"]
    assert "people/Marlowe.md" in preview["wiki_pages_rebuilt"]


def test_a_wiki_regeneration_that_fails_is_finished_on_the_next_try(world, monkeypatch):
    """The facts are committed before the wiki is rebuilt. If the rebuild fails, the next purge must
    finish it, even though that purge's plan finds no facts left to retract."""
    plan = _plan(world)
    real = world.compiler.regenerate_all

    def boom():
        raise OSError("disk full")

    monkeypatch.setattr(world.compiler, "regenerate_all", boom)
    with pytest.raises(OSError):
        apply_retraction(plan, memory_store=world.mem, wiki_compiler=world.compiler)
    assert (world.wiki / "clients" / "Zephyr Corp.md").exists(), "the stale page is still there"

    monkeypatch.setattr(world.compiler, "regenerate_all", real)
    retry = _plan(world)
    assert retry.removed == ()
    apply_retraction(retry, memory_store=world.mem, wiki_compiler=world.compiler)
    assert not (world.wiki / "clients" / "Zephyr Corp.md").exists()


# ── the rest of what a chat leaves in the memory database ──────────────

def test_the_chats_extractor_cursor_goes_and_others_stay(world):
    world.mem.set_extractor_cursor(DOOMED, 5)
    world.mem.set_extractor_cursor(KEEPER, 7)
    apply_retraction(_plan(world), memory_store=world.mem, wiki_compiler=world.compiler)
    assert world.mem.get_extractor_cursor(DOOMED) == 0
    assert world.mem.get_extractor_cursor(KEEPER) == 7


def test_the_deprecated_message_and_summary_tables_are_cleaned_too(world):
    lm1 = world.mem.add_message(DOOMED, "user", f"legacy copy {SECRET}")
    lm2 = world.mem.add_message(KEEPER, "user", "legacy keeper")
    world.mem.add_summary(f"about {SECRET}", [lm1], summary_id="sum-doomed")
    world.mem.add_summary("mixed", [lm1, lm2], summary_id="sum-mixed")

    apply_retraction(_plan(world), memory_store=world.mem, wiki_compiler=world.compiler)

    assert world.mem.get_messages(DOOMED) == []
    assert len(world.mem.get_messages(KEEPER)) == 1
    kept = {s["id"] for s in world.mem.get_summaries(limit=10)}
    assert kept == {"sum-mixed"}, "a summary drawn only from the purged chat goes; a mixed one stays"


# ── what must NOT change ───────────────────────────────────────────────

def test_automatic_retention_never_retracts_facts(world):
    """What was learned outlives chat retention; only the owner's explicit purge retracts."""
    now = 1_800_000_000.0
    world.lcm._conn.execute("UPDATE lcm_messages SET timestamp = ? WHERE session_id = ?",
                            (now - 400 * DAY_SECONDS, DOOMED))
    world.lcm._conn.execute(
        "INSERT OR REPLACE INTO session_tombstones (session_id, deleted_at) VALUES (?, ?)",
        (DOOMED, now - 300 * DAY_SECONDS))
    world.lcm._conn.commit()
    before = _snapshot(world)

    result = apply_retention(world.lcm, plan_retention(world.lcm, now=now))

    assert result["purged"] == 1
    assert world.lcm.message_ids_for_session(DOOMED) == []
    assert _snapshot(world) == before, "retention must leave facts and the wiki exactly as they were"


def test_a_chat_that_taught_nothing_changes_nothing(world):
    quiet = _say(world.lcm, "slack:quiet", "just chatting about nothing")
    del quiet
    before = _snapshot(world)
    plan = plan_retraction("slack:quiet", conversation_store=world.lcm, memory_store=world.mem,
                           wiki_root=world.wiki)
    assert plan.empty
    apply_retraction(plan, memory_store=world.mem, wiki_compiler=world.compiler)
    assert _snapshot(world) == before
