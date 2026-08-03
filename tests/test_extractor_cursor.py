"""The MemoryExtractor's progress cursor — durable, per-scope, rowid-based.

What it replaced: ``self._last_processed_ts``, a single in-memory float
initialised to 0.0 and persisted nowhere. One value serving two scopes, which
made it fail in BOTH directions from the same root cause:

* **Over-extraction.** Every daemon restart reset it to 0.0, so the first pass
  read ``timestamp > 0.0`` — the 500 oldest uncompacted rows in the whole
  database, across every session, mined again. ``persist_memory`` increments
  ``mention_count`` on a dedup hit, and ``mention_count >= 2`` is exactly the
  wiki page-worthiness threshold, so a re-mine could promote a thing mentioned
  once months ago into a wiki page. The visible symptom is a page appearing for
  no reason, which reads as the extractor being noisy rather than as a replay.
* **Under-extraction.** ``pre_compaction_flush`` calls ``run_once(session_id=X)``,
  which advanced that same GLOBAL value to session X's newest timestamp. Rows
  belonging to other sessions older than that were then never read again.

Three tests, and the third is the reason the other two are not enough: "never
mine anything" would pass both of the first two perfectly. A suppression-only
suite cannot tell a working cursor from a broken extractor.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from prometheus.memory.extractor import MemoryExtractor
from prometheus.memory.lcm_conversation_store import LCMConversationStore
from prometheus.memory.lcm_types import MessagePart
from prometheus.memory.store import EXTRACTOR_GLOBAL_SCOPE, MemoryStore

SESSION_A = "telegram:111"
SESSION_B = "telegram:222"


class _NullProvider:
    """The one double: an external service. Never called — every test drives
    the read/cursor path, and ``_process_batch`` is stubbed out below."""


@pytest.fixture
def conv(tmp_path) -> LCMConversationStore:
    return LCMConversationStore(tmp_path / "lcm.db")


@pytest.fixture
def store(tmp_path) -> MemoryStore:
    return MemoryStore(db_path=tmp_path / "memory.db")


def _ingest(conv: LCMConversationStore, session_id: str, text: str) -> int:
    """Append one user message; return its durable rowid."""
    msg = MessagePart(role="user", content=text, session_id=session_id)
    conv.insert_message(msg)
    return msg.row_id


def _extractor(store: MemoryStore, conv: LCMConversationStore) -> MemoryExtractor:
    """A real extractor with the model call replaced by a recorder.

    ``_process_batch`` is the boundary between "which rows did the cursor
    select" (what these tests are about) and "what did the LLM make of them"
    (what it is not). Stubbing exactly there keeps the cursor logic, the read,
    the filters and the persistence all real.
    """
    ext = MemoryExtractor(
        store=store, provider=_NullProvider(), lcm_conversation_store=conv,
    )
    ext.mined: list[str] = []

    async def _record(messages):
        ext.mined.extend(m["content"] for m in messages)
        return len(messages), []

    ext._process_batch = _record  # type: ignore[method-assign]
    return ext


def _run(ext: MemoryExtractor, session_id: str | None = None) -> list[str]:
    """One pass; return what it mined THIS pass."""
    before = len(ext.mined)
    asyncio.run(ext.run_once(session_id=session_id))
    return ext.mined[before:]


# ---------------------------------------------------------------------------
# 1. Over-extraction — a restart must not re-mine
# ---------------------------------------------------------------------------

def test_a_restart_does_not_re_mine_what_was_already_mined(store, conv):
    """THE HEADLINE BUG. A fresh MemoryExtractor is exactly what a daemon
    restart produces: same stores on disk, brand-new instance with brand-new
    instance state. Under the old float this re-read from 0.0."""
    # Cursor at the beginning BEFORE construction — the migration floor is
    # seeded in __init__, so setting it afterwards would be too late (which is
    # how this test first caught the lazy-seeding bug).
    store.set_extractor_cursor(EXTRACTOR_GLOBAL_SCOPE, 0)
    _ingest(conv, SESSION_A, "first message")
    _ingest(conv, SESSION_A, "second message")

    first = _extractor(store, conv)
    assert _run(first) == ["first message", "second message"]

    # --- restart ---
    after_restart = _extractor(store, conv)
    assert _run(after_restart) == [], (
        "the extractor re-mined messages it had already processed — every "
        "re-mine increments mention_count, and mention_count >= 2 is the wiki "
        "page-worthiness threshold"
    )


def test_the_cursor_is_seeded_from_max_rowid_not_zero(store, conv):
    """THE MIGRATION TRADE, pinned.

    Seeding at 0 would make the first pass after this change perform exactly
    the full-history re-mine the change exists to prevent. So the floor is
    seeded from the current max rowid: pre-existing history is treated as
    consumed, and anything genuinely unmined before the deploy stays unmined.
    That is a one-way door and it belongs in a test, not only in a PR body."""
    _ingest(conv, SESSION_A, "history from before the fix")
    top = _ingest(conv, SESSION_A, "more history")

    ext = _extractor(store, conv)

    # Asserted AT SEED TIME, before any pass runs. Checking it after a pass
    # tested nothing: the per-session audit rows drop the same rows the global
    # floor should have excluded, and the pass then advances the global cursor
    # to the same value — so a floor seeded at 0 produced an identical
    # end-state and this test stayed green. Found by mutation W2 being killed
    # by three other tests and not by the one named after it (§3b's
    # defence-in-depth variant: assert the control's identity, not an outcome
    # its neighbour also produces).
    assert store.get_extractor_cursor(EXTRACTOR_GLOBAL_SCOPE) == top, (
        "the global floor was not seeded from the store's max rowid"
    )

    assert _run(ext) == [], "the migration re-mined pre-existing history"
    assert store.get_extractor_cursor(EXTRACTOR_GLOBAL_SCOPE) == top

    # And it seeds ONCE — a later restart must not re-seed over a live cursor.
    _ingest(conv, SESSION_A, "new message after the fix")
    assert _run(_extractor(store, conv)) == ["new message after the fix"]


# ---------------------------------------------------------------------------
# 2. Under-extraction — a per-session flush must not strand other sessions
# ---------------------------------------------------------------------------

def test_a_per_session_flush_does_not_advance_the_global_cursor(store, conv):
    """THE QUIET HALF. ``pre_compaction_flush`` runs ``run_once(session_id=X)``.
    Under the old float that advanced the one global watermark to X's newest
    timestamp, so B's older rows fell below it and were never read again —
    silently, forever, with no error anywhere."""
    store.set_extractor_cursor(EXTRACTOR_GLOBAL_SCOPE, 0)

    _ingest(conv, SESSION_B, "B: older, not yet mined")
    a_row = _ingest(conv, SESSION_A, "A: newer, flushed first")

    ext = _extractor(store, conv)
    assert _run(ext, session_id=SESSION_A) == ["A: newer, flushed first"]

    # Asserted directly, not only through its effect: the flush moved A's
    # cursor and left the global floor alone.
    assert store.get_extractor_cursor(SESSION_A) == a_row
    assert store.get_extractor_cursor(EXTRACTOR_GLOBAL_SCOPE) == 0, (
        "a per-session flush advanced the GLOBAL cursor — every other "
        "session's rows below it are now stranded permanently"
    )

    # And the effect: the next sweep still sees B's older row.
    assert "B: older, not yet mined" in _run(ext)


def test_a_sweep_does_not_re_mine_rows_a_flush_already_took(store, conv):
    """The other half of the same invariant, and a SEPARATE mechanism.

    The test above pins "a flush must not move the global cursor". This pins
    "a sweep must respect per-session cursors" — the sweep issues one query
    against the global floor, so it needs a per-row check to see a flush that
    ran ahead of it. Two controls, so two tests: when they shared one
    assertion, disabling either turned the same test red and the matrix could
    not tell them apart."""
    store.set_extractor_cursor(EXTRACTOR_GLOBAL_SCOPE, 0)
    _ingest(conv, SESSION_B, "B: untouched by the flush")
    _ingest(conv, SESSION_A, "A: taken by the flush")

    ext = _extractor(store, conv)
    assert _run(ext, session_id=SESSION_A) == ["A: taken by the flush"]

    swept = _run(ext)
    assert "A: taken by the flush" not in swept, (
        "the cross-session sweep re-mined a row the flush had already "
        "consumed — mention_count is now incremented twice for one message"
    )
    assert swept == ["B: untouched by the flush"]


def test_a_flush_does_not_re_mine_what_the_sweep_already_took(store, conv):
    """Per-session cursors are authoritative when present, so a flush cannot
    re-read the sweep's rows either."""
    store.set_extractor_cursor(EXTRACTOR_GLOBAL_SCOPE, 0)
    _ingest(conv, SESSION_A, "taken by the sweep")

    ext = _extractor(store, conv)
    assert _run(ext) == ["taken by the sweep"]
    assert _run(ext, session_id=SESSION_A) == [], (
        "the flush re-mined a row the cross-session sweep had already consumed"
    )


def test_a_session_with_no_cursor_resolves_to_the_global_floor_not_zero(store, conv):
    """A session with no cursor row of its own must floor at the GLOBAL value,
    not at 0. Falling back to 0 would make the first flush on such a chat read
    its entire history — the original bug wearing a different hat.

    Two traps this test had to be rewritten to avoid:

    * If the session's only rows are NEWER than the global floor, floor-0 and
      floor-global select the same rows and it passes either way. It needs a
      row BELOW the floor to distinguish them.
    * Since the migration now seeds one row per session, most sessions HAVE a
      cursor. A genuinely absent one needs a session the seed could not see —
      and there is a real one: ``list_sessions()`` excludes tombstoned
      sessions (Beacon's "forget session"), so a forgotten chat that later
      speaks again is exactly this case in production."""
    _ingest(conv, SESSION_A, "old history")
    _ingest(conv, SESSION_B, "B's old history, below the floor")
    conv.tombstone_session(SESSION_B)  # hidden from list_sessions → not seeded

    ext = _extractor(store, conv)
    assert store.get_extractor_cursor(SESSION_B) == 0, (
        "precondition: B must have no cursor row of its own"
    )
    assert store.get_extractor_cursor(EXTRACTOR_GLOBAL_SCOPE) > 0

    _ingest(conv, SESSION_B, "B's first NEW message")
    assert _run(ext, session_id=SESSION_B) == ["B's first NEW message"], (
        "a session with no cursor row read from rowid 0 and re-mined history "
        "the global floor had already accounted for"
    )


# ---------------------------------------------------------------------------
# 3. THE ADMISSION HALF — "never mine anything" would pass everything above
# ---------------------------------------------------------------------------

def test_new_messages_are_still_mined_after_a_restart(store, conv):
    """The test the other two cannot replace.

    Every assertion above is of the form "X was NOT mined". An extractor that
    had been broken into mining nothing at all — a cursor stuck at the maximum,
    a read that always returns empty, a filter that drops everything — would
    satisfy all of them and look like a clean fix. This is the only test here
    that fails in that direction, and it is why §2c exists: a suite whose every
    case points the same way cannot see an entire class of failure."""
    _ingest(conv, SESSION_A, "pre-existing history")

    ext = _extractor(store, conv)
    assert _run(ext) == []  # seeded above the history, correctly

    _ingest(conv, SESSION_A, "genuinely new, after the seed")
    _ingest(conv, SESSION_B, "new on another session too")

    # --- restart, then a normal sweep ---
    after_restart = _extractor(store, conv)
    mined = _run(after_restart)
    assert mined == [
        "genuinely new, after the seed",
        "new on another session too",
    ], (
        f"new messages were NOT mined after a restart (got {mined!r}) — the "
        f"cursor is over-suppressing, which reads exactly like a working fix "
        f"until someone notices memory has stopped growing"
    )


def test_the_migration_seeds_one_row_per_session_for_audit(store, conv):
    """A one-way door earns an auditable record.

    The global floor alone is equally CORRECT — an absent per-session cursor
    resolves to it — but it shows one number for every session, so "did the
    migration do the right thing for MY chat" cannot be answered after the
    fact without re-deriving it. This migration cannot be re-run, so the state
    it leaves behind is the only evidence there will ever be."""
    a_top = _ingest(conv, SESSION_A, "A's history")
    b_top = _ingest(conv, SESSION_B, "B's history")

    _extractor(store, conv)  # seeds at construction

    assert store.get_extractor_cursor(EXTRACTOR_GLOBAL_SCOPE) == max(a_top, b_top)
    assert store.get_extractor_cursor(SESSION_A) == a_top, (
        "per-session seed missing or wrong — the migration's decision for this "
        "session is not recoverable from the table"
    )
    assert store.get_extractor_cursor(SESSION_B) == b_top


def test_per_session_seeding_is_behaviourally_inert(store, conv, tmp_path):
    """It must change WHAT IS RECORDED, never WHAT IS MINED.

    Seeding a session at its own (lower) max rowid rather than at the global
    floor could in principle re-open rows between the two. It does not,
    because ``_cursor_for`` takes ``max(global, own)``. Proven by running the
    same history both ways and comparing the mined set, not by arguing it."""
    from prometheus.memory.store import MemoryStore

    _ingest(conv, SESSION_A, "A's history")
    _ingest(conv, SESSION_B, "B's history")

    # With per-session seeding (the shipped path).
    with_seeds = _extractor(store, conv)
    # Capture the floor BEFORE the run — reading it afterwards would pick up
    # the cursor the run itself advanced, and the "control" would start above
    # the very row it is supposed to mine. (That is how this test failed the
    # first time: the control mined nothing and looked like a real divergence.)
    global_floor = store.get_extractor_cursor(EXTRACTOR_GLOBAL_SCOPE)

    _ingest(conv, SESSION_A, "new on A")
    mined_with = _run(with_seeds)

    # Global-only seeding, same history, a fresh facts store.
    bare = MemoryStore(db_path=tmp_path / "bare.db")
    bare.set_extractor_cursor(EXTRACTOR_GLOBAL_SCOPE, global_floor)
    without_seeds = _extractor(bare, conv)
    mined_without = _run(without_seeds)

    assert mined_with == mined_without == ["new on A"], (
        f"per-session seeding changed what gets mined: {mined_with!r} vs "
        f"{mined_without!r} — it is supposed to be an audit record, not a "
        f"behaviour change"
    )


def test_messages_arriving_after_construction_are_always_mined(store, conv):
    """The bug the migration itself nearly introduced.

    Seeding the floor lazily — on the first ``run_once`` — looked equivalent to
    seeding at construction and was not. The first pass is 30 minutes after
    boot (or whenever a compaction flush fires), so on a fresh install
    everything said in between would be seeded PAST and never mined: the
    under-extraction failure this whole change exists to cure, reintroduced by
    its own migration.

    Seeding now happens in ``__init__``, so anything ingested after the
    extractor exists is always in scope."""
    _ingest(conv, SESSION_A, "history from before the daemon started")

    ext = _extractor(store, conv)  # seeds here, above the history

    _ingest(conv, SESSION_A, "said one minute after boot")
    _ingest(conv, SESSION_B, "and on another chat")

    assert _run(ext) == [
        "said one minute after boot",
        "and on another chat",
    ], (
        "messages that arrived after the extractor was constructed were seeded "
        "past — a fresh install would silently lose everything said before the "
        "first 30-minute pass"
    )


def test_the_cursor_advances_over_filtered_rows_too(store, conv):
    """Non-user and machine-session rows are skipped for mining but must still
    move the cursor, or every pass re-reads the same trailing block forever."""
    store.set_extractor_cursor(EXTRACTOR_GLOBAL_SCOPE, 0)
    msg = MessagePart(
        role="user", content="task output, not the user speaking",
        session_id=SESSION_A, provenance="task_supervisor", is_trusted=False,
    )
    conv.insert_message(msg)

    ext = _extractor(store, conv)
    assert _run(ext) == [], "an untrusted row was mined as a user fact"
    assert store.get_extractor_cursor(EXTRACTOR_GLOBAL_SCOPE) == msg.row_id, (
        "the cursor did not advance past a skipped row — the next pass will "
        "re-read it, and every pass after that"
    )


def test_the_cursor_cannot_rewind(store):
    """Monotonicity is enforced in SQL, not left to callers: a stale or
    out-of-order write must never re-expose already-mined rows."""
    store.set_extractor_cursor(SESSION_A, 500)
    store.set_extractor_cursor(SESSION_A, 100)
    assert store.get_extractor_cursor(SESSION_A) == 500
