"""ChatSession → LCM persistence is exact-once (2026-08-11 duplicate-rows fix).

The 2026-08-11 tool_use_id↔message_id survey found identical re-persisted rows
in the append-only LCM store: in ``desktop:s4-*`` probe sessions the same
tool_result row appeared up to FOUR times (rowids 461/466/481/496, one copy
per concurrently in-flight turn). Mechanism: the WS bridge ran fire-and-forget
turns concurrently on one session; each ``run_loop`` appended in place to the
SHARED ``session.messages`` list, and each finishing turn persisted
``messages[its_own_original_len:]`` — a tail covering every OTHER turn's rows
too. Separately, ``POST /api/chat`` passed ``len(session.messages) - 1`` as
``original_len``, re-persisting (and re-appending) the user row every call.

The fix has two halves: the WS bridge now serializes turns per session
(``tests/test_ws_turn_lock.py``), and — tested here — ``ChatSession`` keeps a
persistence watermark so persisting an overlapping or stale span re-writes
nothing. Every row lands in LCM exactly once no matter how sloppy the caller's
``original_len`` is.
"""

from __future__ import annotations

from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.engine.session import ChatSession


def _asst(text: str) -> ConversationMessage:
    return ConversationMessage(role="assistant", content=[TextBlock(text=text)])


class _RecordingEngine:
    """Fake LCM engine capturing every ingest_sync call."""

    def __init__(self) -> None:
        self.rows: list[tuple[int, str, str]] = []  # (turn_index, role, content)

    def ingest_sync(self, *, session_id, role, content, content_json,
                    turn_index, provenance, is_trusted):
        self.rows.append((turn_index, role, content))

    def turn_indexes(self) -> list[int]:
        return sorted(i for i, _r, _c in self.rows)


class _FailingEngine(_RecordingEngine):
    """Raises on the Nth ingest (1-based), records the successful ones."""

    def __init__(self, fail_on: int) -> None:
        super().__init__()
        self._calls = 0
        self._fail_on = fail_on

    def ingest_sync(self, **kw):
        self._calls += 1
        if self._calls == self._fail_on:
            raise RuntimeError("disk full")
        super().ingest_sync(**kw)


def test_overlapping_tail_persists_write_each_row_once():
    """The s4 fan, replayed: four turns' stale original_lens, zero duplicates.

    This is the exact shape of session desktop:s4-1766869.1781103365257 —
    four rapid-fire sends captured original_len 1/2/3/4, the shared list grew
    to 16 rows, and the four completions persisted overlapping tails. Before
    the fix that wrote 40 rows for 16 messages; every row must now land once.
    """
    eng = _RecordingEngine()
    s = ChatSession("desktop:fan", lcm_engine=eng)
    for i in range(4):
        s.add_user_message(f"m{i}")          # rows 0-3, persisted immediately

    # In-flight turns append in place to the shared list (rows 4-8), then the
    # first turn to finish persists from ITS original_len — which is BELOW
    # rows other sends/turns already persisted.
    for i in range(4, 9):
        s.messages.append(_asst(f"r{i}"))
    s.persist_loop_result(3)                 # finisher #1 (stale base 3)

    for i in range(9, 16):
        s.messages.append(_asst(f"r{i}"))
    s.persist_loop_result(4)                 # finisher #2
    s.persist_loop_result(1)                 # finisher #3 — fully covered span
    s.persist_loop_result(2)                 # finisher #4

    assert eng.turn_indexes() == list(range(16)), (
        "each of the 16 in-memory rows must persist exactly once; "
        f"got {eng.turn_indexes()}"
    )


def test_user_row_landing_mid_turn_is_not_repersisted():
    """A user send during a running turn persists immediately (the WS echo
    needs the rowid); the turn's later tail persist must skip it — and must
    still cover the turn's OWN rows sitting below it."""
    eng = _RecordingEngine()
    s = ChatSession("desktop:midturn", lcm_engine=eng)
    s.add_user_message("m0")                 # row 0
    original_len = len(s.messages)           # turn starts

    s.messages.append(_asst("r1"))           # turn appends rows 1-2,
    s.messages.append(_asst("r2"))           # not yet persisted
    s.add_user_message("m3")                 # mid-turn send → row 3, persisted now

    s.persist_loop_result(original_len)      # turn finishes
    assert eng.turn_indexes() == [0, 1, 2, 3]


def test_partial_ingest_failure_resumes_without_duplicates():
    """A mid-span ingest failure never raises; rows already written stay
    written, and a later overlapping persist picks up the remainder."""
    eng = _FailingEngine(fail_on=3)          # m0 ok, r1 ok, r2 raises
    s = ChatSession("desktop:resume", lcm_engine=eng)
    s.add_user_message("m0")
    s.messages.append(_asst("r1"))
    s.messages.append(_asst("r2"))
    s.messages.append(_asst("r3"))
    s.persist_loop_result(1)                 # writes r1, fails on r2 — silently
    assert eng.turn_indexes() == [0, 1]

    s.persist_loop_result(1)                 # retry same span
    assert eng.turn_indexes() == [0, 1, 2, 3], "r2/r3 recovered, r1 not doubled"


def test_ephemeral_hole_is_sealed_not_backfilled():
    """Rows created while the session was ephemeral (engine=None) must stay
    out of the store forever — a later persist with a low/stale start must
    not backfill them, and the watermark must not wedge below the hole."""
    eng = _RecordingEngine()
    s = ChatSession("desktop:eph", lcm_engine=eng)
    s.add_user_message("m0")                 # row 0 persisted
    s.set_lcm_engine(None)
    s.add_user_message("private")            # row 1 — deliberately unpersisted
    s.set_lcm_engine(eng)
    original_len = len(s.messages)
    s.messages.append(_asst("r2"))           # row 2, the new turn's tail
    s.persist_loop_result(original_len)
    assert eng.turn_indexes() == [0, 2]

    s.persist_loop_result(0)                 # stale low span — a would-be backfill
    assert eng.turn_indexes() == [0, 2], "the ephemeral row must never land"


def test_add_result_messages_with_stale_index_does_not_duplicate_rows():
    """The /api/chat shape: original_len passed one too low. ``run_async``
    returns a SHALLOW copy of the session list plus the loop tail, so the
    prefix entries are the very objects in ``session.messages`` — the
    identity skip must drop them, leaving no duplicate in memory OR in the
    store despite the stale index. (The call site itself is also fixed in
    web/server.py.)"""
    eng = _RecordingEngine()
    s = ChatSession("web:offbyone", lcm_engine=eng)
    s.add_user_message("q")                  # row 0 persisted
    result = [s.messages[0], _asst("a")]     # what run_async actually returns
    s.add_result_messages(result, 0)         # stale: should have been 1
    assert eng.rows == [(0, "user", "q"), (1, "assistant", "a")]
    assert len(s.messages) == 2, "the user turn must not be re-appended"


def test_add_result_messages_correct_caller_unchanged():
    """The gateway-adapter contract (pre_len captured after add_user_message)
    persists exactly the loop tail at the right positions."""
    eng = _RecordingEngine()
    s = ChatSession("telegram:1", lcm_engine=eng)
    s.add_user_message("q")
    pre_len = len(s.messages)
    result = [ConversationMessage.from_user_text("q"), _asst("a1"), _asst("a2")]
    s.add_result_messages(result, pre_len)
    assert eng.turn_indexes() == [0, 1, 2]
    assert len(s.messages) == 3


def test_clear_resets_watermark():
    eng = _RecordingEngine()
    s = ChatSession("desktop:clear", lcm_engine=eng)
    s.add_user_message("m0")
    s.clear()
    s.add_user_message("fresh")              # positions restart at 0
    assert (0, "user", "fresh") in eng.rows


def test_trim_shifts_watermark_with_positions():
    eng = _RecordingEngine()
    s = ChatSession("telegram:trim", lcm_engine=eng)
    for i in range(5):
        s.add_user_message(f"m{i}")          # rows 0-4 persisted
    s.trim(3)                                # keeps m2..m4 at positions 0..2
    s.add_user_message("m5")                 # lands at position 3
    assert (3, "user", "m5") in eng.rows
    # ...and the new row was not skipped as already-persisted:
    assert sum(1 for _i, _r, c in eng.rows if c == "m5") == 1


def test_rollback_retreats_watermark():
    """After rolling back a persisted row, the NEXT message at that position
    must persist (not be skipped as already-written)."""
    eng = _RecordingEngine()
    s = ChatSession("telegram:rb", lcm_engine=eng)
    s.add_user_message("bad")
    s.rollback_last()
    s.add_user_message("good")
    assert (0, "user", "good") in eng.rows


def test_persistence_never_raises():
    """Best-effort contract: a hostile engine must not break any entry point."""

    class _Hostile:
        def ingest_sync(self, **kw):
            raise RuntimeError("boom")

    s = ChatSession("desktop:hostile", lcm_engine=_Hostile())
    s.add_user_message("m0")                                     # no raise
    s.add_result_messages([ConversationMessage.from_user_text("m0"), _asst("a")], 1)
    s.persist_loop_result(0)                                     # no raise
