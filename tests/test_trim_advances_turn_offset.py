"""Trimming a session must not re-use durable turn indices.

THE DEFECT
----------
`turn_index` is stamped at persist time as ``list position +
_turn_index_offset``. `trim()` shifted every surviving message DOWN the list
and adjusted `_lcm_persisted_len` / `_lcm_persisted_ahead` to match — but never
advanced the offset. So after the window filled, the next row re-used an
ordinal a dropped message had already written.

Measured, 60 messages through the 50-message window:

    durable rows written : 60
    distinct turn_index  : 51
    turn_index 50 written TEN times, for m50 ... m59

Four store readers ``ORDER BY turn_index``, including the LCM compactor and
assembler, so two messages sharing an ordinal makes their order undefined.

`_turn_index_offset` already existed for the rehydrate case — `restore()` sets
it so rows written after a rehydrate continue the durable numbering instead of
colliding with the history the restored tail came from. Trimming is the same
situation arrived at from the other direction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prometheus.engine.session import (  # noqa: E402
    MAX_SESSION_MESSAGES,
    ChatSession,
)


class _RecordingEngine:
    """Stands in for the LCM engine; records the turn_index of every row."""

    def __init__(self) -> None:
        self.rows: list[tuple[int, str]] = []

    def ingest_sync(self, *, session_id, role, content, content_json,
                    turn_index, provenance=None, is_trusted=True, **kw):
        self.rows.append((turn_index, content))
        return len(self.rows)


def _drive(n_messages: int) -> _RecordingEngine:
    """The ordinary path: add a message, persist it, trim."""
    engine = _RecordingEngine()
    session = ChatSession(session_id="s")
    session.set_lcm_engine(engine)
    for n in range(n_messages):
        session.add_user_message(f"m{n}")
        session.trim()
    return engine


def test_turn_indices_are_unique_past_the_trim_threshold():
    """THE defect. Every ordinal must belong to exactly one message."""
    total = MAX_SESSION_MESSAGES + 10
    engine = _drive(total)

    indices = [t for t, _ in engine.rows]
    assert len(engine.rows) == total, (
        f"only {len(engine.rows)} of {total} messages persisted — this test "
        f"cannot say anything about their indices"
    )

    duplicates = sorted({i for i in indices if indices.count(i) > 1})
    assert not duplicates, (
        f"turn_index values re-used: {duplicates}. "
        + "; ".join(
            f"{d} written for {[c for t, c in engine.rows if t == d]}"
            for d in duplicates[:3]
        )
    )
    assert len(set(indices)) == total


def test_turn_indices_are_strictly_increasing():
    """Uniqueness alone is not enough — the readers sort on this column.

    A scheme that produced unique-but-unordered ordinals would satisfy the
    test above and still interleave new rows into old history.
    """
    engine = _drive(MAX_SESSION_MESSAGES + 10)
    indices = [t for t, _ in engine.rows]
    assert indices == sorted(indices), f"turn_index went backwards: {indices}"
    assert indices == list(range(len(indices)))


def test_the_message_content_matches_its_ordinal():
    """Assert on CONTENT, not just on the shape of the numbers.

    `m7` must be at ordinal 7. A fix that advanced the offset by the wrong
    amount would still yield unique, increasing indices attached to the wrong
    messages.
    """
    engine = _drive(MAX_SESSION_MESSAGES + 10)
    for turn_index, content in engine.rows:
        assert content == f"m{turn_index}", (
            f"turn_index {turn_index} carries {content!r}"
        )


def test_a_session_that_never_trims_is_unchanged():
    """The offset must stay zero for a session under the window.

    Without this, `_turn_index_offset += 1` on every call — trim or not —
    would pass every test above while breaking the cold-session contract.
    """
    engine = _drive(MAX_SESSION_MESSAGES - 5)
    indices = [t for t, _ in engine.rows]
    assert indices == list(range(MAX_SESSION_MESSAGES - 5))


def test_trim_still_bounds_the_in_memory_history():
    """The fix must not stop trim from trimming."""
    session = ChatSession(session_id="s")
    for n in range(MAX_SESSION_MESSAGES + 10):
        session.add_user_message(f"m{n}")
    session.trim()
    assert len(session.messages) == MAX_SESSION_MESSAGES
    assert session.messages[-1].text == f"m{MAX_SESSION_MESSAGES + 9}"


def test_the_offset_advances_by_exactly_what_was_dropped():
    """The arithmetic, stated directly.

    `position + offset` must be invariant across a trim for every surviving
    message — that is the whole property, and it is worth asserting on the
    number rather than only on its effects.
    """
    session = ChatSession(session_id="s")
    for n in range(MAX_SESSION_MESSAGES + 7):
        session.add_user_message(f"m{n}")

    before_offset = session._turn_index_offset
    before_len = len(session.messages)
    session.trim()
    dropped = before_len - len(session.messages)

    assert dropped == 7
    assert session._turn_index_offset == before_offset + dropped


# ───────────────────────────────────────────────────────────────────────────
# THE SIGN-OFF GUARD
#
# `test_trim_shifts_watermark_with_positions` in
# tests/test_session_persist_exact_once.py had to have an ASSERTION EDITED:
# it asserted `turn_index == 3` for "m5", and that value IS the collision this
# fix removes. Editing an existing assertion is the one change that can turn a
# real regression into a green suite, so the edit does not stand alone.
#
# This test fails on the old behaviour in a way that edit cannot mask. It lives
# in a different file, it never asserts a specific ordinal, and it counts the
# REPEAT — the shape the reproduction actually produced:
#
#     durable rows written : 60
#     distinct turn_index  : 51
#     turn_index 50 written TEN times, for m50 ... m59
#
# If the edited assertion and this test do not BOTH fail against a reverted
# `session.py`, one of them is not earning its place.
# ───────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("extra", [2, 10, 25])
def test_every_written_ordinal_is_distinct_and_ascending(extra):
    """N persists produce N ordinals: all distinct, strictly ascending.

    Parameterised past the window by three different amounts so the property
    is asserted over the RUN, not over one arithmetic coincidence.

    ⚠ `extra=1` IS DELIBERATELY ABSENT, AND WAS IN THE FIRST DRAFT. It passes
    against the BROKEN code: a run of MAX+1 messages trims exactly once, and
    the collision needs a persist AFTER a trim to land on a re-used position.
    One trim with nothing following it cannot collide, so that parameter was a
    green light measuring nothing. Verified — reverted, `[1]` passed while
    `[10]` and `[25]` failed. See the boundary test below, which states that
    property on purpose instead of smuggling it in as a passing case.
    """
    total = MAX_SESSION_MESSAGES + extra
    engine = _drive(total)
    indices = [t for t, _ in engine.rows]

    assert len(indices) == total, (
        f"{len(indices)} rows written for {total} messages — the harness is "
        f"not persisting what this test assumes, so its verdict is worthless"
    )

    # Count the repeats explicitly: this is the number the reproduction
    # reported, and the failure message should read like that reproduction.
    from collections import Counter

    counts = Counter(indices)
    repeated = {idx: n for idx, n in counts.items() if n > 1}
    worst = max(counts.values())

    assert not repeated, (
        f"{len(indices)} rows written, only {len(counts)} distinct turn_index "
        f"values — one ordinal was re-used up to {worst} times.\n  "
        + "\n  ".join(
            f"turn_index {idx} written {n} times, for "
            f"{[c for t, c in engine.rows if t == idx]}"
            for idx, n in sorted(repeated.items())[:3]
        )
    )

    assert indices == sorted(indices), (
        f"turn_index went backwards; four store readers ORDER BY this column: "
        f"{indices}"
    )
    assert len(set(indices)) == total


def test_the_first_trim_alone_cannot_collide():
    """The boundary, asserted as a fact rather than left as a silent pass.

    A collision needs a persist that lands on a position a dropped message
    already wrote. At exactly MAX+1 messages there is one trim and nothing
    after it, so the ordinals are distinct on BROKEN code too.

    This is recorded because it was nearly shipped as a parameter of the test
    above, where it would have looked like evidence and been none.
    """
    engine = _drive(MAX_SESSION_MESSAGES + 1)
    indices = [t for t, _ in engine.rows]
    assert len(set(indices)) == len(indices) == MAX_SESSION_MESSAGES + 1
