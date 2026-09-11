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
