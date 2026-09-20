"""A declared column empty in every row is a defect, not a detail.

`history` was declared, columned, and never populated — every row stored `[]`.
The corpus looked complete and would have scored choosers on an input the
original never saw. It was found by reading the loop, which is not a control.

This is the control. It runs as harvest exit criteria, because a
universally-empty column cannot be repaired afterwards: the value was never
captured, so the harvest has to be re-run.
"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

from prometheus.computer.chooser import RuleChooser
from prometheus.computer.corpus import (
    INTENTIONALLY_ABSENT,
    CorpusStore,
    IncompleteCorpus,
    assert_corpus_complete,
)
from prometheus.computer.driver import FixtureDriver
from prometheus.computer.loop import ComputerUseLoop
from prometheus.computer.types import Element, Observation
from prometheus.permissions.checker import PermissionMode, SecurityGate


def _obs(snapshot="s1"):
    return Observation(
        target="box", app="scratchapp", pid=1, window_id=2,
        snapshot_id=snapshot,
        elements=(
            Element(0, f"tok-send-{snapshot}", "push button", "Send"),
            Element(1, f"tok-cancel-{snapshot}", "push button", "Cancel"),
        ),
    )


async def _accept(*a, **k):
    return True


def _harvest(tmp_path, n=3) -> CorpusStore:
    """A small REAL harvest — the loop, the gate, the approval path.

    Deliberately representative, and the completeness guard is what forced it
    to be. The first version used ``ScriptedChooser`` and a single-observation
    ``FixtureDriver``, and the guard correctly refused it: ScriptedChooser is a
    test double that sets no confidence, and with one observation ``_verify``
    re-observes the SAME snapshot and returns None. Both would have produced a
    corpus with two universally-empty columns — which is the very defect this
    file exists to catch, arriving through the fixture.

    So: ``RuleChooser`` (what a harvest actually uses, and it always sets
    confidence) and TWO observations (so the post-action observe mints a new
    snapshot and ``_verify`` can answer).
    """
    store = CorpusStore(tmp_path / "corpus.db")
    for i in range(n):
        loop = ComputerUseLoop(
            driver=FixtureDriver([_obs(f"s{i}a"), _obs(f"s{i}b")]),
            chooser=RuleChooser(prefer=("send",)),
            gate=SecurityGate(mode=PermissionMode.DEFAULT, audit_logger=None),
            approve=_accept, skip_preconditions=True, corpus=store,
        )
        asyncio.run(loop.step(
            "press send", "box", "scratchapp", 1, 2,
            history=[f"step {i}"], goal_source="human",
            harvest_session="t",
        ))
    return store


def test_a_real_harvest_is_complete(tmp_path):
    """The exit criterion, on data produced by the actual loop."""
    assert_corpus_complete(_harvest(tmp_path))


def test_an_empty_corpus_is_a_failure_not_a_pass(tmp_path):
    """Nothing harvested means completeness is unmeasured, not satisfied."""
    store = CorpusStore(tmp_path / "corpus.db")
    with pytest.raises(IncompleteCorpus, match="not measurable"):
        assert_corpus_complete(store)


@pytest.mark.parametrize("column,reported", [
    # The guard reports the name a READER sees: all_tables() decodes
    # history_json -> history and candidates_json -> candidates, so those are
    # the names in the message. Pinned as a pair so the message stays the
    # actionable one rather than the storage detail.
    ("history_json", "history"),
    ("goal", "goal"),
    ("deterministic_id", "deterministic_id"),
    ("candidates_json", "candidates"),
])
def test_blanking_a_populated_column_is_caught(tmp_path, column, reported):
    """THE MUTATION, as a test rather than a one-off.

    Blank one column that the harvest genuinely fills, across every row, and
    the guard must name it. This is the `history` defect reproduced on demand.
    """
    store = _harvest(tmp_path)
    assert_corpus_complete(store)          # green before

    blank = "[]" if column.endswith("_json") else ""
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(f"UPDATE tables SET {column} = ?", (blank,))

    with pytest.raises(IncompleteCorpus) as exc:
        assert_corpus_complete(store)
    assert reported in str(exc.value), (
        f"the guard fired but did not NAME {reported}; an unnamed column is a "
        f"review task, not a control"
    )


def test_falsy_is_not_empty(tmp_path):
    """0 and False are real values, not gaps.

    `deterministic_abstained` is 0 on every row of a corpus where the chooser
    always answered, and `verified` is legitimately False. A guard that treats
    falsy as empty fires on a healthy corpus and gets switched off.
    """
    store = _harvest(tmp_path)
    with sqlite3.connect(store.db_path) as conn:
        conn.execute("UPDATE tables SET deterministic_abstained = 0, verified = 0")
    assert_corpus_complete(store)


def test_every_intentional_absence_carries_a_reason():
    """A name with no sentence is silence with extra steps."""
    for col, reason in INTENTIONALLY_ABSENT.items():
        assert reason and len(reason) > 15, (
            f"{col!r} is excluded from the completeness guard with no real "
            f"reason. Declaring absence is a decision and has to read like one."
        )


def test_the_exclusion_list_does_not_hide_a_wired_column(tmp_path):
    """An entry that is actually populated is a stale exclusion.

    If a field gets wired and nobody removes its entry, the guard stops
    watching a column that now matters — the exclusion outliving its reason.
    """
    store = _harvest(tmp_path)
    rows = store.all_tables()
    stale = [
        col for col in INTENTIONALLY_ABSENT
        if col in rows[0] and any(
            rows[i].get(col) not in (None, "", "[]", "{}") for i in range(len(rows))
        )
    ]
    assert not stale, (
        f"{stale} are listed in INTENTIONALLY_ABSENT but are populated by a "
        f"real harvest. Remove the entries — the guard is no longer watching "
        f"columns that now carry data."
    )


def test_capture_mode_is_narrower_and_full_mode_still_catches(tmp_path):
    """The capture exclusion is SCOPED, not a permanent widening.

    A capture-only harvest cannot populate the action-outcome columns, and that
    is not a defect. But a corpus of REAL runs must still be held to them —
    blanket-excluding them would stop the guard watching five columns forever
    to make one harvest mode pass, which is weakening the instrument to obtain
    a pass.
    """
    import sqlite3

    store = _harvest(tmp_path)
    assert_corpus_complete(store, mode="full")       # real runs: complete

    # Simulate capture-only: blank everything an action would have produced.
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE tables SET executed_candidate_id=NULL, verified=NULL, "
            "gate_decision=NULL, gate_approval_required=NULL, "
            "gate_approval_granted=NULL"
        )

    assert_corpus_complete(store, mode="capture")    # capture: fine

    with pytest.raises(IncompleteCorpus) as exc:
        assert_corpus_complete(store, mode="full")   # full: still caught
    for col in ("executed_candidate_id", "verified", "gate_decision"):
        assert col in str(exc.value), f"full mode stopped watching {col}"


def test_an_unknown_mode_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown mode"):
        assert_corpus_complete(_harvest(tmp_path), mode="lenient")


def test_the_history_exclusion_names_the_test_that_proves_the_wiring():
    """An exclusion for the field whose ABSENCE was the original defect has to
    carry evidence, not an assertion.

    This guard cannot distinguish "wired but the input was empty" from "never
    wired" — which is precisely how `history` hid. So its capture-mode
    exclusion must point at the test that fails if the loop stops populating
    it, and that test must exist.
    """
    from pathlib import Path

    from prometheus.computer.corpus import CAPTURE_ONLY_ABSENT

    reason = CAPTURE_ONLY_ABSENT["history"]
    assert "test_history_reaches_the_corpus" in reason, (
        "the history exclusion does not name the test that proves the wiring; "
        "without it this is an assertion that the field works, which is what "
        "the original defect also looked like"
    )
    named = (
        Path(__file__).resolve().parent / "test_computer_corpus_reasons.py"
    ).read_text()
    assert "def test_history_reaches_the_corpus" in named, (
        "the exclusion names a test that does not exist"
    )
