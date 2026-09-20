"""A `none_correct` row must say WHY, because the four causes disagree.

Three of the four reasons are TABLE defects — the goal needed a verb, a key or
a modifier that `build_candidates` never offers — and they are fixable
deterministically, with no classifier and no dependency. Only
`not_achievable_here` is a row where abstaining is the correct answer, and it
is therefore the only population against which a chooser that picks something
scores a REGRESSION rather than a gain.

A corpus that cannot separate those measures the table's limits and reports
them as the chooser's. It cannot be backfilled: nobody recorded why.
"""

from __future__ import annotations

import pytest

from prometheus.computer.corpus import (
    ANNOTATION_NONE_CORRECT,
    ANNOTATION_TABLE_UNUSABLE,
    NONE_CORRECT_REASONS,
    REASON_KEY_NOT_OFFERED,
    REASON_NOT_ACHIEVABLE_HERE,
    REASON_VERB_NOT_OFFERED,
    LABEL_HUMAN,
    CorpusStore,
    TableRecord,
    load_corpus,
)


class _C:
    def __init__(self, cid):
        self.candidate_id = cid
        self.description = f"Click {cid}"


def _store(tmp_path):
    return CorpusStore(tmp_path / "corpus.db")


def _capture(store, rid="r1"):
    rec = TableRecord(goal="g", target="box", app="a", window_id=1)
    rec.record_id = rid
    rec.note_candidates([_C("click-0"), _C("click-1")])
    assert store.capture(rec)
    return rid


def test_none_correct_without_a_reason_is_refused(tmp_path):
    """THE amendment. A reason cannot be added later."""
    store = _store(tmp_path)
    _capture(store)
    with pytest.raises(ValueError, match="requires none_correct_reason"):
        store.record_annotation("r1", ANNOTATION_NONE_CORRECT, annotated_by="will", label_source=LABEL_HUMAN)


def test_a_free_text_reason_is_refused(tmp_path):
    """Reasons are aggregated; a free-text one cannot be."""
    store = _store(tmp_path)
    _capture(store)
    with pytest.raises(ValueError, match="not a known reason"):
        store.record_annotation(
            "r1", ANNOTATION_NONE_CORRECT,
            annotated_by="will", label_source=LABEL_HUMAN, none_correct_reason="couldn't do it",
        )


def test_a_reason_beside_a_real_candidate_is_refused(tmp_path):
    """It would read as explaining a choice it does not explain."""
    store = _store(tmp_path)
    _capture(store)
    with pytest.raises(ValueError, match="only meaningful with"):
        store.record_annotation(
            "r1", "click-0", annotated_by="will", label_source=LABEL_HUMAN,
            none_correct_reason=REASON_VERB_NOT_OFFERED,
        )
    with pytest.raises(ValueError, match="only meaningful with"):
        store.record_annotation(
            "r1", ANNOTATION_TABLE_UNUSABLE, annotated_by="will", label_source=LABEL_HUMAN,
            none_correct_reason=REASON_VERB_NOT_OFFERED,
        )


@pytest.mark.parametrize("reason", sorted(NONE_CORRECT_REASONS))
def test_every_declared_reason_round_trips(tmp_path, reason):
    store = _store(tmp_path)
    _capture(store, reason)
    store.record_annotation(
        reason, ANNOTATION_NONE_CORRECT,
        annotated_by="will", label_source=LABEL_HUMAN, none_correct_reason=reason,
    )
    row = load_corpus(store).none_correct[0]
    assert row["none_correct_reason"] == reason


def test_the_split_separates_table_defects_from_genuine_abstains(tmp_path):
    """The number that decides the sequencing.

    Two rows the deterministic path can fix, one where abstaining is right.
    Blending them would report 3 chooser failures and hide that 2 of them are
    `build_candidates` never offering the verb.
    """
    store = _store(tmp_path)
    for rid, reason in (
        ("a", REASON_VERB_NOT_OFFERED),
        ("b", REASON_KEY_NOT_OFFERED),
        ("c", REASON_NOT_ACHIEVABLE_HERE),
    ):
        _capture(store, rid)
        store.record_annotation(
            rid, ANNOTATION_NONE_CORRECT,
            annotated_by="will", label_source=LABEL_HUMAN, none_correct_reason=reason,
        )

    corpus = load_corpus(store)
    assert corpus.none_correct_by_reason() == {
        REASON_VERB_NOT_OFFERED: 1,
        REASON_KEY_NOT_OFFERED: 1,
        REASON_NOT_ACHIEVABLE_HERE: 1,
    }
    summary = corpus.summary()
    assert "TABLE DEFECTS" in summary, (
        "the summary did not flag that some none-correct rows are table "
        "defects — that warning is what stops a classifier being judged "
        "against the table's limits"
    )
    assert "2 row(s) are TABLE DEFECTS" in summary, summary


def test_the_column_and_its_provenance_cannot_drift(tmp_path):
    """Both written in ONE transaction."""
    import sqlite3

    store = _store(tmp_path)
    _capture(store)
    store.record_annotation(
        "r1", ANNOTATION_NONE_CORRECT,
        annotated_by="will", label_source=LABEL_HUMAN, none_correct_reason=REASON_VERB_NOT_OFFERED,
    )
    with sqlite3.connect(store.db_path) as conn:
        on_row = conn.execute(
            "SELECT correct_id, none_correct_reason FROM tables WHERE record_id='r1'"
        ).fetchone()
        in_ann = conn.execute(
            "SELECT correct_candidate_id, none_correct_reason FROM annotations "
            "WHERE record_id='r1'"
        ).fetchone()
    assert on_row == in_ann == (ANNOTATION_NONE_CORRECT, REASON_VERB_NOT_OFFERED)


def test_history_reaches_the_corpus(tmp_path):
    """`history` is part of the replay surface, so it must actually be stored.

    It was declared on TableRecord, written to the history_json column, and
    NEVER POPULATED — every row stored `[]`. A replayed ChoiceRequest would
    then carry empty history where the original had entries, which is a
    different input to the chooser. The same orphan shape this PR diagnoses in
    build_candidates, in the code that diagnoses it.
    """
    import asyncio

    from prometheus.computer.chooser import ScriptedChooser
    from prometheus.computer.driver import FixtureDriver
    from prometheus.computer.loop import ComputerUseLoop
    from prometheus.computer.types import (
        CANDIDATE_ABSTAIN, Element, Observation,
    )
    from prometheus.permissions.checker import PermissionMode, SecurityGate

    store = _store(tmp_path)
    obs = Observation(
        target="box", app="scratchapp", pid=1, window_id=2, snapshot_id="s1",
        elements=(Element(0, "tok-0", "push button", "Send"),),
    )
    loop = ComputerUseLoop(
        driver=FixtureDriver([obs]),
        chooser=ScriptedChooser([CANDIDATE_ABSTAIN]),
        gate=SecurityGate(mode=PermissionMode.DEFAULT, audit_logger=None),
        skip_preconditions=True, corpus=store,
    )
    asyncio.run(loop.step(
        "press send", "box", "scratchapp", 1, 2,
        history=["clicked Open", "typed a filename"],
    ))

    stored = store.all_tables()[0]["history"]
    assert stored == ["clicked Open", "typed a filename"], (
        f"history did not reach the corpus: {stored!r}. A replayed "
        f"ChoiceRequest would differ from the one the chooser answered."
    )
