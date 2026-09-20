"""A pilot corpus is never scored, and a diagnostic is never a label.

Two separations, both structural:

* `pilot` is a property of the SESSION, so it lives in its own table rather
  than being written back onto captured rows. A pilot run exists to shake out
  the harvest; scoring against it imports whatever was wrong with the run it
  was built to find.
* An abstain diagnostic answers "could the table express this goal at all",
  which is a question about the TABLE. It is not `correct_id`, it must never be
  counted as ground truth, and it must not consume the human calibration set —
  so it has its own table and its own method.
"""

from __future__ import annotations

import pytest

from prometheus.computer.corpus import (
    REASON_NOT_ACHIEVABLE_HERE,
    REASON_VERB_NOT_OFFERED,
    CorpusStore,
    TableRecord,
    load_corpus,
)


class _C:
    def __init__(self, cid):
        self.candidate_id = cid
        self.description = f"Click {cid}"


def _store(tmp_path):
    return CorpusStore(tmp_path / "c.db")


def _capture(store, rid, session):
    rec = TableRecord(goal="g", target="box", app="a", window_id=1,
                      harvest_session=session)
    rec.record_id = rid
    rec.note_candidates([_C("click-0"), _C("click-1")])
    assert store.capture(rec)
    return rid


def test_a_pilot_session_is_excluded_from_scoring(tmp_path):
    store = _store(tmp_path)
    _capture(store, "p1", "pilot-run")
    _capture(store, "r1", "real-run")
    store.mark_session("pilot-run", "pilot")

    corpus = load_corpus(store)
    assert [r["record_id"] for r in corpus.pilot_excluded] == ["p1"]
    assert [r["record_id"] for r in corpus.unannotated] == ["r1"]
    assert "EXCLUDED — pilot session" in corpus.summary()


def test_pilot_rows_are_still_reachable_when_asked_for(tmp_path):
    """Excluded from scoring is not the same as hidden."""
    store = _store(tmp_path)
    _capture(store, "p1", "pilot-run")
    store.mark_session("pilot-run", "pilot")
    assert len(load_corpus(store, include_pilot=True).unannotated) == 1


def test_an_unknown_session_status_is_refused(tmp_path):
    with pytest.raises(ValueError, match="pilot' or 'real'"):
        _store(tmp_path).mark_session("s", "probably-fine")


def test_a_diagnostic_does_not_set_correct_id(tmp_path):
    """THE separation. Diagnosing the table must not spend the label."""
    store = _store(tmp_path)
    _capture(store, "r1", "s")
    store.record_abstain_diagnostic(
        "r1", correct_present=False, reason=REASON_VERB_NOT_OFFERED,
        answered_by="will",
    )
    assert store.abstain_diagnostics()["r1"]["reason"] == REASON_VERB_NOT_OFFERED
    # still unannotated: no label was consumed
    assert [r["record_id"] for r in load_corpus(store).unannotated] == ["r1"]
    assert store.get_table("r1")["correct_id"] is None


def test_no_correct_candidate_requires_a_reason(tmp_path):
    """The reason IS the measurement."""
    store = _store(tmp_path)
    _capture(store, "r1", "s")
    with pytest.raises(ValueError, match="needs a reason"):
        store.record_abstain_diagnostic(
            "r1", correct_present=False, answered_by="will"
        )


def test_a_reason_with_a_present_candidate_is_refused(tmp_path):
    store = _store(tmp_path)
    _capture(store, "r1", "s")
    with pytest.raises(ValueError, match="only meaningful when no candidate"):
        store.record_abstain_diagnostic(
            "r1", correct_present=True, reason=REASON_NOT_ACHIEVABLE_HERE,
            answered_by="will",
        )


def test_diagnosing_an_unknown_record_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no captured table"):
        _store(tmp_path).record_abstain_diagnostic(
            "nope", correct_present=True, answered_by="will"
        )
