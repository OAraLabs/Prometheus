"""A score over model-proposed labels is AGREEMENT, not accuracy.

A chooser graded against labels another model produced can only be measured on
how alike they are. `tests/fixtures/divergence_traces.py` states the rule in its
opening paragraph: "a calibration round that cannot tell the two apart is
calibrating against its own author."

Like every other provenance field in this corpus, `label_source` cannot be
retrofitted. Once rows are labelled and nobody recorded who labelled them, the
human calibration set is no longer separable from the model-proposed bulk — and
that separation is the entire point of collecting both.
"""

from __future__ import annotations

import pytest

from prometheus.computer.corpus import (
    ANNOTATION_NONE_CORRECT,
    LABEL_HUMAN,
    LABEL_MODEL,
    LABEL_MODEL_CONFIRMED,
    REASON_NOT_ACHIEVABLE_HERE,
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


def _capture(store, rid):
    rec = TableRecord(goal="g", target="box", app="a", window_id=1)
    rec.record_id = rid
    rec.note_candidates([_C("click-0"), _C("click-1")])
    assert store.capture(rec)
    return rid


def test_setting_correct_id_without_a_label_source_is_refused(tmp_path):
    """THE amendment. There is no default — `human` would be the comfortable
    default and the wrong one."""
    store = _store(tmp_path)
    _capture(store, "r1")
    with pytest.raises(TypeError):
        store.record_annotation("r1", "click-0", annotated_by="will")


def test_an_unknown_label_source_is_refused(tmp_path):
    store = _store(tmp_path)
    _capture(store, "r1")
    with pytest.raises(ValueError, match="label_source must be one of"):
        store.record_annotation(
            "r1", "click-0", annotated_by="will", label_source="probably-fine"
        )


@pytest.mark.parametrize(
    "source", [LABEL_HUMAN, LABEL_MODEL, LABEL_MODEL_CONFIRMED]
)
def test_every_declared_source_round_trips(tmp_path, source):
    store = _store(tmp_path)
    _capture(store, source)
    store.record_annotation(
        source, "click-0", annotated_by="will", label_source=source
    )
    assert load_corpus(store).answered[0]["label_source"] == source


def test_a_model_labelled_corpus_never_says_accuracy(tmp_path):
    """THE REPORTING RULE, and the mutation for it.

    A reader who skims must not be able to pick up the wrong word. So the
    summary over an all-`model` corpus must not contain "accuracy" except
    inside an explicit denial of it.
    """
    store = _store(tmp_path)
    for i in range(4):
        rid = f"m{i}"
        _capture(store, rid)
        store.record_annotation(
            rid, "click-0", annotated_by="claude", label_source=LABEL_MODEL
        )

    corpus = load_corpus(store)
    assert corpus.label_mix() == {LABEL_MODEL: 4}
    assert corpus.score_noun() == "AGREEMENT WITH A MODEL — not accuracy"

    summary = corpus.summary()
    assert "AGREEMENT WITH A MODEL" in summary, summary
    assert "NOT ACCURACY" in summary, summary
    # Every occurrence of the word must be inside a denial, never a claim.
    for line in summary.splitlines():
        if "accuracy" in line.lower():
            assert any(
                marker in line for marker in ("NOT ACCURACY", "not accuracy")
            ), f"a line offers 'accuracy' without denying it: {line!r}"


def test_a_human_labelled_corpus_may_say_accuracy(tmp_path):
    """The guard must not fire on the case it exists to protect."""
    store = _store(tmp_path)
    for i in range(3):
        rid = f"h{i}"
        _capture(store, rid)
        store.record_annotation(
            rid, "click-0", annotated_by="will", label_source=LABEL_HUMAN
        )
    corpus = load_corpus(store)
    assert corpus.score_noun() == "accuracy (labels are human)"
    assert "NOT ACCURACY" not in corpus.summary()


def test_a_mixed_corpus_refuses_to_be_summarised_as_either(tmp_path):
    """Human + model in one number is the blend that hides the calibration set."""
    store = _store(tmp_path)
    _capture(store, "h")
    store.record_annotation("h", "click-0", annotated_by="will",
                            label_source=LABEL_HUMAN)
    _capture(store, "m")
    store.record_annotation("m", "click-1", annotated_by="claude",
                            label_source=LABEL_MODEL)

    corpus = load_corpus(store)
    assert corpus.score_noun().startswith("MIXED LABEL SOURCES")
    assert "split before reporting" in corpus.score_noun()


def test_the_mix_counts_none_correct_rows_too(tmp_path):
    """none_correct is a label like any other and carries a source."""
    store = _store(tmp_path)
    _capture(store, "n")
    store.record_annotation(
        "n", ANNOTATION_NONE_CORRECT, annotated_by="will",
        label_source=LABEL_HUMAN, none_correct_reason=REASON_NOT_ACHIEVABLE_HERE,
    )
    assert load_corpus(store).label_mix() == {LABEL_HUMAN: 1}


def test_a_container_opening_answer_is_counted_and_flagged(tmp_path):
    """Over half the correct answers being "open a container" is a warning
    about the GOALS, not a result about the chooser."""
    from prometheus.computer.corpus import LABEL_MODEL

    store = _store(tmp_path)
    for i in range(3):
        rid = f"m{i}"
        rec = TableRecord(goal="g", target="box", app="a", window_id=1)
        rec.record_id = rid

        class _M:
            candidate_id = "click-9"
            description = "Click the push button 'Main menu'"

        rec.note_candidates([_M()])
        assert store.capture(rec)
        store.record_annotation(rid, "click-9", annotated_by="claude",
                                label_source=LABEL_MODEL)

    corpus = load_corpus(store)
    assert corpus.menu_opening_share() == (3, 3)
    assert "OVER HALF" in corpus.summary()
    assert "too coarse for one bounded step" in corpus.summary()
