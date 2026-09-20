"""The corpus captures every outcome, and cannot confuse "unknown" with "none".

WHAT THESE TESTS ARE FOR
------------------------
Two defects would make the corpus worse than useless, because both produce a
dataset that scores cleanly while meaning something other than it claims:

1. **A recorder that only sees the success path.** The rows worth having are
   disproportionately the ones where something did not go to plan — a refusal,
   an abstain, an operator declining, a crash. A corpus of executed steps would
   capture exactly the rows that teach the least. So the outcome coverage test
   below is parametrised over every terminal state ``step()`` has, and it
   asserts on what reached the STORE, not on what ``step()`` returned.

2. **Unannotated collapsing into "no candidate was correct".** Absence of a
   judgment means nobody has looked. If that were stored as "abstain was right",
   a chooser that always abstains would score 100% on an unreviewed corpus. The
   four states are pinned individually.

No test here reads the host: every one builds its own store in ``tmp_path`` and
its own observations. The results are identical headless and with a display.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest

from prometheus.computer.corpus import (
    ANNOTATION_NONE_CORRECT,
    REASON_NOT_ACHIEVABLE_HERE,
    ANNOTATION_TABLE_UNUSABLE,
    CORPUS_SCHEMA_VERSION,
    LABEL_HUMAN,
    CorpusStore,
    TableRecord,
    load_corpus,
)
from prometheus.computer.candidates import build_candidates
from prometheus.computer.chooser import ScriptedChooser
from prometheus.computer.driver import FixtureDriver
from prometheus.computer.loop import ComputerUseLoop
from prometheus.computer.types import (
    CANDIDATE_ABSTAIN,
    CANDIDATE_REOBSERVE,
    Element,
    Observation,
)
from prometheus.permissions.checker import PermissionMode, SecurityGate


def _obs(snapshot: str = "s1", *, unusable: str | None = None) -> Observation:
    if unusable:
        return Observation(
            target="box", app="scratchapp", pid=1, window_id=2,
            snapshot_id=snapshot, unusable_reason=unusable,
        )
    return Observation(
        target="box", app="scratchapp", pid=1, window_id=2,
        snapshot_id=snapshot,
        elements=(
            Element(0, f"tok-send-{snapshot}", "push button", "Send"),
            Element(1, f"tok-cancel-{snapshot}", "push button", "Cancel"),
            Element(2, f"tok-field-{snapshot}", "text", "Search", editable=True),
        ),
    )


def _store(tmp_path) -> CorpusStore:
    return CorpusStore(tmp_path / "corpus.db")


def _loop(store, chooser, *, observations=None, approve=None, gate=None):
    return ComputerUseLoop(
        driver=FixtureDriver(observations or [_obs()]),
        chooser=chooser,
        gate=gate or SecurityGate(mode=PermissionMode.DEFAULT, audit_logger=None),
        approve=approve,
        skip_preconditions=True,
        corpus=store,
    )


# ── EVERY OUTCOME IS CAPTURED ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,chooser_ids,observations,approve,expected_status",
    [
        ("invalid_id", ["no-such-candidate"], None, None, "refused"),
        ("abstain", [CANDIDATE_ABSTAIN], None, None, "abstained"),
        ("reobserve", [CANDIDATE_REOBSERVE], None, None, "reobserve"),
        (
            "unusable_observation",
            ["click-0"],
            [_obs(unusable="no AT-SPI bus; the tree would be empty")],
            None,
            "blocked",
        ),
        ("operator_declined", ["click-0"], None, "decline", "refused"),
        ("no_approver", ["click-0"], None, None, "refused"),
        ("executed", ["click-0"], None, "accept", "executed"),
    ],
)
def test_a_record_lands_for_every_outcome(
    tmp_path, name, chooser_ids, observations, approve, expected_status
):
    """Parametrised over each terminal state ``step()`` can reach.

    ``step()`` has nine return sites and what is in scope differs at each, so
    there is no single point INSIDE it that could emit for all of them. If the
    emit ever moves back inside the body, this test goes red for whichever
    outcomes that point cannot see.
    """
    store = _store(tmp_path)

    async def _accept(*a, **k):
        return True

    async def _decline(*a, **k):
        return False

    approver = {"accept": _accept, "decline": _decline}.get(approve)
    loop = _loop(
        store, ScriptedChooser(chooser_ids),
        observations=observations, approve=approver,
    )
    result = asyncio.run(
        loop.step("press send", "box", "scratchapp", 1, 2)
    )
    assert result.status == expected_status, f"{name}: setup did not reach the path"

    rows = store.all_tables()
    assert len(rows) == 1, (
        f"{name}: {len(rows)} rows captured for a step that ended "
        f"{result.status!r} — every outcome must leave exactly one record"
    )
    assert rows[0]["status"] == expected_status


def test_a_record_lands_when_the_step_raises(tmp_path):
    """A crash mid-step is exactly the row worth having, and the one a
    return-only recorder would lose."""
    store = _store(tmp_path)

    class Exploding:
        name = "exploding"

        def choose(self, request):
            raise RuntimeError("chooser blew up")

    loop = _loop(store, Exploding())
    with pytest.raises(RuntimeError, match="chooser blew up"):
        asyncio.run(loop.step("press send", "box", "scratchapp", 1, 2))

    rows = store.all_tables()
    assert len(rows) == 1, "a raised step captured nothing"
    assert rows[0]["status"] == "raised"
    assert "chooser blew up" in (rows[0]["exception"] or "")
    expected = len(build_candidates(_obs()))
    assert rows[0]["candidate_count"] == expected, (
        "the table was already built when the chooser raised, so it must be "
        "in the record — otherwise the crash row cannot be scored later"
    )


def test_the_table_is_captured_even_when_nothing_executed(tmp_path):
    """The whole point: a refusal still carries the full table."""
    store = _store(tmp_path)
    loop = _loop(store, ScriptedChooser([CANDIDATE_ABSTAIN]))
    asyncio.run(loop.step("press send", "box", "scratchapp", 1, 2))

    row = store.all_tables()[0]
    assert row["executed_candidate_id"] is None
    assert row["deterministic_id"] == CANDIDATE_ABSTAIN
    assert row["candidate_count"] > 0
    ids = {c["id"] for c in row["candidates"]}
    assert "click-0" in ids and "click-1" in ids


# ── WHAT MUST NEVER ENTER THE CORPUS ────────────────────────────────────────


def test_arguments_and_payloads_never_reach_the_corpus(tmp_path):
    """The chooser never saw arguments; neither does anything scoring it.

    Element tokens die with the snapshot and a typed payload is the caller's
    text, not the table's. Both would be permanent if stored.
    """
    store = _store(tmp_path)
    loop = _loop(store, ScriptedChooser([CANDIDATE_ABSTAIN]))
    asyncio.run(
        loop.step(
            "type it", "box", "scratchapp", 1, 2,
            text_to_type="hunter2-the-secret",
        )
    )
    blob = json.dumps(store.all_tables()[0])
    assert "hunter2-the-secret" not in blob, f"a payload was persisted: {blob}"
    assert "tok-send" not in blob, f"an element token was persisted: {blob}"


def test_the_replay_surface_is_stored_verbatim(tmp_path):
    """SUPERSEDES an earlier test that asserted descriptions were truncated.

    The ruling is that the table is stored exactly as offered, because
    ``replay_request`` must hand a future chooser the object the original
    chooser was handed. A description shortened to 300 characters is a
    DIFFERENT input, and scoring against it measures a different question.

    The diagnostic fields (``reason``, ``exception``) are still scrubbed — the
    line is replay fidelity, not "everything is now raw".
    """
    store = _store(tmp_path)
    rec = TableRecord(goal="g", target="box", app="a", window_id=1)

    long_desc = "Click the push button " + "A" * 400

    class FakeCandidate:
        candidate_id = "click-0"
        description = long_desc

    rec.note_candidates([FakeCandidate()])
    assert store.capture(rec)

    stored = store.all_tables()[0]["candidates"][0]["description"]
    assert stored == long_desc, (
        "the description was altered on the way in, so a replayed "
        "ChoiceRequest is not the one the chooser saw"
    )


# ── THE FOUR ANNOTATION STATES ──────────────────────────────────────────────


def _capture_one(store, record_id="r1") -> str:
    rec = TableRecord(goal="g", target="box", app="a", window_id=1)
    rec.record_id = record_id

    class C:
        def __init__(self, cid):
            self.candidate_id = cid
            self.description = f"Click {cid}"

    rec.note_candidates([C("click-0"), C("click-1")])
    assert store.capture(rec)
    return record_id


def test_unannotated_is_not_none_correct(tmp_path):
    """THE collapse to prevent.

    If absence of a judgment read as "abstain was right", a chooser that always
    abstains would score 100% on a corpus nobody has reviewed.
    """
    store = _store(tmp_path)
    _capture_one(store, "unreviewed")
    _capture_one(store, "reviewed")
    store.record_annotation(
        "reviewed", ANNOTATION_NONE_CORRECT, annotated_by="will", label_source=LABEL_HUMAN,
        none_correct_reason=REASON_NOT_ACHIEVABLE_HERE,
    )

    corpus = load_corpus(store)
    assert [r["record_id"] for r in corpus.unannotated] == ["unreviewed"]
    assert [r["record_id"] for r in corpus.none_correct] == ["reviewed"]
    assert len(corpus.scorable) == 1, (
        "an unannotated row entered the scorable set — it is not scorable and "
        "it is not a loss"
    )


def test_table_unusable_is_excluded_not_counted_wrong(tmp_path):
    """A fault upstream of the chooser must not be charged to the chooser."""
    store = _store(tmp_path)
    _capture_one(store, "broken")
    store.record_annotation(
        "broken", ANNOTATION_TABLE_UNUSABLE, annotated_by="will", label_source=LABEL_HUMAN
    )

    corpus = load_corpus(store)
    assert [r["record_id"] for r in corpus.unusable] == ["broken"]
    assert corpus.scorable == [], "an unusable table entered the scorable set"
    assert corpus.none_correct == [], (
        "unusable collapsed into none-correct — on none-correct an abstaining "
        "chooser is RIGHT, on unusable the row must be dropped entirely"
    )


def test_there_is_no_way_to_write_down_unannotated(tmp_path):
    """Absence is the only representation. A second one is how they get mixed."""
    store = _store(tmp_path)
    _capture_one(store)
    with pytest.raises(ValueError, match="NO row"):
        store.record_annotation("r1", "", annotated_by="will", label_source=LABEL_HUMAN)


def test_an_answer_outside_the_table_is_refused(tmp_path):
    """A typo that lands as a valid-looking judgment is unrecoverable later."""
    store = _store(tmp_path)
    _capture_one(store)
    with pytest.raises(ValueError, match="not a candidate"):
        store.record_annotation("r1", "click-99", annotated_by="will", label_source=LABEL_HUMAN)


def test_a_revision_supersedes_without_erasing(tmp_path):
    store = _store(tmp_path)
    _capture_one(store)
    store.record_annotation("r1", "click-0", annotated_by="will", label_source=LABEL_HUMAN)
    store.record_annotation(
        "r1", "click-1", annotated_by="will", label_source=LABEL_HUMAN, note="looked again"
    )

    corpus = load_corpus(store)
    assert corpus.answered[0]["correct_candidate_id"] == "click-1"
    with sqlite3.connect(store.db_path) as conn:
        n = conn.execute(
            "SELECT count(*) FROM annotations WHERE record_id='r1'"
        ).fetchone()[0]
    assert n == 2, "the earlier judgment was destroyed rather than superseded"


def test_the_executed_answer_is_not_the_correct_answer(tmp_path):
    """They differ exactly when the run was wrong — the case the corpus is for."""
    store = _store(tmp_path)
    loop = _loop(store, ScriptedChooser(["click-0"]))

    async def _accept(*a, **k):
        return True

    loop._approve = _accept
    asyncio.run(loop.step("press cancel", "box", "scratchapp", 1, 2))

    row = store.all_tables()[0]
    assert row["executed_candidate_id"] == "click-0"
    store.record_annotation(
        row["record_id"], "click-1", annotated_by="will", label_source=LABEL_HUMAN,
        note="the goal said cancel; it clicked send",
    )

    scored = load_corpus(store).answered[0]
    assert scored["executed_candidate_id"] == "click-0"
    assert scored["correct_candidate_id"] == "click-1"
    assert scored["executed_candidate_id"] != scored["correct_candidate_id"], (
        "the schema cannot express a wrong run, so it cannot score a chooser"
    )


# ── THE STORE ITSELF ────────────────────────────────────────────────────────


def test_capture_never_breaks_a_step(tmp_path):
    """An observer that can break the thing it observes is not worth having."""
    store = _store(tmp_path)

    def _explode(record):
        raise OSError("disk on fire")

    store.capture = _explode  # type: ignore[method-assign]
    loop = _loop(store, ScriptedChooser([CANDIDATE_ABSTAIN]))
    result = asyncio.run(loop.step("press send", "box", "scratchapp", 1, 2))
    assert result.status == "abstained", (
        "a corpus write failure changed the outcome of the step"
    )


def test_no_corpus_means_no_file_and_no_behaviour_change(tmp_path):
    """Capture is opt-in. Without it the loop is what it was."""
    loop = ComputerUseLoop(
        driver=FixtureDriver([_obs()]),
        chooser=ScriptedChooser([CANDIDATE_ABSTAIN]),
        gate=SecurityGate(mode=PermissionMode.DEFAULT, audit_logger=None),
        skip_preconditions=True,
    )
    result = asyncio.run(loop.step("press send", "box", "scratchapp", 1, 2))
    assert result.status == "abstained"
    assert not list(tmp_path.iterdir()), "a corpus file appeared without opt-in"


def test_a_newer_schema_version_is_refused_not_guessed_at(tmp_path):
    db = tmp_path / "corpus.db"
    CorpusStore(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE schema_meta SET value = ? WHERE key = 'schema_version'",
            (str(CORPUS_SCHEMA_VERSION + 1),),
        )
    with pytest.raises(RuntimeError, match="newer than this code"):
        CorpusStore(db)


def test_an_unknown_row_version_is_refused_not_skipped(tmp_path):
    """Silently dropping rows reports a smaller corpus with no sign it did."""
    store = _store(tmp_path)
    _capture_one(store)
    with sqlite3.connect(store.db_path) as conn:
        conn.execute("UPDATE tables SET schema_version = 99")
    with pytest.raises(ValueError, match="do not skip it"):
        store.all_tables()


def test_summary_names_the_unannotated_count(tmp_path):
    """The number a scoring script is most likely to forget to ask about."""
    store = _store(tmp_path)
    _capture_one(store, "a")
    _capture_one(store, "b")
    store.record_annotation("a", "click-0", annotated_by="will", label_source=LABEL_HUMAN)
    summary = load_corpus(store).summary()
    assert "1 UNANNOTATED" in summary and "50%" in summary, summary


# ── PROVENANCE ──────────────────────────────────────────────────────────────


def test_a_fixture_row_is_not_mistakable_for_a_real_one(tmp_path):
    """``tests/fixtures/divergence_traces.py`` states the rule this enforces:

        "a calibration round that cannot tell the two apart is calibrating
        against its own author."

    Every test in this file drives the loop with a ``FixtureDriver``. If those
    rows entered a corpus indistinguishably from real Cua rows, a chooser would
    be scored against tables this repo's own fixtures invented. The label is
    DERIVED from the driver, so a caller cannot get it wrong.
    """
    store = _store(tmp_path)
    loop = _loop(store, ScriptedChooser([CANDIDATE_ABSTAIN]))
    asyncio.run(loop.step("press send", "box", "scratchapp", 1, 2))

    assert store.all_tables()[0]["driver_kind"] == "fixture", (
        "a FixtureDriver row is not labelled as one — fixture and real tables "
        "are indistinguishable in the corpus"
    )


def test_goal_provenance_is_recorded_not_inferred(tmp_path):
    """Goal phrasing is the experiment, per docs/computer-use-corpus.md §2.2:
    a description-derived goal makes 93% of rows trivial. A corpus that cannot
    tell a human-written goal from a derived one cannot enforce its own spec."""
    store = _store(tmp_path)
    loop = _loop(store, ScriptedChooser([CANDIDATE_ABSTAIN]))
    asyncio.run(
        loop.step(
            "press send", "box", "scratchapp", 1, 2,
            goal_source="human", harvest_session="2026-09-20-a",
        )
    )
    row = store.all_tables()[0]
    assert row["goal_source"] == "human"
    assert row["harvest_session"] == "2026-09-20-a"


def test_an_unlabelled_goal_is_unknown_not_human(tmp_path):
    """The default must not be the one that makes a row look trustworthy."""
    store = _store(tmp_path)
    loop = _loop(store, ScriptedChooser([CANDIDATE_ABSTAIN]))
    asyncio.run(loop.step("press send", "box", "scratchapp", 1, 2))
    assert store.all_tables()[0]["goal_source"] == "unknown"


def test_the_code_fingerprint_moves_when_the_role_sets_move(tmp_path, monkeypatch):
    """The role sets DEFINE what a table is; old rows mean something else
    after they change, and scoring across the change blends two experiments."""
    from prometheus.computer import candidates as c
    from prometheus.computer.corpus import code_fingerprint

    before = code_fingerprint()
    monkeypatch.setattr(
        c, "_CLICKABLE_ROLES", frozenset({*c._CLICKABLE_ROLES, "slider"})
    )
    assert code_fingerprint() != before, (
        "the fingerprint did not change when the candidate-defining constants "
        "did — rows from before and after would look interchangeable"
    )


def test_a_table_less_row_does_not_inflate_the_unannotated_count(tmp_path):
    """A step blocked before it observed anything has no table, so no human
    judgment is possible on it. Counting it as 'awaiting review' would make
    the one number a scoring script is meant to trust systematically wrong."""
    store = _store(tmp_path)
    loop = _loop(
        store, ScriptedChooser(["click-0"]),
        observations=[_obs(unusable="no AT-SPI bus; the tree would be empty")],
    )
    result = asyncio.run(loop.step("press send", "box", "scratchapp", 1, 2))
    assert result.status == "blocked"

    corpus = load_corpus(store)
    assert corpus.unannotated == [], (
        "a row with no table is waiting on a human who can do nothing with it"
    )
    assert len(corpus.unusable) == 1
    assert corpus.scorable == []
