"""The corpus-side gate: corrupt positive labels never get banked or exported.

Companion to tests/test_markup_guard.py. That one keeps corrupt values from
EXECUTING; this one keeps them from becoming TRAINING SIGNAL. The distinction
matters because the two failures are independent: the dispatch gate stops new
ones, and 415 pairs were already banked — 10 of them with a `chosen` side made
of decoder artifacts, which is the side a fine-tune points toward.

`chosen` had no content check at all. Its bar was "the model retried and it
worked", and "it worked" means "pydantic accepted a str".

Flagged, never deleted: those 10 rows are the only surviving evidence of the
upstream leak, so they are marked unfit and skipped at export instead.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from prometheus.learning.pair_capture import (
    QUALITY_KEY,
    QUALITY_TEMPLATE_MARKUP,
    CorruptPairRejected,
    PairStore,
)

CORRUPT = '{"}}<tool_call|><|tool_response>call:[]}'
CLEAN = "Prometheus"
# One marker: legitimate, and must stay bankable (same threshold as dispatch).
ONE_MARKER = "grep for <|tool_call|> in the repo"


def _pair(chosen_value, rejected_value=None, **kw):
    return dict(
        pair_source=kw.get("pair_source", "self_correction"),
        model_id="gemma4-26b",
        tool_name="wiki_compile",
        context={"kind": "test"},
        rejected={"name": "wiki_compile", "input": {"entity_name": rejected_value}}
        if rejected_value is not None else None,
        chosen={"name": "wiki_compile", "input": {"entity_name": chosen_value}},
    )


@pytest.fixture
def store(tmp_path):
    return PairStore(tmp_path / "training.db")


class TestAddPairGate:
    def test_corrupt_chosen_is_refused(self, store):
        with pytest.raises(CorruptPairRejected) as exc:
            store.add_pair(**_pair(CORRUPT, rejected_value={}))
        assert "wiki_compile" in str(exc.value)
        assert "markup" in str(exc.value)

    def test_refusal_leaves_the_store_empty(self, store):
        with pytest.raises(CorruptPairRejected):
            store.add_pair(**_pair(CORRUPT, rejected_value={}))
        assert store.rows_since() == []

    def test_clean_chosen_is_banked(self, store):
        assert store.add_pair(**_pair(CLEAN, rejected_value={})) is True
        assert len(store.rows_since()) == 1

    def test_single_marker_chosen_is_still_banked(self, store):
        """Same measured threshold as the dispatch gate — do not diverge.

        If this store were stricter than dispatch, a call that legitimately
        executed could never become training data, which is a silent and
        confusing hole.
        """
        assert store.add_pair(**_pair(ONE_MARKER, rejected_value={})) is True

    def test_corrupt_REJECTED_side_is_fine(self, store):
        """The negative label is SUPPOSED to hold the bad value.

        This is the shape the dispatch gate now produces: corrupt attempt on
        the rejected side, clean retry on the chosen side. Refusing it would
        throw away the most useful pair in the corpus.
        """
        assert store.add_pair(**_pair(CLEAN, rejected_value=CORRUPT)) is True

    def test_gate_covers_every_pair_source(self, store):
        for src in ("levenshtein_repair", "schema_repair", "retry_success",
                    "malformed_recovery", "gym_harvest"):
            with pytest.raises(CorruptPairRejected):
                store.add_pair(**_pair(CORRUPT, rejected_value={}, pair_source=src))

    def test_nested_corrupt_argument_is_caught(self, store):
        with pytest.raises(CorruptPairRejected):
            store.add_pair(
                pair_source="schema_repair", model_id="m", tool_name="t",
                context={}, rejected={"name": "t", "input": {}},
                chosen={"name": "t", "input": {"outer": {"inner": CORRUPT}}},
            )


class TestCapturePairRecordsRejection:
    """capture_pair must count the refusal, not swallow it."""

    def test_silent_failure_row_is_written(self, tmp_path, monkeypatch):
        import prometheus.learning.pair_capture as pc

        recorded: list[dict] = []

        class _Tel:
            def record_silent_failure(self, **kw):
                recorded.append(kw)

        monkeypatch.setattr(pc, "_store", PairStore(tmp_path / "t.db"))
        monkeypatch.setattr(pc, "_source_override", None)
        pc.capture_pair(**_pair(CORRUPT, rejected_value={}), telemetry=_Tel())

        assert len(recorded) == 1, "the rejection rate must be measurable"
        assert recorded[0]["subsystem"] == "pair_capture"
        assert isinstance(recorded[0]["exc"], CorruptPairRejected)

    def test_capture_pair_never_raises_into_the_turn(self, tmp_path, monkeypatch):
        import prometheus.learning.pair_capture as pc

        monkeypatch.setattr(pc, "_store", PairStore(tmp_path / "t.db"))
        monkeypatch.setattr(pc, "_source_override", None)
        pc.capture_pair(**_pair(CORRUPT, rejected_value={}), telemetry=None)  # no raise


class TestExportExcludesUnfit:
    """Export is the last gate before a fine-tune sees anything."""

    def _run_export(self, tmp_path, db_path, extra=()):
        import subprocess, sys as _s
        from pathlib import Path as _P
        out = tmp_path / "pairs.jsonl"
        root = _P(__file__).resolve().parent.parent
        r = subprocess.run(
            [_s.executable, str(root / "scripts" / "export_training_pairs.py"),
             "--db", str(db_path), "--out", str(out), *extra],
            capture_output=True, text=True,
            env={"PYTHONPATH": str(root / "src"), "PATH": "/usr/bin:/bin"},
        )
        assert r.returncode == 0, r.stderr
        lines = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
        return lines, r.stdout

    def _seed(self, db_path, chosen_value, meta):
        """Insert a row directly — bypassing add_pair, as history did."""
        conn = sqlite3.connect(str(db_path))
        conn.executescript(
            "CREATE TABLE IF NOT EXISTS training_pairs ("
            "id TEXT PRIMARY KEY, timestamp REAL NOT NULL, pair_source TEXT NOT NULL,"
            " model_id TEXT NOT NULL, tool_name TEXT NOT NULL, context TEXT,"
            " rejected TEXT, chosen TEXT NOT NULL, meta TEXT,"
            " context_hash TEXT NOT NULL UNIQUE)"
        )
        conn.execute(
            "INSERT INTO training_pairs VALUES (?,?,?,?,?,?,?,?,?,?)",
            (chosen_value[:20] + str(len(meta)), 1.0, "self_correction", "m", "t",
             "{}", json.dumps({"name": "t", "input": {"a": "bad"}}),
             json.dumps({"name": "t", "input": {"a": chosen_value}}),
             json.dumps(meta), chosen_value[:16] + str(len(meta))),
        )
        conn.commit()
        conn.close()

    def test_flagged_row_is_skipped(self, tmp_path):
        db = tmp_path / "t.db"
        self._seed(db, CLEAN, {QUALITY_KEY: QUALITY_TEMPLATE_MARKUP})
        lines, stdout = self._run_export(tmp_path, db)
        assert lines == []
        assert "unfit to train on" in stdout

    def test_unflagged_but_corrupt_row_is_still_skipped(self, tmp_path):
        """The property, not the reminder.

        A DB the flagging script never ran against must not leak corrupt
        positives into a fine-tune just because nobody remembered.
        """
        db = tmp_path / "t.db"
        self._seed(db, CORRUPT, {})
        lines, _ = self._run_export(tmp_path, db)
        assert lines == []

    def test_clean_row_is_exported(self, tmp_path):
        db = tmp_path / "t.db"
        self._seed(db, CLEAN, {})
        lines, _ = self._run_export(tmp_path, db)
        assert len(lines) == 1

    def test_include_flagged_emits_for_inspection(self, tmp_path):
        db = tmp_path / "t.db"
        self._seed(db, CORRUPT, {QUALITY_KEY: QUALITY_TEMPLATE_MARKUP})
        lines, _ = self._run_export(tmp_path, db, extra=["--include-flagged"])
        assert len(lines) == 1, "evidence must remain reachable — flagged, not deleted"
