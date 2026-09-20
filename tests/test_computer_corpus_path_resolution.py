"""Corpus database path resolution — real files, no mocks.

WHY THIS FILE EXISTS
--------------------
``paths.py`` carries two long post-mortems of what a re-derived path cost this
project, and ``tests/test_lcm_db_path_resolution.py`` exists because four sites
named ``lcm.db`` and did not agree — so "the shared lcm.db" was silently two
files, and a reader querying the wrong one found an empty table, which is
indistinguishable from "nothing recorded yet".

A corpus has that failure mode in a worse form. A harvest writing to one file
while the scoring harness reads another produces **an empty corpus, which reads
exactly like an unannotated one** — and the honest response to an unannotated
corpus is "go and annotate it", so the defect would survive being noticed.

Mocked tests structurally cannot catch that class: asserting
``store.db_path == expected`` passes happily while the process writes somewhere
else. So every test here constructs the real store, writes a real row, and reads
it back **through a separate connection**.

Two locations are made observable:

  data_db   — ``<config>/data/computer_corpus.db``. Everything must land here.
  config_db — ``<config>/computer_corpus.db``, where a caller that reached for
              ``get_config_dir()`` would land. Must stay absent — that is where
              ``telemetry.db`` lives, and it is NOT where a dataset goes.
"""

from __future__ import annotations

import sqlite3

import pytest

from prometheus.computer.corpus import CorpusStore, TableRecord
from prometheus.config.paths import get_computer_corpus_db_path


@pytest.fixture
def roots(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(config_dir))
    monkeypatch.delenv("PROMETHEUS_DATA_DIR", raising=False)
    return {
        "config": config_dir,
        "data_db": config_dir / "data" / "computer_corpus.db",
        "config_db": config_dir / "computer_corpus.db",
    }


def _capture_one(store: CorpusStore) -> str:
    rec = TableRecord(goal="g", target="box", app="scratchapp", window_id=1)

    class C:
        candidate_id = "click-0"
        description = "Click the push button Send"

    rec.note_candidates([C()])
    assert store.capture(rec), "the capture did not land"
    return rec.record_id


def test_the_resolver_returns_the_data_dir_file(roots):
    assert get_computer_corpus_db_path() == roots["data_db"]


def test_a_real_row_lands_in_the_data_dir_file_and_nowhere_else(roots):
    """The assertion that a mocked test cannot make."""
    store = CorpusStore(get_computer_corpus_db_path())
    record_id = _capture_one(store)

    assert roots["data_db"].exists(), "nothing was written to the data-dir file"
    assert not roots["config_db"].exists(), (
        "a corpus file appeared in the config root beside telemetry.db — a "
        "caller re-derived the path from get_config_dir()"
    )

    # Read back through a SEPARATE connection: proves the bytes are in that
    # file, not merely that the object thinks they are.
    with sqlite3.connect(roots["data_db"]) as conn:
        row = conn.execute(
            "SELECT app FROM tables WHERE record_id = ?", (record_id,)
        ).fetchone()
    assert row is not None, "the row is not in the file the resolver names"
    assert row[0] == "scratchapp"


def test_the_data_dir_env_var_moves_the_database(tmp_path, monkeypatch):
    config_dir = tmp_path / "c"
    data_dir = tmp_path / "elsewhere"
    config_dir.mkdir()
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("PROMETHEUS_DATA_DIR", str(data_dir))

    store = CorpusStore(get_computer_corpus_db_path())
    _capture_one(store)

    assert (data_dir / "computer_corpus.db").exists()
    assert not (config_dir / "data" / "computer_corpus.db").exists(), (
        "PROMETHEUS_DATA_DIR was ignored — the default was re-derived"
    )


def test_an_explicit_path_still_wins(tmp_path, monkeypatch):
    """Tests, the gym and throwaway harvests pass a path. That must keep working."""
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path / "c"))
    explicit = tmp_path / "throwaway" / "corpus.db"
    store = CorpusStore(explicit)
    _capture_one(store)

    assert explicit.exists()
    assert not get_computer_corpus_db_path().exists(), (
        "an explicit db_path was overridden by the default"
    )


def test_the_name_is_not_spelled_anywhere_else(roots):
    """One resolution point, enforced.

    ``get_computer_corpus_db_path`` is the only place the filename appears. A
    second spelling is how ``lcm.db`` became two files, and the cost there was
    months of checkpoints written to a file nothing read.

    Checked over the AST rather than by grep, and on string literals whose
    value IS the filename rather than lines that merely contain it. A textual
    guard fires on its own explanatory prose — including the paragraph above —
    and a guard that cannot survive being described is one that gets deleted.
    """
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "src" / "prometheus"
    offenders = []
    for py in src.rglob("*.py"):
        if py.match("config/paths.py"):
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - not our problem here
            continue
        for node in ast.walk(tree):
            # `value ==` and not `in`: a docstring node's value is the whole
            # docstring, so prose naming the file never matches.
            if isinstance(node, ast.Constant) and node.value == "computer_corpus.db":
                offenders.append(f"{py}:{node.lineno}")

    assert not offenders, (
        "computer_corpus.db is constructed outside config/paths.py:\n  "
        + "\n  ".join(offenders)
    )
