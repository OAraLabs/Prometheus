"""Pair-harvest gym mode (tool-calling closeout follow-up #6).

Proves the wiring that makes a gym run accumulate training pairs: configuring
the harvest points the pair-capture singleton at a SEPARATE gym DB, captures
land there (not in the live training.db), and pair_total() reflects them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prometheus.gym.harvest import configure_harvest, default_harvest_db, pair_total
from prometheus.learning import pair_capture


@pytest.fixture(autouse=True)
def _reset_capture():
    # Leave the module singleton off after each test (it is process-global).
    yield
    pair_capture.configure({"capture_enabled": False})


def _capture_one(tool: str = "task_create") -> None:
    pair_capture.capture_pair(
        pair_source="schema_repair",
        model_id="gemma-test",
        tool_name=tool,
        context={"prompt": "do the thing"},
        rejected={"name": tool, "input": {"args": {"x": 1}}},  # phantom-nested
        chosen={"name": tool, "input": {"x": 1}},               # unwrapped
    )


def test_configure_points_store_at_gym_db(tmp_path: Path):
    db = tmp_path / "gym-training.db"
    returned = configure_harvest(str(db))
    assert returned == str(db)
    assert pair_capture.get_store() is not None
    # The store writes to exactly the gym path we gave it.
    _capture_one()
    assert db.exists()


def test_default_harvest_db_is_sibling_of_training_db():
    # ~/.prometheus/data/gym-training.db — beside the live training.db, not it.
    p = default_harvest_db()
    assert p.endswith("gym-training.db")
    assert "training.db" in p and "gym-training.db" in p


def test_capture_lands_in_gym_db_and_counts(tmp_path: Path):
    db = tmp_path / "gym-training.db"
    configure_harvest(str(db))
    assert pair_total() == 0
    _capture_one()
    assert pair_total() == 1
    # A genuinely different pair increments…
    _capture_one(tool="task_list")
    assert pair_total() == 2
    # …and the rows are in the GYM db file, which now exists on disk.
    assert db.exists()


def test_harvest_isolated_from_live_training_db(tmp_path: Path, monkeypatch):
    # Capturing under the harvest config must NOT touch the default training.db.
    live = tmp_path / "data" / "training.db"
    monkeypatch.setattr(
        "prometheus.gym.harvest.get_data_dir", lambda: tmp_path / "data"
    )
    (tmp_path / "data").mkdir()
    configure_harvest(str(tmp_path / "gym-training.db"))
    _capture_one()
    assert pair_total() == 1
    assert not live.exists()  # the live DB was never created/written


def test_pair_total_zero_when_capture_off():
    pair_capture.configure({"capture_enabled": False})
    assert pair_total() == 0
