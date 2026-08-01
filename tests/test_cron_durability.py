"""Cron registry durability (audit H7): atomic writes + corrupt-file preservation.

The old failure mode: a plain ``write_text`` interrupted mid-write left an
unparseable file, and ``load_cron_jobs`` silently returned ``[]`` — which the
next save overwrote, erasing every operator job with no trace. These tests pin
the fix: writes are atomic, and a corrupt file is preserved + logged, never
silently nuked.
"""

from __future__ import annotations

import json

import pytest

from prometheus.gateway import cron_service


@pytest.fixture
def registry(tmp_path, monkeypatch):
    path = tmp_path / "cron_jobs.json"
    monkeypatch.setattr(cron_service, "get_cron_registry_path", lambda: path)
    return path


def test_save_round_trips_and_leaves_no_tmp(registry):
    jobs = [{"name": "nightly", "schedule": "0 3 * * *", "command": "backup"}]
    cron_service.save_cron_jobs(jobs)
    assert cron_service.load_cron_jobs() == jobs
    # the atomic temp file must not linger
    assert not registry.with_suffix(registry.suffix + ".tmp").exists()


def test_overwrite_is_clean(registry):
    cron_service.save_cron_jobs([{"name": "a"}])
    cron_service.save_cron_jobs([{"name": "b"}, {"name": "c"}])
    assert cron_service.load_cron_jobs() == [{"name": "b"}, {"name": "c"}]


def test_corrupt_file_is_preserved_not_erased(registry):
    registry.write_text("{ this is not valid json", encoding="utf-8")
    backup = registry.with_suffix(registry.suffix + ".corrupt")

    result = cron_service.load_cron_jobs()  # must not raise

    assert result == []                       # starts empty
    assert backup.exists()                    # but the bad data is preserved
    assert "not valid json" in backup.read_text()
    assert not registry.exists()              # moved aside, not left to be overwritten


def test_save_after_corrupt_does_not_lose_the_backup(registry):
    registry.write_text("garbage", encoding="utf-8")
    cron_service.load_cron_jobs()             # triggers preservation
    backup = registry.with_suffix(registry.suffix + ".corrupt")

    cron_service.save_cron_jobs([{"name": "recovered"}])

    assert cron_service.load_cron_jobs() == [{"name": "recovered"}]
    assert backup.exists() and "garbage" in backup.read_text()


def test_load_missing_file_is_empty(registry):
    assert cron_service.load_cron_jobs() == []


def test_save_writes_valid_json(registry):
    cron_service.save_cron_jobs([{"name": "x", "n": 1}])
    assert json.loads(registry.read_text()) == [{"name": "x", "n": 1}]
