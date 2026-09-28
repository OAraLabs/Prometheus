"""`oara scrub`: the capture-store scrub ships with the package.

It was scripts/scrub_capture_stores.py, and scripts/ ships in neither the
wheel nor the sdist, so a pip or Homebrew install had no way to clean the
secrets kept before capture-time redaction existed. The scrub itself is
covered by tests/test_capture_redaction.py, tests/test_secret_redaction_x37.py
and tests/test_memory_db_redaction.py (through the scripts/ path, which is now
a shim over the same module). These tests pin the packaged entry point.

In-process only: every store is named on the command line, so no default path
can resolve to the real ~/.prometheus, and no child process is started.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

import prometheus

FAKE_TOKEN = "123456:AAF-FakeTokenForTestsOnly_0123456789x"


def _telemetry_with_a_token(tmp_path: Path) -> Path:
    db = tmp_path / "telemetry.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE tool_calls (id INTEGER PRIMARY KEY, error_detail TEXT, "
                 "raw_model_output TEXT, parsed_tool_call TEXT)")
    conn.execute("INSERT INTO tool_calls (raw_model_output) VALUES (?)",
                 (f'curl -s "https://api.telegram.org/bot{FAKE_TOKEN}/getMe"',))
    conn.commit()
    conn.close()
    return db


def _oara(monkeypatch, *argv: str) -> int:
    from prometheus.__main__ import main

    monkeypatch.setattr(sys, "argv", ["oara", *argv])
    with pytest.raises(SystemExit) as exc:
        main()
    return exc.value.code


def _every_store(tmp_path: Path, telemetry: Path) -> list[str]:
    missing = tmp_path / "absent"
    return ["--telemetry", str(telemetry), "--training", str(missing / "training.db"),
            "--trajectories", str(missing / "trajectories"), "--lcm", str(missing / "lcm.db"),
            "--memory", str(missing / "memory.db")]


def test_the_scrub_is_a_module_of_the_package():
    from prometheus.security import scrub_capture_stores as scrub

    # Inside the package directory, so `packages = ["src/prometheus"]` puts it
    # in the wheel (and the sdist's /src carries it for Homebrew).
    assert Path(scrub.__file__).resolve().parent == Path(prometheus.__file__).resolve().parent / "security"
    assert callable(scrub.main) and callable(scrub.run) and callable(scrub.add_arguments)


def test_oara_scrub_is_a_dry_run_by_default(tmp_path, monkeypatch, capsys):
    tel = _telemetry_with_a_token(tmp_path)
    before = tel.read_bytes()

    assert _oara(monkeypatch, "scrub", *_every_store(tmp_path, tel)) == 0

    out = capsys.readouterr().out
    assert "=== capture-store scrub — DRY RUN ===" in out
    assert "tool_calls.raw_model_output" in out and "1 would change" in out
    assert FAKE_TOKEN not in out  # counts only, never a matched value
    assert tel.read_bytes() == before
    assert not list(tmp_path.glob("*.pre-scrub-*"))


def test_oara_scrub_apply_rewrites_after_a_backup(tmp_path, monkeypatch, capsys):
    tel = _telemetry_with_a_token(tmp_path)

    assert _oara(monkeypatch, "scrub", "--apply", *_every_store(tmp_path, tel)) == 0

    out = capsys.readouterr().out
    assert "=== capture-store scrub — APPLY ===" in out and "1 rewritten" in out
    conn = sqlite3.connect(tel)
    (kept,) = conn.execute("SELECT raw_model_output FROM tool_calls").fetchone()
    conn.close()
    assert FAKE_TOKEN not in kept
    backups = list(tmp_path.glob("telemetry.db.pre-scrub-*"))
    assert len(backups) == 1  # the old values are in the backup, as documented


def test_oara_scrub_fails_loudly_on_a_store_it_cannot_open(tmp_path, monkeypatch, capsys):
    not_a_db = tmp_path / "telemetry.db"
    not_a_db.write_text("not sqlite", encoding="utf-8")

    assert _oara(monkeypatch, "scrub", *_every_store(tmp_path, not_a_db)) == 1

    assert "FAILED to scrub telemetry.db" in capsys.readouterr().out
