"""#474 — backfill_billing_mode's two reporting defects.

1. It printed the resolved provider host — a real infrastructure identifier —
   to stdout. Stdout is persisted state: terminal scrollback, shell history,
   and any transcript or ticket the output is pasted into. The script exists
   precisely because the host must not be persisted; the leak is also upstream
   in ``billing_for``'s reason string, which /api/usage returns to clients as
   ``billing_reason``. Both are pinned to name the MATCHED MARKER (a committed
   codebase constant) instead of the host.

2. "rows to stamp: 0" meant three different things — already done, rows exist
   but outside the window, or the model matched nothing (a typo'd --model). A
   no-op rendered identically to a completed backfill (§4e's shape in a
   reporting path). Each zero must now name itself.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "backfill_billing_mode.py"

# A host with an account-scoped remainder that must NEVER reach stdout.
SECRETISH_HOST = "token-plan.ACCOUNT-SECRET-123.some-region.example.com"
PLAN_URL = f"//{SECRETISH_HOST}/v1"


@pytest.fixture
def db(tmp_path: Path):
    def _make(name: str, rows: list[tuple]) -> Path:
        path = tmp_path / name
        conn = sqlite3.connect(path)
        conn.executescript(
            """
            CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE subsystem_runs (
                id INTEGER PRIMARY KEY, model TEXT, timestamp REAL,
                billing_mode TEXT, input_tokens INTEGER, output_tokens INTEGER);
            INSERT INTO schema_meta VALUES ('billing_recorded_since','1000.0');
            """
        )
        for r in rows:
            conn.execute(
                "INSERT INTO subsystem_runs (model,timestamp,billing_mode,input_tokens)"
                " VALUES (?,?,?,?)", r,
            )
        conn.commit()
        conn.close()
        return path

    return _make


def _run(db_path: Path, model: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--db", str(db_path), "--model", model],
        capture_output=True, text=True, cwd=str(REPO), timeout=60,
        env={"PYTHONPATH": str(REPO / "src"),
             "QWEN_BASE_URL": PLAN_URL, "PATH": "/usr/bin:/bin"},
    )


# ── defect 1: the host must not reach stdout ────────────────────────────────

def test_the_plan_hostname_never_reaches_stdout(db):
    """The account-scoped remainder is the identifier; the marker is not."""
    proc = _run(db("done.db", [("qwen3.8-max", 500.0, "subscription", 10)]),
                "qwen3.8-max")
    assert "ACCOUNT-SECRET-123" not in proc.stdout, proc.stdout
    assert SECRETISH_HOST not in proc.stdout, proc.stdout
    # But the output still SAYS what it classified from — the marker is the
    # part that carries the meaning.
    assert "token-plan." in proc.stdout, proc.stdout


def test_a_non_matching_host_is_redacted_too(db):
    """A host that matches no subscription marker is still an address."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--db",
         str(db("nm.db", [])), "--model", "qwen3.8-max",
         "--host", "metered.host.example.com"],
        capture_output=True, text=True, cwd=str(REPO), timeout=60,
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert "metered.host.example.com" not in proc.stdout, proc.stdout
    assert "matches no subscription marker" in proc.stdout, proc.stdout


def test_billing_for_reason_does_not_embed_the_host():
    """The upstream leak: /api/usage returns this reason to clients verbatim."""
    from prometheus.telemetry.cost import billing_for

    mode, reason = billing_for("qwen3.8-max", PLAN_URL)
    assert mode == "subscription"
    assert SECRETISH_HOST not in reason, reason
    assert "ACCOUNT-SECRET-123" not in reason, reason
    assert "token-plan." in reason, (
        "the reason must still say WHICH marker matched — redaction is not "
        f"deletion: {reason!r}"
    )


# ── defect 2: three zeros, three reports ─────────────────────────────────────

def test_zero_when_already_stamped_says_already_done(db):
    proc = _run(db("done.db", [("qwen3.8-max", 500.0, "subscription", 10)]),
                "qwen3.8-max")
    assert proc.returncode == 0
    assert "already done" in proc.stdout, proc.stdout
    assert "1 row(s)" in proc.stdout, proc.stdout


def test_zero_when_rows_are_outside_the_window_says_so(db):
    """Model rows exist but all post-date the boundary — NOT 'already done'."""
    proc = _run(db("outside.db", [("qwen3.8-max", 2000.0, None, 10)]),
                "qwen3.8-max")
    assert proc.returncode == 0
    assert "outside the boundary" in proc.stdout, proc.stdout
    assert "NOT 'already backfilled'" in proc.stdout, proc.stdout


def test_zero_when_the_model_matched_nothing_warns_about_the_typo(db):
    proc = _run(db("nomatch.db", [("other-model", 500.0, None, 10)]),
                "qwen3.8-max")
    assert proc.returncode == 0
    assert "matched nothing" in proc.stdout, proc.stdout
    assert "NO rows for model" in proc.stdout, proc.stdout
    assert "--model spelling" in proc.stdout, proc.stdout


def test_the_three_zeros_are_pairwise_distinguishable(db):
    """The point of the fix: byte-identical output for different states is the
    defect. Each zero must name itself differently."""
    outs = [
        _run(db("a.db", [("qwen3.8-max", 500.0, "subscription", 10)]),
             "qwen3.8-max").stdout,
        _run(db("b.db", [("qwen3.8-max", 2000.0, None, 10)]),
             "qwen3.8-max").stdout,
        _run(db("c.db", [("other-model", 500.0, None, 10)]),
             "qwen3.8-max").stdout,
    ]
    # The verdict lines (everything after the boundary header) must differ.
    def verdict(out: str) -> str:
        return "\n".join(
            line for line in out.splitlines()
            if any(k in line for k in
                   ("already done", "nothing to do", "matched nothing"))
        )

    verdicts = [verdict(o) for o in outs]
    assert all(verdicts), f"a zero-case printed no verdict: {outs}"
    assert len(set(verdicts)) == 3, (
        f"two zero-cases rendered identically — the ambiguity #474 exists to "
        f"remove:\n{verdicts}"
    )
