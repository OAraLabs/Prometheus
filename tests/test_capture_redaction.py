"""Secrets must not survive into the capture stores — at write time, and in what is already there.

Log redaction (test_log_redaction.py) keeps a token out of what the process EMITS.
These tests pin the other half: what the process KEEPS. The 2026-08-31 finding was
a Telegram bot token sitting in telemetry.db (tool_calls), training.db
(training_pairs) and a trajectories/ export — fine-tune capture, the data that
gets copied around. Four chokepoints are redacted before the row exists, and a
script scrubs rows written before the chokepoints did.

Fixtures sit below .githooks/pre-commit's length floors on purpose (see the note
in test_log_redaction.py). Do not make them look more real.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from prometheus.security import REDACTED, redact_capture
from prometheus.telemetry.tracker import ToolCallTelemetry

FAKE_TOKEN = "123456:AAF-FakeTokenForTestsOnly_0123456789x"
CMD = f'TOK="bot{FAKE_TOKEN}"; curl -s "https://api.telegram.org/$TOK/getMe"'
REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "scrub_capture_stores.py"


# ── the recursive helper ──────────────────────────────────────────────────────────────────


def test_redact_capture_recurses_and_leaves_keys_and_non_strings_alone():
    obj = {"name": "bash", "input": {"command": CMD, "n": 3, "flags": ["-s", f"key=sk-fakekey-0123456789ab"]}, "ok": True, "t": (1, f"bearer fake.bearer.token.0123456789")}
    out = redact_capture(obj)
    dumped = json.dumps(out, default=str)
    assert FAKE_TOKEN not in dumped and "sk-fakekey-0123456789ab" not in dumped and "fake.bearer.token.0123456789" not in dumped
    assert out["input"]["n"] == 3 and out["ok"] is True and out["name"] == "bash"
    assert isinstance(out["t"], tuple) and out["t"][0] == 1
    assert set(out.keys()) == set(obj.keys())


def test_redact_capture_is_identity_on_clean_data():
    obj = {"name": "grep", "input": {"pattern": "TODO", "path": "src"}, "latency": 12.5}
    assert redact_capture(obj) == obj
    assert redact_capture("plain") == "plain" and redact_capture(None) is None and redact_capture(7) == 7


# ── chokepoint 1: telemetry.record ────────────────────────────────────────────────────────


def test_tool_call_rows_are_redacted_at_record_time(tmp_path):
    tel = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
    tel.record(
        model="local/qwen", tool_name="bash", success=True, provider="llama_cpp",
        raw_model_output=f"I will run {CMD}",
        parsed_tool_call=json.dumps({"name": "bash", "input": {"command": CMD}}),
        error_detail=f"401 for https://api.telegram.org/bot{FAKE_TOKEN}/getMe",
    )
    conn = sqlite3.connect(tmp_path / "telemetry.db")
    raw, parsed, err = conn.execute("SELECT raw_model_output, parsed_tool_call, error_detail FROM tool_calls").fetchone()
    for col in (raw, parsed, err):
        assert FAKE_TOKEN not in col and REDACTED in col
    # the row is still a usable capture: valid JSON, the command shape intact
    assert json.loads(parsed)["input"]["command"].startswith('TOK="bot<redacted>"')


def test_silent_failure_rows_are_redacted(tmp_path):
    tel = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
    try:
        raise RuntimeError(f"Client error for url https://api.telegram.org/bot{FAKE_TOKEN}/getMe")
    except RuntimeError as exc:
        tel.record_silent_failure("gateway", "poll", exc, context={"cmd": CMD})
    conn = sqlite3.connect(tmp_path / "telemetry.db")
    row = conn.execute("SELECT exception_msg, traceback, context FROM silent_failures").fetchone()
    assert row is not None
    for col in row:
        assert col is None or FAKE_TOKEN not in col


# ── chokepoint 2: the golden-trace export (rows + resolver context) ───────────────────────


def test_golden_export_redacts_the_resolver_context_too(tmp_path):
    tel = ToolCallTelemetry(db_path=tmp_path / "telemetry.db")
    tel.record(
        model="claude-sonnet", tool_name="bash", success=True, provider="anthropic",
        raw_model_output="Running the check", parsed_tool_call=json.dumps({"name": "bash", "input": {"command": "ls"}}),
    )

    def resolver(_trace):
        # The LCM conversation store is NOT redacted at capture — the export must be.
        return [{"role": "user", "content": f"here is my token bot{FAKE_TOKEN} please check it"}]

    path = tel.export_golden_traces(output_dir=tmp_path, context_resolver=resolver)
    text = path.read_text(encoding="utf-8")
    assert text.strip(), "export wrote nothing"
    assert FAKE_TOKEN not in text and REDACTED in text
    for line in text.splitlines():
        json.loads(line)  # still one valid JSON object per line


# ── chokepoint 3: PairStore.add_pair ──────────────────────────────────────────────────────


def test_training_pairs_are_redacted_on_every_side(tmp_path):
    from prometheus.learning.pair_capture import PairStore
    store = PairStore(tmp_path / "t.db")
    ok = store.add_pair(
        pair_source="self_correction", model_id="local/qwen", tool_name="bash",
        context={"messages": [{"role": "user", "content": f"run bot{FAKE_TOKEN} check"}]},
        rejected={"name": "bash", "input": {"command": CMD + " --bad"}},
        chosen={"name": "bash", "input": {"command": CMD}},
        meta={"error": f"401 https://api.telegram.org/bot{FAKE_TOKEN}/getMe"},
    )
    assert ok
    conn = sqlite3.connect(tmp_path / "t.db")
    ctx, rej, cho, meta = conn.execute("SELECT context, rejected, chosen, meta FROM training_pairs").fetchone()
    for col in (ctx, rej, cho, meta):
        assert FAKE_TOKEN not in col
    assert json.loads(cho)["input"]["command"].startswith('TOK="bot<redacted>"')


def test_training_pair_dedupe_hash_is_over_the_redacted_content(tmp_path):
    """Two captures differing only in the secret are the same pair once redacted."""
    from prometheus.learning.pair_capture import PairStore
    store = PairStore(tmp_path / "t.db")
    base = dict(pair_source="self_correction", model_id="m", tool_name="bash", context={"k": "v"}, chosen={"name": "bash", "input": {"command": "ls"}})
    assert store.add_pair(rejected={"name": "bash", "input": {"command": f"bot{FAKE_TOKEN}"}}, **base)
    assert not store.add_pair(rejected={"name": "bash", "input": {"command": "bot123456:AAF-OtherFakeToken_0123456789012345"}}, **base)


# ── the scrub for rows written before the chokepoints existed ─────────────────────────────


def _seed_old_stores(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Rows/lines exactly as an unpatched daemon would have written them."""
    tel_db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db_path=tel_db)
    tel.record(model="m", tool_name="bash", success=True)  # a clean row too
    conn = sqlite3.connect(tel_db)
    conn.execute("UPDATE tool_calls SET raw_model_output = ?, parsed_tool_call = ?", (f"run {CMD}", json.dumps({"name": "bash", "input": {"command": CMD}})))
    conn.execute("INSERT INTO silent_failures (id, timestamp, subsystem, operation, exception_type, exception_msg, traceback, context, response_body) VALUES ('s1', 0, 'g', 'o', 'E', ?, 'tb', NULL, NULL)", (f"url bot{FAKE_TOKEN}",))
    conn.commit(); conn.close()
    from prometheus.learning.pair_capture import _SCHEMA
    tr_db = tmp_path / "training.db"
    conn = sqlite3.connect(tr_db); conn.executescript(_SCHEMA)
    conn.execute("INSERT INTO training_pairs (id, timestamp, pair_source, model_id, tool_name, context, rejected, chosen, meta, context_hash) VALUES ('p1', 0, 'self_correction', 'm', 'bash', '{}', NULL, ?, '{}', 'h1')", (json.dumps({"name": "bash", "input": {"command": CMD}}),))
    conn.execute("INSERT INTO training_pairs (id, timestamp, pair_source, model_id, tool_name, context, rejected, chosen, meta, context_hash) VALUES ('p2', 0, 'self_correction', 'm', 'grep', '{}', NULL, '{\"name\":\"grep\",\"input\":{}}', '{}', 'h2')")
    conn.commit(); conn.close()
    traj = tmp_path / "trajectories"; traj.mkdir()
    (traj / "golden_traces_1.jsonl").write_text(json.dumps({"messages": [{"role": "user", "content": CMD}]}) + "\n" + json.dumps({"messages": [{"role": "user", "content": "clean"}]}) + "\n", encoding="utf-8")
    return tel_db, tr_db, traj


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=120)


def test_scrub_dry_run_counts_and_touches_nothing(tmp_path):
    tel_db, tr_db, traj = _seed_old_stores(tmp_path)
    before = (tel_db.read_bytes(), tr_db.read_bytes(), (traj / "golden_traces_1.jsonl").read_text())
    r = _run(["--telemetry", str(tel_db), "--training", str(tr_db), "--trajectories", str(traj)])
    assert r.returncode == 0, r.stderr
    assert "DRY RUN" in r.stdout
    assert "tool_calls.raw_model_output                     1 would change" in r.stdout
    assert "tool_calls.parsed_tool_call                     1 would change" in r.stdout
    assert "silent_failures.exception_msg                   1 would change" in r.stdout
    assert "training_pairs.chosen                           1 would change" in r.stdout
    assert "golden_traces_1.jsonl                           1 would change" in r.stdout
    assert "5 row(s)/line(s) would change" in r.stdout
    assert FAKE_TOKEN not in r.stdout, "the scrub must never print a matched value"
    assert (tel_db.read_bytes(), tr_db.read_bytes(), (traj / "golden_traces_1.jsonl").read_text()) == before
    assert not list(tmp_path.glob("*.pre-scrub-*")), "dry run must not create backups"


def test_scrub_apply_rewrites_backs_up_and_is_idempotent(tmp_path):
    tel_db, tr_db, traj = _seed_old_stores(tmp_path)
    r = _run(["--apply", "--telemetry", str(tel_db), "--training", str(tr_db), "--trajectories", str(traj)])
    assert r.returncode == 0, r.stderr
    assert "5 row(s)/line(s) rewritten" in r.stdout
    backups = sorted(tmp_path.glob("*.pre-scrub-*"))
    assert len(backups) == 2, backups
    # the backup is a faithful copy of the UNSCRUBBED database
    assert FAKE_TOKEN in sqlite3.connect(backups[0]).execute("SELECT raw_model_output FROM tool_calls").fetchone()[0] or \
        FAKE_TOKEN in (sqlite3.connect(backups[1]).execute("SELECT chosen FROM training_pairs WHERE id='p1'").fetchone()[0])
    # the live stores are clean and still well-formed
    conn = sqlite3.connect(tel_db)
    raw, parsed = conn.execute("SELECT raw_model_output, parsed_tool_call FROM tool_calls").fetchone()
    assert FAKE_TOKEN not in raw and REDACTED in raw and json.loads(parsed)["name"] == "bash"
    assert FAKE_TOKEN not in conn.execute("SELECT exception_msg FROM silent_failures").fetchone()[0]
    cho = sqlite3.connect(tr_db).execute("SELECT chosen FROM training_pairs WHERE id='p1'").fetchone()[0]
    assert FAKE_TOKEN not in cho and json.loads(cho)["input"]["command"].startswith('TOK="bot<redacted>"')
    lines = (traj / "golden_traces_1.jsonl").read_text().splitlines()
    assert len(lines) == 2 and FAKE_TOKEN not in lines[0] and json.loads(lines[1])["messages"][0]["content"] == "clean"
    # second run: nothing left to do, no new backups
    r2 = _run(["--apply", "--telemetry", str(tel_db), "--training", str(tr_db), "--trajectories", str(traj)])
    assert r2.returncode == 0 and "0 row(s)/line(s) rewritten" in r2.stdout


def test_scrub_missing_stores_are_said_not_skipped_silently(tmp_path):
    r = _run(["--telemetry", str(tmp_path / "no.db"), "--training", str(tmp_path / "no2.db"), "--trajectories", str(tmp_path / "none")])
    assert r.returncode == 0
    assert r.stdout.count("not found (skipped") == 3
