"""The audit trail must not store — or re-feed — the credential it logged.

THE DEFECT
----------
`AuditLogger.log` built its entry with `reason=reason`: no redaction, no
truncation. `tool_input_summary` beside it went through `_summarize_input`,
which does both.

The gate builds its reason FROM the command:

    reason = f"Command requires approval: {command!r}"

so the same string reached `log()` twice and was stored two different ways in
the same row — masked in one column, verbatim in the next:

    tool_input_summary : curl -H "Authorization: Bearer ***" …?access_token=***
    reason             : Command requires approval: 'curl -H "Authorization:
                         Bearer sk-ant-SECRETVALUE001" …?access_token=SECRET…'

System-origin `curl`, `wget` and `ssh` are exactly the commands that reach the
approve tier, and exactly the ones carrying bearer headers and tokenised URLs.
And this is not a write-only log: `audit_query` returns these rows INTO MODEL
CONTEXT, so an unredacted reason re-feeds the secret to the model on every
later query.

TWO CONTROLS, AND WHY THAT SHAPES EVERY TEST HERE
--------------------------------------------------
  SINK      does `reason` get scrubbed AT ALL — redacted and bounded?
  REDACTOR  does `_redact` mask a secret whose NAME it does not recognise?

They fail independently and they are fixed independently, so a test that
cannot say which one fired is not a test of either. Every test below is
written to move exactly one of them.

  * The SINK tests never depend on what the redactor decides. One asserts
    TRUNCATION, which the redactor never performs. The other replaces
    `_redact` with a sentinel and asserts the sentinel's mark reaches the
    stored field — plumbing, with no opinion about redaction rules.

  * The REDACTOR tests call `_redact` directly, and only ever with names the
    NAME-BASED layer does not recognise.

WHY THE FIXTURE NAMES ARE WHAT THEY ARE — READ THIS BEFORE EDITING
-------------------------------------------------------------------
`UPSTREAM_PASSPHRASE`, `DOCS_SEED` and `DOCS_ENDPOINT` are used throughout,
and never `DOCS_TOKEN` or `API_KEY`.

That is deliberate and load-bearing. The name layer masks a value when the
identifier beside it contains one of five words — `api_key`, `token`,
`secret`, `password`, `auth`. A leak test written against `DOCS_TOKEN` passes
on the strength of the word "token" and would pass with the sink completely
broken, because the redactor would have caught it on the way past. It measures
the redactor, not the code under test. That exact mutant survived here once.

`test_the_fixture_names_are_not_caught_by_the_name_layer` pins the property
these fixtures are chosen for, so a future edit that swaps in a "nicer" name
fails loudly instead of quietly going vacuous.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prometheus.permissions.audit import (  # noqa: E402
    AuditDecision,
    AuditLogger,
)

# Secrets whose NAME the name-based layer does not recognise. See the module
# docstring: this is the whole point of the fixture choice.
UNCAUGHT_BY_NAME = {
    "UPSTREAM_PASSPHRASE": "PASSPHRASEVALUE0123456789",
    "DOCS_SEED": "SEEDVALUE0123456789abcd",
    "DOCS_ENDPOINT": "https://internal.example.invalid/v1/docs",
}

# A realistic approve-tier command: the shape the gate actually logs.
BEARER = "sk-ant-BEARERVALUE00000000000000"
QUERY_TOKEN = "QUERYTOKENVALUE000000"
CURL = (
    f'curl -H "Authorization: Bearer {BEARER}" '
    f"https://api.internal.example.invalid/v1/x?access_token={QUERY_TOKEN}"
)


@pytest.fixture
def audit(tmp_path):
    return AuditLogger(tmp_path)


def _stored_rows(tmp_path: Path) -> list[dict]:
    """Every audit row, read back from BOTH durable sinks.

    JSONL and SQLite are written separately, so a fix applied to one and not
    the other would leak from the sink nobody checked.
    """
    rows: list[dict] = []
    jsonl = tmp_path / "permission_audit.jsonl"
    if jsonl.exists():
        rows += [json.loads(line) for line in jsonl.read_text().splitlines() if line]
    conn = sqlite3.connect(tmp_path / "audit.db")
    try:
        conn.row_factory = sqlite3.Row
        rows += [dict(r) for r in conn.execute("SELECT * FROM permission_audit")]
    finally:
        conn.close()
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# THE FIXTURE PROPERTY THIS FILE DEPENDS ON
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name", sorted(UNCAUGHT_BY_NAME))
def test_the_fixture_names_are_not_caught_by_the_name_layer(name, audit):
    """The name layer alone must leave these untouched.

    Without this, every redaction assertion below could be passing because of
    the word in the identifier rather than because of the control under test —
    and nobody would know which.
    """
    probe = f"{name}={UNCAUGHT_BY_NAME[name]}"
    masked = probe
    for pattern, replacement in AuditLogger._NAME_REDACT_PATTERNS:
        masked = pattern.sub(replacement, masked)
    assert masked == probe, (
        f"{name!r} IS recognised by the name-based layer, so any test using it "
        f"measures that layer rather than the shape layer. Pick a name none of "
        f"{['api_key', 'token', 'secret', 'password', 'auth']} appears in."
    )


# ═══════════════════════════════════════════════════════════════════════════
# CONTROL 1 — THE SINK.  No opinion about what the redactor does.
# ═══════════════════════════════════════════════════════════════════════════

def test_the_reason_is_bounded(tmp_path):
    """TRUNCATION is the redactor-independent half of the sink.

    `_redact` never shortens anything, so this test moves only when the sink
    moves. It is the cleanest isolation available: no secret, no masking, no
    dependence on any pattern.
    """
    audit = AuditLogger(tmp_path, max_reason_chars=64)
    long_reason = "B" * 500

    entry = audit.log("bash", AuditDecision.DENY, 0, long_reason)

    assert len(entry.reason) <= 64 + 3, (
        f"reason was stored at {len(entry.reason)} chars against a 64-char "
        f"bound — it is not being truncated"
    )
    assert entry.reason.endswith("...")
    for row in _stored_rows(tmp_path):
        assert len(row["reason"]) <= 64 + 3


def test_the_reason_passes_through_the_redactor(tmp_path, monkeypatch):
    """PLUMBING, proven with a sentinel instead of a real secret.

    Replacing `_redact` with a marker asserts that `reason` is routed through
    it, WITHOUT asserting anything about redaction rules. If the redactor were
    deleted entirely this test still passes; if the sink stops calling it, this
    test fails. That is the isolation the two-control split requires.
    """
    monkeypatch.setattr(
        AuditLogger, "_redact", lambda self, text: f"<<SCRUBBED>>{text}"
    )
    audit = AuditLogger(tmp_path)

    entry = audit.log("bash", AuditDecision.CONFIRM_PENDING, 1, "some reason")

    assert entry.reason.startswith("<<SCRUBBED>>"), (
        "the reason field did not pass through _redact — it is being stored "
        "raw, which is the defect this PR exists to fix"
    )
    for row in _stored_rows(tmp_path):
        assert row["reason"].startswith("<<SCRUBBED>>"), (
            "a durable sink received an unscrubbed reason"
        )


def test_both_fields_are_scrubbed_the_same_way(tmp_path, monkeypatch):
    """The two columns must not disagree about the same string.

    The defect was not "reason is unredacted" in isolation — it was that ONE
    row held the same command masked in one column and verbatim in the next.
    Sentinel again, so this says nothing about which secrets are recognised.
    """
    monkeypatch.setattr(
        AuditLogger, "_redact", lambda self, text: f"<<SCRUBBED>>{text}"
    )
    audit = AuditLogger(tmp_path)

    entry = audit.log(
        "bash", AuditDecision.CONFIRM_PENDING, 1,
        f"Command requires approval: {CURL!r}", tool_input=CURL,
    )

    assert entry.reason.startswith("<<SCRUBBED>>")
    assert entry.tool_input_summary.startswith("<<SCRUBBED>>")


# ═══════════════════════════════════════════════════════════════════════════
# CONTROL 2 — THE REDACTOR.  Direct calls, uncaught names only.
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("name", sorted(UNCAUGHT_BY_NAME))
def test_the_redactor_masks_a_value_whose_name_it_does_not_know(name, audit):
    """Shape, not name.

    These three are the real ones from this repo that the name layer misses.
    `_redact` is called DIRECTLY — no logging, no sink — so a broken sink
    cannot make this pass or fail.
    """
    value = UNCAUGHT_BY_NAME[name]
    out = audit._redact(f"{name}={value}")
    assert value not in out, (
        f"{name}={value!r} survived redaction. The name layer does not "
        f"recognise {name!r} (pinned above), so the shape layer is what should "
        f"have masked it.\n  got: {out}"
    )


def test_the_redactor_masks_a_query_parameter_whatever_it_is_called(audit):
    """A tokenised URL is a credential regardless of the vendor's key name."""
    out = audit._redact(f"https://h.invalid/x?wholly_unknown_name={QUERY_TOKEN}")
    assert QUERY_TOKEN not in out, out


def test_the_redactor_masks_url_userinfo(audit):
    out = audit._redact("https://alice:PASSWORDVALUE123@internal.invalid/x")
    assert "PASSWORDVALUE123" not in out, out


def test_the_redactor_leaves_ordinary_command_text_readable(audit):
    """Over-redaction destroys the log's purpose.

    An audit trail nobody can read is not a safer audit trail. Paths, short
    values and lowercase flags must survive — otherwise the answer to "what
    did the agent try to run" becomes `***`.
    """
    cmd = "curl --retry 3 --output /tmp/some/long/path/file.txt https://h.invalid/a"
    out = audit._redact(cmd)
    assert out == cmd, f"ordinary command text was redacted:\n  {out}"


# ═══════════════════════════════════════════════════════════════════════════
# END TO END — both controls together, on the real logged shape
# ═══════════════════════════════════════════════════════════════════════════

def test_an_approve_tier_curl_leaves_no_credential_in_any_sink(tmp_path):
    """The whole point, measured on every durable surface.

    Deliberately NOT an isolation test — it cannot say which control fired,
    and it is not asked to. It answers the different question of whether the
    secret is anywhere after a realistic call.
    """
    audit = AuditLogger(tmp_path)
    audit.log(
        "bash", AuditDecision.CONFIRM_PENDING, 1,
        f"Command requires approval: {CURL!r}", tool_input=CURL,
    )

    rows = _stored_rows(tmp_path)
    assert rows, "nothing was written — this test would pass vacuously"

    blob = json.dumps(rows)
    for secret in (BEARER, QUERY_TOKEN):
        assert secret not in blob, (
            f"{secret!r} is stored in the audit trail, which audit_query "
            f"returns into model context"
        )
    # And the row is still useful.
    assert any("curl" in json.dumps(r) for r in rows), (
        "the command is unrecognisable in the log — redaction has eaten the "
        "thing the audit trail exists to record"
    )
