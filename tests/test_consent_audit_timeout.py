"""SPRINT-CONSENT Phases 3+4 — the audit records decisions, expiry notifies.

WHY THESE ASSERT WHAT THEY ASSERT
---------------------------------
Phase 3's defect was an ABSENCE, and absences are what green suites miss.
``AuditDecision.CONFIRM_APPROVED`` and ``CONFIRM_REJECTED`` were defined in
``audit.py`` and referenced nowhere in ``src/``; the request half wrote
(``checker.py`` logs CONFIRM_PENDING three times) and the resolution half
wrote nothing. Result: 24,048 rows over four months, zero resolutions,
against at least six demonstrated approvals.

Proven before these were written: a mutation making ``_audit_resolution``
return immediately left the whole grants suite green. So every assertion
below READS ROWS BACK from a real ``AuditLogger`` on disk. Asserting the
writer was called would reproduce the defect one level up.

Phase 4's defect was silence: a 300-second expiry popped the request and said
nothing, so it read as a broken ``/approve always``.
"""

from __future__ import annotations

import asyncio

import pytest
import yaml

from prometheus.gateway import commands as cmds
from prometheus.permissions.approval_queue import (
    DEFAULT_APPROVAL_TIMEOUT_SECONDS, ApprovalQueue, ApprovalResult,
    approve_verbs, normalise_scope)
from prometheus.permissions.audit import AuditDecision, AuditLogger
from prometheus.permissions.checker import SecurityGate


@pytest.fixture
def config(tmp_path):
    p = tmp_path / "prometheus.yaml"
    p.write_text(yaml.dump({"security": {"denied_paths": ["/etc"]}}))
    return p


@pytest.fixture
def queue(config, tmp_path):
    """A real queue with a real gate and a real AuditLogger on disk."""
    q = ApprovalQueue(timeout_seconds=5)
    gate = SecurityGate(approval_queue=q, config_path=str(config))
    gate._audit = AuditLogger(tmp_path / "audit")
    q._security_gate = gate
    return q


def _rows(queue, decision=None):
    """Rows read BACK from the store — never the writer's return value."""
    import sqlite3

    db = queue._security_gate._audit.db_path
    con = sqlite3.connect(str(db))
    try:
        sql = "select tool_name, decision, reason from permission_audit"
        args = ()
        if decision is not None:
            sql += " where decision = ?"
            args = (decision.value,)
        return con.execute(sql, args).fetchall()
    finally:
        con.close()


async def _request(q, **kw):
    task = asyncio.create_task(q.request_approval("write_file", "outside", **kw))
    for _ in range(100):
        if q.pending:
            break
        await asyncio.sleep(0.01)
    q._test_task = task
    return list(q.pending.keys())[0]


# ── Phase 3: resolutions are recorded, both directions ─────────────────────

@pytest.mark.asyncio
async def test_approval_writes_a_confirm_approved_row(queue, tmp_path):
    target = tmp_path / "f.txt"
    target.write_text("x")
    rid = await _request(queue, grant_file_path=str(target))
    assert _rows(queue, AuditDecision.CONFIRM_APPROVED) == []

    await cmds.cmd_approve(queue, f"always {rid}")

    rows = _rows(queue, AuditDecision.CONFIRM_APPROVED)
    assert len(rows) == 1, "no confirm_approved row — the resolution half is silent again"
    tool, _dec, reason = rows[0]
    assert tool == "write_file"
    assert f"request={rid}" in reason, "no join key back to the request"
    assert "scope=always" in reason, "scope not recorded"
    assert "grant=" in reason and "none" not in reason.split("grant=")[1][:8]


@pytest.mark.asyncio
async def test_denial_writes_a_confirm_rejected_row(queue):
    rid = await _request(queue)
    await queue.deny(rid)
    rows = _rows(queue, AuditDecision.CONFIRM_REJECTED)
    assert len(rows) == 1, "a denial left no record"
    assert f"request={rid}" in rows[0][2]


@pytest.mark.asyncio
async def test_timeout_writes_a_confirm_timeout_row(config, tmp_path):
    """An expired request is an OUTCOME, not an absence.

    Without this row a timeout is indistinguishable from a request nobody
    ever answered — the ambiguity that cost a live probe to resolve."""
    q = ApprovalQueue(timeout_seconds=1)
    gate = SecurityGate(approval_queue=q, config_path=str(config))
    gate._audit = AuditLogger(tmp_path / "audit")
    q._security_gate = gate

    result = await q.request_approval("write_file", "outside")
    assert result == ApprovalResult.TIMEOUT

    rows = _rows(q, AuditDecision.CONFIRM_TIMEOUT)
    assert len(rows) == 1, "an expiry left no record"
    assert "scope=once" in rows[0][2]


@pytest.mark.asyncio
async def test_scope_distinguishes_always_that_wrote_no_grant(queue):
    """THE ambiguity this field exists to close.

    A rule-4 request creates no grant. Before scope was recorded, that was
    indistinguishable in the store from `always` never having been invoked —
    and telling those apart cost a live probe."""
    rid = await _request(queue)  # no target -> rule 4 -> no grant
    await cmds.cmd_approve(queue, f"always {rid}")

    rows = _rows(queue, AuditDecision.CONFIRM_APPROVED)
    assert len(rows) == 1
    reason = rows[0][2]
    assert "scope=always" in reason, "the attempt is invisible without scope"
    assert "grant=none" in reason, "cannot tell 'granted nothing' from 'never asked'"


# ── Phase 4: expiry notifies, and the number is defensible ─────────────────

@pytest.mark.asyncio
async def test_expiry_notifies_the_operator(config, tmp_path):
    """Silence on a security surface is the defect. A 300s expiry that said
    nothing was reported as a broken /approve always."""
    sent = []

    class Tg:
        async def send(self, chat, text, **kw):
            sent.append(text)

    q = ApprovalQueue(telegram_adapter=Tg(), timeout_seconds=1, default_chat_id=1)
    gate = SecurityGate(approval_queue=q, config_path=str(config))
    gate._audit = AuditLogger(tmp_path / "audit")
    q._security_gate = gate

    rid = None
    task = asyncio.create_task(
        q.request_approval("write_file", "outside", grant_file_path="/tmp/x.txt")
    )
    for _ in range(100):
        if q.pending:
            rid = list(q.pending)[0]
            break
        await asyncio.sleep(0.01)
    assert await task == ApprovalResult.TIMEOUT

    expiry = [m for m in sent if "EXPIRED" in m]
    assert expiry, f"expiry was silent; messages were {sent!r}"
    msg = expiry[0]
    assert rid in msg, "the expiry notice does not name the request"
    assert "/tmp/x.txt" in msg, "the expiry notice does not name the target"
    assert "did NOT run" in msg


def test_timeout_default_matches_the_shipped_template():
    """The drift guard compares key PRESENCE and cannot see a value
    divergence — that is exactly how live max_tool_iterations: 50 sat against
    a template saying 25. This compares the VALUES."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    tpl = yaml.safe_load((root / "config/prometheus.yaml.default").read_text())
    tpl_value = tpl["security"]["approval_queue"]["timeout_seconds"]
    assert tpl_value == DEFAULT_APPROVAL_TIMEOUT_SECONDS, (
        f"template says {tpl_value}, code default is "
        f"{DEFAULT_APPROVAL_TIMEOUT_SECONDS} — a divergence no guard can see"
    )


def test_timeout_is_longer_than_a_phone_glance():
    """Guards the reasoning, not just the number: 5 minutes assumes the
    operator is already looking at the chat."""
    assert DEFAULT_APPROVAL_TIMEOUT_SECONDS >= 900, (
        "an approval window under 15 minutes assumes the operator is already "
        "watching; the field incident was a 300s expiry read as a bug"
    )


# ── Task 1: one vocabulary, both surfaces ──────────────────────────────────

@pytest.mark.asyncio
async def test_every_offered_verb_is_accepted_by_the_rest_validator(queue, tmp_path):
    """The gap found live: REST 400'd the verbs the prompt offered.

    Both surfaces now derive from ``approve_verbs()``, so this asserts the
    property rather than comparing two lists — there is only one list."""
    from prometheus.permissions.approval_queue import prospective_extents

    target = tmp_path / "f.txt"
    target.write_text("x")
    rid = await _request(queue, grant_file_path=str(target))
    offered = set(prospective_extents(queue.pending[rid]))

    assert offered, "the prompt offered nothing to test"
    for verb in offered:
        assert normalise_scope(verb) == verb, (
            f"the prompt offers {verb!r} and the shared validator rejects it "
            f"— the vocabularies have drifted again"
        )
    await queue.deny(rid)


def test_unknown_verbs_are_still_rejected():
    """The other direction: widening the accepted set must not make it
    accept anything."""
    for bad in ("forever", "always always", "permanent", "here", "sessionn"):
        assert normalise_scope(bad) is None, f"{bad!r} was accepted"


def test_retired_spellings_still_resolve():
    """`session` is an alias, not an offer — muscle memory must not break."""
    assert normalise_scope("session") == "until-restart"
    assert normalise_scope("session here") == "until-restart here"
    assert "session" not in approve_verbs(), "the retired verb is being offered"


def test_daemon_fallback_matches_the_named_default():
    """Binds the THIRD copy of the timeout to the named constant.

    The value lives in three places: this constant, the shipped template, and
    daemon.py's ``.get(..., 1800)`` fallback. #221's guard compares the
    template to daemon.py by STATICALLY PARSING that line, so it must stay a
    literal — a named constant there reads as <no-default> and silently
    defeats the guard. This test closes the remaining edge: literal ==
    constant. Together the two make all three agree.

    Found because #221 fired on the first attempt at this change."""
    import re
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    src = (root / "src/prometheus/daemon.py").read_text()
    m = re.search(r'timeout_seconds=approval_cfg\.get\("timeout_seconds",\s*(\d+)\)', src)
    assert m, "daemon.py's approval timeout fallback is no longer a parseable literal"
    assert int(m.group(1)) == DEFAULT_APPROVAL_TIMEOUT_SECONDS, (
        f"daemon.py falls back to {m.group(1)}, the named default is "
        f"{DEFAULT_APPROVAL_TIMEOUT_SECONDS}"
    )
