"""The ACP approval rule (decided 2026-11-25): a lower-trust channel.

A paired ACP client's own credential may approve ORDINARY actions and deny
anything; a CRITICAL action is refused over ACP on every surface, because the
refusal lives in the one queue every surface reaches. The answer records the
client that gave it (the audit row's user_id), never the daemon's API token.
"""

from __future__ import annotations

import pytest

from prometheus.permissions.approval_queue import ApprovalQueue, ApprovalResult, PendingAction
from prometheus.permissions.approver import ACP, Approver, acp_client
from prometheus.permissions.audit import AuditDecision, AuditLogger
from prometheus.permissions.checker import PermissionMode, SecurityGate

RID = "abc12345"


def _queue(tmp_path):
    gate = SecurityGate(mode=PermissionMode.DEFAULT, audit_logger=AuditLogger(tmp_path / "audit"))
    return ApprovalQueue(security_gate=gate)


def _pending(queue, rid=RID, **kw):
    action = PendingAction(request_id=rid, tool_name=kw.pop("tool_name", "write_file"),
                           description=kw.pop("description", "write a file"), **kw)
    queue.pending[rid] = action
    return action


@pytest.mark.asyncio
async def test_an_ordinary_action_may_be_approved_over_acp(tmp_path):
    q = _queue(tmp_path)
    _pending(q)
    assert await q.approve(RID, by=acp_client("client-1", "Editor")) is True
    rows = q._security_gate._audit.query_recent(limit=10)
    resolutions = [r for r in rows if r.decision == AuditDecision.CONFIRM_APPROVED]
    assert resolutions and resolutions[0].user_id == "acp:client-1"


@pytest.mark.asyncio
async def test_a_critical_action_is_refused_over_acp_and_stays_pending(tmp_path):
    q = _queue(tmp_path)
    # a desktop prompt is critical by construction (permissions/critical.py)
    _pending(q, task_id="task-1")
    assert await q.approve(RID, by=acp_client("client-1")) is False
    assert RID in q.pending  # still pending for a person
    assert q.pending[RID]._result != ApprovalResult.APPROVED


@pytest.mark.asyncio
async def test_the_owners_critical_lists_also_refuse_acp(tmp_path):
    q = _queue(tmp_path)
    _pending(q, critical_matches=("push.critical_commands: git push",))
    assert await q.approve(RID, by=acp_client("client-1")) is False
    assert RID in q.pending


@pytest.mark.asyncio
async def test_a_critical_action_is_still_approvable_by_a_person(tmp_path):
    """The refusal is the CHANNEL's, not the action's: a device answer goes through."""
    q = _queue(tmp_path)
    _pending(q, task_id="task-1")
    assert await q.approve(RID, by=Approver("device-token", "phone-1", "Phone")) is True


@pytest.mark.asyncio
async def test_acp_may_deny_anything_including_critical(tmp_path):
    """Denial narrows; the lower-trust channel may always say no."""
    q = _queue(tmp_path)
    _pending(q, task_id="task-1")
    assert await q.deny(RID, by=acp_client("client-1")) is True
    rows = q._security_gate._audit.query_recent(limit=10)
    resolutions = [r for r in rows if r.decision == AuditDecision.CONFIRM_REJECTED]
    assert resolutions and resolutions[0].user_id == "acp:client-1"


@pytest.mark.asyncio
async def test_an_unnamed_acp_client_is_recorded_as_unknown(tmp_path):
    q = _queue(tmp_path)
    _pending(q)
    by = acp_client("")
    assert by.id == "unknown"
    assert await q.approve(RID, by=by) is True
    rows = q._security_gate._audit.query_recent(limit=10)
    resolutions = [r for r in rows if r.decision == AuditDecision.CONFIRM_APPROVED]
    assert resolutions and resolutions[0].user_id == "acp:unknown"


def test_acp_is_not_a_chat_kind_and_not_the_owner_gate():
    """The chat-owner backstop does not apply; the critical refusal is ACP's own."""
    by = acp_client("client-1")
    assert by.kind == ACP
    assert by.not_the_owner is False


# ── D13 amended 2026-11-26: ACP approvals are ONCE ONLY ─────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["until-restart", "always", "until-restart here", "always here"])
async def test_acp_never_carries_a_lasting_scope(tmp_path, scope):
    q = _queue(tmp_path)
    _pending(q)
    assert await q.approve(RID, by=acp_client("client-1"), scope=scope) is False
    assert RID in q.pending  # still pending for a person
    assert q.pending[RID]._result != ApprovalResult.APPROVED


@pytest.mark.asyncio
async def test_acp_never_creates_a_grant_even_with_once_scope(tmp_path):
    q = _queue(tmp_path)
    _pending(q)
    assert await q.approve(RID, by=acp_client("client-1"), scope="once", grant=object()) is False
    assert RID in q.pending


@pytest.mark.asyncio
async def test_acp_once_approval_writes_nothing_to_config(tmp_path):
    """An ACP approve-once lands, and no grant is minted or persisted: the
    gate's grant list stays empty and no grants file appears."""
    import yaml

    q = _queue(tmp_path)
    _pending(q)
    assert await q.approve(RID, by=acp_client("client-1"), scope="once") is True
    gate = q._security_gate
    assert list(getattr(gate, "_grants", {}) or {}) == []
    # nothing grant-shaped was written anywhere under the tmp tree
    written = [p for p in tmp_path.rglob("*") if p.is_file() and "grant" in p.name.lower()]
    assert written == []
    for p in tmp_path.rglob("*.yaml"):
        data = yaml.safe_load(p.read_text()) or {}
        assert "grants" not in data


@pytest.mark.asyncio
async def test_a_person_may_still_grant_a_lasting_scope(tmp_path):
    """The refusal is the CHANNEL's: a device answer with a lasting scope
    still reaches the grant path (it raises nothing here; the queue's own
    scope handling runs as before)."""
    q = _queue(tmp_path)
    _pending(q)
    # a device approve with 'always' must NOT be refused by the ACP rule;
    # the queue's grant machinery runs (gate present, so no RuntimeError)
    assert await q.approve(RID, by=Approver("device-token", "phone-1", "Phone"),
                           scope="always") is True
