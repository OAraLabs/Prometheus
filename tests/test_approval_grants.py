"""Tests for /approve scopes (once|session|always) and /grants listing.

Covers the shared command core (prometheus.gateway.commands) and its
wiring to SecurityGate grants via the queue's ``_security_gate`` handle.
"""
import asyncio

import pytest

from prometheus.gateway import commands as cmds
from prometheus.permissions.approval_queue import ApprovalQueue
from prometheus.permissions.checker import Grant, SecurityGate

# built at runtime so the source never contains a blocked literal
_CHAINED_BAD = "ls -la /tmp; " + "rm " + "-rf /"


def _make_queue() -> ApprovalQueue:
    queue = ApprovalQueue(timeout_seconds=5)
    queue._security_gate = SecurityGate(approval_queue=queue)
    return queue


async def _request(queue: ApprovalQueue, **kwargs) -> str:
    """Queue a pending action without blocking; return its request id."""
    task = asyncio.create_task(queue.request_approval("bash", "ls -la", **kwargs))
    for _ in range(100):
        if queue.pending:
            break
        await asyncio.sleep(0.01)
    assert queue.pending, "request_approval did not queue"
    rid = list(queue.pending.keys())[0]
    queue._test_task = task  # prevent GC; approve/deny resolves it
    return rid


class TestCmdApproveScopes:
    @pytest.mark.asyncio
    async def test_plain_approve_is_once_no_grant(self):
        queue = _make_queue()
        rid = await _request(queue)
        text = await cmds.cmd_approve(queue, rid)
        assert rid in text
        assert queue._security_gate.list_grants() == []

    @pytest.mark.asyncio
    async def test_session_alias_records_an_until_restart_grant(self):
        """RENAMED by SPRINT-CONSENT: scope "session" -> "until_restart".

        This test asserted ``scope == "session"``. It was not coupled to a
        defect — it faithfully pinned a label, and the LABEL is what changed:
        there is one SecurityGate per process, ``_grants`` is never cleared,
        and ``matches()`` never reads scope, so "session" promised a boundary
        the system does not have. The verb is kept as a silent ALIAS so
        muscle memory still works; only the recorded value and the offered
        vocabulary changed."""
        queue = _make_queue()
        rid = await _request(queue, grant_command="ls -la")
        text = await cmds.cmd_approve(queue, f"session {rid}")
        assert "restart" in text.lower(), text
        grants = queue._security_gate.list_grants()
        assert len(grants) == 1
        assert grants[0].scope == "until_restart"
        assert grants[0].kind == "command_prefix"

    @pytest.mark.asyncio
    async def test_until_restart_verb_is_equivalent_to_the_alias(self):
        """The offered verb and the legacy alias must not diverge."""
        queue = _make_queue()
        rid = await _request(queue, grant_command="ls -la")
        await cmds.cmd_approve(queue, f"until-restart {rid}")
        grants = queue._security_gate.list_grants()
        assert len(grants) == 1
        assert grants[0].scope == "until_restart"

    @pytest.mark.asyncio
    async def test_always_scope_records_persistent_grant(self):
        queue = _make_queue()
        rid = await _request(queue, grant_command="ls -la")
        text = await cmds.cmd_approve(queue, f"always {rid}")
        assert "always" in text.lower() or "permanently" in text.lower()
        grants = queue._security_gate.list_grants()
        assert len(grants) == 1
        assert grants[0].scope == "persistent"

    @pytest.mark.asyncio
    async def test_invalid_scope_rejected_without_approving(self):
        queue = _make_queue()
        rid = await _request(queue)
        text = await cmds.cmd_approve(queue, f"forever {rid}")
        # Vocabulary changed with the rename; the usage must name the
        # verbs it actually accepts, not the retired one.
        assert "Usage" in text and "until-restart" in text and "always" in text
        assert rid in queue.pending  # approve() never called
        await queue.deny(rid)  # clean up background task

    @pytest.mark.asyncio
    async def test_scope_without_gate_falls_back_to_once(self):
        queue = ApprovalQueue(timeout_seconds=5)  # no _security_gate
        rid = await _request(queue, grant_command="ls -la")
        text = await cmds.cmd_approve(queue, f"session {rid}")
        assert rid in text or "security gate" in text.lower()


class TestCmdGrants:
    def test_no_gate_lists_nothing(self):
        queue = ApprovalQueue(timeout_seconds=5)
        assert "No security gate" in cmds.cmd_grants(queue)

    def test_none_queue(self):
        assert cmds.cmd_grants(None)

    def test_lists_recorded_grants(self):
        queue = _make_queue()
        gate = queue._security_gate
        gate.add_grant(Grant(kind="command_prefix", value="pytest",
                             tool_name="bash", scope="persistent"))
        gate.add_grant(Grant(kind="path_prefix", value="/tmp/x",
                             tool_name="file_write", scope="until_restart"))
        text = cmds.cmd_grants(queue)
        assert "pytest" in text
        assert "/tmp/x" in text
        # SPRINT-CONSENT: the listing now renders each grant's EXTENT via
        # Grant.describe() rather than printing the raw scope token, and
        # leads with the id because that is the handle /revoke takes. The
        # duration is still stated — in words the operator can act on.
        assert "permanently" in text
        assert "until the daemon restarts" in text
        for g in gate.list_grants():
            assert g.grant_id in text, "a listed grant showed no revocation handle"
        assert "revoke" in text.lower()


class TestGrantMatching:
    def test_command_prefix_matches_and_blocks_chaining(self):
        g = Grant(kind="command_prefix", value="ls -la", tool_name="bash")
        assert g.matches("bash", None, "ls -la /tmp")
        assert not g.matches("bash", None, _CHAINED_BAD)
        assert not g.matches("bash", None, "echo hi")

    def test_path_prefix_resolves_traversal(self):
        g = Grant(kind="path_prefix", value="/tmp/safe", tool_name="file_write")
        assert g.matches("file_write", "/tmp/safe/x.txt", None)
        assert not g.matches("file_write", "/tmp/safe/../evil.txt", None)
        assert not g.matches("file_write", "/tmp/other.txt", None)

    def test_tool_kind_matches_tool_name(self):
        g = Grant(kind="tool", value="", tool_name="file_write")
        assert g.matches("file_write", "/anything", None)
        assert not g.matches("bash", None, "ls")

    def test_dedup_on_add(self):
        gate = SecurityGate()
        gate.add_grant(Grant(kind="command_prefix", value="ls", tool_name="bash"))
        gate.add_grant(Grant(kind="command_prefix", value="ls", tool_name="bash"))
        assert len(gate.list_grants()) == 1


# --------------------------------------------------------------------------- #
# The id is optional when it is unambiguous
#
# Live 2026-08-14: five consecutive bare `/approve` messages produced five
# usage replies and zero approvals. Every form demanded an explicit 8-hex id,
# so clearing an approval storm meant scrolling back to copy one per prompt.
# --------------------------------------------------------------------------- #

import asyncio as _asyncio
from unittest.mock import MagicMock as _MagicMock

from prometheus.gateway.commands import cmd_approve, cmd_deny
from prometheus.permissions.approval_queue import ApprovalQueue, ApprovalResult


def _result_of(queue, request_id: str):
    """The recorded verdict for a request.

    approve()/deny() set the action's result and wake the waiter; the pending
    dict is drained by request_approval on the WAITING side, which no test
    here runs — so asserting on the dict would assert the wrong half.
    """
    return queue.pending[request_id]._result


def _queue_with(n: int) -> ApprovalQueue:
    """A queue holding *n* pending requests, oldest first."""
    from prometheus.permissions.approval_queue import PendingAction

    q = ApprovalQueue()
    for i in range(n):
        rid = f"{i:08x}"
        q.pending[rid] = PendingAction(
            request_id=rid,
            tool_name="write_file",
            description=f"write /tmp/file{i}.txt",
            created_at=float(i),
        )
    return q


class TestBareApprove:

    def test_bare_approve_takes_the_only_pending_request(self):
        q = _queue_with(1)
        out = _asyncio.run(cmd_approve(q, ""))
        assert "Approved" in out
        assert "Usage" not in out
        assert _result_of(q, "00000000") is ApprovalResult.APPROVED

    def test_bare_approve_with_nothing_pending_says_so(self):
        out = _asyncio.run(cmd_approve(ApprovalQueue(), ""))
        assert "No pending approval requests." == out

    def test_bare_approve_with_several_lists_them_and_approves_nothing(self):
        q = _queue_with(3)
        out = _asyncio.run(cmd_approve(q, ""))
        assert "3 pending requests" in out
        for i in range(3):
            assert f"{i:08x}" in out
        # Ambiguity must never be resolved by guessing.
        assert len(q.pending) == 3

    def test_scope_without_id_also_resolves(self):
        q = _queue_with(1)
        out = _asyncio.run(cmd_approve(q, "session"))
        assert "Usage" not in out
        assert _result_of(q, "00000000") is ApprovalResult.APPROVED

    def test_explicit_id_still_works(self):
        q = _queue_with(2)
        out = _asyncio.run(cmd_approve(q, "00000001"))
        assert "Approved: 00000001" in out
        assert "00000000" in q.pending

    def test_single_request_shortcut_picks_the_oldest(self):
        """A request arriving between prompt and reply must not steal the
        shortcut — but with 2 pending we list rather than pick, so this
        asserts the ordering the listing uses."""
        q = _queue_with(2)
        out = _asyncio.run(cmd_approve(q, ""))
        assert out.index("00000000") < out.index("00000001")

    def test_mistyped_scope_still_returns_usage(self):
        q = _queue_with(1)
        out = _asyncio.run(cmd_approve(q, "forever abcdef12"))
        assert "Usage" in out
        assert len(q.pending) == 1


class TestApproveAll:

    def test_approve_all_clears_the_queue(self):
        q = _queue_with(4)
        out = _asyncio.run(cmd_approve(q, "all"))
        assert "Approved 4 request(s)" in out
        assert all(
            _result_of(q, f"{i:08x}") is ApprovalResult.APPROVED for i in range(4)
        )

    def test_approve_all_grants_nothing_persistent(self):
        """Draining a backlog must not widen trust — that is what `always`
        is for, deliberately chosen one request at a time."""
        q = _queue_with(3)
        gate = _MagicMock()
        q._security_gate = gate
        _asyncio.run(cmd_approve(q, "all"))
        gate.add_grant.assert_not_called()
        gate.persist_grant.assert_not_called()

    def test_approve_all_with_empty_queue(self):
        out = _asyncio.run(cmd_approve(ApprovalQueue(), "all"))
        assert "No pending approval requests." == out


class TestBareDeny:

    def test_bare_deny_takes_the_only_pending_request(self):
        q = _queue_with(1)
        out = _asyncio.run(cmd_deny(q, ""))
        assert "Denied" in out
        assert _result_of(q, "00000000") is ApprovalResult.DENIED

    def test_bare_deny_with_several_lists_them(self):
        q = _queue_with(2)
        out = _asyncio.run(cmd_deny(q, ""))
        assert "2 pending requests" in out
        assert len(q.pending) == 2

    def test_bare_deny_with_nothing_pending(self):
        assert _asyncio.run(cmd_deny(ApprovalQueue(), "")) == "No pending approval requests."
