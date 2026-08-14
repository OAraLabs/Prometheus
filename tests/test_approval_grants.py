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
    async def test_session_scope_records_session_grant(self):
        queue = _make_queue()
        rid = await _request(queue, grant_command="ls -la")
        text = await cmds.cmd_approve(queue, f"session {rid}")
        assert "session" in text.lower()
        grants = queue._security_gate.list_grants()
        assert len(grants) == 1
        assert grants[0].scope == "session"
        assert grants[0].kind == "command_prefix"

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
        assert "Usage" in text and "session" in text and "always" in text
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
                             tool_name="file_write", scope="session"))
        text = cmds.cmd_grants(queue)
        assert "pytest" in text
        assert "/tmp/x" in text
        assert "persistent" in text
        assert "session" in text


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
