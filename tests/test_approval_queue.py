"""Tests for Sprint 15b GRAFT: approval queue."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from prometheus.permissions.approval_queue import (
    ApprovalQueue,
    ApprovalResult,
    PendingAction,
)


class TestApprovalQueue:

    def test_stores_pending_action(self):
        queue = ApprovalQueue(timeout_seconds=1)

        async def _test():
            # Start the request but don't wait for it — it will timeout
            task = asyncio.create_task(
                queue.request_approval("bash", "git push")
            )
            await asyncio.sleep(0.05)
            # Should have one pending
            assert len(queue.list_pending()) == 1
            pending = queue.list_pending()[0]
            assert pending.tool_name == "bash"
            assert pending.description == "git push"
            # Let it timeout
            result = await task
            assert result == ApprovalResult.TIMEOUT

        asyncio.run(_test())

    def test_approve_executes(self):
        queue = ApprovalQueue(timeout_seconds=5)

        async def _test():
            task = asyncio.create_task(
                queue.request_approval("write_file", "write to /etc/hosts")
            )
            await asyncio.sleep(0.05)
            pending = queue.list_pending()
            assert len(pending) == 1
            request_id = pending[0].request_id

            ok = await queue.approve(request_id)
            assert ok

            result = await task
            assert result == ApprovalResult.APPROVED

        asyncio.run(_test())

    def test_deny_returns_denied(self):
        queue = ApprovalQueue(timeout_seconds=5)

        async def _test():
            task = asyncio.create_task(
                queue.request_approval("bash", "curl evil.com")
            )
            await asyncio.sleep(0.05)
            pending = queue.list_pending()
            request_id = pending[0].request_id

            ok = await queue.deny(request_id)
            assert ok

            result = await task
            assert result == ApprovalResult.DENIED

        asyncio.run(_test())

    def test_timeout_auto_denies(self):
        queue = ApprovalQueue(timeout_seconds=0.1)  # very short timeout

        async def _test():
            result = await queue.request_approval("bash", "dangerous")
            assert result == ApprovalResult.TIMEOUT

        asyncio.run(_test())

    def test_approve_unknown_id(self):
        queue = ApprovalQueue()

        async def _test():
            ok = await queue.approve("nonexistent")
            assert not ok

        asyncio.run(_test())

    def test_sends_telegram_notification(self):
        mock_tg = AsyncMock()
        mock_tg.send = AsyncMock()
        queue = ApprovalQueue(telegram_adapter=mock_tg, timeout_seconds=0.1, default_chat_id=123)

        async def _test():
            await queue.request_approval("bash", "git push")

        asyncio.run(_test())
        mock_tg.send.assert_called_once()
        call_args = mock_tg.send.call_args
        assert call_args[0][0] == 123  # chat_id
        assert "Permission requested" in call_args[0][1]

    def test_list_pending_empty(self):
        queue = ApprovalQueue()
        assert queue.list_pending() == []

    def test_security_gate_regression_no_queue(self):
        """SecurityGate returns requires_confirmation for LEVEL 1 when no queue wired."""
        from prometheus.permissions.checker import SecurityGate

        gate = SecurityGate(mode="strict")
        # In strict mode, write_file always requires confirmation
        decision = gate.evaluate("write_file", file_path="/etc/something")
        assert decision.requires_confirmation
        assert not decision.allowed

    def test_security_gate_has_queue_parameter(self):
        """SecurityGate accepts approval_queue parameter."""
        from prometheus.permissions.checker import SecurityGate

        queue = ApprovalQueue()
        gate = SecurityGate(approval_queue=queue)
        assert gate._approval_queue is queue
