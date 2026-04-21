"""Approval queue — Telegram-based confirmation flow for LEVEL 1 actions.

When SecurityGate returns requires_confirmation=True, the queue sends a
Telegram message asking the user to approve or deny.  The agent waits
for the response (with timeout).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

logger = logging.getLogger(__name__)


class ApprovalResult(str, Enum):
    APPROVED = "approved"
    DENIED = "denied"
    TIMEOUT = "timeout"


@dataclass
class PendingAction:
    """A tool call waiting for user approval."""

    request_id: str
    tool_name: str
    description: str
    created_at: float = field(default_factory=time.time)
    _event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _result: ApprovalResult = ApprovalResult.TIMEOUT


class ApprovalQueue:
    """Manages pending LEVEL 1 approval requests via Telegram.

    Usage::

        queue = ApprovalQueue(telegram_adapter=tg, timeout_seconds=300)
        # Wire into SecurityGate:
        gate = SecurityGate(..., approval_queue=queue)

        # In agent loop, when requires_confirmation:
        result = await queue.request_approval("bash", "git push origin main")
        if result == ApprovalResult.APPROVED:
            # execute
    """

    def __init__(
        self,
        telegram_adapter=None,
        timeout_seconds: int = 300,
        default_chat_id: int | None = None,
    ) -> None:
        self._telegram = telegram_adapter
        self._timeout = timeout_seconds
        self._default_chat_id = default_chat_id
        self.pending: dict[str, PendingAction] = {}

    async def request_approval(
        self,
        tool_name: str,
        description: str,
        chat_id: int | None = None,
    ) -> ApprovalResult:
        """Queue an action for user approval.

        Sends a Telegram message and waits for /approve or /deny response.
        Returns APPROVED, DENIED, or TIMEOUT.
        """
        request_id = uuid4().hex[:8]
        action = PendingAction(
            request_id=request_id,
            tool_name=tool_name,
            description=description,
        )
        self.pending[request_id] = action

        # Send notification via Telegram
        target_chat = chat_id or self._default_chat_id
        if self._telegram and target_chat:
            msg = (
                f"Permission requested:\n"
                f"Tool: {tool_name}\n"
                f"Action: {description}\n\n"
                f"/approve {request_id} or /deny {request_id}"
            )
            try:
                await self._telegram.send(target_chat, msg, parse_mode=None)
            except Exception as exc:
                logger.warning("Failed to send approval request: %s", exc)

        # Wait for response or timeout
        try:
            await asyncio.wait_for(action._event.wait(), timeout=self._timeout)
        except asyncio.TimeoutError:
            action._result = ApprovalResult.TIMEOUT

        # Clean up
        self.pending.pop(request_id, None)
        return action._result

    async def approve(self, request_id: str) -> bool:
        """Approve a pending action. Returns True if found and approved."""
        action = self.pending.get(request_id)
        if action is None:
            return False
        action._result = ApprovalResult.APPROVED
        action._event.set()
        return True

    async def deny(self, request_id: str) -> bool:
        """Deny a pending action. Returns True if found and denied."""
        action = self.pending.get(request_id)
        if action is None:
            return False
        action._result = ApprovalResult.DENIED
        action._event.set()
        return True

    def list_pending(self) -> list[PendingAction]:
        """Return all pending approval requests."""
        return list(self.pending.values())
