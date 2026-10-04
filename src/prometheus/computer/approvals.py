"""The computer approval channel — desktop prompts, and only desktop prompts.

WHY A SEPARATE QUEUE (design §5.1.4, Appendix A)
-------------------------------------------------
The gate-wide ``ApprovalQueue`` exists only with Telegram plus
``security.approval_queue.enabled``, which ships off. Building it whenever
computer use is on would override the operator's flag for EVERY tool. So a
desktop task asks through this queue instead, which only ``SessionConsent``
feeds. It emits the same ``approval_pending`` / ``approval_resolved`` frames
(Beacon already renders them) and is answered through the same surfaces,
which see both queues via ``permissions.approval_queue.ApprovalQueues``.

WHAT IS DIFFERENT ABOUT A DESKTOP PROMPT
-----------------------------------------
* **Tagged** with the task and the chat session that asked, so a stop can
  deny exactly its own prompts (:meth:`deny_task`) and a surface can show
  which task is asking.
* **Approve-once only.** No lasting scope is offered and none is accepted
  (:class:`~prometheus.computer.door.OnceOnly`) — the binding is the lasting
  consent, and it is per app and per session or task (W2).
* **Answered only by a person** (W3): the API token, a device nobody marked
  and a stranger on an allowed group chat are refused. DENYING is open to
  any credential: it can only stop something.
* **Sent to the chat that started the task**, never to a default chat.
* **Shows what will be acted on** (D16): the element, as the app labels it
  (capped), and the text to be typed (W5) — and nothing else from the
  driver: no token, no pid, no window id.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from prometheus.computer.door import OnceOnly, PersonCheck
from prometheus.permissions.approval_queue import (
    DEFAULT_APPROVAL_TIMEOUT_SECONDS,
    SCOPE_ONCE,
    ApprovalQueue,
    ApprovalResult,
    PendingAction,
    _humanise_window,
)
from prometheus.permissions.approver import Approver
from prometheus.permissions.argument_view import format_arguments, redact_arguments

logger = logging.getLogger(__name__)


class ComputerApprovalChannel(ApprovalQueue):
    """See the module docstring."""

    def __init__(
        self,
        *,
        security_gate: Any,
        people: PersonCheck,
        telegram_adapter: Any = None,
        timeout_seconds: int = DEFAULT_APPROVAL_TIMEOUT_SECONDS,
    ) -> None:
        # No default chat: a desktop prompt goes to the chat that started
        # the task, or to no chat at all (Beacon still gets the frame).
        super().__init__(security_gate=security_gate,
                         telegram_adapter=telegram_adapter,
                         timeout_seconds=timeout_seconds,
                         default_chat_id=None)
        self.people = people

    async def request_for_task(
        self,
        *,
        tool_name: str,
        description: str,
        extent: Any,
        arguments: dict[str, Any] | None,
        task_id: str,
        session_id: str,
        chat_id: int | None = None,
        on_pending: Any = None,
    ) -> ApprovalResult:
        """Ask the person, tagged with the task. Waits for the answer.

        ``on_pending(action)`` runs once the request exists — the action log
        uses it to say the step is waiting, with the request's id.
        """
        request_id = uuid4().hex[:8]
        raw_text = (arguments or {}).get("text")
        action = PendingAction(
            request_id=request_id,
            tool_name=tool_name,
            description=description,
            grant_computer_action=extent,
            arguments=redact_arguments(arguments),
            task_id=task_id,
            session_id=session_id,
            once_only=True,
            text_chars=len(raw_text) if isinstance(raw_text, str) else None,
        )
        self.pending[request_id] = action
        await self._emit("approval_pending", self.serialize_pending(action))
        if on_pending is not None:
            try:
                await on_pending(action)
            except Exception:  # noqa: BLE001 - the log never blocks consent
                logger.debug("on_pending failed", exc_info=True)
        if self._telegram and chat_id:
            lines = [
                f"Desktop task {task_id} asks:",
                description,
            ]
            arg_lines = format_arguments(action.arguments)
            if arg_lines:
                lines.append("With:")
                lines.extend(arg_lines)
            lines += [
                "",
                "/approve — approve this ONCE (or /deny)",
                "A desktop request is once only: nothing is remembered.",
                "/computer stop — stop the task",
                "",
                f"Expires in {_humanise_window(self._timeout)} if unanswered.",
                f"id: {request_id} (only needed if several are pending)",
            ]
            try:
                await self._telegram.send(chat_id, "\n".join(lines),
                                          parse_mode=None)
            except Exception as exc:  # noqa: BLE001 - the frame still went out
                logger.warning("desktop approval prompt not sent: %s", exc)
        try:
            await self._wait_for_answer(action, request_id, chat_id)
        finally:
            self.pending.pop(request_id, None)
        return action._result

    async def approve(
        self, request_id: str, *, by: Approver,
        scope: str | None = None, grant=None,
    ) -> bool:
        action = self.pending.get(request_id)
        if action is None:
            return False
        if (scope not in (None, SCOPE_ONCE)) or grant is not None:
            raise OnceOnly(
                "A desktop request can only be approved once — nothing is "
                "remembered. The app you picked is the lasting consent; "
                "approve this one with /approve.")
        self.people.require(by)
        return await super().approve(request_id, by=by, scope=None)

    async def deny_task(self, task_id: str, *, by: Approver) -> int:
        """Deny every pending prompt of one task — how a stop unwinds a step
        that is waiting on a person. Returns how many were denied."""
        hits = [rid for rid, a in list(self.pending.items())
                if a.task_id == task_id]
        n = 0
        for rid in hits:
            if await self.deny(rid, by=by):
                n += 1
        return n
