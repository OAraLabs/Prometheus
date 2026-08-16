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
    """A tool call waiting for user approval.

    ``grant_file_path`` / ``grant_command`` carry the STRUCTURED target of the
    pending call (set by SecurityGate.request_approval) so a scoped approval
    (/approve session|always) can derive a Grant without parsing the free-text
    description.
    """

    request_id: str
    tool_name: str
    description: str
    created_at: float = field(default_factory=time.time)
    grant_file_path: str | None = None
    grant_command: str | None = None
    _event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _result: ApprovalResult = ApprovalResult.TIMEOUT


def derive_grant(
    action: PendingAction, root: str | None = None, *, widen: bool = False
):
    """Build the Grant a scoped approval would remember, or None.

    - ``root`` given: grant writes under that path (file tools only).
    - else a file target: grant EXACTLY that file by default; ``widen=True``
      grants its parent directory instead (the opt-in directory semantic).
    - else a command target: grant that exact command (single invocation).
    - else: **None** — extent is unknown, so no grant can be described and
      none is created. See the rule-4 comment below.

    Returns ``Grant | None``. Callers MUST handle None: it means "approve
    once, remember nothing", not "grant everything".
    """
    from pathlib import Path as _Path

    from prometheus.permissions.checker import Grant

    if root:
        return Grant(
            kind="path_prefix",
            value=str(_Path(root).expanduser().resolve()),
            tool_name=action.tool_name,
            request_id=action.request_id,
        )
    if action.grant_file_path:
        target = _Path(action.grant_file_path).expanduser().resolve()
        # SPRINT-CONSENT Phase 1 — THE DEFAULT NARROWS. This returned
        # ``target.parent`` unconditionally, so approving ONE file in $HOME
        # granted write_file across ALL of $HOME, permanently, while the
        # prompt showed only the one path. The directory semantic is
        # deliberate and useful; it is not defensible as a SILENT default.
        # It is now opt-in (``/approve always here``).
        value = str(target.parent) if widen else str(target)
        return Grant(
            kind="path_prefix",
            value=value,
            tool_name=action.tool_name,
            request_id=action.request_id,
        )
    if action.grant_command:
        return Grant(
            kind="command_prefix",
            value=action.grant_command,
            tool_name="bash",
            request_id=action.request_id,
        )
    # SPRINT-CONSENT Phase 1 — RULE 4 PRODUCES NO GRANT.
    #
    # This used to return ``Grant(kind="tool", value="")`` — a grant whose
    # matches() never looks at a path, i.e. the WIDEST grant in the system,
    # produced by the case carrying the LEAST information (a strict-mode
    # prompt whose reason names no target). That inversion should not exist.
    #
    # If extent cannot be determined it cannot be described, and consent that
    # cannot be described cannot be informed. The caller approves ONCE and
    # says why nothing was remembered.
    return None


def prospective_extents(action: PendingAction) -> dict[str, str]:
    """What each scope verb WOULD grant, described in operator terms.

    SPRINT-CONSENT Phase 1 / 0e — THE ONE COMPUTED EXTENT.

    The prompt used to state duration ("remember permanently") and never
    extent, while ``derive_grant`` quietly widened a single file to its whole
    parent directory. The operator learned the true scope only afterwards, in
    the response, by which time the grant existed and could not be revoked.
    That is consent obtained under a false description.

    Both surfaces render from THIS function — Telegram formats it into prose,
    Beacon ships it as a field in ``/api/approvals``. Neither re-derives it,
    and it calls the same ``derive_grant`` the approval path will call, so the
    description and the grant cannot drift (Standing-Principles §17).

    Keys are the scope verbs; a verb is ABSENT when it would create no grant.
    """
    out: dict[str, str] = {}
    exact = derive_grant(action)
    if exact is not None:
        exact.scope = "until_restart"
        out["until-restart"] = exact.describe()
        always = derive_grant(action)
        always.scope = "persistent"
        out["always"] = always.describe()
        if action.grant_file_path:
            here = derive_grant(action, widen=True)
            here.scope = "persistent"
            out["always here"] = here.describe()
    return out


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
        grant_file_path: str | None = None,
        grant_command: str | None = None,
    ) -> ApprovalResult:
        """Queue an action for user approval.

        Sends a Telegram message and waits for /approve or /deny response.
        Returns APPROVED, DENIED, or TIMEOUT.

        ``grant_file_path`` / ``grant_command`` are the structured target of
        the pending call — used to derive a remembered grant when the operator
        answers with /approve session|always.
        """
        request_id = uuid4().hex[:8]
        action = PendingAction(
            request_id=request_id,
            tool_name=tool_name,
            description=description,
            grant_file_path=grant_file_path,
            grant_command=grant_command,
        )
        self.pending[request_id] = action

        # Send notification via Telegram
        target_chat = chat_id or self._default_chat_id
        if self._telegram and target_chat:
            # The bare forms lead: when this is the only pending request the
            # id is optional, and copying an 8-hex id back is precisely the
            # friction that made operators type `/approve` alone and get
            # usage text instead of an approval.
            # Each scope verb is offered WITH the extent it would grant.
            # "session" is gone: there is one gate per process and _grants is
            # never cleared, so it never meant a session — "until restart" is
            # the true boundary and states it at consent time.
            extents = prospective_extents(action)
            if extents:
                offers = "".join(
                    f"/approve {verb} — grants {what}\n"
                    for verb, what in extents.items()
                )
            else:
                offers = (
                    "/approve until-restart, /approve always — NOT OFFERED "
                    "here: this request carries no specific target, so the "
                    "extent of a remembered grant cannot be described. "
                    "Approve once or deny.\n"
                )
            msg = (
                f"Permission requested:\n"
                f"Tool: {tool_name}\n"
                f"Action: {description}\n\n"
                f"/approve — approve this ONCE (or /deny)\n"
                f"{offers}"
                f"/approve all — approve everything pending, once each\n\n"
                f"Expires in {self._timeout // 60} min if unanswered.\n"
                f"id: {request_id} (only needed if several are pending)"
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
