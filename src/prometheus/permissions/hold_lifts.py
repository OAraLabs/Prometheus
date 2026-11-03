"""Lifting a device's hold is a person's decision: the consent prompt for it.

A hold (``PUT /api/devices/{id}/profile``, config/device_store.py) keeps a device to a profile, so a
server-to-server caller beside the owner's own assistant can never run shell, file or write tools. SETTING
one only takes tools away, so the API token may do it. LIFTING or CHANGING one widens what the device may
run, and the token is no proof of a person: it sits in a file a model with a shell can read, which is the
door computer use already closed (computer/door.py, W3). So a lift asks, through this channel:

* the request is a consent prompt on the owner's approval surfaces, beside tool and desktop prompts
  (Telegram ``/approve``, Beacon's approvals, the ``approval_pending`` frame), sent to the owner's own chat;
* the hold stands until a PERSON approves it: the Telegram owner, or a Beacon device a person marked
  (:class:`~prometheus.computer.door.PersonCheck`), never the API token, an open daemon, or the held device
  itself (a held device is never a person, door.py);
* once only, and never part of "approve all": each lift is read and answered on its own;
* a denial, an expiry or a restart leaves the hold where it was.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from prometheus.computer.door import OnceOnly
from prometheus.permissions.approval_queue import (
    DEFAULT_APPROVAL_TIMEOUT_SECONDS,
    SCOPE_ONCE,
    ApprovalQueue,
    ApprovalResult,
    PendingAction,
    _humanise_window,
)
from prometheus.permissions.approver import Approver

logger = logging.getLogger(__name__)

#: The pending request's tool name: what the approval surfaces show as the action.
TOOL_NAME = "device_hold"


class _NoAuditStore:
    """Where no gate is wired (a bare web app): a lift is still decided; its resolution row has nowhere to go."""

    _audit = None


class HoldLiftChannel(ApprovalQueue):
    """See the module docstring."""

    def __init__(
        self,
        *,
        people: Any,
        devices: Callable[[], Any] | None = None,
        security_gate: Any = None,
        telegram_adapter: Any = None,
        owner_chat_id: int | None = None,
        timeout_seconds: int = DEFAULT_APPROVAL_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(security_gate=security_gate if security_gate is not None else _NoAuditStore(),
                         telegram_adapter=telegram_adapter, timeout_seconds=timeout_seconds,
                         default_chat_id=owner_chat_id)
        self.people = people
        #: Where the hold lives: the web launcher's ONE device store, bound once it exists (bind_device_store).
        self._devices = devices
        #: The waits, held so the loop's weak reference cannot drop one before it decides.
        self._waits: set[asyncio.Task] = set()

    def bind_device_store(self, store: Any) -> None:
        """The web launcher's ONE device store (the REST middleware's): the holds and the person marks it holds."""
        if store is not None:
            self._devices = lambda: store
            self.people.bind_device_store(store)

    def request_lift(self, device_id: str, device_name: str, *, current: str, to: str | None,
                     by: Approver) -> str:
        """Ask a person to lift *device_id*'s hold (``to`` None) or change it to *to*. Returns the request id at
        once; the decision lands later, and only an approval moves the hold."""
        request_id = uuid4().hex[:8]
        change = f"to profile {to!r}" if to else "so it runs the instance's default profile"
        description = (f"Lift the hold on device {device_name!r} ({device_id}), held to profile {current!r}, "
                       f"{change}. Asked by {by.label}.")
        action = PendingAction(
            request_id=request_id, tool_name=TOOL_NAME, description=description,
            arguments={"device_id": device_id, "from": current, "to": to},
            once_only=True, hold_device_id=device_id, notify_chat=self._default_chat_id,
        )
        self._register(action)
        task = asyncio.get_running_loop().create_task(self._decide(action, to=to, asked_by=by))
        self._waits.add(task)
        task.add_done_callback(self._waits.discard)
        return request_id

    async def _decide(self, action: PendingAction, *, to: str | None, asked_by: Approver) -> None:
        await self._emit("approval_pending", self.serialize_pending(action))
        chat = self._default_chat_id
        if self._telegram and chat:
            lines = [action.description, "",
                     "/approve — lift it (once; nothing is remembered), or /deny to keep the hold",
                     "", f"Expires in {_humanise_window(self._timeout)} if unanswered; the hold stays.",
                     f"id: {action.request_id} (only needed if several are pending)"]
            try:
                await self._telegram.send(chat, "\n".join(lines), parse_mode=None)
            except Exception as exc:  # noqa: BLE001 - the frame still went out
                logger.warning("hold-lift prompt not sent: %s", exc)
        try:
            await self._wait_for_answer(action, action.request_id, chat)
        finally:
            self.pending.pop(action.request_id, None)
        if action._result != ApprovalResult.APPROVED:
            logger.info("hold on %s kept: lift request %s ended %s", action.hold_device_id,
                        action.request_id, getattr(action._result, "value", action._result))
            return
        approver = getattr(action, "_approved_by", None)
        label = approver.label if isinstance(approver, Approver) else "a person"
        store = self._devices() if self._devices is not None else None
        if store is None or not store.set_profile(action.hold_device_id, to,
                                                  by=f"{label} (asked by {asked_by.label})"):
            logger.warning("approved hold lift %s found no live device %s", action.request_id,
                           action.hold_device_id)
            return
        logger.info("hold on %s %s by %s", action.hold_device_id, f"changed to {to!r}" if to else "lifted", label)

    async def approve(self, request_id: str, *, by: Approver, scope: str | None = None, grant=None) -> bool:
        action = self.pending.get(request_id)
        if action is None:
            return False
        if (scope not in (None, SCOPE_ONCE)) or grant is not None:
            raise OnceOnly("A hold lift can only be approved once: nothing is remembered.")
        self.people.require(by)
        action._approved_by = by  # type: ignore[attr-defined]
        return await super().approve(request_id, by=by, scope=None)
