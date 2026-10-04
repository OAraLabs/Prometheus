"""Who may open the door, and the refusals it answers with (W3, design §5.1.1).

ONLY A PERSON
-------------
Will's ruling W3: only a person may start computer use — an allowed chat
user, or a Beacon device a person marked for computer use. NEVER the API
token. The token sits in a plaintext file that ``bash`` (always loaded) can
read, so any route that accepts it is reachable by a prompt-injected model,
and ``POST /api/devices`` mints a device token from it alone — so a device
counts only once a PERSON has marked it.

The answer is a property of the :class:`~prometheus.permissions.approver.Approver`
the surface resolved (W4's record step), checked here in one place:

======================  =================================================
``telegram:<uid>``      a person when ``<uid>`` is itself on the Telegram
                        allowlist (``allowed_chat_ids``). The chat being
                        allowed is not enough: a group has many members.
``device-token:<id>``   a person when that device is marked for computer
                        use (``DeviceStore.computer_allowed``).
``in-process:<name>``   yes — it can only answer a queue it built itself.
``global-token``        never.
``open``                never: no credential at all.
``slack`` / ``discord`` not in v1.1 — approvals reach only Telegram and
                        Beacon (W5); recorded as a parity gap.
======================  =================================================

WHAT IS NEVER REFUSED
---------------------
Stopping a task, and denying a desktop prompt. Both can only end something;
refusing them to a credential would make the safe direction the hard one.
"""

from __future__ import annotations

from typing import Any, Iterable

from prometheus.permissions import approver as _approver


class DoorRefused(Exception):
    """The door said no. ``status`` is the HTTP answer, ``code`` the stable
    machine word, ``str(exc)`` the sentence a person reads."""

    status = 400
    code = "refused"

    def as_dict(self) -> dict[str, Any]:
        return {"error": self.code, "detail": str(self)}


class NotAPerson(DoorRefused):
    status = 401
    code = "not_a_person"


class InToolContext(DoorRefused):
    """Raised by ``ComputerTaskRunner.start`` inside a tool call (W3)."""

    status = 403
    code = "in_tool_call"


class ComputerUseOff(DoorRefused):
    status = 404
    code = "computer_use_off"


class IntegrationDown(DoorRefused):
    status = 503
    code = "integration_down"


class TaskBusy(DoorRefused):
    status = 409
    code = "busy"


class OnceOnly(DoorRefused):
    """A desktop prompt was answered with a lasting scope."""

    status = 409
    code = "once_only"


class NeedsConsent(DoorRefused):
    """No binding covers this session and app: the person must pick the app,
    and the pick IS the consent. Carries what to ask with."""

    status = 409
    code = "needs_app"

    def __init__(self, message: str, *, options: list[str] | None = None,
                 sentence: str = "", app: str | None = None) -> None:
        super().__init__(message)
        self.options = list(options or [])
        self.sentence = sentence
        self.app = app

    def as_dict(self) -> dict[str, Any]:
        return {"error": self.code, "detail": str(self),
                "options": self.options, "describes": self.sentence,
                "app": self.app}


class PersonCheck:
    """Is this credential a person's? See the module table."""

    def __init__(
        self,
        *,
        device_store: Any = None,
        telegram_user_ids: Iterable[int | str] = (),
    ) -> None:
        self._devices = device_store
        self._telegram = {str(u) for u in telegram_user_ids}

    def bind_device_store(self, store: Any) -> None:
        """The web launcher's ONE device store (the REST middleware's), so a
        mark or a revocation is seen by both at once."""
        if store is not None:
            self._devices = store

    @property
    def devices(self) -> Any:
        """The device store the marks live in (None when there is none)."""
        return self._devices

    def check(self, by: _approver.Approver | None) -> tuple[bool, str]:
        if by is None:
            return False, "no credential was presented"
        kind = by.kind
        if kind == _approver.IN_PROCESS:
            return True, ""
        if kind == _approver.TELEGRAM:
            if by.id != _approver.UNKNOWN_ID and by.id in self._telegram:
                return True, ""
            return False, ("only a person whose own Telegram id is on the "
                           "allowlist may do this")
        if kind == _approver.DEVICE:
            store = self._devices
            try:
                marked = bool(store is not None
                              and store.computer_allowed(by.id))
            except Exception:  # noqa: BLE001 - an unreadable store refuses
                marked = False
            if marked:
                return True, ""
            return False, ("this device is not marked for computer use — mark "
                           "it from Telegram (/computer allow <device id>) or "
                           "from a device that is")
        if kind == _approver.GLOBAL_TOKEN:
            return False, ("the API token cannot do this: a model can read it. "
                           "Only a person may — from Telegram, or a Beacon "
                           "device marked for computer use")
        if kind == _approver.OPEN:
            return False, ("no API token is configured, so no caller can be "
                           "told apart from a model — only Telegram can do "
                           "this here")
        if kind in (_approver.SLACK, _approver.DISCORD):
            return False, ("computer use answers only on Telegram and Beacon "
                           "in v1.1")
        return False, f"{kind!r} is not a person's credential"

    def require(self, by: _approver.Approver | None) -> None:
        ok, why = self.check(by)
        if not ok:
            raise NotAPerson(f"Only a person may do this: {why}.")
