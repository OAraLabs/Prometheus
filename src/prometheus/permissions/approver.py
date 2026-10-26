"""Approver — WHO answered an approval. W4, step 1 of 3: RECORD.

``docs/design/approver-credential.md``. An approval could be answered with the
daemon's API token on every surface, and no surface recorded who answered: the
audit table has carried a ``user_id`` column all along and the resolution row
never filled it. So "who approved this?" had no answer, and the later steps —
warn when an answer did not come from a person, then refuse it — had nothing
to stand on.

This step RECORDS and refuses nothing. Every surface names its caller:

    REST, global API token    global-token
    REST, a device token      device-token:<device id>
    REST, no token (open)     open
    Telegram / Slack / Discord  <platform>:<the SENDER's user id>
                                (the person, not the chat they wrote in)
    in-process (a script)     in-process:<name>

``ApprovalQueue.approve``/``deny`` take it as a REQUIRED keyword, so a new
surface cannot reach the queue without saying who is answering. A sender a
gateway cannot name is recorded as ``unknown`` — honestly, and still answered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Collection
from typing import Any

#: The kinds. For REST, the kind is the CREDENTIAL that answered (a device's
#: token, the global token, or none); for a chat gateway it is the platform of
#: the person who sent the command. Only a device token and the chat platforms
#: can ever be a person; the warn step (not this one) decides which are.
DEVICE = "device-token"
GLOBAL_TOKEN = "global-token"
OPEN = "open"
TELEGRAM = "telegram"
SLACK = "slack"
DISCORD = "discord"
IN_PROCESS = "in-process"

UNKNOWN_ID = "unknown"

#: The chat gateways: their answers come from a person in a chat, and only the gateway's owner may give one.
CHAT_KINDS = frozenset({TELEGRAM, SLACK, DISCORD})


@dataclass(frozen=True)
class Approver:
    """Who answered. ``kind`` from the list above; ``id`` is stable, ``name``
    is for people reading the record and may change."""

    kind: str
    id: str
    name: str = ""
    #: A chat gateway's answer: True when the sender is the gateway's owner, False when not. None for every other
    #: surface (REST, in-process), whose own rules decide. The shared /approve and /deny core refuses False.
    owner: bool | None = field(default=None, compare=False)

    @property
    def not_the_owner(self) -> bool:
        """A chat answer that is not known to be the owner's. ``owner=None`` on a chat kind means the gateway never
        asked, so it is refused too: the approval queue's backstop under the /approve and /deny core."""
        return self.kind in CHAT_KINDS and self.owner is not True

    @property
    def label(self) -> str:
        """The audit row's ``user_id``: ``kind:id``, or the bare kind for the
        two credentials that have no per-caller id."""
        if self.kind in (GLOBAL_TOKEN, OPEN):
            return self.kind
        return f"{self.kind}:{self.id}"

    def to_record(self) -> dict[str, str]:
        """Plain data for a signal payload, a WS frame and prometheus.yaml —
        never the dataclass, which neither ``json.dumps`` nor ``yaml.dump``
        writes as plain data."""
        return {"kind": self.kind, "id": self.id, "name": self.name}


def from_device_identity(identity: Any) -> Approver:
    """REST: the identity the auth middleware put on ``request.state``.

    None means the daemon runs with no API token (open mode): there is no
    credential at all, and that is what gets recorded.
    """
    if identity is None:
        return Approver(OPEN, "none", "no API token configured")
    if getattr(identity, "is_global", False):
        return Approver(GLOBAL_TOKEN, "global", "API token")
    return Approver(DEVICE, str(identity.id), str(getattr(identity, "name", "")))


def from_request(request: Any) -> Approver:
    state = getattr(request, "state", None)
    return from_device_identity(getattr(state, "device_identity", None))


def _owned(owners: Collection[str] | None, user_id: str | None) -> bool | None:
    """Whether *user_id* is the gateway's owner; None when the caller passed no owner list (record only)."""
    if owners is None:
        return None
    return user_id is not None and user_id in owners


def telegram(user: Any, *, owners: Collection[str] | None = None) -> Approver:
    """The sender (``update.effective_user``), not the chat. *owners*: the gateway's owner ids."""
    if user is None or getattr(user, "id", None) is None:
        return Approver(TELEGRAM, UNKNOWN_ID, owner=_owned(owners, None))
    name = getattr(user, "username", None) or getattr(user, "full_name", None)
    return Approver(TELEGRAM, str(user.id), str(name or ""), owner=_owned(owners, str(user.id)))


def slack(command: Any, *, owners: Collection[str] | None = None) -> Approver:
    """The sender of a slash command (its payload's ``user_id``)."""
    get = command.get if hasattr(command, "get") else (lambda k: None)
    user_id = get("user_id")
    if not user_id:
        return Approver(SLACK, UNKNOWN_ID, owner=_owned(owners, None))
    return Approver(SLACK, str(user_id), str(get("user_name") or ""), owner=_owned(owners, str(user_id)))


def discord(interaction: Any, *, owners: Collection[str] | None = None) -> Approver:
    """The user who invoked the interaction."""
    user = getattr(interaction, "user", None)
    user_id = getattr(user, "id", None)
    if user_id is None:
        return Approver(DISCORD, UNKNOWN_ID, owner=_owned(owners, None))
    return Approver(DISCORD, str(user_id), str(getattr(user, "name", "") or ""), owner=_owned(owners, str(user_id)))


def in_process(name: str) -> Approver:
    """A caller inside the process that owns the queue (a probe script).

    It can only ever answer a queue it built itself: the daemon's queue lives
    in the daemon's memory.
    """
    return Approver(IN_PROCESS, name, name)
