"""Who may touch which session — the one rule behind device scoping.

A *device token* (config/device_store.py) sees and manages only the sessions that
device owns. The operator's global token — and a daemon deliberately run with no
token at all — keeps full access. REST (web/server.py) and the WebSocket bridge
(web/ws_server.py) both ask THIS module, so the two doors cannot drift.

The rule, in four lines:

  * a session belongs to the device that brought it into existence;
  * a session nobody claimed belongs to the operator (Telegram, cron, anything
    that predates device scoping);
  * "not yours" is indistinguishable from "not there" to a device: the same 404,
    the same error frame — no oracle for session ids;
  * the operator is never restricted.

How a device brings a session into existence is :meth:`SessionAccess.admit`: it
may take an id that exists NOWHERE (no live session, no durable row) and that is
not in a namespace the daemon itself writes into — otherwise a device could send
to ``telegram:<chat id>`` before the operator's chat existed and own it.

What this is not: a sandbox. It scopes the API surface. The agent a device talks
to is the operator's agent, with the operator's tools (``lcm_grep`` searches every
session), so a device can still ask its own session's model about other sessions.
docs/contracts/device-scoping.md lists what is and is not covered.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from prometheus.memory.session_kind import MACHINE_SESSION_IDS

#: Namespaces the daemon writes into on its own account: the gateway adapters
#: (``gateway.config.Platform``) and the managed coding runs (``coding:<task id>``).
#: A device cannot claim an id here, even one that does not exist yet.
#: ``tests/test_device_session_scoping_rest.py`` pins this to ``Platform``.
DAEMON_GATEWAYS = frozenset({"telegram", "slack", "discord", "cli", "api", "coding"})


@dataclass(frozen=True)
class Scope:
    """Whose sessions a caller may touch. ``device_id is None`` = the operator."""

    device_id: str | None

    @property
    def unrestricted(self) -> bool:
        return self.device_id is None


OPERATOR = Scope(None)
#: Authentication is on, yet the caller resolved to no identity: owns nothing.
#: Fail closed — an unidentified caller must never read as the operator.
NOBODY = Scope("")


def scope_for(identity: Any, *, auth_required: bool) -> Scope:
    """The scope of a caller. *identity* is ``request.state.device_identity`` (REST) or the
    socket's recorded ``DeviceIdentity`` (WS); None when auth is off or nothing was recorded."""
    if not auth_required:
        return OPERATOR  # deliberately open: today's behaviour, unchanged
    if identity is None:
        return NOBODY
    # The master token and an OWNER device (the person's own: same-Mac pairing) are the operator.
    # Anything else is a device scoped to what it created. ``is_operator`` is DeviceIdentity's own
    # test; getattr keeps a bare identity from an older registry on the safe, narrower side.
    if getattr(identity, "is_operator", identity.is_global):
        return OPERATOR
    return Scope(identity.id)


def gateway_of(session_id: str) -> str:
    """The id's namespace — the text before the first colon, lowercased; "" for a bare id."""
    head, sep, _ = session_id.partition(":")
    return head.strip().lower() if sep else ""


def session_exists(session_mgr: Any, conversation_store: Any, session_id: str) -> bool:
    """True if the session is live in memory or has any durable row. A tombstoned session
    still has its rows, so it still exists; a PURGED one does not."""
    if session_mgr is not None and session_mgr.get(session_id) is not None:
        return True
    return conversation_store is not None and conversation_store.max_rowid(session_id) > 0


def event_session_id(event: dict[str, Any]) -> str | None:
    """The session a WS frame or a persisted signal is about, if it names one.

    Looks where a session id actually travels: ``payload.session_id`` (turn frames,
    promoted signals) and ``payload.payload.session_id`` (the generic ``sentinel_signal``
    wrapper nests the signal's own payload one level down). None for daemon-level frames
    (``dream_*``, ``memory_updated`` …), which belong to no conversation.
    """
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return None
    sid = payload.get("session_id")
    if isinstance(sid, str) and sid:
        return sid
    inner = payload.get("payload")
    if isinstance(inner, dict):
        sid = inner.get("session_id")
        if isinstance(sid, str) and sid:
            return sid
    return None


class SessionAccess:
    """The ownership checks, bound to a device registry and an existence test.

    *devices* returns the DeviceStore (called each time: the REST layer may create it lazily);
    *exists* answers "is there already a session with this id, anywhere?".
    """

    def __init__(self, devices: Callable[[], Any], exists: Callable[[str], bool]) -> None:
        self._devices = devices
        self._exists = exists

    def owns(self, scope: Scope, session_id: str) -> bool:
        """May *scope* see and manage *session_id*? Pure: never claims anything."""
        if scope.unrestricted:
            return True
        # A session id off the wire is whatever the client's JSON said it was.
        if not scope.device_id or not isinstance(session_id, str) or not session_id:
            return False
        return bool(self._devices().session_owner(session_id) == scope.device_id)

    def may_use(self, scope: Scope, session_id: str) -> bool:
        """Does *scope* own *session_id*, or could it bring it into existence? Pure."""
        if self.owns(scope, session_id):
            return True
        return self._claimable(scope, session_id)

    def admit(self, scope: Scope, session_id: str) -> bool:
        """Like :meth:`may_use`, and a claimable id becomes the device's. Call it where a
        device may legitimately CREATE a session — and only there."""
        if scope.unrestricted:
            return True
        if self.owns(scope, session_id):
            return True
        if not self._claimable(scope, session_id):
            return False
        assert scope.device_id  # _claimable is False for NOBODY
        return bool(self._devices().claim_session(session_id, scope.device_id))

    def mint(self, scope: Scope, session_id: str) -> bool:
        """Claim an id the SERVER just generated (``<gateway>:<uuid4>``). It cannot collide with
        anything, so the namespace and existence checks of :meth:`admit` do not apply — a device
        may ask for any gateway label in a fresh id. Never use this for a client-chosen id."""
        if scope.unrestricted:
            return True
        if not scope.device_id or not isinstance(session_id, str):
            return False
        return bool(self._devices().claim_session(session_id, scope.device_id))

    def _claimable(self, scope: Scope, session_id: str) -> bool:
        if scope.unrestricted:
            return True
        if not scope.device_id or not isinstance(session_id, str) or not session_id:
            return False
        if self._devices().session_owner(session_id) is not None:
            return False  # somebody's — this device's was handled by owns()
        if gateway_of(session_id) in DAEMON_GATEWAYS or session_id in MACHINE_SESSION_IDS:
            return False
        return not self._exists(session_id)

    def owned_ids(self, scope: Scope) -> set[str] | None:
        """Every session id *scope* may see, or None for the operator (no restriction)."""
        if scope.unrestricted:
            return None
        if not scope.device_id:
            return set()
        owned: set[str] = self._devices().owned_session_ids(scope.device_id)
        return owned

    def frame_visible(self, scope: Scope, event: dict[str, Any]) -> bool:
        """May *scope* receive this frame/event? One that names a session goes only to that
        session's owner; one that names none is daemon-level and goes to everyone."""
        if scope.unrestricted:
            return True
        sid = event_session_id(event)
        return True if sid is None else self.owns(scope, sid)
