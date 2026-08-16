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


#: SPRINT-CONSENT Phase 4. Was 300 (5 minutes), and the field incident is the
#: argument: a request raised at 15:52 was popped at 15:57 EXACTLY, and the
#: operator's `/approve always` arrived after the window had closed. It read
#: as a broken feature and cost hours to diagnose as an expiry.
#:
#: 5 minutes is shorter than a human's response latency to a phone
#: notification — it assumes the operator is already looking at the chat.
#: 30 minutes is chosen because it spans a meeting, a commute or a meal, and
#: the cost of the longer window is bounded: a pending request holds ONE tool
#: call open, it is visible the whole time in `/pending` and `/api/approvals`,
#: and the loop's own iteration cap still ends the turn. An unanswered
#: request now expires with a NOTIFICATION rather than silence, so a long
#: window can no longer be mistaken for a lost one.
#:
#: MUST equal the template value (config/prometheus.yaml.default). The config
#: drift guard compares key PRESENCE and cannot see a value divergence — that
#: is exactly how live `max_tool_iterations: 50` sat against a template
#: saying 25.
DEFAULT_APPROVAL_TIMEOUT_SECONDS = 1800


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
        # ⚠ resolve() RAISES on a symlink loop (RuntimeError) and can raise
        # OSError on a broken or unresponsive mount. That raise pre-dates this
        # sprint — but derive_grant used to run only AFTER the operator
        # answered, so it broke an approval; it now also runs at PROMPT time,
        # where an uncaught raise means the prompt is never sent and the
        # operator never learns a permission was requested. Failing closed to
        # the unresolved literal keeps the request visible; the grant is then
        # narrower than a resolved one, never wider, so the failure direction
        # is safe.
        try:
            target = _Path(action.grant_file_path).expanduser().resolve()
        except (OSError, RuntimeError):
            target = _Path(action.grant_file_path).expanduser()
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


# ---------------------------------------------------------------------------
# THE APPROVE-SCOPE VOCABULARY — one definition, every surface derives from it
# ---------------------------------------------------------------------------
#
# ⚠ THIS EXISTS BECAUSE THE VOCABULARY DRIFTED THE DAY IT WAS INTRODUCED.
# #232 added "until-restart" and "always here" to the chat parser and to the
# prompt, and left `server.py`'s validator asserting the OLD three verbs. The
# REST surface then 400'd the exact verbs the prompt was offering, so Beacon
# could not opt into the directory grant at all. Found by a live outcome
# check, not by any test — the gateway-parity guard covers command
# REGISTRATION, not payload VOCABULARY, so it looked like it protected surface
# parity and protected a subset of it.
#
# A guard asserting "these two lists match" was the obvious fix and is the
# worse one: it keeps two lists and adds a third thing to maintain. A default
# beats a check (Standing-Principles §13). There is now ONE definition, and
# the parser, the REST validator and the prompt all derive from it, so they
# cannot disagree — there is no second list to drift.

SCOPE_ONCE = "once"

#: Scopes that create a remembered grant. Order is the order the prompt
#: offers them: narrowest duration first.
GRANT_SCOPES: tuple[str, ...] = ("until-restart", "always")

#: Suffix opting in to the directory-widening (was the silent default pre-#232).
WIDEN_SUFFIX = "here"

#: Retired spellings kept working so muscle memory does not break. "session"
#: is the pre-#232 name; it never meant a session (one gate per process,
#: _grants never cleared) which is why it is an alias and not an offer.
_SCOPE_ALIASES: dict[str, str] = {
    "session": "until-restart",
    "until_restart": "until-restart",
    "session here": "until-restart here",
    "until_restart here": "until-restart here",
}


def approve_verbs() -> tuple[str, ...]:
    """Every scope verb the system accepts, canonical spellings only."""
    return (SCOPE_ONCE, *GRANT_SCOPES,
            *(f"{g} {WIDEN_SUFFIX}" for g in GRANT_SCOPES))


def normalise_scope(raw: str | None) -> str | None:
    """Canonical scope verb, or None if it is not one.

    THE single source of truth. Callers must treat None as "reject" — never
    as a default, which is how the widest grant in the system used to be
    produced from the least information.
    """
    s = " ".join((raw or "").split()).lower()
    if not s:
        return SCOPE_ONCE
    s = _SCOPE_ALIASES.get(s, s)
    return s if s in approve_verbs() else None


def scope_is_persistent(scope: str) -> bool:
    """Does this verb create a grant that outlives the process?"""
    return scope.split()[0] == "always"


def scope_widens(scope: str) -> bool:
    """Does this verb opt in to the directory grant?"""
    return scope.endswith(f" {WIDEN_SUFFIX}")


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
    if derive_grant(action) is None:
        return out
    # Derived from GRANT_SCOPES, never hand-listed — a verb added there is
    # offered here, accepted by the parser, and accepted by REST, with no
    # second list to update.
    for verb in approve_verbs():
        if verb == SCOPE_ONCE:
            continue
        if scope_widens(verb) and not action.grant_file_path:
            continue  # nothing to widen for a command-scoped request
        g = derive_grant(action, widen=scope_widens(verb))
        if g is None:
            continue
        g.scope = "persistent" if scope_is_persistent(verb) else "until_restart"
        out[verb] = g.describe()
    return out


def _humanise_window(seconds) -> str:
    """'30 min' / '45 s'. Never '0.0 min'.

    ``self._timeout // 60`` renders 0.0 for any sub-minute window — which a
    test with a short timeout surfaced, and which an operator with a 30-second
    window would have read as "expires immediately".
    """
    seconds = float(seconds or 0)
    if seconds >= 60:
        return f"{int(seconds // 60)} min"
    return f"{int(seconds)} s"


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
        timeout_seconds: int = DEFAULT_APPROVAL_TIMEOUT_SECONDS,
        default_chat_id: int | None = None,
    ) -> None:
        self._telegram = telegram_adapter
        self._timeout = timeout_seconds
        self._default_chat_id = default_chat_id
        self.pending: dict[str, PendingAction] = {}

    @property
    def timeout_seconds(self) -> float:
        """The window actually in force on THIS queue, in seconds.

        Public because the alternative is a second surface reading
        ``DEFAULT_APPROVAL_TIMEOUT_SECONDS`` and hoping it matches. It often
        would not: the constant is only the default, while ``self._timeout``
        is what ``asyncio.wait_for`` counts and what the Telegram prose
        renders. A config that sets a different window would leave the
        constant-reading surface confidently wrong.
        """
        return self._timeout

    def expires_at(self, action: PendingAction) -> float:
        """Epoch seconds at which this request's window closes.

        The daemon holds this truth, so the daemon states it. A client that
        computes ``created_at + 1800`` itself is a second surface deriving a
        value the first one already has — the defect class SPRINT-CONSENT
        removed for grant extents (Standing-Principles §17), reintroduced in
        a new place.
        """
        return action.created_at + self._timeout

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
                f"Expires in {_humanise_window(self._timeout)} if unanswered.\n"
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
            # SPRINT-CONSENT Phase 4 — EXPIRY MUST NOTIFY.
            #
            # This popped the request in the `finally` below and said nothing.
            # The operator's only clue was a later, unrelated-looking "No
            # pending approval requests", which is why a 300-second expiry was
            # reported as a broken `/approve always` and cost hours to
            # diagnose. Silence on a security surface is the defect.
            from prometheus.permissions.audit import AuditDecision

            self._audit_resolution(action, AuditDecision.CONFIRM_TIMEOUT)
            await self._notify_expiry(action, target_chat)

        # Clean up
        self.pending.pop(request_id, None)
        return action._result

    async def _notify_expiry(self, action: PendingAction, chat_id) -> None:
        """Tell the operator the window closed, and what it was for."""
        if not (self._telegram and chat_id):
            return
        target = action.grant_file_path or action.grant_command or "(no target)"
        window = _humanise_window(self._timeout)
        msg = (
            f"Approval EXPIRED — not approved.\n"
            f"Tool: {action.tool_name}\n"
            f"Target: {target}\n"
            f"Request {action.request_id} went unanswered for {window} and "
            f"has been withdrawn. The action did NOT run.\n"
            f"Ask again if you still want it."
        )
        try:
            await self._telegram.send(chat_id, msg, parse_mode=None)
        except Exception as exc:  # pragma: no cover - notify is best-effort
            logger.warning("Failed to send approval-expiry notice: %s", exc)

    def _audit_resolution(
        self,
        action: PendingAction,
        decision,
        *,
        scope: str | None = None,
        grant=None,
    ) -> None:
        """Write the resolution row. SPRINT-CONSENT Phase 3.

        THE MISSING WRITE. ``AuditDecision.CONFIRM_APPROVED`` and
        ``CONFIRM_REJECTED`` were defined in ``audit.py`` and referenced
        NOWHERE in ``src/``. The request half wrote (``checker.py`` logs
        CONFIRM_PENDING three times); the resolution half wrote nothing. So
        the accountability record held 24,048 rows across four months —
        23,858 allow, 79 confirm_pending, 111 deny — and **zero resolutions
        against at least six demonstrated approvals.** It could say what was
        asked and never what was decided.

        ``scope`` is what permanently closes the "never invoked vs invoked
        and dropped" ambiguity: an ``always`` that writes no grant is now
        visible here instead of being indistinguishable from silence. That
        ambiguity previously cost a live probe to resolve.

        Best-effort: telemetry must never turn a resolved approval into an
        exception. The queue reaches the logger through the gate rather than
        holding its own, so there is one audit store, not two.
        """
        gate = getattr(self, "_security_gate", None)
        audit = getattr(gate, "_audit", None) if gate is not None else None
        if audit is None:
            return
        bits = [f"request={action.request_id}", f"scope={scope or 'once'}"]
        if grant is not None:
            bits.append(f"grant={grant.grant_id} ({grant.describe()})")
        else:
            bits.append("grant=none")
        target = action.grant_file_path or action.grant_command or ""
        try:
            audit.log(
                tool_name=action.tool_name,
                decision=decision,
                trust_level=getattr(gate, "_mode_trust_level", lambda: 0)(),
                reason=f"{decision.value}: " + ", ".join(bits),
                tool_input=target or None,
            )
        except Exception:  # pragma: no cover - audit must not mask a decision
            logger.debug("approval audit write failed", exc_info=True)

    async def approve(
        self, request_id: str, *, scope: str | None = None, grant=None
    ) -> bool:
        """Approve a pending action. Returns True if found and approved."""
        from prometheus.permissions.audit import AuditDecision

        action = self.pending.get(request_id)
        if action is None:
            return False
        action._result = ApprovalResult.APPROVED
        action._event.set()
        self._audit_resolution(
            action, AuditDecision.CONFIRM_APPROVED, scope=scope, grant=grant
        )
        return True

    async def deny(self, request_id: str) -> bool:
        """Deny a pending action. Returns True if found and denied."""
        from prometheus.permissions.audit import AuditDecision

        action = self.pending.get(request_id)
        if action is None:
            return False
        action._result = ApprovalResult.DENIED
        action._event.set()
        self._audit_resolution(action, AuditDecision.CONFIRM_REJECTED)
        return True

    def list_pending(self) -> list[PendingAction]:
        """Return all pending approval requests."""
        return list(self.pending.values())
