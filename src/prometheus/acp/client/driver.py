"""Driving one outside agent over ACP — prompts out, permissions through the gate.

The driver's shape: initialize (v2 only), session/new, session/prompt,
session/cancel; the agent's ``session/update`` notifications stream to an
observer; the agent's ``session/request_permission`` requests go through
THE SAME GATE as Prometheus's own tool calls — exactly what Prometheus's
own calls get (D13 as written):

- a gate ALLOW (a grant covers it, an allowlisted read, …) answers the
  agent's ``allow_once`` option DIRECTLY, no queue — audited like any
  gate-allowed call, with a row naming the outside agent it answered;
- a gate APPROVE (needs confirmation) goes to the owner's approval queue,
  approve-once; anything the gate escalates goes to Beacon through the
  queue, never answered here;
- a DENY (or a timeout, no gate, no ``allow_once`` option, or a queue
  error) answers the agent with a rejection — fail closed, the agent
  MUST NOT proceed;
- the driver never answers with an ``allow_always`` option: a driven
  agent's consent is per-call, like a spawned worker's.

Hard-disabled for machine/benchmark sessions: a driven agent is a real
conversation with real side effects, never fixture material.

Source: novel code for Prometheus, 2026-11-25.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from prometheus.acp import types
from prometheus.acp.client.connection import AcpClientConnection
from prometheus.acp.client.registry import OutsideAgent
from prometheus.acp.framing import UNSUPPORTED_PROTOCOL_VERSION, FrameError
from prometheus.acp.negotiation import PROTOCOL_VERSION
from prometheus.memory.session_kind import is_machine_session

log = logging.getLogger(__name__)

UpdateObserver = Callable[[str, dict[str, Any]], Awaitable[None]]


class DriverRefused(Exception):
    """The driver will not run this request (machine session, no gate, …)."""


class OutsideAgentDriver:
    """One driven session with one outside agent.

    ``gate`` is a SecurityGate-like object (``evaluate``); ``approval_queue``
    is the owner's queue the APPROVE-tier requests go to. A driver with no
    gate refuses everything; a gate ALLOW is answered directly (audited),
    but an APPROVE with no queue to ask refuses — fail closed.
    """

    def __init__(
        self,
        agent: OutsideAgent,
        *,
        gate: Any = None,
        approval_queue: Any = None,
        observer: UpdateObserver | None = None,
        cwd: Path | None = None,
    ) -> None:
        self._agent = agent
        self._gate = gate
        self._queue = approval_queue
        self._observer = observer
        self._cwd = cwd or Path.cwd()
        self._conn: AcpClientConnection | None = None
        self._session_id: str | None = None
        self._proc: asyncio.subprocess.Process | None = None

    @property
    def session_id(self) -> str | None:
        return self._session_id

    async def start(self, *, session_id: str | None = None) -> str:
        """Launch the agent, initialize (v2 only), open a session. Returns
        the ACP sessionId. ``session_id`` names the PROMETHEUS conversation
        this drive belongs to (for the machine-session refusal), not the
        ACP session's id."""
        if session_id is not None and is_machine_session(session_id):
            raise DriverRefused("a driven outside agent never runs in a machine/benchmark session")
        self._proc = await asyncio.create_subprocess_exec(
            *self._agent.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            cwd=str(self._cwd),
        )
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._conn = AcpClientConnection(
            self._proc.stdout, self._proc.stdin,
            permission_handler=self._answer_permission,
            notification_handler=self._on_notification,
        )
        await self._conn.start()
        result = await self._conn.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "info": {"name": "prometheus", "version": "0"},
            "capabilities": {},
        })
        resp = types.InitializeResponse.model_validate(result)
        if resp.protocol_version != PROTOCOL_VERSION:
            await self.close()
            raise FrameError(UNSUPPORTED_PROTOCOL_VERSION,
                             f"agent answered protocol version {resp.protocol_version}; this client speaks v2 only")
        new = await self._conn.request("session/new", {"cwd": str(self._cwd)})
        self._session_id = types.NewSessionResponse.model_validate(new).session_id
        return self._session_id

    async def prompt(self, text: str, *, idempotency_key: str | None = None) -> Any:
        if self._conn is None or self._session_id is None:
            raise DriverRefused("driver not started")
        params: dict[str, Any] = {
            "sessionId": self._session_id,
            "prompt": [{"type": "text", "text": text}],
        }
        if idempotency_key is not None:
            params["idempotencyKey"] = idempotency_key
        return await self._conn.request("session/prompt", params, timeout=3600.0)

    async def cancel(self) -> None:
        if self._conn is not None and self._session_id is not None:
            await self._conn.notify("session/cancel", {"sessionId": self._session_id})

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
        if self._proc is not None:
            try:
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except (ProcessLookupError, asyncio.TimeoutError):
                try:
                    self._proc.kill()
                except ProcessLookupError:
                    pass
            self._proc = None

    async def _on_notification(self, method: str, params: dict[str, Any] | None) -> None:
        if method == "session/update" and self._observer is not None and params is not None:
            update = params.get("update") or {}
            kind = update.get("sessionUpdate", "")
            await self._observer(kind, update)

    async def _answer_permission(self, params: dict[str, Any] | None) -> dict[str, Any]:
        """The agent's permission request, through the SAME gate.

        Exactly what Prometheus's own tool calls get (D13 as written):
        a gate ALLOW (a grant covers it, an allowlisted read, …) answers the
        agent's ``allow_once`` option DIRECTLY — no queue, audited like any
        gate-allowed call, naming the outside agent. A gate APPROVE (needs
        confirmation) goes to the owner's queue, approve-once. A DENY, no
        gate, no ``allow_once`` option offered, a queue error or an unanswered
        escalation: the rejection option, or ``cancelled`` when the agent
        offered none. Fail closed.

        Every answer is a selection of one of the agent's OWN options (the
        schema's contract) — never an invented option id, and NEVER
        ``allow_always``: a driven agent's consent is per-call.
        """
        req = types.RequestPermissionRequest.model_validate(params or {})
        options = req.options
        reject = next((o for o in options if _option_kind(o) == "reject_once"), None)

        def _refuse() -> dict[str, Any]:
            if reject is not None:
                return {"outcome": {"outcome": "selected", "optionId": _option_id(reject)}}
            return {"outcome": {"outcome": "cancelled"}}

        if self._gate is None:
            return _refuse()
        tool_name, command = _subject_tool_and_command(req)
        try:
            decision = self._gate.evaluate(
                tool_name,
                command=command,
                is_read_only=False,          # an outside agent's ask is never assumed read-only
                origin="system",
            )
        except Exception:  # noqa: BLE001 - a gate that cannot decide refuses
            log.exception("acp driver: gate evaluation failed — refusing")
            return _refuse()
        if not decision.allowed and not decision.requires_confirmation:
            return _refuse()               # DENY tier: never proceeds
        allow = next((o for o in options if _option_kind(o) == "allow_once"), None)
        if allow is None:
            return _refuse()               # no per-call allow offered: nothing to consent to
        if decision.allowed:
            # Gate ALLOW: the same call Prometheus itself would run without a
            # prompt. The gate's evaluate() already wrote its ALLOW audit row;
            # this row names the OUTSIDE AGENT the allow answered, so the
            # record says who ran it, not just what was allowed.
            self._audit_agent_allow(tool_name, decision.reason, command)
            return {"outcome": {"outcome": "selected", "optionId": _option_id(allow)}}
        # Gate APPROVE: the owner's queue, approve-once — a driven agent's
        # consent is per-call, like a spawned worker's.
        if self._queue is None:
            return _refuse()
        from prometheus.permissions.approval_queue import ApprovalResult
        try:
            result = await self._queue.request_approval(
                tool_name,
                f"outside agent {self._agent.name!r}: {req.title}",
                grant_command=command,
                once_only=True,
            )
        except Exception:  # noqa: BLE001 - a broken queue refuses
            log.exception("acp driver: approval queue failed — refusing")
            return _refuse()
        if result != ApprovalResult.APPROVED:
            return _refuse()
        return {"outcome": {"outcome": "selected", "optionId": _option_id(allow)}}

    def _audit_agent_allow(self, tool_name: str, reason: str, command: str | None) -> None:
        """One audit row naming the outside agent a gate-ALLOW answered.

        Best-effort like every audit write: a logger that cannot write must
        not refuse a call the gate allowed. The gate's own ALLOW row (written
        by evaluate) carries the decision; this one carries the asker.
        """
        audit = getattr(self._gate, "_audit", None)
        if audit is None:
            return
        try:
            from prometheus.permissions.audit import AuditDecision
            from prometheus.permissions.modes import TrustLevel
            audit.log(
                tool_name=tool_name,
                decision=AuditDecision.ALLOW,
                trust_level=TrustLevel.AUTO,  # the gate allowed it without a person
                reason=f"outside agent {self._agent.name!r} answered allow_once: {reason}",
                tool_input=command,
                user_id=f"acp-agent:{self._agent.name}",
            )
        except Exception:  # noqa: BLE001 - audit is best-effort
            log.warning("acp driver: could not audit the agent allow", exc_info=True)


def _option_kind(option: Any) -> str:
    kind = getattr(option, "kind", None)
    return getattr(kind, "root", kind) if kind is not None else ""


def _option_id(option: Any) -> str:
    oid = getattr(option, "option_id", None)
    return getattr(oid, "root", oid) if oid is not None else ""


def _subject_tool_and_command(req: Any) -> tuple[str, str | None]:
    """The gate's (tool_name, command) for the request's subject.

    A command subject maps to ``bash`` with the command line; a tool-call
    subject maps to the tool call's title (the gate's own name matching is
    exact, so an unknown name evaluates under the gate's rules for an
    unnamed write — never assumed read-only, never auto-allowed by name).
    """
    subject = getattr(req, "subject", None)
    if subject is None:
        return (req.title or "outside_agent"), None
    root = getattr(subject, "root", subject)
    command = getattr(root, "command", None)
    if command is not None:
        return "bash", str(command)
    tool_call = getattr(root, "tool_call", None)
    if tool_call is not None:
        title = getattr(tool_call, "title", None)
        return str(title or req.title or "outside_agent"), None
    return (req.title or "outside_agent"), None