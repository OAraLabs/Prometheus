"""SecurityGate — permission checker wired into AgentLoop as permission_checker.

Sprint 4: implements the 4-level trust model from prometheus.yaml security config.
Sprint 11: adds audit logging + exfiltration detection.
Sprint TRUST-CONTEXT: ``origin`` parameter distinguishes user-initiated calls
(Telegram, CLI, Web — the user is in the loop and asked for this) from
background/automated calls (SENTINEL, GEPA, AutoDream, smoke-tests, cron —
no human sanction in the moment). User-initiated bash commands skip the
ExfiltrationDetector and the network/install approve-patterns; everything
else still applies for both origins (always-blocked patterns, denied_paths,
denied_commands, write_file workspace gate). The origin is derived from
``LoopContext.session_id`` per the convention at agent_loop.py:538.
Integrates with the permission_checker slot in LoopContext.
"""

from __future__ import annotations

import fnmatch
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prometheus.permissions.audit import AuditDecision, AuditLogger
from prometheus.permissions.exfiltration import ExfiltrationDetector
from prometheus.permissions.modes import PermissionMode, TrustLevel

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Blocked command patterns (applied before prometheus.yaml denied_commands)
# ---------------------------------------------------------------------------

_ALWAYS_BLOCKED_PATTERNS: list[str] = [
    r"rm\s+-rf\s+/",
    r"rm\s+-rf\s+~",
    r"rm\s+--no-preserve-root",
    r"mkfs\b",
    r"dd\s+if=.*of=/dev/",
    r"chmod\s+-R\s+777\s+/",
    r">\s*/dev/sda",
    r":(){ :|:& };:",  # fork bomb
]

# Tools that are always safe for read-only classification
_READONLY_TOOLS: frozenset[str] = frozenset(
    {"read_file", "grep", "glob", "bash_read"}
)

# Tools that qualify for APPROVE (level 1) by default
_APPROVE_TOOLS: frozenset[str] = frozenset(
    {"write_file", "edit_file"}
)

# Bash substrings that bump trust to APPROVE (network / destructive)
_APPROVE_BASH_PATTERNS: list[str] = [
    r"git\s+push",
    r"git\s+push\s+--force",
    r"\bcurl\b",
    r"\bwget\b",
    r"\bnc\b",
    r"\bssh\b",
    r"\bscp\b",
    r"\brsync\b",
    r"pip\s+install",
    r"npm\s+install",
]


# ---------------------------------------------------------------------------
# Origin classification
# ---------------------------------------------------------------------------

ORIGIN_USER = "user"
ORIGIN_SYSTEM = "system"

# Session-id prefixes / values that indicate a real human is in the loop.
# These match the convention documented at agent_loop.py:538-542
# (Telegram: "telegram:<chat_id>", Slack: "slack:<channel_id>", etc.).
_USER_SESSION_PREFIXES: tuple[str, ...] = (
    "telegram:", "slack:", "discord:", "matrix:", "signal:",
)
_USER_SESSION_LITERALS: frozenset[str] = frozenset({"cli", "web"})


def origin_from_session_id(session_id: str | None) -> str:
    """Classify a session_id as user-initiated or system/background.

    Returns ``"user"`` when a real human is in the loop (Telegram chat,
    CLI prompt, Web bridge, etc.) so they can sanction the next tool call.
    Returns ``"system"`` for anything else — the reserved ``"system"``
    sentinel, ``None``, SYMBIOTE/GEPA/SENTINEL UUIDs, smoke-tests,
    benchmarks, cron — none of which represent a present user.
    The default is ``"system"`` (the safer/stricter classification) for
    any unrecognized value.
    """
    if not session_id or session_id == "system":
        return ORIGIN_SYSTEM
    if session_id in _USER_SESSION_LITERALS:
        return ORIGIN_USER
    if any(session_id.startswith(p) for p in _USER_SESSION_PREFIXES):
        return ORIGIN_USER
    return ORIGIN_SYSTEM


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PermissionDecision:
    """Result of a permission evaluation.

    Compatible with both:
    - agent_loop.py: uses .allowed / .requires_confirmation / .reason
    - acceptance test: uses .action ("ALLOW" | "DENY" | "APPROVE")
    """

    allowed: bool
    requires_confirmation: bool
    reason: str
    action: str  # "ALLOW" | "DENY" | "APPROVE"
    trust_level: TrustLevel = TrustLevel.AUTO

    @classmethod
    def allow(cls, reason: str = "", level: TrustLevel = TrustLevel.AUTO) -> PermissionDecision:
        return cls(allowed=True, requires_confirmation=False, reason=reason,
                   action="ALLOW", trust_level=level)

    @classmethod
    def approve(cls, reason: str = "") -> PermissionDecision:
        return cls(allowed=False, requires_confirmation=True, reason=reason,
                   action="APPROVE", trust_level=TrustLevel.APPROVE)

    @classmethod
    def deny(cls, reason: str) -> PermissionDecision:
        return cls(allowed=False, requires_confirmation=False, reason=reason,
                   action="DENY", trust_level=TrustLevel.BLOCKED)


# ---------------------------------------------------------------------------
# SecurityGate
# ---------------------------------------------------------------------------


class SecurityGate:
    """Permission checker for the Prometheus agent loop.

    Implements the 4-level trust model:
      LEVEL 0 (BLOCKED)    — rm -rf, system dirs, credential access → DENY
      LEVEL 1 (APPROVE)    — file writes outside workspace, git push, network → APPROVE
      LEVEL 2 (AUTO)       — reads within workspace, grep, glob, git status → ALLOW
      LEVEL 3 (AUTONOMOUS) — heartbeat checks, status notifications → ALLOW

    Usage (wired into AgentLoop):
        gate = SecurityGate.from_config()
        loop = AgentLoop(provider=..., permission_checker=gate)

    Usage (standalone acceptance test):
        gate = SecurityGate()
        result = gate.pre_tool_use('bash', {'command': 'rm -rf /'}, {})
        assert result.action == 'DENY'
    """

    def __init__(
        self,
        denied_commands: list[str] | None = None,
        denied_paths: list[str] | None = None,
        workspace_root: str | Path | None = None,
        mode: PermissionMode | str = PermissionMode.DEFAULT,
        audit_logger: AuditLogger | None = None,
        exfiltration_detector: ExfiltrationDetector | None = None,
        approval_queue: object | None = None,
    ) -> None:
        self._denied_commands: list[str] = denied_commands or []
        self._denied_paths: list[str] = [
            str(Path(p).expanduser()) for p in (denied_paths or [])
        ]
        self._workspace = Path(workspace_root).expanduser().resolve() if workspace_root else None
        self._mode = PermissionMode(mode) if isinstance(mode, str) else mode

        # Sprint 11: optional audit + exfiltration
        self._audit = audit_logger
        self._exfil = exfiltration_detector

        # Sprint 15b GRAFT: optional approval queue for Telegram confirmation
        self._approval_queue = approval_queue

        # Compile blocked patterns once
        self._blocked_re = [re.compile(p) for p in _ALWAYS_BLOCKED_PATTERNS]
        self._approve_re = [re.compile(p) for p in _APPROVE_BASH_PATTERNS]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str | Path | None = None) -> SecurityGate:
        """Load SecurityGate from prometheus.yaml security section."""
        import yaml

        if config_path is None:
            from prometheus.config.defaults import DEFAULTS_PATH
            config_path = DEFAULTS_PATH

        try:
            with open(Path(config_path).expanduser()) as fh:
                data = yaml.safe_load(fh)
            sec = data.get("security", {})
        except (OSError, Exception):
            sec = {}

        # Sprint 11: optionally create audit logger + exfiltration detector
        audit_logger = None
        exfil_detector = None
        audit_cfg = sec.get("audit", {})
        if audit_cfg.get("enabled", True):
            from prometheus.config.paths import get_data_dir
            audit_logger = AuditLogger(get_data_dir() / "security")

        exfil_cfg = sec.get("exfiltration", {})
        if exfil_cfg.get("enabled", True):
            exfil_detector = ExfiltrationDetector()

        return cls(
            denied_commands=sec.get("denied_commands") or [],
            denied_paths=sec.get("denied_paths") or [],
            workspace_root=sec.get("workspace_root"),
            mode=sec.get("permission_mode", "default"),
            audit_logger=audit_logger,
            exfiltration_detector=exfil_detector,
        )

    # ------------------------------------------------------------------
    # Audit helper
    # ------------------------------------------------------------------

    def _audit_log(
        self,
        tool_name: str,
        decision: AuditDecision,
        reason: str,
        tool_input: dict | str | None = None,
    ) -> None:
        """Write to audit log if an AuditLogger is attached."""
        if self._audit is None:
            return
        trust_val = self._mode_trust_level()
        self._audit.log(
            tool_name=tool_name,
            decision=decision,
            trust_level=trust_val,
            reason=reason,
            tool_input=tool_input,
        )

    def _mode_trust_level(self) -> int:
        if self._mode == PermissionMode.AUTONOMOUS:
            return TrustLevel.AUTONOMOUS
        if self._mode == PermissionMode.STRICT:
            return TrustLevel.APPROVE
        return TrustLevel.AUTO

    # ------------------------------------------------------------------
    # Public interface — used by agent_loop.py permission_checker slot
    # ------------------------------------------------------------------

    def evaluate(
        self,
        tool_name: str,
        *,
        is_read_only: bool = False,
        file_path: str | None = None,
        command: str | None = None,
        origin: str = ORIGIN_SYSTEM,
    ) -> PermissionDecision:
        """Evaluate whether a tool call is permitted.

        Called by agent_loop._execute_tool_call() with keyword args.

        ``origin``:
          ``"user"``    — request comes from a present human (Telegram, CLI,
                          Web). Bash commands skip ExfiltrationDetector and
                          network/install approve-patterns. Always-blocked
                          patterns, denied_commands, denied_paths, and the
                          write_file workspace gate STILL apply.
          ``"system"``  — automated/background (SENTINEL, GEPA, AutoDream,
                          smoke-tests, cron, SYMBIOTE phases). Full
                          restrictions apply. This is the safer default.
        """
        is_user = (origin == ORIGIN_USER)

        # Sprint 11: exfiltration check (system origin only — when the user
        # is not in the loop, network+sensitive-file combos are still blocked.
        # User-initiated bash bypasses exfil per the trust model: a present
        # human is responsible for what they ask the agent to send.)
        if not is_user and self._exfil and tool_name == "bash" and command:
            exfil_match = self._exfil.check_command(command)
            if exfil_match:
                reason = f"Exfiltration blocked: {exfil_match.reason}"
                self._audit_log(tool_name, AuditDecision.DENY, reason, command)
                return PermissionDecision.deny(reason)

        # AUTONOMOUS mode: allow everything except always-blocked patterns
        if self._mode == PermissionMode.AUTONOMOUS:
            if command and self._is_always_blocked(command):
                reason = f"Blocked command pattern: {command!r}"
                self._audit_log(tool_name, AuditDecision.DENY, reason, command)
                return PermissionDecision.deny(reason)
            self._audit_log(tool_name, AuditDecision.ALLOW, "Auto-allowed (autonomous)")
            return PermissionDecision.allow(level=TrustLevel.AUTONOMOUS)

        # --- LEVEL 0: check always-blocked patterns (both origins) ---
        if command:
            reason = self._check_blocked_command(command)
            if reason:
                self._audit_log(tool_name, AuditDecision.DENY, reason, command)
                return PermissionDecision.deny(reason)

        # --- Check denied_paths (both origins) ---
        if file_path:
            reason = self._check_denied_path(file_path)
            if reason:
                self._audit_log(tool_name, AuditDecision.DENY, reason, file_path)
                return PermissionDecision.deny(reason)

        # --- LEVEL 1: write_file / edit_file outside workspace → APPROVE
        # (both origins — this is the path-traversal guarantee) ---
        if tool_name in _APPROVE_TOOLS:
            if self._mode == PermissionMode.STRICT:
                reason = f"{tool_name} requires confirmation in strict mode"
                self._audit_log(tool_name, AuditDecision.CONFIRM_PENDING, reason)
                return PermissionDecision.approve(reason)
            if file_path and not self._within_workspace(file_path):
                reason = f"{tool_name} targets path outside workspace: {file_path}"
                self._audit_log(tool_name, AuditDecision.CONFIRM_PENDING, reason)
                return PermissionDecision.approve(reason)

        # --- LEVEL 1: bash with network/install commands → APPROVE
        # (system origin only — user-initiated curl/pip/wget/ssh allowed) ---
        if tool_name == "bash" and command and not is_user:
            if self._is_approve_pattern(command):
                if self._mode != PermissionMode.AUTONOMOUS:
                    reason = f"Command requires approval: {command!r}"
                    self._audit_log(tool_name, AuditDecision.CONFIRM_PENDING, reason, command)
                    return PermissionDecision.approve(reason)

        # --- LEVEL 2 / 3: allow ---
        level = TrustLevel.AUTO if not is_read_only else TrustLevel.AUTO
        reason = "Auto-allowed (user-initiated)" if is_user else "Auto-allowed"
        self._audit_log(tool_name, AuditDecision.ALLOW, reason)
        return PermissionDecision.allow(level=level)

    # ------------------------------------------------------------------
    # Acceptance-test interface (pre_tool_use convention)
    # ------------------------------------------------------------------

    def pre_tool_use(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        context: dict[str, Any],
    ) -> PermissionDecision:
        """Evaluate a tool call from a raw tool_input dict.

        ``context`` may carry ``"origin"`` (``"user"`` or ``"system"``) or
        ``"session_id"`` from which an origin is derived. Defaults to
        ``"system"`` for backward compatibility with callers that only pass
        the bare two-arg shape (existing acceptance test).

        Compatible with sprint acceptance test:
            result = gate.pre_tool_use('bash', {'command': 'rm -rf /'}, {})
            assert result.action == 'DENY'
        """
        command = tool_input.get("command") or tool_input.get("cmd")
        file_path = (
            tool_input.get("path")
            or tool_input.get("file_path")
            or tool_input.get("filepath")
        )
        origin = context.get("origin") if isinstance(context, dict) else None
        if not origin:
            origin = origin_from_session_id(
                context.get("session_id") if isinstance(context, dict) else None
            )
        return self.evaluate(
            tool_name,
            is_read_only=False,
            file_path=str(file_path) if file_path else None,
            command=str(command) if command else None,
            origin=origin,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_always_blocked(self, command: str) -> bool:
        return any(r.search(command) for r in self._blocked_re)

    def _check_blocked_command(self, command: str) -> str:
        """Return a denial reason if the command matches any blocked pattern."""
        for pattern in self._blocked_re:
            if pattern.search(command):
                return f"Blocked command pattern matched: {pattern.pattern!r}"
        for denied in self._denied_commands:
            if denied.lower() in command.lower():
                return f"Command matches deny list entry: {denied!r}"
        return ""

    def _check_denied_path(self, file_path: str) -> str:
        """Return a denial reason if the path falls under a denied prefix."""
        resolved = str(Path(file_path).expanduser().resolve())
        for denied in self._denied_paths:
            resolved_denied = str(Path(denied).expanduser().resolve())
            if resolved.startswith(resolved_denied):
                return f"Path {file_path!r} is under denied prefix {denied!r}"
        return ""

    def _within_workspace(self, file_path: str) -> bool:
        if self._workspace is None:
            return True
        try:
            Path(file_path).expanduser().resolve().relative_to(self._workspace)
            return True
        except ValueError:
            return False

    def _is_approve_pattern(self, command: str) -> bool:
        return any(r.search(command) for r in self._approve_re)
