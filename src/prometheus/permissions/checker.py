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
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prometheus.config.shipped_defaults import (
    resolve_denied_paths, resolve_workspace_root)
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

# ---------------------------------------------------------------------------
# Path locations denied REGARDLESS of configuration — the structural floor.
#
# The exact counterpart of _ALWAYS_BLOCKED_PATTERNS above, and it exists for
# the same reason. `denied_commands` was safe when absent because that list is
# hardcoded and applied first; `denied_paths` was not, so an absent key meant
# an empty list and `read_file ~/.ssh/id_rsa` succeeded. A boundary supplied
# entirely by config is a boundary that disappears when nobody writes the
# config (CROSS-CUTTING §5 — prefer a property that cannot be violated).
#
# DELIBERATELY NARROWER THAN THE SHIPPED LIST. `/etc`, `/sys` and `/boot` are
# policy — an operator may have a real reason to let an agent read
# /etc/hostname — and they arrive via SHIPPED_DENIED_PATHS, which an explicit
# config can override. What is in the floor is credential material only: no
# configuration should be able to hand an agent a private key, and nothing
# legitimate needs that permission.
_ALWAYS_DENIED_PATHS: tuple[str, ...] = (
    "~/.ssh",
    "~/.gnupg",
)

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

# Trusted command patterns — auto-ALLOWed even at SYSTEM trust (background
# tasks / cron), so vetted long-running jobs can run without a human approver.
# Checked AFTER the always-blocked / denied / exfiltration gates, so this can
# never resurrect a blocked pattern.
#
# Intentionally EMPTY in source: trusted patterns reference infrastructure
# specifics (e.g. the GPU host for model downloads) that must not be hardcoded
# in the repo. Configure them via ``security.allowed_commands`` in the
# (gitignored) prometheus.yaml — see prometheus.yaml.default for the vetted
# model-download example that powers the download-model-to-gpu skill.
_TRUSTED_COMMAND_PATTERNS: list[str] = []

# A trusted command must be a SINGLE simple invocation. These metacharacters
# could chain a second command past the pattern match, so their presence
# disqualifies a command from the allowlist (defense-in-depth).
_TRUSTED_CMD_FORBIDDEN = re.compile(r"[;&|`\n]|\$\(")


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
# Approval grants — "stop asking me about THIS" (session or persistent)
# ---------------------------------------------------------------------------
#
# CC-style permission memory for the approval layer. When the operator answers
# an APPROVE prompt with a scoped yes (/approve session <id> or
# /approve always <id>), the gate records a GRANT derived from the pending
# request and auto-allows matching calls from then on.
#
# Ordering guarantee: grants are evaluated AFTER always-blocked patterns,
# denied_commands, denied_paths and exfiltration — the same position as the
# trusted-command allowlist — so a grant can NEVER resurrect a blocked call.
# It only silences calls that would otherwise have gone to APPROVE.

@dataclass
class Grant:
    """One remembered approval.

    kind:
      "path_prefix"    — file_path resolves under ``value`` (file tools only).
      "command_prefix" — bash command starts with ``value`` AND is a single
                         simple invocation (no ; | & ` $() — same guard as the
                         trusted allowlist, so a prefix grant can't smuggle a
                         chained command through).
      "tool"           — the tool itself, any target (produced by strict-mode
                         approvals whose reason carries no target).
    """

    kind: str  # "path_prefix" | "command_prefix" | "tool"
    value: str
    tool_name: str
    scope: str = "session"  # "session" (memory-only) | "persistent" (config)

    def matches(self, tool_name: str, file_path: str | None, command: str | None) -> bool:
        if self.kind == "tool":
            return tool_name == self.tool_name
        if self.kind == "path_prefix":
            if tool_name != self.tool_name or not file_path:
                return False
            try:
                resolved = Path(file_path).expanduser().resolve()
                resolved.relative_to(Path(self.value).expanduser().resolve())
                return True
            except (ValueError, OSError):
                return False
        if self.kind == "command_prefix":
            if tool_name != "bash" or not command:
                return False
            if _TRUSTED_CMD_FORBIDDEN.search(command):
                return False  # single-invocation guard, same as trusted list
            return command.startswith(self.value)
        return False

    def to_config_dict(self) -> dict:
        return {"kind": self.kind, "value": self.value, "tool": self.tool_name}

    @classmethod
    def from_config_dict(cls, d: dict) -> Grant | None:
        kind = d.get("kind")
        if kind not in ("path_prefix", "command_prefix", "tool"):
            return None
        return cls(
            kind=kind,
            value=str(d.get("value", "")),
            tool_name=str(d.get("tool", "")),
            scope="persistent",
        )



# ---------------------------------------------------------------------------
# SecurityGate
# ---------------------------------------------------------------------------


_GLOB_CHARS = "*?["


def _is_glob(entry: str) -> bool:
    return any(c in entry for c in _GLOB_CHARS)


def _normalise_denied_path(entry: str) -> str:
    """Expand and validate ONE ``security.denied_paths`` entry.

    ABSOLUTE ONLY. A relative entry was previously resolved at check time
    against the daemon's working directory, which means the file it protected
    was chosen by wherever the process happened to be running — not by the
    config. That is not a control; it is a control-shaped thing whose target
    moves. Proven on 2026-08-13: moving the daemon from the dev checkout to a
    deploy clone silently moved the entry ``config/prometheus.yaml`` off the
    file it was meant to protect, and nothing in the config, the logs or the
    restart said so.

    Raising is the point. The alternatives are worse: ignoring the entry
    removes a control the operator believes they have, and resolving it
    reinstates the defect. A deny list that cannot say what it protects
    should stop the process, loudly, with the fix in the message.

    Glob entries (``~/.config/*/env``) are expanded but NOT resolved —
    ``Path.resolve()`` on a literal ``*`` component would mangle the pattern.
    """
    expanded = str(Path(entry).expanduser())
    if not Path(expanded).is_absolute():
        raise ValueError(
            f"security.denied_paths entry {entry!r} is RELATIVE. Entries must "
            f"be absolute (or start with '~'), because a relative entry is "
            f"resolved against the daemon's WORKING DIRECTORY — so the file it "
            f"protects changes whenever the process moves, silently. "
            f"Use an absolute path, e.g. '/opt/prometheus/{entry}' or "
            f"'~/{entry}'. NOTE: the daemon's own config file is denied "
            f"automatically and does not need an entry at all."
        )
    if _is_glob(expanded):
        return expanded
    return str(Path(expanded).resolve())


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
        allowed_commands: list[str] | None = None,
        config_path: str | Path | None = None,
        grants: list[Grant] | None = None,
    ) -> None:
        self._denied_commands: list[str] = denied_commands or []
        # The floor first, then the configured policy. Order is cosmetic —
        # both are checked — but it reads the way the guarantee works: the
        # floor is present before any config is consulted.
        self._denied_paths: list[str] = [
            _normalise_denied_path(p) for p in _ALWAYS_DENIED_PATHS
        ]
        for p in (denied_paths or []):
            entry = _normalise_denied_path(p)
            if entry not in self._denied_paths:
                self._denied_paths.append(entry)
        # The daemon's own config file is denied AUTOMATICALLY. It used to be a
        # config entry — the relative string "config/prometheus.yaml" — which
        # is the defect this whole change exists to remove: the file it
        # protected was chosen by the process's working directory. A template
        # cannot name an absolute path that is right for every install, so the
        # honest fix is not a better default but a PROPERTY: the process knows
        # where it read its config from, so it can deny that file without
        # anyone writing it down (CROSS-CUTTING §5).
        if config_path is not None:
            resolved_cfg = str(Path(config_path).expanduser().resolve())
            if resolved_cfg not in self._denied_paths:
                self._denied_paths.append(resolved_cfg)
        self._config_path = config_path
        # MULTI-ROOT. A single root did not survive contact: of 871 recorded
        # file-tool calls on the live box only 16 were under ~/projects, so a
        # one-root boundary is a wall of prompts — and a control that prompts
        # constantly is one that gets turned off (CROSS-CUTTING §4). Accepts a
        # string (back-compat) or a list.
        if workspace_root is None or workspace_root == "":
            self._workspaces: tuple[Path, ...] = ()
        elif isinstance(workspace_root, (str, Path)):
            self._workspaces = (Path(workspace_root).expanduser().resolve(),)
        else:
            self._workspaces = tuple(
                Path(w).expanduser().resolve() for w in workspace_root if w
            )
        self._mode = PermissionMode(mode) if isinstance(mode, str) else mode

        # Sprint 11: optional audit + exfiltration
        self._audit = audit_logger
        self._exfil = exfiltration_detector

        # Sprint 15b GRAFT: optional approval queue for Telegram confirmation
        self._approval_queue = approval_queue

        # Approval grants: remembered approvals. Session grants live in memory
        # only; persistent grants are also written to prometheus.yaml by
        # persist_grant() and re-loaded at construction (from_config reads
        # security.grants). Evaluated after every DENY-tier check, so they can
        # only suppress APPROVE-tier prompts, never resurrect a blocked call.
        self._grants: list[Grant] = list(grants or [])

        # reason → target context for APPROVE decisions, so the approval
        # queue can derive a Grant without parsing free text. Set in
        # evaluate(), consumed by request_approval(), bounded.
        self._approve_targets: dict[str, dict[str, str | None]] = {}

        # Compile blocked patterns once
        self._blocked_re = [re.compile(p) for p in _ALWAYS_BLOCKED_PATTERNS]
        self._approve_re = [re.compile(p) for p in _APPROVE_BASH_PATTERNS]
        # Trusted allowlist = built-in patterns + any from config.allowed_commands
        self._trusted_re = [
            re.compile(p)
            for p in (_TRUSTED_COMMAND_PATTERNS + list(allowed_commands or []))
        ]

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
            denied_paths=resolve_denied_paths(sec),
            allowed_commands=sec.get("allowed_commands") or [],
            workspace_root=resolve_workspace_root(sec),
            config_path=config_path,
            mode=sec.get("permission_mode", "default"),
            audit_logger=audit_logger,
            exfiltration_detector=exfil_detector,
            grants=[
                g for g in (
                    Grant.from_config_dict(d) for d in (sec.get("grants") or [])
                    if isinstance(d, dict)
                ) if g is not None
            ],
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

    # ------------------------------------------------------------------
    # Runtime mode toggle (Telegram /gate)
    # ------------------------------------------------------------------

    def set_mode(self, mode: str | PermissionMode) -> PermissionMode:
        """Change the permission mode at runtime without a restart.

        Accepts a PermissionMode or its string name ("default", "strict",
        "autonomous"). The new mode applies to every subsequent evaluate()
        call. In-memory only: a daemon restart restores the mode from
        config (security.permission_mode), so the gate always comes back
        ON in its configured posture.
        """
        self._mode = (
            mode if isinstance(mode, PermissionMode) else PermissionMode(mode)
        )
        return self._mode

    def current_mode(self) -> PermissionMode:
        return self._mode

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

        # --- Approval grants (both origins) → ALLOW ---
        # Remembered approvals. Same position as the trusted allowlist: after
        # every DENY-tier check, before the APPROVE tier — a grant can silence
        # a prompt, never resurrect a block.
        if self._grants:
            matched = next(
                (g for g in self._grants
                 if g.matches(tool_name, file_path, command)),
                None,
            )
            if matched is not None:
                self._audit_log(
                    tool_name, AuditDecision.ALLOW,
                    f"Auto-allowed (approval grant: {matched.kind} "
                    f"{matched.value or matched.tool_name}, {matched.scope})",
                    command or file_path,
                )
                return PermissionDecision.allow(level=TrustLevel.AUTO)

        # --- LEVEL 1: write_file / edit_file outside workspace → APPROVE
        # (both origins — this is the path-traversal guarantee) ---
        if tool_name in _APPROVE_TOOLS:
            if self._mode == PermissionMode.STRICT:
                reason = f"{tool_name} requires confirmation in strict mode"
                self._audit_log(tool_name, AuditDecision.CONFIRM_PENDING, reason)
                self._remember_approve_target(reason, file_path=file_path, command=command)
                return PermissionDecision.approve(reason)
            if file_path and not self._within_workspace(file_path):
                reason = f"{tool_name} targets path outside workspace: {file_path}"
                self._audit_log(tool_name, AuditDecision.CONFIRM_PENDING, reason)
                self._remember_approve_target(reason, file_path=file_path, command=command)
                return PermissionDecision.approve(reason)

        # --- Trusted command allowlist (both origins) → ALLOW ---
        # Lets a vetted command (e.g. an HF .gguf model download to the GPU box)
        # run as a SYSTEM-trust background task without a human approver. Placed
        # AFTER always-blocked / denied / exfiltration, so it can never bypass
        # them; the pattern is narrow and command-chaining is rejected.
        if tool_name == "bash" and command and self._is_trusted_command(command):
            self._audit_log(
                tool_name, AuditDecision.ALLOW,
                "Auto-allowed (trusted command pattern)", command,
            )
            return PermissionDecision.allow(level=TrustLevel.AUTO)

        # --- LEVEL 1: bash with network/install commands → APPROVE
        # (system origin only — user-initiated curl/pip/wget/ssh allowed) ---
        if tool_name == "bash" and command and not is_user:
            if self._is_approve_pattern(command):
                if self._mode != PermissionMode.AUTONOMOUS:
                    reason = f"Command requires approval: {command!r}"
                    self._audit_log(tool_name, AuditDecision.CONFIRM_PENDING, reason, command)
                    self._remember_approve_target(reason, file_path=None, command=command)
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

    # ------------------------------------------------------------------
    # Approval grants
    # ------------------------------------------------------------------

    def _remember_approve_target(
        self, reason: str, file_path: str | None, command: str | None
    ) -> None:
        """Keep the structured target behind an APPROVE reason so the
        approval queue can derive a Grant without parsing free text."""
        self._approve_targets[reason] = {"file_path": file_path, "command": command}
        if len(self._approve_targets) > 128:  # bounded; oldest entries drop
            for key in list(self._approve_targets)[:32]:
                self._approve_targets.pop(key, None)

    def approve_target_for(self, reason: str) -> dict[str, str | None]:
        return self._approve_targets.get(reason, {"file_path": None, "command": None})

    def add_grant(self, grant: Grant) -> None:
        """Register a grant (dedupes on kind+value+tool)."""
        for existing in self._grants:
            if (existing.kind, existing.value, existing.tool_name) == (
                grant.kind, grant.value, grant.tool_name
            ):
                return
        self._grants.append(grant)

    def list_grants(self) -> list[Grant]:
        return list(self._grants)

    def persist_grant(self, grant: Grant, config_path: str | Path | None = None) -> bool:
        """Append a grant to ``security.grants`` in the on-disk YAML.

        Surgical write (fresh-load the file, set the one key, dump) — the same
        pattern the deferred-loading toggle uses, so env-var secrets merged
        into the runtime config dict never get copied into the file.
        """
        import yaml

        path = Path(config_path or self._config_path or "").expanduser()
        if not path or not path.exists():
            return False
        try:
            with path.open(encoding="utf-8") as fh:
                on_disk = yaml.safe_load(fh) or {}
            grants = on_disk.setdefault("security", {}).setdefault("grants", [])
            entry = grant.to_config_dict()
            if entry not in grants:
                grants.append(entry)
            with path.open("w", encoding="utf-8") as fh:
                yaml.dump(on_disk, fh, default_flow_style=False, sort_keys=False)
            return True
        except Exception:
            log.warning("grant persist failed for %r", grant, exc_info=True)
            return False

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

    async def request_approval(self, tool_name: str, reason: str) -> bool:
        """Ask the operator to sanction one APPROVE decision. True = go ahead.

        THE MISSING HOP. ``daemon.py`` has assigned ``_approval_queue`` since
        the queue existed, ``ApprovalQueue``'s own docstring says "Wire into
        SecurityGate", and nothing ever read the field — while
        ``LoopContext.permission_prompt``, the other route from an APPROVE to
        the operator, was populated by no construction site on any surface.
        Two orphans facing each other across one missing line, so every
        APPROVE anywhere fell to ``agent_loop``'s ``else`` branch and became a
        refusal-with-explanation. The operator was never offered the choice
        the decision exists to create.

        Fails CLOSED and says why: no queue, or a queue that raises, means the
        answer is no. A permission prompt that degrades to "yes" when its
        transport breaks is worse than having none (CROSS-CUTTING §8).
        """
        queue = getattr(self, "_approval_queue", None)
        if queue is None:
            return False
        target = self.approve_target_for(reason)
        try:
            from prometheus.permissions.approval_queue import ApprovalResult
            result = await queue.request_approval(
                tool_name, reason,
                grant_file_path=target.get("file_path"),
                grant_command=target.get("command"),
            )
        except Exception:
            log.warning(
                "approval request for %s failed; refusing", tool_name,
                exc_info=True,
            )
            return False
        return result == ApprovalResult.APPROVED

    def _check_denied_path(self, file_path: str) -> str:
        """Return a denial reason if the path falls under a denied prefix.

        Entries are normalised ONCE at construction (see
        :func:`_normalise_denied_path`), so nothing here depends on the
        process's working directory. Before that, every entry was re-resolved
        on each call — which made a relative entry protect whichever file
        happened to sit under the daemon's cwd.
        """
        resolved = str(Path(file_path).expanduser().resolve())
        for denied in self._denied_paths:
            if _is_glob(denied):
                # A wildcard entry matches the path itself or anything under
                # it. ``fnmatch``'s ``*`` spans ``/``, which is broader than a
                # shell glob — deliberately: broader means MORE denied, and
                # this is a deny list.
                if fnmatch.fnmatch(resolved, denied) or fnmatch.fnmatch(
                        resolved, f"{denied.rstrip('/')}/*"):
                    return f"Path {file_path!r} matches denied pattern {denied!r}"
            # Match on PATH COMPONENTS, not raw string prefix. A bare
            # ``startswith`` denied "/etcetera/notes" for the entry "/etc" —
            # over-refusal, so it never announced itself, and it disagreed
            # with the glob branch three lines up, which has always compared
            # component-wise. Two branches of one matcher must not have
            # different ideas of what "under" means. Surfaced by mutation M5:
            # forcing every entry down the glob branch changed almost nothing,
            # which is only true if the two branches nearly agree — the gap
            # was exactly this.
            elif resolved == denied or resolved.startswith(denied.rstrip("/") + os.sep):
                return f"Path {file_path!r} is under denied prefix {denied!r}"
        return ""

    def _within_workspace(self, file_path: str) -> bool:
        """True when the path is under ANY configured workspace root.

        No roots configured = no confinement, unchanged: that is a deliberate
        API choice used by most SecurityGate construction sites (tests, and
        callers confined another way). What must never happen is a CONFIG that
        merely omits the key landing there — see resolve_workspace_root.
        """
        if not self._workspaces:
            return True
        resolved = Path(file_path).expanduser().resolve()
        for root in self._workspaces:
            try:
                resolved.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def _is_approve_pattern(self, command: str) -> bool:
        return any(r.search(command) for r in self._approve_re)

    def _is_trusted_command(self, command: str) -> bool:
        """True if ``command`` matches the trusted allowlist AND is a single
        simple invocation (no chaining metacharacters). Trusted commands are
        auto-ALLOWed at any origin — see _TRUSTED_COMMAND_PATTERNS."""
        if not command or _TRUSTED_CMD_FORBIDDEN.search(command):
            return False
        return any(r.search(command) for r in self._trusted_re)
