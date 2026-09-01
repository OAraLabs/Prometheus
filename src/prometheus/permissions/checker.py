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
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from prometheus.config.load import load_config_file
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
# NARROWER THAN THE SHIPPED LIST, and the line is CREDENTIAL vs POLICY —
# not "how few entries can we get away with".
#
# `/etc`, `/sys` and `/boot` are policy: an operator may have a real reason to
# let an agent read /etc/hostname, so they arrive via SHIPPED_DENIED_PATHS
# where a config can override them. The floor holds the patterns that are
# credential-bearing BY CONSTRUCTION — private keys, GPG secrets, and env
# files carrying secrets. Nobody legitimately overrides "do not read the file
# whose purpose is to hold a secret".
#
# ⚠ THIS REVERSES AN EARLIER CALL IN #226, and the evidence is why. There I
# kept `.config/*/*env` OUT of the floor on the argument that a floor which
# cannot be switched off must stay as small as possible. The argument is sound
# in general and it lost to what happened: the deploy clone's live config —
# gitignored, so no PR can reach it — still carried the old `~`-relative
# entries, and `/root/.config/prometheus/env` stayed ALLOWED after #226
# landed. The half that was overridable is exactly the half that went stale.
# `.ssh` and `.gnupg` were immune precisely because they were floored.
#
# So the size of the floor is the wrong axis. The right one is whether an
# override could ever be legitimate, and for these three it cannot be.
#
# ⚠ ANY HOME, NOT JUST THE DAEMON'S. These were `~/.ssh` / `~/.gnupg`, and `~`
# expands to the home of whoever the daemon runs as — so `/root/.ssh` and
# `/home/<anyone-else>/.ssh` were ALLOWED BY THE GATE. Observed live on
# 2026-08-16: an agent grep at `~/.ssh` was denied, and the same grep at
# `/root/.ssh` passed the gate and failed only on an OS permission error. The
# OS is not the control; it happened to be holding a door the gate had left
# open, and it would not hold it for a readable key directory owned by a
# service account, or for a daemon running as root.
#
# The glob form rather than a literal `/root/.ssh`: enumerating the homes you
# happen to think of leaves the same defect one name over, which is the
# name-pattern trap this repo has now hit four times (`_PATH_SHAPED`, deleted
# in #214). `fnmatch`'s `*` spans `/` here — documented below as deliberate,
# "broader means MORE denied" — so `/*/.ssh` covers every home at any depth.
_ALWAYS_DENIED_PATHS: tuple[str, ...] = (
    "/*/.ssh",
    "/*/.gnupg",
    "/*/.config/*/*env",
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
    # SPRINT-CONSENT: "session" renamed to "until_restart".
    #
    # The old name was a promise the system does not keep. There is ONE
    # SecurityGate per process (daemon.py:427), ``_grants`` is never cleared
    # anywhere, and ``matches()`` never reads this field — so a "session"
    # grant lived for the life of the daemon, across every session and every
    # surface (Telegram, Beacon, web, cron). Renaming states the true
    # boundary at consent time, which is this sprint's whole point.
    #
    # Real per-session scoping needs a per-session notion the gate does not
    # have; it is logged as separate work, not faked here.
    scope: str = "until_restart"  # "until_restart" (memory) | "persistent" (config)
    # Provenance (SPRINT-CONSENT 0b). Revocation needs a stable handle and the
    # audit trail needs a join key; the record carried neither.
    grant_id: str = ""      # stable handle, survives persistence
    created_at: float = 0.0  # unix seconds
    request_id: str = ""     # the approval request that produced this grant
    # THE INTENT, carried rather than re-derived. ``describe()`` used to
    # recover this with ``Path(value).is_dir()`` — a stat at RENDER time — so
    # the description was a function of mutable filesystem state instead of a
    # property of the grant. The same grant, unchanged, described itself
    # narrowly before the approved write created the directory and as a
    # subtree afterwards; and because a widening approval's target directory
    # is usually created BY the action being approved, the wrong branch was
    # the normal case. Proven live by Beacon's consent walk on daemon
    # fb73b28: shown "on exactly /home/will/beacon-walk-wide", granted
    # "on anything under /home/will/beacon-walk-wide/".
    #
    #   True  — created to cover a subtree (``/approve … here``, or a root).
    #   False — created to cover exactly this target.
    #   None  — NOT RECORDED. Config rows written before this field existed.
    #           See describe() for how those are worded; the answer is not a
    #           stat, and it is not a guess either.
    #
    # ``scope`` above has the same shape of bug and the same fix. It defaulted
    # to "until_restart" here and callers patched it afterwards — one did,
    # one did not — so ``_audit_resolution`` recorded a PERMANENT grant as
    # "until the daemon restarts". Measured on the live store:
    #
    #   confirm_approved: request=fc12bcae, scope=always, grant=1502e24f3837
    #     (write_file on exactly /home/will/beacon-persist/x.txt
    #      — until the daemon restarts)
    #
    # while that same grant's revoke row says ``persistent``. Both fields are
    # now set at construction by ``derive_grant``; neither is a caller's
    # responsibility.
    widened: bool | None = None

    def __post_init__(self) -> None:
        # Generated here rather than at the call site so EVERY construction
        # path gets one — including Grant.from_config_dict re-materialising a
        # grant written before this field existed.
        if not self.grant_id:
            self.grant_id = uuid.uuid4().hex[:12]
        if not self.created_at:
            self.created_at = time.time()

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
        # Provenance is persisted: without grant_id on disk, a revoke has
        # nothing stable to name, and after a restart the in-memory id would
        # differ from the one the operator was shown.
        return {
            "kind": self.kind,
            "value": self.value,
            "tool": self.tool_name,
            "id": self.grant_id,
            "created_at": self.created_at,
            "request_id": self.request_id,
            # Persisted so a restart does not turn a recorded intent back
            # into an unrecorded one. Written only when known, so rows keep
            # their shape when there is nothing to say.
            **({"widened": self.widened}
               if self.widened is not None else {}),
        }

    @classmethod
    def from_config_dict(cls, d: dict) -> Grant | None:
        kind = d.get("kind")
        if kind not in ("path_prefix", "command_prefix", "tool"):
            return None
        # scope is hardcoded, not read: anything in the config file IS
        # persistent by definition, so a missing or stale value cannot mislabel it.
        return cls(
            kind=kind,
            value=str(d.get("value", "")),
            tool_name=str(d.get("tool", "")),
            scope="persistent",
            # Absent on entries written before SPRINT-CONSENT — __post_init__
            # mints one, so old config entries become revocable rather than
            # being stranded without a handle.
            grant_id=str(d.get("id", "")),
            created_at=float(d.get("created_at") or 0.0),
            request_id=str(d.get("request_id", "")),
            # ABSENT stays None — "not recorded" — and describe() words it
            # from the matching rule. Defaulting to False here would quietly
            # relabel every pre-existing directory grant as an exact-file one.
            widened=(
                bool(d["widened"]) if "widened" in d else None
            ),
        )

    def describe(self) -> str:
        """The grant's EXTENT, in operator terms. One computed description.

        SPRINT-CONSENT Phase 1/0e: both surfaces render from this. Telegram
        formats it into prose, Beacon ships it as a field — neither
        re-derives it, so the two cannot drift (Standing-Principles §17).
        """
        duration = (
            "permanently, until revoked"
            if self.scope == "persistent"
            else "until the daemon restarts"
        )
        if self.kind == "tool":
            what = f"EVERY use of {self.tool_name}, on any target"
        elif self.kind == "path_prefix":
            # NO FILESYSTEM CALL. The wording comes from the intent recorded
            # when the grant was built, not from what the disk looks like when
            # someone happens to render it. See ``widened``.
            if self.widened is True:
                what = f"{self.tool_name} on anything under {self.value}/"
            elif self.widened is False:
                what = f"{self.tool_name} on exactly {self.value}"
            else:
                # Intent unrecorded (a config row from before this field).
                # Guessing either way would be wrong half the time, and the
                # two wrong answers are not symmetric: understating a subtree
                # grant is consent under a narrower description, which is the
                # defect this whole line of work exists to remove.
                #
                # So state the MATCHING RULE instead, which is knowable from
                # the record alone and is true whichever the intent was:
                # ``matches()`` resolves the candidate and calls
                # ``relative_to(value)``, so the grant covers ``value`` and
                # everything beneath it. For a file-shaped value "everything
                # beneath it" is empty, and the sentence stays accurate.
                what = (
                    f"{self.tool_name} on {self.value} and anything under it"
                )
        elif self.kind == "command_prefix":
            what = f"any bash command starting with {self.value!r}"
        else:  # pragma: no cover - kind is validated at construction
            what = f"{self.kind} {self.value}"
        return f"{what} — {duration}"



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


def _splice_grants(original: str, grants: list) -> str:
    """Return ``original`` with only the ``security.grants`` block replaced.

    Everything outside that one block — comments, key order, quoting,
    indentation, blank lines — is carried through byte-for-byte. That is the
    whole point: a whole-file ``yaml.dump`` is what deleted the config's 430
    comment lines once already (see ``_rewrite_config_grants``).

    Handles the three shapes the key actually takes on disk:
      * ``grants: []``            — inline empty, one line
      * ``grants:`` + ``- …``     — block sequence
      * key absent entirely       — inserted under ``security:``
    and the case where ``security:`` itself is absent (appended at the end).
    """
    import yaml

    def render(indent: int) -> list[str]:
        dumped = yaml.dump(
            {"grants": grants}, default_flow_style=False, sort_keys=False
        ).rstrip("\n")
        return [(" " * indent + ln) if ln.strip() else ln
                for ln in dumped.split("\n")]

    lines = original.splitlines()

    def content(ln: str) -> bool:
        return bool(ln.strip()) and not ln.lstrip().startswith("#")

    def indent_of(ln: str) -> int:
        return len(ln) - len(ln.lstrip())

    # Locate `security:` at top level, then its block extent.
    sec = next((i for i, ln in enumerate(lines)
                if content(ln) and indent_of(ln) == 0
                and ln.split(":", 1)[0].strip() == "security"), None)
    if sec is None:
        tail = "" if original.endswith("\n") else "\n"
        return original + tail + "security:\n" + "\n".join(render(2)) + "\n"

    end = len(lines)
    for i in range(sec + 1, len(lines)):
        if content(lines[i]) and indent_of(lines[i]) == 0:
            end = i
            break

    # Locate `grants:` inside that block.
    key = next((i for i in range(sec + 1, end)
                if content(lines[i])
                and lines[i].split(":", 1)[0].strip() == "grants"), None)
    if key is None:
        return "\n".join(lines[:sec + 1] + render(2) + lines[sec + 1:]) + "\n"

    ind = indent_of(lines[key])
    stop = key + 1
    if lines[key].split(":", 1)[1].strip() == "":
        # Block sequence: consume its items and their continuation lines.
        # Stops at a comment at or above the key's indent so documentation
        # that belongs to the NEXT key is never swallowed.
        # `stop` advances only over lines PROVEN to belong to the block, so a
        # trailing blank line before the next section is left where it is. A
        # revoke that swallowed it would drift the file's formatting on every
        # approval cycle, which the round-trip test forbids.
        probe = key + 1
        while probe < end:
            ln = lines[probe]
            if not ln.strip():
                probe += 1
                continue
            if not content(ln) and indent_of(ln) <= ind:
                break
            if content(ln) and indent_of(ln) <= ind \
                    and not ln.lstrip().startswith("- "):
                break
            probe += 1
            stop = probe

    return "\n".join(lines[:key] + render(ind) + lines[stop:]) + "\n"


class SecurityGate:
    """Permission checker for the Prometheus agent loop.

    Implements the 4-level trust model:
      LEVEL 0 (BLOCKED)    — rm -rf, system dirs, credential access → DENY
      LEVEL 1 (APPROVE)    — file writes outside workspace, git push, network → APPROVE
      LEVEL 2 (AUTO)       — reads within workspace, grep, glob, git status → ALLOW
      LEVEL 3 (AUTONOMOUS) — heartbeat checks, status notifications → ALLOW

    Usage (wired into AgentLoop):
        gate = SecurityGate.from_config(config_path)   # ALWAYS pass the path
        loop = AgentLoop(provider=..., permission_checker=gate)

    ⚠ The no-argument form is NOT the recommended usage and was documented
    here as if it were. It falls back to a module-level default path that
    resolves to a file present on no checkout, so the gate it returns has an
    empty ``security`` section: the shipped ``denied_paths`` floor still
    holds, but every configured ``denied_commands`` entry is gone. What the
    daemon actually does is build the gate from the config it already loaded
    (``__main__.create_security_gate``), which is the pattern to copy. The
    default path is fixed in its own change; until then, pass the path.

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
        """Load SecurityGate from prometheus.yaml security section.

        ⚠ Pass *config_path*. Omitting it falls back to the module-level
        default, and a gate built from a config that could not be read keeps
        the hardcoded ``denied_paths`` floor but loses every CONFIG-supplied
        ``denied_commands`` entry — measured, ``cat /etc/shadow`` goes
        DENY -> ALLOW. The read is now loud about that (see
        prometheus.config.load) instead of substituting ``{}`` in silence, but
        loud is a diagnosis, not a fix: the caller still ends up with a gate
        that is missing its deny list.
        """
        explicit = config_path is not None
        if config_path is None:
            from prometheus.config.defaults import DEFAULTS_PATH
            config_path = DEFAULTS_PATH

        load = load_config_file(
            config_path,
            subsystem="security_gate",
            substituting="the shipped denied_paths floor with NO configured "
                         "denied_commands, allowed_commands or workspace roots",
            explicit=explicit,
        )
        sec = load.section("security")

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
                          patterns and denied_commands STILL refuse; the
                          write_file workspace gate still applies, as an
                          approval prompt rather than a refusal.
          ``"system"``  — automated/background (SENTINEL, GEPA, AutoDream,
                          smoke-tests, cron, SYMBIOTE phases). Full
                          restrictions apply. This is the safer default.

        ``denied_paths`` — including the ``_ALWAYS_DENIED_PATHS`` floor — is
        NOT in that list, and this docstring claimed for a long time that it
        was. It is checked only when a ``file_path`` argument is passed:
        ``_check_denied_path`` below is nested under ``if file_path:``. So it
        covers the path-declaring tools and does NOT cover bash, at EITHER
        origin — the gate is handed a command string and never sees the paths
        inside it. Verified by outcome, not inferred: at ``origin="user"``
        ``rm -rf /`` and a denied_command both DENY, while
        ``cat /home/<user>/.gnupg/x`` and ``echo x > /home/<user>/.ssh/x`` both
        ALLOW.

        This is not an origin distinction and tightening ``origin`` does not
        fix it. Enforcing the floor for bash needs a control below the tool
        layer, because any check on the command string is defeated by
        ordinary shell — ``cd ~/.ssh && cat id_*``, ``$HOME`` indirection,
        globs, ``sh -c``.
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

        # AUTONOMOUS mode: suppress the APPROVAL tiers — never the FLOOR.
        #
        # This branch returns before the denied-path check below, so until now
        # `/gate off` allowed write_file and read_file on ~/.ssh, ~/.gnupg and
        # the daemon env file, while its own reply text told the operator that
        # "denied paths still enforced". Measured, not inferred: at mode
        # autonomous, write_file and read_file on ~/.ssh/id_rsa both returned
        # ALLOW where default and strict both returned DENY.
        #
        # The floor is not a mode. What autonomous drops is the APPROVE tier
        # (workspace prompts, strict confirmations) and configured policy —
        # deliberately still `_is_always_blocked` here, not
        # `_check_blocked_command`, so denied_commands stays a policy the mode
        # can waive while the always-blocked patterns cannot be.
        if self._mode == PermissionMode.AUTONOMOUS:
            if command and self._is_always_blocked(command):
                reason = f"Blocked command pattern: {command!r}"
                self._audit_log(tool_name, AuditDecision.DENY, reason, command)
                return PermissionDecision.deny(reason)
            if file_path:
                reason = self._check_denied_path(file_path)
                if reason:
                    self._audit_log(tool_name, AuditDecision.DENY, reason, file_path)
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

        # --- LEVEL 1: third-party MCP tools (FOUNDATION 2.3a) → APPROVE
        # unless the server declares the tool read-only. Both origins: a
        # present human asked for the TASK, not for whatever a third-party
        # server does with it. Placed after grants, so the operator's
        # "/approve always <id>" (a "tool"-kind grant) silences the prompt
        # for exactly that tool; AUTONOMOUS mode returned above and waives
        # this tier like every other APPROVE. Before this rule, every MCP
        # call fell through to auto-allow: the adapter hardcoded
        # is_read_only=True, declared no path params the gate could see,
        # and carried no command — three misses that compose to "the
        # sanctioned third-party surface is the ungated one".
        if tool_name.startswith("mcp__") and not is_read_only:
            reason = (
                f"Third-party MCP tool {tool_name} is not declared "
                "read-only by its server"
            )
            self._audit_log(tool_name, AuditDecision.CONFIRM_PENDING, reason)
            self._remember_approve_target(reason, file_path=file_path, command=command)
            return PermissionDecision.approve(reason)

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

    def add_grant(self, grant: Grant) -> Grant:
        """Register a grant. Returns the EFFECTIVE grant (new or upgraded).

        ⚠ SPRINT-CONSENT — the dedupe used to ignore ``scope``, and that made
        memory and disk disagree. ``/approve always`` on a target already
        covered by an ``until_restart`` grant hit this early ``return``, so
        the in-memory entry kept ``scope="until_restart"`` while
        ``persist_grant`` — a separate call in ``cmd_approve`` — still wrote
        the entry to the config file. Two stores of one truth, inside the
        permission system. Harmless only because ``matches()`` never reads
        scope; a revoke that consulted one view would have missed the other.

        Now: same identity (kind, value, tool) UPGRADES in place rather than
        being dropped. The original ``grant_id`` is kept deliberately — the
        operator may already have been shown it, and revocation names it.
        """
        for existing in self._grants:
            if (existing.kind, existing.value, existing.tool_name) == (
                grant.kind, grant.value, grant.tool_name
            ):
                if existing.scope != "persistent" and grant.scope == "persistent":
                    # THE ONE PERMITTED MUTATION of scope after
                    # construction, and it is a state TRANSITION rather
                    # than a caller filling in a field derive_grant left
                    # blank: an existing until-restart grant is being
                    # upgraded because the operator has now said
                    # "always". test_no_caller_patches_scope_or_widened
                    # asserts this is the only one.
                    existing.scope = "persistent"
                    # Adopt the newer request as the provenance for the
                    # upgrade — it is the approval that widened the duration.
                    existing.request_id = grant.request_id or existing.request_id
                return existing
        self._grants.append(grant)
        return grant

    def list_grants(self) -> list[Grant]:
        return list(self._grants)

    def remove_grant(
        self, grant_id: str, config_path: str | Path | None = None
    ) -> bool:
        """Revoke one grant by id. Clears memory AND the persisted entry.

        BOTH HALVES, deliberately, and this is the whole point of the method:
        a revoke that clears only memory is undone by the next restart, when
        ``from_config`` re-materialises the grant from the file; a revoke that
        clears only the file leaves the grant live until that restart. Either
        alone is a revoke that does not revoke.

        Returns True if a grant with that id was found in memory.
        """
        match = next((g for g in self._grants if g.grant_id == grant_id), None)
        if match is None:
            return False
        self._grants.remove(match)
        if match.scope == "persistent":
            # Best-effort on the file half; the in-memory removal already
            # happened and is reported. A failure here is logged loudly by
            # _rewrite_config_grants rather than swallowed.
            self._unpersist_grant(match, config_path)
        self._audit_log(
            match.tool_name,
            AuditDecision.DENY,
            f"Grant REVOKED ({match.kind} {match.value or match.tool_name}, "
            f"{match.scope}, id={match.grant_id})",
            match.value,
        )
        return True

    def clear_grants(
        self, scope: str | None = None, config_path: str | Path | None = None
    ) -> int:
        """Revoke every grant, or every grant of one scope. Returns the count."""
        doomed = [g for g in self._grants if scope is None or g.scope == scope]
        for g in doomed:
            self.remove_grant(g.grant_id, config_path)
        return len(doomed)

    def persist_grant(self, grant: Grant, config_path: str | Path | None = None) -> bool:
        """Append a grant to ``security.grants`` in the on-disk YAML.

        Surgical write (fresh-load the file, set the one key, dump) — the same
        pattern the deferred-loading toggle uses, so env-var secrets merged
        into the runtime config dict never get copied into the file.
        """
        def _add(grants: list) -> bool:
            if any(
                g.get("id") == grant.grant_id for g in grants if isinstance(g, dict)
            ):
                return False
            grants.append(grant.to_config_dict())
            return True

        return self._rewrite_config_grants(_add, config_path, grant)

    def _unpersist_grant(
        self, grant: Grant, config_path: str | Path | None = None
    ) -> bool:
        """Remove one grant from ``security.grants`` on disk, by id."""

        def _drop(grants: list) -> bool:
            keep = [
                g for g in grants
                if not (isinstance(g, dict) and g.get("id") == grant.grant_id)
            ]
            if len(keep) == len(grants):
                return False
            grants[:] = keep
            return True

        return self._rewrite_config_grants(_drop, config_path, grant)

    def _rewrite_config_grants(self, mutate, config_path, grant: Grant) -> bool:
        """Surgical, ATOMIC read-modify-write of ``security.grants``.

        Surgical (fresh-load the file, touch one key, splice) so env-var
        secrets merged into the runtime config dict never get copied into the
        file.

        ⚠ ATOMIC via temp-file + ``os.replace``, added with revocation. The
        previous form truncated the real file in place, so a crash mid-dump
        left a partial ``prometheus.yaml`` — the file the daemon reads at
        boot. ``os.replace`` is atomic on POSIX: a reader sees the old file or
        the new one, never a truncated one.

        ⚠ COMMENT-PRESERVING, and this is load-bearing now. This function
        used to ``yaml.dump`` the WHOLE file, which drops every comment and
        reformats everything else. That was excused on the premise that "the
        live config is machine-written and already comment-free" — a premise
        that held only because a *different* writer (the config-pin drift
        auto-fix, removed in #242) had already stripped the file's 430
        comment lines. Restoring those comments falsifies the excuse: the
        next persistent grant would delete the operator's documentation a
        second time, by the same mechanism, for the same reason.

        So the write is now a SPLICE. Only the ``security.grants`` block is
        re-serialised; every other byte of the file is carried through
        untouched. Grants are machine-owned, so reformatting inside that one
        block costs nothing.

        ⚠ KNOWN, NOT FIXED: atomic is not LOCKED — two concurrent writers can
        still lose an update (last writer wins). Latent while this is the
        only writer of the file, which it is again as of #242.
        """
        import tempfile

        import yaml

        path = Path(config_path or self._config_path or "").expanduser()
        if not path or not path.exists():
            return False
        try:
            original = path.read_text(encoding="utf-8")
            on_disk = yaml.safe_load(original) or {}
            grants = on_disk.setdefault("security", {}).setdefault("grants", [])
            if not mutate(grants):
                return True  # already in the desired state

            new_text = _splice_grants(original, grants)

            # Prove the splice before it reaches the file: the parsed result
            # must differ from the original in security.grants and NOWHERE
            # else. A text edit that quietly changed another key would be a
            # permissions bug wearing a formatting diff.
            reparsed = yaml.safe_load(new_text) or {}
            expected = yaml.safe_load(original) or {}
            expected.setdefault("security", {})["grants"] = grants
            if reparsed != expected:
                log.error(
                    "grant config splice would alter keys beyond security.grants; "
                    "refusing to write %s", path,
                )
                return False

            fd, tmp = tempfile.mkstemp(
                dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(new_text)
                    fh.flush()
                    os.fsync(fh.fileno())
                shutil.copymode(path, tmp)
                os.replace(tmp, path)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
            return True
        except Exception:
            log.warning("grant config rewrite failed for %r", grant, exc_info=True)
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
