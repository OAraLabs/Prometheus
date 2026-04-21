"""Permission mode and trust level definitions for Sprint 4."""

from __future__ import annotations

from enum import IntEnum, Enum


class TrustLevel(IntEnum):
    """Four-level trust model for tool execution decisions.

    BLOCKED (0)    — always deny; never execute.
    APPROVE (1)    — requires user confirmation before proceeding.
    AUTO (2)       — allow automatically; no prompt needed.
    AUTONOMOUS (3) — allow; used for background / heartbeat operations.
    """

    BLOCKED = 0
    APPROVE = 1
    AUTO = 2
    AUTONOMOUS = 3


class PermissionMode(str, Enum):
    """Named permission profiles loaded from prometheus.yaml security.permission_mode."""

    DEFAULT = "default"       # standard: destructive ops require approval
    STRICT = "strict"         # conservative: file writes + network require approval
    AUTONOMOUS = "autonomous" # fully automatic: no user confirmations
