"""Permission mode and trust level definitions for Sprint 4."""

from __future__ import annotations

import os
from collections.abc import Mapping
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
    CAUTIOUS = "cautious"     # strict, and EVERY shell command the model runs asks too (new app installs)
    AUTONOMOUS = "autonomous" # fully automatic: no user confirmations


def fresh_install_mode(environ: Mapping[str, str] | None = None) -> str:
    """The ``security.permission_mode`` a NEW config is written with.

    ``cautious`` on the app install (``PROMETHEUS_INSTALL_KIND=app``, set by Prometheus.app's launcher): on macOS
    the daemon has no shell floor, and the app is for people who never chose a terminal, so a new install asks
    before every shell command the model runs and every file write. ``default`` everywhere else, as before. An
    EXISTING config is never rewritten with this; its writers carry its mode over.
    """
    env = os.environ if environ is None else environ
    return PermissionMode.CAUTIOUS.value if env.get("PROMETHEUS_INSTALL_KIND") == "app" else PermissionMode.DEFAULT.value

