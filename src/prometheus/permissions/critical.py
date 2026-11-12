"""Which approval requests must be answered in the app, never from a notification (``critical``).

A notification can show a tool name and one line. Some requests cannot be judged from that, and some are too
consequential to approve with a thumb on a lock screen. For those the daemon says ``critical: true`` in the one
serialized request (``ApprovalQueue.serialize_pending``), so ``GET /api/approvals``, the ``approval_pending`` frame
and the push agree by construction, and the push carries no ``category``: no Approve button, the phone opens the app.

The daemon decides; a client never classifies a request itself.

Built in, not configurable (the facts are already on the request):

  * a desktop prompt (``task_id``): its push carries no content at all;
  * a request to lift a device's hold (``hold_device_id``): it widens what a device may run;
  * arguments that were cut or redacted when the request was captured: the part that was cut is the decision;
  * a description over ``DESCRIPTION_MAX_CHARS`` characters or ``DESCRIPTION_MAX_LINES`` lines: it will not be
    read on a lock screen.

Owner-editable (``push.critical_tools``, ``push.critical_commands``, ``push.critical_paths``), with defaults for
deletes, ``git push``, anything that sends or spends, and changes to devices, permissions or the agent's own
configuration. A missing key means the default list; an empty list means none. A value of the wrong shape is
logged and the default is used: the defaults are the narrower choice (more requests answered in the app).

Source: novel code for Prometheus, 2026-11-12.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DESCRIPTION_MAX_CHARS = 600
DESCRIPTION_MAX_LINES = 8

#: Tools whose every call is critical by default: they send something to someone, spend money, or change
#: schedules that run the agent unattended.
DEFAULT_CRITICAL_TOOLS: tuple[str, ...] = (
    "message", "sessions_send",                    # send
    "image_generate", "video_generate",            # spend
    "cron_create", "cron_delete",                  # change what runs unattended
    "symbiote_graft",                              # changes the agent's own code
)

#: Shell commands, as leading words of any command in a line (``a && b | c`` is three). ``sudo``, ``env``,
#: ``nohup``, ``time``, ``command``, ``exec`` and leading ``VAR=value`` words are skipped first.
DEFAULT_CRITICAL_COMMANDS: tuple[str, ...] = (
    # deletes
    "rm", "rmdir", "unlink", "shred", "truncate", "trash",
    "git rm", "git clean", "git branch -d", "git branch -D", "git reset --hard",
    # git push, in every form
    "git push",
    # sends
    "curl", "wget", "ssh", "scp", "sftp", "rsync", "nc", "ncat", "mail", "sendmail", "mutt", "osascript",
    "gh", "npm publish", "twine upload", "uv publish", "docker push",
    # spends
    "aws", "gcloud", "az", "stripe",
    # devices, permissions, the agent's own configuration and services
    "oara", "prometheus", "sudo", "chmod", "chown", "crontab", "launchctl", "systemctl", "security",
)

#: Paths whose writes are critical by default: the agent's own configuration and state.
DEFAULT_CRITICAL_PATHS: tuple[str, ...] = ("~/.prometheus", "~/.config/prometheus")

#: Tools that write the path in ``grant_file_path``.
_WRITE_TOOLS = frozenset({"write_file", "edit_file", "notebook_edit"})

_SKIP_WORDS = frozenset({"sudo", "env", "nohup", "time", "command", "exec", "nice", "doas"})
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_SEPARATORS = re.compile(r"\|\||&&|[;|&\n]|\$\(|`|\)")


def _strings(value: Any, default: tuple[str, ...], key: str) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return tuple(v.strip() for v in value if v.strip())
    logger.warning("push.%s must be a list of strings; using the default list", key)
    return default


@dataclass(frozen=True)
class CriticalRules:
    tools: tuple[str, ...] = DEFAULT_CRITICAL_TOOLS
    commands: tuple[str, ...] = DEFAULT_CRITICAL_COMMANDS
    paths: tuple[str, ...] = DEFAULT_CRITICAL_PATHS

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> CriticalRules:
        push = (config or {}).get("push") if isinstance(config, Mapping) else None
        push = push if isinstance(push, Mapping) else {}
        return cls(
            tools=_strings(push.get("critical_tools"), DEFAULT_CRITICAL_TOOLS, "critical_tools"),
            commands=_strings(push.get("critical_commands"), DEFAULT_CRITICAL_COMMANDS, "critical_commands"),
            paths=_strings(push.get("critical_paths"), DEFAULT_CRITICAL_PATHS, "critical_paths"),
        )

    def matches(self, tool_name: str, arguments: Mapping[str, Any] | None,
                grant_file_path: str | None = None) -> tuple[str, ...]:
        """The owner-list reasons this call is critical, read from the RAW arguments at capture."""
        reasons: list[str] = []
        if tool_name in self.tools:
            reasons.append(f"tool:{tool_name}")
        command = (arguments or {}).get("command") if tool_name == "bash" else None
        if isinstance(command, str):
            hit = _command_match(command, self.commands)
            if hit:
                reasons.append(f"command:{hit}")
        if tool_name in _WRITE_TOOLS:
            path = grant_file_path or (arguments or {}).get("path") or (arguments or {}).get("file_path")
            if isinstance(path, str) and _under_any(path, self.paths):
                reasons.append("path")
        return tuple(reasons)


def _words(segment: str) -> list[str]:
    try:
        words = shlex.split(segment, posix=True)
    except ValueError:
        words = segment.split()
    while words and (words[0] in _SKIP_WORDS or _ASSIGNMENT.match(words[0])):
        words = words[1:]
    if words:
        words[0] = os.path.basename(words[0])
    return words


def _command_match(command: str, patterns: Iterable[str]) -> str | None:
    """The first pattern that leads any command in *command*, or None. Words compare exactly."""
    split = [p.split() for p in patterns if p.split()]
    for segment in _SEPARATORS.split(command):
        words = _words(segment)
        if not words:
            continue
        for pattern in split:
            if words[:len(pattern)] == pattern:
                return " ".join(pattern)
            # ``git -C dir push``: a git option between the command and its verb.
            if pattern[0] == "git" and words[0] == "git" and len(pattern) > 1:
                rest = _drop_option_values(words[1:])
                if rest[:len(pattern) - 1] == pattern[1:]:
                    return " ".join(pattern)
    return None


def _drop_option_values(words: list[str]) -> list[str]:
    """git's global options that take a value (``-C dir``, ``-c k=v``) removed, so the verb comes first."""
    out: list[str] = []
    skip = False
    for i, w in enumerate(words):
        if skip:
            skip = False
            continue
        if not out and w in ("-C", "-c", "--git-dir", "--work-tree", "--namespace"):
            skip = True
            continue
        if not out and w.startswith("-"):
            continue
        out = words[i:]
        break
    return out


def _under_any(path: str, roots: Iterable[str]) -> bool:
    try:
        target = os.path.realpath(os.path.expanduser(path))
    except (OSError, ValueError):
        return True                                  # cannot say where it lands: the narrower answer
    for root in roots:
        try:
            base = os.path.realpath(os.path.expanduser(root))
        except (OSError, ValueError):
            continue
        if target == base or target.startswith(base.rstrip(os.sep) + os.sep):
            return True
    return False


def critical_reasons(action: Any) -> tuple[str, ...]:
    """Why *action* must be answered in the app; empty when it may be answered from a notification.

    Built-in facts first, then what the owner's lists matched when the request was captured
    (``PendingAction.critical_matches``).
    """
    reasons: list[str] = []
    if getattr(action, "task_id", None) is not None:
        reasons.append("desktop")
    if getattr(action, "hold_device_id", None) is not None:
        reasons.append("hold_lift")
    if getattr(action, "arguments_cut", False):
        reasons.append("arguments_cut")
    description = str(getattr(action, "description", "") or "")
    if len(description) > DESCRIPTION_MAX_CHARS or len(description.splitlines()) > DESCRIPTION_MAX_LINES:
        reasons.append("long_description")
    reasons.extend(getattr(action, "critical_matches", ()) or ())
    return tuple(reasons)


def is_critical(action: Any) -> bool:
    return bool(critical_reasons(action))
