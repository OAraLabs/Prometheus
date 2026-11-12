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
    "git rm", "git clean", "git branch -d", "git branch -D", "git branch --delete", "git reset --hard",
    "git checkout --", "git restore", "git stash drop", "git stash clear",
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

#: Under a critical path, but where a tool writes in the ordinary way (``download_file``'s default destination).
_ORDINARY_WRITE_DIRS: tuple[str, ...] = ("~/.prometheus/downloads",)

#: Tools that write a path, and the argument that names it (``grant_file_path`` wins when the gate gave one).
_WRITE_TOOLS: dict[str, tuple[str, ...]] = {
    "write_file": ("path", "file_path"),
    "edit_file": ("path", "file_path"),
    "notebook_edit": ("path", "notebook_path", "file_path"),
    "download_file": ("destination", "path"),
}

#: Words that run the rest of their line as a command: the command after them (and after their own options) is
#: matched as well. ``sudo`` and ``doas`` are in the default list themselves.
_WRAPPERS = frozenset({"sudo", "doas", "env", "nohup", "time", "command", "exec", "nice", "timeout", "stdbuf",
                       "xargs", "caffeinate", "ionice", "chroot", "unbuffer"})
#: Interpreters whose inline program cannot be read here: ``-c`` / ``-e`` makes the call critical.
_INLINE_FLAGS = {"bash": ("-c",), "sh": ("-c",), "zsh": ("-c",), "dash": ("-c",), "ksh": ("-c",),
                 "fish": ("-c",), "python": ("-c",), "python3": ("-c",), "perl": ("-e", "-E"),
                 "ruby": ("-e",), "node": ("-e", "--eval", "-p"), "deno": ("eval",), "php": ("-r",),
                 "osascript": ("-e",)}
#: Shell words that open a command without being one.
_KEYWORDS = frozenset({"!", "if", "then", "else", "elif", "do", "while", "until", "{", "}", "(", ")", "fi",
                       "done"})
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_SEPARATORS = re.compile(r"\|\||&&|[;|&\n(){}]|\$\(|`|<\(|>\(")
_HOME_VARS = re.compile(r"\$\{HOME\}|\$HOME\b")


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
        """The owner-list reasons this call is critical, read from the RAW arguments at capture.

        Never raises: a call the matcher cannot read is critical (``unreadable``), the narrower answer.
        """
        try:
            return self._matches(tool_name, arguments, grant_file_path)
        except Exception:                                  # noqa: BLE001 - fail to the narrower answer
            logger.warning("critical: could not read the call to %s; treating it as critical", tool_name,
                           exc_info=True)
            return ("unreadable",)

    def _matches(self, tool_name: str, arguments: Mapping[str, Any] | None,
                 grant_file_path: str | None) -> tuple[str, ...]:
        args: Mapping[str, Any] = arguments or {}
        reasons: list[str] = []
        if tool_name in self.tools:
            reasons.append(f"tool:{tool_name}")
        command = args.get("command") if tool_name == "bash" else None
        if isinstance(command, str):
            hit = _command_match(command, self.commands)
            if hit:
                reasons.append(f"command:{hit}")
            if self.paths and _mentions_path(command, self.paths):
                reasons.append("path")
        if tool_name in _WRITE_TOOLS:
            path = grant_file_path or next((args.get(k) for k in _WRITE_TOOLS[tool_name]
                                            if isinstance(args.get(k), str)), None)
            if isinstance(path, str) and _writes_under(path, self.paths):
                reasons.append("path")
        return tuple(reasons)


def _words(segment: str) -> list[str]:
    try:
        words = shlex.split(segment, posix=True)
    except ValueError:
        words = segment.split()
    while words and (words[0] in _KEYWORDS or _ASSIGNMENT.match(words[0])):
        words = words[1:]
    return [os.path.basename(w) if i == 0 else w for i, w in enumerate(words)]


def _leads(words: list[str], pattern: list[str]) -> bool:
    if words[:len(pattern)] == pattern:
        return True
    # ``git -C dir push``: git's global options between the command and its verb.
    if len(pattern) > 1 and pattern[0] == "git" and words and words[0] == "git":
        return _drop_option_values(words[1:])[:len(pattern) - 1] == pattern[1:]
    return False


def _commands_in(words: list[str]) -> list[list[str]]:
    """*words* and every command it runs on the rest of its line: after a wrapper (and its options), and after a
    ``find -exec``."""
    found = [words]
    if words and words[0] in _WRAPPERS:
        rest = words[1:]
        for k, w in enumerate(rest):
            if not w.startswith("-") and not _ASSIGNMENT.match(w):
                inner = [os.path.basename(rest[k])] + rest[k + 1:]
                found.extend(_commands_in(inner))
                # an option's value may have been taken for the command (``nice -n 5 rm``): try the next too
                if k + 1 < len(rest):
                    found.extend(_commands_in([os.path.basename(rest[k + 1])] + rest[k + 2:]))
                break
    if words and words[0] == "find":
        for k, w in enumerate(words):
            if w in ("-exec", "-execdir", "-ok", "-okdir") and k + 1 < len(words):
                found.extend(_commands_in([os.path.basename(words[k + 1])] + words[k + 2:]))
    return found


def _command_match(command: str, patterns: Iterable[str]) -> str | None:
    """The first reason any command in *command* is critical, or None. Words compare exactly."""
    split = [p.split() for p in patterns if p.split()]
    for segment in _SEPARATORS.split(command):
        for words in _commands_in(_words(segment)):
            if not words:
                continue
            for pattern in split:
                if _leads(words, pattern):
                    return " ".join(pattern)
            flags = _INLINE_FLAGS.get(words[0])
            if flags and any(w in flags for w in words[1:]):
                return f"{words[0]} {next(w for w in words[1:] if w in flags)}"
            if words[0] == "eval":
                return "eval"
            if words[0] == "find" and "-delete" in words:
                return "find -delete"
    return None


def _drop_option_values(words: list[str]) -> list[str]:
    """git's global options removed (``-C dir``, ``-c k=v`` take a value), so the verb comes first."""
    skip = False
    for i, w in enumerate(words):
        if skip:
            skip = False
            continue
        if w in ("-C", "-c", "--git-dir", "--work-tree", "--namespace"):
            skip = True
            continue
        if w.startswith("-"):
            continue
        return words[i:]
    return []


def _mentions_path(command: str, roots: Iterable[str]) -> bool:
    """Does any word of a shell line name a path under *roots*? A read counts too: the narrower answer."""
    text = _HOME_VARS.sub("~", command)
    try:
        words = shlex.split(text, posix=True)
    except ValueError:
        words = text.split()
    for word in words:
        for part in re.split(r"[<>=]+", word):
            if part and ("/" in part or part.startswith("~")) and _under_any(part, roots, relative_too=True):
                return True
    return False


def _writes_under(path: str, roots: Iterable[str]) -> bool:
    """A write under *roots*, except into a folder a tool writes to in the ordinary way. Resolved first, so
    ``downloads/../config.yaml`` is a write into the configuration."""
    if os.path.isabs(os.path.expanduser(path)) and _under_any(path, _ORDINARY_WRITE_DIRS):
        return False
    return _under_any(path, roots, relative_too=True)


def _under_any(path: str, roots: Iterable[str], *, relative_too: bool = False) -> bool:
    """Is *path* under one of *roots*? Resolved (``~``, ``..``, symlinks). With *relative_too*, a relative path
    is also judged by its own components, because the cwd it will land in (the workspace) is not known here:
    ``.prometheus/x`` or ``a/../.config/prometheus/x`` counts."""
    if relative_too and not os.path.isabs(os.path.expanduser(path)):
        parts = os.path.normpath(path).split(os.sep)
        for root in roots:
            tail = [p for p in os.path.normpath(root).split(os.sep) if p not in ("~", "")]
            for i in range(len(parts)):
                if tail and parts[i:i + len(tail)] == tail:
                    return True
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
