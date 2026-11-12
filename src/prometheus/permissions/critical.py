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

import fnmatch
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
    "git checkout --", "git checkout .", "git restore", "git stash drop", "git stash clear",
    "git worktree remove", "git tag -d", "git tag --delete", "dd", ":",
    "docker rm", "docker rmi", "docker system prune", "docker image prune", "docker volume rm", "docker volume prune",
    "docker compose down", "npm unpublish", "git switch --discard-changes", "git checkout -f", "pip uninstall",
    "uv pip uninstall", "brew uninstall", "npm uninstall",
    # git push, in every form
    "git push",
    # sends
    "curl", "wget", "ssh", "scp", "sftp", "rsync", "nc", "ncat", "socat", "telnet", "ftp", "openssl",
    "mail", "sendmail", "mutt", "osascript", "git send-email",
    "gh", "npm publish", "pnpm publish", "yarn publish", "cargo publish", "poetry publish", "gem push",
    "twine upload", "uv publish", "docker push", "git subtree push", "rclone", "gsutil", "http", "https", "xh",
    "shortcuts",
    # spends
    "aws", "gcloud", "gsutil", "az", "stripe", "kubectl delete", "kubectl apply", "kubectl create",
    "kubectl replace", "kubectl patch", "kubectl scale", "kubectl exec", "terraform apply", "terraform destroy",
    "pulumi up", "pulumi destroy", "wrangler", "vercel", "netlify",
    "flyctl", "fly", "heroku",
    # devices, permissions, the agent's own configuration and services
    "oara", "prometheus", "sudo", "chmod", "chown", "chgrp", "crontab", "launchctl", "systemctl", "security",
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

#: Words that run the rest of their line as a command. After one, EVERY later word is tried as the start of a
#: command (option parsing differs per wrapper, so it is not attempted), and an argument holding a whole command
#: line (``watch 'rm x'``, ``env -S 'rm x'``) is matched as one.
_WRAPPERS = frozenset({"sudo", "doas", "env", "nohup", "time", "command", "exec", "nice", "timeout", "stdbuf",
                       "xargs", "caffeinate", "ionice", "chroot", "unbuffer", "setsid", "flock", "arch", "watch",
                       "parallel", "builtin", "uv", "uvx", "npx", "pipx", "poetry", "pnpm", "yarn", "bunx",
                       "nohup", "taskpolicy", "sandbox-exec"})
#: Run a program that cannot be read here (a script, stdin, a string): always critical as a command word.
_SHELLS = frozenset({"bash", "sh", "zsh", "dash", "ksh", "fish", "csh", "tcsh", "busybox", "su", "script",
                     "source", ".", "eval", "trap", "pwsh", "powershell"})
#: Interpreters: critical with an inline program (``-c``, ``-e`` and friends, also inside a group such as
#: ``-lc``), with ``-`` or fed on stdin; ``-m module`` is matched like a wrapper. Running a named file is not.
_INTERPRETERS = {"python": "c", "python2": "c", "python3": "c", "pypy": "c", "pypy3": "c", "perl": "eE",
                 "ruby": "e", "node": "ep", "bun": "e", "deno": "", "php": "r", "lua": "e", "luajit": "e",
                 "tclsh": "", "rscript": "e", "osascript": "e"}
_AWKS = frozenset({"awk", "gawk", "nawk", "mawk"})
#: Shell words that open a command without being one.
_KEYWORDS = frozenset({"!", "if", "then", "else", "elif", "do", "while", "until", "{", "}", "(", ")", "fi",
                       "done", "case", "esac", "in"})
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
#: Statements: what runs one after another or side by side.
_STATEMENTS = re.compile(r"\|\||&&|[;|&\n]")
#: Finer: every place a command can start, including subshells, groups and substitutions.
_SEPARATORS = re.compile(r"\|\||&&|[;|&\n(){}]|\$\(|`|<\(|>\(")
_HOME_VARS = re.compile(r"\$\{HOME\}|\$HOME\b|\$\{XDG_CONFIG_HOME\}|\$XDG_CONFIG_HOME\b")
_GLOB = re.compile(r"[*?\[\]{}]")



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
    return words


_VERSIONED = re.compile(r"^(python|pypy|perl|ruby|node|lua|php|bash|zsh|tclsh)[0-9][0-9.]*$")


def _base(name: str) -> str:
    match = _VERSIONED.match(name)
    return match.group(1) if match else name


def _name(word: str) -> str:
    """A command word as the shell finds it: the last path part, case folded (the default macOS disk ignores
    case, so ``RM`` runs ``rm``), a version suffix dropped (``python3.12`` is ``python``)."""
    return _base(os.path.basename(word).lower())


def _leads(words: list[str], pattern: list[str]) -> bool:
    """Does the command *words* match *pattern*? The command name, then the verb as the first word after the
    command's own options (``git -C dir push``), then every further pattern word ANYWHERE after the verb
    (``git reset -q --hard``, ``git reset HEAD~3 --hard``)."""
    if not words or _name(words[0]) != _base(pattern[0].lower()):
        return False
    if len(pattern) == 1:
        return True
    if words[1:len(pattern)] == pattern[1:]:
        return True
    rest = _drop_option_values(words[1:])
    return bool(rest) and rest[0] == pattern[1] and all(w in rest[1:] for w in pattern[2:])


def _inline_flag(words: list[str], letters: str) -> bool:
    for w in words[1:]:
        if w == "-":
            return True
        if w in ("--eval", "--command", "eval"):
            return True
        if letters and w.startswith("-") and not w.startswith("--") and any(ch in w[1:] for ch in letters):
            return True
    return False


def _one(words: list[str], patterns: list[list[str]], fed: bool, nested: bool = False) -> str | None:
    """Why the command *words* is critical, or None. *fed*: something is piped or here-doc'd into it.

    *nested*: called from a wrapper's scan, which already tries every later word, so a wrapper met here is not
    scanned again (that recursion was exponential in stacked wrappers: ``env env env ... ls``)."""
    if not words:
        return None
    first = words[0]
    if first.startswith("$") or first.startswith("`") or _GLOB.search(first) or "," in first:
        return "dynamic command"              # its name exists only when the shell runs the line
    name = _name(first)
    for pattern in patterns:
        if _leads(words, pattern):
            return " ".join(pattern)
    if name in _SHELLS:
        return name
    if name in _INTERPRETERS:
        if fed or _inline_flag(words, _INTERPRETERS[name]):
            return f"{name} inline"
        if "-m" in words:                     # python3 -m twine upload: the module and its words
            k = words.index("-m")
            return None if nested else _scan(words[k + 1:], patterns)
        return None
    if name in _AWKS:
        program = " ".join(words[1:])
        if any(t in program for t in ("system", "|", ">", "getline")):
            return "awk program"
        return None
    if name == "find":
        if "-delete" in words:
            return "find -delete"
        # The first -exec only, and through the scan's guard: each later -exec is a later word of that same
        # command, and recursing at every one was exponential (``find . -exec find -exec find ...``).
        for k, w in enumerate(words):
            if w in ("-exec", "-execdir", "-ok", "-okdir"):
                return None if nested else _scan(words[k + 1:], patterns)
        return None
    if name in _WRAPPERS:
        return None if nested else _scan(words[1:], patterns)
    return None


def _scan(words: list[str], patterns: list[list[str]]) -> str | None:
    """After a wrapper: every later word tried as a command, and any word holding a command line matched."""
    for k in range(len(words)):
        if words[k].startswith("-") and _name(words[k]) not in _SHELLS:
            continue
        hit = _one(words[k:], patterns, False, nested=True)
        if hit:
            return hit
        if any(c.isspace() for c in words[k]):
            hit = _match_line(words[k], patterns)
            if hit:
                return hit
    return None


def _match_line(command: str, patterns: list[list[str]]) -> str | None:
    for statement in _STATEMENTS.split(command):
        head = statement.lstrip(" \t\"'({!")
        if head.startswith("$") or head.startswith("`"):
            return "dynamic command"
        if head.startswith(">") and not head.startswith(">>"):
            return "truncate"                 # ``> f`` empties a file
        # awk's program holds braces, which the finer split below cuts apart: judge it whole, here.
        words = _words(statement)
        if words and _name(words[0]) in _AWKS:
            hit = _one(words, patterns, False)
            if hit:
                return hit
    pieces = _SEPARATORS.split(command)
    for i, segment in enumerate(pieces):
        fed = "<<" in segment or (i > 0 and _piped_into(command, segment))
        hit = _one(_words(segment), patterns, fed)
        if hit:
            return hit
    return None


def _piped_into(command: str, segment: str) -> bool:
    at = command.find(segment)
    before = command[:at].rstrip()
    return before.endswith("|") and not before.endswith("||")


#: A shell line longer than this is critical without being read: no lock screen shows it, and it bounds the
#: matcher's cost on hostile input.
MAX_COMMAND_CHARS = 4096
MAX_COMMAND_WORDS = 400


def _command_match(command: str, patterns: Iterable[str]) -> str | None:
    """The first reason any command in *command* is critical, or None."""
    if len(command) > MAX_COMMAND_CHARS or len(command.split()) > MAX_COMMAND_WORDS:
        return "long command"
    return _match_line(command, [p.split() for p in patterns if p.split()])


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


def _home_vars(text: str) -> str:
    return _HOME_VARS.sub(lambda m: "~/.config" if "XDG" in m.group(0) else "~", text)


def _path_words(statement: str) -> list[str]:
    try:
        words = shlex.split(_home_vars(statement), posix=True)
    except ValueError:
        words = _home_vars(statement).split()
    return [part for w in words for part in re.split(r"[<>=]+", w) if part]


def _mentions_path(command: str, roots: Iterable[str]) -> bool:
    """Does the line name a path under *roots*, directly, through a glob, or relative to a ``cd`` it made? A
    read counts too: the narrower answer."""
    roots = tuple(roots)
    cwd: str | None = None
    for statement in _STATEMENTS.split(command):
        words = _path_words(statement)
        if words and words[0] == "cd":
            args = [w for w in words[1:] if w != "--" and not (w.startswith("-") and len(w) == 2)]
            target = args[0] if args else "~"
            if target.startswith(("~", "/")):
                cwd = os.path.expanduser(target)
            elif cwd is not None:
                cwd = os.path.normpath(os.path.join(cwd, target))
            continue
        for part in words:
            if part.startswith("-"):
                continue
            if part.startswith(("~", "/")) or "/" in part:
                if _under_any(part, roots, relative_too=True):
                    return True
            if cwd is not None and not part.startswith(("~", "/")):
                if _under_any(os.path.join(cwd, part), roots):
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
    if _GLOB.search(path):
        return _glob_could_reach(path, roots)
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


def _glob_could_reach(pattern: str, roots: Iterable[str]) -> bool:
    """Could the glob *pattern* name something under one of *roots*? Compared part by part over the root's
    length, so ``~/.prometh?us/x`` and ``~/.config/prometh*/x`` do."""
    parts = os.path.expanduser(pattern).split(os.sep)
    for root in roots:
        base = os.path.expanduser(root).rstrip(os.sep).split(os.sep)
        if len(parts) >= len(base) and all(fnmatch.fnmatchcase(b, p) for b, p in zip(base, parts)):
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
