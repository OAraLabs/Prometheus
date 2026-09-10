"""McpServerStore — REST-managed MCP server definitions (#332, Beacon B1).

Two sources of MCP servers, deliberately separate:

- ``mcp_servers`` in prometheus.yaml — operator-managed, read-only over
  REST. The daemon NEVER writes the YAML: the grant-writer incident (a
  config writer that ate all 540 comments) is the standing reason config
  mutation does not go near that file.
- This store — daemon-owned JSON at ``~/.prometheus/data/mcp_servers.json``
  (the ``devices.db`` precedent: daemon-managed state lives in data/, not
  in the operator's config). REST creates/edits/deletes here; the boot
  path merges these into the config's map, with the YAML winning on a
  name collision so an operator's hand-written entry can never be
  shadowed remotely.

Secrets: a server's ``env`` map may carry credentials for the subprocess
(API keys the MCP server itself needs). They are stored here (0600 file)
and NEVER echoed — readers get ``env_names`` only, same write-only stance
as the provider-key endpoints.

Source: Prometheus (OAra Labs)
License: MIT
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from prometheus.config.paths import get_data_dir

logger = logging.getLogger(__name__)

_STORE_FILENAME = "mcp_servers.json"

# Server names become tool-name prefixes and file keys; same shape the
# sanitizer accepts cleanly, enforced at the door instead of mangled later.
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# The per-server keys the transport resolver understands (camelCase is the
# OpenClaw-donor wire shape resolve_transport parses) plus allowed_tools
# and our own enabled flag. Anything else is refused, not ignored — an
# unknown key silently accepted is how allowed_tools itself sat dead in
# config for a month.
_ALLOWED_KEYS = {
    "command", "args", "env", "cwd", "workingDirectory",
    "connectionTimeoutMs", "url", "headers", "transport",
    "allowed_tools", "enabled",
}

# ── WHAT THIS VALIDATION DOES AND DOES NOT CLAIM ───────────────────────
#
# It does NOT make defining an MCP server safe, and nothing below should be
# read as claiming that. Launching a stdio MCP server IS arbitrary code
# execution by construction: the canonical definition is
# `npx -y some-package`, and whatever that package does is what runs. No
# allowlist of command names changes that — `node -e`, `python -c`,
# `uvx <anything>` and `docker run` are all ordinary, legitimate MCP
# launchers and all of them are "run this code". A filter that appeared to
# make this endpoint safe would be the mechanism-that-reports-itself-working
# failure this codebase keeps finding, in a new place.
#
# The control that actually bounds this surface is at the door, in
# web/server.py: the mutating verbs require the GLOBAL token (a device token
# is the wrong credential for spawning a process), and `mcp.rest_management`
# turns the surface off entirely.
#
# What the checks below DO buy is that a definition MEANS WHAT IT SAYS. An
# operator reading `command: npx, args: [-y, docs-mcp]` off a server card is
# entitled to conclude that is the program that runs. These variables are the
# ones that falsify that reading — they redirect the loader or hand the
# interpreter code to run before it ever reaches the named script, so the
# card and the process stop describing the same thing:
_ENV_HIJACK_NAMES = frozenset({
    # Dynamic-loader injection (glibc / macOS dyld)
    "LD_PRELOAD", "LD_AUDIT", "LD_LIBRARY_PATH",
    "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH",
    # Interpreters that will execute code handed to them via the environment
    "NODE_OPTIONS", "NODE_REPL_EXTERNAL_MODULE",
    "PYTHONSTARTUP", "PYTHONPATH", "PYTHONHOME", "PYTHONEXECUTABLE",
    "BASH_ENV", "ENV", "SHELLOPTS", "PS4",
    "PERL5OPT", "PERL5LIB", "RUBYOPT", "RUBYLIB",
    # Programs that take an executable through their own configuration
    "GIT_SSH", "GIT_SSH_COMMAND", "GIT_EXTERNAL_DIFF", "GIT_PAGER",
})

#: POSIX environment names. A name outside this shape cannot be exported by
#: any normal launcher, so accepting one only stores something unusable.
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _has_control_chars(text: str) -> bool:
    """True if *text* carries a C0 control, DEL, or a NUL.

    argv and environ are NUL-terminated: an embedded NUL truncates the value
    at the exec boundary, so what the store shows and what the kernel receives
    are different strings. Newlines matter for the same reason one layer up —
    they let a single stored field span lines in every log and card that
    renders it.
    """
    return any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in text)


class McpStoreError(ValueError):
    """A definition the store refuses; the message is client-facing."""


class McpServerStore:
    """CRUD over the daemon-owned MCP server definition file."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (get_data_dir() / _STORE_FILENAME)

    # ── IO ─────────────────────────────────────────────────────────

    def load(self) -> dict[str, dict[str, Any]]:
        if not self._path.exists():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            logger.warning(
                "McpServerStore: %s unreadable — treating as empty (REST-"
                "managed servers will be missing until it is fixed)",
                self._path, exc_info=True,
            )
            return {}

    def _save(self, servers: dict[str, dict[str, Any]]) -> None:
        tmp = self._path.with_name(self._path.name + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(servers, fh, indent=2)
        os.replace(tmp, self._path)

    # ── validation ─────────────────────────────────────────────────

    @staticmethod
    def validate(name: str, definition: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(name, str) or not _NAME_RE.match(name):
            raise McpStoreError(
                "server name must be 1-64 chars of [A-Za-z0-9_-]"
            )
        if not isinstance(definition, dict):
            raise McpStoreError("server definition must be an object")
        unknown = set(definition) - _ALLOWED_KEYS
        if unknown:
            raise McpStoreError(
                f"unknown key(s) {sorted(unknown)} — accepted: "
                f"{sorted(_ALLOWED_KEYS)}"
            )
        if not definition.get("command") and not definition.get("url"):
            raise McpStoreError(
                "definition needs a stdio `command` or an http/sse `url`"
            )
        # ── command / args / cwd ───────────────────────────────────────
        # Previously unchecked in full: presence of `command` was the ONLY
        # test, and args/cwd were never looked at on any path. These reach
        # StdioServerParameters verbatim (mcp/runtime.py _connect_stdio).
        command = definition.get("command")
        if command is not None:
            if not isinstance(command, str) or not command.strip():
                raise McpStoreError("command must be a non-empty string")
            if _has_control_chars(command):
                raise McpStoreError("command contains control characters")
        args = definition.get("args")
        if args is not None:
            if not isinstance(args, list) or not all(
                isinstance(a, str) for a in args
            ):
                raise McpStoreError("args must be a list of strings")
            for a in args:
                if _has_control_chars(a):
                    raise McpStoreError(
                        f"args entry {a[:40]!r} contains control characters"
                    )
        for key in ("cwd", "workingDirectory"):
            cwd = definition.get(key)
            if cwd is None:
                continue
            if not isinstance(cwd, str):
                raise McpStoreError(f"{key} must be a string")
            if _has_control_chars(cwd):
                raise McpStoreError(f"{key} contains control characters")
            if cwd.strip() and not Path(cwd).is_absolute():
                raise McpStoreError(
                    f"{key} must be an absolute path — a relative one resolves "
                    "against the DAEMON's working directory, which is not the "
                    "directory whoever wrote this definition was picturing"
                )
            # Deliberately not an existence check: a definition may legitimately
            # be stored before its directory is created. A regular FILE is the
            # error worth catching, because it can never become a valid cwd.
            if cwd.strip() and Path(cwd).is_file():
                raise McpStoreError(f"{key} is a file, not a directory: {cwd}")
        # ── env ────────────────────────────────────────────────────────
        env = definition.get("env")
        if env is not None:
            if not isinstance(env, dict) or not all(
                isinstance(k, str) and isinstance(v, str)
                for k, v in env.items()
            ):
                raise McpStoreError("env must be a {NAME: value} string map")
            for k, v in env.items():
                if _has_control_chars(v) or _has_control_chars(k):
                    raise McpStoreError(
                        f"env {k!r} contains control characters"
                    )
                if not _ENV_NAME_RE.match(k):
                    raise McpStoreError(
                        f"env name {k!r} is not a POSIX environment name "
                        "([A-Za-z_][A-Za-z0-9_]*)"
                    )
                if k.upper() in _ENV_HIJACK_NAMES:
                    raise McpStoreError(
                        f"env {k!r} is refused: it changes WHICH program runs "
                        "rather than configuring the one named in `command`, "
                        "so the stored definition would stop describing the "
                        "process. Set it in the MCP server's own launcher if "
                        "it is genuinely needed."
                    )
        allowed = definition.get("allowed_tools")
        if allowed is not None and (
            not isinstance(allowed, list)
            or not all(isinstance(t, str) for t in allowed)
        ):
            raise McpStoreError("allowed_tools must be a list of strings")
        return definition

    # ── CRUD ───────────────────────────────────────────────────────

    def upsert(self, name: str, definition: dict[str, Any]) -> None:
        definition = self.validate(name, definition)
        servers = self.load()
        servers[name] = definition
        self._save(servers)
        logger.info("MCP store: upserted server %r", name)

    def patch(self, name: str, changes: dict[str, Any]) -> dict[str, Any]:
        servers = self.load()
        if name not in servers:
            raise KeyError(name)
        merged = {**servers[name], **changes}
        # A PATCH that explicitly nulls a key removes it.
        merged = {k: v for k, v in merged.items() if v is not None}
        self.validate(name, merged)
        servers[name] = merged
        self._save(servers)
        logger.info("MCP store: patched server %r (%s)", name, sorted(changes))
        return merged

    def delete(self, name: str) -> bool:
        servers = self.load()
        if name not in servers:
            return False
        del servers[name]
        self._save(servers)
        logger.info("MCP store: deleted server %r", name)
        return True

    # ── projection ─────────────────────────────────────────────────

    @staticmethod
    def public_view(definition: dict[str, Any]) -> dict[str, Any]:
        """The definition with secrets stripped: env VALUES never leave the
        daemon — readers learn the names and that they are set, nothing
        more (the provider-keys stance)."""
        out = {k: v for k, v in definition.items() if k != "env"}
        env = definition.get("env")
        if isinstance(env, dict):
            out["env_names"] = sorted(env)
        return out


def merged_server_configs(config: dict[str, Any],
                          store: McpServerStore) -> dict[str, dict[str, Any]]:
    """YAML servers + store servers, YAML winning on collision.

    ``enabled: false`` (store-managed only) keeps the definition but
    excludes it from the merge — the runtime never sees it, so its tools
    are structurally absent rather than registered-then-hidden.
    """
    merged: dict[str, dict[str, Any]] = {}
    for name, definition in store.load().items():
        if definition.get("enabled", True):
            merged[name] = {
                k: v for k, v in definition.items() if k != "enabled"
            }
    yaml_servers = config.get("mcp_servers") or {}
    if isinstance(yaml_servers, dict):
        for name, definition in yaml_servers.items():
            if name in merged:
                logger.warning(
                    "MCP: server %r defined in BOTH prometheus.yaml and the "
                    "REST store — the yaml definition wins; delete one",
                    name,
                )
            if isinstance(definition, dict):
                merged[name] = definition
    return merged
