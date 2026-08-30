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
        env = definition.get("env")
        if env is not None:
            if not isinstance(env, dict) or not all(
                isinstance(k, str) and isinstance(v, str)
                for k, v in env.items()
            ):
                raise McpStoreError("env must be a {NAME: value} string map")
            for k, v in env.items():
                if any(ord(ch) < 0x20 for ch in v) or any(
                    ord(ch) < 0x20 for ch in k
                ):
                    raise McpStoreError(
                        f"env {k!r} contains control characters"
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
