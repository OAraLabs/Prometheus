"""The outside-agent registry — a named, allowlisted command per agent.

Nothing outside the registry runs: the driver launches ONLY a command an
operator named in ``acp.clients`` (prometheus.yaml), never a string from a
model, a transcript or a request body. The registry ships EMPTY; each
outside agent is an explicit operator grant, like a paired device.

Claude Code is the first intended entry (``claude-code-acp`` or the
vendor's ACP adapter); the registry does not hardcode it — the operator
names the command, and the entry says what the agent IS.

Source: novel code for Prometheus, 2026-11-25.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any, Mapping


class UnknownAgent(Exception):
    """The name is not in the registry. Nothing launches."""

    def __init__(self, name: str) -> None:
        super().__init__(f"no outside agent named {name!r} is registered")
        self.name = name


@dataclass(frozen=True)
class OutsideAgent:
    """One allowlisted outside agent.

    ``command`` is the argv the driver launches (a list, never a shell
    string). The driver evaluates every ask through the SAME gate as
    Prometheus's own tool calls (D13): a gate ALLOW is answered directly
    (audited, naming the agent); a gate APPROVE goes to the owner's queue,
    approve-once. Registering an agent here is the operator's grant to
    LAUNCH it — never a grant to skip the gate.
    """

    name: str
    command: tuple[str, ...]
    description: str = ""


def registry_from_config(config: Mapping[str, Any] | None) -> dict[str, OutsideAgent]:
    """Read ``acp.clients`` — a mapping of name -> {command, description?}.

    ``command`` may be a list (preferred) or a string (shlex-split, never
    shell-executed). A malformed entry is skipped with nothing registered
    under its name: fail closed, an operator typo must not half-register.
    """
    section = (config or {}).get("acp")
    if not isinstance(section, Mapping):
        return {}
    clients = section.get("clients")
    if not isinstance(clients, Mapping):
        return {}
    out: dict[str, OutsideAgent] = {}
    for name, entry in clients.items():
        if not isinstance(name, str) or not name or not isinstance(entry, Mapping):
            continue
        command = entry.get("command")
        if isinstance(command, str):
            try:
                command = shlex.split(command)
            except ValueError:
                continue
        if not isinstance(command, (list, tuple)) or not command:
            continue
        if not all(isinstance(c, str) and c for c in command):
            continue
        out[name] = OutsideAgent(
            name=name,
            command=tuple(command),
            description=str(entry.get("description", "") or ""),
        )
    return out


def resolve(registry: Mapping[str, OutsideAgent], name: str) -> OutsideAgent:
    agent = registry.get(name)
    if agent is None:
        raise UnknownAgent(name)
    return agent