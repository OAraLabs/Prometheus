"""File-backed memory management — MEMORY.md + USER.md.

Adapted from Hermes agent's memory_tool.py (560 lines).
Provides:
  - FileMemoryStore: manages bounded MEMORY.md and USER.md files with
    add/replace/remove, char limit enforcement, file locking, and
    system prompt formatting.
  - MemoryTool: Prometheus BaseTool wrapper around FileMemoryStore.
"""

from __future__ import annotations

import fcntl
import re
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from prometheus.config.paths import get_config_dir
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

_MEMORY_FILE = "MEMORY.md"
_USER_FILE = "USER.md"
_MEMORY_MAX_CHARS = 12_000
_USER_MAX_CHARS = 8_000
_SECURITY_PATTERNS = re.compile(
    r"(ignore previous|disregard|forget everything|system prompt|jailbreak)",
    re.IGNORECASE,
)


def _get_memory_path() -> Path:
    return get_config_dir() / _MEMORY_FILE


def _get_user_path() -> Path:
    return get_config_dir() / _USER_FILE


# ------------------------------------------------------------------
# FileMemoryStore
# ------------------------------------------------------------------


class FileMemoryStore:
    """Manage a bounded markdown memory file with add/replace/remove.

    Parameters
    ----------
    path:
        Path to the markdown file (MEMORY.md or USER.md).
    max_chars:
        Maximum total characters. Oldest entries are pruned when exceeded.
    """

    def __init__(self, path: Path, max_chars: int) -> None:
        self._path = path
        self._max_chars = max_chars
        if not self._path.exists():
            self._path.write_text("", encoding="utf-8")

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add(self, entry: str) -> str:
        """Append an entry. Returns 'added' or an error message."""
        entry = self._sanitize(entry)
        if not entry:
            return "entry is empty or contained prohibited content"
        with _ExclusiveLock(self._path):
            entries = self._parse()
            entries.append(entry)
            entries = self._prune(entries)
            self._flush(entries)
        return "added"

    def replace(self, old_text: str, new_text: str) -> str:
        """Replace the first entry containing *old_text*. Returns status."""
        new_entry = self._sanitize(new_text)
        if not new_entry:
            return "new entry is empty or contained prohibited content"
        needle = old_text.strip().lower()
        with _ExclusiveLock(self._path):
            entries = self._parse()
            for i, e in enumerate(entries):
                if needle in e.lower():
                    entries[i] = new_entry
                    entries = self._prune(entries)
                    self._flush(entries)
                    return "replaced"
        return f"no entry found matching: {old_text}"

    def remove(self, text: str) -> str:
        """Remove the first entry containing *text*. Returns status."""
        needle = text.strip().lower()
        with _ExclusiveLock(self._path):
            entries = self._parse()
            new_entries = [e for e in entries if needle not in e.lower()]
            if len(new_entries) == len(entries):
                return f"no entry found matching: {text}"
            self._flush(new_entries)
        return "removed"

    def clear(self) -> None:
        """Remove all entries."""
        with _ExclusiveLock(self._path):
            self._flush([])

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def list_entries(self) -> list[str]:
        """Return all entries."""
        return self._parse()

    def format_for_prompt(self, header: str) -> str:
        """Return entries as a markdown section for injection into a system prompt."""
        entries = self._parse()
        if not entries:
            return ""
        body = "\n".join(f"- {e}" for e in entries)
        return f"## {header}\n{body}\n"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _parse(self) -> list[str]:
        if not self._path.exists():
            return []
        raw = self._path.read_text(encoding="utf-8")
        return [line.strip() for line in raw.splitlines() if line.strip()]

    def _flush(self, entries: list[str]) -> None:
        self._path.write_text("\n".join(entries) + ("\n" if entries else ""), encoding="utf-8")

    def _prune(self, entries: list[str]) -> list[str]:
        while entries and sum(len(e) + 1 for e in entries) > self._max_chars:
            entries.pop(0)
        return entries

    def _sanitize(self, text: str) -> str:
        cleaned = text.strip().replace("\n", " ")
        if _SECURITY_PATTERNS.search(cleaned):
            return ""
        return cleaned


# ------------------------------------------------------------------
# Module-level convenience stores
# ------------------------------------------------------------------


def get_memory_store() -> FileMemoryStore:
    """Return the MEMORY.md store."""
    return FileMemoryStore(_get_memory_path(), _MEMORY_MAX_CHARS)


def get_user_store() -> FileMemoryStore:
    """Return the USER.md store."""
    return FileMemoryStore(_get_user_path(), _USER_MAX_CHARS)


def format_memory_for_prompt() -> str:
    """Return both MEMORY.md and USER.md formatted for a system prompt."""
    memory_section = get_memory_store().format_for_prompt("Memory")
    user_section = get_user_store().format_for_prompt("User Model")
    parts = [s for s in [memory_section, user_section] if s]
    return "\n".join(parts)


# ------------------------------------------------------------------
# MemoryTool — Prometheus BaseTool wrapper
# ------------------------------------------------------------------

MemoryOperation = Literal["add", "replace", "remove", "list"]
MemoryTarget = Literal["memory", "user"]


class MemoryToolInput(BaseModel):
    """Arguments for memory management."""

    operation: MemoryOperation = Field(
        description="Operation: 'add', 'replace', 'remove', or 'list'."
    )
    target: MemoryTarget = Field(
        default="memory",
        description="File target: 'memory' (MEMORY.md) or 'user' (USER.md).",
    )
    entry: str | None = Field(
        default=None,
        description="Entry text for add/remove, or new entry text for replace.",
    )
    old_entry: str | None = Field(
        default=None,
        description="Existing entry text to match for replace.",
    )


class MemoryTool(BaseTool):
    """Read and write persistent memory entries in MEMORY.md and USER.md."""

    name = "memory"
    description = (
        "Manage persistent memory entries. "
        "Use 'add' to store a new fact, 'replace' to update, "
        "'remove' to delete, 'list' to read all entries."
    )
    input_model = MemoryToolInput

    def is_read_only(self, arguments: MemoryToolInput) -> bool:
        return arguments.operation == "list"

    async def execute(self, arguments: MemoryToolInput, context: ToolExecutionContext) -> ToolResult:
        del context
        store = get_memory_store() if arguments.target == "memory" else get_user_store()

        if arguments.operation == "list":
            entries = store.list_entries()
            if not entries:
                return ToolResult(output="(no entries)")
            return ToolResult(output="\n".join(f"- {e}" for e in entries))

        if arguments.operation == "add":
            if not arguments.entry:
                return ToolResult(output="'entry' is required for add", is_error=True)
            result = store.add(arguments.entry)
            return ToolResult(output=result, is_error=result not in {"added"})

        if arguments.operation == "remove":
            if not arguments.entry:
                return ToolResult(output="'entry' is required for remove", is_error=True)
            result = store.remove(arguments.entry)
            return ToolResult(output=result)

        if arguments.operation == "replace":
            if not arguments.old_entry or not arguments.entry:
                return ToolResult(
                    output="both 'old_entry' and 'entry' are required for replace",
                    is_error=True,
                )
            result = store.replace(arguments.old_entry, arguments.entry)
            return ToolResult(output=result)

        return ToolResult(output=f"unknown operation: {arguments.operation}", is_error=True)


# ------------------------------------------------------------------
# File locking
# ------------------------------------------------------------------


class _ExclusiveLock:
    """Context manager for exclusive fcntl advisory lock."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fh = None

    def __enter__(self) -> _ExclusiveLock:
        self._fh = open(self._path, "a+", encoding="utf-8")
        fcntl.flock(self._fh, fcntl.LOCK_EX)
        return self

    def __exit__(self, *_: object) -> None:
        if self._fh:
            fcntl.flock(self._fh, fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None
