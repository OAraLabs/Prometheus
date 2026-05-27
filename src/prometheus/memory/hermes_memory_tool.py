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

# Sprint S1 Stream 2: module-level SignalBus reference. Wired by daemon.py
# inside the SENTINEL block via ``set_memory_signal_bus(signal_bus)``. None
# until the bus exists; emission is best-effort and never raises.
_signal_bus: object | None = None


def set_memory_signal_bus(bus: object | None) -> None:
    """Register the SignalBus so MemoryTool can emit memory_updated.

    Matches the pattern used by ``tools/builtin/sentinel_status.py``
    (set_sentinel_components). Called once from daemon startup after the
    SignalBus is constructed.
    """
    global _signal_bus
    _signal_bus = bus


class MemoryOverflowError(Exception):
    """Raised when a single entry exceeds the file's character limit.

    Adapted from Hermes's "MEMORY.md at capacity, consolidate before adding
    new facts" pattern, except Hermes upstream does NOT actually enforce a
    char limit (verified against agent/memory_manager.py + prompt_builder.py
    in the public repo) — that's a Prometheus addition. Existing
    prune-oldest semantics in FileMemoryStore handle the common case
    silently; this error surfaces ONLY when a single entry is so large
    that even fully draining the file wouldn't make room. The agent loop
    is expected to catch and inject a consolidation prompt, then retry.

    See Sprint 1 design notes for the policy choice.
    """

    def __init__(self, target: str, entry_chars: int, limit: int) -> None:
        super().__init__(
            f"{target} entry would be {entry_chars} chars, limit is {limit}. "
            f"Consolidate before adding new facts."
        )
        self.target = target
        self.entry_chars = entry_chars
        self.limit = limit


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
        """Append an entry. Returns 'added' or an error message.

        Sprint S1: a single entry longer than ``max_chars`` cannot be made
        to fit even by draining the file completely; we raise
        :class:`MemoryOverflowError` so the agent loop can inject a
        consolidation prompt and retry. Normal (smaller) entries still
        prune-oldest silently — that's the existing contract.
        """
        entry = self._sanitize(entry)
        if not entry:
            return "entry is empty or contained prohibited content"
        # Hard ceiling: even a fully empty file cannot hold this entry.
        if len(entry) > self._max_chars:
            raise MemoryOverflowError(self._path.name, len(entry), self._max_chars)
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
        "Manage persistent memory entries. Call 'add' when the user "
        "shares a durable fact about themselves (preferences, key "
        "contacts, ongoing projects) or about the system "
        "(infrastructure, conventions, decisions) that should persist "
        "across sessions. Use 'replace' to update, 'remove' to delete, "
        "'list' to read all entries."
    )
    input_model = MemoryToolInput

    def is_read_only(self, arguments: MemoryToolInput) -> bool:
        return arguments.operation == "list"

    async def execute(self, arguments: MemoryToolInput, context: ToolExecutionContext) -> ToolResult:
        del context
        store = get_memory_store() if arguments.target == "memory" else get_user_store()

        target_label = "MEMORY.md" if arguments.target == "memory" else "USER.md"

        if arguments.operation == "list":
            entries = store.list_entries()
            if not entries:
                return ToolResult(output="(no entries)")
            return ToolResult(output="\n".join(f"- {e}" for e in entries))

        if arguments.operation == "add":
            if not arguments.entry:
                return ToolResult(output="'entry' is required for add", is_error=True)
            try:
                result = store.add(arguments.entry)
            except MemoryOverflowError as exc:
                # Surface the overflow plainly so the agent loop / caller
                # can react. Result is_error so adapters route it to the
                # retry path.
                return ToolResult(
                    output=(
                        f"MemoryOverflowError: {exc}. "
                        f"Consolidate {target_label} (merge duplicates, drop "
                        f"stale items) and try again."
                    ),
                    is_error=True,
                )
            await _maybe_emit_memory_updated(
                target=target_label,
                operation="add",
                entry=arguments.entry,
            )
            return ToolResult(output=result, is_error=result not in {"added"})

        if arguments.operation == "remove":
            if not arguments.entry:
                return ToolResult(output="'entry' is required for remove", is_error=True)
            result = store.remove(arguments.entry)
            if result == "removed":
                await _maybe_emit_memory_updated(
                    target=target_label,
                    operation="remove",
                    entry=arguments.entry,
                )
            return ToolResult(output=result)

        if arguments.operation == "replace":
            if not arguments.old_entry or not arguments.entry:
                return ToolResult(
                    output="both 'old_entry' and 'entry' are required for replace",
                    is_error=True,
                )
            try:
                result = store.replace(arguments.old_entry, arguments.entry)
            except MemoryOverflowError as exc:
                return ToolResult(
                    output=(
                        f"MemoryOverflowError: {exc}. "
                        f"Consolidate {target_label} and try again."
                    ),
                    is_error=True,
                )
            if result == "replaced":
                await _maybe_emit_memory_updated(
                    target=target_label,
                    operation="replace",
                    entry=arguments.entry,
                )
            return ToolResult(output=result)

        return ToolResult(output=f"unknown operation: {arguments.operation}", is_error=True)


async def _maybe_emit_memory_updated(
    *,
    target: str,
    operation: str,
    entry: str,
) -> None:
    """Best-effort emission of ``memory_updated`` on the module SignalBus."""
    bus = _signal_bus
    if bus is None:
        return
    try:
        from prometheus.sentinel.signals import ActivitySignal

        await bus.emit(ActivitySignal(
            kind="memory_updated",
            payload={
                "target": target,
                "operation": operation,
                "entry_preview": entry[:120],
            },
            source="memory_tool",
        ))
    except Exception:
        # Never let signalling break a memory write.
        pass


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
