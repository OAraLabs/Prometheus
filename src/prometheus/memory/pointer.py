"""MEMORY.md pointer management.

Maintains a lightweight index file (~/.prometheus/MEMORY.md) that holds
short pointer entries — one per line — referencing facts in the SQLite store.
Used to surface high-value memories into the system prompt without loading
the full SQLite store on every turn.
"""

from __future__ import annotations

import fcntl
import time
from pathlib import Path

from prometheus.config.paths import get_config_dir

_POINTER_FILE = "MEMORY.md"
_MAX_CHARS = 8000
_DELIMITER = "\n"


def _get_pointer_path() -> Path:
    return get_config_dir() / _POINTER_FILE


class MemoryPointer:
    """Manage MEMORY.md — a bounded, human-readable pointer index.

    Each entry is a single line of text (no newlines within entries).
    Total file size is capped at ``max_chars`` characters; oldest entries
    are pruned when the limit is exceeded.
    """

    def __init__(
        self,
        pointer_path: str | Path | None = None,
        max_chars: int = _MAX_CHARS,
    ) -> None:
        self._path = Path(pointer_path) if pointer_path is not None else _get_pointer_path()
        self._max_chars = max_chars
        if not self._path.exists():
            self._path.write_text("", encoding="utf-8")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_pointer(self, text: str) -> None:
        """Append a pointer entry.  Prunes oldest entries if over char limit."""
        entry = text.strip().replace("\n", " ")
        if not entry:
            return
        with self._lock():
            entries = self._read_entries()
            entries.append(entry)
            entries = self._prune(entries)
            self._write_entries(entries)

    def remove_pointer(self, text: str) -> bool:
        """Remove the first entry that contains *text* as a substring.

        Returns True if an entry was removed.
        """
        needle = text.strip().lower()
        with self._lock():
            entries = self._read_entries()
            new_entries = [e for e in entries if needle not in e.lower()]
            removed = len(new_entries) < len(entries)
            if removed:
                self._write_entries(new_entries)
        return removed

    def replace_pointer(self, old_text: str, new_text: str) -> bool:
        """Replace the first entry matching *old_text* with *new_text*.

        Returns True if a replacement was made.
        """
        needle = old_text.strip().lower()
        replacement = new_text.strip().replace("\n", " ")
        with self._lock():
            entries = self._read_entries()
            replaced = False
            for i, entry in enumerate(entries):
                if needle in entry.lower():
                    entries[i] = replacement
                    replaced = True
                    break
            if replaced:
                entries = self._prune(entries)
                self._write_entries(entries)
        return replaced

    def get_all(self) -> list[str]:
        """Return all pointer entries."""
        return self._read_entries()

    def format_for_prompt(self) -> str:
        """Return entries formatted for injection into a system prompt."""
        entries = self._read_entries()
        if not entries:
            return ""
        lines = "\n".join(f"- {e}" for e in entries)
        return f"## Memory Pointers\n{lines}\n"

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock():
            self._write_entries([])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _read_entries(self) -> list[str]:
        if not self._path.exists():
            return []
        raw = self._path.read_text(encoding="utf-8")
        return [line for line in raw.split(_DELIMITER) if line.strip()]

    def _write_entries(self, entries: list[str]) -> None:
        self._path.write_text(_DELIMITER.join(entries) + ("\n" if entries else ""), encoding="utf-8")

    def _prune(self, entries: list[str]) -> list[str]:
        """Drop oldest entries until total chars fit within max_chars."""
        while entries and sum(len(e) + 1 for e in entries) > self._max_chars:
            entries.pop(0)
        return entries

    def _lock(self):
        """Return a context manager that holds an exclusive fcntl lock."""
        return _FileLock(self._path)


class _FileLock:
    """Exclusive advisory lock on *path* via fcntl.LOCK_EX."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fh = None

    def __enter__(self) -> _FileLock:
        self._fh = open(self._path, "a+", encoding="utf-8")
        fcntl.flock(self._fh, fcntl.LOCK_EX)
        return self

    def __exit__(self, *_: object) -> None:
        if self._fh:
            fcntl.flock(self._fh, fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None
