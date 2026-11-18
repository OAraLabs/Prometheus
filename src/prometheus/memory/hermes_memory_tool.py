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
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from prometheus.config.paths import get_config_dir
from prometheus.security.invisible_text import describe_invisible, find_invisible
from prometheus.engine.tool_context import APPROVED_CARD, OWNER_APPROVED
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
        # Not created here: building an approval card makes a store, and a card the owner denies must leave nothing
        # behind. A missing file reads as empty; the first write creates it.

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
        hidden = find_invisible(entry or "")
        if hidden:
            return f"refused: the entry contains {describe_invisible(hidden)}, which hide text from a person reading it"
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

    def read_entries(self) -> list[str] | None:
        """The entries, or None when the file exists but can't be read as text. Never creates the file."""
        try:
            return self._parse()
        except (OSError, UnicodeDecodeError):
            return None

    def change(self, arguments: "MemoryToolInput", approved_card: str | None) -> str:
        """Make the one change *arguments* asks for, in ONE locked step from the card to the write.

        Under the file lock: read the entries, rebuild the approval card from them and refuse if it differs from
        *approved_card* (another approval or an edit came first), then change exactly what that card showed:
        an add or replace that would overflow the file is refused (nothing is pruned to make room), and a remove or
        replace touches exactly one entry. Returns ``"added"``/``"replaced"``/``"removed"`` or a refusal.
        """
        label = self._path.name
        with _ExclusiveLock(self._path):
            entries = self.read_entries()
            if entries is None:
                return f"refused: {label} can't be read as text; fix it in the memory editor"
            # Fail closed: no approved card, no change.
            if approved_card is None or _card(arguments, entries, self._max_chars, label) != approved_card:
                return ("refused: memory changed after the owner approved this, so it no longer does what they were "
                        "shown. Ask again.")
            new = self._sanitize(arguments.entry or "")
            if arguments.operation in ("add", "replace"):
                hidden = find_invisible(arguments.entry or "")
                if hidden:
                    return (f"refused: the entry contains {describe_invisible(hidden)}, which hide text from a "
                            "person reading it")
                if not new:
                    return "entry is empty or contained prohibited content"
            if arguments.operation == "add":
                if _size(entries) + len(new) + 1 > self._max_chars:
                    return f"refused: {label} is full; remove or replace an entry first (each asks the owner)"
                entries.append(new)
                self._flush(entries)
                return "added"
            needle = (arguments.old_entry if arguments.operation == "replace" else arguments.entry) or ""
            hits = _hits(entries, needle)
            if not hits:
                return f"no entry found matching: {needle}"
            if len(hits) > 1:
                return f"refused: {len(hits)} entries contain {needle!r}; quote more of the one you mean"
            (i,) = hits
            if arguments.operation == "remove":
                del entries[i]
                self._flush(entries)
                return "removed"
            if _size(entries) - len(entries[i]) + len(new) > self._max_chars:
                return f"refused: {label} would be full; remove an entry first (each asks the owner)"
            entries[i] = new
            self._flush(entries)
            return "replaced"


    def replace(self, old_text: str, new_text: str) -> str:
        """Replace the one entry containing *old_text*; refused when it matches several. Returns status."""
        hidden = find_invisible(new_text or "")
        if hidden:
            return (f"refused: the new entry contains {describe_invisible(hidden)}, which hide text from a person "
                    "reading it")
        new_entry = self._sanitize(new_text)
        if not new_entry:
            return "new entry is empty or contained prohibited content"
        needle = old_text.strip().lower()
        with _ExclusiveLock(self._path):
            entries = self._parse()
            hits = [i for i, e in enumerate(entries) if needle and needle in e.lower()]
            if len(hits) > 1:
                return f"refused: {len(hits)} entries contain {old_text!r}; quote more of the one you mean"
            if hits:
                entries[hits[0]] = new_entry
                entries = self._prune(entries)
                self._flush(entries)
                return "replaced"
        return f"no entry found matching: {old_text}"

    def remove(self, text: str) -> str:
        """Remove the one entry containing *text*; refused when it matches several. Returns status.

        It used to remove EVERY entry containing the text, so "the" could empty the file while the approval showed
        only "the" (GHSA-f68r-xfv4-83fr): a change is now exactly the one entry the owner was shown.
        """
        needle = text.strip().lower()
        with _ExclusiveLock(self._path):
            entries = self._parse()
            hits = [i for i, e in enumerate(entries) if needle and needle in e.lower()]
            if not hits:
                return f"no entry found matching: {text}"
            if len(hits) > 1:
                return f"refused: {len(hits)} entries contain {text!r}; quote more of the one you mean"
            del entries[hits[0]]
            self._flush(entries)
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


#: The longest entry the memory tool writes. Every change is shown to the owner for approval, and every surface
#: must show the new text whole: the worker path caps a reason at 500 characters (GHSA-f68r-xfv4-83fr).
MEMORY_ENTRY_MAX_CHARS = 400


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
        max_length=MEMORY_ENTRY_MAX_CHARS,
        description=f"Entry text for add/remove, or new entry text for replace (at most {MEMORY_ENTRY_MAX_CHARS} "
                    "characters: one fact per entry).",
    )
    old_entry: str | None = Field(
        default=None,
        description="Existing entry text to match for replace.",
    )

    @field_validator("entry")
    @classmethod
    def _shown_whole(cls, value: str | None) -> str | None:
        """One line, no control characters, and short enough as SHOWN (its repr) to fit every approval surface.

        The file keeps one entry per line, so a newline would split it; and the approval shows the entry's repr,
        where each control character, quote or backslash takes more room than one (GHSA-f68r-xfv4-83fr).
        """
        if value is None:
            return value
        import unicodedata

        if any(unicodedata.category(ch) in ("Cc", "Zl", "Zp") for ch in value) or len(value.splitlines()) > 1:
            raise ValueError("an entry is one line of text: no newlines, line separators or other control characters")
        if len(repr(value)) - 2 > MEMORY_ENTRY_MAX_CHARS:
            raise ValueError(f"an entry is at most {MEMORY_ENTRY_MAX_CHARS} characters as shown")
        return value


#: The shortest card an approval is shown on (the worker path caps a reason at 500 characters).
_CARD_MAX = 500


def _size(entries: list[str]) -> int:
    """The file's size for its limit: each entry and its newline."""
    return sum(len(e) + 1 for e in entries)


def _hits(entries: list[str], needle: str) -> list[int]:
    """The positions of the entries that contain *needle*, case-insensitively."""
    folded = (needle or "").strip().lower()
    return [i for i, e in enumerate(entries) if folded and folded in e.lower()]


def _fit(text: str, room: int) -> str:
    """*text* quoted as the card shows it (its repr), at most *room* characters.

    Cut short, it ends in "…" and the first 16 hex of the whole text's sha256, so a different text with the same
    start never makes the same card (the card is what the write is checked against).
    """
    shown = repr(text)
    if len(shown) <= room:
        return shown
    import hashlib

    mark = " #" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    keep = text
    while keep and len(repr(keep + "…")) + len(mark) > room:
        keep = keep[: max(len(keep) - max(len(repr(keep)) + len(mark) - room, 1), 0)]
    return repr(keep + "…") + mark if keep else mark.strip()


def _card(arguments: "MemoryToolInput", entries: list[str] | None, max_chars: int, label: str) -> str:
    """The approval card for *arguments* against the file's current *entries* (None: unreadable).

    An add shows the new entry whole. A remove or replace shows the entry it will actually touch, not the text the
    call matched it by; when nothing or several match, or the change would not fit, the card says nothing will
    change. The new text comes first and is always whole (``MEMORY_ENTRY_MAX_CHARS``, counted as shown); anything
    after it is shortened to fit ``_CARD_MAX``.
    """
    head = f"{label}, in every prompt: "
    nothing = "; nothing will change"
    if entries is None:
        return f"{head}{arguments.operation}: the file can't be read as text{nothing}"
    new = (arguments.entry or "").strip()
    if arguments.operation == "add":
        full = "; the file is full, so nothing will change" if _size(entries) + len(new) + 1 > max_chars else ""
        return f"{head}add {arguments.entry!r}{full}"
    needle = (arguments.old_entry if arguments.operation == "replace" else arguments.entry) or ""
    hits = _hits(entries, needle)
    lead = f"{head}replace with {arguments.entry!r}, " if arguments.operation == "replace" else head
    if len(hits) == 1:
        hit = entries[hits[0]]
        if arguments.operation == "replace" and _size(entries) - len(hit) + len(new) > max_chars:
            return f"{lead}the file would be full{nothing}"
        verb = "was " if arguments.operation == "replace" else "remove "
        return f"{lead}{verb}{_fit(hit, _CARD_MAX - len(lead) - len(verb))}"
    what = "no entry matches" if not hits else f"{len(hits)} entries match"
    return f"{lead}{what} {_fit(needle, _CARD_MAX - len(lead) - len(what) - 1 - len(nothing))}{nothing}"


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
    find_me = ("remember a fact about the user", "save to persistent memory")

    def is_read_only(self, arguments: MemoryToolInput) -> bool:
        return arguments.operation == "list"

    def requires_owner_approval(self, arguments: MemoryToolInput) -> str | None:
        """Every change asks the owner, with a card showing exactly what will change (``_card``)."""
        if arguments.operation == "list":
            return None
        store = get_memory_store() if arguments.target == "memory" else get_user_store()
        label = "MEMORY.md" if arguments.target == "memory" else "USER.md"
        return _card(arguments, store.read_entries(), store._max_chars, label)

    async def execute(self, arguments: MemoryToolInput, context: ToolExecutionContext) -> ToolResult:
        # Gateway trust (0.9.17): only the owner changes what the agent remembers. A turn a group member asked for
        # may list memory, never add, replace or remove.
        if arguments.operation != "list" and (context.metadata or {}).get("turn_trusted") is False:
            return ToolResult(
                output="Not saved: only your owner can change what I remember. This request came from "
                       f"{(context.metadata or {}).get('turn_author') or 'someone else in this chat'}.",
                is_error=True,
            )
        if arguments.operation != "list" and not OWNER_APPROVED.get():
            # GHSA-f68r-xfv4-83fr: these files ride every system prompt, so a change is the owner's own, approved in
            # the approval queue (requires_owner_approval below). Called any other way, with no approval channel or
            # outside the agent loop, it is refused.
            return ToolResult(
                output="Not saved: a change to memory needs the owner's approval, and none was given for this one.",
                is_error=True,
            )
        store = get_memory_store() if arguments.target == "memory" else get_user_store()

        target_label = "MEMORY.md" if arguments.target == "memory" else "USER.md"

        if arguments.operation == "list":
            entries = store.list_entries()
            if not entries:
                return ToolResult(output="(no entries)")
            return ToolResult(output="\n".join(f"- {e}" for e in entries))

        if arguments.operation not in ("add", "remove", "replace"):
            return ToolResult(output=f"unknown operation: {arguments.operation}", is_error=True)
        if not arguments.entry or (arguments.operation == "replace" and not arguments.old_entry):
            need = "both 'old_entry' and 'entry' are" if arguments.operation == "replace" else "'entry' is"
            return ToolResult(output=f"{need} required for {arguments.operation}", is_error=True)
        result = store.change(arguments, APPROVED_CARD.get())
        done = {"add": "added", "remove": "removed", "replace": "replaced"}[arguments.operation]
        if result != done:
            return ToolResult(output=result, is_error=True)
        await _maybe_emit_memory_updated(target=target_label, operation=arguments.operation, entry=arguments.entry)
        return ToolResult(output=result)


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
