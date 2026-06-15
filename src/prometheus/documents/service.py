"""Documents Editor service — confined disk-backed read/save/edit (Phase A).

ADDITIVE by construction — this service builds NO new edit engine. It reuses:

- :meth:`prometheus.coding.sandbox.ProcessSandbox.resolve` for path
  confinement (the #37 validation pattern: join → ``resolve()`` follows
  symlinks → must be under root → not under a denied path), rooted at a
  GENERAL documents root instead of coding mode's per-run clone jail. We never
  call the sandbox's ``run()`` — only its path confinement.
- :class:`prometheus.coding.tools.CodeStrReplaceTool` for edits — the exact
  unique-match find/replace with the same loud no-match / multi-match errors.

It then composes a SecurityGate's denied-paths ON TOP (defense in depth: a path
that is *under the documents root but also under a denied prefix* is still
rejected — same pattern as coding mode's sandbox validation in #37).

The model never carries the whole document in context per turn: the manual ops
(read / save / edit) are model-free, and the AI ``suggest`` path takes an
injected generator so the model wiring lives at the call site (the REST route),
not in this service — keeping the unit tests deterministic and honouring the
bakeoff overhead finding (no full-document-in-system-prompt).
"""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Iterable

from prometheus.coding.sandbox import ProcessSandbox, SandboxViolation
from prometheus.coding.tools import CodeStrReplaceArgs, CodeStrReplaceTool
from prometheus.tools.base import ToolExecutionContext

log = logging.getLogger(__name__)

# Read/preview cap — matches the files-preview and coding-diff caps (256 KB).
_READ_CAP = 256 * 1024
# A single-shot suggest includes the (bounded) document once. Larger docs need
# span-scoped viewing — explicitly a follow-up (see module docstring).
_SUGGEST_DOC_CAP = 64 * 1024
_LIST_MAX_ENTRIES = 2_000


# --------------------------------------------------------------------------- #
# Result / error types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DocumentEntry:
    """One child of a listed directory."""

    name: str
    type: str  # "dir" | "file"
    size: int
    mtime: float


@dataclass(frozen=True)
class DocumentEditResult:
    """Outcome of an apply-edit. ``ok=False`` carries the primitive's loud error
    text (no-match / multi-match / empty-old_str) UNCHANGED — never a write."""

    ok: bool
    diff: str
    error: str


@dataclass(frozen=True)
class SuggestedEdit:
    """A proposed redline — returned WITHOUT touching disk (suggest mode)."""

    find: str
    replace: str
    reason: str
    applicable: bool  # `find` occurs exactly once in the current document
    note: str         # why not applicable (0 / N matches), else ""


class DocumentsError(Exception):
    """A documents operation failed with an HTTP-mappable cause."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


# (document_text, instruction) -> list of raw {find, replace, reason} dicts.
# Injected at the call site; the REST route supplies a model-backed generator,
# tests supply a deterministic stub. The service validates + never applies.
EditGenerator = Callable[[str, str], Awaitable[list[dict]]]


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #


class DocumentsService:
    """Confined read / save / edit / suggest over a single documents root."""

    def __init__(
        self,
        root: str | Path,
        *,
        denied_paths: Iterable[str] = (),
        gate: object | None = None,  # SecurityGate-like (.evaluate(...) -> decision)
        read_cap: int = _READ_CAP,
    ) -> None:
        root_path = Path(root).expanduser()
        root_path.mkdir(parents=True, exist_ok=True)
        # Reuse the coding sandbox PURELY for its path confinement (resolve()).
        # The sandbox's run()/env-scrub machinery is never exercised here.
        self._sandbox = ProcessSandbox(root=root_path, denied_paths=tuple(denied_paths))
        self._root = self._sandbox.root
        self._gate = gate
        self._read_cap = read_cap
        self._str_replace = CodeStrReplaceTool(self._sandbox)
        self._ctx = ToolExecutionContext(cwd=self._root)

    @property
    def root(self) -> Path:
        return self._root

    # ------------------------------------------------------------------ #
    # Confinement + SecurityGate composition
    # ------------------------------------------------------------------ #

    def _resolve(self, rel: str, *, tool_name: str) -> Path:
        """Resolve *rel* under the documents root and clear SecurityGate.

        Raises :class:`DocumentsError` (403) on traversal / absolute-outside /
        symlink-escape (the sandbox) OR on a SecurityGate denial (a path
        under-root but under a denied prefix). No I/O happens before this
        passes, so a rejected path never touches the filesystem.
        """
        try:
            real = self._sandbox.resolve(rel)
        except SandboxViolation as exc:
            raise DocumentsError(403, f"path escapes the documents root: {exc}") from exc
        self._gate_check(real, tool_name)
        return real

    def _gate_check(self, real: Path, tool_name: str) -> None:
        if self._gate is None:
            return
        decision = self._gate.evaluate(
            tool_name=tool_name,
            is_read_only=(tool_name == "read_file"),
            file_path=str(real),
            origin="user",
        )
        if not getattr(decision, "allowed", True):
            reason = getattr(decision, "reason", "") or "policy"
            raise DocumentsError(403, f"denied by SecurityGate: {reason}")

    # ------------------------------------------------------------------ #
    # List
    # ------------------------------------------------------------------ #

    def list_dir(self, rel: str = "") -> tuple[str, list[DocumentEntry]]:
        """List the immediate children of one directory under the root.

        A tree browser calls this per expanded directory (bounded: one level,
        capped entry count) — the same lazy model as ``/api/files``.
        """
        target = self._resolve(rel or ".", tool_name="read_file")
        if not target.exists():
            raise DocumentsError(404, "directory not found")
        if not target.is_dir():
            raise DocumentsError(400, "not a directory")
        try:
            children = sorted(
                target.iterdir(), key=lambda p: (p.is_file(), p.name.lower())
            )
        except OSError as exc:
            raise DocumentsError(500, f"listdir failed: {exc}") from exc
        entries: list[DocumentEntry] = []
        for child in children[:_LIST_MAX_ENTRIES]:
            try:
                st = child.stat()
                is_dir = child.is_dir()
            except OSError:
                continue  # skip dangling symlinks / unreadable entries
            entries.append(
                DocumentEntry(
                    name=child.name,
                    type="dir" if is_dir else "file",
                    size=0 if is_dir else st.st_size,
                    mtime=st.st_mtime,
                )
            )
        rel_out = "" if target == self._root else str(target.relative_to(self._root))
        return rel_out, entries

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #

    def read(self, rel: str) -> dict:
        """Read one file's content (raw, not line-numbered)."""
        target = self._resolve(rel, tool_name="read_file")
        if not target.exists() or not target.is_file():
            raise DocumentsError(404, "file not found")
        size = target.stat().st_size
        with target.open("rb") as fh:
            raw = fh.read(self._read_cap)
        binary = b"\x00" in raw
        content = ""
        if not binary:
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError:
                binary = True
        return {
            "path": str(target.relative_to(self._root)),
            "size": size,
            "truncated": size > self._read_cap,
            "binary": binary,
            "content": content,
        }

    # ------------------------------------------------------------------ #
    # Save (upsert — the manual-save path; creates a new file or overwrites)
    # ------------------------------------------------------------------ #

    def save(self, rel: str, content: str) -> dict:
        """Write *content* to *rel*, creating it (and parents) if new.

        This is the panel's manual save — distinct from the edit primitive
        (whole-file overwrite, not a unique-match find/replace). It is the one
        write that has no coding-mode tool analogue (``code_create`` refuses an
        existing file; ``code_str_replace`` needs a unique ``old_str``).
        """
        target = self._resolve(rel, tool_name="write_file")
        if target.exists() and target.is_dir():
            raise DocumentsError(400, "path is a directory")
        created = not target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode("utf-8")
        target.write_bytes(data)
        return {
            "path": str(target.relative_to(self._root)),
            "bytes_written": len(data),
            "created": created,
        }

    # ------------------------------------------------------------------ #
    # Edit (unique-match str_replace via the existing primitive) → diff
    # ------------------------------------------------------------------ #

    async def apply_edit(self, rel: str, old_str: str, new_str: str) -> DocumentEditResult:
        """Apply one unique-match find/replace and return the unified diff.

        Routes through the EXISTING ``CodeStrReplaceTool`` — same exact
        unique-match semantics, same loud no-match / multi-match / empty errors,
        which are returned (``ok=False``) UNCHANGED with the file untouched.
        """
        target = self._resolve(rel, tool_name="edit_file")
        if not target.exists() or not target.is_file():
            raise DocumentsError(404, "file not found")
        before = target.read_text(encoding="utf-8", errors="replace")
        result = await self._str_replace.execute(
            CodeStrReplaceArgs(path=rel, old_str=old_str, new_str=new_str),
            self._ctx,
        )
        if result.is_error:
            return DocumentEditResult(ok=False, diff="", error=result.output)
        after = target.read_text(encoding="utf-8", errors="replace")
        diff = "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
            )
        )
        return DocumentEditResult(ok=True, diff=diff, error="")

    # ------------------------------------------------------------------ #
    # Suggest (redline mode — propose edits, NEVER apply)
    # ------------------------------------------------------------------ #

    async def suggest(
        self, rel: str, instruction: str, generator: EditGenerator
    ) -> list[SuggestedEdit]:
        """Return proposed ``{find, replace, reason}`` redlines without applying.

        The *generator* (model-backed at the route, a stub in tests) sees the
        bounded document once and returns raw edit dicts; this method validates
        each ``find`` against the SAME exact-match rule the apply path enforces
        (occurs exactly once) and annotates applicability. Nothing is written.
        """
        target = self._resolve(rel, tool_name="read_file")
        if not target.exists() or not target.is_file():
            raise DocumentsError(404, "file not found")
        size = target.stat().st_size
        if size > _SUGGEST_DOC_CAP:
            raise DocumentsError(
                413,
                f"document too large for single-shot suggest ({size} bytes > "
                f"{_SUGGEST_DOC_CAP}); span-scoped suggest is a follow-up",
            )
        content = target.read_text(encoding="utf-8", errors="replace")
        raw_edits = await generator(content, instruction)
        out: list[SuggestedEdit] = []
        for raw in raw_edits or []:
            find = str(raw.get("find", ""))
            replace = str(raw.get("replace", ""))
            reason = str(raw.get("reason", ""))
            if not find:
                continue
            count = content.count(find)  # the same exact-match rule as str_replace
            applicable = count == 1
            if applicable:
                note = ""
            elif count == 0:
                note = "find not present in document"
            else:
                note = f"find is ambiguous ({count} matches) — widen it"
            out.append(
                SuggestedEdit(
                    find=find,
                    replace=replace,
                    reason=reason,
                    applicable=applicable,
                    note=note,
                )
            )
        return out
