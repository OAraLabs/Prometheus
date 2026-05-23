"""SPRINT-2 WS2 — File-mutation verifier.

Catches silent failures where a tool *claims* a write succeeded but the
side effect didn't land on disk: Gemma saying "wrote 47 lines to foo.py"
while the editor returns success but the file is unchanged; bash exiting
0 without the side effect; permission-denied surfacing as "success" in a
buggy tool wrapper. These are the Adapter Layer's blind spot — the
*response shape* was fine, but the bytes on disk disagree.

How it works:
  - Pre-tool-use: for any FS-touching tool call, ``os.stat`` the target
    path (or each match from the bash regex) and stash the result on the
    in-flight turn record.
  - Post-tool-use: ``os.stat`` again. Diff with the snapshot. Tag the
    mutation as ``created``, ``modified``, ``deleted``, ``failed``, or
    ``no_change`` (claimed write but disk unchanged — the load-bearing
    case).
  - Post-turn: if any mutations accumulated, emit a one-block summary as
    a synthetic ``user``-role message so the model sees it on its NEXT
    turn. Same pattern as PeriodicNudge.

No Hermes precedent: their hooks docs explicitly state file-mutation
verification "isn't provided as a ready-made feature" (see
``website/docs/user-guide/features/hooks.md`` on the upstream). Built
native — note in commit message.

Config:
  hooks:
    file_mutation_verifier:
      enabled: true              # opt-out, on by default
      show_in_telegram: false    # quiet by default — only fed to model
      truncate_after_n_mutations: 20
"""

from __future__ import annotations

import logging
import os
import re
import stat as _stat_mod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# Tool names that touch the filesystem. Path extraction is per-tool —
# see ``_extract_paths`` below.
_FS_TOOLS = frozenset({
    "file_write", "write_file",       # writes
    "file_edit", "edit_file",
    "notebook_edit",
})

# Bash command patterns whose side effect is a path mutation. The verifier
# is heuristic — if a pattern doesn't match (compound commands, complex
# pipelines, custom aliases), the mutation just doesn't get tracked. That
# is preferable to false positives.
_BASH_FS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r'(?<![A-Za-z0-9_])mv\s+(?:-\w+\s+)*(\S+)\s+(\S+)'),   'move'),
    (re.compile(r'(?<![A-Za-z0-9_])rm\s+(?:-\w+\s+)*(\S+)'),            'delete'),
    (re.compile(r'(?<![A-Za-z0-9_])cp\s+(?:-\w+\s+)*(\S+)\s+(\S+)'),    'copy'),
    (re.compile(r'(?<![A-Za-z0-9_])touch\s+(\S+)'),                     'touch'),
    (re.compile(r'(?<![A-Za-z0-9_])mkdir\s+(?:-\w+\s+)*(\S+)'),         'mkdir'),
    # Redirects (anchored loosely — picks the LAST '>' / '>>' in the line).
    (re.compile(r'(?<![<>])>\s*(\S+)\s*$'),                             'redirect_write'),
    (re.compile(r'>>\s*(\S+)\s*$'),                                     'redirect_append'),
]


@dataclass
class _Snapshot:
    """os.stat result captured before a tool runs (None = file absent)."""
    exists: bool
    size: int = 0
    mtime: float = 0.0
    mode: int = 0


@dataclass
class _Mutation:
    """One tracked filesystem touch this turn."""
    tool: str
    path: str
    claimed_action: str           # "write", "edit", "delete", "create", ...
    before: _Snapshot
    after: _Snapshot
    error: str | None = None      # populated when the tool itself reported failure


@dataclass
class _TurnRecord:
    """Per-turn accumulator. Reset on each PostTurn."""
    mutations: list[_Mutation] = field(default_factory=list)
    # Map turn-scoped pre-snapshots by (tool_use_id, path) so post_tool_use
    # can pair them up even when one tool call touches multiple paths.
    _pending: dict[tuple[str, str], _Snapshot] = field(default_factory=dict)


def _snapshot(path: str) -> _Snapshot:
    """Cheap os.stat wrapper. Returns an absent-marker on any error."""
    try:
        st = os.stat(path)
        return _Snapshot(
            exists=True,
            size=int(st.st_size),
            mtime=float(st.st_mtime),
            mode=int(st.st_mode),
        )
    except (OSError, ValueError):
        return _Snapshot(exists=False)


def _classify(before: _Snapshot, after: _Snapshot) -> str:
    """Compare before/after snapshots and assign a status tag."""
    if before.exists and not after.exists:
        return "deleted"
    if not before.exists and after.exists:
        return "created"
    if not before.exists and not after.exists:
        return "missing"      # claimed something but path never existed
    # both exist — compare
    if before.size != after.size or before.mtime != after.mtime:
        return "modified"
    return "no_change"        # the load-bearing silent-failure case


def _extract_paths(tool_name: str, tool_input: dict[str, Any]) -> list[str]:
    """Best-effort extraction of paths from tool input."""
    out: list[str] = []
    # file_write / file_edit / notebook_edit all use ``file_path`` (or
    # ``path``) — the Prometheus convention.
    for key in ("file_path", "path", "notebook_path"):
        val = tool_input.get(key)
        if isinstance(val, str) and val:
            out.append(val)
            break
    return out


def _extract_bash_paths(command: str) -> list[tuple[str, str]]:
    """Return ``(path, claimed_action)`` tuples extracted from a bash line.

    Compound commands (``a && b``, ``foo; bar``) are scanned per-clause
    so an ``mkdir foo && touch foo/x.md`` reports two tracked paths.
    """
    out: list[tuple[str, str]] = []
    for clause in re.split(r"\s*(?:;|&&|\|\|)\s*", command or ""):
        for pat, action in _BASH_FS_PATTERNS:
            for m in pat.finditer(clause):
                # mv/cp: groups (src, dst) — track dst (the new home).
                # For mv we also want the src "deleted" effect; tracking
                # the dst alone is conservative but captures the
                # creation. False negatives < false positives.
                target = m.group(m.lastindex or 1)
                # Strip quotes that survive shell-style argv splitting.
                target = target.strip("'\"")
                if target:
                    out.append((target, action))
    return out


class FileMutationVerifier:
    """Per-turn tracker for claimed vs actual filesystem mutations.

    Lifecycle:
      pre_tool_use(tool_name, tool_input, tool_use_id)
        snapshots the affected path(s) before execution.
      post_tool_use(tool_name, tool_input, tool_use_id, output, is_error)
        snapshots again and records the diff.
      post_turn() -> str | None
        returns a summary string when any mutations are pending, ``None``
        otherwise. Resets internal state. The caller (agent_loop) decides
        where the summary goes (default: append as a synthetic user-role
        message so the model sees it on its next turn).
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        show_in_telegram: bool = False,
        truncate_after_n_mutations: int = 20,
    ) -> None:
        self._enabled = bool(enabled)
        self.show_in_telegram = bool(show_in_telegram)
        self._truncate_n = max(1, int(truncate_after_n_mutations))
        self._turn = _TurnRecord()

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------
    # Hook entry points — called by agent_loop
    # ------------------------------------------------------------------

    def pre_tool_use(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_use_id: str,
    ) -> None:
        """Snapshot every path the tool is expected to touch."""
        if not self._enabled:
            return
        try:
            for p in self._paths_for(tool_name, tool_input):
                self._turn._pending[(tool_use_id, p)] = _snapshot(p)
        except Exception:
            log.debug("FileMutationVerifier.pre_tool_use raised", exc_info=True)

    def post_tool_use(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_use_id: str,
        *,
        output: str | None = None,
        is_error: bool = False,
    ) -> None:
        """Diff snapshots and record one ``_Mutation`` per tracked path."""
        if not self._enabled:
            return
        try:
            paths = self._paths_for(tool_name, tool_input)
            for p in paths:
                before = self._turn._pending.pop((tool_use_id, p), None)
                if before is None:
                    # No pre-snapshot — happens if pre_tool_use raised or
                    # the post_tool_use receives a path the pre couldn't
                    # extract. Treat ``before`` as absent.
                    before = _Snapshot(exists=False)
                after = _snapshot(p)
                self._turn.mutations.append(_Mutation(
                    tool=tool_name,
                    path=p,
                    claimed_action=self._claim_from(tool_name, tool_input),
                    before=before,
                    after=after,
                    error=(output or "")[:200] if is_error else None,
                ))
        except Exception:
            log.debug(
                "FileMutationVerifier.post_tool_use raised", exc_info=True,
            )

    def post_turn(self) -> str | None:
        """Render and reset the per-turn summary. Returns ``None`` when
        nothing was tracked."""
        muts = list(self._turn.mutations)
        # Drop unmatched pre-snapshots that never got a post — they
        # likely came from a tool that failed before execution, or one
        # the pre handler didn't recognise; reset to avoid leaking
        # across turns.
        self._turn = _TurnRecord()
        if not muts:
            return None
        return self._format_summary(muts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _paths_for(
        self, tool_name: str, tool_input: dict[str, Any],
    ) -> list[str]:
        """Return every path this tool call may touch."""
        if tool_name in _FS_TOOLS:
            return _extract_paths(tool_name, tool_input)
        if tool_name == "bash":
            command = str(tool_input.get("command", ""))
            return [p for p, _ in _extract_bash_paths(command)]
        return []

    def _claim_from(
        self, tool_name: str, tool_input: dict[str, Any],
    ) -> str:
        """Human-readable claim of what the tool said it would do."""
        if tool_name == "file_write":
            return "write"
        if tool_name == "file_edit":
            return "edit"
        if tool_name == "notebook_edit":
            return "edit_notebook"
        if tool_name == "bash":
            command = str(tool_input.get("command", ""))
            actions = [a for _, a in _extract_bash_paths(command)]
            if not actions:
                return "bash"
            return "/".join(sorted(set(actions)))
        return tool_name

    def _format_summary(self, muts: list[_Mutation]) -> str:
        """Render the per-turn list into a single string. Truncates."""
        lines = ["[FILE MUTATION VERIFIER]", "Files touched this turn:"]
        shown = muts[: self._truncate_n]
        for m in shown:
            tag, badge = self._tag(m)
            size_note = ""
            if m.after.exists and tag in {"modified", "created"}:
                delta = m.after.size - m.before.size
                size_note = (
                    f" (+{delta} bytes)" if delta > 0
                    else f" ({delta} bytes)" if delta < 0
                    else f" ({m.after.size} bytes)"
                )
            error_note = f" — {m.error}" if m.error else ""
            lines.append(
                f"   {badge} {m.path} — {m.claimed_action}: "
                f"{tag}{size_note}{error_note}"
            )
        if len(muts) > self._truncate_n:
            lines.append(
                f"   ... and {len(muts) - self._truncate_n} more "
                f"(truncated at {self._truncate_n})"
            )
        return "\n".join(lines)

    @staticmethod
    def _tag(m: _Mutation) -> tuple[str, str]:
        """Map a _Mutation into a (status, unicode-badge) pair."""
        status = _classify(m.before, m.after)
        if m.error:
            return status, "✗"
        if status == "no_change":
            # The load-bearing silent-failure case: claimed write, no disk change.
            return "CLAIMED but NO CHANGE ON DISK", "⚠"
        if status == "missing":
            return "CLAIMED but FILE ABSENT", "⚠"
        return status, "✓"


def make_default_verifier(config: dict[str, Any] | None = None) -> "FileMutationVerifier":
    """Build a verifier from a (possibly partial) config block.

    Honours the spec:
        hooks:
          file_mutation_verifier:
            enabled: true
            show_in_telegram: false
            truncate_after_n_mutations: 20
    """
    cfg = (
        ((config or {}).get("hooks") or {}).get("file_mutation_verifier")
        or {}
    )
    return FileMutationVerifier(
        enabled=cfg.get("enabled", True),
        show_in_telegram=cfg.get("show_in_telegram", False),
        truncate_after_n_mutations=cfg.get("truncate_after_n_mutations", 20),
    )
