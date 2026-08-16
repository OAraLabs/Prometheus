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
    a synthetic injected turn so the model sees it on its NEXT turn. Same
    channel as PeriodicNudge, but tagged ``provenance="file_mutation_
    verifier"`` rather than masquerading as something the user typed.

TURN SCOPING: ``run_daemon`` builds exactly ONE verifier and hands it to
every surface — telegram, CLI, cron, and (since this change) web/Beacon.
State is therefore keyed by ``turn_key``: one key per ``run_loop``
invocation, minted by the loop itself. Without it a single flat
accumulator is shared by every concurrent turn, so the first turn to
finish drains the other's mutations and reports them as its own while
the second reports nothing — which inverts a feature whose entire job is
checking that the writes YOU claimed actually landed.

No Hermes precedent: their hooks docs explicitly state file-mutation
verification "isn't provided as a ready-made feature" (see
``website/docs/user-guide/features/hooks.md`` on the upstream). Built
native — note in commit message.

Config:
  hooks:
    file_mutation_verifier:
      enabled: true              # opt-out, on by default
      truncate_after_n_mutations: 20
"""

from __future__ import annotations

import logging
import os
import re
import stat as _stat_mod
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

log = logging.getLogger(__name__)

# Callers that don't mint a turn key (unit tests, duck-typed embedders, any
# future single-threaded driver) share this one. Safe there BECAUSE they are
# single-threaded; every concurrent surface goes through run_loop, which always
# passes a real key.
DEFAULT_TURN_KEY = "__unscoped__"

# Backstop against records that never get drained: a turn that ends by raising,
# or one whose run_loop generator is abandoned without being closed, leaves its
# record behind. run_loop discards in a ``finally`` so this should stay cold —
# it exists so a caller that forgets cannot leak without bound. Evicting the
# least-recently-touched turn degrades to "no summary for that turn", never to
# cross-turn contamination.
MAX_LIVE_TURNS = 64


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


def _changed(before: _Snapshot, after: _Snapshot) -> bool:
    """True when the filesystem actually moved under this path.

    Creation, deletion, or any change to size/mtime. Content is not captured
    (see ``_Snapshot``), so an in-place rewrite of identical length within the
    same mtime granularity is invisible — a known false negative, and the
    reason this layer is described as detection rather than containment.
    """
    if before.exists != after.exists:
        return True
    if not after.exists:
        return False
    return (before.size, before.mtime) != (after.size, after.mtime)


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
    """Per-turn accumulator. Created on first touch, dropped on PostTurn."""
    mutations: list[_Mutation] = field(default_factory=list)
    # Map turn-scoped pre-snapshots by (tool_use_id, path) so post_tool_use
    # can pair them up even when one tool call touches multiple paths.
    _pending: dict[tuple[str, str], _Snapshot] = field(default_factory=dict)


def _expand_user(path: str) -> str:
    """Expand a leading ``~`` the way the shell already did.

    The shell expands ``~/.ssh/x`` before the write lands; ``os.stat`` does
    not. Snapshotting the unexpanded literal stats a path that can never
    exist, so before/after is absent->absent, ``_classify`` returns
    "missing", and a mutation that really happened is reported as nothing at
    all. Three writes under the denied-path floor landed this way with zero
    lines emitted.

    Expanding here also matters downstream: ``landed_paths()`` is handed to
    the permission gate by ``agent_loop._boundary_escapes``, and the floor
    globs (``/*/.ssh``) cannot match a ``~``-prefixed literal.

    Only a leading ``~`` is touched. Relative paths are deliberately left
    alone: a bash clause can ``cd`` first, so resolving them against the
    daemon's cwd would invent a path the command never used.
    """
    if not path.startswith("~"):
        return path
    return os.path.expanduser(path)


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
            out.append(_expand_user(val))
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
                target = _expand_user(target.strip("'\""))
                if target and not _is_device_sink(target):
                    out.append((target, action))
    return out


def _is_device_sink(target: str) -> bool:
    """True for redirect targets that are kernel device sinks, not files.

    ``> /dev/null`` (or /dev/zero, /dev/stdout, /dev/urandom, …) never
    changes on disk, so snapshotting it yields a guaranteed "CLAIMED but
    NO CHANGE ON DISK" — a false positive that poisons the summary. The
    exception is ``/dev/shm/``: a real tmpfs where files genuinely land.
    """
    return target.startswith("/dev/") and not target.startswith("/dev/shm/")


class FileMutationVerifier:
    """Per-turn tracker for claimed vs actual filesystem mutations.

    ONE instance is shared process-wide (see ``run_daemon``), so every entry
    point takes a ``turn_key`` identifying which in-flight turn it belongs to.
    Keys are minted per ``run_loop`` invocation; omitting one falls back to
    :data:`DEFAULT_TURN_KEY`, which is correct only for single-threaded callers.

    Lifecycle:
      pre_tool_use(tool_name, tool_input, tool_use_id, turn_key=...)
        snapshots the affected path(s) before execution.
      post_tool_use(tool_name, tool_input, tool_use_id, output, is_error,
                    turn_key=...)
        snapshots again and records the diff.
      post_turn(turn_key=...) -> str | None
        returns a summary string when that turn has mutations pending,
        ``None`` otherwise, and drops the turn's record. The caller
        (agent_loop) decides where the summary goes (default: append as an
        injected turn so the model sees it on its next turn).
      discard_turn(turn_key=...)
        drops a turn's record without rendering — the cleanup path for turns
        that end early (iteration cap, circuit breaker, interrupt).
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        truncate_after_n_mutations: int = 20,
    ) -> None:
        self._enabled = bool(enabled)
        self._truncate_n = max(1, int(truncate_after_n_mutations))
        # turn_key -> record, least-recently-touched first. Guarded by
        # ``_lock``: turns are driven by asyncio and interleave at every
        # ``await``, and gateways are free to drive the loop from a worker
        # thread, so map mutation must not race.
        self._turns: OrderedDict[str, _TurnRecord] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @staticmethod
    def new_turn_key(session_id: str | None = None) -> str:
        """Mint a key for one turn. Unique per call — a session id alone is
        NOT enough, since a session can have more than one turn in flight."""
        return f"{session_id or 'anon'}:{uuid4().hex}"

    @property
    def live_turns(self) -> int:
        """Number of turns currently holding state. Diagnostics only; a
        number that keeps climbing means a caller isn't draining."""
        with self._lock:
            return len(self._turns)

    # ------------------------------------------------------------------
    # Hook entry points — called by agent_loop
    # ------------------------------------------------------------------

    def pre_tool_use(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_use_id: str,
        *,
        turn_key: str | None = None,
    ) -> None:
        """Snapshot every path the tool is expected to touch."""
        if not self._enabled:
            return
        try:
            paths = self._paths_for(tool_name, tool_input)
            if not paths:
                # Don't materialise a record for a tool that touches nothing —
                # otherwise every bash `ls` allocates a turn slot.
                return
            snaps = {(tool_use_id, p): _snapshot(p) for p in paths}
            with self._lock:
                self._record(turn_key)._pending.update(snaps)
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
        turn_key: str | None = None,
    ) -> None:
        """Diff snapshots and record one ``_Mutation`` per tracked path."""
        if not self._enabled:
            return
        try:
            paths = self._paths_for(tool_name, tool_input)
            if not paths:
                return
            claim = self._claim_from(tool_name, tool_input)
            err = (output or "")[:200] if is_error else None
            with self._lock:
                turn = self._record(turn_key)
                for p in paths:
                    before = turn._pending.pop((tool_use_id, p), None)
                    if before is None:
                        # No pre-snapshot — happens if pre_tool_use raised or
                        # the post_tool_use receives a path the pre couldn't
                        # extract. Treat ``before`` as absent.
                        before = _Snapshot(exists=False)
                    turn.mutations.append(_Mutation(
                        tool=tool_name,
                        path=p,
                        claimed_action=claim,
                        before=before,
                        after=_snapshot(p),
                        error=err,
                    ))
        except Exception:
            log.debug(
                "FileMutationVerifier.post_tool_use raised", exc_info=True,
            )

    def landed_paths(self, *, turn_key: str | None = None) -> list[str]:
        """Paths this turn ACTUALLY changed on disk. Non-draining.

        The teeth of the outcome layer read this, so it is deliberately
        narrower than what :meth:`post_turn` renders: a mutation is included
        only when the before/after ``os.stat`` diff shows a real change. A
        tool that claimed a write and produced none is a REPORTING matter, not
        a boundary violation — nothing escaped.

        Ground truth, unlike anything available before dispatch. For ``bash``
        the pre-execution path guess is the regex heuristic in
        ``_BASH_FS_PATTERNS``, which is deliberately incomplete ("False
        negatives < false positives") because it is a reporter. This is the
        after-the-fact diff, so a redirect that the heuristic DID catch is
        confirmed by bytes rather than by pattern.

        ⚠ HONEST LIMIT, and it decides what the caller may do with this:
        ``_Snapshot`` holds ``exists``/``size``/``mtime``/``mode`` and NO
        CONTENT. A caller can learn that a file was overwritten. It can never
        put the old bytes back. Detection, never containment.
        """
        with self._lock:
            turn = self._turns.get(self._key(turn_key))
            if turn is None:
                return []
            return [
                m.path for m in turn.mutations
                if _changed(m.before, m.after)
            ]

    def post_turn(self, *, turn_key: str | None = None) -> str | None:
        """Render THIS turn's summary and drop its record. Returns ``None``
        when the turn tracked nothing.

        Dropping the record also discards unmatched pre-snapshots — those came
        from a tool that failed before execution, or one the pre handler
        didn't recognise, and must not leak into a later turn.
        """
        with self._lock:
            turn = self._turns.pop(self._key(turn_key), None)
        if turn is None or not turn.mutations:
            return None
        return self._format_summary(turn.mutations)

    def discard_turn(self, *, turn_key: str | None = None) -> None:
        """Drop a turn's record without rendering. Idempotent — safe to call
        after ``post_turn`` has already drained it."""
        with self._lock:
            self._turns.pop(self._key(turn_key), None)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(turn_key: str | None) -> str:
        return turn_key or DEFAULT_TURN_KEY

    def _record(self, turn_key: str | None) -> _TurnRecord:
        """Get-or-create this turn's record. Caller holds ``_lock``."""
        key = self._key(turn_key)
        turn = self._turns.get(key)
        if turn is None:
            turn = _TurnRecord()
            self._turns[key] = turn
            while len(self._turns) > MAX_LIVE_TURNS:
                evicted, _ = self._turns.popitem(last=False)
                log.warning(
                    "FileMutationVerifier: evicted undrained turn %r "
                    "(>%d live) — that turn gets no summary; a caller is "
                    "not calling post_turn/discard_turn",
                    evicted, MAX_LIVE_TURNS,
                )
        else:
            self._turns.move_to_end(key)
        return turn

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
            truncate_after_n_mutations: 20

    ``show_in_telegram`` was specified in SPRINT-2 WS2 and implemented as far
    as an attribute, but no code ever read it — the summary has always been
    model-facing only, on every surface. It is gone rather than left as a
    setting that silently does nothing; an unrecognised key here is ignored,
    so a config that still carries it keeps loading.
    """
    cfg = (
        ((config or {}).get("hooks") or {}).get("file_mutation_verifier")
        or {}
    )
    return FileMutationVerifier(
        enabled=cfg.get("enabled", True),
        truncate_after_n_mutations=cfg.get("truncate_after_n_mutations", 20),
    )
