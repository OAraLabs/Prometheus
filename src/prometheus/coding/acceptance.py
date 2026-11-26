"""Post-merge acceptance check (blackboard protocol §7, experimental).

The authoritative "is this item done" gate: re-run the item's acceptance
command on the INTEGRATED tree (the integration branch's tip after the
merge), and store the result SEPARATELY from the branch-local one. Green
plus green does not equal green — they are different trees, different
runs, different files.

This module is the check itself, callable by the merge flow (PR #966's
tool wires it accept-before-advance) and by tests. It is off by default
in the sense that nothing calls it until the blackboard pieces are
enabled; the function itself never touches the network or the model.

The command is read VERBATIM from the roadmap on the board, by item id
(review #2: nothing but the human-owned roadmap decides what runs) — a
caller cannot supply one. It runs through the SAME coding sandbox a
coding run's acceptance uses (ProcessSandbox: floored_argv read/write
floors with the temp checkout as the only workspace, and the sandbox's
env allowlist — never the daemon's environment with its API keys).

The result record is the protocol's (§7): kind=merged, the tree it ran
on, expected_exit_code, exit_code, daemon-computed passed, output
reference, timestamps. It is written to the board's orphan ref under
results/merged/<item-id>/<merge-id>.json with git plumbing (hash-object,
mktree, commit-tree, update-ref) — never a checkout, never a working-tree
write, so coordination writes stay off CI. The output is stored IN the
board tree next to the record (a referenced blob — a dangling
hash-object blob would be pruned by git gc). Every board write is a
compare-and-swap (update-ref <ref> <new> <old>, retry on mismatch): the
run harness and the merge flow are both writers.

A mission id is REQUIRED: the bare ref refs/blackboard would block every
refs/blackboard/<mission> ref (git cannot hold both).

Source: novel code for Prometheus, 2026-11-26.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_GIT_TIMEOUT = 120
#: Acceptance commands get the same wall budget a coding run's acceptance
#: gets by default (coding/session.py acceptance_timeout_seconds).
_ACCEPT_TIMEOUT = 240.0
#: How much of the command's output the record's output file keeps.
_OUTPUT_CAP = 64_000
#: CAS retries when a concurrent writer moved the board ref between our
#: read and our update-ref.
_CAS_RETRIES = 5


class AcceptanceError(Exception):
    """The check could not run (not a repo, unknown ref, board write failed)."""


def _git(repo: Path, *args: str, timeout: float = _GIT_TIMEOUT) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, timeout=timeout,
    )


def board_ref_for(board_ref: str, mission: str) -> str:
    """The full board ref. A mission id is REQUIRED — the bare
    ``refs/blackboard`` would block every ``refs/blackboard/<mission>``
    ref (git cannot hold both a ref and a directory of the same name)."""
    mission = (mission or "").strip()
    if not mission or "/" in mission or mission.startswith("."):
        raise AcceptanceError(f"mission id required (got {mission!r})")
    return f"{board_ref}/{mission}"


def read_roadmap_item(repo: Path, *, mission: str, item_id: str,
                      board_ref: str = "refs/blackboard") -> dict[str, Any]:
    """One item from roadmap.json on the board — never a checkout.

    The roadmap generator (§3, its own PR) emits either a top-level list
    of items or {"items": [...]}; both are read here. A malformed roadmap
    or a missing item is a named error, never a partial parse.
    """
    ref = board_ref_for(board_ref, mission)
    show = _git(repo, "show", f"{ref}:roadmap.json")
    if show.returncode != 0:
        raise AcceptanceError(f"no roadmap.json on {ref}")
    try:
        doc = json.loads(show.stdout)
    except json.JSONDecodeError as exc:
        raise AcceptanceError(f"roadmap.json on {ref} is not valid JSON: {exc}") from exc
    items = doc.get("items") if isinstance(doc, dict) else doc
    if not isinstance(items, list):
        raise AcceptanceError(f"roadmap.json on {ref}: expected a list of items")
    for item in items:
        if isinstance(item, dict) and item.get("id") == item_id:
            return item
    raise AcceptanceError(f"item {item_id!r} not on the roadmap ({ref})")


def run_post_merge_acceptance(
    repo: str | Path,
    *,
    item_id: str,
    integration_ref: str,
    mission: str,
    merge_id: str | None = None,
    board_ref: str = "refs/blackboard",
    timeout_seconds: float = _ACCEPT_TIMEOUT,
    sandbox_backend: str = "process",
) -> dict[str, Any]:
    """Run the item's acceptance command on the integration ref's tip; record it.

    The command and expected_exit_code come from the roadmap on the board
    (``roadmap.json``, by item id) — the caller supplies neither. The
    command runs in a TEMPORARY detached checkout of ``integration_ref``
    (``git worktree add --detach`` into a sibling temp dir, removed after),
    through the coding sandbox (floors + env allowlist) with that checkout
    as its only workspace, so the operator's working tree is never touched
    and the tree the command sees is exactly the merge result. The record
    is written to the board ref (compare-and-swap) and returned. Raises
    AcceptanceError when the check cannot run (including a refused shell
    floor — nothing runs unfloored); a non-zero exit is a RESULT
    (passed=False), never an exception.
    """
    repo = Path(repo).expanduser().resolve()
    if not (repo / ".git").exists():
        raise AcceptanceError(f"not a git repository: {repo}")
    board_ref_for(board_ref, mission)  # mission id required, before anything runs
    item = read_roadmap_item(repo, mission=mission, item_id=item_id, board_ref=board_ref)
    acceptance = item.get("acceptance")
    if isinstance(acceptance, dict):
        command = acceptance.get("command")
        expected_exit_code = int(acceptance.get("expected_exit_code", 0))
    elif isinstance(acceptance, str):
        command = acceptance
        expected_exit_code = int(item.get("expected_exit_code", 0))
    else:
        command = None
        expected_exit_code = 0
    if not isinstance(command, str) or not command.strip():
        raise AcceptanceError(f"item {item_id!r} has no acceptance command on the roadmap")

    tip = _git(repo, "rev-parse", "--verify", "--quiet", integration_ref + "^{commit}")
    if tip.returncode != 0:
        raise AcceptanceError(f"unknown integration ref: {integration_ref}")
    tree_sha = tip.stdout.strip()
    merge_id = merge_id or uuid.uuid4().hex[:10]

    import tempfile

    from prometheus.coding.sandbox import create_sandbox
    from prometheus.security.shell_floor import ShellFloorRefused

    started = time.time()
    with tempfile.TemporaryDirectory(prefix="bb-accept-") as tmp:
        checkout = Path(tmp) / "tree"
        add = _git(repo, "worktree", "add", "--detach", str(checkout), tree_sha)
        if add.returncode != 0:
            raise AcceptanceError(f"cannot check out {integration_ref}: {add.stderr.strip()}")
        try:
            box = create_sandbox(checkout, task_id=f"bb-accept-{merge_id}",
                                 backend=sandbox_backend)
            try:
                result = asyncio.run(box.run(command, timeout_seconds=timeout_seconds))
            except ShellFloorRefused as exc:
                # Fail closed: a required floor this host cannot provide
                # means nothing ran — never fall back to an unfloored shell.
                raise AcceptanceError(f"shell floor refused, nothing ran: {exc}") from exc
            finally:
                box.close()
            exit_code = None if result.timed_out else result.exit_code
            output = result.output or ""
            if result.timed_out:
                output = f"acceptance timed out after {timeout_seconds}s\n" + output
        finally:
            _git(repo, "worktree", "remove", "--force", str(checkout))
            _git(repo, "worktree", "prune")
    ended = time.time()

    passed = exit_code is not None and exit_code == expected_exit_code
    record_path = f"results/merged/{item_id}/{merge_id}.json"
    # The output lives IN the board tree next to the record — a referenced
    # blob. A bare hash-object blob nothing references is pruned by git gc.
    output_path = f"results/merged/{item_id}/{merge_id}.output.txt"

    record = {
        "item_id": item_id,
        "kind": "merged",
        "merge_id": merge_id,
        "tree": tree_sha,
        "command": command,
        "expected_exit_code": expected_exit_code,
        "exit_code": exit_code,
        "passed": passed,
        "source": "daemon-run",
        "output_ref": output_path,
        "started_at": started,
        "ended_at": ended,
    }
    _write_board_files(repo, board_ref=board_ref, mission=mission,
                       files={record_path: json.dumps(record, indent=2),
                              output_path: output[-_OUTPUT_CAP:]})
    return record


def _write_board_files(repo: Path, *, board_ref: str, mission: str,
                       files: dict[str, str]) -> None:
    """Add files to the board's orphan ref with git plumbing + CAS.

    Reads the ref's current tree (or starts empty), adds the entries,
    builds nested trees bottom-up, commit-tree with the old tip as parent,
    then ``update-ref <ref> <new> <old>`` — compare-and-swap. On a
    mismatch (a concurrent writer moved the ref: the run harness writes
    branch results while the merge flow writes merged results) the tree
    is re-read and the write retried. Never checks out, never touches the
    working tree. ``ls-tree -r -z`` so quoted/awkward paths round-trip.
    """
    ref = board_ref_for(board_ref, mission)
    blobs: dict[str, str] = {}
    for path, content in files.items():
        blobs[path] = subprocess.run(
            ["git", "-C", str(repo), "hash-object", "-w", "--stdin"],
            input=content, capture_output=True, text=True, check=True,
        ).stdout.strip()

    def mktree(entries: dict[str, str]) -> str:
        dirs: dict[str, dict] = {}
        plain: dict[str, str] = {}
        for p, sha in entries.items():
            if "/" in p:
                head, rest = p.split("/", 1)
                dirs.setdefault(head, {})[rest] = sha
            else:
                plain[p] = sha
        lines = [f"100644 blob {sha}\t{p}\n" for p, sha in sorted(plain.items())]
        for d, sub in sorted(dirs.items()):
            lines.append(f"040000 tree {mktree(sub)}\t{d}\n")
        return subprocess.run(
            ["git", "-C", str(repo), "mktree", "-z"],
            input="".join(line[:-1] + "\0" for line in lines),
            capture_output=True, text=True, check=True,
        ).stdout.strip()

    for attempt in range(_CAS_RETRIES):
        parent = _git(repo, "rev-parse", "--verify", "--quiet", ref)
        old = parent.stdout.strip() if parent.returncode == 0 else ""
        entries: dict[str, str] = {}
        if old:
            listing = subprocess.run(
                ["git", "-C", str(repo), "ls-tree", "-r", "-z", old],
                capture_output=True, text=True, check=True,
            )
            for entry in listing.stdout.split("\0"):
                if not entry:
                    continue
                meta, p = entry.split("\t", 1)
                entries[p] = meta.split()[2]
        entries.update(blobs)
        tree = mktree(entries)
        args = ["git", "-C", str(repo), "commit-tree", tree, "-m",
                f"board: {', '.join(sorted(files))}"]
        if old:
            args += ["-p", old]
        commit = subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
        # compare-and-swap: the ref must still be where we read it
        upd = _git(repo, "update-ref", ref, commit, old or "0" * 40)
        if upd.returncode == 0:
            return
        log.info("board CAS mismatch on %s (attempt %d) — re-reading", ref, attempt + 1)
    raise AcceptanceError(f"board ref update failed after {_CAS_RETRIES} CAS attempts: {ref}")


def _write_board_record(repo: Path, *, board_ref: str, mission: str,
                        path: str, content: str) -> None:
    """One-file convenience over :func:`_write_board_files` (the run
    harness's branch-local records use the same CAS plumbing)."""
    _write_board_files(repo, board_ref=board_ref, mission=mission,
                       files={path: content})