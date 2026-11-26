"""Post-merge acceptance check (blackboard protocol §7, experimental).

The authoritative "is this item done" gate: re-run the item's acceptance
command on the INTEGRATED tree (the integration branch's tip after the
merge), and store the result SEPARATELY from the branch-local one. Green
plus green does not equal green — they are different trees, different
runs, different files.

This module is the check itself, callable by the merge flow (PR #966's
tool leaves acceptance pending; a follow-up wires the two) and by tests.
It is off by default in the sense that nothing calls it until the
blackboard pieces are enabled; the function itself is pure — repo, item,
command — and never touches the network or the model.

The result record is the protocol's (§7): kind=merged, the tree it ran
on, expected_exit_code, exit_code, daemon-computed passed, output
reference, timestamps. It is written to the board's orphan ref under
results/merged/<item-id>/<merge-id>.json with git plumbing (hash-object,
mktree, commit-tree, update-ref) — never a checkout, never a working-tree
write, so coordination writes stay off CI.

Source: novel code for Prometheus, 2026-11-26.
"""

from __future__ import annotations

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
#: How much of the command's output the record's output_ref file keeps.
_OUTPUT_CAP = 64_000


class AcceptanceError(Exception):
    """The check could not run (not a repo, unknown ref, board write failed)."""


def _git(repo: Path, *args: str, timeout: float = _GIT_TIMEOUT) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, timeout=timeout,
    )


def run_post_merge_acceptance(
    repo: str | Path,
    *,
    item_id: str,
    acceptance_command: str,
    expected_exit_code: int = 0,
    integration_ref: str,
    merge_id: str | None = None,
    board_ref: str = "refs/blackboard",
    mission: str = "",
    timeout_seconds: float = _ACCEPT_TIMEOUT,
) -> dict[str, Any]:
    """Run the acceptance command on the integration ref's tip; record it.

    The command runs in a TEMPORARY detached checkout of ``integration_ref``
    (``git worktree add --detach`` into a sibling temp dir, removed after),
    so the operator's working tree is never touched and the tree the command
    sees is exactly the merge result. The record is written to the board ref
    and returned. Raises AcceptanceError when the check cannot run; a
    non-zero exit is a RESULT (passed=False), never an exception.
    """
    repo = Path(repo).expanduser().resolve()
    if not (repo / ".git").exists():
        raise AcceptanceError(f"not a git repository: {repo}")
    tip = _git(repo, "rev-parse", "--verify", "--quiet", integration_ref + "^{commit}")
    if tip.returncode != 0:
        raise AcceptanceError(f"unknown integration ref: {integration_ref}")
    tree_sha = tip.stdout.strip()
    merge_id = merge_id or uuid.uuid4().hex[:10]

    import tempfile

    started = time.time()
    with tempfile.TemporaryDirectory(prefix="bb-accept-") as tmp:
        checkout = Path(tmp) / "tree"
        add = _git(repo, "worktree", "add", "--detach", str(checkout), tree_sha)
        if add.returncode != 0:
            raise AcceptanceError(f"cannot check out {integration_ref}: {add.stderr.strip()}")
        try:
            try:
                proc = subprocess.run(
                    acceptance_command, shell=True, cwd=str(checkout),
                    capture_output=True, text=True, timeout=timeout_seconds,
                )
                exit_code: int | None = proc.returncode
                output = (proc.stdout or "") + (proc.stderr or "")
            except subprocess.TimeoutExpired as exc:
                exit_code = None
                output = f"acceptance timed out after {timeout_seconds}s\n"
                output += (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        finally:
            _git(repo, "worktree", "remove", "--force", str(checkout))
            _git(repo, "worktree", "prune")
    ended = time.time()

    passed = exit_code is not None and exit_code == expected_exit_code
    # The output lives beside the board's results in the repo's data area —
    # a blob on the board ref itself, so the record's output_ref resolves
    # with git show and never depends on a local path surviving.
    out_blob = subprocess.run(
        ["git", "-C", str(repo), "hash-object", "-w", "--stdin"],
        input=output[-_OUTPUT_CAP:], capture_output=True, text=True,
    )
    output_ref = out_blob.stdout.strip() if out_blob.returncode == 0 else ""

    record = {
        "item_id": item_id,
        "kind": "merged",
        "merge_id": merge_id,
        "tree": tree_sha,
        "command": acceptance_command,
        "expected_exit_code": expected_exit_code,
        "exit_code": exit_code,
        "passed": passed,
        "source": "daemon-run",
        "output_ref": output_ref,
        "started_at": started,
        "ended_at": ended,
    }
    _write_board_record(repo, board_ref=board_ref, mission=mission,
                        path=f"results/merged/{item_id}/{merge_id}.json",
                        content=json.dumps(record, indent=2))
    return record


def _write_board_record(repo: Path, *, board_ref: str, mission: str,
                        path: str, content: str) -> None:
    """Append one file to the board's orphan ref with git plumbing.

    Reads the ref's current tree (or starts empty), adds the entry, builds
    nested trees bottom-up, commit-tree with the old tip as parent,
    update-ref. Never checks out, never touches the working tree. A
    concurrent board write between read and update-ref is lost-update
    territory; the caller serializes board writes (the coordinator is a
    single stateless decision-maker, D10) — this function does not invent
    a lock the protocol has not decided.
    """
    ref = f"{board_ref}/{mission}" if mission else board_ref
    blob = subprocess.run(
        ["git", "-C", str(repo), "hash-object", "-w", "--stdin"],
        input=content, capture_output=True, text=True, check=True,
    ).stdout.strip()

    parent = _git(repo, "rev-parse", "--verify", "--quiet", ref)
    entries: dict[str, str] = {}
    if parent.returncode == 0:
        listing = _git(repo, "ls-tree", "-r", parent.stdout.strip())
        for line in listing.stdout.splitlines():
            meta, p = line.split("\t", 1)
            entries[p] = meta.split()[2]
    entries[path] = blob

    def mktree(files: dict[str, str]) -> str:
        dirs: dict[str, dict] = {}
        plain: dict[str, str] = {}
        for p, sha in files.items():
            if "/" in p:
                head, rest = p.split("/", 1)
                dirs.setdefault(head, {})[rest] = sha
            else:
                plain[p] = sha
        lines = [f"100644 blob {sha}\t{p}\n" for p, sha in sorted(plain.items())]
        for d, sub in sorted(dirs.items()):
            lines.append(f"040000 tree {mktree(sub)}\t{d}\n")
        return subprocess.run(
            ["git", "-C", str(repo), "mktree"], input="".join(lines),
            capture_output=True, text=True, check=True,
        ).stdout.strip()

    tree = mktree(entries)
    args = ["git", "-C", str(repo), "commit-tree", tree, "-m",
            f"board: post-merge acceptance {path}"]
    if parent.returncode == 0:
        args += ["-p", parent.stdout.strip()]
    commit = subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
    upd = _git(repo, "update-ref", ref, commit)
    if upd.returncode != 0:
        raise AcceptanceError(f"board ref update failed: {upd.stderr.strip()}")