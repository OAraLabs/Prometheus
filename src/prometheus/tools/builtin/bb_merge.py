"""The blackboard merge tool (experimental, off by default).

Blackboard protocol §7 (docs/contracts/blackboard.md): the coordinator
DECIDES a merge; this open tool ENFORCES it. The rules, in the tool,
not in instructions:

1. **The intent block is required as an argument.** No intent, no merge —
   the call is refused before any git command runs.
2. **Dependencies first.** Every item id in ``depends_on`` must carry a
   green post-merge acceptance result on the board, else the call is
   refused with the missing ids named.
3. **A conflict is a RESULT, never a resolution.** The merge runs with
   ``--no-commit --no-ff`` in a TEMPORARY worktree; a conflict aborts and
   returns the conflicted paths and the two refs. No strategy flag
   (``-X ours`` etc.) exists in this tool's vocabulary.
4. **Only the mission's integration branch.** ``into`` must be exactly
   ``bb/<mission>/integrate`` and the mission id must be non-empty —
   this tool never merges into ``main`` or any other ref.
5. **Accept-before-advance.** The merge commit is built in the temp
   worktree; the post-merge acceptance check (coding.acceptance, #967)
   runs on THAT commit; the integration ref moves with a compare-and-swap
   ``update-ref <ref> <merge> <old>`` ONLY when the check is green. On
   red the ref does not move and the result carries the red record —
   the integration branch is always green.

The operator's checkout is NEVER touched: no checkout, no merge, no
commit happens in it (review #2). ``repo`` is a declared WRITTEN
DIRECTORY path, so the gate's workspace boundary and denied-path floor
apply to it like any file-writing tool (review #2: an undeclared path
meant the gate saw no file_path and fell through to ALLOW anywhere on
disk).

The tool is registered only when ``blackboard.merge_tool.enabled`` is
true (literal true only; ships false), under the same gate as every
tool call.

Source: novel code for Prometheus, 2026-11-26.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from prometheus.permissions.path_schema import DIR_FIELD_WRITE
from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolResult

log = logging.getLogger(__name__)

_GIT_TIMEOUT = 120


def merge_tool_enabled(config: Any) -> bool:
    """Literal true only — same shape as every experimental gate."""
    section = (config or {}).get("blackboard") if isinstance(config, dict) else None
    if not isinstance(section, dict):
        return False
    tool = section.get("merge_tool")
    if not isinstance(tool, dict):
        return False
    return tool.get("enabled", False) is True


class MergeIntent(BaseModel):
    """The worker's brief statement of what changed and why. REQUIRED."""

    what: str = Field(min_length=1, max_length=2000)
    why: str = Field(min_length=1, max_length=2000)


class BbMergeInput(BaseModel):
    repo: str = Field(
        json_schema_extra=DIR_FIELD_WRITE,
        description="Absolute path to the git repo the mission works on (written: the merge "
                    "commit and the board ref live in its object store)",
    )
    branch: str = Field(description="The worker branch to merge, bb/<mission>/<item>/<agent>")
    into: str = Field(description="The mission's integration branch — must be exactly bb/<mission>/integrate")
    item_id: str = Field(description="The roadmap item id this merge completes")
    intent: MergeIntent = Field(description="What changed and why — required, refused without it")
    depends_on: list[str] = Field(
        default_factory=list,
        description="Item ids that must already have a green post-merge result on the board",
    )
    board_ref: str = Field(
        default="refs/blackboard",
        description="The board ref prefix; results read from <ref>/<mission>/results/merged/",
    )
    mission: str = Field(default="", description="Mission id; defaults to the branch's bb/<mission>/ segment")


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, timeout=_GIT_TIMEOUT,
    )


def _green_dependencies(repo: Path, board_ref: str, mission: str,
                        depends_on: list[str]) -> list[str]:
    """Item ids whose post-merge result is NOT green on the board.

    Reads ``results/merged/<item-id>/`` entries from the board ref with
    ``git show`` — never checks the board out. A result is green when its
    JSON carries the daemon-computed ``passed: true`` and ``kind: merged``
    (or, for a record predating ``passed``, ``exit_code == expected_exit_code``
    with expected defaulting to 0). Missing, unreadable or non-green all
    count as not-green: the refusal names them.
    """
    ref = f"{board_ref}/{mission}"
    missing: list[str] = []
    for item in depends_on:
        path = f"results/merged/{item}"
        listing = _git(repo, "ls-tree", "-r", "-z", "--name-only", ref, "--", path + "/")
        names = [n for n in listing.stdout.split("\0") if n.strip()]
        if listing.returncode != 0 or not names:
            missing.append(item)
            continue
        green = False
        for name in names:
            show = _git(repo, "show", f"{ref}:{name}")
            if show.returncode != 0:
                continue
            try:
                rec = json.loads(show.stdout)
            except json.JSONDecodeError:
                continue
            if rec.get("kind") != "merged":
                continue
            if rec.get("passed") is True or (
                "passed" not in rec
                and rec.get("exit_code") == rec.get("expected_exit_code", 0)
                and rec.get("exit_code") is not None
            ):
                green = True
                break
        if not green:
            missing.append(item)
    return missing


class BbMergeTool(BaseTool):
    name = "bb_merge"
    description = (
        "Merge a blackboard worker branch into the mission's integration branch "
        "(bb/<mission>/integrate only). Requires the intent block (what/why) and green "
        "post-merge results for every dependency. The merge is built in a temporary "
        "worktree; the post-merge acceptance check runs on the merge commit and the "
        "integration branch advances ONLY on green (accept-before-advance). A conflict "
        "or a red check is returned as a result — never resolved with a strategy flag. "
        "Experimental; the coordinator decides, this tool enforces."
    )
    input_model = BbMergeInput
    find_me = ("merge a worker branch", "complete a roadmap item")

    async def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolResult:
        args: BbMergeInput = arguments  # type: ignore[assignment]
        repo = Path(args.repo).expanduser().resolve()
        if not (repo / ".git").exists():
            return ToolResult(output=f"not a git repository: {repo}", is_error=True)

        # Rule 1: the intent block. pydantic already refused an empty/missing
        # one at validation; this is the belt for a duck-typed caller.
        if not args.intent.what.strip() or not args.intent.why.strip():
            return ToolResult(output="the intent block (what/why) is required to merge", is_error=True)

        mission = args.mission or (args.branch.split("/")[1] if args.branch.startswith("bb/") and len(args.branch.split("/")) > 2 else "")
        # Rule 4: a mission id is required (the bare refs/blackboard would
        # block every per-mission ref), and the ONLY merge target is the
        # mission's integration branch — never main, never another ref.
        if not mission.strip() or "/" in mission or mission.startswith("."):
            return ToolResult(output=f"mission id required (got {mission!r})", is_error=True)
        if args.into != f"bb/{mission}/integrate":
            return ToolResult(
                output=f"into must be the mission's integration branch bb/{mission}/integrate "
                       f"(got {args.into!r}); this tool never merges into any other ref",
                is_error=True,
            )

        # Rule 2: dependencies.
        not_green = _green_dependencies(repo, args.board_ref, mission, args.depends_on)
        if not_green:
            return ToolResult(
                output="dependencies without a green post-merge result: " + ", ".join(not_green),
                is_error=True, metadata={"missing": not_green},
            )

        # The branches must exist. The operator's working tree is NEVER
        # touched — no cleanliness requirement, no checkout: the merge is
        # built in a temporary worktree of the integration ref's tip.
        for ref in (args.branch, args.into):
            check = _git(repo, "rev-parse", "--verify", "--quiet", ref + "^{commit}")
            if check.returncode != 0:
                return ToolResult(output=f"unknown branch or ref: {ref}", is_error=True)
        old_tip = _git(repo, "rev-parse", args.into).stdout.strip()

        import tempfile

        from prometheus.coding.acceptance import AcceptanceError, run_post_merge_acceptance

        merge_id = f"m{__import__('uuid').uuid4().hex[:10]}"
        with tempfile.TemporaryDirectory(prefix="bb-merge-") as tmp:
            wt = Path(tmp) / "merge"
            add = _git(repo, "worktree", "add", "--detach", str(wt), old_tip)
            if add.returncode != 0:
                return ToolResult(output=f"cannot check out {args.into}: {add.stderr.strip()}", is_error=True)
            try:
                # Rule 3: merge, conflict as a result. --no-ff keeps the
                # merge commit even when fast-forwardable, so the item's
                # merge is a named event in git log.
                merge = _git(wt, "merge", "--no-ff", "--no-commit", args.branch)
                if merge.returncode != 0:
                    conflicts = _git(wt, "diff", "--name-only", "--diff-filter=U").stdout.splitlines()
                    _git(wt, "merge", "--abort")
                    return ToolResult(
                        output=json.dumps({
                            "merged": False,
                            "conflict": [c for c in conflicts if c.strip()],
                            "branch": args.branch,
                            "into": args.into,
                            "detail": merge.stderr.strip() or merge.stdout.strip(),
                        }, indent=2),
                        is_error=True,
                        metadata={"conflict": conflicts},
                    )
                commit = _git(wt, "commit", "--quiet", "-m",
                              f"bb merge: {args.item_id} from {args.branch}\n\n"
                              f"what: {args.intent.what.strip()}\nwhy: {args.intent.why.strip()}")
                if commit.returncode != 0:
                    _git(wt, "merge", "--abort")
                    return ToolResult(output=f"merge commit failed: {commit.stderr.strip()}", is_error=True)
                merge_sha = _git(wt, "rev-parse", "HEAD").stdout.strip()
            finally:
                _git(repo, "worktree", "remove", "--force", str(wt))
                _git(repo, "worktree", "prune")

        # Rule 5: accept-before-advance. The check runs on the MERGE COMMIT
        # (still referenced by nothing but this call — pin it on a temp ref
        # so gc cannot take it mid-check), through the coding sandbox, with
        # the command read from the roadmap (#967). The integration branch
        # moves ONLY on green, by compare-and-swap against the tip we read.
        pin = f"refs/bb-merge-pending/{merge_id}"
        _git(repo, "update-ref", pin, merge_sha)
        try:
            try:
                import asyncio as _aio
                record = await _aio.to_thread(
                    run_post_merge_acceptance,
                    repo, item_id=args.item_id, mission=mission,
                    integration_ref=pin, merge_id=merge_id, board_ref=args.board_ref,
                )
            except AcceptanceError as exc:
                return ToolResult(
                    output=json.dumps({
                        "merged": False, "item_id": args.item_id,
                        "commit": merge_sha, "into": args.into,
                        "acceptance": "could not run", "detail": str(exc),
                        "note": "the integration branch did not move",
                    }, indent=2),
                    is_error=True,
                )
        finally:
            _git(repo, "update-ref", "-d", pin)

        if not record.get("passed"):
            return ToolResult(
                output=json.dumps({
                    "merged": False,
                    "item_id": args.item_id,
                    "commit": merge_sha,
                    "into": args.into,
                    "acceptance": "red — the integration branch did not move",
                    "merged_acceptance": record,
                }, indent=2),
                is_error=True,
                metadata={"item_id": args.item_id, "acceptance_passed": False},
            )

        # Green: advance the integration branch, compare-and-swap. A
        # concurrent mover of the ref means our merge is stale — refuse,
        # never clobber.
        upd = _git(repo, "update-ref", "refs/heads/" + args.into, merge_sha, old_tip)
        if upd.returncode != 0:
            return ToolResult(
                output=json.dumps({
                    "merged": False, "item_id": args.item_id, "commit": merge_sha,
                    "into": args.into,
                    "acceptance": "green, but the integration ref moved during the check",
                    "detail": upd.stderr.strip(),
                    "note": "compare-and-swap refused; re-run the merge against the new tip",
                }, indent=2),
                is_error=True,
            )
        return ToolResult(
            output=json.dumps({
                "merged": True,
                "item_id": args.item_id,
                "branch": args.branch,
                "into": args.into,
                "commit": merge_sha,
                "intent": {"what": args.intent.what.strip(), "why": args.intent.why.strip()},
                "acceptance": "green — the integration branch advanced to the merge commit",
                "merged_acceptance": record,
            }, indent=2),
            metadata={"commit": merge_sha, "item_id": args.item_id, "acceptance_passed": True},
        )