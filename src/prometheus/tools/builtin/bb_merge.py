"""The blackboard merge tool (experimental, off by default).

Blackboard protocol §7 (docs/contracts/blackboard.md): the coordinator
DECIDES a merge; this open tool ENFORCES it. Three rules, in the tool,
not in instructions:

1. **The intent block is required as an argument.** No intent, no merge —
   the call is refused before any git command runs.
2. **Dependencies first.** Every item id in ``depends_on`` must carry a
   green post-merge acceptance result on the board, else the call is
   refused with the missing ids named.
3. **A conflict is a RESULT, never a resolution.** The merge runs with
   ``--no-commit --no-ff``; a conflict aborts the merge and returns the
   conflicted paths and the two refs. No strategy flag (``-X ours`` etc.)
   exists in this tool's vocabulary — the coordinator escalates the
   conflict through the amendment/interpretation classifier.

The tool is registered only when ``blackboard.merge_tool.enabled`` is
true (literal true only; ships false). It runs git in the repo the
operator named, under the same gate as every tool call — the gate's
APPROVE tier and the owner's queue are unchanged; this tool adds no
permission of its own. The post-merge acceptance CHECK is a separate PR
(blackboard.md §12); this tool records the merge and leaves the
acceptance result to that check, never inventing one.

Source: novel code for Prometheus, 2026-11-26.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

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
    repo: str = Field(description="Absolute path to the git repo the mission works on")
    branch: str = Field(description="The worker branch to merge, bb/<mission>/<item>/<agent>")
    into: str = Field(description="The integration branch to merge into (must exist)")
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
    JSON carries ``exit_code: 0`` and ``kind: merged``. Missing, unreadable
    or non-green all count as not-green: the refusal names them.
    """
    ref = f"{board_ref}/{mission}" if mission else board_ref
    missing: list[str] = []
    for item in depends_on:
        path = f"results/merged/{item}"
        # ls-tree <commit-ish> -- <path>: the board ref is a commit, the
        # results live under a path in its tree. Never checks the board out.
        listing = _git(repo, "ls-tree", "--name-only", ref, "--", path + "/")
        names = [n for n in listing.stdout.splitlines() if n.strip()]
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
            if rec.get("kind") == "merged" and rec.get("exit_code") == 0:
                green = True
                break
        if not green:
            missing.append(item)
    return missing


class BbMergeTool(BaseTool):
    name = "bb_merge"
    description = (
        "Merge a blackboard worker branch into the integration branch. Requires the intent "
        "block (what/why) and green post-merge results for every dependency. A conflict is "
        "returned as a result — never resolved with a strategy flag. Experimental; the "
        "coordinator decides, this tool enforces."
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

        # Rule 2: dependencies.
        not_green = _green_dependencies(repo, args.board_ref, mission, args.depends_on)
        if not_green:
            return ToolResult(
                output="dependencies without a green post-merge result: " + ", ".join(not_green),
                is_error=True, metadata={"missing": not_green},
            )

        # The branches must exist; the working tree must be clean enough to
        # merge into (a dirty tree is the operator's, never ours to stash).
        for ref in (args.branch, args.into):
            check = _git(repo, "rev-parse", "--verify", "--quiet", ref + "^{commit}")
            if check.returncode != 0:
                return ToolResult(output=f"unknown branch or ref: {ref}", is_error=True)
        dirty = _git(repo, "status", "--porcelain")
        if dirty.stdout.strip():
            return ToolResult(
                output="the working tree is not clean; commit or stash before merging",
                is_error=True,
            )
        head_before = _git(repo, "rev-parse", "HEAD").stdout.strip()
        on_into = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        if on_into != args.into:
            co = _git(repo, "checkout", "--quiet", args.into)
            if co.returncode != 0:
                return ToolResult(output=f"cannot check out {args.into}: {co.stderr.strip()}", is_error=True)

        # Rule 3: merge, conflict as a result. --no-commit leaves the merge
        # staged for the post-merge acceptance check (separate PR) to run on
        # the integrated tree BEFORE the commit is finalized; --no-ff keeps
        # the merge commit even when fast-forwardable, so the item's merge
        # is a named event in git log.
        merge = _git(repo, "merge", "--no-ff", "--no-commit", args.branch)
        if merge.returncode != 0:
            conflicts = _git(repo, "diff", "--name-only", "--diff-filter=U").stdout.splitlines()
            _git(repo, "merge", "--abort")
            # restore the operator's starting point exactly
            if on_into != args.into:
                _git(repo, "checkout", "--quiet", on_into or head_before)
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

        # Clean merge: finalize it. The acceptance check re-runs on this
        # tree in its own PR; this tool does NOT claim acceptance ran.
        commit = _git(repo, "commit", "--quiet", "-m",
                      f"bb merge: {args.item_id} from {args.branch}\n\n"
                      f"what: {args.intent.what.strip()}\nwhy: {args.intent.why.strip()}")
        if commit.returncode != 0:
            _git(repo, "merge", "--abort")
            return ToolResult(output=f"merge commit failed: {commit.stderr.strip()}", is_error=True)
        sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
        return ToolResult(
            output=json.dumps({
                "merged": True,
                "item_id": args.item_id,
                "branch": args.branch,
                "into": args.into,
                "commit": sha,
                "intent": {"what": args.intent.what.strip(), "why": args.intent.why.strip()},
                "acceptance": "pending — the post-merge check runs separately",
            }, indent=2),
            metadata={"commit": sha, "item_id": args.item_id},
        )