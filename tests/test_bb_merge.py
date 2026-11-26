"""The blackboard merge tool — intent required, dependencies green, conflict
as a result. Real git repos in tmp_path; the board is a real orphan ref."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prometheus.tools.base import ToolExecutionContext  # noqa: E402
from prometheus.tools.builtin.bb_merge import (  # noqa: E402
    BbMergeInput,
    BbMergeTool,
    MergeIntent,
    merge_tool_enabled,
)


def _git(cwd: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(cwd), *args],
                       capture_output=True, text=True, check=True)
    return r.stdout.strip()


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "test@example.invalid")
    _git(r, "config", "user.name", "Test")
    (r / "base.txt").write_text("base\n")
    _git(r, "add", ".")
    _git(r, "commit", "-q", "-m", "init")
    return r


def _worker_branch(repo: Path, name: str, file: str, text: str) -> None:
    _git(repo, "checkout", "-q", "-b", name)
    (repo / file).write_text(text)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", f"work on {name}")
    _git(repo, "checkout", "-q", "main")


def _board_result(repo: Path, mission: str, item: str, exit_code: int) -> None:
    """Write a post-merge result onto an orphan board ref, no checkout."""
    rec = json.dumps({"item_id": item, "kind": "merged", "exit_code": exit_code,
                      "tree": "x", "command": "true"})
    blob = subprocess.run(["git", "-C", str(repo), "hash-object", "-w", "--stdin"],
                          input=rec, capture_output=True, text=True, check=True).stdout.strip()
    ref = f"refs/blackboard/{mission}"
    # read the current tree (or start empty), add the entry, commit-tree
    path = f"results/merged/{item}/r1.json"
    try:
        parent = _git(repo, "rev-parse", ref)
        tree = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", parent],
                              capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError:
        parent = None
        tree = ""
    entries = {}
    for line in tree.splitlines():
        meta, p = line.split("\t", 1)
        entries[p] = meta.split()[2]
    entries[path] = blob
    # nested paths need nested trees; build them bottom-up
    new_tree = _write_tree(repo, entries)
    args = ["git", "-C", str(repo), "commit-tree", new_tree, "-m", "board"]
    if parent:
        args += ["-p", parent]
    commit = subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
    _git(repo, "update-ref", ref, commit)


def _write_tree(repo: Path, entries: dict[str, str]) -> str:
    """Build a git tree object from {path: blob_sha}, nesting as needed."""
    dirs: dict = {}
    files: dict = {}
    for p, sha in entries.items():
        if "/" in p:
            head, rest = p.split("/", 1)
            dirs.setdefault(head, {})[rest] = sha
        else:
            files[p] = sha
    lines = [f"100644 blob {sha}\t{p}\n" for p, sha in sorted(files.items())]
    for d, sub in sorted(dirs.items()):
        sub_tree = _write_tree(repo, sub)
        lines.append(f"040000 tree {sub_tree}\t{d}\n")
    return subprocess.run(["git", "-C", str(repo), "mktree"],
                          input="".join(lines), capture_output=True, text=True,
                          check=True).stdout.strip()


def _args(repo: Path, **kw) -> BbMergeInput:
    base = dict(repo=str(repo), branch="bb/m1/i1/agent-a", into="main",
                item_id="m1.i1", intent=MergeIntent(what="added the thing", why="the item asks for it"))
    base.update(kw)
    return BbMergeInput(**base)


@pytest.fixture()
def tool() -> BbMergeTool:
    return BbMergeTool()


@pytest.fixture()
def ctx(tmp_path: Path) -> ToolExecutionContext:
    return ToolExecutionContext(cwd=tmp_path)


# ── the gate ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("config,expected", [
    (None, False), ({}, False),
    ({"blackboard": {}}, False),
    ({"blackboard": {"merge_tool": {}}}, False),
    ({"blackboard": {"merge_tool": {"enabled": "true"}}}, False),
    ({"blackboard": {"merge_tool": {"enabled": True}}}, True),
])
def test_the_gate_is_literal_true_only(config, expected):
    assert merge_tool_enabled(config) is expected


def test_shipped_template_keeps_the_tool_off():
    import yaml
    template = Path(__file__).parent.parent / "config" / "prometheus.yaml.default"
    cfg = yaml.safe_load(template.read_text())
    assert merge_tool_enabled(cfg) is False


def test_the_intent_block_is_required_by_the_schema():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        BbMergeInput(repo="/x", branch="b", into="main", item_id="i")  # no intent
    with pytest.raises(ValidationError):
        MergeIntent(what="", why="y")
    with pytest.raises(ValidationError):
        MergeIntent(what="x", why="")


# ── rule 1: no intent, no merge ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_blank_intent_is_refused_before_any_git(repo, tool, ctx):
    _worker_branch(repo, "bb/m1/i1/agent-a", "f.txt", "work\n")
    args = _args(repo)
    # pydantic models are mutable by default; blank the field post-validation
    args.intent.what = "   "
    res = await tool.execute(args, ctx)
    assert res.is_error
    assert "intent" in res.output
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "main"


# ── rule 2: dependencies ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_missing_dependency_result_refuses_with_the_id_named(repo, tool, ctx):
    _worker_branch(repo, "bb/m1/i2/agent-a", "f.txt", "work\n")
    res = await tool.execute(_args(repo, branch="bb/m1/i2/agent-a", item_id="m1.i2",
                                 depends_on=["m1.i1"]), ctx)
    assert res.is_error
    assert "m1.i1" in res.output
    assert _git(repo, "rev-parse", "HEAD") == _git(repo, "rev-parse", "main")


@pytest.mark.asyncio
async def test_a_non_green_dependency_refuses(repo, tool, ctx):
    _board_result(repo, "m1", "m1.i1", exit_code=1)
    _worker_branch(repo, "bb/m1/i2/agent-a", "f.txt", "work\n")
    res = await tool.execute(_args(repo, branch="bb/m1/i2/agent-a", item_id="m1.i2",
                                 depends_on=["m1.i1"], mission="m1"), ctx)
    assert res.is_error
    assert "m1.i1" in res.output


@pytest.mark.asyncio
async def test_a_green_dependency_allows_the_merge(repo, tool, ctx):
    _board_result(repo, "m1", "m1.i1", exit_code=0)
    _worker_branch(repo, "bb/m1/i2/agent-a", "f.txt", "work\n")
    res = await tool.execute(_args(repo, branch="bb/m1/i2/agent-a", item_id="m1.i2",
                                 depends_on=["m1.i1"], mission="m1"), ctx)
    assert not res.is_error, res.output
    body = json.loads(res.output)
    assert body["merged"] is True
    assert body["acceptance"].startswith("pending")
    assert (repo / "f.txt").read_text() == "work\n"


# ── rule 3: conflict is a result ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_conflict_is_returned_and_the_merge_aborted(repo, tool, ctx):
    # two branches touching the same line
    _git(repo, "checkout", "-q", "-b", "bb/m1/i1/agent-a")
    (repo / "base.txt").write_text("side A\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "a")
    _git(repo, "checkout", "-q", "main")
    (repo / "base.txt").write_text("side B\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "b")
    head = _git(repo, "rev-parse", "HEAD")
    res = await tool.execute(_args(repo), ctx)
    assert res.is_error
    body = json.loads(res.output)
    assert body["merged"] is False
    assert "base.txt" in body["conflict"]
    # the tree is exactly where it started: merge aborted, no MERGE_HEAD
    assert _git(repo, "rev-parse", "HEAD") == head
    assert not (repo / ".git" / "MERGE_HEAD").exists()
    assert (repo / "base.txt").read_text() == "side B\n"


@pytest.mark.asyncio
async def test_a_clean_merge_commits_with_the_intent_in_the_message(repo, tool, ctx):
    _worker_branch(repo, "bb/m1/i1/agent-a", "new.txt", "new\n")
    res = await tool.execute(_args(repo), ctx)
    assert not res.is_error, res.output
    msg = _git(repo, "log", "-1", "--format=%B")
    assert "bb merge: m1.i1" in msg
    assert "what: added the thing" in msg
    assert "why: the item asks for it" in msg
    # a merge commit, not a fast-forward
    parents = _git(repo, "log", "-1", "--format=%P").split()
    assert len(parents) == 2


@pytest.mark.asyncio
async def test_unknown_branch_refused(repo, tool, ctx):
    res = await tool.execute(_args(repo, branch="bb/m1/nope/agent-a"), ctx)
    assert res.is_error
    assert "unknown branch" in res.output


@pytest.mark.asyncio
async def test_a_dirty_tree_refused(repo, tool, ctx):
    _worker_branch(repo, "bb/m1/i1/agent-a", "f.txt", "work\n")
    (repo / "base.txt").write_text("operator's uncommitted work\n")
    res = await tool.execute(_args(repo), ctx)
    assert res.is_error
    assert "not clean" in res.output
    assert (repo / "base.txt").read_text() == "operator's uncommitted work\n"


@pytest.mark.asyncio
async def test_not_a_repo_refused(tmp_path, tool, ctx):
    res = await tool.execute(_args(tmp_path), ctx)
    assert res.is_error
    assert "not a git repository" in res.output