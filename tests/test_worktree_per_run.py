"""Worktree per run (blackboard protocol §5, experimental, off by default).

The clone path is unchanged; these tests pin the NEW worktree path and the
off-by-default gate. Real git repos in tmp_path — no mocks of git itself.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prometheus.coding.sandbox import (  # noqa: E402
    clone_repo_for_sandbox,
    remove_run_worktree,
)


def _git(cwd: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(cwd), *args],
                       capture_output=True, text=True, check=True)
    return r.stdout.strip()


@pytest.fixture()
def source_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    (repo / "file.txt").write_text("hello\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def test_the_shipped_default_is_off():
    import yaml
    template = Path(__file__).parent.parent / "config" / "prometheus.yaml.default"
    cfg = yaml.safe_load(template.read_text())
    assert (cfg.get("coding") or {}).get("worktree_per_run", False) is False


def test_default_path_is_still_a_full_clone(source_repo, tmp_path):
    box = clone_repo_for_sandbox(source_repo, tmp_path / "sandboxes", name="run1")
    assert (box.root / ".git").is_dir()          # a clone has its OWN .git dir
    assert (box.root / "file.txt").read_text() == "hello\n"
    # the source's worktree list is untouched
    assert str(box.root) not in _git(source_repo, "worktree", "list")


def test_worktree_path_creates_a_worktree_and_remove_cleans_it(source_repo, tmp_path):
    box = clone_repo_for_sandbox(source_repo, tmp_path / "sandboxes", name="run2",
                                 worktree=True)
    assert (box.root / ".git").is_file()         # a worktree has a .git FILE
    assert (box.root / "file.txt").read_text() == "hello\n"
    assert str(box.root) in _git(source_repo, "worktree", "list")
    removed, wip = remove_run_worktree(source_repo, box.root)
    assert removed is True and wip is None
    assert not box.root.exists()
    assert str(box.root) not in _git(source_repo, "worktree", "list")


def test_a_branch_created_in_the_worktree_survives_removal(source_repo, tmp_path):
    """The run's work is in the SOURCE's refs — removing the checkout loses
    nothing the report names. This is the property that makes cleanup safe."""
    box = clone_repo_for_sandbox(source_repo, tmp_path / "sandboxes", name="run3",
                                 worktree=True)
    _git(box.root, "checkout", "-q", "-b", "coding/run3-abc123")
    (box.root / "file.txt").write_text("changed\n")
    _git(box.root, "add", ".")
    _git(box.root, "commit", "-q", "-m", "run work")
    sha = _git(box.root, "rev-parse", "HEAD")
    removed, wip = remove_run_worktree(source_repo, box.root)
    assert removed is True and wip is None
    # the branch and its commit are readable from the source
    assert _git(source_repo, "rev-parse", "coding/run3-abc123") == sha
    assert _git(source_repo, "show", "coding/run3-abc123:file.txt") == "changed"


def test_two_worktrees_of_one_source_coexist(source_repo, tmp_path):
    a = clone_repo_for_sandbox(source_repo, tmp_path / "sandboxes", name="a", worktree=True)
    b = clone_repo_for_sandbox(source_repo, tmp_path / "sandboxes", name="b", worktree=True)
    _git(a.root, "checkout", "-q", "-b", "coding/t-a")
    _git(b.root, "checkout", "-q", "-b", "coding/t-b")
    (a.root / "file.txt").write_text("a\n")
    (b.root / "file.txt").write_text("b\n")
    assert (a.root / "file.txt").read_text() == "a\n"
    assert (b.root / "file.txt").read_text() == "b\n"
    assert (source_repo / "file.txt").read_text() == "hello\n"  # source untouched
    assert remove_run_worktree(source_repo, a.root)[0]
    assert remove_run_worktree(source_repo, b.root)[0]


def test_remove_is_idempotent_and_never_raises(source_repo, tmp_path):
    box = clone_repo_for_sandbox(source_repo, tmp_path / "sandboxes", name="run4",
                                 worktree=True)
    assert remove_run_worktree(source_repo, box.root)[0] is True
    assert remove_run_worktree(source_repo, box.root)[0] is True   # already gone
    # a path that was never a worktree: the git command refuses, the backstop
    # removes the directory (it is under the sandbox parent by construction)
    stray = tmp_path / "sandboxes" / "stray"
    stray.mkdir(parents=True)
    assert remove_run_worktree(source_repo, stray)[0] is True


def test_existing_target_still_refused(source_repo, tmp_path):
    dest = tmp_path / "sandboxes" / "run5"
    dest.mkdir(parents=True)
    with pytest.raises(ValueError, match="already exists"):
        clone_repo_for_sandbox(source_repo, tmp_path / "sandboxes", name="run5",
                               worktree=True)


def test_the_session_branch_suffix_is_unique_per_run():
    """CodingSession's branch carries the suffix the runner passes; the clone
    path passes nothing and keeps coding/<task_id> exactly."""
    from prometheus.coding.session import CodingSession
    import inspect
    sig = inspect.signature(CodingSession.__init__)
    assert "branch_suffix" in sig.parameters
    assert sig.parameters["branch_suffix"].default == ""
    # the branch construction uses it (read the source line, not a mock run)
    src = inspect.getsource(CodingSession.__init__)
    assert 'coding/{task.task_id}{branch_suffix}' in src

def test_uncommitted_work_is_wip_committed_before_removal(source_repo, tmp_path):
    """Review #2: a failed/stopped/timed-out run leaves uncommitted work.
    Removal must commit it to the run's branch as a WIP commit — never
    --force it away — and name the branch."""
    box = clone_repo_for_sandbox(source_repo, tmp_path / "sandboxes", name="run6",
                                 worktree=True)
    _git(box.root, "checkout", "-q", "-b", "coding/run6-suffix")
    (box.root / "file.txt").write_text("half-done edit\n")     # tracked, unstaged
    (box.root / "new.txt").write_text("untracked work\n")      # untracked
    removed, wip = remove_run_worktree(source_repo, box.root,
                                       branch="coding/run6-suffix")
    assert removed is True
    assert wip == "coding/run6-suffix"
    # both the edit and the untracked file survive on the branch
    assert _git(source_repo, "show", "coding/run6-suffix:file.txt") == "half-done edit"
    assert _git(source_repo, "show", "coding/run6-suffix:new.txt") == "untracked work"
    assert "WIP" in _git(source_repo, "log", "-1", "--format=%s", "coding/run6-suffix")


def test_detached_dirty_worktree_gets_a_wip_branch(source_repo, tmp_path):
    """A run that died before its branch was created: the worktree is still
    detached. The dirty state gets its own WIP branch, named in the result."""
    box = clone_repo_for_sandbox(source_repo, tmp_path / "sandboxes", name="run7",
                                 worktree=True)
    (box.root / "file.txt").write_text("orphaned work\n")
    removed, wip = remove_run_worktree(source_repo, box.root)
    assert removed is True
    assert wip is not None and wip.startswith("coding/wip-")
    assert _git(source_repo, "show", f"{wip}:file.txt") == "orphaned work"


def test_unsaveable_work_leaves_the_worktree_in_place(source_repo, tmp_path, monkeypatch):
    """If the WIP commit cannot be made, removal REFUSES — the checkout
    stays so nothing uncommitted is destroyed."""
    box = clone_repo_for_sandbox(source_repo, tmp_path / "sandboxes", name="run8",
                                 worktree=True)
    _git(box.root, "checkout", "-q", "-b", "coding/run8")
    (box.root / "file.txt").write_text("precious\n")
    # break commit-making: an index.lock blocks `git add`
    gitdir = _git(box.root, "rev-parse", "--absolute-git-dir")
    (Path(gitdir) / "index.lock").write_text("")
    removed, wip = remove_run_worktree(source_repo, box.root, branch="coding/run8")
    assert removed is False
    assert wip is None
    assert box.root.exists()
    assert (box.root / "file.txt").read_text() == "precious\n"


def test_clean_run_reports_no_wip_branch(source_repo, tmp_path):
    box = clone_repo_for_sandbox(source_repo, tmp_path / "sandboxes", name="run9",
                                 worktree=True)
    _git(box.root, "checkout", "-q", "-b", "coding/run9")
    (box.root / "file.txt").write_text("done\n")
    _git(box.root, "add", ".")
    _git(box.root, "commit", "-q", "-m", "finished")
    removed, wip = remove_run_worktree(source_repo, box.root, branch="coding/run9")
    assert removed is True and wip is None
