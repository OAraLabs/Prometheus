"""Every staged file is scanned from the index, and a hit never prints the secret.

WHAT HAPPENED
-------------
`.githooks/pre-commit` built its scan list from a list of status letters, and
each letter it lacked was a way past it:

- `--diff-filter=ACM`: git detects renames by default, so a file renamed AND
  edited in one commit has status R (`R083 a.py b.py`) and was dropped. A key
  added during a `git mv` committed while the hook said "All clean".
- `--diff-filter=ACMR`: a committed symlink replaced by a regular file has
  status T, and was dropped the same way.

And check_pattern skipped any listed file missing from disk (`[ -f "$file" ]`),
although it reads the staged blob: a key staged and then deleted from the
working tree was committed from the index while the hook said "All clean".

Separately, a hit printed `BLOCKED  <file>:<line>:<the whole line>` -- the
secret itself, echoed to the terminal and to anything capturing it.

WHAT THIS FILE ASSERTS
----------------------
In a throwaway repo, running the real hook:

    renamed and edited, carrying a key       refused; names file:line and the pattern
    staged, then deleted from disk           refused
    symlink replaced by a file with a key    refused
    any hit                                  the key never appears in the output
    controls                                 a clean rename, a pure deletion, a new
                                             clean symlink and a docs-only commit
                                             all pass

Each refusal failed against the hook as it was before its fix.

The key is generated at run time, so this file holds nothing key-shaped for the
hook itself, or for tests/test_sdist_contents.py, to find.
"""

from __future__ import annotations

import os
import secrets
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
HOOK = REPO / ".githooks" / "pre-commit"

pytestmark = [
    pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH"),
    pytest.mark.skipif(shutil.which("bash") is None, reason="bash not on PATH"),
]

# Ten ordinary lines, so one appended line keeps git's similarity score far
# above its 50% rename threshold.
BODY = "".join(f"setting_{i} = {i}\n" for i in range(10))
KEY_LINE_NO = len(BODY.splitlines()) + 1
LABEL = "Provider API key"
PREFIX = "sk-ant-"


def _fake_key() -> str:
    """A fresh Anthropic-shaped key, never the same twice.

    Hex without 0: the hook's placeholder exemption skips any line holding
    "0000", so a random run of zeros would turn this hit into a pass.
    """
    return PREFIX + "".join(secrets.choice("123456789abcdef") for _ in range(48))


def _env() -> dict[str, str]:
    env = dict(os.environ)
    # The throwaway repo must not inherit this machine's git config: a global
    # diff.renames=false would hide the rename, and a global hooksPath or
    # commit.gpgsign would run hooks or ask for a passphrase on the setup commit.
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    # Guard 1 refuses commits from a non-worktree checkout; this repo is one.
    # That guard is not what these tests are about.
    env["PROMETHEUS_ALLOW_DEV_COMMIT"] = "1"
    return env


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, env=_env(), check=True, capture_output=True, text=True
    ).stdout


def _repo(tmp_path: Path) -> Path:
    """A throwaway repo with the real hook in place."""
    repo = tmp_path / "r"
    (repo / ".githooks").mkdir(parents=True)
    shutil.copy2(HOOK, repo / ".githooks" / "pre-commit")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "diff.renames", "true")
    return repo


def _run_hook(repo: Path) -> tuple[int, str]:
    proc = subprocess.run(
        ["bash", str(repo / ".githooks" / "pre-commit")],
        cwd=repo, env=_env(), capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


def _symlink(link: Path, target: str) -> None:
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks not supported on this platform")


def _stage_added(repo: Path, key: str) -> str:
    (repo / "app.py").write_text(BODY + f'qzv_assigned = "{key}"\n', encoding="utf-8")
    _git(repo, "add", "app.py")
    return "app.py"


def _stage_renamed_and_edited(repo: Path, key: str | None) -> str:
    """Commit a clean file, `git mv` it, append `key` (if any), stage it."""
    (repo / "settings.py").write_text(BODY, encoding="utf-8")
    _git(repo, "add", "settings.py")
    _git(repo, "commit", "-q", "-m", "clean")
    _git(repo, "mv", "settings.py", "config.py")
    if key is not None:
        with (repo / "config.py").open("a", encoding="utf-8") as f:
            f.write(f'qzv_assigned = "{key}"\n')
        _git(repo, "add", "config.py")
    status = _git(repo, "diff", "--cached", "--name-status")
    assert status.startswith("R"), f"git saw no rename, so this proves nothing:\n{status}"
    return "config.py"


def test_a_renamed_and_edited_file_is_scanned(tmp_path: Path) -> None:
    """THE GAP: --diff-filter=ACM dropped status R, so this file was never read."""
    repo = _repo(tmp_path)
    path = _stage_renamed_and_edited(repo, _fake_key())
    code, out = _run_hook(repo)
    assert code != 0, "a key added during a rename was allowed through"
    assert "All clean" not in out
    assert f"{path}:{KEY_LINE_NO}" in out
    assert LABEL in out


def test_a_clean_rename_still_passes(tmp_path: Path) -> None:
    """Scanning renames must not turn every `git mv` into a refusal.

    "All clean" is the proof the file was read: before the fix a pure rename
    left the scan list empty, and the hook exited 0 without scanning or saying so.
    """
    repo = _repo(tmp_path)
    _stage_renamed_and_edited(repo, None)
    code, out = _run_hook(repo)
    assert code == 0, out
    assert "All clean" in out


def test_a_key_staged_then_deleted_from_disk_is_refused(tmp_path: Path) -> None:
    """THE SECOND GAP: `[ -f "$file" ]` skipped it, but the index is what commits."""
    key = _fake_key()
    repo = _repo(tmp_path)
    path = _stage_added(repo, key)
    (repo / path).unlink()
    status = _git(repo, "diff", "--cached", "--name-status")
    assert status.startswith("A"), f"expected a staged add:\n{status}"
    assert not (repo / path).exists(), "the file must be gone from disk to prove anything"
    code, out = _run_hook(repo)
    shown = out.replace(key, "<THE KEY>")
    printed_key = key in out
    assert not printed_key, f"the hook printed the key it found:\n{shown}"
    assert code != 0, f"a key committed from the index was allowed through:\n{shown}"
    assert "All clean" not in out
    assert f"{path}:{KEY_LINE_NO}" in out
    assert LABEL in out


def test_a_symlink_replaced_by_a_file_holding_a_key_is_refused(tmp_path: Path) -> None:
    """THE THIRD GAP: a type change is status T, which ACMR dropped too."""
    key = _fake_key()
    repo = _repo(tmp_path)
    (repo / "settings.py").write_text(BODY, encoding="utf-8")
    _symlink(repo / "current.py", "settings.py")
    _git(repo, "add", "settings.py", "current.py")
    _git(repo, "commit", "-q", "-m", "clean")
    (repo / "current.py").unlink()
    (repo / "current.py").write_text(BODY + f'qzv_assigned = "{key}"\n', encoding="utf-8")
    _git(repo, "add", "current.py")
    status = _git(repo, "diff", "--cached", "--name-status")
    assert status.startswith("T"), f"git saw no type change, so this proves nothing:\n{status}"
    code, out = _run_hook(repo)
    shown = out.replace(key, "<THE KEY>")
    printed_key = key in out
    assert not printed_key, f"the hook printed the key it found:\n{shown}"
    assert code != 0, f"a key arriving as a type change was allowed through:\n{shown}"
    assert "All clean" not in out
    assert f"current.py:{KEY_LINE_NO}" in out
    assert LABEL in out


def test_a_pure_deletion_still_passes(tmp_path: Path) -> None:
    """Deletions are the one status not scanned: nothing is staged to read.

    The deleted file holds a key on purpose. Deleting it is how a key gets
    removed, so that must never be refused.
    """
    key = _fake_key()
    repo = _repo(tmp_path)
    _stage_added(repo, key)
    _git(repo, "commit", "-q", "-m", "a key that has to go")
    _git(repo, "rm", "-q", "app.py")
    status = _git(repo, "diff", "--cached", "--name-status")
    assert status.startswith("D"), status
    code, out = _run_hook(repo)
    assert code == 0, out.replace(key, "<THE KEY>")


def test_a_new_clean_symlink_passes(tmp_path: Path) -> None:
    """A symlink's staged blob is its target path; a clean one is read and passes."""
    repo = _repo(tmp_path)
    (repo / "settings.py").write_text(BODY, encoding="utf-8")
    _git(repo, "add", "settings.py")
    _git(repo, "commit", "-q", "-m", "clean")
    _symlink(repo / "current.py", "settings.py")
    _git(repo, "add", "current.py")
    status = _git(repo, "diff", "--cached", "--name-status")
    assert status.startswith("A"), status
    code, out = _run_hook(repo)
    assert code == 0, out
    assert "All clean" in out


def test_a_docs_only_commit_still_passes(tmp_path: Path) -> None:
    """The on-disk check also skipped the empty line `<<<` yields for an empty list.

    A docs-only commit leaves the code-scoped list empty. Dropping the check with
    nothing in its place refused every such commit ("SCANNER DID NOT RUN: git
    show : failed"), so an empty name is still skipped; only the disk is not.
    """
    repo = _repo(tmp_path)
    (repo / "docs").mkdir()
    (repo / "docs" / "notes.md").write_text("ordinary prose\n", encoding="utf-8")
    _git(repo, "add", "docs/notes.md")
    code, out = _run_hook(repo)
    assert code == 0, out
    assert "All clean" in out


@pytest.mark.parametrize(
    "stage", [_stage_added, _stage_renamed_and_edited], ids=["added", "renamed"]
)
def test_a_hit_prints_file_line_and_pattern_never_the_value(
    tmp_path: Path, stage: Callable[[Path, str], str]
) -> None:
    """The refusal says where and why, and never repeats what it found."""
    key = _fake_key()
    repo = _repo(tmp_path)
    path = stage(repo, key)
    code, out = _run_hook(repo)
    # Booleans first: `assert key not in out` would make pytest's own report
    # quote the key while explaining the failure.
    shown = out.replace(key, "<THE KEY>")
    printed_key = key in out
    printed_part = key[len(PREFIX):][:12] in out
    assert not printed_key, f"the hook printed the key it found:\n{shown}"
    assert not printed_part, f"the hook printed part of the key:\n{shown}"
    assert "qzv_assigned" not in out, f"the hook printed the matched line:\n{shown}"
    assert code != 0, f"the key was not refused:\n{shown}"
    assert f"{path}:{KEY_LINE_NO}" in out, f"the refusal must name file:line:\n{shown}"
    assert LABEL in out, f"the refusal must name the pattern:\n{shown}"
