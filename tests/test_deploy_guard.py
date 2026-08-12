"""The deploy guard refuses to boot the daemon from an unmerged checkout.

WHY THIS EXISTS
---------------
The daemon deploys from a checkout that concurrent Claude Code sessions
share, so ``tree_head`` is whatever branch that checkout happens to be on.
Three times in eleven days it was left on a feature branch. On 2026-08-12 the
live reading was ``running_sha 19e9db4`` (main) against ``tree_head d8a6115``
(``feat/firstlight-fl2u-absence-defaults``), ``stale: true`` — a restart at
that moment would have deployed an unmerged branch to production, and nothing
in the merge, the CI, or the PR state would have hinted at it.

Standing-Principles CROSS-CUTTING §7 documented this on 2026-08-02 and it
happened twice more. §13 is why: *"writing a trap down does not stop you
walking into it. If the failure is mechanical, the countermeasure has to be
mechanical."* ``scripts/deploy_guard.sh`` is the mechanical form, wired as
``ExecStartPre`` so the refusal lands where the harm is — the boot — rather
than at the keystroke that checks out a branch.

BOTH DIRECTIONS (§2c)
---------------------
A guard that refuses everything is indistinguishable from a working one if
the suite only tests refusals — that is exactly how the media surface shipped
PDF-only. So the good state gets its own test: the guard must ALLOW a
checkout that is on main and equal to origin/main. If it ever stops doing
that, the daemon cannot boot at all.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

GUARD = Path(__file__).resolve().parent.parent / "scripts" / "deploy_guard.sh"


def _run(repo: Path, **env_overrides: str) -> subprocess.CompletedProcess:
    env = {**os.environ, **env_overrides}
    return subprocess.run(
        ["bash", str(GUARD), str(repo)],
        capture_output=True, text=True, env=env, timeout=30,
    )


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A checkout on `main`, equal to a local origin/main ref."""
    r = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "-b", "main", str(r)], check=True)
    _git(r, "config", "user.email", "t@example.com")
    _git(r, "config", "user.name", "t")
    (r / "a.txt").write_text("a\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "one")
    _git(r, "update-ref", "refs/remotes/origin/main", "HEAD")
    return r


def test_the_script_is_executable_and_present():
    """An ExecStartPre pointing at a missing file fails the start — safe, but
    the reason would be systemd's, not the guard's."""
    assert GUARD.exists(), f"{GUARD} is missing"
    assert os.access(GUARD, os.X_OK), f"{GUARD} is not executable"


# ── ADMISSION direction — the state that MUST boot ──────────────────────────

def test_allows_main_equal_to_origin_main(repo):
    """The good state. Without this test, a guard that refuses everything
    looks identical to one that works — and the daemon never starts."""
    r = _run(repo)
    assert r.returncode == 0, f"guard refused a clean main checkout:\n{r.stderr}"
    assert "OK:" in r.stderr


# ── BREACH direction — every state that must NOT boot ───────────────────────

def test_refuses_a_feature_branch(repo):
    """THE incident: the deploy tree left on someone's branch."""
    _git(repo, "checkout", "-qb", "feat/x")
    r = _run(repo)
    assert r.returncode == 1
    assert "not main" in r.stderr
    # The message must name the harm, not just the condition — an operator
    # who reads "wrong branch" and shrugs is why this happened three times.
    assert "UNMERGED BRANCH" in r.stderr
    assert "git -C" in r.stderr and "checkout main" in r.stderr


def test_refuses_a_feature_branch_even_at_the_same_commit(repo):
    """A branch created at main's tip has an identical tree — and is still
    not main. The check is on the BRANCH, not just the SHA, because the next
    commit on it is what would silently deploy."""
    _git(repo, "checkout", "-qb", "tmp/same-sha")
    assert _git(repo, "rev-parse", "HEAD") == _git(
        repo, "rev-parse", "refs/remotes/origin/main")
    r = _run(repo)
    assert r.returncode == 1
    assert "not main" in r.stderr


def test_refuses_main_ahead_of_origin(repo):
    """Unpushed commits would deploy without ever having been reviewed."""
    (repo / "b.txt").write_text("b\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "two")
    r = _run(repo)
    assert r.returncode == 1
    assert "ahead 1" in r.stderr
    assert "unpushed" in r.stderr


def test_refuses_main_behind_origin(repo):
    """A stale checkout deploys code older than what was merged."""
    (repo / "b.txt").write_text("b\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "two")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    _git(repo, "reset", "-q", "--hard", "HEAD~1")
    r = _run(repo)
    assert r.returncode == 1
    assert "behind 1" in r.stderr
    assert "pull --ff-only" in r.stderr


def test_refuses_detached_head(repo):
    _git(repo, "checkout", "-q", "--detach", "HEAD")
    r = _run(repo)
    assert r.returncode == 1
    assert "DETACHED HEAD" in r.stderr


def test_refuses_when_origin_main_ref_is_missing(tmp_path):
    """A repo with no origin/main cannot be compared, so it cannot be
    cleared. The DIAGNOSIS matters as much as the refusal here: written with
    a bare ``rev-parse``, this case still refused but reported "not equal to
    origin/main" with a garbled SHA, because ``git rev-parse <missing-ref>``
    ECHOES THE REF NAME to stdout while exiting non-zero. ``--verify
    --quiet`` is what makes the message true (§2b — a check that answers
    cleanly about the wrong subject)."""
    r_ = tmp_path / "noorigin"
    subprocess.run(["git", "init", "-q", "-b", "main", str(r_)], check=True)
    _git(r_, "config", "user.email", "t@example.com")
    _git(r_, "config", "user.name", "t")
    (r_ / "a.txt").write_text("a\n")
    _git(r_, "add", "-A")
    _git(r_, "commit", "-qm", "one")
    r = _run(r_)
    assert r.returncode == 1
    assert "no local refs/remotes/origin/main" in r.stderr
    assert "fetch origin" in r.stderr


def test_refuses_a_directory_that_is_not_a_checkout(tmp_path):
    """Fail CLOSED on a broken detector. A guard that boots anyway when it
    cannot tell what would deploy is not a guard (CROSS-CUTTING §8)."""
    d = tmp_path / "plain"
    d.mkdir()
    r = _run(d)
    assert r.returncode == 1
    assert "not a git checkout" in r.stderr


# ── The override ────────────────────────────────────────────────────────────

def test_override_allows_the_worst_case_and_announces_itself(repo):
    """A guard with no override is one that gets commented out the first time
    it blocks something legitimate — the ``--no-verify`` reflex, which
    CROSS-CUTTING §4 names explicitly. The override must therefore exist AND
    be loud: a silent bypass is how a disabled guard looks healthy."""
    _git(repo, "checkout", "-qb", "feat/x")
    r = _run(repo, PROMETHEUS_ALLOW_UNMERGED_DEPLOY="1")
    assert r.returncode == 0
    assert "OVERRIDE ACTIVE" in r.stderr
    assert "not on main" in r.stderr


def test_override_is_off_unless_it_is_exactly_1(repo):
    """A truthy-looking value must not silently disable the guard."""
    _git(repo, "checkout", "-qb", "feat/x")
    for value in ("0", "", "false", "no", "true", "yes"):
        r = _run(repo, PROMETHEUS_ALLOW_UNMERGED_DEPLOY=value)
        assert r.returncode == 1, (
            f"PROMETHEUS_ALLOW_UNMERGED_DEPLOY={value!r} disabled the guard"
        )
