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

BEHIND IS NOT A SECURITY CONDITION (2026-08-16)
-----------------------------------------------
``test_refuses_main_behind_origin`` used to live here and asserted the
opposite of ``test_warns_and_starts_when_behind_origin`` below. It was not
testing a bug — it faithfully pinned a deliberate decision, and that decision
is what changed. ``!=`` is symmetric, so one comparison was refusing in two
very different situations: AHEAD (unpushed commits deploying unreviewed — the
incident) and BEHIND (a checkout whose every commit is on origin/main and was
reviewed). Refusing on BEHIND made deliberate dark-merge incompatible with
surviving an unrelated reboot: ``ExecStartPre`` failure plus
``StartLimitBurst=3`` parks the unit in ``failed``, unattended, for a reason
that has nothing to do with why the machine restarted.

The script already computed ``ahead`` and ``behind`` separately and branched
on them to choose a MESSAGE, then discarded the distinction for the VERDICT.
This suite now pins the verdict in both directions.
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
    """Unpushed commits would deploy without ever having been reviewed.

    This is the original incident and the one direction that must never
    become permissive."""
    (repo / "b.txt").write_text("b\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "two")
    head = _git(repo, "rev-parse", "HEAD")
    origin = _git(repo, "rev-parse", "refs/remotes/origin/main")

    r = _run(repo)
    assert r.returncode == 1
    assert "AHEAD" in r.stderr
    assert "ahead 1" in r.stderr
    assert "unpushed" in r.stderr
    # Both SHAs, so the operator can see WHICH two commits disagree without
    # running git themselves.
    assert head[:7] in r.stderr, f"HEAD sha absent from message:\n{r.stderr}"
    assert origin[:7] in r.stderr, f"origin sha absent from message:\n{r.stderr}"
    # The stale-tracking-ref trap: an "ahead" refusal is the one that can be
    # WRONG, so the message must point at fetch before the override.
    assert "fetch origin" in r.stderr


# ── The reversal: BEHIND warns and starts ───────────────────────────────────

def test_warns_and_starts_when_behind_origin(repo):
    """Old code, not unmerged code — every commit here is on origin/main.

    Replaces ``test_refuses_main_behind_origin``. See the module docstring:
    the decision changed, not the implementation's correctness."""
    (repo / "b.txt").write_text("b\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "two")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    _git(repo, "reset", "-q", "--hard", "HEAD~1")
    head = _git(repo, "rev-parse", "HEAD")
    origin = _git(repo, "rev-parse", "refs/remotes/origin/main")

    r = _run(repo)
    assert r.returncode == 0, f"guard refused a pure fast-forward gap:\n{r.stderr}"
    assert "WARNING" in r.stderr
    assert "BEHIND" in r.stderr
    assert "behind 1" in r.stderr
    assert head[:7] in r.stderr, f"HEAD sha absent from message:\n{r.stderr}"
    assert origin[:7] in r.stderr, f"origin sha absent from message:\n{r.stderr}"
    # Starting is not the same as being silent about it.
    assert "pull --ff-only" in r.stderr
    assert "OK:" not in r.stderr, "a behind checkout must not report itself as OK"


def test_behind_by_many_reports_the_real_count(repo):
    """The count is load-bearing: '3 behind' and '1 behind' are different
    operational situations and the operator reads the number."""
    for n in ("two", "three", "four"):
        (repo / f"{n}.txt").write_text(f"{n}\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", n)
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    _git(repo, "reset", "-q", "--hard", "HEAD~3")

    r = _run(repo)
    assert r.returncode == 0
    assert "behind 3" in r.stderr


# ── Divergence must still refuse ────────────────────────────────────────────

def test_refuses_divergent_history(repo):
    """A checkout that is BOTH ahead and behind is not a fast-forward gap.

    NOTE on what this pins. Divergence is caught by the ``ahead != 0``
    branch, which fires first — so this asserts the OUTCOME (refusal, and
    specifically not the permissive behind-branch). The explicit
    ``merge-base --is-ancestor`` in the script is a SECOND lock on the same
    door: given ``ahead`` comes from ``rev-list --count origin/main..HEAD``,
    ``ahead == 0`` already implies ancestry, so no real git state can reach
    the ancestor check with a divergent history. It is defence in depth, and
    §3b's lesson is that defence in depth is exactly what makes a control
    hard to test independently — recorded here rather than papered over with
    a test that would not really exercise it."""
    base = _git(repo, "rev-parse", "HEAD")
    # A commit only origin/main has.
    _git(repo, "checkout", "-q", "-b", "tmp-origin")
    (repo / "origin-side.txt").write_text("o\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "origin side")
    origin_side = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "reset", "-q", "--hard", base)
    # A different commit only main has.
    (repo / "local-side.txt").write_text("l\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "local side")
    _git(repo, "update-ref", "refs/remotes/origin/main", origin_side)
    head = _git(repo, "rev-parse", "HEAD")

    r = _run(repo)
    assert r.returncode == 1, f"divergent history was allowed to boot:\n{r.stderr}"
    assert "ahead 1" in r.stderr and "behind 1" in r.stderr
    assert head[:7] in r.stderr
    assert origin_side[:7] in r.stderr
    # The load-bearing negative: divergence must NOT take the permissive path.
    assert "starting anyway" not in r.stderr


# ── A broken detector must not fall through to permissive ───────────────────

def _git_shim(shim_dir: Path, fail_when: str) -> Path:
    """A ``git`` earlier on PATH that exits non-zero when the joined argv
    contains *fail_when*, and delegates everything else to the real binary."""
    shim_dir.mkdir(exist_ok=True)
    real_git = subprocess.run(
        ["bash", "-c", "command -v git"], capture_output=True, text=True, check=True,
    ).stdout.strip()
    shim = shim_dir / "git"
    shim.write_text(
        "#!/usr/bin/env bash\n"
        'case " $* " in\n'
        f'  *{fail_when}*) exit 3 ;;\n'
        "esac\n"
        f'exec {real_git} "$@"\n'
    )
    shim.chmod(0o755)
    return shim_dir


@pytest.mark.parametrize(
    "fail_when, label",
    [
        ("rev-list", "both counts fail"),
        ("refs/remotes/origin/main..HEAD", "only AHEAD fails"),
        ("HEAD..refs/remotes/origin/main", "only BEHIND fails"),
    ],
)
def test_refuses_when_rev_list_cannot_count(repo, fail_when, label):
    """CROSS-CUTTING §8. ``rev-list`` failing yields '?', which is UNKNOWN —
    not zero. If '?' reached the ahead==0 path it would be treated as a
    fast-forward gap and START, which is the fail-by-exception state nobody
    chose.

    PARAMETRIZED OVER EACH OPERAND, and that is not decoration. The check is
    ``[ "$ahead" = "?" ] || [ "$behind" = "?" ]`` — an OR with two operands,
    and a single test only ever exercised one of them. Mutation M10 (delete
    the ``behind`` half) SURVIVED the suite as first written. Each operand
    now has a case that fails without it.

    Induced with a ``git`` shim rather than repo state. The obvious
    construction — pointing origin/main at a blob — does NOT work: ``git
    rev-list --count <blob>..HEAD`` SUCCEEDS and returns 1, exercising the
    ahead branch instead. Verified by hand before writing this."""
    shim_dir = _git_shim(repo.parent / "shimbin", fail_when)
    # Make HEAD != origin so the comparison is actually reached.
    (repo / "b.txt").write_text("b\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "two")

    r = _run(repo, PATH=f"{shim_dir}:{os.environ['PATH']}")
    assert r.returncode == 1, (
        f"a broken comparison ({label}) was allowed to boot:\n{r.stderr}"
    )
    assert "cannot count commits" in r.stderr
    assert "UNKNOWN" in r.stderr
    assert "starting anyway" not in r.stderr


# ── Working tree: TRACKED refuses, untracked does not ───────────────────────

def test_refuses_dirty_tracked_file(repo):
    """REVERSAL of the pre-2026-08-16 decision. Uncommitted tracked edits mean
    the running code is not any reviewed commit, so ``running_sha`` names a
    commit whose content is not what is on disk."""
    (repo / "a.txt").write_text("locally edited\n")

    r = _run(repo)
    assert r.returncode == 1, f"a dirty tracked tree was allowed to boot:\n{r.stderr}"
    assert "TRACKED" in r.stderr
    assert "a.txt" in r.stderr, "the message must name the offending file"
    assert "running_sha" in r.stderr


def test_refuses_staged_but_uncommitted(repo):
    """Staged is still uncommitted — ``diff-index HEAD`` sees the index too."""
    (repo / "a.txt").write_text("staged edit\n")
    _git(repo, "add", "-A")

    r = _run(repo)
    assert r.returncode == 1
    assert "TRACKED" in r.stderr


def test_untracked_files_do_NOT_refuse(repo):
    """The over-refusal direction (§2c). A stray scratch file must never be
    able to take the daemon down — that is a boot gate reading as 'safe'
    while actually being unbootable."""
    (repo / "scratch.log").write_text("noise\n")
    (repo / "another-stray.tmp").write_text("noise\n")

    r = _run(repo)
    assert r.returncode == 0, f"an untracked file caused an outage:\n{r.stderr}"
    assert "OK:" in r.stderr


def test_untracked_files_do_not_refuse_even_when_behind(repo):
    """The two new branches must compose: untracked noise on a behind
    checkout still warns-and-starts rather than refusing."""
    (repo / "b.txt").write_text("b\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "two")
    _git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
    _git(repo, "reset", "-q", "--hard", "HEAD~1")
    (repo / "scratch.log").write_text("noise\n")

    r = _run(repo)
    assert r.returncode == 0
    assert "BEHIND" in r.stderr


def test_dirty_refuses_even_when_otherwise_equal(repo):
    """Guard identity (§3b): assert the refusal is the DIRTY one, not some
    other check happening to fire. Here refs are equal, so only the tracked
    modification can be the reason."""
    (repo / "a.txt").write_text("edited\n")

    r = _run(repo)
    assert r.returncode == 1
    assert "TRACKED" in r.stderr
    assert "AHEAD" not in r.stderr and "BEHIND" not in r.stderr


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
