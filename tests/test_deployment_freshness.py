"""Deployment freshness — the axis `stale` could never see.

`stale` compares the RUNNING process to the CHECKED-OUT TREE. Both operands
come from the same clone, so a clone that is itself behind origin/main
reports `stale: false`. That happened in production on 2026-09-19: the
daemon served `stale: false` with two merged commits undeployed, and the one
instrument built to catch merged-not-deployed could not see it.

These tests pin the SECOND axis and, specifically, that it compares against
origin rather than against the tree — see the mutation guard at the bottom,
which is the test that would have failed on the original defect.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import prometheus.context.environment as env_mod
from prometheus.context.environment import deployment_freshness


# ── Unit: the rollup, with the git primitives stubbed ───────────────────

def _stub(monkeypatch, *, tree, origin, ahead=0, behind=0, age=30):
    """Pin every git read `deployment_freshness` makes."""
    monkeypatch.setattr(env_mod, "git_head_sha", lambda *a, **k: tree)
    monkeypatch.setattr(env_mod, "origin_main_sha", lambda *a, **k: origin)
    monkeypatch.setattr(env_mod, "origin_ref_age_seconds", lambda *a, **k: age)

    def _counts(range_expr, repo_dir):
        # `origin..HEAD` = ahead, `HEAD..origin` = behind
        return ahead if range_expr.endswith("..HEAD") else behind

    monkeypatch.setattr(env_mod, "_count_commits", _counts)


def test_current_requires_both_axes_in_sync(monkeypatch):
    _stub(monkeypatch, tree="aaaa", origin="aaaa")
    d = deployment_freshness("aaaa")
    assert d["state"] == "current"
    assert d["process_vs_tree"] == "in_sync"
    assert d["tree_vs_origin"] == "in_sync"


def test_behind_origin_is_reported_while_the_process_matches_its_tree(monkeypatch):
    """THE DEFECT, stated as a test.

    running == tree, so `stale` is false and always was. The clone is two
    commits behind origin/main. Before this axis existed the daemon had
    nothing to say here; now it names the state and the remedy.
    """
    _stub(monkeypatch, tree="aaaa", origin="bbbb", ahead=0, behind=2)
    d = deployment_freshness("aaaa")
    assert d["state"] == "behind_origin"
    assert d["process_vs_tree"] == "in_sync"   # `stale` would be False here
    assert d["tree_vs_origin"] == "behind"
    assert d["behind_origin"] == 2
    assert "pull --ff-only" in d["detail"]


def test_behind_tree_outranks_behind_origin(monkeypatch):
    """Both axes bad → name the cheaper remedy. A restart is seconds; a pull
    first would discard the running process's own provenance."""
    _stub(monkeypatch, tree="bbbb", origin="cccc", ahead=0, behind=1)
    d = deployment_freshness("aaaa")
    assert d["state"] == "behind_tree"
    assert d["process_vs_tree"] == "behind"
    assert d["tree_vs_origin"] == "behind"


def test_ahead_of_origin_is_reported_not_enforced(monkeypatch):
    """deploy_guard.sh REFUSES to boot here. /api/status only reports —
    a status endpoint that refused to answer would be a second gate."""
    _stub(monkeypatch, tree="bbbb", origin="aaaa", ahead=3, behind=0)
    d = deployment_freshness("bbbb")
    assert d["state"] == "ahead_of_origin"
    assert d["ahead_of_origin"] == 3


def test_diverged_is_its_own_state(monkeypatch):
    _stub(monkeypatch, tree="bbbb", origin="cccc", ahead=1, behind=1)
    d = deployment_freshness("bbbb")
    assert d["state"] == "diverged"
    assert d["tree_vs_origin"] == "diverged"


# ── Unknown is a THIRD ANSWER and must never read as fresh ──────────────

def test_missing_origin_ref_is_unknown_not_current(monkeypatch):
    """No tracking ref → we cannot know. The whole failure class this work
    exists for is a check reporting clean when it cannot see."""
    _stub(monkeypatch, tree="aaaa", origin="unknown")
    d = deployment_freshness("aaaa")
    assert d["state"] == "unknown"
    assert d["tree_vs_origin"] == "unknown"
    assert d["state"] != "current"


def test_rev_list_failure_is_unknown_not_in_sync(monkeypatch):
    """`rev-list` broke. deploy_guard.sh refuses on its '?' sentinel for the
    same reason: a broken detector must not fall through to permissive."""
    _stub(monkeypatch, tree="aaaa", origin="bbbb")
    monkeypatch.setattr(env_mod, "_count_commits", lambda *a, **k: None)
    d = deployment_freshness("aaaa")
    assert d["state"] == "unknown"
    assert d["tree_vs_origin"] == "unknown"


def test_unknown_tree_is_unknown_on_both_axes(monkeypatch):
    _stub(monkeypatch, tree="unknown", origin="bbbb")
    d = deployment_freshness("aaaa")
    assert d["state"] == "unknown"
    assert d["process_vs_tree"] == "unknown"


def test_unknown_boot_sha_does_not_claim_current(monkeypatch):
    _stub(monkeypatch, tree="aaaa", origin="aaaa")
    d = deployment_freshness("unknown")
    assert d["state"] == "unknown"


def test_every_state_carries_a_remedy_sentence(monkeypatch):
    """A state name with no action is a puzzle at 2am, not a signal."""
    for tree, origin, ahead, behind, running in [
        ("aaaa", "aaaa", 0, 0, "aaaa"),
        ("aaaa", "bbbb", 0, 2, "aaaa"),
        ("bbbb", "cccc", 0, 1, "aaaa"),
        ("bbbb", "aaaa", 3, 0, "bbbb"),
        ("bbbb", "cccc", 1, 1, "bbbb"),
        ("aaaa", "unknown", 0, 0, "aaaa"),
    ]:
        _stub(monkeypatch, tree=tree, origin=origin, ahead=ahead, behind=behind)
        d = deployment_freshness(running)
        assert isinstance(d["detail"], str) and len(d["detail"]) > 20


# ── Integration: a REAL git repo, real refs, no stubs ───────────────────

def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "clone"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "t@example.com")
    _git(r, "config", "user.name", "t")
    (r / "f.txt").write_text("1")
    _git(r, "add", "f.txt")
    _git(r, "commit", "-qm", "one")
    return r


def test_real_repo_behind_origin(repo: Path):
    """Two real commits, origin/main pointed at the newer one, HEAD at the
    older. The plumbing — rev-parse, rev-list, the ref name — is exercised
    for real, not against doubles."""
    first = _git(repo, "rev-parse", "HEAD")
    (repo / "f.txt").write_text("2")
    _git(repo, "commit", "-aqm", "two")
    second = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", env_mod.ORIGIN_MAIN_REF, second)
    _git(repo, "reset", "-q", "--hard", first)

    d = deployment_freshness(first, repo_dir=repo)
    assert d["state"] == "behind_origin"
    assert d["tree_head"] == first and d["origin_main"] == second
    assert d["behind_origin"] == 1 and d["ahead_of_origin"] == 0


def test_real_repo_current_after_ff(repo: Path):
    """The other direction, on the same real repo: fast-forward to origin and
    the verdict flips to `current`."""
    (repo / "f.txt").write_text("2")
    _git(repo, "commit", "-aqm", "two")
    head = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", env_mod.ORIGIN_MAIN_REF, head)

    d = deployment_freshness(head, repo_dir=repo)
    assert d["state"] == "current"


def test_real_repo_without_tracking_ref_is_unknown(repo: Path):
    """A clone that has never fetched. Reports unknown, NOT current."""
    head = _git(repo, "rev-parse", "HEAD")
    d = deployment_freshness(head, repo_dir=repo)
    assert d["origin_main"] == "unknown"
    assert d["state"] == "unknown"


def test_origin_ref_age_is_exposed_so_never_fetched_is_visible(repo: Path):
    """`in_sync` can mean 'current' or 'nobody fetched since before the gap
    opened'. Those are different claims; the age is what separates them.
    This is how the 2026-09-19 gap stayed invisible until a manual fetch."""
    head = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", env_mod.ORIGIN_MAIN_REF, head)
    age = env_mod.origin_ref_age_seconds(repo)
    assert age is not None and age >= 0


# ── The mutation guard ──────────────────────────────────────────────────

def test_origin_axis_compares_against_origin_not_against_the_tree(repo: Path):
    """MUTATION GUARD — this is the defect, written as a test.

    Point the origin side of the comparison back at the checked-out tree and
    this test goes red. That mutation is precisely the original bug: both
    operands sourced from the same clone, so the answer is 'in sync' by
    construction and the check can never fail.

    It is deliberately an INTEGRATION test against a real repo. A unit test
    that stubbed `origin_main_sha` would be bypassed by the mutation rather
    than failed by it — the stub would simply stop being called, and a green
    suite would survive the very change it is meant to catch.
    """
    first = _git(repo, "rev-parse", "HEAD")
    (repo / "f.txt").write_text("2")
    _git(repo, "commit", "-aqm", "two")
    second = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", env_mod.ORIGIN_MAIN_REF, second)
    _git(repo, "reset", "-q", "--hard", first)

    d = deployment_freshness(first, repo_dir=repo)

    # The two operands must be DIFFERENT values read from DIFFERENT refs.
    assert d["tree_head"] == first
    assert d["origin_main"] == second
    assert d["tree_head"] != d["origin_main"], (
        "tree_head and origin_main are equal on a clone that is demonstrably "
        "behind origin — the origin side is being read from the tree"
    )
    assert d["state"] == "behind_origin", (
        f"expected behind_origin, got {d['state']!r}: the origin comparison "
        "is not reaching refs/remotes/origin/main"
    )


# ── Doc guard: the README claim, asserted against the code ──────────────

def test_readme_claim_names_both_axes_and_the_unknown_answer():
    """The README bullet is the user-facing contract for this feature. It
    previously claimed merged-but-not-restarted "can never masquerade as
    deployed" while describing only ONE axis — the claim outran the code by
    a whole failure mode. Assert the DOC against the CODE, never the reverse.
    """
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    bullet = next(
        (ln for ln in readme.splitlines() if "masquerade as deployed" in ln), None
    )
    assert bullet, "the staleness bullet is gone from the README"
    assert "origin/main" in bullet, (
        "the bullet does not mention the origin axis — it describes the "
        "pre-fix behaviour, which reported stale:false on an undeployed merge"
    )
    assert "unknown" in bullet, (
        "the bullet does not state that undeterminable freshness reads as "
        "`unknown`; that third answer is the whole point of the fix"
    )
    # Every state the code can return must be a real state, so the doc and
    # the enum cannot drift apart silently.
    assert set(env_mod._FRESHNESS_DETAIL) == {
        "current", "behind_tree", "behind_origin",
        "ahead_of_origin", "diverged", "unknown",
    }
