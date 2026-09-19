"""The origin axis needs an independent refresh AND a staleness threshold.

THE DEFECT
----------
``stale`` compared the running process to the checked-out tree — both operands
from one clone — so a clone behind ``origin/main`` reported ``stale: false``.
#518 added ``tree_vs_origin``. But ``origin_main_sha`` reads
``refs/remotes/origin/main``, a LOCAL CACHED COPY updated only by a fetch, and
on the deploy clone the only thing that fetched was a deploy. The cache
refreshed exactly when the clone became up to date, so the axis could report
``in_sync`` and nothing else, indefinitely.

Measured on production 2026-09-19: the block said ``current`` with a merged
commit unpulled; a bare fetch flipped it to ``behind_origin``.

⚠ BOTH HALVES OR NEITHER, and these tests hold that:

* a fetch that starts FAILING goes silent — the ref stops moving and
  ``in_sync`` keeps printing;
* a threshold with NO fetch is permanently ``unknown``, which trains an
  operator to ignore the third answer.

The case that matters is therefore **the degradation**: break the fetch, pass
the threshold, and confirm the block says ``unknown``. ``in_sync`` after a
working fetch proves much less.

REAL REPOSITORIES, NOT MOCKS. Every test here builds an actual git repo with
an actual remote and runs actual fetches. The defect was a relationship
between two operands' REFRESH SCHEDULES, and a mock has no schedule — mocking
the fetch is precisely how this would have been missed again.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from prometheus.context.environment import deployment_freshness, git_head_sha
from prometheus.context.origin_fetch import (
    DEFAULT_INTERVAL_SECONDS,
    DEFAULT_STALE_AFTER_SECONDS,
    FetchState,
    OriginFetcher,
    fetch_origin_main,
    fetcher_from_config,
)


def _git(*args: str, cwd: Path) -> str:
    out = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                         text=True, check=True)
    return out.stdout.strip()


@pytest.fixture
def repos(tmp_path):
    """An upstream repo and a clone of it. Real git, real refs, real fetches."""
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git("init", "-q", "-b", "main", cwd=upstream)
    _git("config", "user.email", "t@test", cwd=upstream)
    _git("config", "user.name", "t", cwd=upstream)
    (upstream / "a.txt").write_text("one\n")
    _git("add", "a.txt", cwd=upstream)
    _git("commit", "-qm", "one", cwd=upstream)

    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(upstream), str(clone)], check=True)
    _git("config", "user.email", "t@test", cwd=clone)
    _git("config", "user.name", "t", cwd=clone)

    def advance_upstream(msg: str = "two") -> str:
        (upstream / "a.txt").write_text(msg + "\n")
        _git("add", "a.txt", cwd=upstream)
        _git("commit", "-qm", msg, cwd=upstream)
        return _git("rev-parse", "HEAD", cwd=upstream)

    return {"upstream": upstream, "clone": clone, "advance": advance_upstream}


# ── THE FETCH TOUCHES ONE REF AND NOTHING ELSE ──────────────────────────────

def test_the_fetch_touches_the_ref_and_nothing_else(repos):
    """Never HEAD, never the working tree. Asserted, not trusted to a sentence."""
    clone = repos["clone"]
    head_before = _git("rev-parse", "HEAD", cwd=clone)
    tree_before = _git("status", "--porcelain", cwd=clone)
    ref_before = _git("rev-parse", "refs/remotes/origin/main", cwd=clone)

    repos["advance"]()
    ok, err = fetch_origin_main(clone)

    assert ok, f"fetch failed: {err}"
    assert _git("rev-parse", "HEAD", cwd=clone) == head_before, "the fetch moved HEAD"
    assert _git("status", "--porcelain", cwd=clone) == tree_before, (
        "the fetch changed the working tree"
    )
    assert _git("rev-parse", "refs/remotes/origin/main", cwd=clone) != ref_before, (
        "the fetch did not update the ref it exists to update"
    )


def test_the_fetch_does_not_write_fetch_head_or_tags(repos):
    clone = repos["clone"]
    fh = clone / ".git" / "FETCH_HEAD"
    before = fh.read_text() if fh.exists() else None
    repos["advance"]()
    assert fetch_origin_main(clone)[0]
    after = fh.read_text() if fh.exists() else None
    assert after == before, "the fetch wrote FETCH_HEAD"


def test_a_broken_remote_reports_the_error_rather_than_raising(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    _git("init", "-q", "-b", "main", cwd=repo)
    _git("remote", "add", "origin", str(tmp_path / "does-not-exist"), cwd=repo)
    ok, err = fetch_origin_main(repo, timeout=20)
    assert ok is False
    assert err, "a failed fetch produced no error text to show an operator"


# ── THE REFRESH IS INDEPENDENT OF DEPLOYING — THE WHOLE POINT ───────────────

def test_the_axis_sees_a_merge_WITHOUT_a_pull(repos):
    """The property the axis exists for, which had never held on the box.

    Something merged and not deployed must be visible without anyone running
    a manual fetch.
    """
    clone = repos["clone"]
    fetcher = OriginFetcher(repo_dir=clone)

    repos["advance"]()          # a commit lands on origin; nobody pulls

    fetcher.fetch_once()        # the independent refresh, not a deploy
    block = deployment_freshness(
        git_head_sha(clone), repo_dir=clone, fetch_state=fetcher.state)

    assert block["tree_vs_origin"] == "behind", (
        f"a merged-but-unpulled commit was invisible: {block['tree_vs_origin']}"
    )
    assert block["state"] == "behind_origin"
    assert block["behind_origin"] == 1


def test_without_the_refresh_the_axis_says_in_sync_while_behind(repos):
    """THE ORIGINAL DEFECT, pinned so it cannot come back.

    No fetcher and no staleness threshold: the cached ref is from clone time,
    so the block confidently reports in_sync while a commit sits unpulled.
    """
    clone = repos["clone"]
    repos["advance"]()

    block = deployment_freshness(git_head_sha(clone), repo_dir=clone,
                                 fetch_state=None)
    assert block["tree_vs_origin"] == "in_sync", (
        "the fixture no longer reproduces the defect this change fixes"
    )


# ── THE DEGRADATION: THE CASE THAT MATTERS ──────────────────────────────────

def test_a_dead_fetch_degrades_to_unknown_not_to_in_sync(repos):
    """BREAK THE FETCH, PASS THE THRESHOLD, EXPECT UNKNOWN.

    `in_sync` after a working fetch proves much less than this does: it is
    consistent both with the control working and with it being absent. This
    is the case that distinguishes them.
    """
    clone = repos["clone"]
    fetcher = OriginFetcher(repo_dir=clone, stale_after_seconds=1200)
    fetcher.fetch_once()                      # one good fetch
    assert fetcher.state.state() == "ok"

    repos["advance"]()                        # origin moves...
    # ...and the fetch dies. Point the remote at nothing.
    _git("remote", "set-url", "origin", str(clone.parent / "gone"), cwd=clone)
    assert fetcher.fetch_once() is False
    assert fetcher.state.state() == "failing"

    # Inside the threshold: still trusted, because three failed fetches are a
    # blip and a third answer that cries wolf gets ignored.
    block = deployment_freshness(git_head_sha(clone), repo_dir=clone,
                                 fetch_state=fetcher.state)
    assert block["tree_vs_origin"] != "unknown"

    # Past the threshold: the evidence is stale and the conclusion goes with it.
    fetcher.state.last_success_ts = time.time() - 1201
    block = deployment_freshness(git_head_sha(clone), repo_dir=clone,
                                 fetch_state=fetcher.state)
    assert block["tree_vs_origin"] == "unknown", (
        "a dead fetch past the threshold still reported a confident verdict "
        "against a cached ref — the exact false assurance this closes"
    )
    assert block["state"] == "unknown"


def test_no_evidence_at_all_is_unknown_not_in_sync(repos):
    """No fetch success and no ref reflog = nothing to stand on.

    ⚠ A fresh `git clone` writes NO reflog for `refs/remotes/origin/main`
    (verified), so a just-cloned repo with no fetcher has genuinely no
    evidence of when origin was last checked — even though it happens to be
    up to date. `unknown` is the honest answer, and the window is bounded:
    the fetcher's first success ends it within one interval.

    The alternative — treating "no evidence" as `in_sync` because a clone is
    usually current — is the precise mistake this whole change removes.
    """
    clone = repos["clone"]
    state = FetchState(last_success_ts=None, stale_after_seconds=1200)
    block = deployment_freshness(git_head_sha(clone), repo_dir=clone,
                                 fetch_state=state)
    assert block["origin_evidence_age_seconds"] is None
    assert block["tree_vs_origin"] == "unknown", (
        "no evidence about origin produced a confident verdict"
    )
    assert block["state"] == "unknown"


def test_ref_movement_counts_as_evidence_even_with_no_fetcher(repos):
    """A manual fetch or a deploy is real evidence; do not call it unknown.

    Ref movement cannot happen without a fetch, so it is positive proof one
    occurred. Ignoring it would make an operator who just ran `git pull` stare
    at `unknown`.
    """
    clone = repos["clone"]
    repos["advance"]()
    subprocess.run(["git", "fetch", "-q", "origin"], cwd=str(clone), check=True)

    state = FetchState(last_success_ts=None, stale_after_seconds=1200)
    block = deployment_freshness(git_head_sha(clone), repo_dir=clone,
                                 fetch_state=state)
    assert block["tree_vs_origin"] == "behind", (
        "a just-fetched ref was treated as no evidence"
    )


def test_the_ref_age_alone_would_have_been_the_wrong_operand(repos):
    """Why `origin_ref_age_seconds` is not the staleness signal.

    It reads the reflog, so it measures when the ref MOVED, not when it was
    FETCHED. Verified on production: 766 s before a successful fetch and 766 s
    after, because origin had not changed. Keying the threshold on it would
    report `unknown` whenever the repo is merely QUIET — which is most of the
    time, and would turn the third answer into noise within a day.
    """
    from prometheus.context.environment import origin_ref_age_seconds

    clone = repos["clone"]
    fetcher = OriginFetcher(repo_dir=clone)
    fetcher.fetch_once()
    age_before = origin_ref_age_seconds(clone)

    fetcher.fetch_once()          # a second successful fetch, origin unchanged
    age_after = origin_ref_age_seconds(clone)

    assert age_before == age_after, (
        "the ref age DID reset on a no-op fetch; re-derive which operand the "
        "threshold should use"
    )
    # ...while the fetcher's own record did move.
    assert fetcher.state.success_age_seconds() is not None


# ── A FAILING FETCH IS VISIBLE IN THE BLOCK, NOT ONLY IN A LOG ──────────────

def test_a_failing_fetch_appears_in_the_status_block(repos):
    clone = repos["clone"]
    fetcher = OriginFetcher(repo_dir=clone)
    _git("remote", "set-url", "origin", str(clone.parent / "gone"), cwd=clone)
    fetcher.fetch_once()

    block = deployment_freshness(git_head_sha(clone), repo_dir=clone,
                                 fetch_state=fetcher.state)
    of = block["origin_fetch"]
    assert of is not None, "the refresh is invisible in the block"
    assert of["state"] == "failing"
    assert of["consecutive_failures"] == 1
    assert of["last_error"], "a failing refresh shows no reason"
    assert "FAILING" in of["detail"] or "failing" in of["detail"].lower()


def test_a_disabled_fetch_says_so_rather_than_looking_healthy():
    state = FetchState(enabled=False)
    block = state.block()
    assert block["state"] == "disabled"
    assert "DISABLED" in block["detail"]


def test_every_fetch_state_has_a_remedy_sentence():
    from prometheus.context.origin_fetch import _FETCH_DETAIL

    for st in ("ok", "failing", "never", "disabled"):
        assert _FETCH_DETAIL.get(st), f"{st} has no sentence"


# ── THE THRESHOLD'S SHAPE ───────────────────────────────────────────────────

def test_the_threshold_is_comfortably_longer_than_the_interval():
    """A threshold at 1-2x flips on a single transient failure, and a third
    answer that cries wolf is one an operator learns to scroll past."""
    assert DEFAULT_STALE_AFTER_SECONDS >= 3 * DEFAULT_INTERVAL_SECONDS, (
        "the threshold would trip on one or two transient network failures"
    )


def test_the_threshold_is_comfortably_shorter_than_a_merge_to_deploy_cycle():
    """It bounds the window in which `in_sync` can stand on stale evidence.

    The two real merge-to-deploy cycles measured on this box were ~50 min and
    ~5 h; 20 min is ~2.5x shorter than the faster one.
    """
    assert DEFAULT_STALE_AFTER_SECONDS <= 30 * 60


# ── CONFIG ──────────────────────────────────────────────────────────────────

def test_enabled_by_default():
    """The axis is not optional-correct: without the refresh it is decorative,
    and a key whose false default silently disables a control is a shape this
    repo has already paid for."""
    assert fetcher_from_config(None).state.enabled is True
    assert fetcher_from_config({}).state.enabled is True


def test_config_can_disable_and_retune():
    f = fetcher_from_config({"deployment": {"origin_fetch": {
        "enabled": False, "interval_seconds": 60, "stale_after_seconds": 240}}})
    assert f.state.enabled is False
    assert f.state.interval_seconds == 60
    assert f.state.stale_after_seconds == 240


def test_a_disabled_fetcher_does_not_loop():
    import asyncio

    f = fetcher_from_config({"deployment": {"origin_fetch": {"enabled": False}}})
    asyncio.run(asyncio.wait_for(f.run_forever(), timeout=5))


# ── NEVER IN A REQUEST PATH ─────────────────────────────────────────────────

def test_the_status_route_never_fetches():
    """`deployment_freshness` reads; it must not reach the network.

    A refresh in a request path is a status endpoint that can hang on DNS.
    """
    import inspect

    from prometheus.context import environment

    src = inspect.getsource(environment)
    assert "fetch_origin_main" not in src, (
        "environment.py now calls the fetch — a status read must never reach "
        "the network"
    )
    assert '"fetch"' not in src and "'fetch'" not in src, (
        "environment.py appears to run a git fetch"
    )


def test_the_realistic_stale_state_is_unknown_and_would_have_been_in_sync(repos):
    """THE OUTCOME-CHECKED CASE, as a test.

    The shape that actually occurs: a clone set up by a deploy (its remote ref
    written at clone time, which leaves NO reflog entry) whose refresh has
    been failing ever since. Neither operand has moved.

    ⚠ An earlier attempt at this scenario was IMPOSSIBLE and quietly passed
    for the wrong reason: it aged the fetcher's success to 1300 s while the
    ref had moved seconds earlier. Only a fetch moves the ref, so a dead
    fetcher and a just-moved ref cannot coexist — and `min()` correctly
    reported the evidence as fresh. The lesson is about the FIXTURE: a state
    that cannot arise proves nothing about a state that can.
    """
    clone = repos["clone"]
    repos["advance"]()                        # origin moves; nobody learns

    # Without the gate, this is what the block said — false assurance.
    before = deployment_freshness(git_head_sha(clone), repo_dir=clone,
                                  fetch_state=None)
    assert before["tree_vs_origin"] == "in_sync"

    # With a refresh that has been dead past the threshold: cannot tell.
    _git("remote", "set-url", "origin", str(clone.parent / "gone"), cwd=clone)
    fetcher = OriginFetcher(repo_dir=clone, stale_after_seconds=1200)
    fetcher.fetch_once()
    fetcher.state.last_success_ts = time.time() - 1300

    after = deployment_freshness(git_head_sha(clone), repo_dir=clone,
                                 fetch_state=fetcher.state)
    assert after["tree_vs_origin"] == "unknown", (
        "a clone with no fresh evidence still reported a confident verdict"
    )
    assert after["state"] == "unknown"
    assert after["origin_fetch"]["state"] == "failing"
    assert after["origin_fetch"]["last_error"]


def test_the_daemon_starts_the_refresh_and_registers_it_for_shutdown():
    """Structural: a refresh nobody starts is the defect, restated.

    Also pins the shutdown registration — the watchdog was once the one
    long-running task missing from `tasks`, and logged "Task was destroyed
    but it is pending".
    """
    from pathlib import Path

    import prometheus.daemon as dmod

    src = Path(dmod.__file__).read_text()
    assert "fetcher_from_config(config)" in src, (
        "the daemon does not construct the origin fetcher"
    )
    assert "origin_fetcher.run_forever()" in src, (
        "the daemon constructs the fetcher but never runs it"
    )
    assert "tasks.append(origin_fetch_task)" in src, (
        "the refresh task is not registered for shutdown"
    )
    assert "origin_fetcher=origin_fetcher" in src, (
        "the fetcher never reaches the web app, so /api/status cannot see it"
    )
