# Source: OpenHarness (HKUDS/OpenHarness)
# Original: src/openharness/prompts/environment.py
# License: MIT
# Modified: renamed module path to prometheus.context.environment

"""Environment detection for system prompt construction.

Gathers OS, shell, platform, working directory, date, and git info.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class EnvironmentInfo:
    """Snapshot of the current runtime environment."""

    os_name: str
    os_version: str
    platform_machine: str
    shell: str
    cwd: str
    home_dir: str
    date: str
    python_version: str
    is_git_repo: bool
    git_branch: str | None = None
    hostname: str = ""
    model_name: str = ""
    model_provider: str = ""
    extra: dict[str, str] = field(default_factory=dict)


def detect_os() -> tuple[str, str]:
    """Return (os_name, os_version) for the current platform."""
    system = platform.system()
    if system == "Linux":
        try:
            import distro  # type: ignore[import-untyped]
            return "Linux", distro.version(pretty=True) or platform.release()
        except ImportError:
            return "Linux", platform.release()
    elif system == "Darwin":
        mac_ver = platform.mac_ver()[0]
        return "macOS", mac_ver or platform.release()
    elif system == "Windows":
        win_ver = platform.version()
        return "Windows", win_ver
    return system, platform.release()


def detect_shell() -> str:
    """Detect the user's shell."""
    shell = os.environ.get("SHELL", "")
    if shell:
        return Path(shell).name

    # Fallback: check for common shells on PATH
    for candidate in ("bash", "zsh", "fish", "sh"):
        if shutil.which(candidate):
            return candidate

    return "unknown"


def detect_git_info(cwd: str) -> tuple[bool, str | None]:
    """Check if cwd is inside a git repo and return (is_git_repo, branch_name)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
        )
        is_git = result.returncode == 0 and result.stdout.strip() == "true"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, None

    if not is_git:
        return False, None

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
        )
        branch = result.stdout.strip() if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        branch = None

    return True, branch


def booted_from() -> dict[str, str | bool]:
    """Which checkout is this process actually running, on disk.

    WHY THIS EXISTS. ``stale`` compares ``running_sha`` (boot) to ``tree_head``
    (HEAD now) — and both are read from the SAME checkout, the one the loaded
    package came from. So deploying the wrong tree returns ``stale: false``:
    the two shas agree, because the update landed somewhere this process never
    reads. The guard is structurally blind to that failure while appearing to
    cover it.

    Observed 2026-08-25: ``~/Prometheus`` was pulled and the service restarted,
    ``/api/status`` reported ``stale: false``, and the daemon was still running
    week-old code — its WorkingDirectory is ``~/prometheus-deploy``, a separate
    clone of the same repo. Two importable copies existed on the box, and
    nothing on the wire said which one was live.

    ``package`` is the decisive field: it is the directory Python actually
    imported, not a working directory or a config value, so a reader can
    compare it against the tree they pushed to instead of inferring.
    """
    pkg = Path(__file__).resolve().parents[1]      # …/src/prometheus
    root = Path(__file__).resolve().parents[3]     # repo root
    return {
        "package": str(pkg),
        "path": str(root),
        "is_git": (root / ".git").exists(),
    }


def git_head_sha(repo_dir: str | Path | None = None) -> str:
    """Return the HEAD commit SHA of the git repo containing *repo_dir*.

    Defaults to the repository this package is checked out in (anchored from
    this file: ``context/environment.py`` → ``parents[3]`` = repo root). ``git
    rev-parse`` walks up to ``.git`` regardless, so any in-repo path works.
    Returns ``"unknown"`` if git is unavailable or the path is not a git
    checkout — callers treat that as "can't determine staleness" (never stale).
    """
    if repo_dir is None:
        repo_dir = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(repo_dir),
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def get_environment_info(cwd: str | None = None) -> EnvironmentInfo:
    """Gather all environment information into an EnvironmentInfo snapshot."""
    if cwd is None:
        cwd = os.getcwd()

    os_name, os_version = detect_os()
    shell = detect_shell()
    is_git, branch = detect_git_info(cwd)

    return EnvironmentInfo(
        os_name=os_name,
        os_version=os_version,
        platform_machine=platform.machine(),
        shell=shell,
        cwd=cwd,
        home_dir=str(Path.home()),
        date=datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
        python_version=platform.python_version(),
        is_git_repo=is_git,
        git_branch=branch,
        hostname=platform.node(),
    )


# ── Deployment freshness: the SECOND axis ────────────────────────────────
#
# WHY THIS EXISTS. ``git_head_sha`` answers "is the process behind the tree?"
# and that is the ONLY question ``stale`` has ever answered. It cannot see the
# failure it is most often quoted about: the tree itself being behind the
# remote. On 2026-09-19 the daemon reported ``stale: false`` while origin/main
# was two commits ahead — running code matched the tree it booted from, so the
# check returned clean and the merged-but-undeployed commits were invisible.
#
# Two axes, deliberately kept apart, because they have DIFFERENT REMEDIES:
#   * process behind tree  → restart the daemon.
#   * tree behind origin   → ``git pull --ff-only`` and then restart.
# Collapsing them into one boolean tells an operator something is wrong and
# not which of the two things to do.
#
# ⚠ NO FETCH. This runs in a request handler (``/api/status`` served 54,618
# times in 30 days), and ``scripts/deploy_guard.sh`` documents at length why
# a network call is the wrong thing to put on a safety check. We read the
# LOCAL tracking ref — the SAME operand the guard reads, so the two can never
# disagree about what "origin/main" means.
#
# ⚠ THE COST OF NOT FETCHING, AND WHY ``origin_ref_age_seconds`` EXISTS.
# The local ref is only as fresh as the last fetch, so "in sync" can mean
# "genuinely current" OR "nobody has fetched since before the gap opened".
# Those are not the same claim and this module refuses to conflate them: the
# age of the ref ships alongside the verdict so a reader can tell which one
# they are looking at. This is not hypothetical — it is exactly how the
# 2026-09-19 gap stayed invisible until someone fetched by hand.

#: The one ref name. Shared with ``scripts/deploy_guard.sh`` on purpose: a
#: status surface that disagreed with the boot guard about which ref is
#: "origin/main" would be a second source of truth, which is the defect this
#: module exists to remove.
ORIGIN_MAIN_REF = "refs/remotes/origin/main"


def _git(args: list[str], repo_dir: str | Path | None) -> str | None:
    """Run a git command in *repo_dir*. ``None`` on ANY failure.

    ``None`` means "could not determine", never "zero" and never "equal" —
    the distinction ``deploy_guard.sh`` makes with its ``'?'`` sentinel and
    for the same reason (CROSS-CUTTING §8: a detector that broke must not
    fall through into the permissive branch).
    """
    if repo_dir is None:
        repo_dir = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            cwd=str(repo_dir),
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def origin_main_sha(repo_dir: str | Path | None = None) -> str:
    """The LOCAL ``refs/remotes/origin/main`` commit, or ``"unknown"``.

    No fetch, no network (see the module note above). ``"unknown"`` covers a
    missing tracking ref, a non-repo, and a broken/absent git — all cases
    where the honest answer is "cannot determine", which callers must NOT
    render as "in sync".
    """
    return _git(["rev-parse", "--verify", "--quiet", ORIGIN_MAIN_REF], repo_dir) or "unknown"


def origin_ref_age_seconds(repo_dir: str | Path | None = None) -> int | None:
    """Seconds since the local origin/main ref last moved, or ``None``.

    Read from the ref's own reflog, so it measures when THIS clone last
    learned about origin/main — not when the remote last changed. A large
    age next to ``in_sync`` means "nobody fetched", which is the difference
    between a verdict and a guess.
    """
    ts = _git(["reflog", "show", "--date=unix", "-1", "--format=%gd", ORIGIN_MAIN_REF], repo_dir)
    if not ts:
        return None
    # `%gd` with --date=unix renders as e.g. `refs/remotes/origin/main@{1789826546}`
    start, sep, rest = ts.partition("@{")
    if not sep or not rest.endswith("}"):
        return None
    try:
        moved_at = int(rest[:-1])
    except ValueError:
        return None
    age = int(datetime.now(timezone.utc).timestamp()) - moved_at
    return age if age >= 0 else 0


def _count_commits(range_expr: str, repo_dir: str | Path | None) -> int | None:
    """``rev-list --count <range>``. ``None`` when the count is UNKNOWN."""
    out = _git(["rev-list", "--count", range_expr], repo_dir)
    if out is None:
        return None
    try:
        return int(out)
    except ValueError:
        return None


def deployment_freshness(
    running_sha: str,
    repo_dir: str | Path | None = None,
) -> dict[str, object]:
    """Both axes of "is what is running what was merged?", as one block.

    Returns ``process_vs_tree`` (``in_sync`` | ``behind`` | ``unknown``),
    ``tree_vs_origin`` (``in_sync`` | ``behind`` | ``ahead`` | ``diverged`` |
    ``unknown``) and a rollup ``state``:

    ``current``          both axes in sync — the only all-clear.
    ``behind_tree``      new code on disk this process is not executing.
    ``behind_origin``    merged commits this clone has not pulled.
    ``ahead_of_origin``  local commits not on origin (the guard REFUSES this
                         at boot; here it is reported, not enforced).
    ``diverged``         histories have forked — not a fast-forward gap.
    ``unknown``          at least one axis could not be determined.

    ``unknown`` is a THIRD ANSWER and never collapses into ``current``. That
    collapse is the entire failure class this function was written for: a
    check that cannot see a problem must say so, not report clean.

    Precedence when several are true at once is by remedy urgency:
    ``diverged`` > ``ahead_of_origin`` > ``behind_tree`` > ``behind_origin``.
    ``behind_tree`` outranks ``behind_origin`` because a restart is the
    cheaper, faster fix and pulling first would discard the running process's
    own provenance.
    """
    tree = git_head_sha(repo_dir)
    origin = origin_main_sha(repo_dir)

    if running_sha == "unknown" or tree == "unknown":
        process_vs_tree = "unknown"
    elif running_sha == tree:
        process_vs_tree = "in_sync"
    else:
        process_vs_tree = "behind"

    ahead: int | None = None
    behind: int | None = None
    if tree == "unknown" or origin == "unknown":
        tree_vs_origin = "unknown"
    elif tree == origin:
        tree_vs_origin, ahead, behind = "in_sync", 0, 0
    else:
        ahead = _count_commits(f"{ORIGIN_MAIN_REF}..HEAD", repo_dir)
        behind = _count_commits(f"HEAD..{ORIGIN_MAIN_REF}", repo_dir)
        if ahead is None or behind is None:
            # rev-list itself failed. Same ruling as the guard's '?' branch:
            # UNKNOWN is not zero and is not equal.
            tree_vs_origin = "unknown"
        elif ahead > 0 and behind > 0:
            tree_vs_origin = "diverged"
        elif ahead > 0:
            tree_vs_origin = "ahead"
        else:
            tree_vs_origin = "behind"

    if process_vs_tree == "unknown" or tree_vs_origin == "unknown":
        state = "unknown"
    elif tree_vs_origin == "diverged":
        state = "diverged"
    elif tree_vs_origin == "ahead":
        state = "ahead_of_origin"
    elif process_vs_tree == "behind":
        state = "behind_tree"
    elif tree_vs_origin == "behind":
        state = "behind_origin"
    else:
        state = "current"

    return {
        "state": state,
        "process_vs_tree": process_vs_tree,
        "tree_vs_origin": tree_vs_origin,
        "running_sha": running_sha,
        "tree_head": tree,
        "origin_main": origin,
        "ahead_of_origin": ahead,
        "behind_origin": behind,
        "origin_ref_age_seconds": origin_ref_age_seconds(repo_dir),
        "detail": _FRESHNESS_DETAIL[state],
    }


#: One sentence per state, naming the REMEDY. The rollup is read by people at
#: 2am; a state name without an action is a puzzle, not a signal.
_FRESHNESS_DETAIL: dict[str, str] = {
    "current": "Running code matches this clone, and this clone matches origin/main.",
    "behind_tree": "New code is on disk that this process is not executing — restart the daemon.",
    "behind_origin": "origin/main has commits this clone has not pulled — git pull --ff-only, then restart.",
    "ahead_of_origin": "This clone has commits that are not on origin/main; the boot guard REFUSES to start here.",
    "diverged": "This clone and origin/main have forked — this is not a fast-forward gap.",
    "unknown": "Freshness could NOT be determined; this is not the same as up to date.",
}
