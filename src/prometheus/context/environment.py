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
import sys
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


# ── Is this process running from a Prometheus checkout? ──────────────────
#
# WHY THIS EXISTS. Every git read below used to run in ``parents[3]`` of this
# file and trust whatever repository git found there. On a Homebrew install
# that path is inside ``/opt/homebrew`` — Homebrew's own repository — so
# ``running_sha``, ``tree_head``, ``stale`` and the deployment block described
# Homebrew's HEAD, and ``brew update`` moving it read as "new code on disk this
# process isn't executing". A venv inside a user's own project did the same
# with that project. The origin refresh had the write-side form of the same
# bug (origin_fetch.py, #826), and both now ask this one question.

#: The distribution name a Prometheus checkout's pyproject.toml declares.
PROJECT_NAME = "oara-prometheus"

#: The file every "which checkout am I" read is anchored on: this module, inside
#: the loaded package. A module name rather than ``__file__`` inline so a test
#: can stand the process inside a fake install.
_PACKAGE_FILE = __file__


def _locate_checkout(package_file: str | Path | None) -> tuple[Path | None, bool]:
    """``(root or None, settled)``. *settled* is False only when git itself
    could not be run, so "could not tell" is never taken for "not a checkout"."""
    here = Path(package_file or __file__).resolve()
    root = here.parents[3]
    if here.parents[1] != (root / "src" / "prometheus").resolve():
        return None, True
    try:
        project = tomllib.loads(
            (root / "pyproject.toml").read_text(encoding="utf-8")).get("project") or {}
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return None, True
    if project.get("name") != PROJECT_NAME:
        return None, True
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=str(root),
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, False
    if proc.returncode != 0 or not proc.stdout.strip():
        return None, True
    if Path(proc.stdout.strip()).resolve() != root:
        return None, True
    return root, True


def source_checkout_root(package_file: str | Path | None = None) -> Path | None:
    """The Prometheus checkout this package was loaded from, or None.

    All three must hold, so that a repository which merely CONTAINS the
    installed package is never taken for one (Homebrew's ``/opt/homebrew``, a
    user's own project with a venv inside it):

    * the package is ``<root>/src/prometheus`` (the source layout);
    * ``<root>/pyproject.toml`` names :data:`PROJECT_NAME`;
    * ``git rev-parse --show-toplevel`` run from ``<root>`` is ``<root>``.

    *package_file* stands in for a module at ``prometheus/<sub>/<file>.py``
    (tests point it at a fake install). Never raises: anything unreadable is
    "not a checkout". Uncached; this process's own answer is
    :func:`own_checkout_root`.
    """
    return _locate_checkout(package_file)[0]


#: This process's answer, per anchor. Only a SETTLED answer is kept: a git that
#: could not be run is asked again next time rather than being remembered as
#: "not a checkout" — on the deploy clone that would switch the freshness
#: signal off until a restart, without a word.
_OWN_CHECKOUT: dict[str, Path | None] = {}


def _own_checkout() -> tuple[Path | None, bool]:
    key = str(_PACKAGE_FILE)
    if key in _OWN_CHECKOUT:
        return _OWN_CHECKOUT[key], True
    root, settled = _locate_checkout(_PACKAGE_FILE)
    if settled:
        _OWN_CHECKOUT[key] = root
    return root, settled


def own_checkout_root() -> Path | None:
    """The checkout the loaded package came from, or None (not one, or git
    could not be asked). Resolved once per process; may run ``git rev-parse``
    the first time."""
    return _own_checkout()[0]


def is_source_checkout() -> bool | None:
    """True / False, or None when git could not be run to tell."""
    root, settled = _own_checkout()
    if root is not None:
        return True
    return False if settled else None


def booted_from() -> dict[str, str | bool | None]:
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

    ``source_checkout`` says whether ``path`` is a Prometheus checkout at all
    (:func:`is_source_checkout`; None = git could not be asked). When it is
    false — Homebrew, pip, ``uv tool``, the Mac app — ``path`` is just three
    directories above the package, and the shas beside this block are null.
    """
    here = Path(_PACKAGE_FILE).resolve()
    pkg = here.parents[1]      # …/src/prometheus
    root = here.parents[3]     # repo root, in a checkout
    return {
        "package": str(pkg),
        "path": str(root),
        "is_git": (root / ".git").exists(),
        "source_checkout": is_source_checkout(),
    }


#: A path under any user's per-user site-packages — the substring deploy.sh's B5
#: grep used on /proc/<pid>/maps, kept identical so the check means what it did.
_USER_SITE_MARK = "/.local/lib/python"


def _maps_evidence(maps_text: str, prefix: str) -> tuple[bool, bool]:
    """(something is mapped from under *prefix*, something is mapped from a user
    site) for the text of a /proc/<pid>/maps file. The path is the sixth field
    and may contain spaces."""
    under_prefix = prefix.rstrip("/") + "/"
    from_prefix = from_user_site = False
    for line in maps_text.splitlines():
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        path = parts[5]
        from_prefix = from_prefix or path.startswith(under_prefix)
        from_user_site = from_user_site or _USER_SITE_MARK in path
    return from_prefix, from_user_site


def runtime_report() -> dict[str, Any]:
    """What this process has loaded, read from the inside: deploy.sh B5's evidence.

    WHY THIS EXISTS. deploy.sh read ``/proc/<MainPID>/maps`` to prove the
    restarted daemon runs from the new venv and maps nothing from ``~/.local``.
    The daemon is non-dumpable (security/nondumpable.py), so that file is
    root-owned to every other process — but a process may always read its OWN
    maps. This is the same evidence, read by the one process still allowed to,
    with ``pid`` so the reader can check it came from the process it meant.

    ``mapped_from_prefix`` / ``user_site_mapped`` are None where there is no
    ``/proc/self/maps`` (macOS). ``user_site_modules`` counts imported modules
    whose file is in a user site, on every platform; unlike maps it also sees
    pure-Python modules. ``dumpable`` is the kernel's answer, None off Linux.
    """
    from prometheus.security.nondumpable import is_dumpable

    prefix = os.path.realpath(sys.prefix)
    from_prefix: bool | None
    from_user_site: bool | None
    try:
        maps = Path("/proc/self/maps").read_text(encoding="utf-8", errors="replace")
        from_prefix, from_user_site = _maps_evidence(maps, prefix)
    except OSError:
        from_prefix = from_user_site = None
    user_site_modules = sum(
        1 for mod in list(sys.modules.values())
        if _USER_SITE_MARK in (getattr(mod, "__file__", None) or "")
    )
    return {
        "pid": os.getpid(),
        "prefix": prefix,
        "mapped_from_prefix": from_prefix,
        "user_site_mapped": from_user_site,
        "user_site_modules": user_site_modules,
        "dumpable": is_dumpable(),
    }


def git_head_sha(repo_dir: str | Path | None = None) -> str:
    """Return the HEAD commit SHA of the git repo containing *repo_dir*.

    Defaults to the Prometheus checkout this package was loaded from
    (:func:`own_checkout_root`) — never to a repository that merely encloses an
    installed package. ``git rev-parse`` walks up to ``.git`` regardless, so
    any in-repo path works. Returns ``"unknown"`` if git is unavailable, the
    path is not a git checkout, or (by default) this is not a Prometheus
    checkout — callers treat that as "can't determine staleness" (never stale).
    """
    if repo_dir is None:
        repo_dir = own_checkout_root()
        if repo_dir is None:
            return "unknown"
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
    fall through into the permissive branch). With no *repo_dir* it runs only
    in this package's own checkout (:func:`own_checkout_root`).
    """
    if repo_dir is None:
        repo_dir = own_checkout_root()
        if repo_dir is None:
            return None
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
    fetch_state: Any | None = None,
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
    ``not_checkout``     this install is not a Prometheus checkout (Homebrew,
                         pip, the Mac app), so neither axis applies. Both axes
                         say ``not_checkout`` too, and every sha is null.

    ``unknown`` is a THIRD ANSWER and never collapses into ``current``. That
    collapse is the entire failure class this function was written for: a
    check that cannot see a problem must say so, not report clean.
    ``not_checkout`` is not a weaker ``unknown``: it is a settled answer
    (:func:`is_source_checkout` is False), given only with no *repo_dir*.

    Precedence when several are true at once is by remedy urgency:
    ``diverged`` > ``ahead_of_origin`` > ``behind_tree`` > ``behind_origin``.
    ``behind_tree`` outranks ``behind_origin`` because a restart is the
    cheaper, faster fix and pulling first would discard the running process's
    own provenance.
    """
    if repo_dir is None and is_source_checkout() is False:
        # Any repository git would find above the package is someone else's
        # (Homebrew's, the user's own project): its shas are not ours to
        # report, and comparing them would be the bug, not the answer.
        return {
            "state": "not_checkout",
            "process_vs_tree": "not_checkout",
            "tree_vs_origin": "not_checkout",
            "running_sha": None,
            "tree_head": None,
            "origin_main": None,
            "ahead_of_origin": None,
            "behind_origin": None,
            "origin_ref_age_seconds": None,
            "origin_evidence_age_seconds": None,
            "origin_fetch": _fetch_block(fetch_state),
            "detail": _FRESHNESS_DETAIL["not_checkout"],
        }

    tree = git_head_sha(repo_dir)
    origin = origin_main_sha(repo_dir)

    if running_sha == "unknown" or tree == "unknown":
        process_vs_tree = "unknown"
    elif running_sha == tree:
        process_vs_tree = "in_sync"
    else:
        process_vs_tree = "behind"

    # ── THE STALENESS GATE ──────────────────────────────────────────────
    #
    # ⚠ `origin` above is a LOCAL CACHED COPY of the other side, refreshed
    # only by a fetch. On the deploy clone the only thing that fetched was a
    # deploy — so the cache refreshed exactly when the clone became up to
    # date, and this axis could report `in_sync` and nothing else,
    # indefinitely. Measured on production 2026-09-19: the block said
    # `current` with a merged commit unpulled; a bare fetch flipped it to
    # `behind_origin` with nothing else changed.
    #
    # Publishing `origin_ref_age_seconds` was necessary and NOT SUFFICIENT: a
    # block that labels its own evidence stale and lets the conclusion stand
    # has still asserted the conclusion. So the age is now ACTED ON.
    #
    # Evidence freshness is the FRESHER of two independent facts, because
    # each is positive evidence and neither is complete on its own:
    #   * the fetcher's own last SUCCESS — the operand that moves even when
    #     origin is quiet;
    #   * the ref's last MOVEMENT — proof a fetch happened then, including a
    #     manual one by an operator or a deploy.
    # ⚠ The ref age alone will NOT do: it reads the reflog, so it measures
    # movement rather than fetch recency, and a quiet repository would look
    # identical to a dead fetch. Verified: 766 s before a successful fetch and
    # 766 s after.
    evidence_age = _evidence_age_seconds(repo_dir, fetch_state)
    stale_after = getattr(fetch_state, "stale_after_seconds", None)
    evidence_stale = (
        stale_after is not None
        and (evidence_age is None or evidence_age > stale_after)
    )

    ahead: int | None = None
    behind: int | None = None
    if evidence_stale:
        # The cached ref may be arbitrarily old. "Cannot tell" — never the
        # healthy answer on evidence we have just called stale.
        tree_vs_origin = "unknown"
    elif tree == "unknown" or origin == "unknown":
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
        # How old the FRESHEST evidence about origin/main is — the number the
        # staleness gate actually rules on. Distinct from the ref age above,
        # which only moves when origin does.
        "origin_evidence_age_seconds": evidence_age,
        # ⚠ A FAILING REFRESH IS VISIBLE HERE, not only in a log. Silence
        # about the refresh is how this class of defect recurs: the previous
        # version failed by never running, and nothing anywhere said so.
        "origin_fetch": _fetch_block(fetch_state),
        "detail": _FRESHNESS_DETAIL[state],
    }


def _fetch_block(fetch_state: Any | None) -> dict[str, Any] | None:
    """The refresh's own report, or None when there is no refresh to report.

    ``None`` is meaningful: it says this process has no independent refresh at
    all, which is the state the whole change exists to make visible rather
    than silent.
    """
    if fetch_state is None or not hasattr(fetch_state, "block"):
        return None
    try:
        return dict(fetch_state.block())
    except Exception:  # noqa: BLE001 - a status read must not raise
        return None


def _evidence_age_seconds(
    repo_dir: str | Path | None, fetch_state: Any | None
) -> int | None:
    """Seconds since origin/main was last CONFIRMED, by any means.

    The fresher of the fetcher's last success and the ref's last movement.
    ``None`` when neither is available, which the caller treats as stale.
    """
    ages: list[int] = []
    if fetch_state is not None:
        try:
            age = fetch_state.success_age_seconds()
        except Exception:  # noqa: BLE001 - a status read must not raise
            age = None
        if age is not None:
            ages.append(int(age))
    ref_age = origin_ref_age_seconds(repo_dir)
    if ref_age is not None:
        ages.append(int(ref_age))
    return min(ages) if ages else None


#: One sentence per state, naming the REMEDY. The rollup is read by people at
#: 2am; a state name without an action is a puzzle, not a signal.
_FRESHNESS_DETAIL: dict[str, str] = {
    "current": "Running code matches this clone, and this clone matches origin/main.",
    "behind_tree": "New code is on disk that this process is not executing — restart the daemon.",
    "behind_origin": "origin/main has commits this clone has not pulled — git pull --ff-only, then restart.",
    "ahead_of_origin": "This clone has commits that are not on origin/main; the boot guard REFUSES to start here.",
    "diverged": "This clone and origin/main have forked — this is not a fast-forward gap.",
    "unknown": (
        "Freshness could NOT be determined; this is not the same as up to "
        "date. If origin_fetch is failing or disabled, the comparison "
        "against origin/main is the part that cannot be trusted."
    ),
    "not_checkout": (
        "This install is not a Prometheus source checkout (a Homebrew, pip, "
        "uv tool or app install), so there are no commits to compare; the "
        "package version (GET /) says what is running. Nothing to fix."
    ),
}
