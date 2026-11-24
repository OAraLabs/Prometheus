"""A conversation's pull request, through ``gh`` on the computer that has the repository.

Three small git reads, under the same floor as ``git_read`` (no new process, fixed arguments, bare environment),
find where the current branch is pushed: its remote and its name there. Then ``gh`` is asked about that branch of
that repository.

* ``gh`` is run OUTSIDE the repository: working directory ``/``, a fixed environment, named the repository
  (``--repo owner/name``, parsed here from the remote's address by a strict pattern) and the branch as arguments.
  It never reads the work folder's own git settings, and nothing the folder says reaches it except a branch name
  and a repository that passed the patterns below.
* It cannot run under the no-new-process limit (it starts the system's keychain tool to sign in), so everything it
  is given is validated here first, and if it has to be stopped, its whole process group is.
* Only ``github.com`` remotes. Any other address has no pull request state.
* No token is set or passed on: ``gh`` signs in with whatever it is signed in with. Without ``gh``, or not signed in,
  there is simply none.
* The link is built from the repository and the number, never taken from the answer.
* Answers are remembered for a minute, no more than a few questions are put to GitHub in any minute whatever is
  asked, and one at a time.

Contract: Beacon's ``git-status.md`` (``pr``), and Beacon's own lookup (``src/shared/git-pr.ts``). Source: novel code
for Prometheus, 2026-11.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from prometheus.security import git_read as gr

# ── the three git reads ──────────────────────────────────────────────────────────────────────────────────────

HEAD_ARGS: tuple[str, ...] = (*gr.SWITCHES, "rev-parse", "--symbolic-full-name", "HEAD")
BRANCHES_ARGS: tuple[str, ...] = (
    *gr.SWITCHES, "for-each-ref", "--format=%(refname)%00%(upstream:remotename)%00%(upstream:remoteref)", "refs/heads",
)
REMOTES_ARGS: tuple[str, ...] = (*gr.SWITCHES, "config", "--null", "--get-regexp", r"^remote\..*\.url$")


def upstream_of(head_ref: str, branches_out: str) -> tuple[str, str] | None:
    """Where the current branch is pushed: the remote's name and the branch's name there."""
    head = head_ref.strip()
    if not head.startswith("refs/heads/"):
        return None
    for line in branches_out.split("\n"):
        parts = line.split("\0")
        if len(parts) < 3 or parts[0] != head:
            continue
        remote, remote_ref = parts[1], parts[2]
        if not remote or not remote_ref.startswith("refs/heads/"):
            return None
        return remote, remote_ref[len("refs/heads/"):]
    return None


def remote_url(remotes_out: str, remote: str) -> str | None:
    """The address of one remote, from ``REMOTES_ARGS`` output (``remote.<name>.url``, a line break, the address, NUL)."""
    for rec in remotes_out.split("\0"):
        cut = rec.find("\n")
        if cut > 0 and rec[:cut] == f"remote.{remote}.url":
            return rec[cut + 1:]
    return None


_NAME = r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,98}[A-Za-z0-9_])?"
_FORMS = tuple(re.compile(f) for f in (
    rf"https://github\.com/({_NAME})/({_NAME}?)(?:\.git)?/?",
    rf"git@github\.com:({_NAME})/({_NAME}?)(?:\.git)?",
    rf"ssh://git@github\.com/({_NAME})/({_NAME}?)(?:\.git)?/?",
))


def github_repo(url: str | None) -> str | None:
    """``owner/name`` for a github.com remote address, or None for anything else (another host, a port, credentials
    in the address, a path with more in it, a name made of dots)."""
    u = (url or "").strip()
    for form in _FORMS:
        m = form.fullmatch(u)
        if not m:
            continue
        owner, name = m.group(1), re.sub(r"\.git$", "", m.group(2))
        if name in ("", ".", "..") or ".." in owner or ".." in name:
            return None
        return f"{owner}/{name}"
    return None


_BRANCH = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._/@+-]{0,199}")


def branch_ok(branch: object) -> bool:
    """A branch name as an argument to ``gh``: the characters branch names are made of, nothing an option could be
    mistaken for."""
    return (isinstance(branch, str) and _BRANCH.fullmatch(branch) is not None and ".." not in branch
            and "//" not in branch and not branch.endswith("/") and not branch.endswith(".lock"))


_REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")


def gh_args(repo: str | None, branch: object) -> list[str] | None:
    """The arguments for ``gh``, or None when the repository or the branch is not one to ask about."""
    if not repo or _REPO.fullmatch(repo) is None or not branch_ok(branch):
        return None
    return ["pr", "list", "--repo", repo, "--head", str(branch), "--state", "all", "--limit", "5", "--json",
            "number,state,isDraft,mergeable,url,statusCheckRollup,headRefName,isCrossRepository"]


def gh_env(home: str | None = None) -> dict[str, str]:
    """The environment ``gh`` runs in: where to find it and the keychain tool, the person's home (its own sign-in is
    theirs), no prompts, no colour. ``GH_HOST`` is github.com whatever gh would otherwise prefer. No token variable is
    ever set or passed on, and nothing of the daemon's own environment but a home and where gh keeps its settings."""
    env = {"PATH": gr.GIT_PATH, "LANG": "C", "GH_HOST": "github.com", "GH_PROMPT_DISABLED": "1",
           "GH_NO_UPDATE_NOTIFIER": "1", "NO_COLOR": "1", "GH_PAGER": "cat"}
    h = home if home is not None else os.environ.get("HOME", "")
    if h.startswith("/") and "\0" not in h:
        env["HOME"] = h
    for k in ("XDG_CONFIG_HOME", "GH_CONFIG_DIR"):
        v = os.environ.get(k, "")
        if v.startswith("/") and "\0" not in v:
            env[k] = v
    return env


#: Where ``gh`` may be: fixed places, tried in order. Never looked up through the daemon's own ``PATH``.
GH_PLACES: tuple[str, ...] = ("/opt/homebrew/bin/gh", "/usr/local/bin/gh", "/usr/bin/gh",
                              "/home/linuxbrew/.linuxbrew/bin/gh")

_FAILED = frozenset({"FAILURE", "ERROR", "TIMED_OUT", "CANCELLED", "ACTION_REQUIRED", "STARTUP_FAILURE"})
_PASSED = frozenset({"SUCCESS", "NEUTRAL", "SKIPPED"})


def checks_of(rollup: object) -> str:
    """``passing | failing | pending | none`` from a ``statusCheckRollup``. A check run says ``status`` then
    ``conclusion``; a commit status says ``state``."""
    if not isinstance(rollup, list) or not rollup:
        return "none"
    pending = False
    for c in rollup:
        if not isinstance(c, dict):
            continue
        conclusion, state, status = c.get("conclusion"), c.get("state"), c.get("status")
        verdict = conclusion if isinstance(conclusion, str) and conclusion else state if isinstance(state, str) else ""
        if verdict in _FAILED:
            return "failing"
        if verdict not in _PASSED or (isinstance(status, str) and status not in ("COMPLETED", "")):
            pending = True
    return "pending" if pending else "passing"


def parse_gh_prs(out: str | bytes, repo: str, branch: str) -> dict[str, object] | None:
    """The pull request for a branch, from ``gh pr list … --json …`` output: the open one if there is one, otherwise
    the first. None when there is none, or the answer is not what was asked for (not JSON, another branch's, from a
    fork, a number that is not one, a state it does not know)."""
    try:
        listing = json.loads(out)
    except (ValueError, TypeError):
        return None
    if not isinstance(listing, list):
        return None
    # Only this repository's own branch: ``--head`` matches by name, so a pull request from someone's fork, from a
    # branch of the same name, would otherwise be taken for this one.
    mine = [p for p in listing if isinstance(p, dict) and p.get("headRefName") == branch
            and p.get("isCrossRepository") is False and type(p.get("number")) is int and p["number"] > 0]
    pick = next((p for p in mine if p.get("state") == "OPEN"), mine[0] if mine else None)
    if pick is None:
        return None
    state = {"OPEN": "open", "MERGED": "merged", "CLOSED": "closed"}.get(str(pick.get("state")))
    if state is None:
        return None
    mergeable = {"MERGEABLE": "mergeable", "CONFLICTING": "conflicting"}.get(str(pick.get("mergeable")), "unknown")
    number = pick["number"]
    return {"number": number, "state": state, "draft": state == "open" and pick.get("isDraft") is True,
            "mergeable": mergeable if state == "open" else "unknown", "checks": checks_of(pick.get("statusCheckRollup")),
            "url": f"https://github.com/{repo}/pull/{number}"}


# ── asking ───────────────────────────────────────────────────────────────────────────────────────────────────

GH_TIME_S = 8.0
GH_OUT_MAX = 1024 * 1024
FRESH_S = 60.0
NO_GH_S = 600.0
FAILED_S = 120.0
STALLED_S = 300.0
PER_MINUTE = 6


@dataclass
class PrRead:
    """``pr`` (with the pull request), ``none``, or ``unknown`` (could not be told just now)."""

    kind: str
    pr: dict[str, object] | None = None


NONE = PrRead("none")
UNKNOWN = PrRead("unknown")

_known: dict[str, tuple[float, bool, PrRead]] = {}
_asking: dict[str, asyncio.Future[PrRead]] = {}
_asked: list[float] = []


class _Memory:
    """What was learned about gh: when it was last found absent, when it had to be stopped, where it is."""

    no_gh_since = 0.0
    stalled_at = 0.0
    place = ""


_state = _Memory()
_turn: asyncio.Lock | None = None


def reset(places: tuple[str, ...] | None = None) -> None:
    """Forget what was learned (tests), and look for ``gh`` in these places instead."""
    global GH_PLACES, _turn
    _known.clear()
    _asking.clear()
    _asked.clear()
    _state.no_gh_since = 0.0
    _state.stalled_at = 0.0
    _state.place = ""
    _turn = None
    if places is not None:
        GH_PLACES = places


def _remember(key: str, good: bool, read: PrRead) -> None:
    _known.pop(key, None)
    _known[key] = (time.monotonic(), good, read)
    while len(_known) > 300:
        _known.pop(next(iter(_known)), None)


async def _ask(repo: str, branch: str, args: list[str]) -> PrRead:
    now = time.monotonic()
    if _state.no_gh_since and now - _state.no_gh_since < NO_GH_S:
        return NONE
    if _state.stalled_at and now - _state.stalled_at < STALLED_S:
        return UNKNOWN
    while _asked and now - _asked[0] >= 60:
        _asked.pop(0)
    if len(_asked) >= PER_MINUTE:
        return UNKNOWN                       # asked again later; nothing is remembered
    _asked.append(now)
    env = gh_env()
    key = f"{repo}\n{branch}"
    places = [p for p in ([_state.place] if _state.place else [])] + [p for p in GH_PLACES if p != _state.place]
    for place in places:
        if not (os.path.isfile(place) and os.access(place, os.X_OK)):
            continue
        # Not under the process limit; its own session, so a helper it started is stopped with it.
        r = await gr.run([place, *args], "/", env, script='exec "$@"', timeout=GH_TIME_S, limit=GH_OUT_MAX)
        _state.place = place
        _state.no_gh_since = 0.0
        if r.timed_out:
            _state.stalled_at = time.monotonic()
            return UNKNOWN
        if r.code != 0 or r.too_big:
            # Not signed in, no network, no such repository for this account: not known, for a while.
            _remember(key, False, UNKNOWN)
            return UNKNOWN
        pr = parse_gh_prs(r.out, repo, branch)
        read = PrRead("pr", pr) if pr else NONE
        _remember(key, True, read)
        return read
    _state.place = ""
    _state.no_gh_since = time.monotonic()
    return NONE


async def read_pr(folder: str, top_allowed: Callable[[str], bool]) -> PrRead:
    """The pull request for the branch a folder's repository has checked out. Never raises."""
    global _turn
    try:
        opened = await gr.open_repo(folder, top_allowed)
        if isinstance(opened, gr.GitRead):
            return NONE
        head = await gr.run([opened.git, *HEAD_ARGS], folder, opened.env)
        if head.code != 0:
            return NONE
        branches = await gr.run([opened.git, *BRANCHES_ARGS], folder, opened.env)
        up = upstream_of(head.out.decode("utf-8", "replace"),
                         branches.out.decode("utf-8", "replace")) if branches.code == 0 else None
        if up is None:
            return NONE
        remotes = await gr.run([opened.git, *REMOTES_ARGS], folder, opened.env)
        url = remote_url(remotes.out.decode("utf-8", "replace"), up[0]) if remotes.code == 0 else None
        repo = github_repo(url)
        args = gh_args(repo, up[1])
        if repo is None or args is None:
            return NONE
        key = f"{repo}\n{up[1]}"
        kept = _known.get(key)
        if kept and time.monotonic() - kept[0] < (FRESH_S if kept[1] else FAILED_S):
            return kept[2]
        going = _asking.get(key)
        if going is not None:
            return await asyncio.shield(going)
        if _turn is None:
            _turn = asyncio.Lock()
        fut: asyncio.Future[PrRead] = asyncio.get_running_loop().create_future()
        _asking[key] = fut
        try:
            async with _turn:                 # one gh at a time, in the order asked
                result = await _ask(repo, up[1], args)
            fut.set_result(result)
            return result
        except BaseException as exc:
            if not fut.done():
                fut.set_exception(exc) if not isinstance(exc, asyncio.CancelledError) else fut.cancel()
            raise
        finally:
            _asking.pop(key, None)
            if fut.done() and not fut.cancelled():
                fut.exception()
    except asyncio.CancelledError:
        raise
    except Exception:
        gr.logger.warning("pull request lookup failed", exc_info=True)
        return UNKNOWN
