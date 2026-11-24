"""Reading git in a folder the assistant controls: read-only, and git cannot start a program.

A conversation's work folder is where the agent works. It can write ``.git/config``, ``.git/info/attributes``,
``.gitattributes`` and anything else in it, and plain ``git status`` in such a folder runs the programs that config
names (``core.fsmonitor``, a ``filter.<name>.clean`` or ``.process`` an attributes line selects, a submodule's own
config). The agent's own commands pass the permission checker; a program the DAEMON's git started on its behalf
would not, and would run with the daemon's rights. So this module is the one place the daemon runs git, and it
runs it like this:

* **No new process.** git runs under ``RLIMIT_NPROC`` of zero, set by a small ``sh`` that then BECOMES git
  (``exec``). ``git status`` needs no child, so a config key that names a program fails to start it, whichever key
  it is. The limit does not bind root; a self-test starts a child under the limit and the answer is
  ``unsafe_here`` (and the capability is not advertised) wherever the child starts.
* **Named switches** as well, so status succeeds instead of failing on a refused fork.
* **Fixed arguments.** Nothing the folder, the session or the request says is ever an argument; the folder is
  git's working directory and nothing else.
* **A bare environment** (a short fixed list), never the daemon's own: no ``GIT_DIR``, ``GIT_CONFIG_*``,
  ``GIT_SSH``, loader variable or ``PATH`` of the daemon reaches git.
* **git by absolute path**, found on a fixed list of system folders (on a Mac, the real git ``xcrun`` names),
  never through the daemon's ``PATH``: a ``PATH`` entry that is empty, ``.`` or ``node_modules/.bin`` would find a
  file named ``git`` the agent wrote in the folder, and becoming that program starts no new process.
* **The repository must be plainly there**, judged BEFORE ``status`` walks anything: the git folder is
  ``<top>/.git``, a linked worktree's (which names this folder back) or a submodule's (inside a repository above
  the top), the top is itself a folder the daemon would accept as a workspace, and the repository does not borrow
  another's history (a ``commondir`` file, an alternates list, links in place of its index, refs or objects).
* **A time limit (4 s) and an output limit (4 MB)** per command.

Nothing here writes, fetches or pushes, and nothing here is a tool: it is the daemon's own read, never the
agent's action. The answer holds counts and names git calls its own (branch, upstream); never a file name, a
diff, a commit message or a remote address.

Contract: Beacon's ``git-status.md`` (the shape of the answer), and Beacon's own reader (``src/main/git-reader.ts``,
``src/shared/git-state.ts``), which found the holes this closes. Source: novel code for Prometheus, 2026-11.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import signal
import stat
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ── what git is, and how it is started ───────────────────────────────────────────────────────────────────────

#: The folders git is looked for in, and the ``PATH`` it runs with. Absolute system folders only.
GIT_PATH = "/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/homebrew/bin"
_GIT_DIRS = tuple(GIT_PATH.split(":"))

#: What runs first: lower the process limit to zero, stop if it cannot, then BE the command. The command's
#: arguments are ``"$@"``, never part of the script text.
NO_FORK_SCRIPT = 'ulimit -u 0 2>/dev/null || ulimit -p 0 2>/dev/null || exit 97; exec "$@"'
#: What ``sh`` exits with when it could not lower the limit.
NO_FORK_UNSET = 97
SH = "/bin/sh"

#: Named switches. The process limit is the floor; these make status succeed under it.
SWITCHES: tuple[str, ...] = (
    "-c", "core.fsmonitor=false",
    "-c", "core.hooksPath=/dev/null",
    "-c", "core.preloadIndex=false",
    "-c", "index.threads=1",
    "-c", "core.untrackedCache=false",
    "-c", "safe.bareRepository=explicit",
    "-c", "status.submoduleSummary=false",
    "-c", "advice.statusHints=false",
    "-c", "color.ui=false",
    "--no-optional-locks", "--no-pager",
)
#: Where is the repository: inside a work tree, its top, its git folder, and the folder it shares its history in.
WHERE_ARGS: tuple[str, ...] = (
    *SWITCHES, "rev-parse", "--is-inside-work-tree", "--show-toplevel", "--absolute-git-dir",
    "--path-format=absolute", "--git-common-dir",
)
STATUS_ARGS: tuple[str, ...] = (
    *SWITCHES, "status", "--porcelain=v2", "--branch", "-z", "--untracked-files=normal",
    "--ignore-submodules=all", "--no-renames",
)

TIME_LIMIT_S = 4.0
OUTPUT_LIMIT = 4 * 1024 * 1024
MAX_ENTRIES = 5000
#: The walk of refs/ and objects/ for links is bounded too.
STORE_WALK_LIMIT = 400_000
STORE_WALK_S = 2.0
#: A second read of one folder within this long is the first one's answer.
FRESH_S = 1.5
#: At most this many git reads at once, whatever is asked.
MAX_PARALLEL = 2

# Why a repository could not be read. Fixed words the contract names.
UNSAFE_HERE = "unsafe_here"
TOO_SLOW = "too_slow"
TOO_BIG = "too_big"
NOT_READ = "not_read"


def read_env(home: str | None = None) -> dict[str, str]:
    """git's whole environment: a short fixed list. ``home`` defaults to the daemon's own, and is left out unless it
    is an absolute path (then ``GIT_CEILING_DIRECTORIES`` is it too, so discovery never climbs into it)."""
    env = {"PATH": GIT_PATH, "LANG": "C", "LC_ALL": "C", "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0",
           "GIT_PAGER": "cat"}
    h = home if home is not None else os.environ.get("HOME", "")
    if h.startswith("/") and "\0" not in h:
        env["HOME"] = h
        env["GIT_CEILING_DIRECTORIES"] = h
    return env


_git_program: str | None = None
_git_checked = False


def _find_git() -> str | None:
    """The git to run, as an absolute path. On a Mac, ``/usr/bin/git`` is a launcher that finds the real git through
    ``xcrun``, and starting that lookup is a new process the limit refuses; so the real one is asked for once, with
    no limit, at the top of the disk, in the fixed environment, and its absolute path is what runs afterwards.
    Anywhere else: the first executable ``git`` in the fixed system folders."""
    if sys.platform == "darwin":
        import subprocess

        try:
            tools = subprocess.run(["/usr/bin/xcode-select", "-p"], cwd="/", env=read_env(), capture_output=True,
                                   timeout=5, check=False)
            if tools.returncode != 0:
                return None        # no developer tools: /usr/bin/git would put up an install dialog
            found = subprocess.run(["/usr/bin/xcrun", "--find", "git"], cwd="/", env=read_env(),
                                   capture_output=True, text=True, timeout=10, check=False)
            path = found.stdout.strip() if found.returncode == 0 else ""
            if (re.fullmatch(r"/(Library/Developer/CommandLineTools|Applications/Xcode[^/]*\.app/Contents/Developer)"
                             r"/[A-Za-z0-9_./-]+/git", path) and ".." not in path and os.access(path, os.X_OK)):
                return path
        except (OSError, subprocess.SubprocessError):
            pass
        return None
    for d in _GIT_DIRS:
        candidate = os.path.join(d, "git")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def git_program() -> str | None:
    """The absolute path of git, found once; None when there is none."""
    global _git_program, _git_checked
    if not _git_checked:
        _git_program = _find_git()
        _git_checked = True
    return _git_program


# ── running a program under the limit ────────────────────────────────────────────────────────────────────────

@dataclass
class Ran:
    code: int | None = None
    out: bytes = b""
    timed_out: bool = False
    too_big: bool = False


def _kill_group(proc: asyncio.subprocess.Process) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(ProcessLookupError, OSError):
        proc.kill()


async def run(argv: list[str], cwd: str, env: dict[str, str], *, script: str | None = None,
              timeout: float | None = None, limit: int | None = None) -> Ran:
    """Run ``argv`` (absolute program first) as ``sh -c script sh argv...`` in ``cwd`` with exactly ``env``.

    The program is a fresh session leader so the whole group can be stopped; stdin is nothing; stderr is dropped.
    Over ``timeout`` or ``limit`` it is killed, group and all.
    """
    script = NO_FORK_SCRIPT if script is None else script
    timeout = TIME_LIMIT_S if timeout is None else timeout
    limit = OUTPUT_LIMIT if limit is None else limit
    try:
        proc = await asyncio.create_subprocess_exec(
            SH, "-c", script, "sh", *argv, cwd=cwd, env=env, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, start_new_session=True,
        )
    except (OSError, ValueError):
        return Ran()
    result = Ran()
    chunks: list[bytes] = []
    size = 0

    async def pump() -> None:
        nonlocal size
        assert proc.stdout is not None
        while True:
            data = await proc.stdout.read(65536)
            if not data:
                return
            size += len(data)
            if size > limit:
                result.too_big = True
                return
            chunks.append(data)

    try:
        await asyncio.wait_for(pump(), timeout)
        if not result.too_big:
            await asyncio.wait_for(proc.wait(), max(0.1, timeout))
    except asyncio.TimeoutError:
        result.timed_out = True
    except asyncio.CancelledError:
        _kill_group(proc)
        with contextlib.suppress(Exception):
            await proc.wait()
        raise
    if result.timed_out or result.too_big:
        _kill_group(proc)
        with contextlib.suppress(Exception):
            await proc.wait()
    else:
        result.code = proc.returncode
        # Anything the program left behind in its group goes with it.
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(proc.pid, signal.SIGKILL)
    result.out = b"".join(chunks)
    return result


_PROBE = [SH, "-c", f"{SH} -c 'exit 0'; echo started:$?"]
_enforced: bool | None = None
_enforced_lock: asyncio.Lock | None = None


async def fork_limit_enforced() -> bool:
    """Whether the limit really stops a program being started on this host: a pair probe. Run the same program
    once with no limit (it must start its child), then under the limit (it must not). Root, a container that does
    not count processes per user, or a limit ``sh`` cannot set: not enforced, and git is not read here. A probe that
    could not run at all is "not known" and is asked again next time."""
    global _enforced, _enforced_lock
    if _enforced is not None:
        return _enforced
    if _enforced_lock is None:
        _enforced_lock = asyncio.Lock()
    async with _enforced_lock:
        if _enforced is not None:
            return _enforced
        env = read_env()
        free = await run(_PROBE, "/", env, script='exec "$@"', timeout=10)
        if free.code != 0 or not re.search(rb"^started:0$", free.out, re.M):
            return False
        limited = await run(_PROBE, "/", env, timeout=10)
        if limited.timed_out or limited.code is None:
            return False
        _enforced = limited.code != NO_FORK_UNSET and b"started:0" not in limited.out
        return _enforced


def forget_probe() -> None:
    """Forget what the self-test found (tests)."""
    global _enforced, _enforced_lock
    _enforced = None
    _enforced_lock = None


# ── what git said ────────────────────────────────────────────────────────────────────────────────────────────

_OID = re.compile(r"[0-9a-f]{7,64}")
_AB = re.compile(r"\+(\d{1,9}) -(\d{1,9})")
_NAME_MAX = 255


@dataclass
class Counts:
    branch: str | None = None
    commit: str | None = None
    upstream: str | None = None
    ahead: int | None = None
    behind: int | None = None
    staged: int = 0
    unstaged: int = 0
    untracked: int = 0
    conflicted: int = 0
    changed: int = 0
    truncated: bool = False


def _name(raw: str) -> str | None:
    """A name git gave (branch, upstream): as written, but bounded, and never empty."""
    return raw[:_NAME_MAX] if raw else None


def parse_status(out: bytes | str) -> Counts | None:
    """``git status --porcelain=v2 --branch -z`` as counts, or None when it is not that (so garbage is never "a
    clean repository"). Records end at NUL, so a file name with a line break in it is one file. Headers count only
    before the first entry: a later record that looks like one cannot restate the branch."""
    text = out.decode("utf-8", "replace") if isinstance(out, bytes) else out
    if not isinstance(text, str) or not text.startswith("# branch."):
        return None
    c = Counts()
    in_header = True
    seen = 0
    for rec in text.split("\0"):
        if rec == "":
            continue
        if in_header and rec.startswith("# "):
            m = re.fullmatch(r"# branch\.(oid|head|upstream|ab) (.*)", rec, re.S)
            if not m:
                continue
            key, val = m.group(1), m.group(2)
            if key == "oid":
                c.commit = val[:7] if _OID.fullmatch(val) else None
            elif key == "head":
                c.branch = None if val in ("(detached)", "(unknown)") else _name(val)
            elif key == "upstream":
                c.upstream = _name(val)
            else:
                ab = _AB.fullmatch(val)
                if ab:
                    c.ahead, c.behind = int(ab.group(1)), int(ab.group(2))
            continue
        in_header = False
        seen += 1
        if seen > MAX_ENTRIES:
            c.truncated = True
            break
        kind = rec[0]
        if kind == "?":
            c.untracked += 1
            c.changed += 1
        elif kind == "u":
            c.conflicted += 1
            c.changed += 1
        elif kind == "1":
            if len(rec) < 4 or rec[1] != " ":
                return None
            x, y = rec[2], rec[3]
            if x != ".":
                c.staged += 1
            if y != ".":
                c.unstaged += 1
            c.changed += 1
        elif kind == "!":
            continue                    # ignored: not counted
        else:
            return None                 # "2" (a rename: never written with --no-renames) or anything else
    return c


def parse_where(out: bytes | str) -> tuple[str, str, str, bool] | None:
    """``WHERE_ARGS`` output: ``(top, git_dir, common, linked)``, or None when the repository is not plainly HERE.

    It must say ``true`` for being inside the work tree, give three absolute paths, and name a git folder that is
    ``<top>/.git``, a linked worktree's (``…/.git/worktrees/<name>``) or a submodule's (``<a folder above top>/
    .git/modules/…``). A repository's settings can point git at any folder (``core.worktree``, a ``.git`` file
    naming another place); then the top is not the folder that holds ``.git``, and ``status`` would walk another
    folder. Nothing is read for that.
    """
    text = out.decode("utf-8", "replace") if isinstance(out, bytes) else out
    lines = [ln for ln in text.split("\n") if ln != ""]
    if len(lines) != 4 or lines[0] != "true" or not all(ln.startswith("/") for ln in lines[1:]):
        return None
    _, top, git_dir, common = lines
    own = git_dir == ("/.git" if top == "/" else f"{top}/.git")
    worktree = re.search(r"/\.git/worktrees/[^/]+$", git_dir) is not None
    at = git_dir.find("/.git/modules/")
    submodule = at > 0 and top.startswith(f"{git_dir[:at]}/")
    if not (own or worktree or submodule):
        return None
    return top, git_dir, common, worktree


def operation_of(names: list[str]) -> str | None:
    """What is under way, from the names directly inside the git folder."""
    has = set(names).__contains__
    if has("rebase-merge") or has("rebase-apply"):
        return "rebase"
    if has("MERGE_HEAD"):
        return "merge"
    if has("CHERRY_PICK_HEAD"):
        return "cherry-pick"
    if has("REVERT_HEAD"):
        return "revert"
    if has("sequencer"):                # between two commits of a several-commit cherry-pick or revert
        return "cherry-pick"
    if has("BISECT_LOG"):
        return "bisect"
    return None


def _squash(p: str) -> str:
    out: list[str] = []
    for part in p.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if out:
                out.pop()
        else:
            out.append(part)
    return "/" + "/".join(out)


def _unprivate(p: str) -> str:
    """On a Mac /tmp, /var and /etc are also /private/tmp and so on: one folder, two spellings."""
    return re.sub(r"^/private/(tmp|var|etc)(?=/|$)", r"/\1", p)


def worktree_names_back(top: str, git_dir: str, gitdir_file: str) -> bool:
    """Whether a linked worktree's store names THIS folder back. A linked worktree is two halves: the folder's
    ``.git`` file names a store (``<main>/.git/worktrees/<name>``), and that store's ``gitdir`` file names the
    folder's ``.git``. The assistant can write the first half and point it at any repository's existing worktree
    store on this computer; it cannot write the second (that is inside the other repository)."""
    named = gitdir_file.split("\n", 1)[0].strip()
    if not named:
        return False
    full = _squash(named if named.startswith("/") else f"{git_dir}/{named}")
    return _unprivate(full) == _unprivate(_squash(f"{top}/.git"))


#: The names directly inside a git folder that, as LINKS, would have the repository read another's.
_STORE_NAMES = frozenset({"HEAD", "index", "packed-refs", "objects", "refs", "config", "commondir", "info", "shallow"})


def store_is_own(git_dir: str, common: str, info_names: list[str], top_links: list[str],
                 deep_links: list[str]) -> bool:
    """Whether a repository's history and index are its OWN: the common folder git reports is the git folder itself
    or the one a linked worktree hangs off; there is no alternates list; no store name at the top of a git folder is
    a link; nothing under refs or objects is. This is not a wall (whoever can write the folder can copy files into
    it); it keeps the read from being pointed at another repository by a line of configuration or a link."""
    prefix = f"{common}/worktrees/"
    worktree = git_dir.startswith(prefix) and "/" not in git_dir[len(prefix):]
    if common != git_dir and not worktree:
        return False
    if "alternates" in info_names or "http-alternates" in info_names:
        return False
    if any(deep_links):
        return False
    for link in top_links:
        if not link:
            continue
        parent, _, leaf = link.rpartition("/")
        if parent in (common, git_dir) and leaf in _STORE_NAMES:
            return False
    return True


class _Walk(Exception):
    """The walk of refs/objects ran past its bounds."""


def _links_under(root: str, deadline: float, budget: list[int]) -> list[str]:
    """Every symbolic link at any depth under ``root`` (not followed), found by listing, bounded."""
    found: list[str] = []
    stack = [root]
    while stack:
        here = stack.pop()
        try:
            with os.scandir(here) as it:
                for e in it:
                    budget[0] -= 1
                    if budget[0] < 0 or time.monotonic() > deadline:
                        raise _Walk
                    if e.is_symlink():
                        found.append(e.path)
                    elif e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
        except FileNotFoundError:
            continue
        except _Walk:
            raise
        except OSError:
            continue
    return found


def _store_facts(git_dir: str, common: str) -> tuple[list[str], list[str], list[str]]:
    """What ``store_is_own`` is given: the names in ``<common>/objects/info``, the links directly inside each git
    folder, and the links anywhere under refs and objects. Passive listing only: nothing is opened or followed."""
    info: list[str] = []
    with contextlib.suppress(OSError):
        info = os.listdir(f"{common}/objects/info")
    top: list[str] = []
    for d in dict.fromkeys((common, git_dir)):
        with contextlib.suppress(OSError), os.scandir(d) as it:
            top.extend(e.path for e in it if e.is_symlink())
    deadline = time.monotonic() + STORE_WALK_S
    budget = [STORE_WALK_LIMIT]
    deep: list[str] = []
    for sub in ("refs", "objects"):
        path = f"{common}/{sub}"
        if os.path.islink(path):
            deep.append(path)
        else:
            deep.extend(_links_under(path, deadline, budget))
    return info, top, deep


def _read_small(path: str, limit: int = 4096) -> str | None:
    """A small regular file's text, never blocking on a pipe or following a link; None otherwise."""
    try:
        st = os.lstat(path)
        if not stat.S_ISREG(st.st_mode):
            return None
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        return None
    try:
        return os.read(fd, limit).decode("utf-8", "replace")
    except OSError:
        return None
    finally:
        os.close(fd)


# ── reading a folder ─────────────────────────────────────────────────────────────────────────────────────────

@dataclass
class GitRead:
    """What reading a folder came to. ``kind`` is state, none or unreadable (with ``why``)."""

    kind: str
    top: str | None = None
    state: dict[str, object] = field(default_factory=dict)
    truncated: bool = False
    why: str | None = None
    #: The branch and its upstream, for the pull-request lookup; never in the answer.
    git_dir: str | None = None

    def fingerprint(self) -> tuple:
        """What a client would see change (everything but the time)."""
        return (self.kind, self.top, tuple(sorted(self.state.items(), key=lambda kv: kv[0])), self.truncated, self.why)


NONE = GitRead("none")


def _unreadable(why: str) -> GitRead:
    return GitRead("unreadable", why=why)


_sem: asyncio.Semaphore | None = None
_fresh: dict[str, tuple[float, GitRead]] = {}
_inflight: dict[str, asyncio.Future[GitRead]] = {}


def forget_reads() -> None:
    """Forget remembered reads (tests, and a changed binding)."""
    _fresh.clear()


async def _guarded(folder: str, top_allowed: Callable[[str], bool]) -> GitRead:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(MAX_PARALLEL)
    async with _sem:
        try:
            result = await _read(folder, top_allowed)
        except Exception:
            logger.warning("git read of a work folder failed", exc_info=True)
            result = _unreadable(NOT_READ)
    _fresh[folder] = (time.monotonic(), result)
    if len(_fresh) > 256:
        for k in sorted(_fresh, key=lambda key: _fresh[key][0])[:128]:
            _fresh.pop(k, None)
    return result


async def read_git(folder: str, top_allowed: Callable[[str], bool]) -> GitRead:
    """Read ``folder``'s git state. ``top_allowed`` is the daemon's own rule for a repository top (the workspace
    validator): it is asked BEFORE ``status`` runs, so a top the daemon would not accept is never walked.

    Never raises. Reads of one folder at once share one run; a second read within ``FRESH_S`` is the first one's.
    """
    if not isinstance(folder, str) or not folder.startswith("/") or "\0" in folder or "\n" in folder:
        return NONE
    hit = _fresh.get(folder)
    if hit is not None and time.monotonic() - hit[0] < FRESH_S:
        return hit[1]
    task = _inflight.get(folder)
    if task is None:
        task = asyncio.ensure_future(_guarded(folder, top_allowed))
        _inflight[folder] = task
        task.add_done_callback(lambda _t, f=folder: _inflight.pop(f, None))
    return await asyncio.shield(task)


async def _read(folder: str, top_allowed: Callable[[str], bool]) -> GitRead:
    if not await fork_limit_enforced():
        return _unreadable(UNSAFE_HERE)
    git = git_program()
    if git is None:
        return NONE
    env = read_env()
    where = await run([git, *WHERE_ARGS], folder, env)
    if where.timed_out:
        return _unreadable(TOO_SLOW)
    if where.code == NO_FORK_UNSET:
        return _unreadable(UNSAFE_HERE)
    at = parse_where(where.out) if where.code == 0 else None
    if at is None:
        return NONE
    top, git_dir, common, linked = at
    # Judged BEFORE status, the command that walks the folder: a top the daemon would not accept is never walked.
    try:
        if not top_allowed(top):
            return NONE
    except Exception:
        return NONE
    if linked:
        back = _read_small(f"{git_dir}/gitdir")
        if back is None or not worktree_names_back(top, git_dir, back):
            return NONE
    try:
        info, top_links, deep_links = await asyncio.to_thread(_store_facts, git_dir, common)
    except _Walk:
        return _unreadable(TOO_SLOW)
    if not store_is_own(git_dir, common, info, top_links, deep_links):
        return _unreadable(NOT_READ)
    st = await run([git, *STATUS_ARGS], folder, env)
    if st.timed_out:
        return _unreadable(TOO_SLOW)
    if st.too_big:
        return _unreadable(TOO_BIG)
    if st.code != 0:
        return _unreadable(NOT_READ)
    parsed = parse_status(st.out)
    if parsed is None:
        return _unreadable(NOT_READ)
    names: list[str] = []
    with contextlib.suppress(OSError):
        names = os.listdir(git_dir)
    state: dict[str, object] = {
        "branch": parsed.branch, "commit": parsed.commit, "upstream": parsed.upstream,
        "ahead": parsed.ahead if parsed.upstream else None, "behind": parsed.behind if parsed.upstream else None,
        "staged": parsed.staged, "unstaged": parsed.unstaged, "untracked": parsed.untracked,
        "conflicted": parsed.conflicted, "changed": parsed.changed, "operation": operation_of(names),
    }
    return GitRead("state", top=top, state=state, truncated=parsed.truncated, git_dir=git_dir)


def payload(read: GitRead, *, pr: dict[str, object] | None = None, read_at: float | None = None) -> dict[str, object]:
    """The route's answer (contract ``v: 1``). ``none`` and ``unreadable`` carry no state, top or pr."""
    out: dict[str, object] = {"v": 1, "kind": read.kind}
    if read.kind == "state":
        out.update({"top": read.top, "state": dict(read.state), "truncated": read.truncated, "pr": pr})
    elif read.kind == "unreadable":
        out["why"] = read.why or NOT_READ
    out["read_at"] = read_at if read_at is not None else time.time()
    return out


def acceptable_top(security_cfg: dict | None) -> Callable[[str], bool]:
    """The rule for a repository top: a folder the daemon would accept as a workspace, never the filesystem root or
    the whole home."""
    from prometheus.context.workspace import validate_workspace_path

    home = os.path.realpath(os.path.expanduser("~"))

    def ok(top: str) -> bool:
        if os.path.realpath(top) == home:
            return False
        resolved, _why = validate_workspace_path(top, security_cfg or {})
        return resolved is not None

    return ok
