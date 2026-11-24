"""Reading git in a folder the assistant controls (``prometheus/security/git_read.py``).

Built on REAL git and throwaway repositories, ported from the hostile-repository cases Beacon's own reader was
reviewed against. Each hostile repository names a program in its config; the program leaves a mark if it runs. For
each there is a CONTROL: plain ``git status`` in the same folder does run it. Without the control, "the mark is
absent" proves nothing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from prometheus.security import git_read as gr

pytestmark = pytest.mark.skipif(sys.platform == "win32" or shutil.which("git") is None, reason="needs git and a POSIX host")

ANY = lambda top: True  # noqa: E731 - the daemon's rule for a top, when a test is not about it


@pytest.fixture
def world(tmp_path, monkeypatch):
    """An empty home, plain C messages, no system git config, and a clean slate for the reader's memory."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for k in list(os.environ):
        if k.startswith("GIT_") and k not in ("GIT_CONFIG_NOSYSTEM",):
            monkeypatch.delenv(k, raising=False)
    gr.forget_reads()
    gr.forget_probe()
    root = tmp_path / "w"
    root.mkdir()
    return root


def _env() -> dict[str, str]:
    return {**os.environ, "LANG": "C", "LC_ALL": "C", "GIT_TERMINAL_PROMPT": "0"}


def g(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=Smoke", "-c", "user.email=smoke@example.invalid", "-c", "init.defaultBranch=main",
         "-c", "commit.gpgsign=false", *args],
        cwd=cwd, env=_env(), capture_output=True, text=True, timeout=30, check=True,
    ).stdout


def make(root: Path, name: str) -> Path:
    d = root / name
    d.mkdir()
    g(d, "init", "-q", ".")
    (d / "a.txt").write_text("one\n")
    (d / "b.txt").write_text("two\n")
    g(d, "add", "-A")
    g(d, "commit", "-q", "-m", "first")
    return d


async def read(folder: Path, allow=ANY) -> gr.GitRead:
    gr.forget_reads()
    return await gr.read_git(str(folder), allow)


# ── the parser ───────────────────────────────────────────────────────────────────────────────────────────────

def rec(*lines: str) -> str:
    return "\0".join(lines) + "\0"


HEAD = ("# branch.oid 0a0dc05b3846652a51c9c81c98470216165089cc", "# branch.head main")


def _one(xy: str, path: str = "file.txt") -> str:
    return f"1 {xy} N... 100644 100644 100644 {'a' * 40} {'b' * 40} {path}"


class TestParseStatus:
    def test_a_clean_repository_with_no_upstream(self):
        c = gr.parse_status(rec(*HEAD))
        assert (c.branch, c.commit, c.changed, c.upstream, c.ahead) == ("main", "0a0dc05", 0, None, None)

    def test_an_upstream_with_commits_ahead_and_behind(self):
        c = gr.parse_status(rec(*HEAD, "# branch.upstream origin/main", "# branch.ab +3 -2"))
        assert (c.upstream, c.ahead, c.behind) == ("origin/main", 3, 2)

    def test_an_upstream_that_is_gone_has_no_distance(self):
        c = gr.parse_status(rec(*HEAD, "# branch.upstream origin/gone"))
        assert (c.upstream, c.ahead, c.behind) == ("origin/gone", None, None)

    def test_staged_unstaged_untracked_and_conflicted_are_counted_and_a_file_in_both_places_is_one(self):
        out = rec(*HEAD, _one(".M"), _one("M."), _one("MM"), _one("A."), _one(".D"), "? new file.txt", "? dir/",
                  f"u UU N... 100644 100644 100644 100644 {'a' * 40} {'b' * 40} {'c' * 40} both.txt", "! ignored.log")
        c = gr.parse_status(out)
        assert (c.staged, c.unstaged, c.untracked, c.conflicted, c.changed) == (3, 3, 2, 1, 8)

    def test_no_commit_yet_and_a_detached_head(self):
        assert gr.parse_status(rec("# branch.oid (initial)", "# branch.head main")).commit is None
        d = gr.parse_status(rec(HEAD[0], "# branch.head (detached)"))
        assert d.branch is None and d.commit == "0a0dc05"
        assert gr.parse_status(rec(HEAD[0], "# branch.head (unknown)")).branch is None

    def test_a_file_name_with_a_line_break_is_one_file(self):
        assert gr.parse_status(rec(*HEAD, "? a\nb.txt", "? c.txt")).untracked == 2

    @pytest.mark.parametrize("bad", [
        "", "fatal: not a git repository", "On branch main\nnothing to commit", rec(*HEAD, "X weird"), rec(*HEAD, "1 M"),
        rec(*HEAD, "1"),
        rec(*HEAD, f"2 R. N... 100644 100644 100644 {'a' * 40} {'b' * 40} R100 new.txt", "old.txt"),
    ])
    def test_what_is_not_this_format_is_never_read_as_clean(self, bad):
        assert gr.parse_status(bad) is None

    def test_past_the_most_entries_the_count_stops_and_says_so(self, monkeypatch):
        monkeypatch.setattr(gr, "MAX_ENTRIES", 20)
        c = gr.parse_status(rec(*HEAD, *[f"? f{i}" for i in range(70)]))
        assert c.truncated and c.untracked == 20

    def test_a_number_or_commit_that_is_not_one_is_left_out(self):
        assert gr.parse_status(rec(*HEAD, "# branch.ab +x -1")).ahead is None
        assert gr.parse_status(rec("# branch.oid zzzz", HEAD[1])).commit is None

    def test_a_record_after_the_first_entry_that_looks_like_a_header_restates_nothing(self):
        c = gr.parse_status(rec(*HEAD, "? note.txt", "# branch.head evil", "# branch.ab +9 -9"))
        assert c is None or (c.branch == "main" and c.ahead is None)

    def test_a_long_branch_name_is_bounded(self):
        assert len(gr.parse_status(rec(HEAD[0], "# branch.head " + "x" * 5000)).branch) == 255

    @pytest.mark.parametrize("names,want", [
        (["HEAD", "MERGE_HEAD"], "merge"), (["rebase-merge", "MERGE_HEAD"], "rebase"), (["rebase-apply"], "rebase"),
        (["CHERRY_PICK_HEAD"], "cherry-pick"), (["REVERT_HEAD"], "revert"), (["BISECT_LOG"], "bisect"),
        (["sequencer", "HEAD"], "cherry-pick"), (["HEAD", "config"], None),
    ])
    def test_what_is_under_way_is_read_from_the_names_in_the_git_folder(self, names, want):
        assert gr.operation_of(names) == want


class TestParseWhere:
    def test_inside_the_work_tree_with_three_absolute_paths_and_a_common_folder(self):
        assert gr.parse_where("true\n/a/b\n/a/b/.git\n/a/b/.git\n") == ("/a/b", "/a/b/.git", "/a/b/.git", False)

    @pytest.mark.parametrize("out", [
        "", "true\n/a/b\n/a/b/.git\n", "true\nrelative\n/a/b/.git\n/a/b/.git\n", "/a/b\n/a/b/.git\n/a/b/.git\n",
        "false\n/a/b\n/a/b/.git\n/a/b/.git\n",
    ])
    def test_anything_else_is_not_a_repository_here(self, out):
        assert gr.parse_where(out) is None

    def test_a_top_that_is_not_the_folder_holding_dot_git_is_not_read(self):
        # core.worktree: the repository's settings name another folder as its work tree
        assert gr.parse_where("true\n/home/demo\n/home/demo/proj/.git\n/home/demo/proj/.git\n") is None
        assert gr.parse_where("true\n/etc\n/home/demo/proj/.git\n/home/demo/proj/.git\n") is None

    def test_a_linked_worktree_a_submodule_and_the_top_of_the_disk_are_read(self):
        wt = gr.parse_where("true\n/a/wt\n/a/main/.git/worktrees/wt\n/a/main/.git\n")
        assert wt is not None and wt[3] is True
        assert gr.parse_where("true\n/a/main/sub\n/a/main/.git/modules/sub\n/a/main/.git/modules/sub\n") is not None
        assert gr.parse_where("true\n/\n/.git\n/.git\n") is not None

    def test_a_git_folder_that_is_neither_is_not(self):
        assert gr.parse_where("true\n/a/b\n/somewhere/else\n/somewhere/else\n") is None
        assert gr.parse_where("true\n/a/b\n/a/b/.git/../../c\n/a/b/.git/../../c\n") is None

    def test_a_submodule_s_git_folder_must_be_inside_a_repository_above_the_top(self):
        assert gr.parse_where("true\n/home/demo/mine\n/victim/.git/modules/x\n/victim/.git/modules/x\n") is None
        assert gr.parse_where("true\n/a/mainly/sub\n/a/main/.git/modules/sub\n/a/main/.git/modules/sub\n") is None


class TestWorktreeAndStore:
    def test_a_real_linked_worktree_is_named_back(self):
        assert gr.worktree_names_back("/w/tree", "/main/.git/worktrees/tree", "/w/tree/.git\n")
        assert gr.worktree_names_back("/private/tmp/w", "/m/.git/worktrees/w", "/tmp/w/.git\n")
        assert gr.worktree_names_back("/m/trees/w", "/m/.git/worktrees/w", "../../../trees/w/.git\n")

    def test_a_store_that_names_another_folder_does_not(self):
        assert not gr.worktree_names_back("/home/demo/mine", "/victim/.git/worktrees/theirs", "/victim-trees/theirs/.git\n")
        assert not gr.worktree_names_back("/a", "/m/.git/worktrees/w", "")
        assert not gr.worktree_names_back("/a/b", "/m/.git/worktrees/w", "/a/.git\n")

    def test_own_stores(self):
        assert gr.store_is_own("/r/.git", "/r/.git", ["pack", ""], [""], [""])
        assert gr.store_is_own("/r/.git/worktrees/w", "/r/.git", [], [], [])
        assert gr.store_is_own("/r/.git", "/r/.git", [], ["/r/.git/hooks", "/r/.git/description"], [])

    def test_a_common_folder_somewhere_else_is_not_own(self):
        assert not gr.store_is_own("/b/.git", "/victim/.git", [], [], [])
        assert not gr.store_is_own("/b/.git/worktrees/a/b", "/b/.git", [], [], [])
        assert not gr.store_is_own("/x/worktrees/w", "/b/.git", [], [], [])

    def test_a_list_of_other_object_stores_is_not_own(self):
        assert not gr.store_is_own("/r/.git", "/r/.git", ["alternates"], [], [])
        assert not gr.store_is_own("/r/.git", "/r/.git", ["http-alternates"], [], [])

    @pytest.mark.parametrize("leaf", ["index", "HEAD", "objects", "refs", "packed-refs", "config", "commondir", "info", "shallow"])
    def test_a_store_name_that_is_a_link_is_not_own(self, leaf):
        assert not gr.store_is_own("/r/.git", "/r/.git", [], [f"/r/.git/{leaf}"], [])

    @pytest.mark.parametrize("link", ["refs/heads", "refs/heads/main", "objects/ab", "objects/ab/cdef0123", "objects/pack/pack-1.pack"])
    def test_a_link_under_refs_or_objects_at_any_depth_is_not_own(self, link):
        assert not gr.store_is_own("/r/.git", "/r/.git", [], [], [f"/r/.git/{link}"])


class TestTheCommandsAndTheEnvironment:
    def test_the_arguments_are_fixed_and_carry_the_named_switches(self):
        for key in ("core.fsmonitor=false", "core.hooksPath=/dev/null", "core.preloadIndex=false", "index.threads=1",
                    "core.untrackedCache=false", "safe.bareRepository=explicit"):
            for args in (gr.WHERE_ARGS, gr.STATUS_ARGS):
                assert key in args and args[args.index(key) - 1] == "-c"
        assert {"--no-optional-locks", "--ignore-submodules=all", "-z", "--porcelain=v2", "--no-renames"} <= set(gr.STATUS_ARGS)
        verbs = [a for a in (*gr.WHERE_ARGS, *gr.STATUS_ARGS) if not a.startswith("-") and "=" not in a]
        assert sorted(set(verbs)) == ["rev-parse", "status"]

    def test_the_script_lowers_the_limit_stops_if_it_cannot_and_then_is_the_command(self):
        import re

        assert re.fullmatch(r'ulimit -u 0 2>/dev/null \|\| ulimit -p 0 2>/dev/null \|\| exit 97; exec "\$@"', gr.NO_FORK_SCRIPT)

    def test_the_environment_is_a_short_fixed_list(self, monkeypatch):
        for k, v in {"GIT_DIR": "/evil", "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "core.fsmonitor",
                     "GIT_CONFIG_VALUE_0": "/evil", "GIT_EXEC_PATH": "/evil", "GIT_SSH": "/evil",
                     "LD_PRELOAD": "/evil.so", "DYLD_INSERT_LIBRARIES": "/evil.dylib", "EDITOR": "vi",
                     "PATH": ".:node_modules/.bin::/usr/bin"}.items():
            monkeypatch.setenv(k, v)
        env = gr.read_env("/home/demo")
        assert sorted(env) == sorted(["GIT_CEILING_DIRECTORIES", "GIT_OPTIONAL_LOCKS", "GIT_PAGER", "GIT_TERMINAL_PROMPT",
                                      "HOME", "LANG", "LC_ALL", "PATH"])
        assert env["PATH"] == gr.GIT_PATH and env["HOME"] == "/home/demo" and env["GIT_CEILING_DIRECTORIES"] == "/home/demo"

    def test_a_home_that_is_not_absolute_is_left_out(self):
        assert "HOME" not in gr.read_env("relative/home") and "GIT_CEILING_DIRECTORIES" not in gr.read_env("")

    def test_git_is_looked_for_in_fixed_absolute_system_folders(self):
        dirs = gr.GIT_PATH.split(":")
        assert all(d.startswith("/") and len(d) > 1 for d in dirs) and dirs[0] == "/usr/bin"
        assert os.path.isabs(gr.git_program() or "/")


# ── ordinary repositories ────────────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _needs_the_limit():
    """Where the process limit does not bind (root, some containers) nothing here can run: the route says so."""
    import asyncio

    gr.forget_probe()
    if not asyncio.run(gr.fork_limit_enforced()):
        pytest.skip("the process limit is not enforced for this user")
    gr.forget_probe()


class TestOrdinaryRepositories:
    async def test_clean_changed_staged_untracked(self, world):
        d = make(world, "clean")
        r = await read(d)
        assert r.kind == "state" and r.top == str(d.resolve())
        s = r.state
        assert (s["branch"], s["changed"], s["operation"], s["upstream"], s["ahead"]) == ("main", 0, None, None, None)
        assert len(str(s["commit"])) == 7
        (d / "a.txt").write_text("one, changed\n")
        (d / "new.txt").write_text("new\n")
        (d / "staged.txt").write_text("staged\n")
        g(d, "add", "staged.txt")
        (d / "sub").mkdir()
        (d / "sub" / "deep.txt").write_text("deep\n")
        s = (await read(d)).state
        assert (s["unstaged"], s["staged"], s["untracked"], s["changed"]) == (1, 1, 2, 4)
        # from a folder inside the repository it is the same repository
        inside = await read(d / "sub")
        assert inside.kind == "state" and inside.top == str(d.resolve()) and inside.state["changed"] == 4

    async def test_a_branch_and_a_detached_head(self, world):
        d = make(world, "branchy")
        g(d, "checkout", "-q", "-b", "feat/thing")
        assert (await read(d)).state["branch"] == "feat/thing"
        g(d, "checkout", "-q", "--detach")
        s = (await read(d)).state
        assert s["branch"] is None and s["commit"] is not None

    async def test_ahead_of_a_real_upstream(self, world):
        clean = make(world, "clean")
        up = world / "upstream.git"
        g(world, "clone", "-q", "--bare", str(clean), str(up))
        local = world / "local"
        g(world, "clone", "-q", str(up), str(local))
        (local / "c.txt").write_text("c\n")
        g(local, "add", "-A")
        g(local, "commit", "-q", "-m", "local one")
        g(local, "commit", "-q", "--allow-empty", "-m", "local two")
        s = (await read(local)).state
        assert (s["upstream"], s["ahead"], s["behind"]) == ("origin/main", 2, 0)

    async def test_a_merge_stopped_on_a_conflict(self, world):
        m = make(world, "merging")
        g(m, "checkout", "-q", "-b", "other")
        (m / "a.txt").write_text("theirs\n")
        g(m, "commit", "-q", "-am", "theirs")
        g(m, "checkout", "-q", "main")
        (m / "a.txt").write_text("ours\n")
        g(m, "commit", "-q", "-am", "ours")
        subprocess.run(["git", "merge", "-q", "other"], cwd=m, env=_env(), capture_output=True)  # the conflict is the point
        s = (await read(m)).state
        assert (s["operation"], s["conflicted"], s["branch"]) == ("merge", 1, "main")

    async def test_a_repository_with_no_commit_yet(self, world):
        d = world / "fresh"
        d.mkdir()
        g(d, "init", "-q", ".")
        s = (await read(d)).state
        assert s["commit"] is None and s["branch"] == "main"

    async def test_a_folder_outside_any_repository_or_gone_or_not_a_path_is_none(self, world):
        assert (await read(world)).kind == "none"
        assert (await read(world / "no-such-folder")).kind == "none"
        for bad in ("relative/path", "", f"{world}\0x", f"{world}\nx"):
            assert (await gr.read_git(bad, ANY)).kind == "none"

    async def test_reads_at_once_share_one_run_and_a_second_read_is_the_first_answer(self, world):
        d = make(world, "memory")
        gr.forget_reads()
        a, b, c = await __import__("asyncio").gather(*(gr.read_git(str(d), ANY) for _ in range(3)))
        assert a is b is c
        (d / "later.txt").write_text("x\n")
        assert await gr.read_git(str(d), ANY) is a            # within the moment: the same answer
        assert (await read(d)).state["untracked"] == 1        # forgotten: read afresh

    async def test_a_real_linked_worktree_reads(self, world):
        main = make(world, "main")
        wt = world / "tree"
        g(main, "worktree", "add", "-q", str(wt), "-b", "side")
        (wt / "extra.txt").write_text("x\n")
        r = await read(wt)
        assert r.kind == "state" and r.top == str(wt.resolve()) and r.state["branch"] == "side" and r.state["untracked"] == 1

    async def test_the_answer_holds_no_file_name_message_or_remote(self, world):
        d = make(world, "private")
        (d / "SECRET-FILE-NAME.txt").write_text("x\n")
        g(d, "commit", "-q", "--allow-empty", "-m", "SECRET-COMMIT-MESSAGE")
        g(d, "remote", "add", "origin", "https://github.com/acme/SECRET-REMOTE.git")
        text = json.dumps(gr.payload(await read(d)))
        assert not any(s in text for s in ("SECRET-FILE-NAME", "SECRET-COMMIT-MESSAGE", "SECRET-REMOTE", "acme"))

    async def test_the_answer_is_the_contract_shape(self, world):
        d = make(world, "shape")
        p = gr.payload(await read(d), read_at=1.5)
        assert sorted(p) == ["kind", "pr", "read_at", "state", "top", "truncated", "v"] and p["v"] == 1 and p["read_at"] == 1.5
        assert sorted(p["state"]) == sorted(["branch", "commit", "upstream", "ahead", "behind", "staged", "unstaged",
                                             "untracked", "conflicted", "changed", "operation"])
        none = gr.payload(gr.NONE, read_at=2.0)
        assert none == {"v": 1, "kind": "none", "read_at": 2.0}
        bad = gr.payload(gr._unreadable(gr.TOO_SLOW), read_at=3.0)
        assert bad == {"v": 1, "kind": "unreadable", "why": "too_slow", "read_at": 3.0}


# ── hostile repositories ─────────────────────────────────────────────────────────────────────────────────────

class _Hostile:
    def __init__(self, world: Path):
        self.world = world
        self.n = 0

    async def run(self, label: str, setup, *, plain_runs: bool = True, daemon_env: dict[str, str] | None = None,
                  expect: tuple[str, ...] = ("state", "unreadable", "none")) -> gr.GitRead:
        self.n += 1
        d = make(self.world, f"hostile-{self.n}")
        mark = self.world / f"mark-{self.n}"
        prog = self.world / f"prog-{self.n}.sh"
        prog.write_text(f'#!/bin/sh\necho ran >> "{mark}"\nexit 0\n')
        prog.chmod(0o755)
        setup(d, prog)
        t0 = time.monotonic()
        r = await read(d)
        took = time.monotonic() - t0
        assert not mark.exists(), f"{label}: the daemon's reading ran the program"
        assert r.kind in expect and took < 9, f"{label}: answered {r.kind} in {took:.1f}s"
        if plain_runs:
            # The control: plain git, same folder, does run it. Without this the check above proves nothing.
            subprocess.run(["git", "status", "--porcelain=v2"], cwd=d, env={**_env(), **(daemon_env or {})},
                           capture_output=True, timeout=30)
            assert mark.exists(), f"{label}: (control) plain git status did not run the program"
        return r


def _later(path: Path) -> None:
    t = time.time() + 5000
    os.utime(path, (t, t))


class TestHostileRepositories:
    async def test_core_fsmonitor(self, world):
        await _Hostile(world).run("fsmonitor", lambda d, p: g(d, "config", "core.fsmonitor", str(p)))

    async def test_a_clean_filter_chosen_by_gitattributes(self, world):
        def setup(d, p):
            (d / ".gitattributes").write_text("*.txt filter=evil\n")
            g(d, "config", "filter.evil.clean", str(p))
            (d / "a.txt").write_text("ONE\n")
            _later(d / "a.txt")

        await _Hostile(world).run("clean filter", setup)

    async def test_a_filter_named_in_info_attributes(self, world):
        def setup(d, p):
            (d / ".git" / "info").mkdir(exist_ok=True)
            (d / ".git" / "info" / "attributes").write_text("* filter=sneaky\n")
            g(d, "config", "filter.sneaky.clean", str(p))
            (d / "b.txt").write_text("TWO\n")
            _later(d / "b.txt")

        await _Hostile(world).run("info/attributes filter", setup)

    async def test_a_long_running_filter_process(self, world):
        def setup(d, p):
            (d / ".gitattributes").write_text("* filter=proc\n")
            g(d, "config", "filter.proc.process", str(p))
            (d / "a.txt").write_text("ONE\n")
            _later(d / "a.txt")

        await _Hostile(world).run("filter process", setup)

    async def test_config_included_from_another_file(self, world):
        def setup(d, p):
            (d / "extra.config").write_text(f"[core]\n\tfsmonitor = {p}\n")
            g(d, "config", "include.path", str(d / "extra.config"))

        await _Hostile(world).run("include.path", setup)

    async def test_a_program_that_never_returns(self, world):
        h = _Hostile(world)

        def setup(d, p):
            p.write_text(f'#!/bin/sh\necho ran >> "{world / "mark-1"}"\nsleep 600\n')
            g(d, "config", "core.fsmonitor", str(p))

        r = await h.run("never returns", setup, plain_runs=False)
        assert r.kind in ("state", "unreadable")

    async def test_a_file_named_git_in_the_folder_with_dot_on_the_daemons_path(self, world, monkeypatch):
        gr.git_program()                                   # found once, by absolute path, before PATH is hostile
        monkeypatch.setenv("PATH", f".:node_modules/.bin::{os.environ.get('PATH', '/usr/bin:/bin')}")

        def setup(d, p):
            for where in (d / "git", d / "node_modules" / ".bin" / "git"):
                where.parent.mkdir(parents=True, exist_ok=True)
                where.write_text(p.read_text())
                where.chmod(0o755)

        h = _Hostile(world)
        r = await h.run("a program named git", setup, plain_runs=False)
        assert r.kind == "state"
        # The control: a shell given that PATH, in that folder, does run the file named git.
        d = world / "hostile-1"
        subprocess.run(["/bin/sh", "-c", 'exec "$@"', "sh", "git", "status"], cwd=d, env={**_env(), "PATH": f".:{os.environ['PATH']}"},
                       capture_output=True, timeout=30)
        assert (world / "mark-1").exists()

    async def test_settings_smuggled_in_through_the_daemons_own_environment(self, world, monkeypatch):
        prog = world / "prog-env.sh"
        mark = world / "mark-env"
        prog.write_text(f'#!/bin/sh\necho ran >> "{mark}"\nexit 0\n')
        prog.chmod(0o755)
        smuggled = {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "core.fsmonitor", "GIT_CONFIG_VALUE_0": str(prog),
                    "GIT_SSH": str(prog), "GIT_EXEC_PATH": str(world / "nowhere")}
        d = make(world, "env")
        for k, v in smuggled.items():
            monkeypatch.setenv(k, v)
        r = await read(d)
        assert r.kind == "state" and not mark.exists()
        # The control: plain git with that environment runs it.
        subprocess.run(["git", "-c", "core.untrackedCache=false", "status", "--porcelain=v2"], cwd=d,
                       env={**os.environ, "GIT_EXEC_PATH": subprocess.run(["git", "--exec-path"], capture_output=True, text=True,
                                                                          env={k: v for k, v in os.environ.items() if k != "GIT_EXEC_PATH"}).stdout.strip()},
                       capture_output=True, timeout=30)
        assert mark.exists()

    async def test_gitdir_in_the_daemons_environment_does_not_redirect_it(self, world, monkeypatch):
        other = make(world, "other")
        (other / "OTHER-ONLY.txt").write_text("x\n")
        mine = make(world, "mine")
        monkeypatch.setenv("GIT_DIR", str(other / ".git"))
        monkeypatch.setenv("GIT_WORK_TREE", str(other))
        r = await read(mine)
        assert r.kind == "state" and r.top == str(mine.resolve()) and r.state["untracked"] == 0

    async def test_core_worktree_naming_another_folder_is_not_read(self, world):
        d = make(world, "points-elsewhere")
        other = world / "other-folder"
        other.mkdir()
        for i in range(5):
            (other / f"secret-{i}.txt").write_text("x\n")
        g(d, "config", "core.worktree", str(other))
        assert (await read(d)).kind == "none"
        # The control: plain git status there does list the other folder's files.
        assert "secret-0.txt" in g(d, "status", "--porcelain=v2")

    async def test_a_submodule_shaped_git_folder_naming_the_folder_above_is_not_read(self, world):
        outer = world / "outer"
        inner = outer / "s"
        inner.mkdir(parents=True)
        for i in range(4):
            (outer / f"private-{i}.txt").write_text("x\n")
        g(outer, "init", "-q", ".")
        (outer / ".git" / "modules").mkdir(exist_ok=True)
        clean = make(world, "clean")
        g(world, "clone", "-q", "--bare", str(clean), str(outer / ".git" / "modules" / "m"))
        g(outer / ".git" / "modules" / "m", "config", "core.bare", "false")
        g(outer / ".git" / "modules" / "m", "config", "core.worktree", str(outer))
        (inner / ".git").write_text("gitdir: ../.git/modules/m\n")
        assert (await read(inner)).kind == "none"
        assert (await read(inner, allow=lambda top: top == str(inner.resolve()))).kind == "none"

    async def test_a_folder_dressed_up_as_a_bare_repository(self, world):
        clean = make(world, "clean")
        dressed = world / "dressed-up"
        mark = world / "mark-bare"
        prog = world / "prog-bare.sh"
        prog.write_text(f'#!/bin/sh\necho ran >> "{mark}"\n')
        prog.chmod(0o755)
        g(world, "clone", "-q", "--bare", str(clean), str(dressed))
        g(dressed, "config", "core.fsmonitor", str(prog))
        g(dressed, "config", "core.bare", "false")
        g(dressed, "config", "core.worktree", str(clean))
        r = await read(dressed)
        assert r.kind == "none" and not mark.exists()
        # The control: plain git status in that folder does treat it as a repository and runs the program.
        subprocess.run(["git", "status", "--porcelain=v2"], cwd=dressed, env=_env(), capture_output=True, timeout=30)
        assert mark.exists()

    async def test_a_submodule_with_hostile_config_is_not_entered(self, world):
        sub = make(world, "subrepo")
        parent = make(world, "parent")
        g(parent, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(sub), "sm")
        g(parent, "commit", "-q", "-m", "add the submodule")
        mark = world / "mark-sub"
        prog = world / "prog-sub.sh"
        prog.write_text(f'#!/bin/sh\necho ran >> "{mark}"\nexit 0\n')
        prog.chmod(0o755)
        g(parent / "sm", "config", "core.fsmonitor", str(prog))
        (parent / "sm" / "a.txt").write_text("changed in the submodule\n")
        r = await read(parent)
        assert r.kind == "state" and not mark.exists()
        subprocess.run(["git", "status", "--porcelain=v2"], cwd=parent, env=_env(), capture_output=True, timeout=30)
        assert mark.exists()                                # the control: plain git enters it and runs the program

    async def test_a_top_the_daemon_would_not_accept_is_never_walked(self, world):
        d = make(world, "refused-top")
        asked: list[str] = []

        def refuse(top: str) -> bool:
            asked.append(top)
            return False

        assert (await read(d, allow=refuse)).kind == "none" and asked == [str(d.resolve())]

    async def test_a_rule_that_fails_is_a_refusal(self, world):
        d = make(world, "broken-rule")

        def boom(top: str) -> bool:
            raise RuntimeError("x")

        assert (await read(d, allow=boom)).kind == "none"


class TestRepositoriesThatBorrowAnothersHistory:
    """A repository can be made to read another's: a commondir file, an alternates list, links in place of its index,
    refs or objects, or a ``.git`` file naming another repository's worktree store. All refused, each with its
    control (plain git does show the other repository)."""

    def _victim(self, world: Path) -> Path:
        v = make(world, "victim")
        (v / "private.env").write_text("TOP-SECRET-OF-ANOTHER-REPOSITORY\n")
        g(v, "add", "-A")
        g(v, "commit", "-q", "-m", "the victim keeps a secret subject")
        return v

    async def test_a_commondir_file(self, world):
        v = self._victim(world)
        b = world / "borrow-commondir"
        b.mkdir()
        g(b, "init", "-q", ".")
        (b / ".git" / "commondir").write_text(f"{v / '.git'}\n")
        r = await read(b)
        assert r.kind == "unreadable" and r.why == gr.NOT_READ and "private.env" not in json.dumps(gr.payload(r))
        assert "private.env" in g(b, "status", "--porcelain=v2")          # the control

    async def test_an_alternates_list(self, world):
        v = self._victim(world)
        b = make(world, "borrow-alternates")
        (b / ".git" / "objects" / "info").mkdir(parents=True, exist_ok=True)
        (b / ".git" / "objects" / "info" / "alternates").write_text(f"{v / '.git' / 'objects'}\n")
        r = await read(b)
        assert r.kind == "unreadable" and r.why == gr.NOT_READ

    async def test_a_ref_that_is_a_link(self, world):
        v = self._victim(world)
        b = make(world, "borrow-ref")
        heads = b / ".git" / "refs" / "heads"
        (heads / "main").unlink()
        (heads / "main").symlink_to(v / ".git" / "refs" / "heads" / "main")
        r = await read(b)
        assert r.kind == "unreadable" and r.why == gr.NOT_READ

    async def test_an_index_that_is_a_link(self, world):
        v = self._victim(world)
        b = make(world, "borrow-index")
        (b / ".git" / "index").unlink()
        (b / ".git" / "index").symlink_to(v / ".git" / "index")
        r = await read(b)
        assert r.kind == "unreadable" and r.why == gr.NOT_READ

    async def test_a_dot_git_file_naming_another_repositorys_worktree_store(self, world):
        v = self._victim(world)
        wt = world / "victim-tree"
        g(v, "worktree", "add", "-q", str(wt), "-b", "theirs")
        mine = world / "mine"
        mine.mkdir()
        (mine / ".git").write_text(f"gitdir: {v / '.git' / 'worktrees' / 'victim-tree'}\n")
        r = await read(mine)
        assert r.kind == "none"
        assert "private.env" not in json.dumps(gr.payload(r))
        # The control: plain git there does read the victim's repository as this folder's.
        assert "victim-tree" in g(mine, "worktree", "list") or "private.env" in g(mine, "ls-files")


# ── the limits and the floor ─────────────────────────────────────────────────────────────────────────────────

class TestTheLimitsAndTheFloor:
    async def test_where_the_process_limit_does_not_bind_the_answer_is_unsafe_here_and_nothing_is_read(self, world, monkeypatch):
        d = make(world, "root-like")
        real_run = gr.run

        async def fake(argv, cwd, env, **kw):
            # a host where a child can still be started under the limit: both probes print started:0
            if argv[:2] == gr._PROBE[:2]:
                return gr.Ran(code=0, out=b"started:0\n")
            return await real_run(argv, cwd, env, **kw)

        monkeypatch.setattr(gr, "run", fake)
        gr.forget_probe()
        assert await gr.fork_limit_enforced() is False
        r = await read(d)
        assert r.kind == "unreadable" and r.why == gr.UNSAFE_HERE and r.state == {} and r.top is None

    async def test_where_the_limit_cannot_be_set_the_answer_is_unsafe_here(self, world, monkeypatch):
        real_run = gr.run

        async def fake(argv, cwd, env, **kw):
            if argv[:2] == gr._PROBE[:2]:
                free = kw.get("script") == 'exec "$@"'
                return gr.Ran(code=0, out=b"started:0\n") if free else gr.Ran(code=gr.NO_FORK_UNSET)
            return await real_run(argv, cwd, env, **kw)

        monkeypatch.setattr(gr, "run", fake)
        gr.forget_probe()
        assert await gr.fork_limit_enforced() is False

    async def test_the_probe_really_sees_a_child_refused_here(self, world):
        gr.forget_probe()
        assert await gr.fork_limit_enforced() is True
        free = await gr.run(gr._PROBE, "/", gr.read_env(), script='exec "$@"')
        limited = await gr.run(gr._PROBE, "/", gr.read_env())
        assert b"started:0" in free.out and b"started:0" not in limited.out

    async def test_a_command_that_does_not_answer_in_time_is_too_slow_and_is_stopped(self, world, monkeypatch):
        d = make(world, "slow")
        fake = world / "slow-git"
        fake.write_text("#!/bin/sh\nexec /bin/sleep 30\n")
        fake.chmod(0o755)
        monkeypatch.setattr(gr, "_git_program", str(fake))
        monkeypatch.setattr(gr, "_git_checked", True)
        monkeypatch.setattr(gr, "TIME_LIMIT_S", 0.7)
        t0 = time.monotonic()
        r = await read(d)
        assert r.kind == "unreadable" and r.why == gr.TOO_SLOW and time.monotonic() - t0 < 5

    async def test_more_output_than_the_limit_is_too_big(self, world, monkeypatch):
        d = make(world, "big")
        for i in range(120):
            (d / f"file-with-a-fairly-long-name-{i:04d}.txt").write_text("x\n")
        monkeypatch.setattr(gr, "OUTPUT_LIMIT", 1500)
        r = await read(d)
        assert r.kind == "unreadable" and r.why == gr.TOO_BIG

    async def test_more_entries_than_the_most_read_say_so(self, world, monkeypatch):
        d = make(world, "many")
        for i in range(60):
            (d / f"f{i}.txt").write_text("x\n")
        monkeypatch.setattr(gr, "MAX_ENTRIES", 25)
        r = await read(d)
        assert r.kind == "state" and r.truncated and r.state["untracked"] == 25

    async def test_no_git_on_this_host_is_none(self, world, monkeypatch):
        monkeypatch.setattr(gr, "_git_program", None)
        monkeypatch.setattr(gr, "_git_checked", True)
        assert (await read(make(world, "nogit"))).kind == "none"

    async def test_a_read_that_fails_unexpectedly_is_not_read_never_an_exception(self, world, monkeypatch):
        async def boom(folder, allow):
            raise RuntimeError("x")

        monkeypatch.setattr(gr, "_read", boom)
        r = await read(make(world, "boom"))
        assert r.kind == "unreadable" and r.why == gr.NOT_READ

    async def test_the_reader_runs_git_by_absolute_path_never_through_a_search(self):
        p = gr.git_program()
        assert p is None or os.path.isabs(p)
        assert not p or Path(p).name == "git"
