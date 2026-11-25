"""A conversation's pull request (``prometheus/security/git_pr.py``).

The mapping is tested against answers captured from the real ``gh pr list --json`` (``tests/fixtures/gh-pr-list.json``),
and the lookup against real repositories with a stand-in ``gh`` that records where it was run and with what.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from prometheus.security import git_pr as pr
from prometheus.security import git_read as gr

pytestmark = pytest.mark.skipif(sys.platform == "win32" or shutil.which("git") is None, reason="needs git and a POSIX host")

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "gh-pr-list.json").read_text())
ANY = lambda top: True  # noqa: E731


def _gh_answer(out: str) -> str:
    return json.dumps(json.loads(out))


class TestWhereTheBranchIsPushed:
    BRANCHES = "\n".join([
        "refs/heads/main\0origin\0refs/heads/main",
        "refs/heads/feat/x\0origin\0refs/heads/feat/x-remote",
        "refs/heads/local-only\0\0",
        "refs/heads/odd\0origin\0refs/tags/v1",
    ])

    def test_the_remote_and_the_name_there(self):
        assert pr.upstream_of("refs/heads/feat/x\n", self.BRANCHES) == ("origin", "feat/x-remote")

    @pytest.mark.parametrize("head", ["refs/heads/local-only", "HEAD", "refs/heads/missing", "refs/heads/odd"])
    def test_none_for_no_upstream_a_detached_head_a_missing_branch_or_an_upstream_that_is_not_a_branch(self, head):
        assert pr.upstream_of(head, self.BRANCHES) is None

    def test_a_remote_s_address_is_the_one_under_exactly_its_name(self):
        out = "remote.origin.url\nhttps://github.com/acme/site.git\0remote.fork.url\ngit@github.com:someone/site.git\0"
        assert pr.remote_url(out, "origin") == "https://github.com/acme/site.git"
        assert pr.remote_url(out, "fork") == "git@github.com:someone/site.git"
        assert pr.remote_url(out, "nope") is None and pr.remote_url(out, "origi") is None


class TestWhichRepositories:
    @pytest.mark.parametrize("url,want", [
        ("https://github.com/acme/site.git", "acme/site"), ("https://github.com/acme/site", "acme/site"),
        ("https://github.com/acme/site/", "acme/site"), ("git@github.com:acme/site.git", "acme/site"),
        ("ssh://git@github.com/acme/my.repo-name_1.git", "acme/my.repo-name_1"),
    ])
    def test_a_github_com_address_is_owner_slash_name(self, url, want):
        assert pr.github_repo(url) == want

    @pytest.mark.parametrize("url", [
        "https://gitlab.com/acme/site.git", "https://github.com.evil.example/acme/site",
        "https://user:secret@github.com/acme/site.git", "https://github.com:8443/acme/site",
        "https://github.com/acme/site/extra", "https://github.com/acme", "https://github.com/acme/..",
        "https://github.com/-flag/site", "https://github.com/acme/site?x=1", "https://github.com/ac me/site",
        "/home/demo/repos/site.git", "file:///home/demo/site", "git@github.com:acme/site.git --upload-pack=x",
        "https://github.com/acme/site\n--repo=evil/x", "", None,
    ])
    def test_anything_else_is_not_asked_about(self, url):
        assert pr.github_repo(url) is None

    @pytest.mark.parametrize("b", ["main", "feat/x", "release-1.2", "user@thing", "a+b", "_x"])
    def test_a_branch_name_to_ask_about(self, b):
        assert pr.branch_ok(b)

    @pytest.mark.parametrize("b", ["-x", "--repo=evil/x", "a b", "a..b", "a//b", "x/", "x.lock", "a\nb", "a;b", "a$b", "",
                                   "x" * 201, None, 7])
    def test_a_name_an_option_could_be_mistaken_for_is_not(self, b):
        assert not pr.branch_ok(b)

    def test_gh_is_asked_about_that_branch_of_that_repository_by_name_and_nothing_else(self):
        assert pr.gh_args("acme/site", "feat/x") == [
            "pr", "list", "--repo", "acme/site", "--head", "feat/x", "--state", "all", "--limit", "5", "--json",
            "number,state,isDraft,mergeable,url,statusCheckRollup,headRefName,isCrossRepository"]
        for repo, branch in ((None, "x"), ("acme/site", "-x"), ("acme", "x"), ("acme/site/extra", "x"), ("--flag/x", "x")):
            assert pr.gh_args(repo, branch) is None


class TestTheEnvironmentGhRunsIn:
    def test_a_short_fixed_list_no_token_and_the_host_is_github_com(self, monkeypatch):
        for k in ("XDG_CONFIG_HOME", "GH_CONFIG_DIR"):       # a CI runner sets some of these; this test is about the rest
            monkeypatch.delenv(k, raising=False)
        for k, v in {"GH_TOKEN": "never", "GITHUB_TOKEN": "never", "GH_HOST": "evil.example", "GH_ENTERPRISE_TOKEN": "never",
                     "GIT_DIR": "/evil", "PATH": ".:/evil", "GH_CONFIG_DIR": "relative/evil"}.items():
            monkeypatch.setenv(k, v)
        env = pr.gh_env("/home/demo")
        assert sorted(env) == sorted(["GH_HOST", "GH_NO_UPDATE_NOTIFIER", "GH_PAGER", "GH_PROMPT_DISABLED", "HOME", "LANG",
                                      "NO_COLOR", "PATH"])
        assert env["GH_HOST"] == "github.com" and ".:" not in env["PATH"] and all(d.startswith("/") for d in env["PATH"].split(":"))

    def test_where_the_person_keeps_gh_s_settings_if_they_moved_them_is_passed_on_when_absolute(self, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", "/home/demo/.config-elsewhere")
        monkeypatch.setenv("GH_CONFIG_DIR", "/home/demo/gh")
        env = pr.gh_env("/home/demo")
        assert env["XDG_CONFIG_HOME"] == "/home/demo/.config-elsewhere" and env["GH_CONFIG_DIR"] == "/home/demo/gh"

    def test_gh_is_looked_for_in_fixed_places_only(self):
        assert all(p.startswith("/") and p.endswith("/gh") for p in pr.GH_PLACES)


class TestTheMapping:
    """Answers captured from the real gh (tests/fixtures/gh-pr-list.json)."""

    def test_a_merged_pull_request(self):
        got = pr.parse_gh_prs(json.dumps(FIXTURES["merged"]), "OAraLabs/Prometheus", "fix/parity-goldens-memory-approval")
        assert got is not None
        assert got["number"] == 941 and got["state"] == "merged" and got["draft"] is False
        assert got["checks"] == "passing" and got["mergeable"] == "unknown"
        assert got["url"] == "https://github.com/OAraLabs/Prometheus/pull/941"
        assert sorted(got) == ["checks", "draft", "mergeable", "number", "state", "url"]

    def test_a_draft_pull_request_is_open_and_a_draft(self):
        got = pr.parse_gh_prs(json.dumps(FIXTURES["draft"]), "OAraLabs/Prometheus", "docs/https-over-tailscale")
        assert got is not None and got["state"] == "open" and got["draft"] is True and got["number"] == 929

    def test_no_pull_request_and_an_answer_that_is_not_a_list_are_none(self):
        assert pr.parse_gh_prs(json.dumps(FIXTURES["none"]), "acme/site", "x") is None
        for bad in ("", '{"number":1}', "not json", "null"):
            assert pr.parse_gh_prs(bad, "acme/site", "x") is None

    def _one(self, **over) -> str:
        return json.dumps([{"number": 7, "state": "OPEN", "isDraft": False, "mergeable": "MERGEABLE", "headRefName": "feat/x",
                            "isCrossRepository": False, "url": "https://evil.example/phish", "statusCheckRollup": [], **over}])

    def test_the_link_is_built_from_the_repository_and_the_number_never_from_the_answer(self):
        assert pr.parse_gh_prs(self._one(), "acme/site", "feat/x")["url"] == "https://github.com/acme/site/pull/7"

    @pytest.mark.parametrize("over", [
        {"headRefName": "other"}, {"number": "7"}, {"number": -1}, {"number": 0}, {"number": 1e21}, {"number": 7.5},
        {"number": True}, {"state": "WEIRD"}, {"isCrossRepository": True},
    ])
    def test_an_answer_that_is_not_this_branch_s_is_none(self, over):
        assert pr.parse_gh_prs(self._one(**over), "acme/site", "feat/x") is None

    def test_a_pull_request_that_does_not_say_whether_it_is_from_a_fork_is_not_this_branch_s(self):
        listing = json.loads(self._one())
        del listing[0]["isCrossRepository"]
        assert pr.parse_gh_prs(json.dumps(listing), "acme/site", "feat/x") is None

    def test_states_and_mergeable(self):
        assert pr.parse_gh_prs(self._one(state="CLOSED"), "a/b", "feat/x")["state"] == "closed"
        conflicting = pr.parse_gh_prs(self._one(mergeable="CONFLICTING"), "a/b", "feat/x")
        assert conflicting["mergeable"] == "conflicting" and conflicting["state"] == "open"
        assert pr.parse_gh_prs(self._one(mergeable="UNKNOWN"), "a/b", "feat/x")["mergeable"] == "unknown"
        # a finished one has nothing left to merge
        assert pr.parse_gh_prs(self._one(state="MERGED", mergeable="CONFLICTING"), "a/b", "feat/x")["mergeable"] == "unknown"

    def test_with_several_for_one_branch_the_open_one_is_shown(self):
        two = json.dumps([{"number": 3, "state": "CLOSED", "headRefName": "feat/x", "isCrossRepository": False},
                          {"number": 9, "state": "OPEN", "headRefName": "feat/x", "isDraft": False, "isCrossRepository": False}])
        assert pr.parse_gh_prs(two, "a/b", "feat/x")["number"] == 9

    @pytest.mark.parametrize("rollup,want", [
        ([], "none"), (None, "none"),
        ([{"status": "COMPLETED", "conclusion": "SUCCESS"}, {"status": "COMPLETED", "conclusion": "SKIPPED"}], "passing"),
        ([{"status": "COMPLETED", "conclusion": "SUCCESS"}, {"status": "IN_PROGRESS", "conclusion": ""}], "pending"),
        ([{"status": "COMPLETED", "conclusion": "FAILURE"}, {"status": "IN_PROGRESS", "conclusion": ""}], "failing"),
        ([{"state": "SUCCESS"}], "passing"), ([{"state": "PENDING"}], "pending"), ([{"state": "FAILURE"}], "failing"),
        ([{"status": "COMPLETED", "conclusion": "CANCELLED"}], "failing"),
    ])
    def test_checks(self, rollup, want):
        assert pr.checks_of(rollup) == want


# ── asking, with real repositories and a stand-in gh ─────────────────────────────────────────────────────────

def _g(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=Smoke", "-c", "user.email=smoke@example.invalid", "-c", "init.defaultBranch=main",
         "-c", "commit.gpgsign=false", *args], cwd=cwd, capture_output=True, text=True, timeout=30, check=True,
        env={**os.environ, "LANG": "C", "LC_ALL": "C"},
    ).stdout


@pytest.fixture
def world(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for k in list(os.environ):
        if k.startswith("GIT_") and k != "GIT_CONFIG_NOSYSTEM":
            monkeypatch.delenv(k, raising=False)
    gr.forget_reads()
    gr.forget_probe()
    import asyncio

    if not asyncio.run(gr.fork_limit_enforced()):
        pytest.skip("the process limit is not enforced for this user")
    gr.forget_probe()
    pr.reset(())
    yield tmp_path
    pr.reset(("/opt/homebrew/bin/gh", "/usr/local/bin/gh", "/usr/bin/gh", "/home/linuxbrew/.linuxbrew/bin/gh"))


def _repo(root: Path, name: str, *, remote: str | None, branch: str = "feat/git-panel", upstream: bool = True) -> Path:
    d = root / name
    d.mkdir()
    _g(d, "init", "-q", ".")
    (d / "a.txt").write_text("one\n")
    _g(d, "add", "-A")
    _g(d, "commit", "-q", "-m", "first")
    _g(d, "checkout", "-q", "-b", branch)
    if remote is not None:
        _g(d, "remote", "add", "origin", remote)
        if upstream:
            _g(d, "config", f"branch.{branch}.remote", "origin")
            _g(d, "config", f"branch.{branch}.merge", f"refs/heads/{branch}")
    return d


def _fake_gh(root: Path, answer: dict | list | str, *, code: int = 0, sleep: float = 0.0, helper: bool = False) -> tuple[str, Path]:
    """A stand-in gh: records where it ran, with what arguments and which environment, then gives *answer*."""
    log = root / "gh.log"
    payload = root / "gh-answer.json"
    payload.write_text(answer if isinstance(answer, str) else json.dumps(answer))
    script = root / "gh"
    body = f'''#!/bin/sh
echo "cwd=$(pwd) args=$*" >> "{log}"
env | sort | grep -E '^(GIT_|GH_|GITHUB_|HOME)' | tr '\\n' ' ' >> "{log}"; echo >> "{log}"
'''
    if helper:
        body += f'(sleep 7321; echo helper-survived >> "{root / "helper.mark"}") &\n'
    if sleep:
        body += f"sleep {sleep}\n"
    body += f'cat "{payload}"\nexit {code}\n'
    script.write_text(body)
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script), log


def calls(log: Path) -> int:
    return log.read_text().count("cwd=") if log.exists() else 0


class TestAskingGh:
    async def test_a_branch_pushed_to_a_github_remote_reads_its_pull_request_from_gh_s_answer(self, world):
        d = _repo(world, "repo", remote="https://github.com/OAraLabs/Prometheus.git")
        place, log = _fake_gh(world, FIXTURES["merged"])
        pr.reset((place,))
        # the stand-in answers the captured real answer, for the branch the fixture is about
        _g(d, "checkout", "-q", "-b", "fix/parity-goldens-memory-approval")
        _g(d, "config", "branch.fix/parity-goldens-memory-approval.remote", "origin")
        _g(d, "config", "branch.fix/parity-goldens-memory-approval.merge", "refs/heads/fix/parity-goldens-memory-approval")
        got = await pr.read_pr(str(d), ANY)
        assert got.kind == "pr" and got.pr["number"] == 941 and got.pr["state"] == "merged"
        assert got.pr["url"] == "https://github.com/OAraLabs/Prometheus/pull/941"

    async def test_gh_is_run_at_the_top_of_the_disk_never_in_the_repository_and_told_the_repository_by_name(self, world):
        d = _repo(world, "repo", remote="https://github.com/acme/site.git")
        place, log = _fake_gh(world, [])
        pr.reset((place,))
        await pr.read_pr(str(d), ANY)
        text = log.read_text()
        assert "cwd=/ " in text and str(d) not in text.split("args=")[0]
        assert "args=pr list --repo acme/site --head feat/git-panel --state all --limit 5 --json " in text

    async def test_with_no_git_or_token_variable_in_gh_s_environment_and_the_host_fixed(self, world, monkeypatch):
        monkeypatch.setenv("GH_TOKEN", "never-pass-this")
        monkeypatch.setenv("GITHUB_TOKEN", "never-pass-this")
        monkeypatch.setenv("GH_HOST", "evil.example")
        d = _repo(world, "repo", remote="https://github.com/acme/site.git")
        monkeypatch.setenv("GIT_DIR", "/evil")      # after the fixture's own git calls
        place, log = _fake_gh(world, [])
        pr.reset((place,))
        await pr.read_pr(str(d), ANY)
        env_line = log.read_text().splitlines()[1]
        assert "never-pass-this" not in env_line and "GIT_" not in env_line and "GITHUB_TOKEN" not in env_line
        assert "GH_HOST=github.com" in env_line and "GH_TOKEN" not in env_line

    async def test_asked_again_within_the_minute_github_is_not_asked_again(self, world):
        d = _repo(world, "repo", remote="https://github.com/acme/site.git")
        place, log = _fake_gh(world, FIXTURES["none"])
        pr.reset((place,))
        await pr.read_pr(str(d), ANY)
        await pr.read_pr(str(d), ANY)
        assert calls(log) == 1

    async def test_nothing_to_ask_about(self, world):
        place, log = _fake_gh(world, FIXTURES["merged"])
        pr.reset((place,))
        no_up = _repo(world, "no-upstream", remote="https://github.com/acme/site.git", upstream=False)
        no_remote = _repo(world, "no-remote", remote=None)
        assert (await pr.read_pr(str(no_up), ANY)).kind == "none"
        assert (await pr.read_pr(str(no_remote), ANY)).kind == "none"
        assert (await pr.read_pr(str(world), ANY)).kind == "none"
        assert calls(log) == 0

    async def test_a_remote_that_is_not_github_com_asks_gh_nothing(self, world):
        place, log = _fake_gh(world, FIXTURES["merged"])
        pr.reset((place,))
        d = _repo(world, "lab", remote="https://gitlab.com/acme/site.git")
        assert (await pr.read_pr(str(d), ANY)).kind == "none" and calls(log) == 0

    async def test_a_remote_address_with_anything_after_the_repository_in_it_is_not_asked_about(self, world):
        place, log = _fake_gh(world, FIXTURES["merged"])
        pr.reset((place,))
        for i, url in enumerate(("https://github.com/acme/site/--repo=evil/x", "https://github.com/acme/site?x=1",
                                 "https://user:secret@github.com/acme/site.git")):
            d = _repo(world, f"sly{i}", remote=url)
            assert (await pr.read_pr(str(d), ANY)).kind == "none"
        assert calls(log) == 0

    async def test_the_daemon_s_rule_for_repository_tops_holds_here_too(self, world):
        place, log = _fake_gh(world, FIXTURES["merged"])
        pr.reset((place,))
        d = _repo(world, "repo", remote="https://github.com/acme/site.git")
        assert (await pr.read_pr(str(d), lambda top: False)).kind == "none" and calls(log) == 0

    async def test_no_more_than_a_few_questions_a_minute_whatever_is_asked(self, world):
        place, log = _fake_gh(world, [])
        pr.reset((place,))
        d = _repo(world, "repo", remote="https://github.com/acme/site.git", branch="b0")
        kinds = []
        for i in range(9):
            _g(d, "checkout", "-q", "-B", f"b{i}")
            _g(d, "config", f"branch.b{i}.remote", "origin")
            _g(d, "config", f"branch.b{i}.merge", f"refs/heads/b{i}")
            gr.forget_reads()
            kinds.append((await pr.read_pr(str(d), ANY)).kind)
        assert calls(log) == pr.PER_MINUTE and kinds[: pr.PER_MINUTE] == ["none"] * pr.PER_MINUTE
        assert kinds[pr.PER_MINUTE:] == ["unknown"] * (9 - pr.PER_MINUTE)

    async def test_with_no_gh_on_this_computer_there_is_simply_no_pull_request_state_and_it_is_quick(self, world):
        pr.reset((str(world / "nowhere" / "gh"),))
        d = _repo(world, "repo", remote="https://github.com/acme/site.git")
        t0 = time.monotonic()
        assert (await pr.read_pr(str(d), ANY)).kind == "none"
        assert (await pr.read_pr(str(d), ANY)).kind == "none" and time.monotonic() - t0 < 4

    async def test_a_gh_that_is_not_signed_in_is_not_known_just_now_not_no_pull_request(self, world):
        place, _ = _fake_gh(world, "", code=4)
        pr.reset((place,))
        d = _repo(world, "repo", remote="https://github.com/acme/site.git")
        assert (await pr.read_pr(str(d), ANY)).kind == "unknown"

    async def test_a_gh_that_never_answers_is_stopped_with_its_helper_and_not_asked_again_for_a_while(self, world, monkeypatch):
        place, log = _fake_gh(world, [], sleep=60, helper=True)
        pr.reset((place,))
        monkeypatch.setattr(pr, "GH_TIME_S", 1.0)
        d = _repo(world, "repo", remote="https://github.com/acme/site.git")
        t0 = time.monotonic()
        assert (await pr.read_pr(str(d), ANY)).kind == "unknown"
        assert 0.9 < time.monotonic() - t0 < 6
        n = calls(log)
        again = _repo(world, "again", remote="https://github.com/acme/other.git")
        assert (await pr.read_pr(str(again), ANY)).kind == "unknown" and calls(log) == n
        # the helper the stand-in started ("sleep 7321") went with it
        for _ in range(30):                       # a killed process takes a moment to be gone
            if subprocess.run(["pgrep", "-f", "sleep 7321"], capture_output=True, text=True).stdout.strip() == "":
                break
            time.sleep(0.1)
        assert subprocess.run(["pgrep", "-f", "sleep 7321"], capture_output=True, text=True).stdout.strip() == ""
        assert not (world / "helper.mark").exists()
