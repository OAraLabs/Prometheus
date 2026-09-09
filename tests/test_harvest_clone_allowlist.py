"""SYMBIOTE harvest clones GitHub, or it does not clone.

WHAT THIS IS, stated at its real severity rather than inflated. Measured
against git 2.43, which is what runs here:

    git clone 'ext::sh -c "touch PWNED"'  -> fatal: transport 'ext' not allowed
                                             (nothing executed)
    git clone file:///path/to/repo        -> SUCCEEDS, local repo readable

So the command-execution shape was already closed by git's own default protocol
policy, and what remained was reading an arbitrary LOCAL repository. Reaching
even that means changing the ``url`` on a candidate inside the persisted scout
report — a row in the symbiote_sessions SQLite table, not a value any tool
argument carries: ``symbiote_harvest`` accepts only a candidate ``full_name``
and the coordinator resolves the URL itself.

This is therefore defence in depth, not a live prompt-injection path, and it
costs nothing: Scout is the only producer of these URLs, its GITHUB_API_BASE is
hardcoded, and the candidate's ``url`` is GitHub's own ``html_url``.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

from prometheus.symbiote.harvest import (
    ALLOWED_CLONE_HOST,
    HarvestEngine,
    _refuse_clone_url,
)

ALLOWED = [
    "https://github.com/owner/repo",
    "https://github.com/owner/repo.git",
    "https://GitHub.COM/owner/repo",          # host comparison is case-folded
    "  https://github.com/owner/repo  ",      # surrounding whitespace
]

REFUSED = [
    "file:///tmp/secret-repo",                # the one that actually worked
    "file://github.com/etc/passwd",           # host passes, SCHEME must refuse it
    'ext::sh -c "touch PWNED"',               # git already refuses; so do we
    "git@github.com:owner/repo.git",          # ssh — Scout never yields it
    "http://github.com/owner/repo",           # downgrade
    "https://github.com@evil.example/x",      # host is evil.example: userinfo trick
    "https://github.com.evil.example/x",      # host is a github.com SUBDOMAIN of evil
    "https://evil.example/github.com/x",      # name in the path, not the host
    "https://raw.githubusercontent.com/o/r",  # a real GitHub host, still not this one
    "",
]


@pytest.mark.parametrize("url", ALLOWED)
def test_scout_shaped_urls_are_allowed(url):
    assert _refuse_clone_url(url) is None, url


@pytest.mark.parametrize("url", REFUSED)
def test_everything_else_is_refused(url):
    reason = _refuse_clone_url(url)
    assert reason is not None, f"{url!r} was allowed"
    assert reason.startswith("refused:")


def test_the_two_confusable_hosts_are_refused_for_the_right_reason():
    """A bare startswith check would pass these only by accident of a trailing
    slash. Parsing makes it explicit, so the reason names the HOST."""
    for url in ("https://github.com@evil.example/x", "https://github.com.evil.example/x"):
        reason = _refuse_clone_url(url)
        assert ALLOWED_CLONE_HOST in reason and "host" in reason, (url, reason)


def test_the_scheme_check_is_load_bearing_on_its_own():
    """``file:///tmp/x`` has no host, so the HOST check already refuses it and
    the scheme check looks redundant. It is not: a file URL can carry a host,
    and ``file://github.com/etc/passwd`` satisfies the host check exactly.
    Found by mutation — removing the scheme check left every other test green.
    """
    import urllib.parse

    assert urllib.parse.urlparse("file://github.com/etc/passwd").hostname == "github.com"
    reason = _refuse_clone_url("file://github.com/etc/passwd")
    assert reason is not None and "scheme" in reason, reason


# --------------------------------------------------------------------------- #
# The refusal reaches the real clone path, before any process starts
# --------------------------------------------------------------------------- #


def test_clone_refuses_before_spawning_git(tmp_path, monkeypatch):
    """A refusal that still ran git would leave a process and a partial tree."""
    spawned: list[tuple] = []

    async def _never(*args, **kwargs):
        spawned.append(args)
        raise AssertionError("git was executed for a refused URL")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _never)

    engine = HarvestEngine(
        sandbox_root=tmp_path / "sandbox", harvest_root=tmp_path / "harvest"
    )
    with pytest.raises(RuntimeError) as exc:
        asyncio.run(engine._clone("file:///tmp/secret-repo", tmp_path / "out"))

    assert "refused" in str(exc.value)
    assert spawned == []
    assert not (tmp_path / "out").exists()


def test_an_allowed_url_still_reaches_git(tmp_path, monkeypatch):
    """Guard the guard: a check that refused everything would pass the tests
    above while breaking the feature."""
    seen: list = []

    class _Proc:
        returncode = 0

        async def communicate(self):
            return b"", b""

    async def _fake(*args, **kwargs):
        seen.append(args)
        return _Proc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake)

    engine = HarvestEngine(
        sandbox_root=tmp_path / "sandbox", harvest_root=tmp_path / "harvest"
    )
    asyncio.run(engine._clone("https://github.com/owner/repo", tmp_path / "out"))

    assert len(seen) == 1
    assert "https://github.com/owner/repo" in seen[0]


def test_harvest_reports_the_refusal_instead_of_raising(tmp_path, monkeypatch):
    """harvest() turns a clone failure into an abort reason; a refusal must
    travel that same path rather than escaping as an exception."""
    from prometheus.symbiote.license_gate import LicenseCheck, LicenseVerdict

    async def _never(*args, **kwargs):
        raise AssertionError("git was executed for a refused URL")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _never)

    engine = HarvestEngine(
        sandbox_root=tmp_path / "sandbox", harvest_root=tmp_path / "harvest"
    )
    report = asyncio.run(
        engine.harvest(
            repo_full_name="owner/repo",
            repo_url="file:///tmp/secret-repo",
            problem_statement="x",
            license_check=LicenseCheck(
                spdx_id="MIT", verdict=LicenseVerdict.ALLOW,
                source="github_api", obligations=[],
            ),
        )
    )

    assert report.aborted is True
    assert "refused" in report.abort_reason


# --------------------------------------------------------------------------- #
# The producer, and the premise about git
# --------------------------------------------------------------------------- #


def test_the_url_scout_actually_produces_is_accepted():
    """Ties the allowlist to its only producer, so a change to either is caught.

    github_search yields the candidate's ``url`` from GitHub's ``html_url``,
    which has this shape.
    """
    from prometheus.symbiote.github_search import GITHUB_API_BASE

    assert GITHUB_API_BASE == "https://api.github.com"
    assert _refuse_clone_url("https://github.com/psf/requests") is None


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_itself_still_refuses_the_ext_transport(tmp_path):
    """The premise for calling this defence in depth rather than a fix for a
    live RCE. If git's default policy ever loosens, our check becomes the only
    barrier — which is fine, but the severity note above would be stale."""
    result = subprocess.run(
        ["git", "clone", "--depth", "1", 'ext::sh -c "touch PWNED"', "out"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "not allowed" in (result.stderr or "").lower()
    assert not (tmp_path / "PWNED").exists()
