"""The origin refresh runs only in a Prometheus checkout (context/origin_fetch.py).

THE DEFECT
----------
``fetch_origin_main`` ran ``git fetch origin +refs/heads/main:...`` in
``parents[3]`` of its own file, whatever repository git found there. On a
Homebrew install that is inside ``/opt/homebrew`` — Homebrew's own repository,
whose origin has a ``main`` branch — so every five minutes a brew-installed
daemon fetched github.com/Homebrew/brew and wrote ``refs/remotes/origin/main``
into it. A venv inside a user's own project would have had the same effect on
that project.

Real repositories, as in test_origin_fetch.py: each test builds an actual
"enclosing" repository with an actual remote, puts a fake installed package
inside it, and checks whether a ref was written. The Homebrew case also runs the
old default path as a control, so the test cannot pass against a setup in which
the hazard never existed.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path

import pytest

from prometheus.context.origin_fetch import (
    PROJECT_NAME,
    OriginFetcher,
    fetch_origin_main,
    source_checkout_root,
)

REF = "refs/remotes/origin/main"


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                          text=True, check=False)


def _has_ref(repo: Path) -> bool:
    return _git("rev-parse", "--verify", "-q", REF, cwd=repo).returncode == 0


@pytest.fixture
def upstream(tmp_path):
    """A repository with a ``main`` branch to fetch from (Homebrew's has one)."""
    up = tmp_path / "upstream"
    up.mkdir()
    _git("init", "-q", "-b", "main", cwd=up)
    (up / "f.txt").write_text("x\n")
    _git("add", "f.txt", cwd=up)
    _git("-c", "user.email=t@test", "-c", "user.name=t", "commit", "-qm", "one", cwd=up)
    return up


def _repo_with_origin(path: Path, origin: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    assert _git("init", "-q", "-b", "main", cwd=path).returncode == 0
    assert _git("remote", "add", "origin", str(origin), cwd=path).returncode == 0
    return path


def _installed_module(package_dir: Path) -> Path:
    """A fake ``prometheus/context/origin_fetch.py`` under *package_dir*."""
    module = package_dir / "prometheus" / "context" / "origin_fetch.py"
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text("# stand-in for the installed module\n")
    return module


def _pyproject(root: Path, name: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(f'[project]\nname = "{name}"\nversion = "0"\n')


# ── NOT A CHECKOUT: nothing is fetched, nothing is written ──────────────────

def test_a_homebrew_style_prefix_is_not_a_checkout_and_gets_no_ref(tmp_path, upstream):
    prefix = _repo_with_origin(tmp_path / "homebrew", upstream)
    module = _installed_module(
        prefix / "Cellar" / "oara" / "0.9.14" / "libexec" / "lib" / "python3.14"
        / "site-packages")

    assert source_checkout_root(module) is None
    fetcher = OriginFetcher(package_file=module)
    assert fetcher.fetch_once() is False
    assert not _has_ref(prefix), "the refresh wrote a ref into Homebrew's repository"
    assert fetcher.state.state() == "not_checkout"

    # Control: the OLD default (parents[3] of the module, no identity check)
    # really does fetch into the enclosing repository. Without this, the test
    # above could pass in a setup where the hazard never existed.
    ok, err = fetch_origin_main(module.parents[3])
    assert ok, err
    assert _has_ref(prefix), "control failed: the old default path did not reach the prefix repo"


def test_a_users_own_repo_with_a_venv_inside_is_not_a_checkout(tmp_path, upstream):
    project = _repo_with_origin(tmp_path / "myproject", upstream)
    _pyproject(project, "myproject")
    module = _installed_module(project / ".venv" / "lib" / "python3.12" / "site-packages")

    assert source_checkout_root(module) is None
    fetcher = OriginFetcher(package_file=module)
    assert fetcher.fetch_once() is False
    assert not _has_ref(project), "the refresh wrote a ref into the user's own repository"


def test_the_source_layout_under_another_projects_name_is_not_a_checkout(tmp_path, upstream):
    """Identity, not layout: a user's repo that vendors src/prometheus under its
    own project name is still the user's repository."""
    repo = _repo_with_origin(tmp_path / "vendoring", upstream)
    _pyproject(repo, "someone-elses-app")
    module = _installed_module(repo / "src")

    assert source_checkout_root(module) is None
    assert OriginFetcher(package_file=module).fetch_once() is False
    assert not _has_ref(repo)


def test_a_prometheus_tree_inside_another_repository_is_not_a_checkout(tmp_path, upstream):
    """Not just "a .git somewhere above": git's toplevel from the root must be
    the root itself. Here the Prometheus-shaped tree has no repository of its
    own and git finds the enclosing one."""
    outer = _repo_with_origin(tmp_path / "outer", upstream)
    inner = outer / "vendor" / "prometheus-copy"
    _pyproject(inner, PROJECT_NAME)
    module = _installed_module(inner / "src")

    assert source_checkout_root(module) is None
    assert OriginFetcher(package_file=module).fetch_once() is False
    assert not _has_ref(outer)


def test_not_a_checkout_is_skipped_silently(tmp_path, upstream, caplog):
    prefix = _repo_with_origin(tmp_path / "homebrew", upstream)
    module = _installed_module(prefix / "lib" / "python3.14" / "site-packages")
    fetcher = OriginFetcher(package_file=module)

    with caplog.at_level(logging.DEBUG, logger="prometheus.context.origin_fetch"):
        # Returns instead of looping: there is nothing to refresh.
        asyncio.run(asyncio.wait_for(fetcher.run_forever(), timeout=20))

    assert [r for r in caplog.records if r.levelno >= logging.INFO] == [], (
        "a non-checkout install logged about the refresh")
    block = fetcher.state.block()
    assert block["state"] == "not_checkout"
    assert block["last_error"] is None and block["consecutive_failures"] == 0
    assert block["detail"]
    assert not _has_ref(prefix)


# ── A CHECKOUT: still fetched ───────────────────────────────────────────────

def test_a_prometheus_checkout_is_still_fetched(tmp_path, upstream):
    root = _repo_with_origin(tmp_path / "Prometheus", upstream)
    _pyproject(root, PROJECT_NAME)
    module = _installed_module(root / "src")

    assert source_checkout_root(module) == root.resolve()
    fetcher = OriginFetcher(package_file=module)
    assert fetcher.fetch_once() is True
    assert _has_ref(root)
    assert fetcher.state.state() == "ok"


def test_a_fork_is_a_checkout(tmp_path, upstream):
    """Identity comes from the tree, not the origin URL: a fork's origin is the
    fork, and refreshing it is exactly what the axis wants."""
    fork_origin = tmp_path / "someone" / "Prometheus.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(upstream), str(fork_origin)],
                   check=True)
    root = _repo_with_origin(tmp_path / "fork", fork_origin)
    _pyproject(root, PROJECT_NAME)
    module = _installed_module(root / "src")

    assert source_checkout_root(module) == root.resolve()
    assert OriginFetcher(package_file=module).fetch_once() is True


def test_the_running_package_finds_its_own_checkout():
    """From a source checkout (CI, a worktree, a deploy clone) the real module
    resolves to the repository it sits in."""
    import prometheus.context.origin_fetch as mod

    root = Path(mod.__file__).resolve().parents[3]
    if not (root / ".git").exists() or not (root / "pyproject.toml").exists():
        pytest.skip("this test run is not from a source checkout")
    assert source_checkout_root() == root


def test_an_explicit_repo_dir_is_used_as_given(tmp_path, upstream):
    """Callers that name a repository keep working (the axis tests do)."""
    clone = _repo_with_origin(tmp_path / "plain", upstream)
    fetcher = OriginFetcher(repo_dir=clone)
    assert fetcher.fetch_once() is True
    assert _has_ref(clone)
