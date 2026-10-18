"""The shas on /api/status come only from a Prometheus checkout (context/environment.py).

THE DEFECT
----------
``git_head_sha``, ``booted_from`` and the deployment block's ref reads ran git in
``parents[3]`` of ``context/environment.py`` and reported whatever repository git
found there. On a Homebrew install that is inside ``/opt/homebrew`` — Homebrew's
own repository — so ``running_sha``, ``tree_head``, ``stale`` and ``deployment``
described Homebrew's HEAD, and the heartbeat read ``brew update`` moving it as
"new code on disk this process isn't executing". A venv inside a user's own
project did the same with that project. #826 fixed the write side (the origin
refresh); this is the read side, asking the same question.

Real repositories, as in test_origin_fetch_checkout_only.py: each test builds an
actual enclosing repository with commits and an origin ref, stands this process
inside a fake install there (``environment._PACKAGE_FILE``), and reads the
fields. The Homebrew cases also run the old default path as a control, so they
cannot pass in a setup where the hazard never existed.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path

import pytest

import prometheus.context.environment as env_mod
from prometheus.context.environment import (
    PROJECT_NAME,
    booted_from,
    deployment_freshness,
    git_head_sha,
    is_source_checkout,
    origin_main_sha,
    origin_ref_age_seconds,
    own_checkout_root,
)
from prometheus.context.origin_fetch import OriginFetcher

REF = env_mod.ORIGIN_MAIN_REF


def _git(*args: str, cwd: Path) -> str:
    proc = subprocess.run(
        ["git", "-c", "user.email=t@test", "-c", "user.name=t", *args],
        cwd=str(cwd), capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def _repo(path: Path) -> Path:
    """A repository with one commit and an origin/main ref at it (Homebrew's
    has both), so every read below has something real to find."""
    path.mkdir(parents=True, exist_ok=True)
    _git("init", "-q", "-b", "main", cwd=path)
    (path / "f.txt").write_text("one\n")
    _git("add", "f.txt", cwd=path)
    _git("commit", "-qm", "one", cwd=path)
    _git("update-ref", REF, _git("rev-parse", "HEAD", cwd=path), cwd=path)
    return path


def _commit(repo: Path, text: str) -> str:
    (repo / "f.txt").write_text(text)
    _git("commit", "-qam", text.strip(), cwd=repo)
    return _git("rev-parse", "HEAD", cwd=repo)


def _install_in(site: Path, monkeypatch) -> Path:
    """Stand this process inside a fake install whose package dir is
    ``site/prometheus``. Returns the stand-in ``context/environment.py``."""
    module = site / "prometheus" / "context" / "environment.py"
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text("# stand-in for the installed module\n")
    monkeypatch.setattr(env_mod, "_PACKAGE_FILE", str(module))
    return module


def _pyproject(root: Path, name: str) -> None:
    (root / "pyproject.toml").write_text(f'[project]\nname = "{name}"\nversion = "0"\n')


@pytest.fixture
def homebrew(tmp_path, monkeypatch):
    """``/opt/homebrew`` shape: a repository whose Cellar holds the package."""
    prefix = _repo(tmp_path / "homebrew")
    module = _install_in(
        prefix / "Cellar" / "oara" / "0.9.14" / "libexec" / "lib" / "python3.14"
        / "site-packages", monkeypatch)
    return prefix, module


# ── NOT A CHECKOUT: no shas, a clear state ──────────────────────────────────

def test_a_homebrew_style_prefix_reports_no_shas(homebrew):
    prefix, module = homebrew
    brew_head = _git("rev-parse", "HEAD", cwd=prefix)

    # Control: the OLD default (parents[3] of the module) reads Homebrew's repo.
    assert git_head_sha(module.parents[3]) == brew_head
    assert origin_main_sha(module.parents[3]) == brew_head

    assert is_source_checkout() is False
    assert own_checkout_root() is None
    assert git_head_sha() == "unknown"
    assert origin_main_sha() == "unknown"
    assert origin_ref_age_seconds() is None


def test_the_deployment_block_says_not_checkout(homebrew):
    prefix, _ = homebrew
    d = deployment_freshness(git_head_sha())

    assert d["state"] == "not_checkout"
    assert d["process_vs_tree"] == "not_checkout"
    assert d["tree_vs_origin"] == "not_checkout"
    for key in ("running_sha", "tree_head", "origin_main", "ahead_of_origin",
                "behind_origin", "origin_ref_age_seconds",
                "origin_evidence_age_seconds"):
        assert d[key] is None, f"{key} carries {d[key]!r} on a non-checkout"
    assert _git("rev-parse", "HEAD", cwd=prefix) not in repr(d)
    assert len(d["detail"].split()) >= 5
    # Additive: the same keys as a checkout's block, never fewer.
    assert set(d) == set(deployment_freshness("x", repo_dir=prefix))


def test_booted_from_says_it_is_not_a_source_checkout(homebrew):
    _, module = homebrew
    info = booted_from()
    assert info["source_checkout"] is False
    assert Path(info["package"]) == module.parents[1].resolve()
    assert Path(info["path"]) == module.parents[3].resolve()


def test_brew_update_does_not_nudge_restart(homebrew):
    """The user-visible half: Homebrew's HEAD moving is not new Prometheus code."""
    from prometheus.gateway.heartbeat import Heartbeat

    prefix, module = homebrew
    boot = git_head_sha()
    before = git_head_sha(module.parents[3])
    _commit(prefix, "brew update\n")
    # Control: the enclosing repository really moved, so the old default
    # (boot = Homebrew's HEAD, tree = Homebrew's HEAD now) would have drifted.
    assert git_head_sha(module.parents[3]) != before

    hb = Heartbeat(boot_sha=boot)
    sent: list[str] = []

    async def _rec(text, *, chat_id=None):
        sent.append(text)

    hb._notify = _rec  # type: ignore[assignment]
    asyncio.run(hb._check_staleness())
    assert sent == []


def test_a_users_own_repo_with_a_venv_inside_reports_no_shas(tmp_path, monkeypatch):
    project = _repo(tmp_path / "myproject")
    _pyproject(project, "myproject")
    module = _install_in(project / ".venv" / "lib" / "python3.12" / "site-packages",
                         monkeypatch)

    assert git_head_sha(module.parents[3]) == _git("rev-parse", "HEAD", cwd=project)  # control
    assert is_source_checkout() is False
    assert git_head_sha() == "unknown"
    assert deployment_freshness(git_head_sha())["state"] == "not_checkout"
    assert booted_from()["source_checkout"] is False


def test_status_and_health_serve_it(homebrew, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from prometheus.web.server import create_app

    monkeypatch.delenv("PROMETHEUS_API_TOKEN", raising=False)
    client = TestClient(create_app({}, boot_sha=git_head_sha()))

    body = client.get("/api/status").json()
    assert body["running_sha"] is None and body["tree_head"] is None
    assert body["stale"] is False
    assert body["checkout"]["source_checkout"] is False
    assert body["deployment"]["state"] == "not_checkout"

    health = client.get("/health").json()
    assert health["stale"] is False
    assert health["deployment_state"] == "not_checkout"


def test_the_ladder_stamps_no_commit_from_an_enclosing_repository(homebrew):
    from prometheus.gym.ladder.runner import harness_commit

    assert harness_commit() is None


def test_the_fetcher_default_uses_the_same_answer(homebrew):
    """No package_file: the fetcher asks own_checkout_root, as the shas do."""
    prefix, _ = homebrew
    _git("update-ref", "-d", REF, cwd=prefix)
    upstream = _repo(prefix.parent / "upstream")
    _git("remote", "add", "origin", str(upstream), cwd=prefix)

    fetcher = OriginFetcher()
    assert fetcher.fetch_once() is False
    assert fetcher.state.state() == "not_checkout"
    assert subprocess.run(["git", "rev-parse", "--verify", "-q", REF], cwd=prefix,
                          capture_output=True).returncode != 0, (
        "the default fetcher wrote a ref into Homebrew's repository")


# ── A CHECKOUT: still reported ──────────────────────────────────────────────

def test_a_prometheus_checkout_still_reports_its_shas(tmp_path, monkeypatch):
    root = _repo(tmp_path / "Prometheus")
    _pyproject(root, PROJECT_NAME)
    _install_in(root / "src", monkeypatch)
    _git("add", "-A", cwd=root)
    head = _commit(root, "two\n")
    _git("update-ref", REF, head, cwd=root)

    assert is_source_checkout() is True
    assert own_checkout_root() == root.resolve()
    assert git_head_sha() == head
    d = deployment_freshness(head)
    assert d["state"] == "current"
    assert d["running_sha"] == d["tree_head"] == d["origin_main"] == head
    assert booted_from()["source_checkout"] is True

    from prometheus.gym.ladder.runner import harness_commit
    assert harness_commit() == head


def test_the_running_package_is_a_checkout():
    """From a source checkout (CI, a worktree, the deploy clone) the real
    module resolves to the repository it sits in."""
    root = Path(env_mod.__file__).resolve().parents[3]
    if not (root / ".git").exists() or not (root / "pyproject.toml").exists():
        pytest.skip("this test run is not from a source checkout")
    assert own_checkout_root() == root
    assert git_head_sha() != "unknown"


# ── COULD NOT TELL is not NOT A CHECKOUT ────────────────────────────────────

def test_git_that_cannot_run_is_unknown_and_asked_again(tmp_path, monkeypatch):
    """A checkout whose git cannot be run is ``unknown``, never ``not_checkout``
    — and the answer is not kept, so the deploy clone does not lose its
    freshness signal until a restart over one failed call."""
    root = _repo(tmp_path / "Prometheus")
    _pyproject(root, PROJECT_NAME)
    _install_in(root / "src", monkeypatch)
    head = _git("rev-parse", "HEAD", cwd=root)

    path = os.environ.get("PATH", "")
    monkeypatch.setenv("PATH", str(tmp_path / "no-git-here"))
    assert is_source_checkout() is None
    assert git_head_sha() == "unknown"
    d = deployment_freshness(head)
    assert d["state"] == "unknown"
    assert booted_from()["source_checkout"] is None

    monkeypatch.setenv("PATH", path)
    assert is_source_checkout() is True
    assert git_head_sha() == head
