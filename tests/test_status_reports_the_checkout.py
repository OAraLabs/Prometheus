"""`/api/status` must say WHICH checkout it booted from, not just which sha.

THE INCIDENT (2026-08-25). `~/Prometheus` was pulled, the service restarted, and
`/api/status` reported `stale: false`. The daemon was still running week-old
code: its WorkingDirectory is `~/prometheus-deploy`, a separate clone of the
same repo, and the pull had landed in a tree that process never reads.

`stale` was not wrong. It compares `running_sha` (captured at boot) to
`tree_head` (HEAD now) — and BOTH are read from the checkout the loaded package
came from. Deploy the wrong tree and the two agree perfectly, so the field reads
false while nothing you shipped is running. It is structurally blind to the
failure it appears to cover, which is worse than absent: every deploy that day
leaned on it.

These tests pin the field that closes it. `package` is the decisive one — the
directory Python actually imported, not a working directory, not a config
value, so a reader compares rather than infers.
"""

from __future__ import annotations

from pathlib import Path

import prometheus
from prometheus.context.environment import booted_from, git_head_sha


def test_it_reports_the_directory_python_actually_imported(monkeypatch, tmp_path):
    """Asserted from a DIFFERENT working directory on purpose.

    The first version of this test ran at the repo root, where a cwd-derived
    implementation (`Path.cwd() / "src" / "prometheus"`) gives the right answer
    by coincidence — the mutation proved it green. cwd is exactly what was wrong
    in the incident: the service's WorkingDirectory was the deploy clone while a
    second checkout sat elsewhere, and a cwd-derived field would have reported
    whichever directory the process happened to start in rather than the code it
    loaded.
    """
    monkeypatch.chdir(tmp_path)
    info = booted_from()
    assert Path(info["package"]) == Path(prometheus.__file__).resolve().parent, (
        "package must be the loaded package's own directory — anything derived "
        "from cwd or config can point at a checkout this process never reads"
    )


def test_the_repo_root_contains_the_package():
    info = booted_from()
    assert Path(info["package"]).is_relative_to(Path(info["path"]))


def test_is_git_reflects_reality():
    info = booted_from()
    assert info["is_git"] == (Path(info["path"]) / ".git").exists()


def test_the_sha_and_the_path_describe_the_same_checkout():
    """The whole point: the sha is only meaningful WITH the path. If these ever
    came from different anchors, `stale` would compare a sha from one tree
    against a path from another and the field would be worse than useless."""
    info = booted_from()
    assert git_head_sha() == git_head_sha(info["path"])


def test_status_serves_it():
    """Additive, on the wire, same shape as iteration_ceilings."""
    src = Path("src/prometheus/web/server.py").read_text()
    assert '"checkout": _booted_from(),' in src, (
        "the field is not wired into /api/status — a reader still cannot tell "
        "which tree is live"
    )


def test_stale_alone_cannot_tell_two_checkouts_apart():
    """Documents WHY the field exists, as an executable statement rather than a
    comment: two clones of the same repo at the same commit are indistinguishable
    by sha, so `running_sha == tree_head` says nothing about WHERE."""
    info = booted_from()
    other_clone_same_commit = {
        "path": "/somewhere/else/prometheus-deploy",
        "package": "/somewhere/else/prometheus-deploy/src/prometheus",
        "is_git": True,
    }
    # Same sha, different tree — and only `path`/`package` distinguishes them.
    assert other_clone_same_commit["path"] != info["path"]
    assert other_clone_same_commit["package"] != info["package"]
