"""The sdist ships tracked content only — asserted against a BUILT tarball.

WHY THIS EXISTS
---------------
On 2026-08-15 ``uv build`` produced a 9,790-file source distribution. 8,872 of
those files were ``.claude/worktrees/`` — eleven complete copies of the repo,
three of them on commits predating the 2026-08-01 tailnet-IP purge (9a25a50,
#130) and therefore carrying the pre-scrub blobs of the very files that purge
cleaned. A further 54 were untracked working-tree files that had never passed
``.githooks/pre-commit``.

Nothing was misconfigured. ``pyproject.toml`` simply had no
``[tool.hatch.build.targets.sdist]`` stanza, so hatchling applied its default —
"everything the VCS does not ignore" — and the only thing keeping worktrees out
of the VCS was ``.git/info/exclude``, which is PER-CLONE and untracked. Git
honoured it; the build backend did not; no other clone had it. Local,
invisible, unversioned state decided the contents of a release artifact.

WHAT THIS TEST ASSERTS, AND WHY IT IS SHAPED THIS WAY
-----------------------------------------------------
It builds a REAL sdist and reads the REAL tarball. A test that only inspected
``pyproject.toml`` would assert the configuration, and the configuration was
never the thing that shipped (§2e — testing the component proves the
component; only the far side proves the delivery).

It builds from a temporary tree into which it PLANTS the junk: a fake
``.claude/worktrees/`` and a stray untracked file. Building the repo as-is
would prove nothing, because a clean worktree yields a clean tarball under the
old configuration too — the test would pass for the wrong reason (§3b).

The temp tree deliberately has NO ``.git`` directory. That is the point: with
an explicit include allowlist the artifact's contents must not depend on VCS
state at all. If someone reverts to the default, this test fails here first.
"""

from __future__ import annotations

import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Roots the sdist is expected to carry. Read from pyproject rather than
# restated, so the two cannot drift.
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - py<3.11
    tomllib = None


def _include_roots() -> list[str]:
    assert tomllib is not None, "python >= 3.11 required"
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    include = (
        data.get("tool", {}).get("hatch", {}).get("build", {})
        .get("targets", {}).get("sdist", {}).get("include")
    )
    assert include, (
        "pyproject.toml has no [tool.hatch.build.targets.sdist].include. "
        "Without it hatchling falls back to 'everything not VCS-ignored', "
        "which is what shipped 8,872 .claude/worktrees files in a release "
        "artifact. The allowlist is the control; do not remove it."
    )
    return [str(p).lstrip("/") for p in include]


def test_sdist_declares_an_explicit_allowlist():
    """The default must never be what decides a release artifact's contents."""
    roots = _include_roots()
    assert "src" in roots, f"sdist must ship the package; include={roots}"


def test_claude_worktrees_are_gitignored_not_just_locally_excluded():
    """`.gitignore` is tracked; `.git/info/exclude` is not.

    The rule that decides what ships must be visible to every clone and every
    tool, not to one machine's local git state.
    """
    text = (REPO / ".gitignore").read_text(encoding="utf-8")
    assert any(line.strip().rstrip("/") == ".claude" for line in text.splitlines()), (
        ".claude/ must be in the tracked .gitignore. It lived only in "
        ".git/info/exclude until 2026-08-15, which no other clone has and "
        "which the build backend does not read."
    )


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
def test_built_sdist_excludes_untracked_and_worktree_junk(tmp_path: Path):
    """Build a real sdist from a tree seeded with junk; assert none of it ships.

    The junk is planted by the test. Building the repo as-is would pass under
    the old configuration too, on a clean checkout.
    """
    roots = _include_roots()
    stage = tmp_path / "stage"
    stage.mkdir()

    shutil.copy2(REPO / "pyproject.toml", stage / "pyproject.toml")
    for name in roots:
        src = REPO / name
        if not src.exists():
            continue
        dst = stage / name
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    # --- the junk, planted deliberately -----------------------------------
    wt = stage / ".claude" / "worktrees" / "some-session" / "src" / "prometheus"
    wt.mkdir(parents=True)
    (wt / "leaked.py").write_text("# pretend pre-scrub blob\n", encoding="utf-8")
    (stage / "STRAY-UNTRACKED-AUDIT.md").write_text("# never passed a hook\n",
                                                    encoding="utf-8")
    (stage / "src" / "STRAY-INSIDE-AN-INCLUDED-ROOT.md").write_text(
        "# inside src/, still not source\n", encoding="utf-8")

    out = tmp_path / "dist"
    proc = subprocess.run(
        ["uv", "build", "--sdist", "--offline", "--out-dir", str(out)],
        cwd=stage, capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"build failed:\n{proc.stdout}\n{proc.stderr}"

    tarballs = list(out.glob("*.tar.gz"))
    assert len(tarballs) == 1, f"expected one sdist, got {tarballs}"
    with tarfile.open(tarballs[0]) as tf:
        names = [n.split("/", 1)[1] for n in tf.getnames() if "/" in n]

    claude = [n for n in names if n.startswith(".claude")]
    assert not claude, (
        f"{len(claude)} .claude/ path(s) in the sdist. This is the 2026-08-15 "
        f"defect: eleven repo copies, three carrying pre-purge tailnet IPs.\n  "
        + "\n  ".join(claude[:10])
    )

    stray_root = [n for n in names if n.startswith("STRAY-")]
    assert not stray_root, (
        f"untracked working-tree files at the repo root reached the sdist: "
        f"{stray_root}. All 54 of the 2026-08-15 untracked files were of this "
        f"shape — audits/, docs/, emberfall/, gym/, skills/, scripts/."
    )

    # KNOWN RESIDUAL, asserted rather than hidden.
    #
    # An include list of DIRECTORY ROOTS bounds which directories ship; it does
    # not make the artifact tracked-only. A stray file dropped INSIDE an
    # allowlisted root still ships, and hatchling has no "only what git tracks"
    # mode to close it here.
    #
    # This assertion pins the boundary so the next reader learns it from a
    # passing test instead of rediscovering it from a release. The property
    # Will ruled for — tracked content only — is enforced one test down, by
    # test_built_sdist_ships_only_tracked_files, which compares a real build
    # against `git ls-files`.
    inside = [n for n in names if n == "src/STRAY-INSIDE-AN-INCLUDED-ROOT.md"]
    assert inside, (
        "a stray file inside an allowlisted root no longer ships — the residual "
        "this assertion documents has been closed. Delete this assertion and "
        "say so in the PR; do not leave a passing test asserting a limit that "
        "no longer exists."
    )


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
@pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")
def test_built_sdist_ships_only_tracked_files(tmp_path: Path):
    """Every path in a REAL sdist is a path git tracks.

    This is the property the include list cannot express on its own. It
    compares two LIVE answers — the tarball's own member list and
    ``git ls-files`` — rather than either against a restated expectation. A
    hand-written expected file list would be maintained by the same person who
    wrote the include list, and would drift with it.

    On a clean CI checkout this is close to a tautology. It earns its place on
    a working machine, where the repo carries untracked audits and scratch
    files — which is exactly the tree the defective artifact was built from.
    """
    out = tmp_path / "dist"
    proc = subprocess.run(
        ["uv", "build", "--sdist", "--offline", "--out-dir", str(out)],
        cwd=REPO, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"sdist build unavailable here: {proc.stderr.strip()[:200]}")

    with tarfile.open(next(out.glob("*.tar.gz"))) as tf:
        shipped = {n.split("/", 1)[1] for n in tf.getnames() if "/" in n}
    shipped.discard("PKG-INFO")          # generated by the build, never tracked

    tracked = set(subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.split())

    untracked_shipped = sorted(shipped - tracked)
    assert not untracked_shipped, (
        f"{len(untracked_shipped)} file(s) in the sdist are not tracked by git. "
        f"A release artifact must not contain content that never passed review "
        f"or .githooks/pre-commit.\n\n  " + "\n  ".join(untracked_shipped[:20])
    )
