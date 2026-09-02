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

import re
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HOOK = REPO / ".githooks" / "pre-commit"

# ---------------------------------------------------------------------------
# FIXTURES, NOT EXCEPTIONS.
#
# Every entry below is deliberate TEST DATA that trips one of the hook's
# patterns by design. None is a credential, and the host names among them are
# MagicDNS labels that do not resolve outside the tailnet.
#
# Scrubbing them to make a scanner quiet would delete the very data that
# proves the controls work (§3c — the test that banned the words, not the
# claim). So the guard allowlists them BY FILE AND BY COUNT instead: a hit in
# any other file fails, and a SIXTEENTH hit in one of these files fails too.
# The count is the ratchet; without it this would be a place to hide.
#
# Adding an entry here is a claim that the string is test data. Make that
# claim explicitly, in the reason, or do not add it.
#
# NOTE, and it is §3c biting its own author: the first draft of this block
# QUOTED the flagged tokens while explaining that they are harmless, and the
# guard promptly failed against this file. The reasons below therefore
# describe the fixtures without spelling them — the same fix §3c prescribes
# for prose guards, applied to the allowlist's own justification. Do not
# re-add the literals here, and do not allowlist this file to work around it.
# ---------------------------------------------------------------------------
HOOK_PATTERN_FIXTURES: dict[str, tuple[int, str]] = {
    "tests/test_wiki.py": (
        13, "wiki-dedup fixture entities, deliberately case-varied to exercise "
            "entity normalization; MagicDNS labels, no IPs",
    ),
    "tests/test_turn_errors.py": (
        1, "the synthetic provider key in the test asserting a URL query "
           "string is never echoed to clients — the fixture IS the control",
    ),
}


def _hook_checks() -> tuple[list[tuple[str, str, str, str]], str, str]:
    """The hook's own patterns, resolved BY BASH.

    Bash is the only correct parser of bash quoting. A python-side regex over
    the hook text got this wrong twice while this guard was being written: it
    broke on the single-quote-embedding idiom in the last pattern, and it read
    the skip_placeholders/scope arguments as absent whenever they sat past a
    line continuation — which silently switched off the provider-key check's
    placeholder exemption and manufactured a false positive against a test
    fixture.

    So the patterns are never transcribed here. Bash re-reads its own file.
    """
    script = r"""
set -uo pipefail
check_pattern() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "${3:-0}" "${4:-all}"; }
eval "$(sed -n '/^PLACEHOLDER_REGEX=/p; /^ALLOWLIST_REGEX=/p' "$1")"
eval "$(awk '/^check_pattern /{f=1} f{print} f && !/\\$/{f=0}' "$1")"
printf 'PLACEHOLDER_REGEX\t%s\n' "$PLACEHOLDER_REGEX" >&2
printf 'ALLOWLIST_REGEX\t%s\n' "$ALLOWLIST_REGEX" >&2
"""
    proc = subprocess.run(["bash", "-c", script, "_", str(HOOK)],
                          capture_output=True, text=True)
    assert proc.returncode == 0 and proc.stdout.strip(), (
        f"could not extract patterns from {HOOK}: {proc.stderr}")
    env = dict(l.split("\t", 1) for l in proc.stderr.splitlines() if "\t" in l)
    checks = [tuple(l.split("\t")) for l in proc.stdout.splitlines()
              if l.count("\t") == 3]
    assert len(checks) >= 8, f"only {len(checks)} patterns parsed — extractor is stale"
    return checks, env["PLACEHOLDER_REGEX"].replace("(?i)", ""), env["ALLOWLIST_REGEX"]


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


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not on PATH")
@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not on PATH")
def test_built_sdist_has_no_hook_pattern_hits_outside_the_fixture_allowlist(
    tmp_path: Path,
):
    """Run .githooks/pre-commit's OWN patterns over a REAL sdist.

    The hook gates what ENTERS the repo. Nothing gated what LEAVES it as a
    release artifact — which is how 22 tailnet-IP hits and eleven repo copies
    reached a tarball while every commit along the way was clean. This closes
    that direction, on the artifact rather than on the configuration.

    Hits are compared against HOOK_PATTERN_FIXTURES by file AND by count, so a
    new hit fails whether it appears in a new file or in an allowlisted one.
    """
    out = tmp_path / "dist"
    proc = subprocess.run(
        ["uv", "build", "--sdist", "--offline", "--out-dir", str(out)],
        cwd=REPO, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"sdist build unavailable here: {proc.stderr.strip()[:200]}")

    extracted = tmp_path / "x"
    with tarfile.open(next(out.glob("*.tar.gz"))) as tf:
        tf.extractall(extracted)
    root = next(extracted.iterdir())

    checks, placeholder, allowlist = _hook_checks()
    # The hook excludes ITSELF from its own scan (STAGED_ALL drops
    # ^\.githooks/): that file necessarily contains every pattern it hunts for.
    files = [p for p in root.rglob("*") if p.is_file()
             and ".githooks/" not in str(p.relative_to(root))]

    hits: dict[str, list[str]] = {}
    for label, pattern, skip_ph, scope in checks:
        subject = files
        if scope == "code":
            subject = [p for p in files if p.suffix != ".md"
                       and not str(p.relative_to(root)).startswith("docs/")]
        rx = re.compile(pattern)
        for p in subject:
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if not rx.search(text):
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if not rx.search(line):
                    continue
                if not rx.search(re.sub(allowlist, "", line)):   # hook's own mask
                    continue
                if skip_ph == "1" and re.search(placeholder, line, re.I):
                    continue
                hits.setdefault(str(p.relative_to(root)), []).append(f"{i}: {label}")

    unexpected = {f: v for f, v in hits.items() if f not in HOOK_PATTERN_FIXTURES}
    assert not unexpected, (
        "the sdist carries hook-pattern hits in files that are not declared "
        "test fixtures. Each is a credential, a private IP, or infrastructure "
        "that must not ship in a release artifact.\n\n"
        + "\n".join(f"  {f}\n    " + "\n    ".join(v) for f, v in unexpected.items())
    )

    drift = {
        f: (len(hits.get(f, [])), expected)
        for f, (expected, _) in HOOK_PATTERN_FIXTURES.items()
        if len(hits.get(f, [])) != expected
    }
    assert not drift, (
        "declared-fixture hit counts changed. This allowlist is FIXTURES, not "
        "exceptions: a new hit in one of these files is a new claim that a "
        "flagged string is test data, and it must be made deliberately.\n\n"
        + "\n".join(f"  {f}: found {got}, allowlist says {exp}"
                    for f, (got, exp) in drift.items())
    )
