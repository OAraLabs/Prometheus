"""The secret scanner must never report clean for a scan it did not perform.

WHAT HAPPENED
-------------
`.githooks/pre-commit` ran every pattern through `grep -nP`. BSD grep — the
grep git invokes on macOS, whatever the interactive shell aliases — has no -P:

    $ /usr/bin/grep -qP abc ; echo $?
    grep: invalid option -- P
    2

The call site was `... | grep -nP "$pattern" || true`. That `|| true` collapses
grep's three outcomes into two: exit 1 ("no matches") and exit 2 ("the scanner
did not run") both produced an empty result, and the loop moved on. All nine
patterns evaluated nothing, `BLOCKED` stayed 0, and the hook printed
"All clean. No sensitive data found in staged files." and exited 0 — with a
real match staged.

CONTRIBUTING calls this hook the only line of defense on this repo, and it is
the mechanism behind the standing rule about infrastructure identifiers in
persisted content. On macOS it was not a weak net; it was no net that said
otherwise.

WHAT THIS FILE ASSERTS
----------------------
The flag was not the defect — the failure MODE was. Swapping -P for -E fixes
today and leaves the next missing flag, missing binary, or unreadable blob to
read as "no matches" all over again. So these tests replay the known-bad
shapes and assert the three outcomes stay DISTINGUISHABLE:

    clean          exit 0, says "All clean"
    secret found   exit 1, says "BLOCKED"
    scanner broke  exit 1, says "SCANNER DID NOT RUN", never "All clean"

`test_scanner_failure_and_no_matches_never_render_the_same` is the one that
encodes the actual lesson.

A note on the fixtures: the IP-shaped strings here are chosen, not observed,
and name nothing that exists. `SECRET_LINE` is assembled from two halves at
import — see the comment at its definition for why that is necessary rather
than evasive. `ALLOWLISTED_LINE` sits inside the /24 the hook exempts on
purpose, so it can be written out in full.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
HOOK = REPO / ".githooks" / "pre-commit"

pytestmark = [
    pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH"),
    pytest.mark.skipif(shutil.which("bash") is None, reason="bash not on PATH"),
]


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------
def _repo_with_staged(tmp_path: Path, content: str) -> Path:
    """A throwaway git repo with `content` staged and the real hook in place."""
    repo = tmp_path / "r"
    (repo / ".githooks").mkdir(parents=True)
    shutil.copy2(HOOK, repo / ".githooks" / "pre-commit")
    (repo / ".githooks" / "pre-commit").chmod(0o755)

    run = lambda *a: subprocess.run(a, cwd=repo, check=True, capture_output=True)
    run("git", "init", "-q")
    run("git", "config", "user.email", "t@example.invalid")
    run("git", "config", "user.name", "t")
    (repo / "staged.txt").write_text(content, encoding="utf-8")
    run("git", "add", "staged.txt")
    return repo


def _run_hook(repo: Path, grep: str | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    # Guard 1 of the hook refuses commits from a non-worktree checkout; this
    # throwaway repo is one. That guard is not what these tests are about.
    env["PROMETHEUS_ALLOW_DEV_COMMIT"] = "1"
    if grep is not None:
        env["PROMETHEUS_PRECOMMIT_GREP"] = grep
    return subprocess.run(
        ["bash", str(repo / ".githooks" / "pre-commit")],
        cwd=repo, capture_output=True, text=True, env=env,
    )


def _stub_grep(tmp_path: Path, name: str, body: str) -> str:
    p = tmp_path / name
    p.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    p.chmod(0o755)
    return str(p)


# ASSEMBLED AT IMPORT, NOT WRITTEN OUT — and that is not scanner-dodging.
#
# This file lives inside the repository the hook guards. A test proving that a
# real match gets BLOCKED needs a string that really matches, so a literal here
# would be blocked by the very pattern under test — correctly, and it was:
# the first attempt to commit this file was refused by the fixed hook, which is
# the clearest possible evidence the fix works.
#
# Unlike tests/test_sdist_contents.py, which has HOOK_PATTERN_FIXTURES for
# declaring known test data with a counted justification, the pre-commit hook
# has no fixture mechanism — it blocks on content, full stop. So composition is
# the only way a test of a blocking scanner can be committed at all. The value
# is arbitrary, sits in the range the Tailscale pattern covers, is outside the
# /24 the hook allowlists, and names nothing that exists.
_SHAPED_LIKE_A_TAILNET_IP = "100." + "99.123.45"
SECRET_LINE = f'host = "{_SHAPED_LIKE_A_TAILNET_IP}"\n'

# RFC 6598 shared space — the /24 the hook exempts on purpose, because
# security/url_guard.py must carry the constant literally. Safe to write out,
# which is what makes it a fair test of the allowlist.
ALLOWLISTED_LINE = 'shared = "100.64.0.7"\n'
CLEAN_LINE = "just some ordinary prose\n"


def _out(proc) -> str:
    return proc.stdout + proc.stderr


# --------------------------------------------------------------------------
# The four replayed shapes
# --------------------------------------------------------------------------
def test_a_real_match_is_blocked(tmp_path):
    """The baseline the broken hook failed: a staged secret must not pass."""
    proc = _run_hook(_repo_with_staged(tmp_path, SECRET_LINE))
    assert proc.returncode != 0, (
        f"a staged secret was allowed through.\n{_out(proc)}")
    assert "BLOCKED" in _out(proc)
    assert "All clean" not in _out(proc)


def test_an_allowlisted_value_is_not_blocked(tmp_path):
    """The allowlist still works — a fix that blocks everything is not a fix.

    100.64.0.x is RFC 6598 shared space, exempted deliberately because
    security/url_guard.py must carry the constant literally.
    """
    proc = _run_hook(_repo_with_staged(tmp_path, ALLOWLISTED_LINE))
    assert proc.returncode == 0, (
        f"an allowlisted value was blocked — this cries wolf.\n{_out(proc)}")
    assert "All clean" in _out(proc)


def test_a_grep_that_rejects_the_flag_refuses_the_commit(tmp_path):
    """THE ORIGINAL DEFECT. Exit 2 is 'did not run', not 'nothing found'."""
    grep = _stub_grep(tmp_path, "badflag",
                      'echo "grep: invalid option" >&2\nexit 2')
    proc = _run_hook(_repo_with_staged(tmp_path, SECRET_LINE), grep=grep)
    assert proc.returncode != 0, (
        f"the hook reported success using a grep that cannot run.\n{_out(proc)}")
    assert "SCANNER DID NOT RUN" in _out(proc)


def test_a_missing_grep_refuses_the_commit(tmp_path):
    """No binary at all is the same class: unknown, not clean."""
    proc = _run_hook(_repo_with_staged(tmp_path, SECRET_LINE),
                     grep=str(tmp_path / "definitely-not-here"))
    assert proc.returncode != 0, (
        f"the hook reported success with no grep at all.\n{_out(proc)}")
    assert "SCANNER DID NOT RUN" in _out(proc)


def test_a_grep_that_always_reports_no_matches_refuses_the_commit(tmp_path):
    """Exit 1 for everything is a LEGAL code, so exit-code checking alone
    cannot catch it. The engine self-test's positive control is what does.

    Without the self-test this stub would sail through: every pattern would
    return a truthful-looking "no matches" and the hook would report clean.
    """
    grep = _stub_grep(tmp_path, "liar", "exit 1")
    proc = _run_hook(_repo_with_staged(tmp_path, SECRET_LINE), grep=grep)
    assert proc.returncode != 0, (
        f"a grep that never matches anything produced a clean report.\n{_out(proc)}")
    assert "SCANNER DID NOT RUN" in _out(proc)


# --------------------------------------------------------------------------
# The lesson itself
# --------------------------------------------------------------------------
def test_scanner_failure_and_no_matches_never_render_the_same(tmp_path):
    """`unknown` and `zero` must not be the same output.

    This is the assertion that would have failed before the fix: the broken
    hook produced BYTE-IDENTICAL stdout and an identical exit code for
    "scanned a clean file" and "could not scan at all".
    """
    clean = _run_hook(_repo_with_staged(tmp_path / "a", CLEAN_LINE))
    broken = _run_hook(
        _repo_with_staged(tmp_path / "b", CLEAN_LINE),
        grep=_stub_grep(tmp_path, "badflag2",
                        'echo "grep: invalid option" >&2\nexit 2'),
    )

    assert clean.returncode == 0, _out(clean)
    assert broken.returncode != 0, (
        "a scanner that could not run reported the same success as a clean "
        f"scan.\n{_out(broken)}")
    assert clean.returncode != broken.returncode
    assert _out(clean) != _out(broken)
    assert "All clean" in _out(clean)
    assert "All clean" not in _out(broken), (
        "the failure path still says 'All clean' — the two outcomes are still "
        "indistinguishable to anyone reading the terminal.")


# --------------------------------------------------------------------------
# Static guards against the shapes that caused this
# --------------------------------------------------------------------------
def test_no_grep_invocation_is_followed_by_or_true():
    """`|| true` after a grep is what erased the distinction in the first place."""
    offenders = [
        f"{i}: {l.strip()}"
        for i, l in enumerate(HOOK.read_text(encoding="utf-8").splitlines(), 1)
        if not l.lstrip().startswith("#")
        and re.search(r"\bgrep\b|\$GREP|\bscan(_q)?\b", l)
        and "|| true" in l
    ]
    assert not offenders, (
        "a grep result is being swallowed by `|| true`, which makes "
        "'the scanner failed' indistinguishable from 'no matches':\n  "
        + "\n  ".join(offenders)
    )


def test_the_hook_asks_for_no_regex_dialect_beyond_posix_ere():
    """PCRE is not portable; POSIX ERE is what both BSD and GNU grep have."""
    code = [l for l in HOOK.read_text(encoding="utf-8").splitlines()
            if not l.lstrip().startswith("#")]
    pcre_flag = [l.strip() for l in code if re.search(r"grep[^|]*\s-\w*P\b", l)]
    assert not pcre_flag, (
        "the hook asks grep for PCRE (-P). BSD grep does not have it and exits "
        "2, which is how this scanner spent its life reporting clean:\n  "
        + "\n  ".join(pcre_flag)
    )
    inline_flags = [l.strip() for l in code if "(?i)" in l]
    assert not inline_flags, (
        "an inline `(?i)` is a PCRE construct. BSD grep happens to honour it "
        "inside an ERE and GNU grep does not — depending on that difference is "
        "depending on undefined behaviour. Use the -i flag:\n  "
        + "\n  ".join(inline_flags)
    )
