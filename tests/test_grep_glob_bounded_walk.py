"""grep and glob walk lazily, skip dependency/VCS folders and stop at a cap.

The defect (2026-11-27, a daemon at 42 GB): grep did
``sorted(root.glob("**/*"))`` and glob did ``list(root.glob(pattern))``, so one
call over a big workspace built every path under it — 5.8 million ``Path``
objects for ~/projects — before applying a limit of 200. These tests hold the
replacement (tools/tree_walk.py) to: the same matches as pathlib, no entry into
.git/node_modules/.venv/.claude/worktrees by wildcard, a walk that stops when
the caller stops, a hard cap with a note, and memory that stays flat as the
tree grows. The ripgrep listing is tested where ``rg`` is installed.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from prometheus.tools import tree_walk
from prometheus.tools.builtin.glob import GlobTool, GlobToolInput
from prometheus.tools.builtin.grep import GrepTool, GrepToolInput


def _run(tool, args, cwd: Path):
    return asyncio.run(tool.execute(args, SimpleNamespace(cwd=cwd)))


@pytest.fixture()
def no_rg(monkeypatch):
    monkeypatch.setattr(tree_walk, "ripgrep", lambda: None)


def _tree(root: Path) -> Path:
    files = [
        "a.py", "b.txt", ".hidden.py", "c.PY",
        "src/x.py", "src/y.txt", "src/pkg/z.py", "src/pkg/deep/w.py",
        "src2/q.py", "docs/readme.md", ".config/s.py",
    ]
    for f in files:
        p = root / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"needle in {f}\n")
    return root


def _skippy(root: Path) -> Path:
    _tree(root)
    for f in [
        ".git/objects/needle.py", "node_modules/lib/index.py", ".venv/lib/site.py",
        "pkg/__pycache__/m.py", ".claude/worktrees/wt/copy.py", ".claude/notes.py",
    ]:
        p = root / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("needle\n")
    return root


# --------------------------------------------------------------------------- #
# Matching: the same answers pathlib gave, for the patterns the tools see
# --------------------------------------------------------------------------- #

PATTERNS = [
    "*", "*.py", "**/*", "**/*.py", "src/*.py", "src/**/*.py", "*/x.py",
    "s?c/*", "[ab].*", "[!a].py", "src/pkg/*", "**/pkg/*.py", "docs/readme.md",
    ".config/*", "*.PY",
]


@pytest.mark.parametrize("pattern", PATTERNS)
def test_matches_what_pathlib_matched(tmp_path: Path, pattern: str) -> None:
    root = _tree(tmp_path)
    ours = sorted(str(p.relative_to(root)) for p in tree_walk.walk(root, pattern))
    theirs = sorted(str(p.relative_to(root)) for p in root.glob(pattern))
    assert ours == theirs


def test_double_star_matches_zero_folders(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    got = {str(p.relative_to(root)) for p in tree_walk.walk(root, "src/**/*.py")}
    assert "src/x.py" in got and "src/pkg/deep/w.py" in got


def test_non_recursive_pattern_does_not_descend(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    stats = tree_walk.WalkStats()
    list(tree_walk.walk(root, "*.py", stats=stats))
    top = len(os.listdir(root))
    assert stats.entries == top  # never looked inside a subfolder


def test_literal_prefix_anchors_the_walk(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    stats = tree_walk.WalkStats()
    got = [p.relative_to(root).as_posix() for p in tree_walk.walk(root, "src/pkg/*.py", stats=stats)]
    assert got == ["src/pkg/z.py"]
    assert stats.entries == len(os.listdir(root / "src" / "pkg"))


def test_dotdot_after_wildcard_is_a_tool_error(tmp_path: Path, no_rg) -> None:
    _tree(tmp_path)
    r = _run(GlobTool(), GlobToolInput(pattern="*/../*.py"), tmp_path)
    assert r.is_error and ".." in r.output
    r = _run(GrepTool(), GrepToolInput(pattern="x", file_glob="*/../*.py"), tmp_path)
    assert r.is_error and ".." in r.output


@pytest.mark.parametrize("pattern", ["[z-a].py", "[[]x].txt", "[a&&b].py", "[^a].py"])
def test_odd_bracket_classes_answer_like_pathlib_or_say_why(tmp_path: Path, pattern: str, no_rg) -> None:
    root = _tree(tmp_path)
    (root / "[x].txt").write_text("needle\n")
    (root / "^.py").write_text("needle\n")
    try:
        ours = sorted(p.relative_to(root).as_posix() for p in tree_walk.walk(root, pattern))
    except tree_walk.PatternError:
        r = _run(GlobTool(), GlobToolInput(pattern=pattern), root)
        assert r.is_error and "not a valid pattern" in r.output
        assert list(root.glob(pattern)) == []
        return
    assert ours == sorted(p.relative_to(root).as_posix() for p in root.glob(pattern))


def test_trailing_slash_lists_folders_only(tmp_path: Path, no_rg) -> None:
    root = _tree(tmp_path)
    got = sorted(p.relative_to(root).as_posix() for p in tree_walk.walk(root, "*/"))
    assert got == sorted(p.relative_to(root).as_posix() for p in root.glob("*/"))
    assert "a.py" not in got


def test_bounded_pattern_follows_a_symlinked_folder(tmp_path: Path, no_rg) -> None:
    root = _tree(tmp_path)
    (root / "linked").symlink_to(root / "src" / "pkg")
    got = [p.relative_to(root).as_posix() for p in tree_walk.walk(root, "*/z.py")]
    assert "linked/z.py" in got


def test_symlink_loop_does_not_hang(tmp_path: Path, no_rg) -> None:
    _tree(tmp_path)
    (tmp_path / "src" / "loop").symlink_to(tmp_path)
    r = _run(GlobTool(), GlobToolInput(pattern="**/*.py", limit=5000), tmp_path)
    assert "src/loop/src" not in r.output


# --------------------------------------------------------------------------- #
# Pruning
# --------------------------------------------------------------------------- #

def test_wildcards_skip_dependency_and_vcs_folders(tmp_path: Path, no_rg) -> None:
    root = _skippy(tmp_path)
    r = _run(GlobTool(), GlobToolInput(pattern="**/*.py", limit=5000), root)
    for bad in (".git/", "node_modules/", ".venv/", "__pycache__/", ".claude/worktrees/"):
        assert bad not in r.output.split("\n(")[0], bad
    assert ".claude/notes.py" in r.output  # .claude itself is still searched
    assert "skipped" in r.output and "node_modules" in r.output

    g = _run(GrepTool(), GrepToolInput(pattern="needle", limit=2000), root)
    assert "node_modules/lib/index.py" not in g.output
    assert ".claude/worktrees/wt/copy.py" not in g.output
    assert "src/x.py:1:" in g.output


def test_a_folder_named_in_the_pattern_is_searched(tmp_path: Path, no_rg) -> None:
    root = _skippy(tmp_path)
    r = _run(GlobTool(), GlobToolInput(pattern="node_modules/**/*.py"), root)
    assert "node_modules/lib/index.py" in r.output
    r = _run(GlobTool(), GlobToolInput(pattern="**/node_modules/*/*.py"), root)
    assert "node_modules/lib/index.py" in r.output
    g = _run(GrepTool(), GrepToolInput(pattern="needle", root=str(root / ".claude/worktrees")), root)
    assert "wt/copy.py:1:" in g.output


# --------------------------------------------------------------------------- #
# Streaming, the cap and memory
# --------------------------------------------------------------------------- #

def _wide(root: Path, dirs: int, per: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for d in range(dirs):
        sub = root / f"d{d:04d}"
        sub.mkdir()
        for f in range(per):
            (sub / f"f{f:03d}.txt").write_text("needle\n")
    return root


def test_walk_is_lazy(tmp_path: Path) -> None:
    root = _wide(tmp_path, 50, 20)
    stats = tree_walk.WalkStats()
    it = tree_walk.walk(root, "**/*", stats=stats)
    first = [next(it) for _ in range(3)]
    assert len(first) == 3
    assert stats.entries < 100  # one folder or so, not all 1,050 entries


def test_grep_stops_walking_at_its_limit(tmp_path: Path, monkeypatch, no_rg) -> None:
    root = _wide(tmp_path, 200, 10)
    calls = 0
    real = os.scandir

    def counting(path):
        nonlocal calls
        calls += 1
        return real(path)

    monkeypatch.setattr(tree_walk.os, "scandir", counting)
    r = _run(GrepTool(), GrepToolInput(pattern="needle", limit=3), root)
    assert r.output.count(":1:needle") == 3
    assert "stopped at the limit of 3" in r.output
    assert calls < 5, f"walked {calls} folders for 3 matches"


def test_glob_stops_at_its_limit_and_says_so(tmp_path: Path, no_rg) -> None:
    root = _wide(tmp_path, 20, 20)
    r = _run(GlobTool(), GlobToolInput(pattern="**/*.txt", limit=7), root)
    listed = r.output.split("\n(")[0].splitlines()
    assert len(listed) == 7 and listed == sorted(listed)
    assert "stopped at the limit of 7" in r.output


def test_hard_cap_stops_an_unbounded_tree(tmp_path: Path, monkeypatch, no_rg) -> None:
    root = _wide(tmp_path, 30, 30)
    monkeypatch.setattr(tree_walk, "DEFAULT_MAX_ENTRIES", 100)
    r = _run(GrepTool(), GrepToolInput(pattern="no-such-text"), root)
    assert "(no matches)" in r.output
    assert "stopped after looking at 100 entries" in r.output
    r = _run(GlobTool(), GlobToolInput(pattern="**/*.md"), root)
    assert "stopped after looking at 100 entries" in r.output


#: Measured in a fresh interpreter: an audit hook another test file installs
#: (sys.addaudithook cannot be removed) records every path opened for the rest
#: of the session, which grows with the tree and is not this code's memory.
_MEASURE = r"""
import asyncio, json, sys, tracemalloc
from pathlib import Path
from types import SimpleNamespace
from prometheus.tools import tree_walk
from prometheus.tools.builtin.glob import GlobTool, GlobToolInput
from prometheus.tools.builtin.grep import GrepTool, GrepToolInput

tree_walk.ripgrep = lambda: None
small, big = Path(sys.argv[1]), Path(sys.argv[2])
calls = {
    "grep": lambda r: GrepTool().execute(GrepToolInput(pattern="needle", limit=20), SimpleNamespace(cwd=r)),
    "glob": lambda r: GlobTool().execute(GlobToolInput(pattern="**/*.txt", limit=20), SimpleNamespace(cwd=r)),
    "miss": lambda r: GrepTool().execute(GrepToolInput(pattern="absent"), SimpleNamespace(cwd=r)),
}

def peak(name, root):
    tracemalloc.start()
    try:
        asyncio.run(calls[name](root))
        return tracemalloc.get_traced_memory()[1] // 1024
    finally:
        tracemalloc.stop()

out = {}
for name in calls:
    peak(name, small)  # warm imports and caches
    out[name] = (peak(name, small), peak(name, big))
print(json.dumps(out))
"""


def test_memory_stays_flat_as_the_tree_grows(tmp_path: Path) -> None:
    """The defect's signature: peak memory grew with the tree, not the result.

    The old code fails this by a wide margin (sorted(root.glob("**/*")): 420
    KiB on the small tree, 8,389 KiB on the big one). A no-match grep has to
    visit every file, so it is the strictest case: it may hold one folder's
    listing at a time, never every path.
    """
    small = _wide(tmp_path / "small", 10, 50)        # 500 files
    big = _wide(tmp_path / "big", 200, 50)           # 10,000 files
    src = Path(tree_walk.__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": str(src)}
    proc = subprocess.run(
        [sys.executable, "-P", "-s", "-c", _MEASURE, str(small), str(big)],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    peaks = json.loads(proc.stdout.strip().splitlines()[-1])
    for name, (s, b) in peaks.items():
        assert b < s * 2 + 512, f"{name}: peak {s} KiB on 500 files, {b} KiB on 10,000"


def test_grep_skips_huge_files(tmp_path: Path, monkeypatch, no_rg) -> None:
    import prometheus.tools.builtin.grep as grep_mod

    monkeypatch.setattr(grep_mod, "MAX_FILE_BYTES", 10)
    (tmp_path / "big.txt").write_text("needle " * 10)
    (tmp_path / "ok.txt").write_text("needle\n")
    r = _run(GrepTool(), GrepToolInput(pattern="needle"), tmp_path)
    assert "ok.txt:1:" in r.output and "big.txt" not in r.output.split("\n(")[0]
    assert "1 file(s) over" in r.output


# --------------------------------------------------------------------------- #
# ripgrep listing (only where rg is installed)
# --------------------------------------------------------------------------- #

needs_rg = pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")


@needs_rg
def test_rg_lists_the_same_files_minus_gitignored(tmp_path: Path) -> None:
    root = _skippy(tmp_path)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text("docs/\n")
    rg = shutil.which("rg")
    got = sorted(p.relative_to(root).as_posix() for p in tree_walk.walk_files_rg(rg, root, "**/*.py"))
    want = sorted(
        p.relative_to(root).as_posix() for p in tree_walk.walk(root, "**/*.py") if p.is_file()
    )
    assert got == want
    every = {p.relative_to(root).as_posix() for p in tree_walk.walk_files_rg(rg, root, "**/*")}
    assert "docs/readme.md" not in every  # gitignored
    assert ".hidden.py" in every and ".claude/notes.py" in every
    assert not any(e.startswith((".git/", "node_modules/", ".claude/worktrees/")) for e in every)


@needs_rg
def test_rg_respects_depth_and_anchor(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    rg = shutil.which("rg")
    got = sorted(p.relative_to(root).as_posix() for p in tree_walk.walk_files_rg(rg, root, "*.py"))
    assert got == [".hidden.py", "a.py"]
    got = sorted(p.relative_to(root).as_posix() for p in tree_walk.walk_files_rg(rg, root, "src/pkg/**/*.py"))
    assert got == ["src/pkg/deep/w.py", "src/pkg/z.py"]


@needs_rg
def test_grep_with_rg_stops_and_reaps_the_child(tmp_path: Path, monkeypatch) -> None:
    # Enough names (~300 KB) that rg cannot finish into the pipe buffer: if
    # grep stops at its limit, the child must be killed, not left to run.
    root = _wide(tmp_path, 400, 25)
    procs: list[subprocess.Popen] = []
    real_popen = subprocess.Popen

    def recording(*a, **kw):
        p = real_popen(*a, **kw)
        procs.append(p)
        return p

    monkeypatch.setattr(tree_walk.subprocess, "Popen", recording)
    r = _run(GrepTool(), GrepToolInput(pattern="needle", limit=2), root)
    assert r.output.count(":1:needle") == 2
    assert len(procs) == 1
    assert procs[0].returncode is not None, "rg child still running"
    assert procs[0].returncode < 0, "rg ran to the end: the listing did not stop at the limit"


@needs_rg
def test_grep_rg_miss_says_gitignored_files_were_left_out(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("secret/\n")
    (tmp_path / "secret").mkdir()
    (tmp_path / "secret" / "a.txt").write_text("needle\n")
    r = _run(GrepTool(), GrepToolInput(pattern="needle"), tmp_path)
    assert "(no matches)" in r.output and ".gitignore" in r.output
    r = _run(GrepTool(), GrepToolInput(pattern="needle", file_glob="secret/*"), tmp_path)
    assert "secret/a.txt:1:needle" in r.output


def test_rg_that_cannot_run_falls_back_to_the_walk(tmp_path: Path, monkeypatch) -> None:
    root = _tree(tmp_path)
    fake = tmp_path / "bin" / "rg"
    fake.parent.mkdir()
    fake.write_text("#!/bin/sh\nexit 2\n")
    fake.chmod(0o755)
    monkeypatch.setattr(tree_walk, "ripgrep", lambda: str(fake))
    r = _run(GrepTool(), GrepToolInput(pattern="needle", file_glob="src/*.py"), root)
    assert "x.py:1:needle" in r.output


# --------------------------------------------------------------------------- #
# Guards (Will, 2026-11-28): an explicit search of .claude/private still works,
# and nothing under gym/ is skipped — in both modes.
# --------------------------------------------------------------------------- #

@pytest.fixture(params=["walk", "rg"])
def mode(request, monkeypatch):
    if request.param == "rg":
        if shutil.which("rg") is None:
            pytest.skip("ripgrep not installed")
    else:
        monkeypatch.setattr(tree_walk, "ripgrep", lambda: None)
    return request.param


def _repo_with_private(root: Path) -> Path:
    """A repo shaped like Prometheus: .claude/ is gitignored, private notes inside."""
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text(".claude/\n")
    for f in [".claude/private/PLAN.md", ".claude/private/notes/deep.md", "src/a.py"]:
        p = root / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("needle\n")
    (root / ".claude/worktrees/wt").mkdir(parents=True)
    (root / ".claude/worktrees/wt/copy.md").write_text("needle\n")
    return root


def test_explicit_search_of_claude_private_works(tmp_path: Path, mode) -> None:
    root = _repo_with_private(tmp_path)
    by_root = _run(GrepTool(), GrepToolInput(pattern="needle", root=".claude/private"), root)
    assert "PLAN.md:1:needle" in by_root.output
    assert "notes/deep.md:1:needle" in by_root.output

    by_glob = _run(GrepTool(), GrepToolInput(pattern="needle", file_glob=".claude/private/**/*"), root)
    assert ".claude/private/PLAN.md:1:needle" in by_glob.output
    assert ".claude/private/notes/deep.md:1:needle" in by_glob.output
    assert "worktrees" not in by_glob.output

    listed = _run(GlobTool(), GlobToolInput(pattern=".claude/private/**/*.md"), root)
    assert ".claude/private/PLAN.md" in listed.output
    assert ".claude/private/notes/deep.md" in listed.output


GYM_FILES = [
    "gym/experiments/s1-exp0-baseline.yaml",
    "gym/ladder/v1/suite.yaml",
    "gym/results/ladder/r27b-smoke.md",
    "gym/tasksets/v2.yaml",
    "gym/runs/r5/out/log.txt",
    "gym/recipes/r3/train.yaml",
]


def test_nothing_under_gym_is_skipped(tmp_path: Path, mode) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for f in GYM_FILES:
        p = tmp_path / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("needle\n")
    g = _run(GrepTool(), GrepToolInput(pattern="needle", limit=2000), tmp_path)
    for f in GYM_FILES:
        assert f"{f}:1:needle" in g.output, f
    assert "skipped" not in g.output
    listed = _run(GlobTool(), GlobToolInput(pattern="gym/**/*", limit=5000), tmp_path).output
    for f in GYM_FILES:
        assert f in listed, f


def test_every_tracked_gym_file_is_reached_in_this_repo(mode) -> None:
    """The real gym/ tree: every tracked file is listed from the repo root."""
    repo = Path(__file__).resolve().parents[1]
    tracked = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "gym"], capture_output=True, text=True,
    ).stdout.split()
    if not tracked:
        pytest.skip("no gym/ in this checkout")
    rg = tree_walk.ripgrep()
    walked = tree_walk.walk_files_rg(rg, repo, "gym/**/*") if rg else None
    if walked is None:
        walked = (p for p in tree_walk.walk(repo, "gym/**/*") if p.is_file())
    found = {p.relative_to(repo).as_posix() for p in walked}
    assert set(tracked) <= found, sorted(set(tracked) - found)[:10]
