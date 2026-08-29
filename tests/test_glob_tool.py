"""GlobTool — absolute-pattern handling (the #134 defect's unfixed sibling).

grep got _split_absolute_glob in #134 after 24 NotImplementedError kills in
one day of gym runs; glob had the identical `root.glob(pattern)` shape and
no fix. Reproduced before fixing: an absolute pattern raised
``NotImplementedError: Non-relative patterns are unsupported`` and killed
the call.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from prometheus.tools.builtin.glob import GlobTool, GlobToolInput


def _run(tool: GlobTool, args: GlobToolInput, cwd: Path):
    return asyncio.run(tool.execute(args, SimpleNamespace(cwd=cwd)))


def test_relative_pattern_still_works(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.txt").write_text("x")
    result = _run(GlobTool(), GlobToolInput(pattern="*.py"), tmp_path)
    assert result.output.strip() == "a.py"


def test_absolute_pattern_does_not_raise(tmp_path: Path) -> None:
    # The kill shape: pattern carries the absolute directory itself.
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.py").write_text("x")
    result = _run(
        GlobTool(), GlobToolInput(pattern=f"{tmp_path}/*.py"), Path("/ignored-cwd")
    )
    assert not result.is_error
    assert "a.py" in result.output and "b.py" in result.output


def test_absolute_pattern_with_recursive_glob(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "deep.py").write_text("x")
    result = _run(
        GlobTool(), GlobToolInput(pattern=f"{tmp_path}/**/*.py"), Path("/ignored-cwd")
    )
    assert not result.is_error
    assert "deep.py" in result.output


def test_absolute_pattern_no_matches_is_a_result_not_a_crash(tmp_path: Path) -> None:
    result = _run(
        GlobTool(), GlobToolInput(pattern=f"{tmp_path}/*.nope"), Path("/ignored-cwd")
    )
    assert not result.is_error
    assert "(no matches)" in result.output
