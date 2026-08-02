"""GrepTool — absolute file_glob handling (live bug: NotImplementedError).

The model frequently passes absolute glob patterns; pathlib's Path.glob
rejects non-relative patterns outright. 24 occurrences in one day of gym
runs (2026-06-10) before the fix.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from prometheus.tools.builtin.grep import GrepTool, GrepToolInput, _split_absolute_glob
from prometheus.tools.base import ToolExecutionContext


def _run(tmp_path: Path, **kw):
    tool = GrepTool()
    return asyncio.run(
        tool.execute(
            GrepToolInput(**kw),
            ToolExecutionContext(cwd=tmp_path, metadata={}),
        )
    )


class TestSplitAbsoluteGlob:

    def test_relative_passthrough(self, tmp_path):
        assert _split_absolute_glob(tmp_path, "**/*.py") == (tmp_path, "**/*.py")

    def test_absolute_with_wildcard_tail(self, tmp_path):
        root, pat = _split_absolute_glob(Path("/ignored"), f"{tmp_path}/src/*.py")
        assert root == tmp_path / "src"
        assert pat == "*.py"

    def test_absolute_wildcard_mid_path(self, tmp_path):
        root, pat = _split_absolute_glob(Path("/ignored"), f"{tmp_path}/*/x.py")
        assert root == tmp_path
        assert pat == str(Path("*") / "x.py")

    def test_absolute_concrete_path(self, tmp_path):
        root, pat = _split_absolute_glob(Path("/ignored"), f"{tmp_path}/a.py")
        assert root == tmp_path
        assert pat == "a.py"


class TestGrepAbsoluteGlob:

    def test_absolute_glob_no_longer_raises(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "x.py").write_text("# TODO: fix this\n")
        (tmp_path / "src" / "y.py").write_text("# all good\n")
        # the live failure shape: absolute glob, root defaulting to cwd
        result = _run(tmp_path, pattern="TODO", file_glob=f"{tmp_path}/src/*.py")
        assert not result.is_error
        assert "x.py" in result.output
        assert "y.py" not in result.output

    def test_relative_glob_unchanged(self, tmp_path):
        (tmp_path / "a.txt").write_text("needle here\n")
        result = _run(tmp_path, pattern="needle", file_glob="*.txt")
        assert "a.txt:1:needle here" in result.output

    def test_absolute_glob_outside_root_renders_full_path(self, tmp_path):
        # anchor disjoint from cwd-root: relative_to would ValueError;
        # the match must render with its full path instead of crashing.
        other = tmp_path / "other"
        other.mkdir()
        (other / "z.txt").write_text("needle\n")
        result = _run(
            tmp_path / "elsewhere_nonexistent",
            pattern="needle",
            file_glob=f"{other}/*.txt",
        )
        assert not result.is_error
        assert "z.txt" in result.output

    def test_example_call_uses_real_param_names(self):
        # the shipped example taught the model a nonexistent 'path' param
        schema_params = set(GrepToolInput.model_json_schema()["properties"])
        assert set(GrepTool.example_call) <= schema_params
