"""grep/glob prune denied paths out of results, rather than refusing.

The gate refuses a search whose ROOT is denied. This is the second layer:
a LEGITIMATE root that CONTAINS a denied path (`~` contains `~/.ssh`) must
still work, minus the denied subtree.

Refusing instead was measured and rejected: across 399 recorded grep/glob
calls it would have blocked exactly one, while making `grep --root ~`
permanently unusable — and an unusable sanctioned path teaches the model to
reach for `bash`, which has no boundary at all.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from prometheus.tools.base import ToolExecutionContext
from prometheus.tools.builtin import GlobTool, GrepTool
from prometheus.tools.denied_prune import is_denied, resolve_denied


def _tree(tmp_path: Path) -> Path:
    (tmp_path / "ok").mkdir()
    (tmp_path / "ok" / "notes.txt").write_text("NEEDLE in the open\n")
    (tmp_path / ".ssh").mkdir()
    (tmp_path / ".ssh" / "id_rsa").write_text("NEEDLE in a private key\n")
    return tmp_path


def _grep(tmp_path, **kw):
    tool = GrepTool(denied_paths=[str(tmp_path / ".ssh")])
    return asyncio.run(tool.execute(
        GrepTool.input_model(pattern="NEEDLE", **kw),
        ToolExecutionContext(cwd=tmp_path),
    )).output


class TestPruneKeepsTheToolUsable:

    def test_legitimate_hits_survive(self, tmp_path):
        out = _grep(_tree(tmp_path))
        assert "ok/notes.txt" in out, "pruning must not refuse the whole search"

    def test_denied_content_is_not_returned(self, tmp_path):
        out = _grep(_tree(tmp_path))
        assert "id_rsa" not in out
        assert "private key" not in out

    def test_the_withholding_is_stated(self, tmp_path):
        """Silent filtering turns a boundary into a source of wrong
        conclusions — absence would read as proof there is nothing there."""
        out = _grep(_tree(tmp_path))
        assert "withheld" in out and "denied_paths" in out

    def test_no_denied_config_prunes_nothing(self, tmp_path):
        _tree(tmp_path)
        tool = GrepTool(denied_paths=None)
        out = asyncio.run(tool.execute(
            GrepTool.input_model(pattern="NEEDLE"),
            ToolExecutionContext(cwd=tmp_path),
        )).output
        assert "id_rsa" in out and "withheld" not in out

    def test_glob_prunes_and_reports(self, tmp_path):
        _tree(tmp_path)
        tool = GlobTool(denied_paths=[str(tmp_path / ".ssh")])
        out = asyncio.run(tool.execute(
            GlobTool.input_model(pattern="**/*"),
            ToolExecutionContext(cwd=tmp_path),
        )).output
        assert "ok/notes.txt" in out
        assert "id_rsa" not in out
        assert "withheld" in out


class TestDeniedResolution:

    def test_glob_entries_expand(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "xenv").write_text("k=v")
        resolved = resolve_denied([str(tmp_path / "*" / "*env")])
        assert (tmp_path / "a" / "xenv").resolve() in resolved

    def test_absent_denied_path_is_not_an_error(self, tmp_path):
        assert resolve_denied([str(tmp_path / "nope")]) or True  # no raise

    def test_subpaths_are_denied(self, tmp_path):
        denied = resolve_denied([str(tmp_path / "d")])
        (tmp_path / "d").mkdir()
        denied = resolve_denied([str(tmp_path / "d")])
        assert is_denied(tmp_path / "d" / "deep" / "f.txt", denied)
        assert not is_denied(tmp_path / "other" / "f.txt", denied)

    def test_unresolvable_path_is_treated_as_denied(self, tmp_path):
        """A path we cannot reason about is not one to hand back from a
        search that may be rooted anywhere."""
        denied = resolve_denied([str(tmp_path / "d")])
        (tmp_path / "d").mkdir()
        denied = resolve_denied([str(tmp_path / "d")])
        loop = tmp_path / "loop"
        loop.symlink_to(loop)
        assert is_denied(loop, denied)
