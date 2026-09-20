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

    def test_glob_entries_deny_the_paths_they_match(self, tmp_path):
        """A glob entry is kept AS A PATTERN and matched with fnmatch (the
        gate's semantics, where ``*`` spans ``/``), not expanded against the
        filesystem with ``Path.glob`` (where ``*`` is one component). The old
        test asserted the expansion — that expansion was the defect: the shipped
        ``/*/.ssh`` floor expanded to nothing. Assert the outcome instead, so the
        contract holds whichever way the matching is implemented.
        """
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "xenv").write_text("k=v")
        entry = str(tmp_path / "*" / "*env")
        denied = resolve_denied([entry])
        # The pattern is preserved verbatim (not expanded to concrete paths)...
        assert entry in denied
        # ...and still denies a file that matches it, plus its subtree.
        assert is_denied(tmp_path / "a" / "xenv", denied)
        # Created for the same reason as in test_subpaths_are_denied: an
        # absent path is denied because it cannot be verified, which would
        # fail this assertion while the PATTERN logic under test was correct.
        (tmp_path / "a" / "other.txt").write_text("x")
        assert not is_denied(tmp_path / "a" / "other.txt", denied)

    def test_a_star_glob_spans_path_separators_like_the_gate(self, tmp_path):
        """The exact #214 guarantee the expansion broke: a ``*`` entry must
        match a path MORE THAN ONE component below the anchor, because the gate's
        fnmatch does and the two layers must not disagree."""
        deep = tmp_path / "home" / "will" / ".ssh"
        deep.mkdir(parents=True)
        (deep / "id_rsa").write_text("PRIVATE KEY")
        denied = resolve_denied([str(tmp_path / "*" / ".ssh")])
        # `*` spans `home/will` — pathlib's glob would NOT, and returned () here.
        assert is_denied(deep / "id_rsa", denied), (
            "a /*/.ssh-style entry must deny a key two components down; this is "
            "the matcher drift that left the credential floor inert"
        )

    def test_absent_denied_path_is_not_an_error(self, tmp_path):
        assert resolve_denied([str(tmp_path / "nope")]) or True  # no raise

    def test_subpaths_are_denied(self, tmp_path):
        """Both paths are CREATED, and that is load-bearing now.

        They used not to exist, which made both assertions prove something
        other than what they say. Since the guard fails closed on anything it
        cannot verify, a nonexistent path is denied for THAT reason — so the
        positive assertion would pass even against a matcher that had stopped
        matching subpaths, and the negative one would fail against a matcher
        that was working perfectly. Existing files put the matching rule back
        under test.
        """
        denied = resolve_denied([str(tmp_path / "d")])
        (tmp_path / "d").mkdir()
        denied = resolve_denied([str(tmp_path / "d")])
        (tmp_path / "d" / "deep").mkdir()
        (tmp_path / "d" / "deep" / "f.txt").write_text("x")
        (tmp_path / "other").mkdir()
        (tmp_path / "other" / "f.txt").write_text("x")
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

    def test_a_BROKEN_SYMLINK_is_treated_as_denied(self, tmp_path):
        """The loop case above was the only unresolvable input covered, and it
        was the only one that happened to RAISE.

        ``matches_any_denied`` promises to fail closed on "a broken symlink or
        a loop". It delivered that for a loop only by accident: non-strict
        ``Path.resolve()`` raises for a symlink cycle and returns the path
        unchanged for a dangling one, so the ``except`` never fired here and
        the guard fell through to the membership test. Measured on 3.12.3
        before the fix: ``is_denied`` returned False.
        """
        denied = resolve_denied([str(tmp_path / "d")])
        (tmp_path / "d").mkdir()
        denied = resolve_denied([str(tmp_path / "d")])
        broken = tmp_path / "broken"
        broken.symlink_to(tmp_path / "target-that-does-not-exist")
        assert is_denied(broken, denied), (
            "a dangling symlink resolved to a path nobody can verify and was "
            "handed back as allowed"
        )

    def test_a_NONEXISTENT_path_is_treated_as_denied(self, tmp_path):
        """The third unresolvable shape, and the one with no symlink in it.

        Non-strict resolve() returns a nonexistent path unchanged, so the
        guard compared a path that does not exist against the deny list and
        answered on its SPELLING. Every production caller enumerates the
        filesystem (glob/grep results), so a path arriving here that does not
        exist means reality disagrees with the caller -- which is precisely
        when a security boundary should refuse rather than guess.
        """
        denied = resolve_denied([str(tmp_path / "d")])
        (tmp_path / "d").mkdir()
        denied = resolve_denied([str(tmp_path / "d")])
        assert is_denied(tmp_path / "vanished.txt", denied), (
            "a path that does not exist was verified as allowed"
        )


class TestTheShippedFloorActuallyWithholdsCredentials:
    """THE #214 GUARANTEE, end to end, with the SHIPPED denied_paths.

    The audit measured this on a real box: with the shipped floor, a grep rooted
    at the home directory for 'PRIVATE KEY' returned the id_rsa lines and emitted
    NO '[N paths withheld]' note — the model read the result as complete. The
    cause was matcher drift: the gate denied `/*/.ssh` via fnmatch but this layer
    expanded it via Path.glob to nothing.

    These tests drive the REAL GrepTool/GlobTool with SHIPPED_DENIED_PATHS over a
    fake home (a `.ssh/id_rsa` under a tmp anchor), asserting the key is withheld
    AND the withholding is stated. A fake home rather than the operator's real
    `~`: the shipped globs are `/*/.ssh` and fnmatch's `*` spans `/`, so a
    `.ssh` any depth under `/` matches — the same property the drift broke.
    """

    def _fake_home_with_key(self, tmp_path: Path) -> Path:
        # A `.ssh/id_rsa` several components under the anchor, exactly the shape
        # `/*/.ssh` must catch and Path.glob('*' one component) could not.
        home = tmp_path / "home" / "operator"
        (home / "projects").mkdir(parents=True)
        (home / "projects" / "ok.txt").write_text("NEEDLE in open source\n")
        (home / ".ssh").mkdir()
        (home / ".ssh" / "id_rsa").write_text("NEEDLE PRIVATE KEY material\n")
        return home

    def test_grep_withhelds_the_private_key_under_the_shipped_floor(self, tmp_path):
        from prometheus.config.shipped_defaults import SHIPPED_DENIED_PATHS

        home = self._fake_home_with_key(tmp_path)
        tool = GrepTool(denied_paths=list(SHIPPED_DENIED_PATHS))
        out = asyncio.run(tool.execute(
            GrepTool.input_model(pattern="NEEDLE", root=str(home)),
            ToolExecutionContext(cwd=tmp_path),
        )).output

        # The legitimate hit survives...
        assert "ok.txt" in out, "pruning must not refuse the whole search"
        # ...the credential content does not...
        assert "id_rsa" not in out, (
            "the private key was returned despite /*/.ssh being in the shipped "
            "floor — the matcher drift is back"
        )
        assert "PRIVATE KEY material" not in out
        # ...and the withholding is STATED, not silent.
        assert "withheld" in out and "denied_paths" in out, (
            "a filtered result with no note reads as complete"
        )

    def test_glob_withhelds_the_ssh_dir_under_the_shipped_floor(self, tmp_path):
        from prometheus.config.shipped_defaults import SHIPPED_DENIED_PATHS

        home = self._fake_home_with_key(tmp_path)
        tool = GlobTool(denied_paths=list(SHIPPED_DENIED_PATHS))
        out = asyncio.run(tool.execute(
            GlobTool.input_model(pattern="**/*", root=str(home)),
            ToolExecutionContext(cwd=tmp_path),
        )).output

        assert "ok.txt" in out
        assert "id_rsa" not in out, ".ssh contents leaked past the shipped floor"
        assert "withheld" in out

    def test_the_gate_and_the_prune_layer_agree(self, tmp_path):
        """The two layers must give the SAME answer on the same path — that
        symmetry is the whole fix. Assert it directly rather than trusting that
        both happen to call one matcher today."""
        from prometheus.config.shipped_defaults import SHIPPED_DENIED_PATHS
        from prometheus.permissions.checker import SecurityGate

        home = self._fake_home_with_key(tmp_path)
        key = home / ".ssh" / "id_rsa"

        denied = resolve_denied(list(SHIPPED_DENIED_PATHS))
        prune_says = is_denied(key, denied)

        gate = SecurityGate(denied_paths=list(SHIPPED_DENIED_PATHS))
        gate_says = gate.evaluate("read_file", file_path=str(key)).action == "DENY"

        assert prune_says is True and gate_says is True, (
            f"the layers disagree on {key}: prune={prune_says} gate={gate_says}"
        )
