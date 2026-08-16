"""A grant's description must be a property of the GRANT, not of the disk.

``describe()`` recovered "is this a directory grant?" with
``Path(value).is_dir()`` — a stat at RENDER time. So the same grant, unchanged,
described itself narrowly before the approved write created the directory and
as a subtree afterwards. Because a widening approval's target directory is
normally created BY the action being approved, the wrong branch was the normal
case.

Proven live by Beacon's consent walk (daemon fb73b28), both verbs, one session:

    until-restart        card and describes byte-identical (sha16 5131d72a21b6ff71)
    until-restart here   card:  "on exactly /home/will/beacon-walk-wide"
                         grant: "on anything under /home/will/beacon-walk-wide/"

The displayed extent was NARROWER than the grant created — consent under a
narrower description. Third instance of an extent being re-derived instead of
carried, and the first that drifts over TIME rather than across surfaces.

The regression that matters is ``test_describe_is_identical_before_and_after``.
Every other test here can pass while the defect is live.
"""

from __future__ import annotations

import builtins
import os
from pathlib import Path

import pytest

from prometheus.permissions.approval_queue import PendingAction, derive_grant
from prometheus.permissions.checker import Grant

pytestmark = pytest.mark.integration


def _action(target: str) -> PendingAction:
    return PendingAction(
        request_id="req1", tool_name="write_file",
        description="d", grant_file_path=target,
    )


# --------------------------------------------------------------------------- #
# The regression: identical across a filesystem change
# --------------------------------------------------------------------------- #


class TestDescriptionDoesNotVaryOverTime:
    def test_describe_is_identical_before_and_after(self, tmp_path):
        """THE test. The directory appears between the two calls.

        This is the live sequence: the operator is shown an extent, approves,
        the approved write creates the directory, and the grants list is then
        rendered from the same record. Before the fix these two strings
        differed, and the first was the narrower one.
        """
        target = tmp_path / "newdir" / "x.txt"
        g = derive_grant(_action(str(target)), widen=True)
        assert g is not None

        before = g.describe()
        assert not Path(g.value).exists(), "precondition: the directory is absent"

        # The approved action lands, creating the directory the grant covers.
        Path(g.value).mkdir(parents=True)
        assert Path(g.value).is_dir(), "precondition: the directory now exists"

        after = g.describe()
        assert before == after, (
            "the same grant described itself differently once the directory "
            f"existed:\n  before: {before!r}\n  after:  {after!r}"
        )
        assert "anything under" in before

    def test_narrow_grant_also_stable_across_a_filesystem_change(self, tmp_path):
        """The other direction — the one that was correct only by luck.

        A narrow grant whose value later becomes a directory used to flip to
        the subtree wording, misdescribing a file-shaped grant as a wide one.
        """
        target = tmp_path / "thing"
        g = derive_grant(_action(str(target)), widen=False)
        assert g is not None
        before = g.describe()
        Path(g.value).mkdir(parents=True)   # the value is now a real directory
        assert g.describe() == before
        assert "on exactly" in before


# --------------------------------------------------------------------------- #
# The rule, both directions, independent of the filesystem
# --------------------------------------------------------------------------- #


class TestWordingComesFromIntent:
    @pytest.mark.parametrize("make_parent", [False, True], ids=["absent", "present"])
    def test_widen_true_says_anything_under(self, tmp_path, make_parent):
        target = tmp_path / "d" / "x.txt"
        if make_parent:
            target.parent.mkdir(parents=True)
        g = derive_grant(_action(str(target)), widen=True)
        assert g is not None
        assert g.covers_subtree is True
        assert g.describe() == (
            f"write_file on anything under {tmp_path / 'd'}/ — until the daemon restarts"
        )

    def test_widen_true_wording_matches_whether_parent_exists(self, tmp_path):
        a = tmp_path / "one" / "x.txt"
        b = tmp_path / "two" / "x.txt"
        b.parent.mkdir(parents=True)
        ga = derive_grant(_action(str(a)), widen=True)
        gb = derive_grant(_action(str(b)), widen=True)
        assert ga is not None and gb is not None
        # Same shape of sentence; only the path differs.
        assert ga.describe().replace(str(a.parent), "P") == \
               gb.describe().replace(str(b.parent), "P")

    def test_widen_false_on_an_existing_directory_says_on_exactly(self, tmp_path):
        """Fix the rule, not the branch.

        Under the stat this rendered "anything under", turning a grant built to
        cover one target into a subtree in the operator's eyes.
        """
        d = tmp_path / "already-a-dir"
        d.mkdir()
        g = derive_grant(_action(str(d)), widen=False)
        assert g is not None
        assert g.covers_subtree is False
        assert g.describe() == f"write_file on exactly {d} — until the daemon restarts"

    def test_root_grants_are_subtrees_by_construction(self, tmp_path):
        g = derive_grant(_action(str(tmp_path / "x.txt")), root=str(tmp_path))
        assert g is not None
        assert g.covers_subtree is True
        assert "anything under" in g.describe()


# --------------------------------------------------------------------------- #
# Legacy rows: decided, not guessed, and not stat-ed
# --------------------------------------------------------------------------- #


class TestLegacyRowsWithNoRecordedIntent:
    """Config rows written before ``covers_subtree`` existed.

    Neither wording can be right for all of them, and the two errors are not
    symmetric: understating a subtree grant is consent under a narrower
    description, the defect this work exists to remove. So describe the
    MATCHING RULE, which is knowable from the record alone —
    ``matches()`` resolves the candidate and calls ``relative_to(value)``, so
    the grant covers ``value`` and everything beneath it, whatever the intent
    was. For a file-shaped value "anything under it" is empty and the sentence
    is still true.
    """

    def test_absent_field_rehydrates_as_unknown_not_false(self):
        g = Grant.from_config_dict(
            {"kind": "path_prefix", "value": "/tmp/legacy", "tool": "write_file"})
        assert g is not None
        assert g.covers_subtree is None, (
            "defaulting to False would silently relabel every pre-existing "
            "directory grant as an exact-file one"
        )

    def test_unknown_intent_states_the_matching_rule(self):
        g = Grant(kind="path_prefix", value="/tmp/legacy", tool_name="write_file",
                  scope="persistent")
        assert g.describe() == (
            "write_file on /tmp/legacy and anything under it — permanently, until revoked"
        )

    def test_unknown_intent_wording_is_also_filesystem_independent(self, tmp_path):
        d = tmp_path / "legacy-dir"
        g = Grant(kind="path_prefix", value=str(d), tool_name="write_file")
        before = g.describe()
        d.mkdir()
        assert g.describe() == before

    def test_recorded_intent_survives_a_persistence_round_trip(self, tmp_path):
        for widen in (True, False):
            src = derive_grant(_action(str(tmp_path / "d" / "x.txt")), widen=widen)
            assert src is not None
            back = Grant.from_config_dict(src.to_config_dict())
            assert back is not None
            assert back.covers_subtree is src.covers_subtree
            # scope differs by design (config rows are persistent), so compare
            # the half this change owns.
            assert back.describe().split(" — ")[0] == src.describe().split(" — ")[0]

    def test_unknown_intent_is_not_written_to_config(self):
        g = Grant(kind="path_prefix", value="/tmp/x", tool_name="write_file")
        assert "covers_subtree" not in g.to_config_dict()


# --------------------------------------------------------------------------- #
# Assert the absence of the stat. Do not assume it.
# --------------------------------------------------------------------------- #


class TestDescribeTouchesNoFilesystem:
    @pytest.mark.parametrize("covers", [True, False, None])
    def test_describe_makes_no_filesystem_call(self, monkeypatch, covers):
        """Trip every syscall describe() could plausibly reach.

        A comment saying "no stat here" is not enforcement; the next edit can
        reintroduce one and every other test in this file would still pass.
        """
        called: list[str] = []
        # Scoped to THIS grant's value. A blanket tripwire on Path.exists also
        # trips pytest's own tmp-dir cleanup at interpreter exit, which turns a
        # clean failure into a crash and hides which test caught what — seen
        # while verifying M-RESTORE-STAT.
        sentinel = "/tmp/describe-tripwire-target"

        def tripwire(name, orig):
            def _wrapped(*a, **k):
                if any(sentinel in str(x) for x in a):
                    called.append(name)
                    raise AssertionError(f"describe() called {name}{a!r}")
                return orig(*a, **k)
            return _wrapped

        for mod, name in [
            (os, "stat"), (os, "lstat"), (os, "listdir"), (os, "scandir"),
            (os.path, "exists"), (os.path, "isdir"), (os.path, "isfile"),
        ]:
            monkeypatch.setattr(mod, name, tripwire(f"os.{name}", getattr(mod, name)))
        for name in ("is_dir", "exists", "stat", "resolve"):
            monkeypatch.setattr(Path, name, tripwire(f"Path.{name}", getattr(Path, name)))
        monkeypatch.setattr(builtins, "open", tripwire("open", builtins.open))

        g = Grant(kind="path_prefix", value=sentinel,
                  tool_name="write_file", covers_subtree=covers)
        out = g.describe()
        assert isinstance(out, str) and out
        assert called == []

    def test_the_tripwire_itself_fires(self, monkeypatch):
        """Prove the guard can FAIL. An inert tripwire proves nothing.

        Not a lambda that raises unconditionally — that would only prove
        monkeypatch works. This installs the real tripwire and then makes
        describe() reach a stat, exactly as M-RESTORE-STAT does.
        """
        sentinel = "/tmp/describe-tripwire-target"
        tripped: list[str] = []
        real = Path.is_dir

        def _wrapped(self, *a, **k):
            if sentinel in str(self):
                tripped.append("Path.is_dir")
                raise AssertionError("describe() called Path.is_dir")
            return real(self, *a, **k)

        monkeypatch.setattr(Path, "is_dir", _wrapped)
        with pytest.raises(AssertionError, match="Path.is_dir"):
            Path(sentinel).is_dir()
        assert tripped == ["Path.is_dir"]

    def test_other_kinds_also_touch_nothing(self, monkeypatch):
        monkeypatch.setattr(Path, "is_dir",
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError("stat!")))
        assert "EVERY use of" in Grant(
            kind="tool", value="", tool_name="bash").describe()
        assert "starting with" in Grant(
            kind="command_prefix", value="git ", tool_name="bash").describe()
