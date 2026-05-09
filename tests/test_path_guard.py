"""path_guard — write-boundary helper used by autonomous components."""

from __future__ import annotations

from pathlib import Path

import pytest

from prometheus.security.path_guard import (
    assert_path_under_roots,
    is_path_under_roots,
)


class TestAssertPathUnderRoots:
    def test_path_inside_root_returns_resolved(self, tmp_path):
        target = tmp_path / "subdir" / "file.md"
        target.parent.mkdir()
        target.write_text("hi")
        result = assert_path_under_roots(target, [tmp_path])
        assert result == target.resolve()

    def test_path_equal_to_root_is_allowed(self, tmp_path):
        result = assert_path_under_roots(tmp_path, [tmp_path])
        assert result == tmp_path.resolve()

    def test_path_outside_all_roots_raises(self, tmp_path):
        outside = tmp_path / "x"
        with pytest.raises(ValueError, match="not under any allowed root"):
            assert_path_under_roots(outside, [tmp_path / "other_root"])

    def test_traversal_resolved_before_check(self, tmp_path):
        """``../../etc`` that escapes the root is rejected even when
        the literal candidate string starts inside the root."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        # Construct a candidate that LITERALLY starts under allowed/
        # but resolves outside of it via ..
        sneaky = str(allowed) + "/../escape.md"
        with pytest.raises(ValueError):
            assert_path_under_roots(sneaky, [allowed])

    def test_empty_roots_raises(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            assert_path_under_roots(tmp_path, [])

    def test_user_home_expansion(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        # ~/.prometheus/MEMORY.md should be under [~/.prometheus]
        (tmp_path / ".prometheus").mkdir()
        result = assert_path_under_roots(
            "~/.prometheus/MEMORY.md",
            ["~/.prometheus"],
        )
        assert "MEMORY.md" in str(result)


class TestIsPathUnderRoots:
    def test_returns_true_for_allowed(self, tmp_path):
        assert is_path_under_roots(tmp_path / "x", [tmp_path]) is True

    def test_returns_false_for_disallowed(self, tmp_path):
        assert is_path_under_roots(
            tmp_path / "x",
            [tmp_path / "other_root"],
        ) is False

    def test_returns_false_for_empty_roots(self, tmp_path):
        assert is_path_under_roots(tmp_path, []) is False
