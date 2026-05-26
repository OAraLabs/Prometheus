"""Tests for SkillCreator filename derivation (PR #20).

Pre-PR-#20, ``SkillCreator.maybe_create`` slugified the raw user message
(``task_description``) into a filename. The LLM was correctly emitting
``name: <kebab-case>`` in YAML frontmatter, but the code never read it.
Result: pathological filenames in the shape
``<long-run-on-user-message-truncated-mid-word>-.md`` (the trailing
dash a separate strip-before-truncate bug in ``_slugify``) even though
the file's frontmatter contained a clean ``name: <kebab-case>``.

These tests assert the post-fix invariants:

1. Filename derives from the LLM's ``name:`` frontmatter, not from
   ``task_description``.
2. Missing/empty/unslugifiable ``name:`` → no file is written and a
   ``silent_failure`` row is recorded with
   ``subsystem="skill_creator"``, ``operation="extract_name"``.
3. The slug itself is filesystem-safe and bounded.

See ``~/PROMETHEUS-MEMORY-DIAGNOSIS-2026-05-26.md`` Phase 2 for the
forensic walkthrough of the original bug.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from prometheus.learning.skill_creator import (
    SkillCreator,
    SkillNameExtractionError,
    _slugify,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _good_skill_content(name: str = "refactor-auth-module") -> str:
    """Return a valid SKILL.md body with the given ``name:`` in frontmatter."""
    return (
        f"---\n"
        f"name: {name}\n"
        f"description: Refactor authentication module for clarity.\n"
        f"---\n"
        f"\n"
        f"# Refactor Auth Module\n"
        f"\n"
        f"## When to use\n"
        f"When the auth code is messy.\n"
        f"\n"
        f"## Steps\n"
        f"1. Read it.\n"
        f"2. Refactor it.\n"
    )


def _make_creator(
    tmp_path: Path,
    llm_response: str | None,
    *,
    telemetry: MagicMock | None = None,
) -> SkillCreator:
    """Build a SkillCreator whose LLM call returns ``llm_response``.

    Stubs ``_envelope.call`` so no real provider is touched. The provider
    handle itself is a MagicMock — ignored by the envelope stub.
    """
    creator = SkillCreator(
        provider=MagicMock(),
        model="test-model",
        auto_dir=tmp_path,
        telemetry=telemetry,
    )
    creator._envelope.call = AsyncMock(return_value=llm_response)
    return creator


def _trivial_trace(n: int = 5) -> list[dict]:
    """Tool trace just long enough to meet ``_MIN_TOOL_CALLS`` (3)."""
    return [
        {"tool_name": "bash", "arguments": {"command": f"echo {i}"}, "result": "ok"}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSkillFilenameUsesExtractedName:
    """The PR's headline invariant: filename comes from LLM ``name:``."""

    def test_filename_comes_from_frontmatter_not_user_input(self, tmp_path: Path) -> None:
        """User input is a noisy run-on; LLM ``name:`` is clean → clean filename wins."""
        creator = _make_creator(
            tmp_path,
            _good_skill_content("refactor-auth-module"),
        )

        path = asyncio.run(
            creator.maybe_create(
                task_description=(
                    "please go ahead and take down the legacy dashboard that's old "
                    "and was before all this other infrastructure"
                ),
                tool_trace=_trivial_trace(),
            )
        )

        assert path is not None
        assert path.name == "refactor-auth-module.md"
        # And explicitly NOT a slug derived from the user message:
        assert "please-go-ahead" not in path.name
        assert "legacy-dashboard" not in path.name
        assert "infrastructure" not in path.name

    def test_filename_uses_name_even_when_userinput_would_have_been_clean(
        self, tmp_path: Path
    ) -> None:
        """Even if ``task_description`` slugifies fine, ``name:`` still wins."""
        creator = _make_creator(
            tmp_path,
            _good_skill_content("fizz-buzz-the-thing"),
        )

        path = asyncio.run(
            creator.maybe_create(
                task_description="add a logging helper",
                tool_trace=_trivial_trace(),
            )
        )

        assert path is not None
        assert path.name == "fizz-buzz-the-thing.md"

    def test_content_preserved_verbatim_in_file(self, tmp_path: Path) -> None:
        """File body matches the LLM output (only stripped + newline-terminated)."""
        body = _good_skill_content("preserved-content")
        creator = _make_creator(tmp_path, body)
        path = asyncio.run(
            creator.maybe_create("anything", _trivial_trace())
        )
        assert path is not None
        assert path.read_text(encoding="utf-8") == body.strip() + "\n"


class TestSkillCreationSkippedOnMalformedLLMOutput:
    """Missing / empty / unusable ``name:`` → no file, telemetry recorded."""

    def test_no_file_written_when_llm_returns_no_frontmatter(
        self, tmp_path: Path
    ) -> None:
        bad = "# Just a heading\nSome content but no frontmatter at all."
        creator = _make_creator(tmp_path, bad)

        result = asyncio.run(creator.maybe_create("anything", _trivial_trace()))

        assert result is None
        assert list(tmp_path.iterdir()) == []

    def test_no_file_written_when_name_field_is_empty(self, tmp_path: Path) -> None:
        bad = "---\nname:\ndescription: foo\n---\n# x"
        creator = _make_creator(tmp_path, bad)
        result = asyncio.run(creator.maybe_create("anything", _trivial_trace()))
        assert result is None
        assert list(tmp_path.iterdir()) == []

    def test_no_file_written_when_name_field_is_only_quotes(
        self, tmp_path: Path
    ) -> None:
        """``name: ""`` → strip → empty → treated as missing."""
        bad = '---\nname: ""\ndescription: foo\n---\n# x'
        creator = _make_creator(tmp_path, bad)
        result = asyncio.run(creator.maybe_create("anything", _trivial_trace()))
        assert result is None
        assert list(tmp_path.iterdir()) == []

    def test_no_file_written_when_name_is_unslugifiable(
        self, tmp_path: Path
    ) -> None:
        """``name: "!!!"`` slugifies to empty string → no file."""
        bad = '---\nname: "!!!"\ndescription: foo\n---\n# x'
        creator = _make_creator(tmp_path, bad)
        result = asyncio.run(creator.maybe_create("anything", _trivial_trace()))
        assert result is None
        assert list(tmp_path.iterdir()) == []

    def test_silent_failure_recorded_on_missing_name(self, tmp_path: Path) -> None:
        telemetry = MagicMock()
        creator = _make_creator(tmp_path, "no frontmatter", telemetry=telemetry)

        asyncio.run(creator.maybe_create("anything", _trivial_trace()))

        telemetry.record_silent_failure.assert_called_once()
        kwargs = telemetry.record_silent_failure.call_args.kwargs
        assert kwargs["subsystem"] == "skill_creator"
        assert kwargs["operation"] == "extract_name"
        assert isinstance(kwargs["exc"], SkillNameExtractionError)
        # Context payload should preview both the LLM output and the user
        # input so the failure is debuggable from the silent_failures row.
        assert "content_preview" in kwargs["context"]
        assert "task_description" in kwargs["context"]

    def test_silent_failure_recorded_on_unslugifiable_name(
        self, tmp_path: Path
    ) -> None:
        telemetry = MagicMock()
        bad = '---\nname: "!!!"\ndescription: foo\n---\n# x'
        creator = _make_creator(tmp_path, bad, telemetry=telemetry)

        asyncio.run(creator.maybe_create("anything", _trivial_trace()))

        telemetry.record_silent_failure.assert_called_once()
        kwargs = telemetry.record_silent_failure.call_args.kwargs
        assert kwargs["operation"] == "extract_name"
        # The raw (pre-slugify) name should be present so the operator can
        # see exactly what the model emitted.
        assert kwargs["context"].get("name_raw") == "!!!"

    def test_no_telemetry_no_crash_on_missing_name(self, tmp_path: Path) -> None:
        """When telemetry is None, the failure path still completes cleanly."""
        creator = _make_creator(tmp_path, "no frontmatter", telemetry=None)
        result = asyncio.run(creator.maybe_create("anything", _trivial_trace()))
        assert result is None  # no crash, just a None return


class TestSlugifySafety:
    """The ``_slugify`` helper itself: bounded length, filesystem-safe, no trailing dash."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("simple", "simple"),
            ("Multi Word Name", "multi-word-name"),
            ("With Punctuation!!! And/Slashes", "with-punctuation-and-slashes"),
            ("", ""),
            ("!!!", ""),
            ("---leading", "leading"),
            ("trailing---", "trailing"),
            ("inner   spaces", "inner-spaces"),
            # Already-slugified input is idempotent.
            ("already-slugified", "already-slugified"),
        ],
    )
    def test_basic_cases(self, raw: str, expected: str) -> None:
        assert _slugify(raw) == expected

    def test_max_length_64(self) -> None:
        assert _slugify("a" * 200) == "a" * 64

    def test_truncation_never_leaves_trailing_dash(self) -> None:
        """If the 64th char lands on or right after a dash, rstrip removes it.

        The pre-PR-#20 ``slug.strip("-")[:60]`` shape stripped BEFORE the
        truncation, so a truncation that landed in the middle of
        ``"-"`` (the regex sub result for a punctuation/space) left a
        bare trailing dash. The new ``[:64].rstrip("-")`` order catches it.
        """
        raw = "this is a very long task description that will truncate mid word here for sure"
        slug = _slugify(raw)
        assert len(slug) <= 64
        assert not slug.endswith("-")

    def test_pathological_input_shape_reproduces_correctly(self) -> None:
        """Sanity: an input of the shape that produced the original
        pathological filename — a long run-on user message — now produces
        a 64-char slug with no trailing dash."""
        raw = (
            "please go ahead and take down the legacy dashboard that's old "
            "and was before all this other stuff"
        )
        slug = _slugify(raw)
        assert len(slug) <= 64
        assert not slug.endswith("-")
        # Sanity check on the actual produced slug — should start the same
        # but should not have a trailing dash like the on-disk evidence did.
        assert slug.startswith("please-go-ahead")
