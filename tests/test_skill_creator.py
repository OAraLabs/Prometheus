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


# ---------------------------------------------------------------------------
# Quality gate (2026-08 sprint) — Stage 0 deterministic checks + Stage 1
# LLM opt-out. The blocked shapes below replicate the 2026-08-03 junk crop
# (survey: audits/20260803T215431Z-skillcreator-quality-gate-survey.md):
# a trace with a failed call, an all-calls-succeeded negative lookup (a
# locate that found nothing), a capability test ("testing to see if you
# can get this .md?"), and near-duplicates minted minutes apart (three
# skills from one release-check question).
# ---------------------------------------------------------------------------


def _trace(n: int = 5, *, errors: frozenset[int] = frozenset()) -> list[dict]:
    """Tool trace in the exact shape run_async hands to post-task hooks."""
    return [
        {
            "tool_name": "bash",
            "arguments": {"command": f"step {i}"},
            "result": "exit 1" if i in errors else "ok",
            "is_error": i in errors,
        }
        for i in range(n)
    ]


# Phrases that mark a junk OUTCOME. They reach the prompt only through
# ``final_text`` / ``task_description`` — none of them appear in
# ``_GENERATION_PROMPT`` itself (guarded by
# test_junk_markers_stay_out_of_the_template below), so the fake model's
# decisions are driven by the turn data, exactly like a real model's.
_JUNK_OUTCOME_MARKERS = (
    "couldn't find",
    "could not find",
    "not released",
    "isn't released",
    "testing to see",
)


class _CompetentModel:
    """Stands in for a well-behaved skill-generation model.

    Declines junk turns (negative/failed/test outcomes) IF AND ONLY IF the
    prompt both authorizes ``SKIP:`` and carries the outcome evidence. Two
    tripwires follow from that conditionality:

    - Remove the SKIP opt-out from ``_GENERATION_PROMPT`` → this emits a
      full skill for the junk turns → the blocking tests go red.
    - Stop passing ``final_text`` into the prompt → the outcome evidence
      vanishes → same result.

    Both are deliberate: the failed-lookup class (every call exits 0,
    only the ANSWER is negative) has no deterministic catch, so the prompt
    plumbing IS the gate.
    """

    def __init__(self, skill_name: str = "generated-skill") -> None:
        self.skill_name = skill_name
        self.prompts: list[str] = []

    async def __call__(self, **kwargs) -> str:
        prompt = kwargs["prompt"]
        self.prompts.append(prompt)
        low = prompt.lower()
        if "skip:" in low and any(m in low for m in _JUNK_OUTCOME_MARKERS):
            return "SKIP: not a reusable procedure"
        return _good_skill_content(self.skill_name)


def _creator_with_model(
    tmp_path: Path, model: _CompetentModel
) -> SkillCreator:
    creator = SkillCreator(
        provider=MagicMock(), model="test-model", auto_dir=tmp_path,
    )
    # Bound method, not AsyncMock(side_effect=...): mock does not await an
    # instance whose __call__ is async, it would hand back the coroutine.
    creator._envelope.call = model.__call__
    return creator


class TestStage0DeterministicGate:
    """Pre-LLM checks: bounded call count, zero failed calls, no dupes."""

    def test_failed_call_in_trace_skips_before_llm(self, tmp_path: Path) -> None:
        """One errored call disqualifies the turn — and costs no LLM call.

        The wordpress-feed-migration shape: 2026-08-03 03:16 authored a
        skill from a trace whose third bash call failed.
        """
        creator = _make_creator(tmp_path, _good_skill_content())

        result = asyncio.run(
            creator.maybe_create(
                "testing to see if you can get this .md?",
                _trace(5, errors=frozenset({2})),
            )
        )

        assert result is None
        creator._envelope.call.assert_not_awaited()
        assert list(tmp_path.iterdir()) == []

    def test_oversized_trace_skips_before_llm(self, tmp_path: Path) -> None:
        """51 calls is a saga, not a procedure (five-check upper bound)."""
        creator = _make_creator(tmp_path, _good_skill_content())
        result = asyncio.run(creator.maybe_create("huge task", _trace(51)))
        assert result is None
        creator._envelope.call.assert_not_awaited()

    def test_trace_at_max_bound_still_reaches_llm(self, tmp_path: Path) -> None:
        creator = _make_creator(tmp_path, _good_skill_content())
        result = asyncio.run(creator.maybe_create("big but bounded", _trace(50)))
        assert result is not None

    def test_trace_at_min_bound_still_reaches_llm(self, tmp_path: Path) -> None:
        creator = _make_creator(tmp_path, _good_skill_content())
        result = asyncio.run(creator.maybe_create("small but real", _trace(3)))
        assert result is not None

    def test_name_collision_skips_instead_of_timestamp_duplicate(
        self, tmp_path: Path
    ) -> None:
        """An existing ``<slug>.md`` is near-duplicate evidence → no write.

        The old suffix behaviour is how ``debug-cron-job-failure`` came to
        exist three times and how the qwen trio kept landing.
        """
        original = "---\nname: refactor-auth-module\n---\n# The original\n"
        (tmp_path / "refactor-auth-module.md").write_text(original)
        creator = _make_creator(
            tmp_path, _good_skill_content("refactor-auth-module")
        )

        result = asyncio.run(creator.maybe_create("again", _trace(5)))

        assert result is None
        creator._envelope.call.assert_awaited_once()  # post-LLM backstop
        assert [p.name for p in tmp_path.iterdir()] == ["refactor-auth-module.md"]
        assert (tmp_path / "refactor-auth-module.md").read_text() == original

    def test_persist_default_still_suffixes_for_deliberate_writers(
        self, tmp_path: Path
    ) -> None:
        """Direct ``persist_skill_content`` callers (teacher escalation,
        record-a-skill, ACCEPTed drafts) keep the no-content-loss suffix."""
        (tmp_path / "refactor-auth-module.md").write_text("---\nname: refactor-auth-module\n---\n# old\n")
        creator = _make_creator(tmp_path, None)

        path = asyncio.run(
            creator.persist_skill_content(
                _good_skill_content("refactor-auth-module"), trigger="t"
            )
        )

        assert path is not None
        assert path.name.startswith("refactor-auth-module-")
        assert len(list(tmp_path.iterdir())) == 2


class TestStage1LLMOptOut:
    """The generation model may decline — and sees what it needs to."""

    def test_failed_lookup_blocked_only_by_the_opt_out(self, tmp_path: Path) -> None:
        """THE tripwire — the failed-lookup shape.

        2026-08-03: "can you access <X>?" ran three ``ls``-style bash calls
        that all exited 0, the answer was "it doesn't exist", and a skill
        got authored anyway. No deterministic check can see that failure;
        only the generation model can, and only if the prompt (a)
        authorizes SKIP and (b) carries the final reply. Removing either
        from ``_GENERATION_PROMPT``/``maybe_create`` turns this test red —
        by design.
        """
        model = _CompetentModel()
        creator = _creator_with_model(tmp_path, model)

        result = asyncio.run(
            creator.maybe_create(
                "can you access the research vault?",
                _trace(3),
                final_text=(
                    "I checked ~/.prometheus and the working directory — I "
                    "couldn't find any research-vault directory on this machine."
                ),
            )
        )

        assert result is None
        assert len(model.prompts) == 1, "Stage 0 must pass this trace to the LLM"
        assert list(tmp_path.iterdir()) == []

    def test_negative_release_check_blocked(self, tmp_path: Path) -> None:
        """The qwen shape: a lookup whose answer is 'that doesn't exist'."""
        model = _CompetentModel()
        creator = _creator_with_model(tmp_path, model)

        result = asyncio.run(
            creator.maybe_create(
                "qwen 27b 3.8 is out. can you check on github?",
                _trace(4),
                final_text=(
                    "Qwen 3.8 27B is not released — the closest real release "
                    "is Qwen 3.6."
                ),
            )
        )

        assert result is None
        assert list(tmp_path.iterdir()) == []

    def test_capability_test_turn_blocked(self, tmp_path: Path) -> None:
        """The .md-read-test shape: the user is probing, not asking for work."""
        model = _CompetentModel()
        creator = _creator_with_model(tmp_path, model)

        result = asyncio.run(
            creator.maybe_create(
                "testing to see if you can get this .md? [30KB document]",
                _trace(3),
                final_text="Yes — I can read it. Here's a summary of the plan.",
            )
        )

        assert result is None
        assert list(tmp_path.iterdir()) == []

    def test_reusable_procedure_still_authored(self, tmp_path: Path) -> None:
        """The gate must not eat real procedures — the other direction.

        The switch-llama-server-model shape: multi-step, mutating, completed
        successfully. The same competent model that declines junk emits a
        skill here, and the file lands.
        """
        model = _CompetentModel("switch-llama-server-model")
        creator = _creator_with_model(tmp_path, model)

        result = asyncio.run(
            creator.maybe_create(
                "switch the llama-server model to the new Qwen build",
                _trace(6),
                final_text=(
                    "Done — updated the systemd service file and restarted "
                    "llama-server; Qwen3.6 is now serving."
                ),
            )
        )

        assert result is not None
        assert result.name == "switch-llama-server-model.md"
        assert "name: switch-llama-server-model" in result.read_text()

    def test_prompt_carries_outcome_and_existing_skills(self, tmp_path: Path) -> None:
        """Stage 1's inputs actually reach the prompt: SKIP authorization,
        the final reply, the task, and the existing skill list."""
        (tmp_path / "check-ai-model-release-github.md").write_text(
            "---\n"
            "name: check-ai-model-release-github\n"
            "description: Investigate the existence of a model release on GitHub.\n"
            "---\n# x\n"
        )
        model = _CompetentModel("switch-llama-server-model")
        creator = _creator_with_model(tmp_path, model)

        asyncio.run(
            creator.maybe_create(
                "switch the llama-server model to the new Qwen build",
                _trace(6),
                final_text="Done — service restarted; Qwen3.6 now serving.",
            )
        )

        prompt = model.prompts[0]
        assert "SKIP:" in prompt
        assert "Done — service restarted; Qwen3.6 now serving." in prompt
        assert "switch the llama-server model" in prompt
        assert "check-ai-model-release-github" in prompt
        assert "Investigate the existence of a model release" in prompt

    def test_junk_markers_stay_out_of_the_template(self) -> None:
        """The fake model keys on outcome phrases from the TURN, so the
        template must never contain them — otherwise every prompt would
        look junk-shaped and the positive-direction test would lie."""
        from prometheus.learning.skill_creator import _GENERATION_PROMPT

        low = _GENERATION_PROMPT.lower()
        for marker in _JUNK_OUTCOME_MARKERS:
            assert marker not in low, (
                f"_GENERATION_PROMPT now contains {marker!r}; reword the "
                "template or pick a different marker"
            )

    def test_error_flags_render_in_the_trace_format(self) -> None:
        """[ERROR] marks failed calls so Stage 1 stays honest even if the
        Stage 0 any-error skip is ever relaxed."""
        text = SkillCreator._format_trace(_trace(3, errors=frozenset({1})))
        lines = text.splitlines()
        assert "[ERROR]" not in lines[0]
        assert lines[1].endswith("[ERROR]")
        assert "[ERROR]" not in lines[2]
