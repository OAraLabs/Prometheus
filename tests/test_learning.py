"""Tests for the learning loop components: PeriodicNudge, SkillCreator, SkillRefiner."""

from __future__ import annotations

import pytest

# Import engine.messages first to avoid circular import in providers
from prometheus.engine.messages import ConversationMessage  # noqa: F401
from prometheus.learning.nudge import PeriodicNudge
from prometheus.learning.skill_creator import SkillCreator
from prometheus.learning.skill_refiner import SkillRefiner
from prometheus.providers.base import ApiTextDeltaEvent


# ---------------------------------------------------------------------------
# Mock ModelProvider
# ---------------------------------------------------------------------------


class MockProvider:
    """Minimal provider that yields a single text delta with the canned response."""

    def __init__(self, response: str) -> None:
        self._response = response

    async def stream_message(self, request):  # noqa: ANN001
        yield ApiTextDeltaEvent(text=self._response)


def _async_return(value):
    """Return an async function that returns *value*. Useful for mocking _call_model."""
    async def _fn(*_args, **_kwargs):
        return value
    return _fn


# ---------------------------------------------------------------------------
# PeriodicNudge
# ---------------------------------------------------------------------------


class TestPeriodicNudge:
    def test_periodic_nudge_fires_at_interval(self) -> None:
        nudge = PeriodicNudge(interval=15)
        # Should fire at exact multiples of 15
        assert nudge.maybe_inject(15) is not None
        assert nudge.maybe_inject(30) is not None
        # Should NOT fire at non-multiples
        assert nudge.maybe_inject(14) is None
        assert nudge.maybe_inject(16) is None

    def test_periodic_nudge_disabled(self) -> None:
        nudge = PeriodicNudge(interval=15, enabled=False)
        assert nudge.maybe_inject(15) is None
        assert nudge.maybe_inject(30) is None

    def test_periodic_nudge_content(self) -> None:
        nudge = PeriodicNudge(interval=5)
        msg = nudge.maybe_inject(5)
        assert msg is not None
        assert msg["role"] == "user"
        assert msg["_nudge"] is True
        assert "_nudge_number" in msg
        assert msg["_nudge_number"] == 1
        assert "[system-internal]" in msg["content"]

    def test_nudge_reset(self) -> None:
        nudge = PeriodicNudge(interval=5)
        nudge.maybe_inject(5)
        nudge.maybe_inject(10)
        assert nudge.nudge_count == 2
        nudge.reset()
        assert nudge.nudge_count == 0


# ---------------------------------------------------------------------------
# SkillCreator
# ---------------------------------------------------------------------------


class TestSkillCreator:
    def _make_trace(self, count: int) -> list[dict]:
        return [
            {"tool_name": f"tool_{i}", "arguments": {"a": i}, "result": f"ok_{i}"}
            for i in range(count)
        ]

    async def test_skill_creator_below_threshold(self, tmp_path) -> None:
        provider = MockProvider("should not be called")
        creator = SkillCreator(provider, auto_dir=tmp_path)
        result = await creator.maybe_create("small task", self._make_trace(2))
        assert result is None

    async def test_skill_creator_creates_file(self, tmp_path) -> None:
        skill_content = (
            "---\nname: deploy-app\ndescription: Deploy the application\n---\n"
            "# Deploy App\n\n## When to use\nWhen deploying.\n\n## Steps\n1. Build\n2. Deploy\n"
        )
        provider = MockProvider(skill_content)
        creator = SkillCreator(provider, auto_dir=tmp_path)
        # Mock _call_model to bypass ConversationMessage content-type mismatch
        creator._call_model = _async_return(skill_content)
        result = await creator.maybe_create("deploy the application", self._make_trace(5))
        assert result is not None
        assert result.exists()
        assert result.suffix == ".md"
        text = result.read_text()
        assert "deploy" in text.lower()


# ---------------------------------------------------------------------------
# SkillRefiner
# ---------------------------------------------------------------------------


class TestSkillRefiner:
    def _make_trace(self, count: int) -> list[dict]:
        return [
            {"tool_name": f"tool_{i}", "arguments": {"a": i}, "result": f"ok_{i}"}
            for i in range(count)
        ]

    async def test_skill_refiner_no_change(self, tmp_path) -> None:
        skill_path = tmp_path / "my-skill.md"
        original = "---\nname: my-skill\n---\n# My Skill\n\n## Steps\n1. Do thing\n"
        skill_path.write_text(original)

        provider = MockProvider("NO_CHANGE")
        refiner = SkillRefiner(provider)
        refiner._call_model = _async_return("NO_CHANGE")

        updated = await refiner.maybe_refine(skill_path, self._make_trace(3), "success")
        assert updated is False
        # Content should be unchanged
        assert skill_path.read_text() == original

    async def test_skill_refiner_updates_skill(self, tmp_path) -> None:
        skill_path = tmp_path / "my-skill.md"
        original = "---\nname: my-skill\n---\n# My Skill\n\n## Steps\n1. Do thing\n"
        skill_path.write_text(original)

        new_content = "---\nname: my-skill\n---\n# My Skill\n\n## Steps\n1. Do better thing\n2. Extra step\n"
        provider = MockProvider(new_content)
        refiner = SkillRefiner(provider)
        refiner._call_model = _async_return(new_content)

        updated = await refiner.maybe_refine(skill_path, self._make_trace(3), "success")
        assert updated is True
        # Skill file should be updated
        assert "better thing" in skill_path.read_text()
        # Backup should exist
        backups = list(tmp_path.glob("my-skill.bak-*.md"))
        assert len(backups) == 1
        assert backups[0].read_text() == original
