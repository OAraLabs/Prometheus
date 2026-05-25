"""Tests for natural-language cron parsing (Polish & Platform sprint, WS3)."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from prometheus.gateway.cron_service import validate_cron_expression
from prometheus.tools.builtin.cron_create import CronCreateTool, CronCreateToolInput
from prometheus.tools.builtin.cron_nl import (
    ParsedSchedule,
    parse_natural_schedule,
    set_llm_fallback,
    try_llm_fallback,
)


# ---------------------------------------------------------------------------
# Pure parser tests — no I/O, fully deterministic
# ---------------------------------------------------------------------------


class TestParseNaturalSchedule:
    """Patterns the regex parser must handle without an LLM call."""

    def test_every_monday_at_9am(self):
        result = parse_natural_schedule("every Monday at 9am")
        assert result is not None
        assert result.cron == "0 9 * * 1"
        assert result.one_shot is False
        assert validate_cron_expression(result.cron)

    def test_every_monday_at_9_30am(self):
        result = parse_natural_schedule("every Monday at 9:30am")
        assert result is not None
        assert result.cron == "30 9 * * 1"

    def test_every_weekday_at_noon(self):
        result = parse_natural_schedule("every weekday at noon")
        assert result is not None
        assert result.cron == "0 12 * * 1-5"
        assert validate_cron_expression(result.cron)

    def test_every_weekend_at_8am(self):
        result = parse_natural_schedule("every weekend at 8am")
        assert result is not None
        # cron 0=Sunday, 6=Saturday
        assert result.cron == "0 8 * * 0,6"

    def test_every_day_at_midnight(self):
        result = parse_natural_schedule("every day at midnight")
        assert result is not None
        assert result.cron == "0 0 * * *"

    def test_daily_at_3pm(self):
        result = parse_natural_schedule("daily at 3pm")
        assert result is not None
        assert result.cron == "0 15 * * *"

    def test_every_5_minutes(self):
        result = parse_natural_schedule("every 5 minutes")
        assert result is not None
        assert result.cron == "*/5 * * * *"
        assert validate_cron_expression(result.cron)

    def test_every_2_hours(self):
        result = parse_natural_schedule("every 2 hours")
        assert result is not None
        assert result.cron == "0 */2 * * *"

    def test_hourly_keyword(self):
        result = parse_natural_schedule("hourly")
        assert result is not None
        assert result.cron == "0 * * * *"

    # ---- one-shot patterns ----

    def test_in_30_minutes_is_one_shot(self):
        now = datetime(2026, 5, 23, 14, 0)
        result = parse_natural_schedule("in 30 minutes", now=now)
        assert result is not None
        assert result.one_shot is True
        # 14:00 + 30min = 14:30
        assert result.cron == "30 14 23 5 *"
        assert validate_cron_expression(result.cron)

    def test_tomorrow_at_3pm_is_one_shot(self):
        now = datetime(2026, 5, 23, 14, 0)
        result = parse_natural_schedule("tomorrow at 3pm", now=now)
        assert result is not None
        assert result.one_shot is True
        # Tomorrow = May 24, 15:00
        assert result.cron == "0 15 24 5 *"

    def test_at_3pm_today_when_future(self):
        now = datetime(2026, 5, 23, 10, 0)  # 10am, so 3pm today is future
        result = parse_natural_schedule("at 3pm", now=now)
        assert result is not None
        assert result.one_shot is True
        assert result.cron == "0 15 23 5 *"

    def test_at_3pm_tomorrow_when_past(self):
        now = datetime(2026, 5, 23, 17, 0)  # 5pm — 3pm has passed
        result = parse_natural_schedule("at 3pm", now=now)
        assert result is not None
        assert result.one_shot is True
        # Should bump to tomorrow
        assert result.cron == "0 15 24 5 *"

    # ---- unparseable / edge cases ----

    def test_gibberish_returns_none(self):
        assert parse_natural_schedule("when the moon hits your eye") is None

    def test_empty_returns_none(self):
        assert parse_natural_schedule("") is None

    def test_unknown_weekday_returns_none(self):
        assert parse_natural_schedule("every funday at 9am") is None

    def test_invalid_minute_count_returns_none(self):
        # every 60 minutes is out of cron range
        assert parse_natural_schedule("every 60 minutes") is None

    def test_case_insensitive(self):
        result = parse_natural_schedule("Every Monday At 9AM")
        assert result is not None
        assert result.cron == "0 9 * * 1"


# ---------------------------------------------------------------------------
# LLM fallback hook
# ---------------------------------------------------------------------------


class TestLlmFallback:
    def setup_method(self):
        set_llm_fallback(None)

    def teardown_method(self):
        set_llm_fallback(None)

    def test_no_fallback_installed_returns_none(self):
        assert try_llm_fallback("weird input") is None

    def test_installed_fallback_is_called(self):
        called = {}

        def fake(text):
            called["text"] = text
            return ParsedSchedule(cron="0 9 * * *")

        set_llm_fallback(fake)
        result = try_llm_fallback("at the third hour of dawn")
        assert called["text"] == "at the third hour of dawn"
        assert result is not None
        assert result.cron == "0 9 * * *"

    def test_fallback_exception_swallowed(self):
        def boom(_text):
            raise RuntimeError("model down")

        set_llm_fallback(boom)
        # Must NOT raise — fallback is best-effort.
        assert try_llm_fallback("anything") is None


# ---------------------------------------------------------------------------
# CronCreateTool integration
# ---------------------------------------------------------------------------


@pytest.fixture
def tool_context(tmp_path, monkeypatch):
    """Provide a ToolExecutionContext with a tmp registry path."""
    from prometheus.gateway import cron_service
    from prometheus.tools.base import ToolExecutionContext

    registry = tmp_path / "cron.json"
    monkeypatch.setattr(cron_service, "get_cron_registry_path", lambda: registry)

    return ToolExecutionContext(cwd=tmp_path)


class TestCronCreateToolStandardSyntax:
    """Standard cron syntax keeps working unchanged (no regression)."""

    @pytest.mark.asyncio
    async def test_standard_cron_syntax_passes_through(self, tool_context, tmp_path):
        from prometheus.gateway.cron_service import get_cron_job

        tool = CronCreateTool()
        result = await tool.execute(
            CronCreateToolInput(
                name="weekday_9am",
                schedule="0 9 * * 1-5",
                command="echo hi",
            ),
            tool_context,
        )
        assert not result.is_error
        assert "0 9 * * 1-5" in result.output

        job = get_cron_job("weekday_9am")
        assert job is not None
        assert job["schedule"] == "0 9 * * 1-5"


class TestCronCreateToolNaturalLanguage:
    """NL phrases get translated to cron before being saved."""

    @pytest.mark.asyncio
    async def test_every_monday_at_9am(self, tool_context):
        from prometheus.gateway.cron_service import get_cron_job

        tool = CronCreateTool()
        result = await tool.execute(
            CronCreateToolInput(
                name="weekly_report",
                schedule="every Monday at 9am",
                command="prometheus run weekly-report",
            ),
            tool_context,
        )
        assert not result.is_error
        job = get_cron_job("weekly_report")
        assert job is not None
        assert job["schedule"] == "0 9 * * 1"
        # User sees the translation in the output.
        assert "every Monday at 9am" in result.output
        assert "0 9 * * 1" in result.output

    @pytest.mark.asyncio
    async def test_in_30_minutes_is_one_shot_with_warning(self, tool_context):
        tool = CronCreateTool()
        result = await tool.execute(
            CronCreateToolInput(
                name="oneshot_test",
                schedule="in 30 minutes",
                command="echo go",
            ),
            tool_context,
        )
        assert not result.is_error
        # Warning about annual re-fire should be present for one-shots.
        assert "one-shot" in result.output.lower()
        assert "annually" in result.output.lower()


class TestCronCreateToolUnparseable:
    """Unparseable strings hit the LLM fallback; if that's absent, error."""

    def setup_method(self):
        set_llm_fallback(None)

    def teardown_method(self):
        set_llm_fallback(None)

    @pytest.mark.asyncio
    async def test_gibberish_with_no_llm_fallback_errors(self, tool_context):
        tool = CronCreateTool()
        result = await tool.execute(
            CronCreateToolInput(
                name="bad",
                schedule="when pigs fly",
                command="echo never",
            ),
            tool_context,
        )
        assert result.is_error
        assert "couldn't parse" in result.output.lower()
        # Useful guidance, not a stack trace.
        assert "cron expression" in result.output.lower()
        assert "natural" in result.output.lower() or "phrase" in result.output.lower()

    @pytest.mark.asyncio
    async def test_gibberish_falls_through_to_llm_fallback(self, tool_context):
        # Install a fake LLM fallback that rescues "when pigs fly".
        set_llm_fallback(lambda _text: ParsedSchedule(cron="0 9 * * *"))

        tool = CronCreateTool()
        result = await tool.execute(
            CronCreateToolInput(
                name="rescued",
                schedule="when pigs fly",
                command="echo go",
            ),
            tool_context,
        )
        assert not result.is_error
        assert "0 9 * * *" in result.output
