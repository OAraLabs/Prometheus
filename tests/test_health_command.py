"""Sprint S4 A3 — /health command tests.

Per Sprint 4 Work Stream 3 acceptance:

  - test_health_command_surfaces_zero_failures
  - test_health_command_flags_silent_failures
  - test_health_command_groups_by_subsystem
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from prometheus.gateway.commands import cmd_health
from prometheus.telemetry.tracker import (
    ToolCallTelemetry,
    set_telemetry_handle,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def tel_with_handle(tmp_path: Path) -> ToolCallTelemetry:
    """Construct a tracker and register it as the singleton for cmd_health."""
    tel = ToolCallTelemetry(db_path=tmp_path / "t.db")
    set_telemetry_handle(tel)
    yield tel
    set_telemetry_handle(None)


class TestHealthCommand:
    """The three Sprint 4 spec tests, plus a few sanity checks."""

    def test_health_command_surfaces_zero_failures(
        self, tel_with_handle: ToolCallTelemetry
    ) -> None:
        """Clean state: no silent failures, healthy banner."""
        tel = tel_with_handle
        # Some healthy runs across two subsystems.
        for _ in range(3):
            tel.record_run("curator", "run_once", "success", duration_ms=42.0)
        for _ in range(2):
            tel.record_run(
                "memory_extractor", "extract_memory_batch", "success",
                duration_ms=10.0,
            )
        tel.record(
            model="m", tool_name="bash", success=True, retries=0, latency_ms=50.0
        )

        text = cmd_health()
        assert "Prometheus Health" in text
        assert "no silent failures detected" in text
        assert "✨" in text
        # Subsystem rows
        assert "curator" in text
        assert "memory_extractor" in text
        # No "SILENT FAILURES" marker
        assert "SILENT FAILURES" not in text

    def test_health_command_flags_silent_failures(
        self, tel_with_handle: ToolCallTelemetry
    ) -> None:
        """A silent_failures row → /health flags the subsystem + lists recent failure."""
        tel = tel_with_handle
        # Two successful runs + two silent failures for the same subsystem.
        for _ in range(2):
            tel.record_run(
                "skill_creator", "generate_skill", "success", duration_ms=15.0
            )
        for i in range(2):
            try:
                raise ValueError(f"synthetic failure #{i}")
            except Exception as exc:
                tel.record_silent_failure(
                    "skill_creator", "_call_model", exc,
                    context={"prompt_len": 100 + i},
                )

        text = cmd_health()
        assert "skill_creator" in text
        assert "SILENT FAILURES" in text
        assert "Silent failures (most recent" in text
        # The most-recent-5 list should show the latest synthetic failure.
        assert "synthetic failure" in text
        assert "ValueError" in text

    def test_health_command_groups_by_subsystem(
        self, tel_with_handle: ToolCallTelemetry
    ) -> None:
        """Multiple failures from the same subsystem aggregate to one row."""
        tel = tel_with_handle
        # 3 failures in skill_creator, 1 in memory_extractor.
        for i in range(3):
            try:
                raise RuntimeError(f"creator-error-{i}")
            except Exception as exc:
                tel.record_silent_failure("skill_creator", "_call_model", exc)
        try:
            raise RuntimeError("extractor-error")
        except Exception as exc:
            tel.record_silent_failure("memory_extractor", "_call_model", exc)

        text = cmd_health()
        # Each subsystem appears exactly once in the per-subsystem block.
        # Use a tight match to avoid counting the recent-failures list.
        per_subsystem_block = text.split("Silent failures (most recent")[0]
        assert per_subsystem_block.count("skill_creator") == 1, per_subsystem_block
        assert per_subsystem_block.count("memory_extractor") == 1, per_subsystem_block


class TestHealthCommandEdgeCases:
    """Defensive checks: no telemetry, custom window, verbose mode."""

    def test_no_telemetry_handle_returns_explanatory_text(self) -> None:
        set_telemetry_handle(None)
        text = cmd_health()
        assert "telemetry not wired" in text

    def test_custom_lookback_window_honored(
        self, tel_with_handle: ToolCallTelemetry
    ) -> None:
        tel = tel_with_handle
        # Insert an "old" silent-failure row (raw INSERT bypasses time.time()).
        old_ts = time.time() - 48 * 3600  # 2 days ago
        try:
            raise ValueError("ancient")
        except Exception as exc:
            tel.record_silent_failure("test_old", "_call_model", exc)
        # Manually backdate the row we just inserted.
        tel._conn.execute(
            "UPDATE silent_failures SET timestamp = ? WHERE subsystem = ?",
            (old_ts, "test_old"),
        )
        tel._conn.commit()

        # 24h window: should NOT see it.
        recent = cmd_health(since_hours=24.0)
        assert "test_old" not in recent
        # 168h window: should see it.
        weekly = cmd_health(since_hours=168.0)
        assert "test_old" in weekly

    def test_verbose_includes_tracebacks(
        self, tel_with_handle: ToolCallTelemetry
    ) -> None:
        tel = tel_with_handle
        try:
            raise ValueError("verbose-test-failure")
        except Exception as exc:
            tel.record_silent_failure("test_sub", "_call_model", exc)

        text = cmd_health(verbose=True)
        # The traceback section header is present
        assert "Recent silent-failure tracebacks" in text
        # The function name from the traceback shows up
        assert "ValueError" in text
        # Non-verbose mode does NOT include the section header
        plain = cmd_health(verbose=False)
        assert "Recent silent-failure tracebacks" not in plain

    def test_zero_tool_calls_renders_n_a_instead_of_0_pct(
        self, tel_with_handle: ToolCallTelemetry
    ) -> None:
        """Post-PR-#7 cleanup: with 0 tool calls in the window, the
        success-rate string must be ``n/a`` rather than the mathematically
        meaningless ``0.0% success``. Behaviour is identical for non-zero
        cases."""
        # Telemetry handle is wired by the fixture but we deliberately record
        # zero tool calls. A non-empty subsystems block keeps the rest of
        # the output well-formed.
        tel = tel_with_handle
        tel.record_run("curator", "run_once", "success", duration_ms=1.0)

        text = cmd_health()
        # The /health line for tool calls should show "n/a" — never "0.0%".
        assert "0.0% success" not in text, (
            "Zero tool calls should render 'n/a', not '0.0% success'. "
            f"Got:\n{text}"
        )
        # And the new "n/a" token must be present, in the tool-calls line.
        tool_calls_line = next(
            (ln for ln in text.splitlines() if "Tool calls:" in ln),
            "",
        )
        assert "n/a" in tool_calls_line, (
            f"Expected 'n/a' in tool-calls line. Got: {tool_calls_line!r}"
        )


class TestHealthCommandWiring:
    """Sanity: both Telegram and Slack adapters expose a /health handler."""

    def test_telegram_has_cmd_health(self) -> None:
        from prometheus.gateway.telegram import TelegramAdapter

        assert hasattr(TelegramAdapter, "_cmd_health")

    def test_slack_has_slash_health(self) -> None:
        from prometheus.gateway.slack import SlackAdapter

        assert hasattr(SlackAdapter, "_slash_health")
