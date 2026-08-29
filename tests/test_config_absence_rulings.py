"""PR-B config-absence rulings — three shapes, three answers (handoff item 6).

Registered May 2026, never ruled on; re-measured 2026-08-18 as three
distinct shapes. The rulings pinned here:

A. ABSENCE → PERMISSIVE is a defect (the #219 class): an absent
   ``sentinel.enabled`` no longer STARTS an autonomous subsystem. Fail
   closed, warn loudly, name the line to write.
B. ABSENCE → SILENTLY DISABLED keeps failing closed (a missing line must
   not start a web server), but stops being silent: the operator who
   believes the template's ``true`` is in force is told at boot.
   compaction.enabled already bit exactly this way — config-dark since
   birth.
C. DOCUMENTED VALUE ≠ CODE FALLBACK is a defect: the coding fallbacks now
   match the shipped template (50 rounds / 120 minutes), which is the
   operator-visible contract. The guard below reads BOTH the template and
   the code so the pair cannot drift apart again.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest
import yaml

from prometheus.daemon import _sentinel_enabled, _warn_absent_gating_keys


class TestShapeA_SentinelAbsence:
    def test_absent_key_does_not_start_sentinel(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            assert _sentinel_enabled({}) is False
        assert "sentinel.enabled" in caplog.text
        assert "ABSENT" in caplog.text

    def test_explicit_true_starts_it_without_noise(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            assert _sentinel_enabled({"enabled": True}) is True
        assert "sentinel.enabled" not in caplog.text

    def test_explicit_false_stays_off(self) -> None:
        assert _sentinel_enabled({"enabled": False}) is False


class TestShapeB_AbsentGateWarnings:
    def test_empty_config_names_all_three_gates(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            _warn_absent_gating_keys({})
        for key in ("web.enabled", "trajectory_export.enabled",
                    "compaction.enabled"):
            assert key in caplog.text

    def test_fully_keyed_config_is_silent(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        config = {
            "web": {"enabled": True},
            "trajectory_export": {"enabled": False},
            "compaction": {"enabled": True},
        }
        with caplog.at_level(logging.WARNING):
            _warn_absent_gating_keys(config)
        assert "ABSENT" not in caplog.text

    def test_only_the_missing_gate_is_named(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # An explicit false is a DECISION, not an absence — no warning.
        config = {"web": {"enabled": False}, "compaction": {"enabled": True}}
        with caplog.at_level(logging.WARNING):
            _warn_absent_gating_keys(config)
        assert "trajectory_export.enabled" in caplog.text
        assert "web.enabled" not in caplog.text
        assert "compaction.enabled" not in caplog.text

    def test_run_daemon_calls_the_warner_after_pins(self) -> None:
        import prometheus.daemon as daemon_mod
        source = Path(daemon_mod.__file__).read_text(encoding="utf-8")
        pins = source.index('apply_config_pins(config, get_config_dir()')
        warn = source.index("_warn_absent_gating_keys(config)")
        assert pins < warn, (
            "the absence warnings must judge the PINNED config — pins can "
            "add keys"
        )


class TestShapeC_CodingFallbacksMatchTheTemplate:
    def test_fallbacks_equal_the_shipped_template(self) -> None:
        # Reads BOTH sides so neither can drift alone: the template's
        # coding section and the literal fallbacks in __main__'s coding
        # invocation. Before this ruling the pair read 50/120 in the
        # template and 30/20 in the code — and with the live config's keys
        # commented out, absence was the NORMAL state, so the documented
        # wall clock was 6x the effective one.
        import prometheus.__main__ as entry

        repo_root = Path(entry.__file__).resolve().parents[2]
        template = repo_root / "config" / "prometheus.yaml.default"
        coding = yaml.safe_load(template.read_text())["coding"]

        source = Path(entry.__file__).read_text(encoding="utf-8")
        m_iter = re.search(r'coding_cfg\.get\("max_iterations", (\d+)\)', source)
        m_wall = re.search(
            r'coding_cfg\.get\("max_task_duration_minutes", (\d+)\)', source
        )
        assert m_iter and m_wall, "coding fallback reads not found"
        assert int(m_iter.group(1)) == int(coding["max_iterations"])
        assert int(m_wall.group(1)) == int(coding["max_task_duration_minutes"])
