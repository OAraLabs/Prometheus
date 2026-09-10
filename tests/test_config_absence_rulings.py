"""PR-B config-absence rulings — three shapes, three answers (handoff item 6).

Registered May 2026, never ruled on; re-measured 2026-08-18 as three
distinct shapes. The rulings pinned here:

A. ABSENCE → PERMISSIVE is a defect (the #219 class): an absent
   ``sentinel.enabled`` no longer STARTS an autonomous subsystem. Fail
   closed, warn loudly, name the line to write.
B. ABSENCE → SILENTLY DISABLED keeps failing closed (a missing line must
   not start a web server), but stops being silent: the operator who
   believes the template's value is in force is told at boot.
   compaction.enabled already bit exactly this way — config-dark since
   birth. The template's value is NOT uniform across the three gates —
   web and compaction ship true, trajectory_export ships false — so the
   warning names the real one per key, pinned below.
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


# ---------------------------------------------------------------------------
# THE PROSE MUST NOT MISQUOTE THE TEMPLATE (audit P11.1)
#
# `_sentinel_enabled`'s docstring said "The shipped template says
# ``enabled: true``" and its boot WARNING called `true` "the shipped
# template's value". The template has shipped `sentinel.enabled: false`
# since the initial commit (cfebf6c, 2026-04-20); the claim landed with the
# ruling itself (#325, 2026-08-28), so it was WRONG FROM THE DAY IT WAS
# WRITTEN — the same shape as the model-router bullet #420 corrected.
#
# It was not merely a docstring. The claim also reached the OPERATOR twice
# at runtime: the sentinel warning's parenthetical, and shape B's warning
# asserting "the shipped template sets it true" for all three gates — false
# for trajectory_export, which the template ships OFF. That one told the
# operator to switch on an exporter the template deliberately leaves off.
#
# The fix that lasts is not new prose, it is prose that cannot drift: the
# values are DATA in `_ABSENT_GATE_WARNINGS`, and the tests below read the
# template. A claim about a config file, checked against that config file.
# ---------------------------------------------------------------------------

# ONE definition, used by BOTH the prohibition and its self-verification.
# Duplicating it would make the self-check decorative: weakening the real
# pattern would leave the copy — and the test — green.
_CLAIMS_TEMPLATE_ENABLES_SENTINEL = re.compile(
    # "the shipped template says/sets/ships `enabled: true`" — the DOCSTRING
    # form #325 carried.
    r"template[^.\n]{0,80}(?:says|sets|ships)[^.\n]{0,40}"
    r"(?:``)?enabled:?\s*true|"
    # "`sentinel.enabled` ... the template ... true" — template BEFORE true.
    r"(?:``)?sentinel\.enabled[^.\n]{0,60}template[^.\n]{0,40}true|"
    # ...and TRUE BEFORE TEMPLATE, which is how the WARNING TEXT said it:
    # "Write `sentinel.enabled: true` (the shipped template's value)". The
    # same claim with no verb, and both alternatives above miss it — the
    # first wants says/sets/ships, the second wants `template` to come
    # first. That sentence was the OTHER HALF of the very defect this file
    # forbids: #424 deleted it from daemon.py, and the detector would not
    # have stopped it coming back. Found by restoring it verbatim and
    # watching all fifteen tests pass.
    r"(?:``|`)?sentinel\.enabled[^.\n]{0,40}true[^.\n]{0,40}template",
    re.I,
)


def _shipped_template() -> dict:
    import prometheus.daemon as daemon_mod
    root = Path(daemon_mod.__file__).resolve().parents[2]
    return yaml.safe_load(
        (root / "config" / "prometheus.yaml.default").read_text(encoding="utf-8")
    )


class TestTemplateClaimsAreTrue:
    def test_the_template_really_ships_sentinel_off(self) -> None:
        """The fact every sentence about SENTINEL depends on."""
        assert _shipped_template()["sentinel"]["enabled"] is False

    def test_the_shipped_template_produces_sentinel_off(self) -> None:
        """Read the template through the REAL reader, not by eye.

        Pairs the two halves: the value the file carries and what the
        daemon does with it. A fresh install gets SENTINEL off.
        """
        assert _sentinel_enabled(_shipped_template()["sentinel"]) is False

    def test_no_daemon_prose_claims_the_template_enables_sentinel(self) -> None:
        """The defect itself, in the file that carried it.

        Deliberately not a pin on today's wording — it forbids the CLAIM.
        Any sentence putting `sentinel.enabled` next to a template that
        `says`/`sets`/`ships` it true fails, however it is phrased.
        """
        import prometheus.daemon as daemon_mod
        source = Path(daemon_mod.__file__).read_text(encoding="utf-8")

        offenders = [m.group(0) for m in
                     _CLAIMS_TEMPLATE_ENABLES_SENTINEL.finditer(source)]
        assert not offenders, (
            "daemon.py claims the shipped template enables SENTINEL. It "
            "ships `sentinel.enabled: false`, and always has:\n  "
            + "\n  ".join(offenders)
        )

    def test_the_claim_detector_catches_the_sentence_that_shipped(self) -> None:
        """Self-verification: the real #325 text must still trip it.

        Without this the regex above could be softened to match nothing and
        stay green — the standing failure mode of a prohibition.
        """
        docstring_form = (
            "    subsystem is opted INTO, never inherited from a missing "
            "line. The\n    shipped template says ``enabled: true``, so "
            "fresh installs are\n    unchanged; only a hand-written config "
            "that omits the key changes\n    behaviour."
        )
        # BOTH halves of #325, not one. The warning text made the same claim
        # in the possessive, and a detector fed only the docstring form was
        # green while blind to the sentence #424 actually had to delete.
        warning_form = (
            '            "Write `sentinel.enabled: true` (the shipped '
            'template\'s value) "\n'
            '            "to start it. Before this ruling an absent key '
            'STARTED the "\n            "subsystem."'
        )
        for label, text in (("docstring", docstring_form),
                            ("warning text", warning_form)):
            assert _CLAIMS_TEMPLATE_ENABLES_SENTINEL.search(text), (
                f"the detector no longer catches the {label} form that "
                f"actually shipped in #325 — it has been weakened past "
                f"proving anything"
            )


class TestShapeB_TemplateValuesAreReal:
    def test_each_entry_matches_the_template(self) -> None:
        """The ratchet. Every declared value is read back out of the file.

        This is what stops the blanket "sets it true" returning: the
        warning can only say what the template actually says, because the
        value it prints comes from a table this test pins to the template.
        """
        from prometheus.daemon import _ABSENT_GATE_WARNINGS

        template = _shipped_template()
        wrong = []
        for section, key, _effect, shipped in _ABSENT_GATE_WARNINGS:
            actual = (template.get(section) or {}).get(key, "<ABSENT>")
            if actual is not shipped:
                wrong.append(f"{section}.{key}: table says {shipped!r}, "
                             f"template says {actual!r}")
        assert not wrong, (
            "_ABSENT_GATE_WARNINGS disagrees with the shipped template. The "
            "operator is told a default the file does not carry:\n  "
            + "\n  ".join(wrong)
        )

    def test_the_three_gates_are_not_all_the_same(self) -> None:
        """Guards the REASON the table exists.

        If someone flips trajectory_export on in the template, a single
        blanket sentence becomes correct again and the next author may
        collapse the table back. This fails first and says why.
        """
        from prometheus.daemon import _ABSENT_GATE_WARNINGS

        values = {shipped for *_rest, shipped in _ABSENT_GATE_WARNINGS}
        assert values == {True, False}, (
            "the three shape-B gates now ship the SAME template value. The "
            "per-key value in _ABSENT_GATE_WARNINGS exists because they "
            "differed (web/compaction true, trajectory_export false) and a "
            "blanket claim was wrong for one of them. If that is genuinely "
            "no longer true, re-read the warning text before simplifying."
        )

    def test_the_exporter_warning_says_false_not_true(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The operator-facing half, asserted on the emitted message.

        The table being right is not enough — the WARNING is what the
        operator reads, and it used to say `true` for this key.
        """
        with caplog.at_level(logging.WARNING):
            _warn_absent_gating_keys({"web": {"enabled": True},
                                      "compaction": {"enabled": True}})
        line = next(r.getMessage() for r in caplog.records
                    if "trajectory_export.enabled" in r.getMessage())
        assert "sets it false" in line, line
