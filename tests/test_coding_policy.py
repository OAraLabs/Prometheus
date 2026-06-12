"""IterateToGreenPolicy — pure-logic units (verdict, signatures, prompts)."""

from __future__ import annotations

from prometheus.coding.policy import (
    IterateToGreenPolicy,
    failure_signature,
    parse_exit_code,
)


class TestParseExitCode:

    def test_parses_code_run_header(self):
        assert parse_exit_code("exit code: 1 (in 2.3s)\nFAILED x") == 1
        assert parse_exit_code("exit code: 0 (in 0.1s)\nok") == 0

    def test_unparseable_is_none(self):
        assert parse_exit_code("TIMED OUT after 240s") is None


class TestFailureSignature:

    def test_stable_across_timings_and_addresses(self):
        a = failure_signature("FAILED test_x - assert <obj at 0xdeadbeef> in 1.23s")
        b = failure_signature("FAILED test_x - assert <obj at 0xfeedface> in 4.56s")
        assert a == b

    def test_different_failures_differ(self):
        assert failure_signature("FAILED test_x") != failure_signature("FAILED test_y")


class TestEvidence:

    def _policy(self) -> IterateToGreenPolicy:
        return IterateToGreenPolicy(acceptance_command="python3 -m pytest tests/ -q")

    def test_green_acceptance_run_is_evidence(self):
        p = self._policy()
        p.observe_round()
        p.observe_code_run("python3 -m pytest tests/ -q", "exit code: 0 (in 1s)\nok")
        assert p.has_recent_green_evidence()

    def test_green_pytest_variant_is_evidence(self):
        p = self._policy()
        p.observe_round()
        p.observe_code_run("python3 -m pytest tests/test_one.py::test_a -q",
                           "exit code: 0 (in 1s)\nok")
        assert p.has_recent_green_evidence()

    def test_red_run_is_not_evidence(self):
        p = self._policy()
        p.observe_round()
        p.observe_code_run("python3 -m pytest tests/ -q", "exit code: 1 (in 1s)\nFAILED")
        assert not p.has_recent_green_evidence()

    def test_non_test_command_is_not_evidence(self):
        p = self._policy()
        p.observe_round()
        p.observe_code_run("true", "exit code: 0 (in 0s)\n")
        assert not p.has_recent_green_evidence()

    def test_stale_evidence_outside_window_does_not_count(self):
        p = self._policy()
        p.observe_round()
        p.observe_code_run("python3 -m pytest tests/ -q", "exit code: 0 (in 1s)\nok")
        for _ in range(3):  # three more rounds with no test run
            p.observe_round()
        assert not p.has_recent_green_evidence()


class TestCapsAndStepBack:

    def test_round_cap(self):
        p = IterateToGreenPolicy(acceptance_command="x", max_rounds=2)
        p.observe_round(); p.observe_round()
        assert "round cap" in (p.over_cap(0.0) or "")

    def test_wall_cap(self):
        p = IterateToGreenPolicy(acceptance_command="x", max_wall_seconds=10)
        assert "wall-clock cap" in (p.over_cap(11.0) or "")

    def test_step_back_appears_on_second_identical_failure(self):
        p = IterateToGreenPolicy(acceptance_command="pytest")
        p.record_ground_truth_failure("FAILED test_x — assert 1 == 2")
        assert "[STEP BACK]" not in p.ground_truth_rejection("FAILED test_x — assert 1 == 2")
        p.record_ground_truth_failure("FAILED test_x — assert 1 == 2")
        msg = p.ground_truth_rejection("FAILED test_x — assert 1 == 2")
        assert "[STEP BACK]" in msg
        p.record_ground_truth_failure("FAILED test_x — assert 1 == 2")
        assert "[STEP BACK — FINAL]" in p.ground_truth_rejection("FAILED test_x — assert 1 == 2")

    def test_different_failure_resets_repeat_counter(self):
        p = IterateToGreenPolicy(acceptance_command="pytest")
        p.record_ground_truth_failure("FAILED test_x")
        p.record_ground_truth_failure("FAILED test_y")
        assert p.repeat_failures == 0

    def test_success_resets(self):
        p = IterateToGreenPolicy(acceptance_command="pytest")
        p.record_ground_truth_failure("FAILED test_x")
        p.record_ground_truth_success()
        assert p.repeat_failures == 0
