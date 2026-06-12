"""Tier-1 failure detector — every pattern unit-tested + fixture transcripts.

Fixtures live in tests/fixtures/escalation/*.json: ``pos_*`` must fail,
``neg_*`` are failure-adjacent and must NOT (spec requires >= 3 negatives).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prometheus.escalation.detector import (
    CAPABILITY_DENIAL_PATTERNS,
    CLARIFICATION_STALL_MAX_CHARS,
    REPETITION_THRESHOLD,
    RETRY_ACK_PATTERNS,
    FailureVerdict,
    detect_failure,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "escalation"
FIXTURES = sorted(FIXTURE_DIR.glob("*.json"))


# ---------------------------------------------------------------------------
# Fixture transcripts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_fixture_transcript(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    verdict = detect_failure(data["tool_results"], data["final_reply"])
    assert verdict.failed == data["expect_failed"], (
        data["description"], verdict.reasons)
    for pattern in data.get("expect_patterns_include", []):
        assert pattern in verdict.matched_patterns, (
            pattern, verdict.matched_patterns)
    if not data["expect_failed"]:
        # A clean verdict carries no residue.
        assert verdict.reasons == []
        assert verdict.matched_patterns == []


def test_fixture_inventory() -> None:
    """Spec minimums: >=3 negative fixtures, every signal exercised."""
    names = {p.stem for p in FIXTURES}
    assert len([n for n in names if n.startswith("neg_")]) >= 3
    assert len([n for n in names if n.startswith("pos_")]) >= 5


# ---------------------------------------------------------------------------
# Every capability-denial pattern fires (and the example table stays in sync)
# ---------------------------------------------------------------------------

_DENIAL_EXAMPLES = {
    "no_tool": "I don't have a tool that can send emails.",
    "unknown_tool": "Unknown action requested, so nothing was run.",
    "unable_to_act": "I am unable to execute shell commands here.",
    "cannot_act": "I can't access the network from this environment.",
    "no_access": "I don't have permission to modify that database.",
    "breaker_terminal": (
        "Circuit breaker tripped: repeated malformed calls. The model "
        "cannot produce valid tool calls for this request."
    ),
}


@pytest.mark.parametrize("name", [n for n, _ in CAPABILITY_DENIAL_PATTERNS])
def test_every_denial_pattern_fires(name: str) -> None:
    verdict = detect_failure([], _DENIAL_EXAMPLES[name])
    assert verdict.failed
    assert f"capability_denial:{name}" in verdict.matched_patterns


def test_denial_example_table_covers_all_patterns() -> None:
    assert set(_DENIAL_EXAMPLES) == {n for n, _ in CAPABILITY_DENIAL_PATTERNS}


# ---------------------------------------------------------------------------
# Unrecovered tool error vs retry acknowledgment
# ---------------------------------------------------------------------------

_LAST_ERROR = [
    {"tool_name": "bash", "arguments": {"command": "make"},
     "result": "Error: target not found", "is_error": True},
]

_RETRY_ACK_EXAMPLES = [
    "Let me retry that with a longer timeout.",
    "I'll try again with the full path.",
    "Trying a different branch name now.",
    "Retrying with elevated permissions.",
    "That failed, so I'm switching to the REST endpoint instead.",
    "I will use the fallback mirror for the next attempt.",
]


def test_unrecovered_error_fires_without_retry_plan() -> None:
    verdict = detect_failure(_LAST_ERROR, "The build did not work.")
    assert verdict.failed
    assert "unrecovered_tool_error" in verdict.matched_patterns


@pytest.mark.parametrize("reply", _RETRY_ACK_EXAMPLES)
def test_retry_acknowledgment_suppresses_error_signal(reply: str) -> None:
    verdict = detect_failure(_LAST_ERROR, reply)
    assert "unrecovered_tool_error" not in verdict.matched_patterns


def test_every_retry_ack_pattern_matches_an_example() -> None:
    for pattern in RETRY_ACK_PATTERNS:
        assert any(pattern.search(ex) for ex in _RETRY_ACK_EXAMPLES), pattern.pattern


def test_midturn_error_recovered_is_clean() -> None:
    """Error followed by a successful final call is normal agent behavior."""
    results = _LAST_ERROR + [
        {"tool_name": "bash", "arguments": {"command": "make all"},
         "result": "ok", "is_error": False},
    ]
    verdict = detect_failure(results, "Built successfully after fixing the target name.")
    assert not verdict.failed


def test_error_text_fallback_without_flag() -> None:
    """No is_error key: error-shaped text counts, informative text doesn't."""
    errorish = [{"tool_name": "bash", "arguments": {}, "result": "Error: boom"}]
    verdict = detect_failure(errorish, "That did not go well.")
    assert "unrecovered_tool_error" in verdict.matched_patterns

    informative = [{"tool_name": "grep", "arguments": {}, "result": "(no output)"}]
    verdict = detect_failure(informative, "Nothing references that symbol.")
    assert not verdict.failed


def test_timeout_text_counts_as_error() -> None:
    timed = [{"tool_name": "bash", "arguments": {},
              "result": "command timed out after 120s"}]
    verdict = detect_failure(timed, "The script hung.")
    assert "unrecovered_tool_error" in verdict.matched_patterns


# ---------------------------------------------------------------------------
# Repetition
# ---------------------------------------------------------------------------

def _calls(n: int, args: dict) -> list[dict]:
    return [
        {"tool_name": "read_file", "arguments": dict(args),
         "result": "data", "is_error": False}
        for _ in range(n)
    ]


def test_repetition_threshold_boundary() -> None:
    below = detect_failure(_calls(REPETITION_THRESHOLD - 1, {"path": "x"}),
                           "The file holds the same two lines both times I checked.")
    assert "repetition" not in below.matched_patterns

    at = detect_failure(_calls(REPETITION_THRESHOLD, {"path": "x"}),
                        "The file holds the same two lines every time I checked.")
    assert "repetition" in at.matched_patterns


def test_repetition_canonicalizes_dict_order() -> None:
    results = [
        {"tool_name": "grep", "arguments": {"a": 1, "b": 2}, "result": "", "is_error": False},
        {"tool_name": "grep", "arguments": {"b": 2, "a": 1}, "result": "", "is_error": False},
        {"tool_name": "grep", "arguments": {"a": 1, "b": 2}, "result": "", "is_error": False},
    ]
    verdict = detect_failure(results, "Same empty result for that pattern all three times.")
    assert "repetition" in verdict.matched_patterns


def test_different_args_do_not_count_as_repetition() -> None:
    results = [
        {"tool_name": "read_file", "arguments": {"path": f"f{i}.py"},
         "result": "data", "is_error": False}
        for i in range(5)
    ]
    verdict = detect_failure(results, "All five modules import the helper from the same place.")
    assert "repetition" not in verdict.matched_patterns


# ---------------------------------------------------------------------------
# Clarification stall + empty reply
# ---------------------------------------------------------------------------

def test_stall_boundary_lengths() -> None:
    short_q = "A" * (CLARIFICATION_STALL_MAX_CHARS - 1) + "?"
    assert len(short_q) == CLARIFICATION_STALL_MAX_CHARS
    assert "clarification_stall" in detect_failure([], short_q).matched_patterns

    long_q = "A" * CLARIFICATION_STALL_MAX_CHARS + "?"
    assert "clarification_stall" not in detect_failure([], long_q).matched_patterns


def test_stall_requires_no_tool_activity() -> None:
    results = [{"tool_name": "glob", "arguments": {"pattern": "*.py"},
                "result": "a.py b.py", "is_error": False}]
    verdict = detect_failure(results, "Which of the two should I edit?")
    assert "clarification_stall" not in verdict.matched_patterns


def test_empty_reply_requires_tool_activity() -> None:
    # Empty reply with NO tool activity is out of Tier-1 scope.
    assert not detect_failure([], "").failed
    assert not detect_failure([], "   ").failed


def test_verdict_shape() -> None:
    verdict = detect_failure([], "Done — the rename is complete and the suite passes.")
    assert isinstance(verdict, FailureVerdict)
    assert verdict.failed is False
    assert verdict.reasons == []
    assert verdict.matched_patterns == []
