"""The smoke runner's retry and soft-pass rules, tested without a model.

`scripts/smoke_test_tool_calling.py` drives a real local model, so its own
behaviour cannot be pinned by running it — that is the whole reason it was
flaky. These exercise the two rules that decide pass/fail directly, with the
model turn replaced by a scripted sequence.
"""

from __future__ import annotations

import asyncio
import importlib.util
import pathlib

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "smoke_test_tool_calling.py"
pytest.importorskip("fastapi")


def _load():
    spec = importlib.util.spec_from_file_location("_smoke_under_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


smoke = _load()


class _FakeResult:
    """Stands in for RunResult: only .text and .messages are read."""

    def __init__(self, text: str, tools: list[str]):
        self.text = text
        self.messages = [
            type("M", (), {"content": [
                type("B", (), {"type": "tool_use", "name": t})() for t in tools
            ]})()
        ]


def _runner(script: list[tuple[str, list[str]]]):
    """A SmokeTestRunner whose model turns are a fixed sequence."""
    r = smoke.SmokeTestRunner.__new__(smoke.SmokeTestRunner)
    r.results = []
    r.verbose = False
    turns = list(script)

    async def run_agent(message, max_iterations=10):
        text, tools = turns.pop(0)
        return {"result": _FakeResult(text, tools), "elapsed_ms": 1.0,
                "text": text, "tools": tools}

    r.run_agent = run_agent
    return r


def test_a_transient_failure_is_retried_and_reported():
    """THE RETRY. First turn calls nothing; the second does the work."""
    r = _runner([("I'll do it.", []), ("done", ["bash"])])
    res = asyncio.run(r.run_test(
        name="t", category="c", message="m", expect_tools=["bash"], attempts=3,
    ))
    assert res.passed
    assert "passed on attempt 2/3" in res.error, (
        "a retry must be reported, not swallowed — a test that quietly needs "
        "three tries is a degradation nobody can see"
    )


def test_a_persistent_failure_still_fails_after_every_attempt():
    """Retrying must not turn a broken pipeline green."""
    r = _runner([("nope", []), ("nope", []), ("nope", [])])
    res = asyncio.run(r.run_test(
        name="t", category="c", message="m", expect_tools=["bash"], attempts=3,
    ))
    assert not res.passed
    assert "Expected tool(s) ['bash']" in res.error


def test_wording_alone_is_forgiven_when_the_effects_are_right(tmp_path):
    """THE FLAKE. The file assertion passes; only the phrase is missing.

    This hard-failed before, because the soft-pass rule keyed on which
    PARAMETERS were supplied rather than which assertion failed.
    """
    target = tmp_path / "hello.txt"
    target.write_text("smoke test passed\n")
    r = _runner([("Now let me read it back.", ["write_file"])])
    res = asyncio.run(r.run_test(
        name="t", category="c", message="m",
        expect_in_output="smoke test passed",
        expect_file_exists=str(target),
        expect_file_contains="smoke test passed",
        attempts=1,
    ))
    assert res.passed
    assert "SOFT PASS" in res.error


def test_a_missing_effect_is_never_forgiven(tmp_path):
    """The other side of the same rule: no file means no pass, ever."""
    missing = tmp_path / "never-written.txt"
    r = _runner([("smoke test passed", ["write_file"])] * 3)
    res = asyncio.run(r.run_test(
        name="t", category="c", message="m",
        expect_in_output="smoke test passed",
        expect_file_exists=str(missing),
        attempts=3,
    ))
    assert not res.passed
    assert "SOFT PASS" not in res.error


def test_tools_called_is_recorded():
    """The field existed from the start and was never written."""
    r = _runner([("ok", ["bash", "read_file"])])
    res = asyncio.run(r.run_test(name="t", category="c", message="m", attempts=1))
    assert res.tools_called == ["bash", "read_file"]
