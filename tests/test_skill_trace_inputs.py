"""SkillCreator and SkillRefiner see what each tool call was given (option C3).

Both prompts rendered every call as ``tool({})``: the formatters read an
``arguments`` key that ``AgentLoop.run_async`` never sets. A skill written
from "bash({}) → ok, bash({}) → ok, bash({}) → ok" can only be generic.
C1 put each call's real input in the trace under ``tool_input``; these tests
pin that the prompts now show it, cut to a fixed length and passed through
the redactor first, so a pasted token never reaches the learning loop's
model and a cut can never leave half of one behind.

The fake token is assembled at runtime: .githooks/pre-commit scans whole files.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

from prometheus.learning.skill_creator import SkillCreator
from prometheus.learning.skill_refiner import SkillRefiner
from prometheus.learning.trace_format import INPUT_CHARS, format_trace
from prometheus.security import REDACTED

TOKEN = "gh" + "p_" + ("aB3xY9" * 6)


class _Capture:
    """A provider that records every prompt and answers with a fixed text."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    async def stream_message(self, request):  # noqa: ANN001
        from prometheus.providers.base import ApiTextDeltaEvent

        self.prompts.append("".join(block.text for block in request.messages[0].content))
        yield ApiTextDeltaEvent(text=self.answer)


def _call(tool: str, tool_input, result: str = "ok", *, error: bool = False) -> dict:
    return {"tool_name": tool, "tool_input": tool_input, "result": result, "is_error": error}


TRACE = [
    _call("bash", {"command": "git status --short"}),
    _call("write_file", {"path": "deploy/compose.yaml", "content": "services: {}"}),
    _call("bash", {"command": "docker compose -f deploy/compose.yaml up -d"}, "started"),
]


def test_the_creator_prompt_shows_what_each_call_was_given(tmp_path):
    provider = _Capture("SKIP: test")
    creator = SkillCreator(provider, auto_dir=tmp_path, telemetry=MagicMock(),
                           similarity=MagicMock(available=False), catalog=lambda: [])
    asyncio.run(creator.maybe_create("bring the stack up", TRACE, "The stack is up."))
    [prompt] = provider.prompts
    assert 'bash({"command": "git status --short"}) → ok' in prompt
    assert 'write_file({"path": "deploy/compose.yaml", "content": "services: {}"})' in prompt
    assert "({})" not in prompt


def test_the_refiner_prompt_shows_what_each_call_was_given(tmp_path):
    skill = tmp_path / "bring-up.md"
    skill.write_text("---\nname: bring-up\ndescription: bring the stack up\n---\n# Bring up\n1. step\n")
    provider = _Capture("NO_CHANGE")
    refiner = SkillRefiner(provider, auto_dir=tmp_path)
    asyncio.run(refiner.maybe_refine(skill, TRACE, outcome="done"))
    [prompt] = provider.prompts
    assert 'bash({"command": "docker compose -f deploy/compose.yaml up -d"}) → started' in prompt
    assert "({})" not in prompt


def test_a_long_input_is_cut_to_the_stated_length():
    command = "echo " + "x" * 5000
    [line] = format_trace([_call("bash", {"command": command})]).splitlines()
    shown = line[len("1. bash("):line.index(") → ")]
    assert len(shown) == INPUT_CHARS and shown.endswith("…")
    assert shown.startswith('{"command": "echo xxx')


def test_an_input_is_redacted_before_it_is_cut():
    # The token straddles the cut. Cutting first would leave its first 19
    # characters: too short for the pattern, so they would reach the prompt.
    command = "x" * (INPUT_CHARS - 34) + " " + TOKEN
    text = format_trace([_call("bash", {"command": command})])
    assert TOKEN[:8] not in text


def test_inputs_and_results_are_redacted():
    text = format_trace([_call("bash", {"command": f"git push https://{TOKEN}@github.com/o/r"},
                               f"pushed with {TOKEN}")])
    assert TOKEN not in text and text.count(REDACTED) == 2


def test_a_trace_in_the_older_arguments_shape_still_renders():
    """Callers that build their own trace (the teacher, tests) used ``arguments``."""
    text = format_trace([{"tool_name": "bash", "arguments": {"command": "ls"}, "result": "a b"}])
    assert text == '1. bash({"command": "ls"}) → a b'


def test_a_string_input_is_shown_as_it_is():
    assert format_trace([_call("bash", "ls -la")]) == "1. bash(ls -la) → ok"


def test_an_input_that_is_not_json_serialisable_still_renders():
    text = format_trace([_call("read", {"path": Path("/tmp/a.txt")})])
    assert json.loads(text[len("1. read("):text.index(") → ")]) == {"path": "/tmp/a.txt"}


def test_the_creator_still_marks_failed_calls_and_the_refiner_does_not():
    trace = [_call("bash", {"command": "false"}, "exit 1", error=True)]
    assert SkillCreator._format_trace(trace).endswith("[ERROR]")
    assert not SkillRefiner._format_trace(trace).endswith("[ERROR]")
