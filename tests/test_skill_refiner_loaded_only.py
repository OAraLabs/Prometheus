"""SkillRefiner refines a skill only after the task loaded it and completed.

The hook it replaces (``maybe_refine_recent``) took the most recently MODIFIED
auto skill after any task with 3+ tool calls and told the model "A skill was
used to guide a task". On the mini that was 208 refinement calls, none of
them after a real load (docs/audits/SKILL-USAGE.md §6). Now the hook reads
the task's own trace: a successful ``skill`` load of an auto skill, no failed
tool call after it, and a final reply.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from prometheus.engine.agent_loop import AgentLoop
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.learning.skill_refiner import SkillRefiner
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult


def _auto_dir(tmp_path: Path, *names: str) -> Path:
    d = tmp_path / "auto"
    d.mkdir(exist_ok=True)
    for n in names:
        (d / f"{n}.md").write_text(f"---\nname: {n}\ndescription: does {n}\n---\n# {n}\n1. step\n")
    return d


def _load(name: str, *, error: bool = False) -> dict:
    return {"tool_name": "skill", "tool_input": {"name": name}, "result": "---", "is_error": error}


def _call(tool: str = "bash", *, error: bool = False) -> dict:
    return {"tool_name": tool, "tool_input": {"command": "true"}, "result": "ok", "is_error": error}


@pytest.fixture
def refined(monkeypatch) -> list[str]:
    calls: list[str] = []

    async def fake_maybe_refine(self, skill_path, tool_trace, outcome):
        calls.append(Path(skill_path).name)
        return True

    monkeypatch.setattr(SkillRefiner, "maybe_refine", fake_maybe_refine)
    return calls


def _refine(refiner: SkillRefiner, trace: list[dict], final: str = "done") -> bool:
    return asyncio.run(refiner.maybe_refine_loaded("the task", trace, final))


class TestOnlyALoadedSkillIsRefined:
    def test_no_load_means_no_refinement_even_with_a_fresh_auto_skill(self, tmp_path, refined):
        refiner = SkillRefiner(MagicMock(), auto_dir=_auto_dir(tmp_path, "fresh"), min_tool_calls=3)
        assert _refine(refiner, [_call(), _call(), _call(), _call()]) is False
        assert refined == []

    def test_a_loaded_auto_skill_is_refined(self, tmp_path, refined):
        refiner = SkillRefiner(MagicMock(), auto_dir=_auto_dir(tmp_path, "x"), min_tool_calls=3)
        assert _refine(refiner, [_load("x"), _call(), _call()]) is True
        assert refined == ["x.md"]

    def test_the_loaded_skill_not_the_newest_file(self, tmp_path, refined):
        d = _auto_dir(tmp_path, "a")
        _auto_dir(tmp_path, "b")  # written after a: the old hook would have picked b
        refiner = SkillRefiner(MagicMock(), auto_dir=d, min_tool_calls=3)
        _refine(refiner, [_load("a"), _call(), _call()])
        assert refined == ["a.md"]

    def test_user_and_builtin_skills_are_never_rewritten(self, tmp_path, refined):
        refiner = SkillRefiner(MagicMock(), auto_dir=_auto_dir(tmp_path, "x"), min_tool_calls=3)
        assert _refine(refiner, [_load("commit"), _call(), _call()]) is False
        assert refined == []

    def test_each_loaded_skill_once(self, tmp_path, refined):
        refiner = SkillRefiner(MagicMock(), auto_dir=_auto_dir(tmp_path, "a", "b"), min_tool_calls=3)
        _refine(refiner, [_load("a"), _call(), _load("a"), _load("b"), _call()])
        assert sorted(refined) == ["a.md", "b.md"]

    def test_names_match_like_the_registry_case_insensitively(self, tmp_path, refined):
        refiner = SkillRefiner(MagicMock(), auto_dir=_auto_dir(tmp_path, "release-check"),
                               min_tool_calls=3)
        _refine(refiner, [_load("Release-Check"), _call(), _call()])
        assert refined == ["release-check.md"]

    def test_the_frontmatter_name_finds_a_differently_named_file(self, tmp_path, refined):
        d = tmp_path / "auto"
        d.mkdir()
        (d / "release-check-1790000000.md").write_text(
            "---\nname: release-check\ndescription: d\n---\nbody\n")
        refiner = SkillRefiner(MagicMock(), auto_dir=d, min_tool_calls=3)
        _refine(refiner, [_load("release-check"), _call(), _call()])
        assert refined == ["release-check-1790000000.md"]


class TestOnlyACompletedTaskCounts:
    def test_a_failed_load_does_not_count(self, tmp_path, refined):
        refiner = SkillRefiner(MagicMock(), auto_dir=_auto_dir(tmp_path, "x"), min_tool_calls=3)
        assert _refine(refiner, [_load("x", error=True), _call(), _call()]) is False
        assert refined == []

    def test_a_failure_after_the_load_means_the_task_did_not_complete(self, tmp_path, refined):
        refiner = SkillRefiner(MagicMock(), auto_dir=_auto_dir(tmp_path, "x"), min_tool_calls=3)
        assert _refine(refiner, [_load("x"), _call(), _call(error=True)]) is False
        assert refined == []

    def test_a_failure_before_the_load_is_not_held_against_it(self, tmp_path, refined):
        refiner = SkillRefiner(MagicMock(), auto_dir=_auto_dir(tmp_path, "x"), min_tool_calls=3)
        assert _refine(refiner, [_call(error=True), _load("x"), _call(), _call()]) is True
        assert refined == ["x.md"]

    def test_no_final_reply_means_not_completed(self, tmp_path, refined):
        refiner = SkillRefiner(MagicMock(), auto_dir=_auto_dir(tmp_path, "x"), min_tool_calls=3)
        assert _refine(refiner, [_load("x"), _call(), _call()], final="  ") is False
        assert refined == []

    def test_short_traces_are_still_skipped(self, tmp_path, refined):
        refiner = SkillRefiner(MagicMock(), auto_dir=_auto_dir(tmp_path, "x"), min_tool_calls=5)
        assert _refine(refiner, [_load("x"), _call(), _call()]) is False
        assert refined == []


def test_the_newest_file_heuristic_is_gone():
    assert not hasattr(SkillRefiner, "maybe_refine_recent")


# ---------------------------------------------------------------------------
# run_async's trace now says which skill a `skill` call loaded
# ---------------------------------------------------------------------------

class _EchoInput(BaseModel):
    name: str


class _FakeSkillTool(BaseTool):
    name = "skill"
    description = "fake"
    input_model = _EchoInput

    def is_read_only(self, arguments) -> bool:  # noqa: ANN001
        return True

    async def execute(self, arguments, context):  # noqa: ANN001
        return ToolResult(output=f"# {arguments.name}")


class _LoadsOnce(ModelProvider):
    def __init__(self) -> None:
        self.calls = 0

    async def stream_message(self, request):  # noqa: ANN001
        self.calls += 1
        content = ([ToolUseBlock(id="s1", name="skill", input={"name": "release-check"})]
                   if self.calls == 1 else [TextBlock(text="done")])
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=content),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1), stop_reason="stop",
        )


def test_the_post_task_trace_carries_each_calls_input():
    registry = ToolRegistry()
    registry.register(_FakeSkillTool())
    loop = AgentLoop(provider=_LoadsOnce(), model="stub", tool_registry=registry)
    seen: list[list[dict]] = []

    async def hook(task, trace, final):
        seen.append(list(trace))

    loop.add_post_task_hook(hook)
    asyncio.run(loop.run_async("SYSTEM", "go", session_id="telegram:1"))
    [trace] = seen
    [entry] = trace
    assert entry["tool_name"] == "skill"
    assert entry["tool_input"] == {"name": "release-check"}
    assert entry["is_error"] is False
