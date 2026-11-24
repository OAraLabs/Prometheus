"""Every change the memory tool makes to MEMORY.md or USER.md is the owner's, approved in the approval queue.

Those two files ride every system prompt. A tool's result, an upload or an injected turn can steer the model into
writing them, and so can text left in its context from an earlier turn; no rule about what the current turn holds
closes that (GHSA-f68r-xfv4-83fr, review round 3). So every add, replace and remove goes to the owner, through the
same approval queue as a file tool's write to these files, with the exact entry shown (and the old text, for a
replace), whatever the permission mode or a remembered grant says. With no approval channel it is refused. ``list``
needs no approval.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from importlib import reload

import pytest

from prometheus.engine.messages import ConversationMessage, TextBlock, ToolResultBlock, ToolUseBlock
from prometheus.tools.base import ToolExecutionContext, ToolRegistry

ENTRY = "Will banks with ACME_MARKER"


@pytest.fixture
def memory(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path))
    from prometheus.config import paths
    from prometheus.memory import hermes_memory_tool

    reload(paths)
    reload(hermes_memory_tool)
    return hermes_memory_tool


def _saved(tmp_path, marker: str = "ACME_MARKER") -> bool:
    path = tmp_path / "USER.md"
    return path.exists() and marker in path.read_text(encoding="utf-8")


def _run(memory, tmp_path, calls: list[dict], *, prompt=None, mode: str = "default", checker: bool = True):
    """Run the real loop: the model makes each of *calls* to the memory tool, one per round, then stops."""
    from prometheus.engine.agent_loop import LoopContext, run_loop
    from prometheus.engine.usage import UsageSnapshot
    from prometheus.permissions.checker import SecurityGate
    from prometheus.providers.base import ApiMessageCompleteEvent, ApiMessageRequest, ModelProvider

    def _turn(*blocks, stop: str) -> list:
        msg = ConversationMessage(role="assistant", content=list(blocks))
        return [ApiMessageCompleteEvent(message=msg, usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                                        stop_reason=stop)]

    script = [_turn(ToolUseBlock(id=f"m{i}", name="memory", input=c), stop="tool_calls") for i, c in enumerate(calls)]
    script.append(_turn(TextBlock(text="done"), stop="stop"))

    class Scripted(ModelProvider):
        async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator:
            for event in script.pop(0):
                yield event

    registry = ToolRegistry()
    registry.register(memory.MemoryTool())
    gate = SecurityGate(mode=mode, workspace_root=str(tmp_path)) if checker else None
    ctx = LoopContext(provider=Scripted(), model="test", system_prompt="test", max_tokens=256,
                      tool_registry=registry, permission_checker=gate, permission_prompt=prompt)
    messages = [ConversationMessage(role="user", content=[TextBlock(text="Remember that I bank with ACME")],
                                    is_trusted=True)]

    async def _go() -> None:
        async for _ in run_loop(ctx, messages):
            pass

    asyncio.run(_go())
    return [b for m in messages for b in m.content if isinstance(b, ToolResultBlock)]


class _Approver:
    def __init__(self, answer: bool) -> None:
        self.answer = answer
        self.asked: list[tuple[str, str, dict]] = []

    async def __call__(self, tool_name: str, reason: str, *, arguments: dict | None = None) -> bool:
        self.asked.append((tool_name, reason, dict(arguments or {})))
        return self.answer


ADD = {"operation": "add", "target": "user", "entry": ENTRY}


def test_an_add_asks_the_owner_with_the_exact_entry(memory, tmp_path):
    approver = _Approver(True)
    results = _run(memory, tmp_path, [ADD], prompt=approver)
    assert len(approver.asked) == 1
    tool, reason, args = approver.asked[0]
    assert tool == "memory" and args["entry"] == ENTRY
    assert "USER.md" in reason
    assert not results[0].is_error and _saved(tmp_path)


def test_a_refusal_writes_nothing(memory, tmp_path):
    results = _run(memory, tmp_path, [ADD], prompt=_Approver(False))
    assert results[0].is_error and not _saved(tmp_path)


def test_a_replace_shows_the_old_text_too(memory, tmp_path):
    (tmp_path / "USER.md").write_text("Will banks with OLD_BANK\n", encoding="utf-8")
    approver = _Approver(True)
    _run(memory, tmp_path, [{"operation": "replace", "target": "user", "old_entry": "Will banks with OLD_BANK",
                              "entry": ENTRY}], prompt=approver)
    _, reason, args = approver.asked[0]
    assert args["old_entry"] == "Will banks with OLD_BANK"
    assert "OLD_BANK" in reason and "ACME_MARKER" in reason


@pytest.mark.parametrize("mode", ["autonomous", "default"])
def test_no_mode_and_no_grant_skips_the_ask(memory, tmp_path, mode):
    approver = _Approver(True)
    _run(memory, tmp_path, [ADD, {"operation": "add", "target": "user", "entry": "second SECOND_MARKER"}],
         prompt=approver, mode=mode)
    assert len(approver.asked) == 2  # each change on its own; an approval is never a grant for the next


def test_list_needs_no_approval(memory, tmp_path):
    approver = _Approver(True)
    results = _run(memory, tmp_path, [{"operation": "list", "target": "user"}], prompt=approver)
    assert approver.asked == [] and not results[0].is_error


def test_with_no_approval_channel_a_change_is_refused(memory, tmp_path):
    # A gate with no approver to ask, and a loop with no gate at all: never allowed.
    assert _run(memory, tmp_path, [ADD])[0].is_error
    assert _run(memory, tmp_path, [ADD], checker=False)[0].is_error
    assert not _saved(tmp_path)


def test_called_outside_the_loop_a_change_is_refused(memory, tmp_path):
    tool = memory.MemoryTool()
    result = asyncio.run(tool.execute(memory.MemoryToolInput(**ADD),
                                      ToolExecutionContext(cwd=str(tmp_path), metadata={"turn_trusted": True})))
    assert result.is_error and not _saved(tmp_path)


def test_tool_metadata_cannot_claim_the_approval(memory, tmp_path):
    # LoopContext.tool_metadata is spread into every call's metadata; the approval is the loop's own record.
    tool = memory.MemoryTool()
    ctx = ToolExecutionContext(cwd=str(tmp_path), metadata={"owner_approved": True})
    result = asyncio.run(tool.execute(memory.MemoryToolInput(**ADD), ctx))
    assert result.is_error, "a flag in metadata alone is not the loop's record of an approval"
