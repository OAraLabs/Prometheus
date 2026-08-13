"""An APPROVE decision must reach the operator, not become a refusal.

THE DEFECT
----------
``agent_loop`` prompts only when ``context.permission_prompt is not None``.
That field was populated by **no construction site on any surface** — web,
Telegram, CLI, all of them. So every APPROVE fell to the ``else`` branch and
became a refusal-with-explanation, and the operator was never offered the
choice the decision exists to create.

It had a twin: ``daemon.py`` assigns ``security_gate._approval_queue``, and
nothing in ``permissions/`` ever read it — while ``ApprovalQueue``'s own class
docstring says *"Wire into SecurityGate"*. Two orphans facing each other
across one missing line, which is the same shape as FL-4's ``start_task``:
detector, hooks and store all present, one call absent.

Observed live on 2026-08-13: a write outside the workspace logged
``[AUDIT] CONFIRM_PENDING`` and ``/api/approvals`` stayed ``[]``, so Beacon —
whose approval card, poll, notification and tray badge are all complete — had
nothing to render.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from prometheus.engine.agent_loop import LoopContext, _execute_tool_call
from prometheus.permissions.approval_queue import ApprovalResult
from prometheus.permissions.checker import SecurityGate


class _FakeQueue:
    """Stands in for ApprovalQueue. Records what it was asked."""

    def __init__(self, answer: ApprovalResult | Exception):
        self.answer = answer
        self.asked: list[tuple[str, str]] = []

    async def request_approval(self, tool_name, description, chat_id=None):
        self.asked.append((tool_name, description))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def _ctx(tmp_path: Path, queue):
    from prometheus.__main__ import create_tool_registry

    gate = SecurityGate(workspace_root=[str(tmp_path / "ws")])
    gate._approval_queue = queue
    return LoopContext(
        provider=None, model="t", system_prompt="", max_tokens=512,
        tool_registry=create_tool_registry({}, gate),
        permission_checker=gate,
    )


def _write_outside(ctx, tmp_path: Path):
    target = tmp_path / "outside" / "f.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(_execute_tool_call(
        ctx, "write_file", "t1", {"path": str(target), "content": "x\n"}))
    return result, target


def test_an_approve_decision_asks_the_operator(tmp_path):
    """THE regression test. Before this, the queue was never consulted."""
    queue = _FakeQueue(ApprovalResult.APPROVED)
    result, target = _write_outside(_ctx(tmp_path, queue), tmp_path)
    assert queue.asked, (
        "the APPROVE decision never reached the approval queue — it became a "
        "refusal, and Beacon had nothing to render"
    )
    tool_name, reason = queue.asked[0]
    assert tool_name == "write_file"
    assert "outside workspace" in reason, (
        f"the operator must be told WHY, got {reason!r}"
    )
    assert not result.is_error and target.exists(), "APPROVED did not proceed"


def test_denied_means_the_write_does_not_happen(tmp_path):
    queue = _FakeQueue(ApprovalResult.DENIED)
    result, target = _write_outside(_ctx(tmp_path, queue), tmp_path)
    assert queue.asked
    assert result.is_error and not target.exists()


def test_timeout_is_a_refusal_not_a_pass(tmp_path):
    """A prompt nobody answers must not become consent."""
    queue = _FakeQueue(ApprovalResult.TIMEOUT)
    result, target = _write_outside(_ctx(tmp_path, queue), tmp_path)
    assert result.is_error and not target.exists()


def test_a_broken_queue_refuses(tmp_path):
    """Fail CLOSED. A prompt that degrades to 'yes' when its transport breaks
    is worse than no prompt (CROSS-CUTTING §8 — fail-by-exception is the
    third state nobody chose)."""
    queue = _FakeQueue(RuntimeError("telegram down"))
    result, target = _write_outside(_ctx(tmp_path, queue), tmp_path)
    assert result.is_error and not target.exists()


def test_no_queue_refuses(tmp_path):
    from prometheus.__main__ import create_tool_registry

    gate = SecurityGate(workspace_root=[str(tmp_path / "ws")])
    ctx = LoopContext(
        provider=None, model="t", system_prompt="", max_tokens=512,
        tool_registry=create_tool_registry({}, gate),
        permission_checker=gate,
    )
    result, target = _write_outside(ctx, tmp_path)
    assert result.is_error and not target.exists()


def test_an_explicit_permission_prompt_still_wins(tmp_path):
    """The gate's queue is a FALLBACK. A caller that supplies its own prompt
    must keep it — otherwise wiring one becomes impossible."""
    queue = _FakeQueue(ApprovalResult.DENIED)
    ctx = _ctx(tmp_path, queue)
    asked: list = []

    async def explicit(tool_name, reason):
        asked.append(tool_name)
        return True

    ctx.permission_prompt = explicit
    result, target = _write_outside(ctx, tmp_path)
    assert asked == ["write_file"], "the explicit prompt was not used"
    assert not queue.asked, "the fallback fired despite an explicit prompt"
    assert not result.is_error and target.exists()


def test_an_in_workspace_write_never_asks(tmp_path):
    """ADMISSION. Prompting on ordinary work is how a control gets disabled."""
    queue = _FakeQueue(ApprovalResult.APPROVED)
    ctx = _ctx(tmp_path, queue)
    (tmp_path / "ws").mkdir()
    target = tmp_path / "ws" / "ok.txt"
    result = asyncio.run(_execute_tool_call(
        ctx, "write_file", "t1", {"path": str(target), "content": "x\n"}))
    assert not result.is_error and target.exists()
    assert not queue.asked, f"an in-workspace write asked for approval: {queue.asked}"


def test_a_denied_path_never_asks(tmp_path):
    """A DENY must not be offered as a choice — that is the difference
    between denied_paths and workspace_root, and the whole reason both
    exist."""
    denied = tmp_path / "denied"
    denied.mkdir()
    from prometheus.__main__ import create_tool_registry

    queue = _FakeQueue(ApprovalResult.APPROVED)   # would say yes if asked
    gate = SecurityGate(denied_paths=[str(denied)],
                        workspace_root=[str(tmp_path / "ws")])
    gate._approval_queue = queue
    ctx = LoopContext(
        provider=None, model="t", system_prompt="", max_tokens=512,
        tool_registry=create_tool_registry({}, gate),
        permission_checker=gate,
    )
    target = denied / "f.txt"
    result = asyncio.run(_execute_tool_call(
        ctx, "write_file", "t1", {"path": str(target), "content": "x\n"}))
    assert result.is_error and not target.exists()
    assert not queue.asked, (
        "a DENIED path was offered for approval — a denial that can be "
        "approved is not a denial"
    )
