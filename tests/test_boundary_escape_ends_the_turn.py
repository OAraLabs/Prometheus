"""A write that lands outside the permitted area ends the turn.

WHY THIS LAYER EXISTS
---------------------
On 2026-08-13 the workspace boundary started working (#176) and was routed
around within the hour. ``write_file`` was refused; the model wrote the file
anyway with ``bash -c "echo … > path"`` in the same turn, and said so:
*"write_file has a workspace gate that blocks paths outside prometheus-deploy,
so I used bash as a workaround."*

``bash`` is gated on its command string and its own cwd, never on the paths a
command writes to, and no parser fixes that — a command's filesystem effects
live in the programs it invokes (``python3 x.py``, ``make``, ``tar -x``,
``sh -c "$VAR"``). See ``audits/20260813T040000Z-bash-boundary-survey.md``.

So enforcement moves to the outcome layer, where the unit is the CAPABILITY
rather than the tool: the verifier's before/after ``os.stat`` diff sees the
filesystem move regardless of which tool moved it.

DETECTION, NOT CONTAINMENT — AND THE WORDING IS PART OF THE CONTRACT
--------------------------------------------------------------------
``_Snapshot`` holds ``exists``/``size``/``mtime``/``mode`` and no content, so
nothing here can undo a write. The strongest available response is to end the
turn, and the message must say the change LANDED and cannot be restored. A
detection layer that reads like a prevention layer is the overclaim PR #177
removed from the workspace boundary — repeating it one layer in would be
worse, because here the reader has just been told something was "blocked".
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator

import pytest

from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.hooks.file_mutation_verifier import FileMutationVerifier
from prometheus.permissions.checker import SecurityGate
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiTextDeltaEvent,
    ModelProvider,
)


class _Scripted(ModelProvider):
    """Emits one bash call, then plain text so the turn ends."""

    def __init__(self, command: str):
        self._command = command
        self._n = 0

    async def stream_message(self, request: ApiMessageRequest) -> AsyncIterator:
        self._n += 1
        if self._n == 1:
            yield ApiMessageCompleteEvent(
                message=ConversationMessage(
                    role="assistant",
                    content=[ToolUseBlock(id="t1", name="bash",
                                          input={"command": self._command})],
                ),
                usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                stop_reason="tool_calls",
            )
        else:
            yield ApiTextDeltaEvent(text="done")
            yield ApiMessageCompleteEvent(
                message=ConversationMessage(
                    role="assistant", content=[TextBlock(text="done")]),
                usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                stop_reason="stop",
            )


def _run(tmp_path: Path, command: str, *, roots=None):
    """Drive the REAL run_loop with a real gate and a real verifier."""
    from prometheus.__main__ import create_tool_registry

    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    gate = SecurityGate(
        denied_paths=[str(tmp_path / "denied")],
        workspace_root=roots if roots is not None else [str(ws)],
    )
    # The registry's BashTool must see the SAME workspace roots the gate
    # does. With an empty security_cfg it falls back to the shipped default
    # (~/.prometheus/workspace), which does not exist on CI runners — every
    # bash spawn then died with FileNotFoundError instead of running the
    # command under test.
    gate_roots = roots if roots is not None else [str(ws)]
    ctx = LoopContext(
        provider=_Scripted(command), model="t", system_prompt="", max_tokens=256,
        tool_registry=create_tool_registry({"workspace_root": gate_roots}, gate),
        permission_checker=gate,
        file_mutation_verifier=FileMutationVerifier(),
        cwd=ws,
    )
    texts: list[str] = []

    async def go():
        async for event, _ in run_loop(ctx, [
            ConversationMessage.from_user_text("do it")
        ], session_id="s"):
            msg = getattr(event, "message", None)
            if msg is not None and getattr(msg, "text", None):
                texts.append(msg.text)

    asyncio.run(go())
    return texts


# ── THE case: bash writes outside the boundary ─────────────────────────────

def test_a_bash_redirect_outside_the_workspace_ends_the_turn(tmp_path):
    """The exact bypass observed live, now terminal."""
    target = tmp_path / "outside" / "f.txt"
    target.parent.mkdir(parents=True)
    texts = _run(tmp_path, f"echo -n x > {target}")
    assert target.exists(), "precondition: the write must actually land"
    final = texts[-1]
    assert "TURN ENDED" in final, f"the turn did not end on the escape: {final!r}"
    assert str(target) in final, "the message must name the file"


def test_the_message_says_it_could_not_be_undone(tmp_path):
    """The wording IS the feature. This layer cannot restore anything, and a
    message implying otherwise is the #177 overclaim one layer in."""
    target = tmp_path / "outside" / "f.txt"
    target.parent.mkdir(parents=True)
    final = _run(tmp_path, f"echo -n x > {target}")[-1]
    low = final.lower()
    assert "cannot be undone" in low
    assert "already happened" in low
    assert "nothing was blocked or prevented" in low
    for forbidden in ("was blocked", "was prevented", "refused the write",
                      "has been reverted", "restored"):
        assert forbidden not in low.replace("nothing was blocked or prevented", ""), (
            f"the message implies prevention it did not perform: {forbidden!r}"
        )


def test_a_write_into_a_denied_path_ends_the_turn(tmp_path):
    denied = tmp_path / "denied"
    denied.mkdir()
    target = denied / "f.txt"
    final = _run(tmp_path, f"echo -n x > {target}")[-1]
    assert "TURN ENDED" in final and str(target) in final


def test_every_escaped_path_is_named(tmp_path):
    """One line per file: a count without the paths is not actionable."""
    out = tmp_path / "outside"
    out.mkdir()
    a, b = out / "a.txt", out / "b.txt"
    final = _run(tmp_path, f"echo -n x > {a} && echo -n y > {b}")[-1]
    assert str(a) in final and str(b) in final


# ── ADMISSION: ordinary work must not be touched ───────────────────────────

def test_a_write_inside_the_workspace_does_not_end_the_turn(tmp_path):
    """The direction that decides whether this survives contact. A layer that
    ends turns on ordinary work gets disabled within a day."""
    target = tmp_path / "ws" / "ok.txt"
    texts = _run(tmp_path, f"echo -n x > {target}")
    assert target.exists()
    assert not any("TURN ENDED" in t for t in texts), (
        f"an in-workspace write ended the turn: {texts}"
    )


def test_a_turn_that_touches_nothing_is_unaffected(tmp_path):
    texts = _run(tmp_path, "echo hello")
    assert not any("TURN ENDED" in t for t in texts)


def test_a_read_outside_the_workspace_does_not_end_the_turn(tmp_path):
    """Only CHANGES count. Reading is not an escape, and treating it as one
    would end most turns."""
    outside = tmp_path / "outside"
    outside.mkdir()
    src = outside / "src.txt"
    src.write_text("hello\n")
    texts = _run(tmp_path, f"cat {src}")
    assert not any("TURN ENDED" in t for t in texts)


def test_no_gate_means_no_teeth(tmp_path):
    """Back-compat: contexts without a permission_checker (benchmarks, evals)
    must behave exactly as before."""
    from prometheus.__main__ import create_tool_registry

    ws = tmp_path / "ws"
    ws.mkdir()
    target = tmp_path / "outside" / "f.txt"
    target.parent.mkdir(parents=True)
    ctx = LoopContext(
        provider=_Scripted(f"echo -n x > {target}"), model="t", system_prompt="",
        max_tokens=256, tool_registry=create_tool_registry({}, None),
        file_mutation_verifier=FileMutationVerifier(), cwd=ws,
    )
    texts: list[str] = []

    async def go():
        async for event, _ in run_loop(ctx, [
            ConversationMessage.from_user_text("do it")], session_id="s"):
            msg = getattr(event, "message", None)
            if msg is not None and getattr(msg, "text", None):
                texts.append(msg.text)

    asyncio.run(go())
    assert not any("TURN ENDED" in t for t in texts)


def test_the_fmv_summary_still_reaches_the_model(tmp_path):
    """The teeth must not swallow the report. The agent should see BOTH what
    it touched and why the turn stopped."""
    target = tmp_path / "outside" / "f.txt"
    target.parent.mkdir(parents=True)
    texts = _run(tmp_path, f"echo -n x > {target}")
    assert any("TURN ENDED" in t for t in texts)
    # the verifier's own summary is appended as an injected turn, not an
    # assistant message, so assert it did not suppress the terminal one
    assert texts[-1].startswith("TURN ENDED")


# ── the accessor's own contract ────────────────────────────────────────────

def test_landed_paths_reports_only_real_changes(tmp_path):
    """A claimed write that produced nothing is a REPORTING matter, not a
    boundary violation — nothing escaped, so nothing should end a turn."""
    fmv = FileMutationVerifier()
    f = tmp_path / "f.txt"
    f.write_text("a\n")
    fmv.pre_tool_use("write_file", {"path": str(f)}, "t1", turn_key="k")
    fmv.post_tool_use("write_file", {"path": str(f)}, "t1",
                      output="", is_error=False, turn_key="k")
    assert fmv.landed_paths(turn_key="k") == [], (
        "an unchanged file was reported as a landed mutation"
    )


def test_landed_paths_does_not_drain_the_turn(tmp_path):
    """It is read before post_turn, which pops the record — draining here
    would silently empty the summary the model receives."""
    fmv = FileMutationVerifier()
    f = tmp_path / "f.txt"
    fmv.pre_tool_use("write_file", {"path": str(f)}, "t1", turn_key="k")
    f.write_text("new\n")
    fmv.post_tool_use("write_file", {"path": str(f)}, "t1",
                      output="", is_error=False, turn_key="k")
    assert fmv.landed_paths(turn_key="k") == [str(f)]
    assert fmv.landed_paths(turn_key="k") == [str(f)], "second read was empty"
    assert fmv.post_turn(turn_key="k"), "post_turn found nothing left to render"


def test_a_path_the_gate_cannot_classify_is_logged_not_swallowed(tmp_path, caplog):
    """CROSS-CUTTING §8: the direction must be CHOSEN, not inherited from
    whatever the exception does.

    An unclassifiable path is not treated as an escape — ending a turn on a
    classification error is an over-refusal, and this layer only detects. But
    a silent `continue` is fail-open detection with no trace: the coverage has
    a hole and nobody can see it. So it warns, and names the path.
    """
    import logging

    from prometheus.__main__ import create_tool_registry

    class _AngryGate(SecurityGate):
        """Raises only for the BOUNDARY check, not for tool dispatch.

        Written to raise unconditionally first, and the RuntimeError escaped
        the whole run — because the loop also calls evaluate() to permit the
        bash call itself. A fake that breaks more than the thing under test
        tells you nothing about the thing under test.
        """

        def evaluate(self, tool_name, **kw):
            if tool_name == "write_file" and kw.get("file_path"):
                raise RuntimeError("classifier exploded")
            return super().evaluate(tool_name, **kw)

    ws = tmp_path / "ws"
    ws.mkdir()
    target = tmp_path / "outside" / "f.txt"
    target.parent.mkdir(parents=True)
    gate = _AngryGate(workspace_root=[str(ws)])
    ctx = LoopContext(
        provider=_Scripted(f"echo -n x > {target}"), model="t", system_prompt="",
        max_tokens=256, tool_registry=create_tool_registry({"workspace_root": [str(ws)]}, None),
        permission_checker=gate,
        file_mutation_verifier=FileMutationVerifier(), cwd=ws,
    )
    texts: list[str] = []

    async def go():
        async for event, _ in run_loop(ctx, [
            ConversationMessage.from_user_text("do it")], session_id="s"):
            msg = getattr(event, "message", None)
            if msg is not None and getattr(msg, "text", None):
                texts.append(msg.text)

    with caplog.at_level(logging.WARNING):
        asyncio.run(go())

    assert not any("TURN ENDED" in t for t in texts), (
        "an unclassifiable path ended the turn — that is over-refusal"
    )
    assert any("could not classify" in r.message and str(target) in r.getMessage()
               for r in caplog.records), (
        "the coverage gap was swallowed silently"
    )
