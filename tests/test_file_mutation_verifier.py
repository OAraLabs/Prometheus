"""SPRINT-2 WS2 — File-mutation verifier functional tests.

The load-bearing case: a tool returns success but the bytes on disk didn't
change. The verifier MUST flag this with a "CLAIMED but NO CHANGE ON DISK"
marker so the model sees the silent failure on its next turn. If this
sprint ships and that case still slips by, the verifier was wasted effort.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from prometheus.hooks.file_mutation_verifier import (
    FileMutationVerifier,
    _extract_bash_paths,
    make_default_verifier,
)


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Path / bash extraction
# ---------------------------------------------------------------------------


class TestPathExtraction:
    def test_bash_redirect_write(self):
        out = _extract_bash_paths("echo hello > /tmp/x.txt")
        assert ("/tmp/x.txt", "redirect_write") in out

    def test_bash_redirect_append(self):
        out = _extract_bash_paths("printf 'log' >> /tmp/log.txt")
        assert ("/tmp/log.txt", "redirect_append") in out

    def test_bash_mkdir(self):
        out = _extract_bash_paths("mkdir -p /tmp/foo/bar")
        paths = [p for p, _ in out]
        assert "/tmp/foo/bar" in paths

    def test_bash_compound_command(self):
        """``a && b`` should produce mutations from both clauses."""
        out = _extract_bash_paths("mkdir /tmp/foo && touch /tmp/foo/bar")
        actions = {a for _, a in out}
        assert "mkdir" in actions
        assert "touch" in actions

    def test_bash_no_match_returns_empty(self):
        assert _extract_bash_paths("ls -la") == []

    def test_bash_mv_target(self):
        out = _extract_bash_paths("mv /tmp/a /tmp/b")
        # We track at least the destination — a true mv tracks both, but
        # the destination is the load-bearing "did this land?" signal.
        paths = [p for p, _ in out]
        assert "/tmp/b" in paths


# ---------------------------------------------------------------------------
# Lifecycle: pre / post / post_turn
# ---------------------------------------------------------------------------


class TestVerifierLifecycle:
    def test_disabled_verifier_is_a_noop(self, tmp_path: Path):
        v = FileMutationVerifier(enabled=False)
        target = tmp_path / "x.txt"
        v.pre_tool_use("file_write", {"file_path": str(target)}, "t1")
        target.write_text("hi", encoding="utf-8")
        v.post_tool_use(
            "file_write", {"file_path": str(target)}, "t1",
            output="wrote 2 bytes", is_error=False,
        )
        assert v.post_turn() is None  # disabled → no summary

    def test_empty_turn_returns_none(self):
        v = FileMutationVerifier()
        assert v.post_turn() is None

    def test_summary_resets_between_turns(self, tmp_path: Path):
        v = FileMutationVerifier()
        target = tmp_path / "x.txt"
        v.pre_tool_use("file_write", {"file_path": str(target)}, "t1")
        target.write_text("hi", encoding="utf-8")
        v.post_tool_use(
            "file_write", {"file_path": str(target)}, "t1",
            output="ok", is_error=False,
        )
        s1 = v.post_turn()
        assert s1 is not None
        # Second post_turn with no new mutations → None.
        assert v.post_turn() is None


# ---------------------------------------------------------------------------
# Detection — file_write happy path + the silent-failure case
# ---------------------------------------------------------------------------


class TestDetection:
    def test_detects_file_write(self, tmp_path: Path):
        v = FileMutationVerifier()
        target = tmp_path / "foo.py"
        v.pre_tool_use("file_write", {"file_path": str(target)}, "t1")
        # Real write happens here (this is what the file_write tool would do).
        target.write_text("def add(a, b): return a + b\n", encoding="utf-8")
        v.post_tool_use(
            "file_write", {"file_path": str(target)}, "t1",
            output=f"wrote {target.stat().st_size} bytes",
            is_error=False,
        )
        summary = v.post_turn()
        assert summary is not None
        assert str(target) in summary
        assert "✓" in summary
        assert "created" in summary

    def test_detects_bash_redirect(self, tmp_path: Path):
        v = FileMutationVerifier()
        target = tmp_path / "bash.txt"
        cmd = f"echo hello > {target}"
        v.pre_tool_use("bash", {"command": cmd}, "t2")
        target.write_text("hello\n", encoding="utf-8")
        v.post_tool_use(
            "bash", {"command": cmd}, "t2",
            output="", is_error=False,
        )
        summary = v.post_turn()
        assert summary is not None
        assert str(target) in summary
        assert "✓" in summary

    def test_detects_silent_failure_no_change_on_disk(self, tmp_path: Path):
        """THE load-bearing case: tool claimed success, disk unchanged.

        Setup: a pre-existing file, the tool says "I wrote 47 lines" but
        the bytes on disk are identical (or the file was never touched).
        Pre-fix shape this scenario simulated: file_write call returns
        success without actually performing the I/O. The verifier must
        flag this with the "CLAIMED but NO CHANGE ON DISK" marker."""
        v = FileMutationVerifier()
        target = tmp_path / "preexisting.py"
        target.write_text("# original\n", encoding="utf-8")

        v.pre_tool_use("file_write", {"file_path": str(target)}, "t3")
        # The tool CLAIMS to have written, but we deliberately don't
        # modify the file. This is the silent-failure shape.
        v.post_tool_use(
            "file_write", {"file_path": str(target)}, "t3",
            output="wrote 47 lines to preexisting.py",
            is_error=False,
        )
        summary = v.post_turn()
        assert summary is not None
        assert "CLAIMED but NO CHANGE ON DISK" in summary, (
            f"Verifier failed to flag the silent-failure case. Summary:\n{summary}"
        )
        assert "⚠" in summary
        assert str(target) in summary

    def test_detects_permission_denied(self, tmp_path: Path):
        v = FileMutationVerifier()
        target = tmp_path / "permission_denied.txt"
        v.pre_tool_use("file_write", {"file_path": str(target)}, "t4")
        # Tool reports failure.
        v.post_tool_use(
            "file_write", {"file_path": str(target)}, "t4",
            output="Permission denied: '/etc/hosts'",
            is_error=True,
        )
        summary = v.post_turn()
        assert summary is not None
        assert "✗" in summary
        assert "Permission denied" in summary


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


class TestTruncation:
    def test_truncates_at_configured_threshold(self, tmp_path: Path):
        v = FileMutationVerifier(truncate_after_n_mutations=3)
        for i in range(5):
            target = tmp_path / f"file_{i}.txt"
            v.pre_tool_use(
                "file_write", {"file_path": str(target)}, f"t{i}",
            )
            target.write_text(f"content_{i}", encoding="utf-8")
            v.post_tool_use(
                "file_write", {"file_path": str(target)}, f"t{i}",
                output="ok", is_error=False,
            )

        summary = v.post_turn()
        assert summary is not None
        assert "and 2 more" in summary
        assert "truncated at 3" in summary


# ---------------------------------------------------------------------------
# Config wiring
# ---------------------------------------------------------------------------


class TestConfigWiring:
    def test_make_default_verifier_with_no_config(self):
        v = make_default_verifier(None)
        assert v.enabled is True
        assert v.show_in_telegram is False

    def test_make_default_verifier_with_opt_out(self):
        v = make_default_verifier({
            "hooks": {
                "file_mutation_verifier": {
                    "enabled": False,
                    "show_in_telegram": True,
                    "truncate_after_n_mutations": 5,
                },
            },
        })
        assert v.enabled is False
        assert v.show_in_telegram is True
        assert v._truncate_n == 5


# ---------------------------------------------------------------------------
# Agent-loop integration
# ---------------------------------------------------------------------------


class TestAgentLoopIntegration:
    """Wire the verifier through LoopContext and confirm it observes a
    real tool call's filesystem effect end-to-end."""

    @pytest.mark.asyncio
    async def test_verifier_summary_appears_as_user_message_after_turn(
        self, tmp_path: Path,
    ):
        from prometheus.engine.agent_loop import LoopContext, run_loop
        from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
        from prometheus.engine.usage import UsageSnapshot
        from prometheus.providers.base import (
            ApiMessageCompleteEvent, ApiMessageRequest, ApiTextDeltaEvent,
            ModelProvider,
        )
        from typing import AsyncIterator

        target = tmp_path / "out.txt"

        # Tool registry: a single tool that actually writes to the path
        # (so the verifier's pre/post snapshots disagree).
        class _WriteTool:
            name = "file_write"
            description = "write file"
            class input_model:
                @staticmethod
                def model_validate(d):
                    class _A:
                        def __init__(self_, d):
                            self_.file_path = d["file_path"]
                            self_.content = d["content"]
                    return _A(d)
            def is_read_only(self, parsed): return False
            async def execute(self, parsed, ctx):
                from prometheus.tools.base import ToolResult
                Path(parsed.file_path).write_text(parsed.content, encoding="utf-8")
                return ToolResult(output=f"wrote {len(parsed.content)} bytes")

        class _R:
            def __init__(self): self._t = _WriteTool()
            def get(self, n): return self._t if n == "file_write" else None
            def get_tool(self, n): return self.get(n)
            def list_tools(self): return [self._t]
            def list_schemas(self): return [{"name": "file_write", "input_schema": {}}]

        class _Prov(ModelProvider):
            def __init__(self):
                self._call = 0
            async def stream_message(self, request) -> AsyncIterator:
                if self._call == 0:
                    msg = ConversationMessage(
                        role="assistant",
                        content=[ToolUseBlock(
                            id="c1", name="file_write",
                            input={"file_path": str(target), "content": "hello\n"},
                        )],
                    )
                    self._call += 1
                    yield ApiMessageCompleteEvent(
                        message=msg, usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                        stop_reason="tool_calls",
                    )
                else:
                    msg = ConversationMessage(
                        role="assistant", content=[TextBlock(text="done")],
                    )
                    self._call += 1
                    yield ApiMessageCompleteEvent(
                        message=msg, usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                        stop_reason="stop",
                    )

        v = FileMutationVerifier()
        ctx = LoopContext(
            provider=_Prov(), model="qwen-test",
            system_prompt="sys", max_tokens=1024,
            tool_registry=_R(),
            file_mutation_verifier=v,
        )
        messages = [ConversationMessage.from_user_text(
            f"write hello to {target}",
        )]
        async for _ in run_loop(ctx, messages):
            pass

        # Verifier-summary message landed at the end of the conversation.
        last = messages[-1]
        assert last.role == "user"
        assert "[FILE MUTATION VERIFIER]" in last.text
        assert str(target) in last.text
        assert "✓" in last.text
        # The actual file was indeed written, so it's a real success path.
        assert target.read_text(encoding="utf-8") == "hello\n"
