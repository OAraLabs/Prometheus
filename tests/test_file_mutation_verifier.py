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
# Turn scoping — the reason this hook could not be shared across surfaces
# ---------------------------------------------------------------------------


class TestTurnScoping:
    """``run_daemon`` builds ONE verifier and every surface shares it —
    telegram, CLI, cron, and (since the web-bridge wiring) each concurrent
    Beacon turn. Before turn keys, ``_TurnRecord.mutations`` was a flat list
    that ``post_turn()`` drained globally: the turn that finished first
    reported the other's writes as its own and the second reported nothing.
    That inverts a feature whose entire job is checking that the writes YOU
    claimed actually landed."""

    @staticmethod
    def _write(v, path: Path, turn_key: str, call_id: str) -> None:
        v.pre_tool_use("file_write", {"file_path": str(path)}, call_id, turn_key=turn_key)
        path.write_text("x", encoding="utf-8")
        v.post_tool_use(
            "file_write", {"file_path": str(path)}, call_id,
            output="ok", is_error=False, turn_key=turn_key,
        )

    def test_concurrent_turns_do_not_see_each_others_mutations(self, tmp_path: Path):
        """The load-bearing case. Two turns interleaved the way concurrent
        Beacon sessions interleave: each summary names its OWN file only."""
        v = FileMutationVerifier()
        a, b = tmp_path / "turn-a.txt", tmp_path / "turn-b.txt"

        # Interleaved: A pre, B pre, A post, B post.
        v.pre_tool_use("file_write", {"file_path": str(a)}, "call-a", turn_key="A")
        v.pre_tool_use("file_write", {"file_path": str(b)}, "call-b", turn_key="B")
        a.write_text("a", encoding="utf-8")
        b.write_text("b", encoding="utf-8")
        v.post_tool_use(
            "file_write", {"file_path": str(a)}, "call-a",
            output="ok", is_error=False, turn_key="A",
        )
        v.post_tool_use(
            "file_write", {"file_path": str(b)}, "call-b",
            output="ok", is_error=False, turn_key="B",
        )

        sa = v.post_turn(turn_key="A")
        assert sa is not None
        assert str(a) in sa
        assert str(b) not in sa, "turn A reported turn B's write as its own"

        sb = v.post_turn(turn_key="B")
        assert sb is not None, (
            "turn B lost its mutations to turn A's drain — the pre-fix bug"
        )
        assert str(b) in sb
        assert str(a) not in sb

    def test_draining_one_turn_leaves_the_others_intact(self, tmp_path: Path):
        v = FileMutationVerifier()
        for i in range(3):
            self._write(v, tmp_path / f"t{i}.txt", f"T{i}", f"c{i}")
        assert v.live_turns == 3
        assert v.post_turn(turn_key="T1") is not None
        assert v.live_turns == 2
        # Draining T1 did not touch T0/T2.
        assert v.post_turn(turn_key="T0") is not None
        assert v.post_turn(turn_key="T2") is not None
        assert v.live_turns == 0

    def test_post_turn_drops_the_record(self, tmp_path: Path):
        """A second drain of the same turn is empty — including the unmatched
        pre-snapshots, which must not survive into a later turn."""
        v = FileMutationVerifier()
        self._write(v, tmp_path / "x.txt", "T", "c1")
        # An unmatched pre (tool never reported back) on the same turn.
        v.pre_tool_use(
            "file_write", {"file_path": str(tmp_path / "never.txt")}, "c2", turn_key="T",
        )
        assert v.post_turn(turn_key="T") is not None
        assert v.post_turn(turn_key="T") is None
        assert v.live_turns == 0

    def test_discard_turn_is_idempotent_and_silent(self, tmp_path: Path):
        """The cleanup path for turns that end early (iteration cap, circuit
        breaker, interrupt) — drops state without rendering a summary."""
        v = FileMutationVerifier()
        self._write(v, tmp_path / "x.txt", "T", "c1")
        v.discard_turn(turn_key="T")
        assert v.live_turns == 0
        assert v.post_turn(turn_key="T") is None
        v.discard_turn(turn_key="T")  # already gone — must not raise

    def test_keys_are_unique_per_turn_not_per_session(self):
        """A session can have more than one turn in flight, so the key cannot
        just be the session id."""
        v = FileMutationVerifier()
        assert v.new_turn_key("sess-1") != v.new_turn_key("sess-1")
        assert v.new_turn_key(None) != v.new_turn_key(None)

    def test_undrained_turns_are_bounded(self, tmp_path: Path):
        """Backstop for a caller that never drains: evicting the oldest turn
        loses that turn's summary, which is strictly better than growing
        without bound inside a daemon-lifetime singleton."""
        from prometheus.hooks.file_mutation_verifier import MAX_LIVE_TURNS

        v = FileMutationVerifier()
        for i in range(MAX_LIVE_TURNS + 5):
            self._write(v, tmp_path / f"f{i}.txt", f"T{i}", f"c{i}")
        assert v.live_turns == MAX_LIVE_TURNS
        assert v.post_turn(turn_key="T0") is None      # evicted
        assert v.post_turn(turn_key=f"T{MAX_LIVE_TURNS + 4}") is not None  # newest kept

    def test_callers_without_a_key_share_one_scope(self, tmp_path: Path):
        """Back-compat: an omitted key means DEFAULT_TURN_KEY. Correct only
        for single-threaded callers — which is why run_loop always passes
        one."""
        v = FileMutationVerifier()
        target = tmp_path / "x.txt"
        v.pre_tool_use("file_write", {"file_path": str(target)}, "c1")
        target.write_text("x", encoding="utf-8")
        v.post_tool_use(
            "file_write", {"file_path": str(target)}, "c1",
            output="ok", is_error=False,
        )
        assert v.post_turn() is not None
        assert v.post_turn() is None

    def test_no_record_is_allocated_for_tools_that_touch_nothing(self):
        """Otherwise every `ls` would consume a turn slot and push real turns
        out of the bounded map."""
        v = FileMutationVerifier()
        v.pre_tool_use("bash", {"command": "ls -la"}, "c1", turn_key="T")
        v.post_tool_use(
            "bash", {"command": "ls -la"}, "c1",
            output="a\nb\n", is_error=False, turn_key="T",
        )
        assert v.live_turns == 0
        assert v.post_turn(turn_key="T") is None


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

    def test_make_default_verifier_with_opt_out(self):
        v = make_default_verifier({
            "hooks": {
                "file_mutation_verifier": {
                    "enabled": False,
                    "truncate_after_n_mutations": 5,
                },
            },
        })
        assert v.enabled is False
        assert v._truncate_n == 5

    def test_stale_show_in_telegram_key_does_not_break_loading(self):
        """The knob was specified, implemented as an attribute, and never read
        by any code — so it was deleted rather than left as a setting that
        silently does nothing. A config that still carries it must keep
        loading, and must NOT resurrect the attribute."""
        v = make_default_verifier({
            "hooks": {"file_mutation_verifier": {"show_in_telegram": True}},
        })
        assert v.enabled is True
        assert not hasattr(v, "show_in_telegram")


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

        # ...and it is NOT dressed up as something the human typed. The role
        # stays "user" (that is its wire role to the model), but provenance is
        # what LCM persists and what the REST history replays, so this is the
        # field a UI filters on. is_trusted stays True — machinery-authored,
        # so no untrusted-input banner and byte-identical model-facing text.
        assert last.provenance == "file_mutation_verifier"
        assert last.is_trusted is True

        # The loop dropped the turn's state on the way out.
        assert v.live_turns == 0

    @pytest.mark.asyncio
    async def test_concurrent_run_loop_turns_get_their_own_summaries(
        self, tmp_path: Path,
    ):
        """End-to-end version of the turn-scoping bug: two turns driven
        CONCURRENTLY through one shared verifier — the arrangement the daemon
        now has on the web bridge, where a single LoopContext serves every
        Beacon session."""
        import asyncio

        from prometheus.engine.agent_loop import LoopContext, run_loop
        from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
        from prometheus.engine.usage import UsageSnapshot
        from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
        from typing import AsyncIterator

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
                # Yield control mid-tool so the two turns genuinely interleave.
                await asyncio.sleep(0)
                Path(parsed.file_path).write_text(parsed.content, encoding="utf-8")
                await asyncio.sleep(0)
                return ToolResult(output=f"wrote {len(parsed.content)} bytes")

        class _R:
            def __init__(self): self._t = _WriteTool()
            def get(self, n): return self._t if n == "file_write" else None
            def get_tool(self, n): return self.get(n)
            def list_tools(self): return [self._t]
            def list_schemas(self): return [{"name": "file_write", "input_schema": {}}]

        class _Prov(ModelProvider):
            """Writes ``path`` on the first call, then stops."""
            def __init__(self, path: Path):
                self._path = path
                self._call = 0
            async def stream_message(self, request) -> AsyncIterator:
                await asyncio.sleep(0)
                if self._call == 0:
                    msg = ConversationMessage(
                        role="assistant",
                        content=[ToolUseBlock(
                            id=f"c-{self._path.name}", name="file_write",
                            input={"file_path": str(self._path), "content": "hi\n"},
                        )],
                    )
                    stop = "tool_calls"
                else:
                    msg = ConversationMessage(
                        role="assistant", content=[TextBlock(text="done")],
                    )
                    stop = "stop"
                self._call += 1
                yield ApiMessageCompleteEvent(
                    message=msg,
                    usage=UsageSnapshot(input_tokens=1, output_tokens=1),
                    stop_reason=stop,
                )

        # ONE verifier, as run_daemon builds it.
        v = FileMutationVerifier()
        registry = _R()

        async def _turn(name: str) -> list:
            target = tmp_path / f"{name}.txt"
            ctx = LoopContext(
                provider=_Prov(target), model="qwen-test",
                system_prompt="sys", max_tokens=1024,
                tool_registry=registry,
                file_mutation_verifier=v,
            )
            msgs = [ConversationMessage.from_user_text(f"write {target}")]
            async for _ in run_loop(ctx, msgs, session_id=f"sess-{name}"):
                pass
            return msgs

        msgs_a, msgs_b = await asyncio.gather(_turn("alpha"), _turn("beta"))

        for name, msgs in (("alpha", msgs_a), ("beta", msgs_b)):
            summary = msgs[-1]
            assert summary.provenance == "file_mutation_verifier", (
                f"turn {name} never got a summary — its mutations were drained "
                f"by the other turn (the pre-fix bug)"
            )
            other = "beta" if name == "alpha" else "alpha"
            assert f"{name}.txt" in summary.text
            assert f"{other}.txt" not in summary.text, (
                f"turn {name} reported turn {other}'s write as its own"
            )

        assert v.live_turns == 0, "run_loop leaked turn state on the way out"
