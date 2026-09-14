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
    redirect_without_target,
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

    def test_bash_dev_null_redirect_is_not_a_mutation(self):
        """> /dev/null never changes on disk; tracking it would emit a
        guaranteed "CLAIMED but NO CHANGE ON DISK" false positive."""
        assert _extract_bash_paths("echo hello > /dev/null") == []
        assert _extract_bash_paths("cat foo.log >> /dev/null") == []

    def test_bash_real_redirect_next_to_dev_null_still_tracked(self):
        out = _extract_bash_paths(
            "echo a > /dev/null && echo b > /tmp/kept.txt"
        )
        paths = [p for p, _ in out]
        assert "/tmp/kept.txt" in paths
        assert "/dev/null" not in paths

    def test_bash_dev_shm_redirect_still_tracked(self):
        """/dev/shm is a real tmpfs — writes DO land there."""
        out = _extract_bash_paths("echo x > /dev/shm/note.txt")
        assert ("/dev/shm/note.txt", "redirect_write") in out

    def test_bash_fd_redirect_is_not_a_mutation(self):
        """>&1 / 2>&1 duplicate a file descriptor — '&1' is not a path.
        Tracking it emits a guaranteed "CLAIMED but FILE ABSENT" false
        positive (observed live, 2026-08-25)."""
        assert _extract_bash_paths("echo hello >&1") == []
        assert _extract_bash_paths("cmd 2>&1") == []
        # UPDATED by issue #275. This used to assert `== []` — the real write in the
        # same clause went untracked because the patterns anchored on the TRAILING
        # redirect, and the comment called it an accepted false negative. Live, it
        # was worse than a missed path: the only extracted target was `&1`, correctly
        # dropped here, so the turn produced NO verifier row at all while a file was
        # genuinely written. Silence that means "nothing happened" and silence that
        # means "nothing was looked at" cannot be the same signal.
        assert _extract_bash_paths("cmd > file.txt 2>&1") == [("file.txt", "redirect_write")]


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

    def test_dev_null_redirect_produces_no_summary(self):
        """> /dev/null must not emit a 'CLAIMED but NO CHANGE ON DISK'
        warning — /dev/null never changes on disk by definition."""
        v = FileMutationVerifier()
        v.pre_tool_use(
            "bash", {"command": "echo hi > /dev/null"}, "c1", turn_key="T",
        )
        v.post_tool_use(
            "bash", {"command": "echo hi > /dev/null"}, "c1",
            output="", is_error=False, turn_key="T",
        )
        assert v.live_turns == 0
        summary = v.post_turn(turn_key="T")
        assert summary is None


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


class TestSilenceIsNotAnAudit:
    """Issue #275: the hook must not report a clean turn when it was blind.

    `post_turn` returned None both when a turn touched no files and when its writes
    were never extracted. Live, `echo hello > /tmp/fdcheck.txt 2>&1` produced NO row
    while the file genuinely landed on disk — and the daemon had logged
    `FileMutationVerifier: enabled (turn-end audit)` at boot. An audit that cannot
    distinguish "nothing happened" from "nothing was looked at" is counted on for a
    guarantee it is not making.

    These drive the REAL hook — pre_tool_use then post_turn — not the predicate
    alone, because the defect was in what the hook DID with the predicate's answer.
    """

    def _verifier(self):
        from prometheus.hooks.file_mutation_verifier import FileMutationVerifier

        return FileMutationVerifier(enabled=True)

    def test_the_write_behind_a_trailing_2to1_is_now_tracked(self, tmp_path):
        """The original defect, end to end."""
        v = self._verifier()
        target = tmp_path / "fdcheck.txt"
        cmd = f"echo hello > {target} 2>&1"
        v.pre_tool_use("bash", {"command": cmd}, "t1", turn_key="k")
        target.write_text("hello\n")
        v.post_tool_use("bash", {"command": cmd}, "t1", turn_key="k")
        out = v.post_turn(turn_key="k")
        assert out is not None, "a turn that wrote a file produced no row at all"
        assert "fdcheck.txt" in out

    def test_an_unnameable_redirect_speaks_up_instead_of_going_quiet(self):
        """A quoted destination cannot be named — so say so, rather than return None
        and let it read as a clean audit."""
        v = self._verifier()
        cmd = 'cmd > "some file.txt"'
        v.pre_tool_use("bash", {"command": cmd}, "t1", turn_key="k")
        out = v.post_turn(turn_key="k")
        assert out is not None, "blind turn stayed silent — indistinguishable from clean"
        assert "no nameable target" in out
        assert "may have written files this hook could not see" in out

    def test_a_genuinely_clean_turn_stays_quiet(self):
        """Only speak up when it might be blind. A turn with nothing to audit must
        not start emitting a row in every conversation."""
        v = self._verifier()
        for cmd in ["ls -la", "cmd 2>&1", "echo x > /dev/null", 'grep -r "a>b" .']:
            v.pre_tool_use("bash", {"command": cmd}, "t1", turn_key="k")
        assert v.post_turn(turn_key="k") is None

    def test_a_partial_audit_says_it_is_partial(self, tmp_path):
        """Listing what WAS seen while staying quiet about what could not be is the
        same misleading silence, just harder to notice."""
        v = self._verifier()
        target = tmp_path / "seen.txt"
        seen = f"echo a > {target}"
        unseen = 'echo b > "un seen.txt"'
        v.pre_tool_use("bash", {"command": seen}, "t1", turn_key="k")
        target.write_text("a\n")
        v.post_tool_use("bash", {"command": seen}, "t1", turn_key="k")
        v.pre_tool_use("bash", {"command": unseen}, "t2", turn_key="k")
        out = v.post_turn(turn_key="k")
        assert "seen.txt" in out
        assert "no nameable target" in out, "the partial audit claimed to be complete"

    def test_a_quoted_arrow_is_not_read_as_a_redirect(self):
        """Widening the patterns must not reintroduce the false-positive class #198
        and #274 removed. `->` inside a string is prose, not a redirect."""
        from prometheus.hooks.file_mutation_verifier import _extract_bash_paths

        assert _extract_bash_paths('echo "a -> b" > out.txt') == [("out.txt", "redirect_write")]
        assert _extract_bash_paths('grep -r "a>b" .') == []


class TestShellSyntaxIsNotAFilePath:
    """A command that answers an adjacent question (RECURRING.md §4j).

    Four false-positive classes were observed on real turns, all of them the same
    shape: the instrument reported a plausible file that was never the subject of the
    command. Each had a different mechanism, and each is pinned here.

    The two controls are asserted IN ONE TEST per class. "Make the warnings go away"
    is trivially achievable by weakening extraction into uselessness, so a test that
    only checks the noise disappeared would pass on a verifier that tracks nothing.
    Every negative control below is paired with the positive case it must not break.
    """

    # The fixtures are real: each came from a command in this session's history.

    def test_quoted_prose_with_a_semicolon_is_not_split_into_commands(self):
        """`echo "… does not touch them."` reported a missing file named `them.`

        Splitting on `;` before dequoting cut a quoted string in half; each half then
        carried an unbalanced quote, so `_dequote` could not pair it, the prose
        survived as bare text, and a sentence-ending period became a filename.
        """
        cmd = (
            'echo "  refs are permanent unless deleted; reflog expiry does not '
            'touch them."'
        )
        # NEGATIVE control: no fragment of the sentence is a claimed path.
        assert _extract_bash_paths(cmd) == []
        # POSITIVE control: a real write in the same shape is still caught, so this
        # did not pass by stopping extraction of quoted-adjacent clauses altogether.
        assert _extract_bash_paths('echo "  a; b" > out.txt') == [
            ("out.txt", "redirect_write")
        ]

    def test_a_flag_left_by_a_blanked_quoted_target_is_not_a_file(self):
        """`rm -f "$P"` claimed a file named `-f`.

        Blanking the quoted target leaves nothing for `(\\S+)` to match, so
        `(?:-\\w+\\s+)*` backtracks to zero matches and the flag itself becomes the
        operand. Same mechanism produced `2>/dev/null` as a claimed `touch` target.
        """
        # NEGATIVE control.
        assert _extract_bash_paths('rm -f "$PROBE"') == []
        assert _extract_bash_paths('mkdir -p "$gd/hooks"') == []
        assert _extract_bash_paths(
            'if touch "$PROBE" 2>/dev/null; then rm -f "$PROBE"; fi'
        ) == []
        # POSITIVE control: the same commands with a literal target are tracked.
        assert _extract_bash_paths("rm -f node_modules") == [
            ("node_modules", "delete")
        ]
        assert _extract_bash_paths("mkdir -p a/b && touch a/b/x.md") == [
            ("a/b", "mkdir"),
            ("a/b/x.md", "touch"),
        ]

    def test_an_arrow_function_is_not_a_redirect(self):
        """`(s) => /^smoke:/.test(s)` claimed a file `/^smoke:/.test(s))`.

        The `>` in `=>` is JavaScript, not a redirect. `(?<![<>])` did not cover it —
        that lookbehind exists to keep the single-redirect pattern off the first char
        of `>>`, and says nothing about an `=` before the `>`. The operator position
        guard from `_REDIRECT_OP` is what distinguishes them.
        """
        # NEGATIVE control.
        assert _extract_bash_paths("const f = (s) => /^smoke:/.test(s)") == []
        assert _extract_bash_paths(
            "const smokes = Object.keys(pkg.scripts).filter((s) => /^smoke:/.test(s))"
        ) == []
        # POSITIVE control: real redirects in every legal operator position still
        # match. This is the assertion that stops the guard being tuned so tight that
        # it drops genuine writes.
        assert _extract_bash_paths("cmd > out.txt") == [
            ("out.txt", "redirect_write")
        ]
        assert _extract_bash_paths("cmd >> out.txt") == [
            ("out.txt", "redirect_append")
        ]
        assert _extract_bash_paths("echo hi 2> err.txt") == [
            ("err.txt", "redirect_write")
        ]
        assert _extract_bash_paths("cmd > out.txt 2>&1") == [
            ("out.txt", "redirect_write")
        ]

    def test_a_function_definition_is_not_a_touch(self):
        """`say() { … }` and `touch nothing` inside a quoted span, after a real write."""
        cmd = 'cat > /tmp/f.sh <<\'EOF\'\nsay() { echo hi; }\nEOF'
        assert _extract_bash_paths(cmd) == [("/tmp/f.sh", "redirect_write")]
        # The heredoc body contributes nothing of its own.
        assert _extract_bash_paths("say() { echo hi; }") == []


class TestDeletionSemantics:
    """Absence is the outcome a deletion asks for, not evidence it failed.

    The condition this satisfies is the inverted one: for a delete, absent-after is
    SUCCESS. Both directions are in one test, because fixing only the noisy direction
    is the cheap way to pass.
    """

    def test_a_delete_of_an_absent_file_is_a_noop_success_not_a_failure(
        self, tmp_path: Path
    ):
        gone = tmp_path / "never-existed.txt"
        v = FileMutationVerifier(enabled=True)
        seen = f"rm -f {gone}"
        v.pre_tool_use("bash", {"command": seen}, "t1", turn_key="k")
        v.post_tool_use("bash", {"command": seen}, "t1", turn_key="k")
        out = v.post_turn(turn_key="k")
        assert "CLAIMED but FILE ABSENT" not in out, (
            "a successful no-op delete was reported as a failed claim"
        )
        assert "✓" in out and "already absent" in out

    def test_a_delete_that_actually_removed_something_is_still_recorded(
        self, tmp_path: Path
    ):
        target = tmp_path / "real.txt"
        target.write_text("content\n")
        v = FileMutationVerifier(enabled=True)
        cmd = f"rm -f {target}"
        v.pre_tool_use("bash", {"command": cmd}, "t1", turn_key="k")
        target.unlink()
        v.post_tool_use("bash", {"command": cmd}, "t1", turn_key="k")
        out = v.post_turn(turn_key="k")
        assert "deleted" in out and "✓" in out
        assert "already absent" not in out, (
            "a real deletion was reported as a no-op — the fix over-corrected"
        )

    def test_a_write_that_never_landed_still_warns(self, tmp_path: Path):
        """The rule this file exists for must survive the deletion fix.

        A claimed WRITE with nothing on disk is still `CLAIMED but FILE ABSENT`.
        Scoping the no-op tag to `delete` is what keeps this true.
        """
        v = FileMutationVerifier(enabled=True)
        cmd = f"echo x > {tmp_path}/set-ok.txt"
        v.pre_tool_use("bash", {"command": cmd}, "t1", turn_key="k")
        # Deliberately do NOT create the file — the write never happened.
        v.post_tool_use("bash", {"command": cmd}, "t1", turn_key="k")
        out = v.post_turn(turn_key="k")
        assert "CLAIMED but FILE ABSENT" in out, (
            "the deletion fix silenced the write case too — the instrument is gone"
        )

    def test_a_failed_rm_is_not_laundered_into_a_success(self, tmp_path: Path):
        """An `rm` that exited non-zero still reports ✗, not the no-op tag."""
        gone = tmp_path / "absent.txt"
        v = FileMutationVerifier(enabled=True)
        cmd = f"rm {gone}"  # no -f: exits 1 when the file is absent
        v.pre_tool_use("bash", {"command": cmd}, "t1", turn_key="k")
        v.post_tool_use(
            "bash", {"command": cmd}, "t1", turn_key="k",
            output="rm: cannot remove: No such file or directory", is_error=True,
        )
        out = v.post_turn(turn_key="k")
        assert "✗" in out
        assert "already absent" not in out


class TestPerPathActionIsNotAJoinedVerb:
    """One bash call can claim a delete AND a write; they cannot share one label."""

    def test_mixed_actions_get_their_own_claim(self, tmp_path: Path):
        stale = tmp_path / "stale.txt"   # absent — delete becomes a no-op success
        fresh = tmp_path / "fresh.txt"   # absent — write becomes FILE ABSENT
        v = FileMutationVerifier(enabled=True)
        cmd = f"rm -f {stale} && echo x > {fresh}"
        v.pre_tool_use("bash", {"command": cmd}, "t1", turn_key="k")
        v.post_tool_use("bash", {"command": cmd}, "t1", turn_key="k")
        out = v.post_turn(turn_key="k")
        # The delete is a no-op success...
        assert "already absent" in out
        # ...and the write in the SAME command still warns. If both paths had been
        # stamped with the joined "delete/redirect_write" label, the delete branch
        # would have matched for `fresh.txt` too and the write warning would have
        # been swallowed — which is the failure this test exists to prevent.
        assert "CLAIMED but FILE ABSENT" in out


class TestLastClaimWinsForAPath:
    """The mirror of `test_a_failed_rm_is_not_laundered_into_a_success`.

    That test pins the case where a delete is judged correctly. This one pins the
    case where a path is claimed TWICE in one command and the wrong claim is the
    one kept — a delete followed by a write to the same path, where the write fails
    and the file ends up absent.

    The last operation against a path is what the final on-disk state should be
    judged against. Keeping the first (`dict.setdefault`) turned a write that never
    landed into a reported successful no-op deletion, because the deletion branch
    and the write branch disagree about the very same evidence:

        delete         + absent->absent  -> "deleted (no-op: already absent)"  ✓
        redirect_write + absent->absent  -> "CLAIMED but FILE ABSENT"          ⚠
    """

    def test_a_failed_write_is_not_laundered_by_a_preceding_delete_of_the_same_path(
        self, tmp_path: Path
    ):
        victim = tmp_path / "x"          # never created: the write does not land
        v = FileMutationVerifier(enabled=True)
        cmd = f"rm -f {victim} && echo y > {victim}"
        v.pre_tool_use("bash", {"command": cmd}, "t1", turn_key="k")
        v.post_tool_use("bash", {"command": cmd}, "t1", turn_key="k")
        out = v.post_turn(turn_key="k")
        assert "CLAIMED but FILE ABSENT" in out, (
            "a write that never landed was reported as a clean no-op delete — "
            "the first claim for this path was kept instead of the last"
        )
        assert "already absent" not in out

    def test_claim_map_keeps_the_last_action_per_path(self):
        """Direct assertion on the mapping, so the rule is pinned by name."""
        v = FileMutationVerifier(enabled=True)
        cmd = "rm -f x && echo y > x"
        assert v._claim_map("bash", {"command": cmd}) == {"x": "redirect_write"}

    def test_a_delete_that_is_the_last_claim_still_reads_as_a_noop(self, tmp_path: Path):
        """Reversed order: the delete is now the last claim, so the ✓ is correct.

        This is the control on the control — pinning last-wins must not become
        "always prefer the write", which would break the original fix.
        """
        gone = tmp_path / "y"            # absent, and stays absent
        v = FileMutationVerifier(enabled=True)
        cmd = f"touch {gone} && rm -f {gone}"
        v.pre_tool_use("bash", {"command": cmd}, "t1", turn_key="k")
        gone.touch()                     # touch lands...
        gone.unlink()                    # ...and the delete removes it
        v.post_tool_use("bash", {"command": cmd}, "t1", turn_key="k")
        out = v.post_turn(turn_key="k")
        assert "deleted" in out and "✓" in out
        assert "CLAIMED but FILE ABSENT" not in out


class TestNoSpaceRedirectIsADeliberateFalseNegative:
    """PINNED ON PURPOSE. Do not "fix" this without reading the trade below.

    `echo hi>out.txt` and `cmd>>app.log` are NOT tracked. That is a real false
    negative, accepted deliberately, and this test exists so that it stays a
    decision with a receipt rather than becoming an accident nobody knows about.

    WHY IT IS GIVEN UP. The `>` of a redirect and the `>` of a JS/TS arrow
    function are the same character. The only thing separating them in shell text
    is what comes immediately before: whitespace, clause start, an fd digit, or
    `&`. `echo hi>out.txt` has an `i` before the `>`, and so does `(s) =>`:

        echo hi>out.txt      <- `i` before `>`   a real redirect
        const f = (s) => …   <- `=` before `>`   JavaScript

    Any lookbehind permissive enough to admit the first admits forms of the second.
    Loosening it to `(?<=[A-Za-z0-9_])` would track `hi>out.txt` and would equally
    track `a>b` inside a heredoc body, which dequoting cannot blank.

    WHAT IT COSTS, MEASURED RATHER THAN ASSUMED. The untracked write is also NOT
    reported blind: `_REDIRECT_OP` carries the same lookbehind, so it does not even
    see an operator in `hi>out.txt`, `redirect_without_target` returns False, and
    the turn is recorded as clean. The gap is silent, not flagged. That is a
    deliberate acceptance for a reporter — and it is the single strongest argument
    for the lexer rewrite, where clause structure is known instead of guessed.

    If the no-space form must be tracked, the fix is a real lexer (shlex), not a
    looser lookbehind. See the rewrite issue.
    """

    def test_no_space_redirect_is_not_tracked(self):
        from prometheus.hooks.file_mutation_verifier import redirect_without_target

        # The false negative, pinned.
        assert _extract_bash_paths("echo hi>out.txt") == []
        assert _extract_bash_paths("cmd>>app.log") == []

        # AND the honest part: it is silent, not flagged. This assertion is here so
        # that if a future change starts reporting the no-space form as blind, the
        # change is deliberate and this test gets updated on purpose rather than
        # the docstring quietly going stale.
        assert redirect_without_target("echo hi>out.txt") is False

    def test_every_spaced_and_fd_prefixed_form_is_still_tracked(self):
        """The positive control. Tightening the lookbehind must not have cost
        any legal form — this is what stops the trade being paid for twice."""
        assert _extract_bash_paths("echo hi > out.txt") == [
            ("out.txt", "redirect_write")
        ]
        assert _extract_bash_paths("cmd >> app.log") == [
            ("app.log", "redirect_append")
        ]
        assert _extract_bash_paths("echo hi 2> err.log") == [
            ("err.log", "redirect_write")
        ]
        assert _extract_bash_paths("echo hi 2>> err.log") == [
            ("err.log", "redirect_append")
        ]
        assert _extract_bash_paths("cmd > out.txt 2>&1") == [
            ("out.txt", "redirect_write")
        ]

    def test_the_arrow_function_stays_unmatched(self):
        """What the trade buys. Loosening the lookbehind reintroduces this."""
        assert _extract_bash_paths("const f = (s) => /^smoke:/.test(s)") == []
        assert _extract_bash_paths("if a <= b: pass") == []


class TestUnresolvedTargetIsBlindnessNotAMissingFile:
    """An unexpanded variable in a redirect is a write whose name was LOST.

    `> "$LOG"` is blanked by dequoting and vanishes; `> /tmp/x.log` tracks
    normally; `> $LOG` — unquoted — survived into the patterns and was captured by
    `\\S+` as a literal filename. It can never be stat'd as existing, so every turn
    containing one emitted a permanent `⚠ CLAIMED but FILE ABSENT` about a file
    that was never named.

    The fix routes it to the #275 blindness contract rather than blocklisting `$`:
    one honest row saying *a write happened here and I could not name the
    destination*, in place of a false row saying *this file is missing*. Adding `$`
    to `_NOT_A_PATH` would have dropped the target SILENTLY — reproducing exactly
    the failure the no-space redirect has, and making it the fifth extension of a
    blocklist with a documented floor.

    Measured on 7,200 real commands from telemetry `tool_calls` before shipping:
    13 false FILE ABSENT rows removed, 2 blindness rows added, 0.028% of commands
    affected. A control that fires constantly is the failure this file is about;
    these numbers are why this ships rather than becoming a finding.
    """

    def test_an_unresolved_redirect_target_is_blind_and_untracked(self, tmp_path: Path):
        """BOTH directions in one test, so neither can be bought with the other."""
        v = FileMutationVerifier(enabled=True)

        # (a) NEGATIVE: the unresolved target is not tracked as a phantom file...
        assert _extract_bash_paths("echo x > $LOG") == []
        assert _extract_bash_paths("echo x > $UNKNOWN") == []
        assert _extract_bash_paths("echo x > ${LOG_DIR}/out.txt") == []
        # ...and it IS reported, not silent. This is the whole point of routing it
        # to the blindness contract instead of blocklisting the character.
        assert redirect_without_target("echo x > $LOG") is True
        assert redirect_without_target("echo x > ${LOG_DIR}/out.txt") is True

        # (b) POSITIVE: a literal destination still tracks AND stays non-blind, so
        # this did not pass by making every redirect blind.
        real = tmp_path / "real.log"
        assert _extract_bash_paths(f"echo x > {real}") == [
            (str(real), "redirect_write")
        ]
        assert redirect_without_target(f"echo x > {real}") is False

    def test_an_unresolved_target_beats_an_fd_sibling(self):
        """`> $LOG 2>&1` must be BLIND, not excused by its fd duplicate.

        The order of the two rules inside `redirect_without_target` matters: the
        sink/fd rule says "nothing to audit here", which would otherwise swallow the
        unresolved-target signal and make the turn look clean.
        """
        assert redirect_without_target("cmd > $LOG 2>&1") is True

    def test_a_pure_fd_duplicate_is_still_not_blind(self):
        """The #275 distinction must survive: `2>&1` alone is genuinely nothing."""
        assert redirect_without_target("echo hi 2>&1") is False
        assert redirect_without_target("cmd > out.txt 2>&1") is False

    def test_a_device_sink_is_still_not_blind(self):
        """And #198's case: `/dev/null` is not a lost name, it is no file at all."""
        assert redirect_without_target("cmd > /dev/null") is False
        assert _extract_bash_paths("cmd > /dev/null") == []

    def test_no_phantom_absent_row_reaches_the_summary(self):
        """End to end: the false `CLAIMED but FILE ABSENT` row is gone."""
        v = FileMutationVerifier(enabled=True)
        cmd = "echo x > $LOG"
        v.pre_tool_use("bash", {"command": cmd}, "t1", turn_key="k")
        v.post_tool_use("bash", {"command": cmd}, "t1", turn_key="k")
        out = v.post_turn(turn_key="k")
        assert out is not None, "the turn went silent — blindness must speak up"
        assert "no nameable target" in out
        assert "$LOG" not in out.split("—")[0], "a phantom path was tracked"
        assert "CLAIMED but FILE ABSENT" not in out


class TestUnresolvedOperandOutsideARedirectIsTheOpenGap:
    """PINNED AS A KNOWN GAP, not as correct behaviour. See #484.

    The blindness contract only fires on clauses carrying a REDIRECT operator. A
    mutation operator with an unresolved operand — `cp a /tmp/bak-$(date +%s)`,
    `mkdir -p $HOME/x` — now drops the operand and produces NO row at all: neither
    tracked nor blind.

    **What changed here, stated honestly, because it is not purely an improvement.**
    On `origin/main` these were not silent either — they tracked a PHANTOM path
    (`/tmp/bak-$(date`, `$HOME/x`) that could never be stat'd as existing, so they
    emitted a false `⚠ CLAIMED but FILE ABSENT`. This change removes the false row
    without adding a blindness row, so the command goes from *wrongly reported* to
    *unreported*. Trading a lie for silence is not automatically a win; it is the
    lesser evil here because a false row trains the reader to skim every row, and
    the 7 affected commands in 7,200 are backup/mkdir operations whose real
    destinations the instrument never had.

    Routing them to a blindness row means extending `redirect_without_target` beyond
    redirects, which is a change of contract rather than a bug fix, and belongs in the
    lexer rewrite where clause structure is known instead of guessed. This test exists
    so the gap is a decision with a receipt: if it ever starts reporting these, the
    test moves on purpose.
    """

    def test_unresolved_operand_in_a_non_redirect_command_is_silent(self):
        assert _extract_bash_paths("cp a.txt /tmp/bak-$(date +%s).txt") == []
        assert redirect_without_target("cp a.txt /tmp/bak-$(date +%s).txt") is False
        assert _extract_bash_paths("mkdir -p $HOME/x") == []
        assert redirect_without_target("mkdir -p $HOME/x") is False

    def test_a_literal_operand_in_the_same_command_still_tracks(self):
        """The control: the gap is about the UNRESOLVED operand only."""
        assert _extract_bash_paths("cp src.txt dst.txt") == [("dst.txt", "copy")]
        assert _extract_bash_paths("mkdir -p /tmp/x && touch /tmp/x/y") == [
            ("/tmp/x", "mkdir"),
            ("/tmp/x/y", "touch"),
        ]
