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
    audit_bash_command,
    FileMutationVerifier,
    _extract_bash_paths,
    make_default_verifier,
    command_has_unnameable_target,
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

    def test_an_fd_duplicate_piped_without_spaces_is_not_a_path(self):
        """`2>&1|head` — the fd duplicate immediately followed by a pipe.

        Observed live in a turn's verifier summary, which reported SIX rows of
        `⚠ &1|head — redirect_write: CLAIMED but FILE ABSENT` and one of `&1|tail`.
        The pre-#483 code captured `&1|head` as a redirect target because
        `_is_device_sink` uses `re.fullmatch(r"&\\d+", ...)`, and the `|head` suffix
        makes that fail — so the fd duplicate stopped looking like a sink and became
        a filename. It is a *fifth* shape, not one of the four #483 fixed: #274 pinned
        `>&1` and `2>&1`, both of which end at the digit.

        Correct on merged main because `_NOT_A_PATH` rejects a leading `&`, which is
        an independent mechanism from the sink filter — so this test exists to pin
        the behaviour rather than to change it. It was unpinned: the suite covered the
        spaced forms (`cmd 2>&1`, `cmd > file.txt 2>&1`) and not this one, and a
        correct-but-untested behaviour is one regression away from being wrong again.

        Both directions, so tightening the fd handling cannot silently trade this
        away: the spaced form, the no-space form, and a real write in the same clause.
        """
        # The shape that fired in the live summary.
        assert _extract_bash_paths("cmd 2>&1|head -3") == []
        assert _extract_bash_paths("cmd 2>&1|tail -2") == []
        assert _extract_bash_paths("cmd 2>&1|head") == []
        # No blindness row either: an fd duplicate is genuinely nothing to audit,
        # which is the #275 distinction. A lost NAME is blindness; this is not one.
        assert command_has_unnameable_target("cmd 2>&1|head -3") is False
        # POSITIVE control: a real write sharing the clause is still tracked, so this
        # did not pass by making the fd handling swallow its neighbours.
        assert _extract_bash_paths("cmd > out.txt 2>&1|head -3") == [
            ("out.txt", "redirect_write")
        ]


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
        # NOTE: this used to be `cmd > "some file.txt"`. Under the lexer a quoted
        # destination is NAMEABLE — that was the floor #484 set out to raise — so it
        # is now tracked rather than flagged, and no longer exercises this contract.
        # An unresolved variable is what genuinely cannot be named.
        cmd = "cmd > $LOG"
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
        unseen = "echo b > $UNSEEN"      # quoted paths are nameable now; a variable is not
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
        carried an unbalanced quote, nothing could pair it, the prose survived as bare
        text, and a sentence-ending period became a filename. A separator that is a
        TOKEN cannot fall inside a quoted span, so the cascade is now unreachable
        rather than merely guarded against.
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

        The `>` in `=>` is JavaScript, not a redirect. Under the regex this needed a
        lookbehind on the preceding character, which could never admit `hi>out.txt`
        and reject `(s) =>` at the same time. The lexer removes the dilemma: `=>` is
        neutralised before tokenizing, so no `>` reaches the walker that was not an
        operator, and both forms are handled correctly rather than traded off.
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


class TestNoSpaceRedirectIsTrackedByTheLexer:
    """WAS a deliberate false negative. Closed by the lexer rewrite (#484).

    THE TRADE THAT USED TO BE HERE. The `>` of a redirect and the `>` of a JS/TS
    arrow function are the same character. Under a regex the only thing separating
    them is what comes immediately before — whitespace, clause start, an fd digit,
    or `&` — and `echo hi>out.txt` has an `i` before the `>`, exactly as `(s) =>`
    has an `=`:

        echo hi>out.txt      <- `i` before `>`   a real redirect
        const f = (s) => …   <- `=` before `>`   JavaScript

    Any lookbehind permissive enough to admit the first admitted forms of the
    second, so the no-space redirect was given up to keep `=>` from claiming a
    file. Worse, the gap was SILENT rather than flagged: the blindness predicate
    carried the same lookbehind, so it did not see an operator there either and the
    turn was recorded clean.

    WHY THE LEXER IS NOT JUST A LOOSER LOOKBEHIND. Both horns came from deciding
    operator-ness by the preceding CHARACTER. A lexer decides it by TOKEN: `>` is
    emitted as an operator wherever it legally is one, and `=>` never reaches the
    walker at all because it is neutralised in the quote-aware pre-pass. So the
    break closes and the false positive stays closed — the tests below assert both
    halves, because a fix that only bought one of them would be a swap.
    """

    def test_no_space_redirect_is_tracked(self):
        """The break, closed. It is TRACKED — not merely reported blind.

        Naming the file is the point. A rewrite that satisfied the invariant by
        calling this blind would have honoured the letter of the contract and given
        up the thing the contract exists to produce.
        """
        assert _extract_bash_paths("echo hi>out.txt") == [("out.txt", "redirect_write")]
        assert _extract_bash_paths("cmd>>app.log") == [("app.log", "redirect_append")]
        # The spacing dimension in full: the operator is a token now, so where the
        # whitespace falls stopped being a thing the extractor can be wrong about.
        assert _extract_bash_paths("echo hi> out.txt") == [("out.txt", "redirect_write")]
        assert _extract_bash_paths("echo hi >out.txt") == [("out.txt", "redirect_write")]

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
        assert command_has_unnameable_target("echo x > $LOG") is True
        assert command_has_unnameable_target("echo x > ${LOG_DIR}/out.txt") is True

        # (b) POSITIVE: a literal destination still tracks AND stays non-blind, so
        # this did not pass by making every redirect blind.
        real = tmp_path / "real.log"
        assert _extract_bash_paths(f"echo x > {real}") == [
            (str(real), "redirect_write")
        ]
        assert command_has_unnameable_target(f"echo x > {real}") is False

    def test_an_unresolved_target_beats_an_fd_sibling(self):
        """`> $LOG 2>&1` must be BLIND, not excused by its fd duplicate.

        The order of the two rules inside `command_has_unnameable_target` matters: the
        sink/fd rule says "nothing to audit here", which would otherwise swallow the
        unresolved-target signal and make the turn look clean.
        """
        assert command_has_unnameable_target("cmd > $LOG 2>&1") is True

    def test_a_pure_fd_duplicate_is_still_not_blind(self):
        """The #275 distinction must survive: `2>&1` alone is genuinely nothing."""
        assert command_has_unnameable_target("echo hi 2>&1") is False
        assert command_has_unnameable_target("cmd > out.txt 2>&1") is False

    def test_a_device_sink_is_still_not_blind(self):
        """And #198's case: `/dev/null` is not a lost name, it is no file at all."""
        assert command_has_unnameable_target("cmd > /dev/null") is False
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


class TestUnresolvedOperandOutsideARedirectIsReported:
    """A mutation operator with an unresolved operand is blindness, not silence.

    THE GAP THIS CLASS USED TO PIN IS CLOSED, and the test moved on purpose — which
    is what its own docstring said would happen.

    #485 stopped tracking the phantom path these produced (`/tmp/bak-$(date`,
    `$HOME/x`), which removed a false `⚠ CLAIMED but FILE ABSENT` row but left the
    command emitting NO row at all, because `command_has_unnameable_target` only
    inspected clauses carrying a REDIRECT operator. Trading a lie for silence is not
    automatically a win, and it was pinned as a known gap rather than shipped quietly.

    Closing it is not a new contract. The extractor already KNEW it had dropped an
    operand — `_is_trackable` returned False and the caller discarded it — so
    reporting it is the drop site admitting a decision the code was already making
    and throwing away. Under the lexer the drop site is `_analyze_clause`, where the
    same operand either becomes a tracked path or sets `blind`, which is the
    invariant written as one branch rather than inferred across two functions.

    Measured before shipping, on 7,232 real commands from telemetry `tool_calls`:
    2 blindness rows before, 9 after. **Seven new rows, 0.124% of the corpus** — not
    the hundreds a naive widening of `mkdir -p $SOMETHING` might have cost. That
    measurement was the gate on whether this ships as a fix or gets handed to #484 as
    a finding; it came out a fix.
    """

    def test_an_unresolved_operand_in_a_mutation_command_is_reported_blind(self):
        """BOTH directions: reported blind, and still not tracked as a phantom."""
        for cmd in (
            "cp a.txt /tmp/bak-$(date +%s).txt",
            "mkdir -p $HOME/x",
            "rm -f inspect-$r",
            "touch /tmp/smoke-$s.log",
        ):
            assert _extract_bash_paths(cmd) == [], (
                f"{cmd!r} tracked a phantom path that can never be stat'd"
            )
            assert command_has_unnameable_target(cmd) is True, (
                f"{cmd!r} dropped an operand and reported nothing"
            )

    def test_a_literal_operand_in_the_same_command_still_tracks(self):
        """The control: only the UNRESOLVED operand goes blind."""
        assert _extract_bash_paths("cp src.txt dst.txt") == [("dst.txt", "copy")]
        assert command_has_unnameable_target("cp src.txt dst.txt") is False
        assert _extract_bash_paths("mkdir -p /tmp/x && touch /tmp/x/y") == [
            ("/tmp/x", "mkdir"),
            ("/tmp/x/y", "touch"),
        ]
        assert command_has_unnameable_target("mkdir -p /tmp/x && touch /tmp/x/y") is False

    def test_a_command_that_touches_nothing_is_not_blind(self):
        """The over-widening control. Generalising must not make every command blind,
        or the blindness row stops meaning anything — the failure this whole file is
        about, reintroduced from the other direction."""
        for cmd in (
            "ls -la /tmp",
            "echo hello",
            "git status --short | head -5",
            "grep -rn 'pattern' src/",
            "cat README.md",
        ):
            assert command_has_unnameable_target(cmd) is False, (
                f"{cmd!r} was reported blind but touches nothing unnameable"
            )

    def test_the_no_space_redirect_gap_is_closed(self):
        """The gap this class used to hold open, now closed. See #484.

        It closed WITHOUT reopening the `=>` false positive it was traded against,
        which was the open question: an operator is a token now, not a character
        identified by what precedes it, so `>` and `=>` are told apart by the lexer
        rather than by a lookbehind that could only ever admit one of them. The
        second pair of assertions is that half of the trade, and it is the reason
        this is a fix rather than a swap.
        """
        assert _extract_bash_paths("echo hi>out.txt") == [("out.txt", "redirect_write")]
        assert command_has_unnameable_target("echo hi>out.txt") is False
        assert _extract_bash_paths("cmd>>app.log") == [("app.log", "redirect_append")]
        assert command_has_unnameable_target("cmd>>app.log") is False
        # The false positive that the gap was the price of — still rejected.
        assert _extract_bash_paths("const f = (s) => /^smoke:/.test(s)") == []
        assert command_has_unnameable_target("const f = (s) => /^smoke:/.test(s)") is False
        assert _extract_bash_paths("if a <= b: pass") == []

    def test_blindness_reaches_the_summary(self, tmp_path: Path):
        """End to end: a cp to an unresolved destination produces a row."""
        v = FileMutationVerifier(enabled=True)
        cmd = "cp a.txt /tmp/bak-$(date +%s).txt"
        v.pre_tool_use("bash", {"command": cmd}, "t1", turn_key="k")
        v.post_tool_use("bash", {"command": cmd}, "t1", turn_key="k")
        out = v.post_turn(turn_key="k")
        assert out is not None, "the turn went silent — blindness must speak up"
        assert "no nameable target" in out


class TestVerbInArgumentPositionIsNotACommand:
    """WAS a known false positive (#486). Closed by the command word (#484).

    `echo touch $FOO` mutates nothing, yet this reports it blind: the pattern table
    matches the mutation verb wherever it appears in the clause, so a verb sitting in
    ARGUMENT position looks like a command, and the unresolved token after it looks
    like a lost destination.

        echo touch $FOO      -> blind True    WRONG: nothing is mutated
        grep touch $FILE     -> blind True    WRONG: nothing is mutated
        echo "touch $FOO"    -> blind False   correct: quoting blanks it first

    Both are silent on `main`, so this is a false positive **introduced by the
    generalization in this PR** rather than inherited — the old contract only fired on
    redirect clauses, and `echo touch $FOO` has no redirect.

    WHY IT IS NOT FIXED HERE. The structural fix is to anchor mutation patterns to the
    clause's command word, since a verb mid-clause is an argument. That needs an
    allowance for legal prefixes — `sudo`, `env VAR=x`, `time`, `do`, `then` — and the
    moment that list is being written it is a blocklist of prefixes, which is the fifth
    extension of the same shape this file has already lost to four times. **A token
    stream says which token is the command word; nothing has to guess.** That is #484,
    and this test is here so the shape is recorded as known rather than merged silently.

    MEASURED FREQUENCY, because "is it worth closing" is a question about rate, not
    principle. Across 7,252 real commands from telemetry `tool_calls`, the 12 rows this
    generalization produces classify as **11 real, 0 false** — the verb-in-argument
    shape does not occur in that corpus at all. The one extra row was the verifier's own
    commit-message heredoc, which telemetry recorded live from this very session (see
    `TestCorpusSelfContamination`). So this is a theoretical gap with a measured
    frequency of zero on real commands, and a reproducible one on synthetic shapes.

    Do not "fix" this by loosening or tightening the pattern table. If it starts being
    reported correctly, this test moves on purpose.
    """

    def test_a_verb_in_argument_position_is_not_a_mutation(self):
        """The defect, closed. A verb mid-clause is an argument, and the token
        stream says which token the shell would actually execute."""
        assert command_has_unnameable_target("echo touch $FOO") is False
        assert command_has_unnameable_target("grep touch $FILE") is False
        assert command_has_unnameable_target("echo cp $A $B") is False
        assert command_has_unnameable_target("cat rm $X") is False

        # Correct today, and must stay correct: quoting blanks the span first, so the
        # verb never reaches the patterns. This is the control on the control — if a
        # future fix makes the four above False by blanking more aggressively, these
        # must not silently break with them.
        assert command_has_unnameable_target('echo "touch $FOO"') is False
        assert command_has_unnameable_target('echo "rm -rf $HOME"') is False
        assert command_has_unnameable_target("test -f $CONFIG && echo yes") is False

    def test_a_real_verb_in_command_position_is_unaffected_by_that_distinction(self):
        """The shapes the generalization exists FOR must stay blind."""
        for cmd in (
            "touch $FOO",
            "mkdir -p $BAR",
            "do mkdir -p inspect-$r",   # shell keyword prefix
            "sudo rm -f $X",            # privilege prefix
            "env FOO=1 touch $X",       # env prefix
        ):
            assert command_has_unnameable_target(cmd) is True, cmd


class TestCorpusSelfContamination:
    """The corpus that justified this change measures itself.

    `tool_calls` records every bash command the daemon runs, **live**. The commands
    that probe this verifier are themselves bash commands, so a measurement taken over
    that table includes the probes that produced it — and a heredoc containing
    `cp a /tmp/bak-$(date +%s).txt` as *prose* is recorded as a command and classified
    as a mutation.

    That is exactly what happened: a twelfth blindness row appeared during review whose
    content was this PR's own commit message. It is recorded here because it is the
    measurement hazard, not because it is fixable in this file — any future number
    quoted from `tool_calls` needs the same check, or it silently counts its own
    instruments. A corpus that grows while you measure it is not a fixed sample.
    """

    def test_heredoc_prose_containing_a_mutation_shape_is_not_a_command(self):
        """The contamination case, closed at the parser rather than at the corpus.

        A heredoc body is DATA — the shell never executes it — so prose inside one is
        not a mutation and must not be reported as anything. The measurement hazard in
        the docstring above is unchanged: a count taken from a live `tool_calls` still
        counts its own instruments. What changed is that this file no longer
        MISREADS the prose once it is counted.
        """
        cmd = "git commit -q -F - <<MSG\nfix: cp a /tmp/bak-$(date +%s).txt\nMSG"
        assert command_has_unnameable_target(cmd) is False
        assert _extract_bash_paths(cmd) == []

    def test_a_heredoc_body_cannot_contribute_a_claimed_path(self):
        """The worse half of the same hazard, and it was live on `main`.

        Prose in a heredoc did not merely raise a spurious blindness row — a redirect
        written inside a body was extracted as a REAL claimed path, producing a
        permanent `⚠ CLAIMED but FILE ABSENT` about a file nobody ever named:

            cat <<EOF          on main -> [('/tmp/EVIL', 'redirect_write')]
            echo hi > /tmp/EVIL
            EOF
        """
        assert _extract_bash_paths("cat <<EOF\necho hi > /tmp/EVIL\nEOF") == []
        assert _extract_bash_paths(
            "git commit -F - <<MSG\nfix: rewrote > docs/out.md\nMSG"
        ) == []
        # The command line around the heredoc is still read normally.
        assert _extract_bash_paths("cat > /tmp/real.sh <<EOF\nanything > /tmp/EVIL\nEOF") == [
            ("/tmp/real.sh", "redirect_write")
        ]


# ---------------------------------------------------------------------------
# The invariant, as a property (issue #484 acceptance criterion 1)
# ---------------------------------------------------------------------------


def _generated_clause_shapes() -> list[tuple[str, str]]:
    """Every combination of the dimensions the extractor has historically lost to.

    Returns ``(label, command)``. The dimensions are the ones named in #484 —
    mutation operator, redirect operator, quoted/unquoted, resolved/unresolved
    operand, spaced/unspaced, fd-prefixed/bare — crossed rather than enumerated,
    because a hand-written list of cases can only ever cover the shapes someone has
    already met. Four patches to this file were each correct and each left the next
    shape open; that is what this function exists to stop.
    """
    # (operand text, does it name a real path?)
    operands = [
        ("out.txt", True),
        ("a/b/out.txt", True),
        ('"my out.txt"', True),          # quoted: invisible to the old blanking
        ("'my out.txt'", True),
        ("$LOG", False),                 # unresolved: a name that was lost
        ("${LOG_DIR}/out.txt", False),
        ("$(date +%s).txt", False),      # substitution split across tokens
        ("$1", False),                   # positional
    ]
    redirect_ops = [">", ">>"]
    fd_prefixes = ["", "2"]
    spacings = ["", " "]
    verbs = ["touch", "rm -f", "mkdir -p", "cp src.txt", "mv src.txt"]
    prefixes = ["", "sudo ", "env FOO=1 ", "do "]

    shapes: list[tuple[str, str]] = []
    for operand, _ in operands:
        for op in redirect_ops:
            for fd in fd_prefixes:
                for lead in spacings:
                    for trail in spacings:
                        shapes.append((
                            f"redirect fd={fd!r} op={op} lead={lead!r} trail={trail!r} operand={operand}",
                            f"echo hi{lead}{fd}{op}{trail}{operand}",
                        ))
        for verb in verbs:
            for prefix in prefixes:
                shapes.append((
                    f"mutation prefix={prefix!r} verb={verb} operand={operand}",
                    f"{prefix}{verb} {operand}",
                ))
    return shapes


class TestTheInvariantHoldsOverGeneratedShapes:
    """NO WRITE FORM MAY BE BOTH UNTRACKED AND UNREPORTED.

    This is #484's acceptance criterion 1, and it is a property rather than a corpus
    because a corpus is what this file has lost to four times. #198 added `/dev/*`,
    #274 added fd duplicates, #483 added flags and operator position, #485 added
    unresolved targets, #486 added mutation operands. Each was correct. Each left the
    next shape open, because a filter extended against the shapes already reported
    will keep missing the one that has not been reported yet.

    The corpus tests elsewhere in this file are the regression suite. THIS is what
    tells you the corpus is incomplete: the day a new combination of dimensions
    becomes reachable, it fails here, on the commit that introduced it, instead of
    being found three months later by someone reading a turn's output.

    THE ONE EXEMPTION, stated rather than hidden: a device sink (`/dev/null`, an fd
    duplicate) is neither tracked nor blind, because nothing reaches disk. That is
    the absence of a write, not silence about one, so it is not a break — and it is
    deliberately not generated here, so the disjunction below stays total.
    """

    def test_every_generated_shape_is_tracked_or_blind(self):
        shapes = _generated_clause_shapes()
        assert len(shapes) > 200, (
            f"the generator collapsed to {len(shapes)} shapes — it is supposed to "
            "cross its dimensions, and a property over a handful of inputs is a "
            "corpus wearing a property's clothes"
        )
        breaks = []
        for label, command in shapes:
            audit = audit_bash_command(command)
            if not (audit.tracked or audit.blind or audit.unaudited):
                breaks.append(f"{label}\n      {command!r}")
        assert not breaks, (
            f"{len(breaks)} of {len(shapes)} generated shapes were both UNTRACKED and "
            "UNREPORTED — each one is a write this instrument would record as a clean "
            "turn:\n   - " + "\n   - ".join(breaks[:20])
        )

    def test_a_resolvable_operand_is_actually_TRACKED_not_merely_reported_blind(self):
        """The other half, and the reason the disjunction above is not a free pass.

        A degenerate implementation satisfies "tracked OR blind" by reporting
        everything blind and tracking nothing. This pins the half that costs
        something: when the operand names a real path, it must be NAMED.
        """
        for label, command in _generated_clause_shapes():
            if "operand=$" in label or "operand=%" in label:
                continue                      # unresolved by construction
            audit = audit_bash_command(command)
            assert audit.tracked, (
                f"a nameable operand was not tracked — {label}\n   {command!r}\n"
                "   (reporting it blind instead would satisfy the invariant while "
                "giving up the name, which is the thing worth having)"
            )


class TestShapesFoundByRunningTheLexerRatherThanReasoningAboutIt:
    """Shapes an adversarial sweep found by EXECUTING candidate commands.

    Every row here was a real defect in the first draft of the lexer rewrite, found by
    running the tokenizer over generated shell rather than by reading the code. They
    are grouped by the root cause they share, because the causes are the reusable part:
    a fix aimed at one row of a group left the rest of the group broken.

    This is the same lesson as the file's patch history, arriving one layer up. The
    first draft replaced a table of regexes with a lexer and then did its heredoc
    stripping, its substitution collapsing and its comment handling with REGEXES OVER
    RAW TEXT — reintroducing quote-blindness, the exact defect the rewrite existed to
    retire, in the pre-pass. `gh pr create --body "use <<EOF for stdin"` matched a
    heredoc start inside quoted prose, found no terminator, and swallowed an `rm -rf`
    on the following line. Silently.
    """

    # --- shlex merges RUNS of punctuation into one token -------------------------
    @pytest.mark.parametrize("command,expected", [
        ("rm -f a.txt;\nrm -f b.txt",              ["a.txt", "b.txt"]),
        ("touch a.txt\n\ntouch b.txt",             ["a.txt", "b.txt"]),
        ("mkdir -p dist &&\ntouch dist/.keep",     ["dist", "dist/.keep"]),
        ("test -f x ||\n  touch x",                ["x"]),
        ("#!/bin/bash\nset -e\n\nmkdir -p dist\ntouch dist/x", ["dist", "dist/x"]),
        ("case $x in\n  a) touch /tmp/a;;\n  b) rm -rf /tmp/b;;\nesac", ["/tmp/a", "/tmp/b"]),
        ("ls |& rm -rf /tmp/junk",                 ["/tmp/junk"]),
    ])
    def test_a_run_of_separator_characters_still_splits_the_clause(self, command, expected):
        """`;` before a newline arrives as the single token `;\\n`, a blank line as
        `\\n\\n`, a trailing `&&` as `&&\\n`. Matching separators by literal spelling
        missed all of them: the clause never split, and for `rm` — a verb whose every
        operand is a target — the verifier claimed to have deleted a file literally
        named `;\\n`. Separators are classified by character CONTENT for this reason.
        """
        assert [p for p, _ in _extract_bash_paths(command)] == expected

    # --- the pre-pass must respect quoting ---------------------------------------
    @pytest.mark.parametrize("command,expected", [
        ('gh pr create --body "use <<EOF for stdin"\nrm -rf /tmp/wip', ["/tmp/wip"]),
        ('grep -rn "<<EOF" scripts/\nrm -f /tmp/stale',                ["/tmp/stale"]),
        ("echo '```' >> a.md\nrm -rf /tmp/build\necho '```' >> b.md",  ["a.md", "/tmp/build", "b.md"]),
        ("grep -q ok <<<yes\nrm -rf /tmp/build",                       ["/tmp/build"]),
    ])
    def test_a_quoted_heredoc_or_backtick_does_not_swallow_the_next_command(self, command, expected):
        """Each of these lost a REAL mutation to a quote-blind pre-pass.

        The markdown case is the sharpest: a fenced code block is three backticks, so
        writing one paired the fence's first backtick with one three lines later and
        deleted everything between — including the `rm -rf`.
        """
        assert [p for p, _ in _extract_bash_paths(command)] == expected

    # --- posix=False: a quoted token is a WORD whatever it spells ----------------
    @pytest.mark.parametrize("command,expected", [
        ("grep -rn '>' src/ > /tmp/hits.txt",     ["/tmp/hits.txt"]),
        ("grep -c '#' notes.md > /tmp/count.txt", ["/tmp/count.txt"]),
        ("echo '>' >> notes.md",                  ["notes.md"]),
    ])
    def test_a_quoted_operator_is_not_an_operator(self, command, expected):
        """Resolving quotes DURING tokenization erases the only thing separating a
        search pattern from a redirect. Under posix mode `grep -rn '>' src/ > out`
        reported `src/` — a directory the command only READ — as written to.
        """
        assert [p for p, _ in _extract_bash_paths(command)] == expected

    # --- adjacency the punctuation lexer destroys --------------------------------
    def test_a_multi_digit_operand_is_not_mistaken_for_a_file_descriptor(self):
        """`2>` and `2 >` lex identically, so the fd is recovered by popping a
        preceding digit — but only a SINGLE digit. `mkdir -p 2024 > /dev/null` popped
        `2024` as a phantom fd and lost the directory entirely.
        """
        assert _extract_bash_paths("mkdir -p 2024 > /dev/null") == [("2024", "mkdir")]
        assert _extract_bash_paths("echo hi 2> err.log") == [("err.log", "redirect_write")]

    def test_a_redirect_ending_in_ampersand_is_only_an_fd_dup_when_a_digit_follows(self):
        """`2>&1` duplicates a descriptor; `make >& build.log` writes a FILE. Both
        operators end in `&`, so the branch is decided by what follows."""
        assert _extract_bash_paths("make >& build.log") == [("build.log", "redirect_write")]
        assert _extract_bash_paths("cmd 2>&1") == []

    def test_an_equals_sign_inside_an_operand_does_not_shatter_it(self):
        """Making `=` a punctuation char merged `=>` into one harmless token, but broke
        `rm -f a=b.txt` into three claimed paths and merged `FOO= rm -f x` into the
        command word `FOO=rm`, losing a real delete. `=>` is neutralised in the
        quote-aware pre-pass instead."""
        assert _extract_bash_paths("rm -f a=b.txt") == [("a=b.txt", "delete")]
        assert [p for p, _ in _extract_bash_paths("FOO= rm -f x")] == ["x"]
        assert _extract_bash_paths("const f = (s) => /^smoke:/.test(s)") == []

    # --- subshells ---------------------------------------------------------------
    @pytest.mark.parametrize("command,expected", [
        ("(cd frontend && rm -rf node_modules)", ["node_modules"]),
        ("( cd /tmp && rm -rf work )",           ["work"]),
        ("(mv a.txt b.txt)",                     ["b.txt"]),
    ])
    def test_subshell_parentheses_do_not_glue_onto_a_path_or_a_verb(self, command, expected):
        """With parens outside the punctuation set they glued to the adjacent word:
        the tight form tracked a file named `node_modules)`, the spaced form tracked
        one named `)`, and `(mv a.txt b.txt)` made the command word `(mv` — not a
        mutation verb — so a real move went silent."""
        assert [p for p, _ in _extract_bash_paths(command)] == expected

    # --- mutations the command word alone cannot reach ---------------------------
    @pytest.mark.parametrize("command", [
        "find . -name '*.tmp' | xargs rm -f",
        r"find . -name '*.pyc' -exec rm -f {} \;",
        "find . -name '*.log' -delete",
        "sudo -u postgres rm -rf /var/lib/x",
        "timeout 5 rm -rf build",
        "cp -t backup/ a.txt b.txt",
    ])
    def test_a_mutation_the_lexer_cannot_name_is_reported_rather_than_dropped(self, command):
        """The invariant's hardest cases, and the ones a command-word rule alone gets
        WRONG by being silent rather than by being loud.

        `xargs rm -f` resolves its command word to `rm` and then finds no operands —
        the targets arrive on stdin. `find -exec` runs a second argv the clause's
        command word does not name. `sudo -u postgres rm` puts a flag ARGUMENT where
        the command word should be, because nothing here knows which flags take one.
        `cp -t DIR` inverts the destination-last order. None can be resolved to a path,
        so each produces a blindness row — which is the contract, not a consolation.
        """
        audit = audit_bash_command(command)
        assert audit.blind, f"{command!r} reported a clean turn for an unnameable mutation"
        assert not audit.tracked, f"{command!r} guessed a path it could not know"

    def test_a_script_passed_to_sh_c_is_audited_rather_than_treated_as_one_operand(self):
        """`sh -c '<script>'` hides a whole command inside a single operand. The
        script is ordinary shell, so the same walker applies to it."""
        assert _extract_bash_paths("sudo sh -c 'rm -rf /tmp/x'") == [("/tmp/x", "delete")]
        assert _extract_bash_paths('bash -c "touch /tmp/made.txt"') == [
            ("/tmp/made.txt", "touch")
        ]

    # --- heredoc termination -----------------------------------------------------
    def test_an_unterminated_heredoc_is_unaudited_rather_than_silently_eating_the_rest(self):
        """Bash accepts an indented terminator only for `<<-`, so `  EOF` does not
        close a plain `<<EOF`: the body runs to the end of the command and whatever
        followed is data of unknown extent. Reporting that as a clean turn would hide
        it; `unaudited` says the true thing."""
        audit = audit_bash_command(
            "cat > /tmp/f.txt <<EOF\nhello\n  EOF\nrm -f /tmp/real.txt"
        )
        assert audit.unaudited is not None
        assert not audit.tracked

    def test_a_correctly_terminated_heredoc_still_reads_the_command_around_it(self):
        assert [p for p, _ in _extract_bash_paths(
            "cat > /tmp/f <<'EOF'\nbody\nEOF\n\nrm -rf /tmp/old"
        )] == ["/tmp/f", "/tmp/old"]
        assert [p for p, _ in _extract_bash_paths(
            "cat <<-EOF\nbody\n\tEOF\nrm -f /tmp/z"
        )] == ["/tmp/z"]

    # --- line continuations ------------------------------------------------------
    def test_a_backslash_line_continuation_is_one_clause(self):
        """The shell deletes a backslash-newline; shlex leaves the newline, which then
        split the clause and stranded every operand after the break."""
        assert [p for p, _ in _extract_bash_paths("rm -rf \\\n  build \\\n  dist")] == [
            "build", "dist"
        ]
        assert _extract_bash_paths("mv old/report.md \\\n   archive/report.md") == [
            ("archive/report.md", "move")
        ]
