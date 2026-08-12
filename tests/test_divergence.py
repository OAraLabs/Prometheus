"""Tests for Divergence Detection — GoalTracker + DivergenceDetector."""

import asyncio
import sqlite3
from pathlib import Path
from typing import AsyncIterator

import pytest
from pydantic import BaseModel

from prometheus.coordinator.divergence import (
    GoalTracker,
    DivergenceDetector,
    Checkpoint,
    CheckpointStore,
    extract_objectives,
    extract_entities,
)
from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolUseBlock,
)
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiTextDeltaEvent,
    ModelProvider,
)
from prometheus.tools.base import (
    BaseTool,
    ToolExecutionContext,
    ToolRegistry,
    ToolResult,
)


class TestGoalExtraction:
    """Test objective and entity extraction."""

    def test_extract_objectives_imperative(self):
        message = "Create a dashboard. Fix the login bug. Deploy to production."
        objectives = extract_objectives(message)
        assert len(objectives) >= 2
        assert any("Create" in o for o in objectives)
        assert any("Fix" in o for o in objectives)

    def test_extract_objectives_fallback(self):
        message = "Hello, how are you today?"
        objectives = extract_objectives(message)
        assert len(objectives) == 1
        assert objectives[0] == message[:200]

    def test_extract_objectives_max_five(self):
        message = (
            "Create a. Build b. Write c. Implement d. Add e. Fix f. Update g."
        )
        objectives = extract_objectives(message)
        assert len(objectives) <= 5

    def test_extract_entities_files(self):
        message = "Edit the file config.yaml and check main.py"
        entities = extract_entities(message)
        assert "config.yaml" in entities
        assert "main.py" in entities

    def test_extract_entities_quoted(self):
        message = 'Set the variable to "hello world"'
        entities = extract_entities(message)
        assert "hello world" in entities

    def test_extract_entities_capitalized(self):
        message = "Talk to John about the Prometheus project"
        entities = extract_entities(message)
        assert "John" in entities or "Prometheus" in entities


class TestGoalTracker:
    """Test goal tracking and alignment."""

    def test_set_goal(self):
        tracker = GoalTracker()
        goal = tracker.set_goal("Create a Python script to parse JSON files")

        assert goal.goal_hash is not None
        assert len(goal.goal_hash) == 16
        assert len(goal.key_objectives) > 0

    def test_check_alignment_good(self):
        tracker = GoalTracker()
        tracker.set_goal("Create a Python script to parse JSON files")

        messages = [
            {"role": "assistant", "content": "I'll create a Python script for JSON parsing"}
        ]
        tool_results = [{"result": "Created parse_json.py"}]

        score = tracker.check_alignment(messages, tool_results)
        assert score > 0.3

    def test_check_alignment_poor(self):
        tracker = GoalTracker()
        tracker.set_goal("Create a Python script to parse JSON files")

        # Completely unrelated activity
        messages = [
            {"role": "assistant", "content": "Let me search for weather data"}
        ]
        tool_results = [{"result": "Weather in Tokyo: sunny, 25C"}]

        score = tracker.check_alignment(messages, tool_results)
        assert score < 0.5

    def test_no_goal_returns_1(self):
        tracker = GoalTracker()
        # No goal set
        score = tracker.check_alignment([], [])
        assert score == 1.0

    def test_clear_goal(self):
        tracker = GoalTracker()
        tracker.set_goal("Some task")
        tracker.clear()
        assert tracker.current_goal is None
        assert tracker.check_alignment([], []) == 1.0


class TestCheckpointStore:
    """Test checkpoint persistence in SQLite."""

    @pytest.fixture
    def store(self, tmp_path):
        db_path = tmp_path / "test_lcm.db"
        s = CheckpointStore(db_path=db_path)
        yield s
        s.close()

    def test_save_and_retrieve(self, store):
        cp = Checkpoint(
            task_id="t1",
            step_number=5,
            goal_description="Test goal",
            goal_hash="abc123",
            messages_snapshot=[{"role": "user", "content": "hello"}],
            tool_calls=[{"tool": "bash", "success": True}],
        )
        store.save(cp)

        latest = store.get_latest("t1")
        assert latest is not None
        assert latest.task_id == "t1"
        assert latest.step_number == 5
        assert len(latest.messages_snapshot) == 1

    def test_get_latest_returns_highest_step(self, store):
        for step in [5, 10, 15]:
            cp = Checkpoint(
                task_id="t1",
                step_number=step,
                goal_description="Test",
                goal_hash="abc",
                messages_snapshot=[],
                tool_calls=[],
            )
            store.save(cp)

        latest = store.get_latest("t1")
        assert latest is not None
        assert latest.step_number == 15

    def test_no_checkpoint_returns_none(self, store):
        assert store.get_latest("nonexistent") is None


class TestDivergenceDetector:
    """Test divergence detection and checkpointing."""

    @pytest.fixture
    def detector(self, tmp_path):
        db_path = tmp_path / "test_lcm.db"
        store = CheckpointStore(db_path=db_path)
        det = DivergenceDetector(
            {"divergence": {"enabled": True, "checkpoint_interval": 5, "threshold": 0.7}},
            checkpoint_store=store,
        )
        yield det
        store.close()

    def test_disabled(self, tmp_path):
        store = CheckpointStore(db_path=tmp_path / "d.db")
        detector = DivergenceDetector(
            {"divergence": {"enabled": False}},
            checkpoint_store=store,
        )
        result = detector.evaluate([], [])
        assert result.score == 0.0
        assert result.reason == "disabled"
        store.close()

    def test_start_task(self, detector):
        detector.start_task("test-1", "Create a Python script")
        assert detector.steps("test-1") == 0
        assert detector.live_tasks == 1

    def test_record_tool_call(self, detector):
        detector.start_task("test-1", "Test goal")
        detector.record_tool_call(
            "bash", {"command": "ls"}, "file.txt", True, task_id="test-1",
        )

        assert detector.steps("test-1") == 1

    def test_checkpoint_interval(self, detector):
        detector.start_task("test-1", "Test goal")

        # Record 4 tool calls - no checkpoint
        for _ in range(4):
            detector.record_tool_call("bash", {}, "ok", True, task_id="test-1")
            cp = detector.maybe_checkpoint([], task_id="test-1")
            assert cp is None

        # 5th call should trigger checkpoint
        detector.record_tool_call("bash", {}, "ok", True, task_id="test-1")
        cp = detector.maybe_checkpoint(
            [{"role": "user", "content": "test"}], task_id="test-1",
        )
        assert cp is not None
        assert cp.step_number == 5

    def test_divergence_scoring(self, detector):
        detector.start_task("test-1", "Create a Python script")

        # Record some failures
        for _ in range(5):
            detector.record_tool_call("bash", {}, "error", False, task_id="test-1")

        result = detector.evaluate([], [], task_id="test-1")

        # High failure rate should increase divergence score
        assert result.score > 0.3

    def test_end_task(self, detector):
        detector.start_task("test-1", "Some task")
        detector.record_tool_call("bash", {}, "ok", True, task_id="test-1")
        detector.end_task("test-1")

        assert detector.live_tasks == 0
        assert detector.steps("test-1") == 0

    def test_end_task_is_idempotent(self, detector):
        """``run_loop`` drains in a ``finally`` that can run after an early
        exit, so a second drop must not raise."""
        detector.start_task("test-1", "Some task")
        detector.end_task("test-1")
        detector.end_task("test-1")
        detector.end_task("never-started")

    def test_disabled_detector_accrues_nothing(self, tmp_path):
        """The buffer that grew for four months.

        ``record_tool_call`` was guarded by neither ``enabled`` nor an open
        task, and the only three methods that cleared
        ``tool_calls_since_checkpoint`` were the three that never ran — so
        every tool call the daemon dispatched appended a dict carrying args
        and 500 chars of result to a list nothing freed for the life of the
        process.
        """
        store = CheckpointStore(db_path=tmp_path / "d.db")
        detector = DivergenceDetector(
            {"divergence": {"enabled": False}}, checkpoint_store=store,
        )
        detector.start_task("t", "goal")
        for _ in range(50):
            detector.record_tool_call("bash", {}, "x" * 5000, True, task_id="t")

        assert detector.live_tasks == 0, "disabled detector opened a task"
        assert detector.steps("t") == 0
        store.close()

    def test_unknown_task_id_records_nothing(self, detector):
        """A call arriving with no open task must not materialise state —
        otherwise the pre-FL-4 unbounded buffer comes back under a new name.

        This is the control that actually stops the leak. ``start_task`` is
        the single ``enabled`` gate; everything downstream keys off "is there
        a record?", so this is the assertion that pins it.
        """
        detector.record_tool_call("bash", {}, "ok", True, task_id="never-started")
        assert detector.live_tasks == 0
        assert detector.steps("never-started") == 0

    def test_unknown_task_id_makes_no_checkpoint(self, detector):
        assert detector.maybe_checkpoint([], task_id="never-started") is None

    def test_evaluate_without_a_task_reports_no_task(self, detector):
        """Distinct from ``disabled`` on purpose — §3b guard identity. If both
        returned the same reason, a test asserting "it didn't score" could not
        tell which control refused.
        """
        result = detector.evaluate([], [], task_id="never-started")
        assert result.score == 0.0
        assert result.reason == "no_task"
        assert result.diverged is False


class TestConcurrentTasks:
    """ONE detector is shared process-wide (daemon.py builds it once and hands
    the same object to the AgentLoop and to the startup-built web
    LoopContext). Task state therefore may not live on the detector.

    This is CROSS-CUTTING §2's mirror image: the naive fix for "a value that
    must be per-call" is to pass it at both construction sites, which is
    correct for duplicate construction and ships cross-talk for a shared
    instance. These tests pin the shape that cannot cross-talk.
    """

    @pytest.fixture
    def detector(self, tmp_path):
        store = CheckpointStore(db_path=tmp_path / "shared.db")
        det = DivergenceDetector(
            {"divergence": {"enabled": True, "checkpoint_interval": 5}},
            checkpoint_store=store,
        )
        yield det
        store.close()

    def test_step_counts_do_not_bleed(self, detector):
        detector.start_task("session-a", "Fix the login bug")
        detector.start_task("session-b", "Write the release notes")

        for _ in range(4):
            detector.record_tool_call("bash", {}, "ok", True, task_id="session-a")
        detector.record_tool_call("grep", {}, "ok", True, task_id="session-b")

        assert detector.steps("session-a") == 4
        assert detector.steps("session-b") == 1

    def test_starting_a_second_task_does_not_reset_the_first(self, detector):
        """The live bug the old shape would have had: B's start_task zeroed
        A's counters mid-turn."""
        detector.start_task("session-a", "Fix the login bug")
        for _ in range(3):
            detector.record_tool_call("bash", {}, "ok", True, task_id="session-a")

        detector.start_task("session-b", "Something else entirely")

        assert detector.steps("session-a") == 3

    def test_checkpoints_carry_their_own_goal(self, detector):
        """A's checkpoint must not be stamped with B's goal hash."""
        detector.start_task("session-a", "Create a Python script to parse JSON")
        detector.start_task("session-b", "Book a flight to Tokyo")

        for _ in range(5):
            detector.record_tool_call("bash", {}, "ok", True, task_id="session-a")
        cp_a = detector.maybe_checkpoint([], task_id="session-a")

        assert cp_a is not None
        assert "JSON" in cp_a.goal_description
        assert "Tokyo" not in cp_a.goal_description

    def test_ending_one_task_leaves_the_other(self, detector):
        detector.start_task("session-a", "A")
        detector.start_task("session-b", "B")
        detector.record_tool_call("bash", {}, "ok", True, task_id="session-b")

        detector.end_task("session-a")

        assert detector.live_tasks == 1
        assert detector.steps("session-b") == 1

    def test_live_tasks_are_bounded(self, detector):
        """A caller that never drains must not grow the map without limit."""
        from prometheus.coordinator.divergence import MAX_LIVE_TASKS

        for i in range(MAX_LIVE_TASKS + 10):
            detector.start_task(f"leaky-{i}", "goal")

        assert detector.live_tasks == MAX_LIVE_TASKS


# ===========================================================================
# The outcome test — §2d, "registered is not advertised"
# ===========================================================================


class _EchoInput(BaseModel):
    text: str = "hello"


class _EchoTool(BaseTool):
    name = "echo"
    description = "Echo text"
    input_model = _EchoInput

    async def execute(
        self, arguments: BaseModel, context: ToolExecutionContext,
    ) -> ToolResult:
        return ToolResult(output=arguments.text)

    def is_read_only(self, arguments: BaseModel) -> bool:
        return True


class _ScriptedProvider(ModelProvider):
    """Emits one tool call, then plain text so the loop terminates."""

    def __init__(self) -> None:
        self._calls = 0

    async def stream_message(
        self, request: ApiMessageRequest,
    ) -> AsyncIterator:
        self._calls += 1
        if self._calls == 1:
            yield ApiMessageCompleteEvent(
                message=ConversationMessage(
                    role="assistant",
                    content=[ToolUseBlock(
                        id="t1", name="echo", input={"text": "hi"},
                    )],
                ),
                usage=UsageSnapshot(input_tokens=10, output_tokens=10),
                stop_reason="tool_calls",
            )
        else:
            yield ApiTextDeltaEvent(text="done")
            yield ApiMessageCompleteEvent(
                message=ConversationMessage(
                    role="assistant", content=[TextBlock(text="done")],
                ),
                usage=UsageSnapshot(input_tokens=10, output_tokens=5),
                stop_reason="stop",
            )


def _checkpoint_rows(db_path: Path) -> list[tuple]:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            "SELECT task_id, step_number, goal_description FROM checkpoints"
        ).fetchall()
    finally:
        conn.close()


class TestCheckpointsLandThroughTheRealLoop:
    """A checkpoint ROW must exist after a real turn.

    Standing Principles §2d — assert the artefact the CONSUMER receives, not
    the container it was put in. Every structural check this feature had was
    green for four months while it wrote nothing:

      * the detector was CONSTRUCTED on every boot, both entry points;
      * ``record_tool_call`` FIRED on every tool call;
      * ``tests/test_wiring.py`` asserted exactly that and passed;
      * ``tests/test_divergence.py`` exercised the whole lifecycle by
        calling ``start_task`` ITSELF — the one call production never made.

    ...and the ``checkpoints`` table had 0 rows in both lcm.db files on the
    live box, with ``divergence.enabled: true`` in the live config.

    So the consumer here is the table, and the assertion is a row — driven
    through the real ``run_loop``, which is the only place the task scope is
    minted.
    """

    def _drive(self, tmp_path: Path, *, enabled: bool = True) -> Path:
        db_path = tmp_path / "lcm.db"
        store = CheckpointStore(db_path=db_path)
        detector = DivergenceDetector(
            # interval 1: one tool call is one checkpoint, so the test does
            # not depend on how many rounds the scripted provider runs.
            {"divergence": {"enabled": enabled, "checkpoint_interval": 1}},
            checkpoint_store=store,
        )
        registry = ToolRegistry()
        registry.register(_EchoTool())

        ctx = LoopContext(
            provider=_ScriptedProvider(),
            model="test",
            system_prompt="sys",
            max_tokens=1024,
            tool_registry=registry,
            divergence_detector=detector,
        )
        messages = [ConversationMessage.from_user_text(
            "Create a Python script to parse JSON files"
        )]

        async def _run() -> None:
            async for _ in run_loop(ctx, messages, session_id="s-1"):
                pass

        asyncio.run(_run())
        store.close()
        return db_path

    def test_a_checkpoint_row_lands(self, tmp_path):
        rows = _checkpoint_rows(self._drive(tmp_path))
        assert rows, (
            "no checkpoint row after a real run_loop turn — this is the "
            "exact state the live box was in for four months"
        )

    def test_the_row_carries_the_real_goal(self, tmp_path):
        """Not just 'a row appeared'.

        ``_create_checkpoint`` falls back to ``goal_description=""`` when no
        goal is set, so a count-only assertion would still pass if
        ``start_task`` were called with an empty goal — or if the goal were
        resolved from the wrong end of the message list. Pinning the text
        makes the row prove the whole chain: run_loop found the user turn,
        start_task set the goal, and the checkpoint carried it.
        """
        rows = _checkpoint_rows(self._drive(tmp_path))
        assert rows
        task_ids = {r[0] for r in rows}
        assert all("s-1:" in t for t in task_ids), (
            f"task id not scoped to the session: {task_ids}"
        )
        assert all("parse JSON files" in r[2] for r in rows), (
            f"checkpoint did not carry the user's goal: {[r[2] for r in rows]}"
        )

    def test_disabled_writes_nothing(self, tmp_path):
        """The other direction — §2c. A control needs both."""
        assert _checkpoint_rows(self._drive(tmp_path, enabled=False)) == []

    def test_task_state_is_drained_after_the_turn(self, tmp_path):
        """``run_loop``'s ``finally`` must drop the record, or the shared
        detector accumulates one entry per turn forever."""
        db_path = tmp_path / "lcm.db"
        store = CheckpointStore(db_path=db_path)
        detector = DivergenceDetector(
            {"divergence": {"enabled": True, "checkpoint_interval": 1}},
            checkpoint_store=store,
        )
        registry = ToolRegistry()
        registry.register(_EchoTool())
        ctx = LoopContext(
            provider=_ScriptedProvider(),
            model="test",
            system_prompt="sys",
            max_tokens=1024,
            tool_registry=registry,
            divergence_detector=detector,
        )

        async def _run() -> None:
            async for _ in run_loop(
                ctx,
                [ConversationMessage.from_user_text("do a thing")],
                session_id="s-1",
            ):
                pass

        asyncio.run(_run())
        assert detector.live_tasks == 0, (
            "run_loop did not drain the task record"
        )
        store.close()
