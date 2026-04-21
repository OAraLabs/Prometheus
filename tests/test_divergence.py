"""Tests for Divergence Detection — GoalTracker + DivergenceDetector."""

import tempfile
from pathlib import Path

import pytest

from prometheus.coordinator.divergence import (
    GoalTracker,
    DivergenceDetector,
    Checkpoint,
    CheckpointStore,
    extract_objectives,
    extract_entities,
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

    def test_delete_after(self, store):
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

        store.delete_after("t1", 5)

        latest = store.get_latest("t1")
        assert latest is not None
        assert latest.step_number == 5

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
        assert detector.current_task_id == "test-1"
        assert detector.step_count == 0
        assert detector.goal_tracker.current_goal is not None

    def test_record_tool_call(self, detector):
        detector.start_task("test-1", "Test goal")
        detector.record_tool_call("bash", {"command": "ls"}, "file.txt", True)

        assert detector.step_count == 1
        assert len(detector.tool_calls_since_checkpoint) == 1

    def test_checkpoint_interval(self, detector):
        detector.start_task("test-1", "Test goal")

        # Record 4 tool calls - no checkpoint
        for i in range(4):
            detector.record_tool_call("bash", {}, "ok", True)
            cp = detector.maybe_checkpoint([])
            assert cp is None

        # 5th call should trigger checkpoint
        detector.record_tool_call("bash", {}, "ok", True)
        cp = detector.maybe_checkpoint([{"role": "user", "content": "test"}])
        assert cp is not None
        assert cp.step_number == 5

    def test_divergence_scoring(self, detector):
        detector.start_task("test-1", "Create a Python script")

        # Record some failures
        for i in range(5):
            detector.record_tool_call("bash", {}, "error", False)

        result = detector.evaluate([], [])

        # High failure rate should increase divergence score
        assert result.score > 0.3

    def test_end_task(self, detector):
        detector.start_task("test-1", "Some task")
        detector.record_tool_call("bash", {}, "ok", True)
        detector.end_task()

        assert detector.current_task_id is None
        assert detector.tool_calls_since_checkpoint == []

    def test_rollback_below_trust(self, detector):
        """Rollback should notify but not execute below trust threshold."""
        notifications: list[str] = []
        detector.notify_callback = notifications.append
        detector.start_task("test-1", "Test")

        cp = Checkpoint(
            task_id="test-1",
            step_number=5,
            goal_description="Test",
            goal_hash="abc",
            messages_snapshot=[{"role": "user", "content": "hello"}],
            tool_calls=[],
        )

        success, msgs = detector.rollback(cp, trust_level=1)  # Below auto threshold
        assert success is False
        assert msgs == []
        assert len(notifications) == 1

    def test_rollback_at_autonomous_trust(self, detector):
        """Rollback should auto-execute at AUTONOMOUS trust."""
        detector.start_task("test-1", "Test")
        detector.step_count = 10

        cp = Checkpoint(
            task_id="test-1",
            step_number=5,
            goal_description="Test",
            goal_hash="abc",
            messages_snapshot=[{"role": "user", "content": "hello"}],
            tool_calls=[],
        )

        success, msgs = detector.rollback(cp, trust_level=3)  # AUTONOMOUS
        assert success is True
        assert len(msgs) == 1
        assert detector.rollback_count == 1
        assert detector.step_count == 5
