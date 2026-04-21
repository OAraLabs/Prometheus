"""
Divergence Detection — Catch when agent goes off-track, checkpoint/rollback.

Donor patterns:
- LCM DAG (memory/lcm/) — message persistence, summary relationships
- OpenClaw memory_extractor — fact extraction patterns
- Claude Code is_read_only — checkpoint before mutating ops

Source: Prometheus (OAra AI Lab)
License: MIT
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, TYPE_CHECKING

from prometheus.config.paths import get_config_dir

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_DB_NAME = "lcm.db"


def _default_db_path() -> Path:
    return get_config_dir() / _DB_NAME


# ============================================================================
# Goal Tracking (adapted from OpenClaw memory_extractor patterns)
# ============================================================================

@dataclass
class TaskGoal:
    """Represents the original task objective."""
    original_message: str
    goal_hash: str
    key_objectives: list[str]
    key_entities: list[str]


def extract_objectives(message: str) -> list[str]:
    """
    Extract key action items from a task message.

    Adapted from OpenClaw's fact extraction patterns.
    Looks for imperative verbs at sentence starts.
    """
    objectives: list[str] = []

    # Split into sentences
    sentences = re.split(r'[.!?]', message)

    action_patterns = [
        r'^(create|build|write|implement|add|fix|update|delete|remove|configure|setup|deploy)',
        r'^(search|find|look|check|verify|test|run|execute)',
        r'^(analyze|compare|evaluate|review|summarize|explain)',
        r'^(make|generate|produce|design|draft|compose)',
    ]

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        for pattern in action_patterns:
            if re.match(pattern, sent.lower()):
                objectives.append(sent)
                break

    # Fallback: first 200 chars if no explicit objectives
    if not objectives:
        objectives = [message[:200]]

    return objectives[:5]  # Max 5


def extract_entities(message: str) -> list[str]:
    """Extract key entities (files, names, concepts) from message."""
    entities: list[str] = []

    # File paths
    entities.extend(re.findall(r'[\w./\\-]+\.\w{1,5}', message))

    # Quoted strings
    entities.extend(re.findall(r'"([^"]+)"', message))
    entities.extend(re.findall(r"'([^']+)'", message))

    # Capitalized words (potential names)
    entities.extend(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', message))

    # Dedupe and limit
    return list(dict.fromkeys(entities))[:10]


class GoalTracker:
    """Track task goals and measure alignment."""

    def __init__(self) -> None:
        self.current_goal: Optional[TaskGoal] = None

    def set_goal(self, message: str) -> TaskGoal:
        """Set goal from initial task message."""
        goal_hash = hashlib.sha256(message.encode()).hexdigest()[:16]
        self.current_goal = TaskGoal(
            original_message=message,
            goal_hash=goal_hash,
            key_objectives=extract_objectives(message),
            key_entities=extract_entities(message),
        )
        logger.debug(
            f"Goal set: {len(self.current_goal.key_objectives)} objectives, "
            f"{len(self.current_goal.key_entities)} entities"
        )
        return self.current_goal

    def check_alignment(
        self,
        recent_messages: list[dict],
        tool_results: list[dict],
    ) -> float:
        """
        Check if recent activity aligns with goal.
        Returns alignment score 0.0 (off-track) to 1.0 (on-track).
        """
        if not self.current_goal:
            return 1.0  # No goal = assume on track

        # Combine recent text
        recent_text = " ".join([
            m.get("content", "")
            for m in recent_messages[-5:]
            if isinstance(m.get("content"), str)
        ])
        recent_text += " " + " ".join([
            str(r.get("result", ""))[:500]
            for r in tool_results[-5:]
        ])
        recent_lower = recent_text.lower()

        # Entity alignment
        entity_hits = sum(
            1 for e in self.current_goal.key_entities
            if e.lower() in recent_lower
        )
        entity_score = entity_hits / max(len(self.current_goal.key_entities), 1)

        # Objective keyword alignment
        objective_text = " ".join(self.current_goal.key_objectives).lower()
        objective_words = set(re.findall(r'\w+', objective_text))
        recent_words = set(re.findall(r'\w+', recent_lower))
        word_overlap = len(objective_words & recent_words) / max(len(objective_words), 1)

        # Combined score (entities weighted higher)
        return (entity_score * 0.4) + (word_overlap * 0.6)

    def clear(self) -> None:
        """Clear current goal."""
        self.current_goal = None


# ============================================================================
# Checkpoint (stored in LCM database)
# ============================================================================

@dataclass
class Checkpoint:
    """Snapshot of agent state at a point in time."""
    task_id: str
    step_number: int
    goal_description: str
    goal_hash: str
    messages_snapshot: list[dict]
    tool_calls: list[dict]
    timestamp: float = field(default_factory=time.time)
    divergence_score: float = 0.0

    def to_db_row(self) -> tuple:
        """Convert to database row values."""
        return (
            self.task_id,
            self.step_number,
            self.goal_hash,
            self.goal_description,
            json.dumps(self.messages_snapshot),
            json.dumps(self.tool_calls),
            self.divergence_score,
            self.timestamp,
        )

    @classmethod
    def from_db_row(cls, row: tuple) -> "Checkpoint":
        """Create from database row."""
        return cls(
            task_id=row[1],
            step_number=row[2],
            goal_hash=row[3],
            goal_description=row[4] or "",
            messages_snapshot=json.loads(row[5]),
            tool_calls=json.loads(row[6]),
            divergence_score=row[7],
            timestamp=row[8],
        )


# ============================================================================
# Checkpoint Store (extends LCM database — no separate db)
# ============================================================================

class CheckpointStore:
    """Manage checkpoint persistence in the shared lcm.db.

    Uses the same database as LCMConversationStore and LCMSummaryStore
    to keep all conversation state in one place.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path if db_path is not None else _default_db_path()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._apply_schema()

    def _apply_schema(self) -> None:
        self._conn.executescript("""
            PRAGMA journal_mode=WAL;

            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                step_number INTEGER NOT NULL,
                goal_hash TEXT NOT NULL,
                goal_description TEXT,
                messages_json TEXT NOT NULL,
                tool_calls_json TEXT NOT NULL,
                divergence_score REAL DEFAULT 0.0,
                created_at REAL NOT NULL,
                UNIQUE(task_id, step_number)
            );

            CREATE INDEX IF NOT EXISTS idx_checkpoints_task
                ON checkpoints(task_id, step_number DESC);
        """)
        self._conn.commit()

    def save(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint to database."""
        self._conn.execute(
            """INSERT OR REPLACE INTO checkpoints
               (task_id, step_number, goal_hash, goal_description,
                messages_json, tool_calls_json, divergence_score, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            checkpoint.to_db_row(),
        )
        self._conn.commit()

    def get_latest(self, task_id: str) -> Optional[Checkpoint]:
        """Get most recent checkpoint for a task."""
        row = self._conn.execute(
            """SELECT * FROM checkpoints
               WHERE task_id = ?
               ORDER BY step_number DESC LIMIT 1""",
            (task_id,),
        ).fetchone()
        if row:
            return Checkpoint.from_db_row(row)
        return None

    def delete_after(self, task_id: str, step_number: int) -> None:
        """Delete checkpoints after a given step (for rollback cleanup)."""
        self._conn.execute(
            "DELETE FROM checkpoints WHERE task_id = ? AND step_number > ?",
            (task_id, step_number),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "CheckpointStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ============================================================================
# Divergence Detector
# ============================================================================

@dataclass
class DivergenceResult:
    """Result of divergence evaluation."""
    score: float              # 0.0 = on track, 1.0 = completely off track
    should_rollback: bool
    reason: str
    checkpoint: Optional[Checkpoint] = None


class DivergenceDetector:
    """
    Detect when agent diverges from task goal.

    Uses LCM database for checkpoint persistence (not a separate database).
    Extends the existing memory infrastructure.
    """

    def __init__(
        self,
        config: dict,
        checkpoint_store: Optional[CheckpointStore] = None,
        notify_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        div_config = config.get("divergence", {})
        self.enabled = div_config.get("enabled", False)
        self.checkpoint_interval = div_config.get("checkpoint_interval", 5)
        self.threshold = div_config.get("threshold", 0.7)
        self.auto_rollback_trust = div_config.get("auto_rollback_trust_level", 3)
        self.max_rollbacks = div_config.get("max_rollbacks", 2)
        self.use_llm_eval = div_config.get("use_llm_eval", False)
        self.llm_eval_budget = div_config.get("llm_eval_budget", 500)

        self.checkpoint_store = checkpoint_store or CheckpointStore()
        self.goal_tracker = GoalTracker()
        self.notify_callback = notify_callback

        # Runtime state
        self.current_task_id: Optional[str] = None
        self.step_count: int = 0
        self.rollback_count: int = 0
        self.tool_calls_since_checkpoint: list[dict] = []

    def start_task(self, task_id: str, goal_message: str) -> None:
        """Initialize tracking for a new task."""
        self.current_task_id = task_id
        self.step_count = 0
        self.rollback_count = 0
        self.tool_calls_since_checkpoint = []
        self.goal_tracker.set_goal(goal_message)
        logger.info(f"Divergence tracking started: task={task_id}")

    def record_tool_call(
        self,
        tool_name: str,
        args: dict,
        result: object,
        success: bool,
    ) -> None:
        """Record a tool call for divergence analysis."""
        self.step_count += 1
        self.tool_calls_since_checkpoint.append({
            "step": self.step_count,
            "tool": tool_name,
            "args": args,
            "result": str(result)[:500],  # Truncate large results
            "success": success,
            "timestamp": time.time(),
        })

    def maybe_checkpoint(self, messages: list[dict]) -> Optional[Checkpoint]:
        """Create checkpoint if interval reached."""
        if not self.enabled or not self.current_task_id:
            return None

        if self.step_count > 0 and self.step_count % self.checkpoint_interval == 0:
            return self._create_checkpoint(messages)
        return None

    def _create_checkpoint(self, messages: list[dict]) -> Checkpoint:
        """Create and persist a checkpoint."""
        goal = self.goal_tracker.current_goal

        checkpoint = Checkpoint(
            task_id=self.current_task_id or "",
            step_number=self.step_count,
            goal_description=goal.original_message if goal else "",
            goal_hash=goal.goal_hash if goal else "",
            messages_snapshot=[
                m.copy() if isinstance(m, dict) else {"content": str(m)}
                for m in messages
            ],
            tool_calls=self.tool_calls_since_checkpoint.copy(),
        )

        # Persist to LCM store
        self.checkpoint_store.save(checkpoint)

        # Clear since-checkpoint buffer
        self.tool_calls_since_checkpoint = []

        logger.info(
            f"Checkpoint created: task={self.current_task_id}, step={self.step_count}"
        )
        return checkpoint

    def evaluate(
        self,
        messages: list[dict],
        tool_results: list[dict],
    ) -> DivergenceResult:
        """Evaluate current divergence from goal."""
        if not self.enabled or not self.current_task_id:
            return DivergenceResult(
                score=0.0,
                should_rollback=False,
                reason="disabled",
            )

        # Calculate divergence score
        score = self._calculate_score(messages, tool_results)

        # Determine if rollback needed
        should_rollback = (
            score >= self.threshold
            and self.rollback_count < self.max_rollbacks
        )

        # Get checkpoint for potential rollback
        checkpoint = None
        if should_rollback:
            checkpoint = self.checkpoint_store.get_latest(self.current_task_id)

        reason = self._build_reason(score, should_rollback)

        return DivergenceResult(
            score=score,
            should_rollback=should_rollback,
            reason=reason,
            checkpoint=checkpoint,
        )

    def _calculate_score(
        self,
        messages: list[dict],
        tool_results: list[dict],
    ) -> float:
        """
        Calculate divergence score 0.0-1.0.

        Heuristic scoring (no LLM cost):
        1. Goal alignment (inverted)
        2. Tool failure rate
        3. Repetition detection
        4. Context growth anomaly
        """
        scores: list[float] = []

        # 1. Goal alignment (inverted: low alignment = high divergence)
        alignment = self.goal_tracker.check_alignment(messages, tool_results)
        scores.append(1.0 - alignment)

        # 2. Tool failure rate
        recent_tools = self.tool_calls_since_checkpoint[-10:]
        if recent_tools:
            failures = sum(1 for t in recent_tools if not t["success"])
            failure_rate = failures / len(recent_tools)
            scores.append(failure_rate)

        # 3. Repetition detection (same tool > 3 times in a row)
        if len(recent_tools) >= 3:
            last_three = [t["tool"] for t in recent_tools[-3:]]
            if len(set(last_three)) == 1:
                scores.append(0.5)  # Repetition penalty

        # 4. Context growth anomaly
        if len(messages) > 20:
            growth_ratio = len(messages) / max(self.step_count, 1)
            if growth_ratio > 5:  # More than 5 messages per step
                scores.append(0.3)

        # Average all scores
        return sum(scores) / len(scores) if scores else 0.0

    def _build_reason(self, score: float, should_rollback: bool) -> str:
        """Build human-readable divergence reason."""
        if score < 0.3:
            return f"on_track (score={score:.2f})"
        elif score < 0.5:
            return f"minor_drift (score={score:.2f})"
        elif score < self.threshold:
            return f"moderate_drift (score={score:.2f})"
        else:
            if should_rollback:
                return f"diverged (score={score:.2f}), rollback_recommended"
            else:
                return f"diverged (score={score:.2f}), max_rollbacks_reached"

    def rollback(
        self,
        checkpoint: Checkpoint,
        trust_level: int,
    ) -> tuple[bool, list[dict]]:
        """
        Execute rollback to checkpoint.

        Returns (success, restored_messages).
        """
        if trust_level < self.auto_rollback_trust:
            # Need user confirmation for non-autonomous
            if self.notify_callback:
                self.notify_callback(
                    f"Task may be off-track.\n"
                    f"Divergence: {checkpoint.divergence_score:.2f}\n"
                    f"Step: {self.step_count}\n"
                    f"Reply 'rollback' to restore to step {checkpoint.step_number}"
                )
            return False, []

        # Auto-rollback for AUTONOMOUS trust
        self.rollback_count += 1
        self.step_count = checkpoint.step_number

        # Delete checkpoints after this one
        self.checkpoint_store.delete_after(
            self.current_task_id or "", checkpoint.step_number
        )

        logger.warning(
            f"Auto-rollback: task={self.current_task_id}, "
            f"to_step={checkpoint.step_number}, "
            f"rollback_count={self.rollback_count}/{self.max_rollbacks}"
        )

        if self.notify_callback:
            self.notify_callback(
                f"Auto-rollback to step {checkpoint.step_number}\n"
                f"Reason: divergence score {checkpoint.divergence_score:.2f}"
            )

        return True, checkpoint.messages_snapshot

    def end_task(self) -> None:
        """Clean up after task completion."""
        if self.current_task_id:
            logger.info(
                f"Task ended: {self.current_task_id}, "
                f"steps={self.step_count}, rollbacks={self.rollback_count}"
            )

        self.current_task_id = None
        self.goal_tracker.clear()
        self.tool_calls_since_checkpoint = []
