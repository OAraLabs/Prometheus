"""
Divergence Detection — observe when the agent drifts from the task goal.

Donor patterns:
- LCM DAG (memory/lcm/) — message persistence, summary relationships
- OpenClaw memory_extractor — fact extraction patterns
- Claude Code is_read_only — checkpoint before mutating ops

DETECTION ONLY — the rollback half was RETIRED (FL-4), not shipped dark.
Sprint 10 built a checkpoint/rollback feature that was broken in four
independent places, so no checkpoint row was ever written on any box:

1. ``start_task`` had NO CALLER anywhere in ``src/``, so ``current_task_id``
   stayed ``None`` and both ``maybe_checkpoint`` and ``evaluate`` returned at
   their first guard. This is the one the survey started from.
2. ``notify_callback`` was never passed by either construction site
   (``__main__.create_divergence_detector``), so the human-in-the-loop
   branch could not fire.
3. The loop hardcoded ``trust = 1`` while ``auto_rollback_trust_level``
   defaults to (and the live config set) ``3`` — so ``rollback()`` took the
   notify branch unconditionally and auto-rollback was unreachable.
4. On the unreachable success path the restored messages were DISCARDED —
   the call site used them only for ``len(restored)`` in a log line. The
   conversation was never rewound. Rollback did not roll back.

Wiring only (1) would have produced checkpoint rows feeding a rollback that
still could not function — a slower version of the same defect. The scoring
heuristic below has also never run once in production, and auto-rewinding a
live conversation on an unvalidated entity/word-overlap score is not a
default worth arming. So the detector now WRITES checkpoints and REPORTS a
divergence score, and does nothing else. Rollback, ``notify_callback``,
``max_rollbacks``, ``auto_rollback_trust_level`` and
``CheckpointStore.delete_after`` are gone rather than left as dead branches.

Not to be confused with FL-3, which landed the same day (see
:class:`CheckpointStore`'s docstring): that fixed a SECOND, independent
reason the table looked empty — this class resolved its default db path to
a different file than the stores it claimed to share. FL-3 removed the rival
explanation; the orphan above is why nothing was ever written to EITHER file.

The ``checkpoints`` table is forensic: ``get_latest`` is its read half and
has no production caller by design. Stated here rather than left to be
rediscovered.

CONCURRENCY — read this before adding state. ONE ``DivergenceDetector`` is
constructed per process (``daemon.py`` builds it once and hands the SAME
object to the ``AgentLoop`` and to the startup-built web ``LoopContext``),
so nothing task-scoped may live on the detector itself. It used to:
``current_task_id``, ``step_count``, ``rollback_count`` and the goal were
plain attributes, which under concurrent sessions is cross-talk, not drift —
session B's ``start_task`` would reset session A's counters mid-turn and A's
checkpoints would land under B's goal hash. State is therefore keyed by a
``task_id`` minted per ``run_loop`` invocation, exactly as
``FileMutationVerifier`` keys by ``turn_key`` and for the same reason.

Source: Prometheus (OAra Labs)
License: MIT
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from uuid import uuid4

from prometheus.config.paths import get_lcm_db_path

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

#: Task key used when a caller supplies none. Correct only for
#: single-threaded callers (benchmarks, evals, the CLI) — every concurrent
#: surface mints its own via :meth:`DivergenceDetector.new_task_id`.
DEFAULT_TASK_KEY = "default"

# How many recent tool calls the SCORING terms look at. Independent of
# checkpoint_interval on purpose: the scoring window must not be a function of
# how often state happens to be persisted.
SCORING_WINDOW = 10

# UNPRODUCTIVE repetition is a FLOOR on the score, not one summand among four.
#
# As a summand it was arithmetically incapable of declaring divergence: an
# agent visibly stuck — seven greps in a row — produced mean([0.2, 0.0, 0.5])
# = 0.23 and read "on_track". A signal that cannot reach the threshold from
# inside a mean is not weak, it is decorative.
#
# ⚠ BUT THE FLOOR MUST NOT SIT UNDER *ANY* REPETITION, and finding that out
# cost a round. Flooring "same tool three times running" made a compliant
# deploy proof (bash x6) and a stuck agent (grep x7) score IDENTICALLY —
# 0.750, diverged, byte-identical components. Tool-name repetition
# discriminates nothing: a run of `bash` is the most ordinary shape in this
# system. Promoting a signal to decisive without asking what it separates
# just makes a noisy term authoritative.
#
# So the floor sits under UNPRODUCTIVE repetition only: the same tool, N times,
# returning nothing or returning the same thing. That is what separates the two
# traces — the deploy proof's echoes each return a distinct non-empty value,
# the stuck greps all return "".
REPETITION_FLOOR = 0.75

# How many same-tool calls in a row before repetition is considered at all.
REPETITION_RUN = 3

# Terms that describe what the agent DID, as opposed to how its text reads.
# At least one must agree before a verdict of "diverged" is issued.
BEHAVIOURAL_TERMS = ("tool_failure_rate", "repetition", "context_growth")

#: Ceiling on simultaneously-tracked tasks. A task is dropped by
#: ``end_task``; this bounds the damage when a caller never calls it.
MAX_LIVE_TASKS = 32


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

    That sentence was false from the first commit until 2026-08-12: this class
    resolved its own default to ``get_config_dir() / "lcm.db"`` while the two
    stores it names were opened by ``LCMEngine`` at ``get_data_dir() /
    "lcm.db"``. Both files carry a ``checkpoints`` table — the conversation
    store's schema creates one too — so a reader who found the table in the
    data-dir file and saw it empty had no way to tell it was the wrong one.
    Everything now resolves through
    :func:`~prometheus.config.paths.get_lcm_db_path`.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path if db_path is not None else get_lcm_db_path()
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
        """Get most recent checkpoint for a task.

        The read half of a forensic table — no production caller by design
        (see the module docstring). Used by tests to prove a row landed.
        """
        row = self._conn.execute(
            """SELECT * FROM checkpoints
               WHERE task_id = ?
               ORDER BY step_number DESC LIMIT 1""",
            (task_id,),
        ).fetchone()
        if row:
            return Checkpoint.from_db_row(row)
        return None

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
    """Result of divergence evaluation. Observational — see the module
    docstring on why there is no ``should_rollback``.

    ``components`` carries every sub-score that fired, by name. The first live
    observation of this detector was a FALSE POSITIVE scoring 0.96, and the
    mechanism had to be reconstructed by reading the arithmetic back weeks
    later — because the only thing recorded was the total. A verdict whose
    inputs are not written down has to be re-derived by whoever next doubts
    it, and re-derivation is where the checkpoint-clearing interaction was
    missed the first time.
    """
    score: float              # 0.0 = on track, 1.0 = completely off track
    reason: str
    diverged: bool = False    # score >= threshold; a SIGNAL, not an action
    components: dict[str, float] = field(default_factory=dict)


@dataclass
class _TaskState:
    """Everything that belongs to ONE task.

    Lives in :attr:`DivergenceDetector._tasks`, never on the detector — the
    detector is a process-wide singleton (see the module docstring).
    """
    task_id: str
    goal_tracker: GoalTracker = field(default_factory=GoalTracker)
    step_count: int = 0
    # Checkpoint PAYLOAD. Cleared every time a checkpoint is written — that is
    # its job, and it is correct for that job.
    tool_calls_since_checkpoint: list[dict] = field(default_factory=list)
    # SCORING WINDOW — deliberately separate, and never cleared by a
    # checkpoint.
    #
    # ⚠ These were one list, and that is the mechanism of the 0.96 false
    # positive. `_create_checkpoint` clears the payload, and `evaluate` runs
    # immediately after `maybe_checkpoint` in the same block — so on EVERY
    # checkpoint-boundary step the tool window was empty, the failure-rate and
    # repetition terms were skipped for want of data, and the average
    # collapsed to the goal-alignment term alone. A turn of six successful
    # `echo` calls scored 0.96 on lexical dissimilarity and was logged as
    # "diverged".
    #
    # One list serving a payload scope and a scoring scope is
    # CROSS-CUTTING §10: over-applied for one reader and under-applied for the
    # other, simultaneously.
    recent_tool_calls: deque[dict] = field(
        default_factory=lambda: deque(maxlen=SCORING_WINDOW))


class DivergenceDetector:
    """
    Observe when the agent diverges from the task goal.

    Uses LCM database for checkpoint persistence (not a separate database).
    Extends the existing memory infrastructure.

    ONE instance is shared process-wide, so every entry point takes a
    ``task_id`` naming which in-flight task it belongs to. Keys are minted
    per ``run_loop`` invocation via :meth:`new_task_id`; omitting one falls
    back to :data:`DEFAULT_TASK_KEY`, which is correct only for
    single-threaded callers.

    Lifecycle:
      start_task(task_id, goal_message)   — opens a task's record
      record_tool_call(..., task_id=...)  — one step
      maybe_checkpoint(messages, task_id=...) — persists every Nth step
      evaluate(messages, results, task_id=...) — scores drift
      end_task(task_id=...)               — drops the record (idempotent)
    """

    def __init__(
        self,
        config: dict,
        checkpoint_store: Optional[CheckpointStore] = None,
    ) -> None:
        div_config = config.get("divergence", {})
        self.enabled = div_config.get("enabled", False)
        self.checkpoint_interval = div_config.get("checkpoint_interval", 5)
        self.threshold = div_config.get("threshold", 0.7)
        # Self-halt (post-FL-4): whether the LOOP may act on a sustained
        # repetition verdict. FL-4 retired the ROLLBACK half — rewinding a
        # conversation on a score was the dangerous action, and this is not
        # that: the loop ends the turn FORWARD, keeping every completed
        # round, after the repetition floor holds for consecutive
        # evaluations (agent_loop._DIVERGENCE_HALT_AFTER). The 2026-08-17
        # reproduction warned on every iteration of a 30-step flail and
        # changed nothing — "a detector that only warns trains everyone to
        # read its warning as weather." Live-traffic calibration (7 days):
        # healthy turns blip the floor for exactly ONE evaluation, so the
        # consecutive requirement is what separates them from the incident.
        self.halt_on_repetition = bool(
            div_config.get("halt_on_repetition", True)
        )
        # ``use_llm_eval`` / ``llm_eval_budget`` were read here into attributes
        # nothing consumed — a config key promising an LLM-backed evaluator
        # that was never built, kept plausible by the assignment itself (the
        # reader-direction drift guard greps for the key name, and an
        # assignment satisfies it). Dropped with the rollback half; scoring is
        # the heuristic below and nothing else.

        self.checkpoint_store = checkpoint_store or CheckpointStore()

        # task_id -> state, least-recently-touched first. Guarded by
        # ``_lock``: turns are driven by asyncio and interleave at every
        # ``await``, and gateways may drive the loop from a worker thread.
        self._tasks: OrderedDict[str, _TaskState] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def new_task_id(session_id: str | None = None) -> str:
        """Mint a key for one task. Unique per call — a session id alone is
        NOT enough, since a session can have more than one turn in flight."""
        return f"{session_id or 'anon'}:{uuid4().hex}"

    @property
    def live_tasks(self) -> int:
        """Number of tasks currently holding state. Diagnostics only; a
        number that keeps climbing means a caller isn't calling end_task."""
        with self._lock:
            return len(self._tasks)

    # ------------------------------------------------------------------
    # Lifecycle — called by agent_loop
    # ------------------------------------------------------------------

    def start_task(self, task_id: str, goal_message: str) -> None:
        """Initialize tracking for a new task.

        A no-op when disabled: without this, ``record_tool_call`` accrues
        state for a feature that will never read it.
        """
        if not self.enabled:
            return
        state = _TaskState(task_id=task_id)
        state.goal_tracker.set_goal(goal_message)
        with self._lock:
            self._tasks[task_id] = state
            while len(self._tasks) > MAX_LIVE_TASKS:
                evicted, _ = self._tasks.popitem(last=False)
                logger.warning(
                    "DivergenceDetector: evicted live task %r (>%d) — that "
                    "task gets no further checkpoints; a caller is not "
                    "calling end_task",
                    evicted, MAX_LIVE_TASKS,
                )
        logger.info("Divergence tracking started: task=%s", task_id)

    def record_tool_call(
        self,
        tool_name: str,
        args: dict,
        result: object,
        success: bool,
        *,
        task_id: str | None = None,
    ) -> None:
        """Record a tool call for divergence analysis.

        Accrues state ONLY for a task with an open record. Before FL-4 it was
        guarded by nothing, and the only three methods that cleared
        ``tool_calls_since_checkpoint`` were the three that never ran — so
        every tool call the daemon ever dispatched appended a dict carrying
        the args and 500 chars of result to a list nothing freed for the
        life of the process.

        No ``enabled`` check here deliberately. ``start_task`` is the single
        gate: a disabled detector opens no record, so an ``enabled`` guard on
        this path is unreachable — and an unreachable guard is not
        defence in depth, it is a control no test can pin. Mutation M3 proved
        it: deleting it left every test green, because the ``state is None``
        return below refuses the same call for a different reason
        (Standing Principles §3b). The invariant this relies on: a
        ``_TaskState`` exists only if ``start_task`` created it.
        """
        with self._lock:
            state = self._get(task_id)
            if state is None:
                return
            state.step_count += 1
            entry = {
                "step": state.step_count,
                "tool": tool_name,
                "args": args,
                "result": str(result)[:500],  # Truncate large results
                "success": success,
                "timestamp": time.time(),
            }
            state.tool_calls_since_checkpoint.append(entry)
            # The scoring window survives checkpointing; see _TaskState.
            state.recent_tool_calls.append(entry)

    def steps(self, task_id: str | None = None) -> int:
        """Steps recorded for a task; 0 when it has no open record.

        A method, not the old ``step_count`` attribute, deliberately: a
        duck-typed caller still reading the attribute gets an
        ``AttributeError`` rather than a bound method that is truthy and
        ``TypeError``s on comparison.
        """
        with self._lock:
            state = self._get(task_id)
            return state.step_count if state else 0

    def maybe_checkpoint(
        self,
        messages: list[dict],
        *,
        task_id: str | None = None,
    ) -> Optional[Checkpoint]:
        """Create checkpoint if interval reached.

        Gated on an open record, not on ``enabled`` — same reasoning as
        ``record_tool_call``.
        """
        with self._lock:
            state = self._get(task_id)
            if state is None:
                return None
            due = (
                state.step_count > 0
                and state.step_count % self.checkpoint_interval == 0
            )
            if not due:
                return None
            return self._create_checkpoint(state, messages)

    def _create_checkpoint(
        self, state: _TaskState, messages: list[dict],
    ) -> Checkpoint:
        """Create and persist a checkpoint. Caller holds ``_lock``."""
        goal = state.goal_tracker.current_goal

        checkpoint = Checkpoint(
            task_id=state.task_id,
            step_number=state.step_count,
            goal_description=goal.original_message if goal else "",
            goal_hash=goal.goal_hash if goal else "",
            messages_snapshot=[
                m.copy() if isinstance(m, dict) else {"content": str(m)}
                for m in messages
            ],
            tool_calls=list(state.tool_calls_since_checkpoint),
        )

        # Persist to LCM store
        self.checkpoint_store.save(checkpoint)

        # Clear since-checkpoint buffer
        state.tool_calls_since_checkpoint = []

        logger.info(
            "Checkpoint created: task=%s, step=%d",
            state.task_id, state.step_count,
        )
        return checkpoint

    def evaluate(
        self,
        messages: list[dict],
        tool_results: list[dict],
        *,
        task_id: str | None = None,
    ) -> DivergenceResult:
        """Evaluate current divergence from goal."""
        if not self.enabled:
            return DivergenceResult(score=0.0, reason="disabled")
        with self._lock:
            state = self._get(task_id)
            if state is None:
                return DivergenceResult(score=0.0, reason="no_task")

            components = self._components(state, messages, tool_results)
            score = self._aggregate(components)

        diverged = self._is_diverged(components, score, self.threshold)
        return DivergenceResult(
            score=score,
            reason=self._build_reason(score, components, diverged),
            diverged=diverged,
            components=components,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get(self, task_id: str | None) -> Optional[_TaskState]:
        """This task's state, or None. Caller holds ``_lock``."""
        key = task_id or DEFAULT_TASK_KEY
        state = self._tasks.get(key)
        if state is not None:
            self._tasks.move_to_end(key)
        return state

    def _components(
        self,
        state: _TaskState,
        messages: list[dict],
        tool_results: list[dict],
    ) -> dict[str, float]:
        """Every sub-score that FIRED, by name. No aggregation here.

        Separated from the aggregation deliberately. The old function computed
        terms and collapsed them in one breath, so the collapse could not be
        tested and the terms could not be reported — and both of those are the
        reasons the 0.96 false positive took weeks to explain.
        """
        out: dict[str, float] = {}

        # 1. Goal alignment, inverted. ALWAYS present, and the weakest signal
        #    here: it is bag-of-words overlap between the recent text and
        #    entities regex-extracted from the original request. A turn whose
        #    work is command output, SHAs and ports shares almost no vocabulary
        #    with the prose that asked for it, and scores as "diverged" for
        #    doing exactly what it was told.
        alignment = state.goal_tracker.check_alignment(messages, tool_results)
        out["goal_alignment"] = 1.0 - alignment

        # The scoring window, NOT the checkpoint payload — see _TaskState.
        recent = list(state.recent_tool_calls)

        # 2. Tool failure rate.
        if recent:
            out["tool_failure_rate"] = (
                sum(1 for c in recent if not c["success"]) / len(recent))

        # 3. UNPRODUCTIVE repetition — the same tool N times running, getting
        #    nowhere. "Getting nowhere" is the load-bearing half: same-tool
        #    repetition ALONE fires on a run of `bash`, which is what ordinary
        #    work looks like here.
        #
        #    KNOWN MISSES, recorded rather than fixed:
        #      * strict adjacency over the last REPETITION_RUN calls, so a
        #        seven-grep flail with anything interleaved is invisible;
        #      * argument-similarity over a window would catch that and is
        #        deliberately out of scope.
        #    Hash-exact repeat detection is NOT a candidate: it is narrower
        #    than this check and goes silent on the exact flailing shape that
        #    motivated the round — different arguments every time.
        #
        #    This signal is no longer warn-only: when it holds for
        #    consecutive evaluations the loop HALTS the turn (see
        #    halt_on_repetition above, and agent_loop's divergence block).
        #    The loop also carries its own always-on varied-args trip for
        #    READ-ONLY tools (_ProgressRepeatDetector) — the two overlap on
        #    read-only flail on purpose; belt and braces have different
        #    failure modes.
        if len(recent) >= REPETITION_RUN:
            run = recent[-REPETITION_RUN:]
            if len({c["tool"] for c in run}) == 1:
                results = [str(c.get("result", "")).strip() for c in run]
                unproductive = (
                    all(not r for r in results)      # returning nothing
                    or len(set(results)) == 1        # returning the same thing
                )
                if unproductive:
                    out["repetition"] = REPETITION_FLOOR

        # 4. Context growth anomaly.
        if len(messages) > 20:
            if len(messages) / max(state.step_count, 1) > 5:
                out["context_growth"] = 0.3

        return out

    @staticmethod
    def _aggregate(components: dict[str, float]) -> float:
        """Collapse the sub-scores to one number.

        MEAN, then FLOORED by the repetition signal. The floor is the whole
        change: as a summand, 0.5 among three healthy terms averaged to 0.23
        and a plainly-stuck agent read as "on_track". A signal meant to say
        "this is going nowhere" must be able to say it alone.
        """
        if not components:
            return 0.0
        mean = sum(components.values()) / len(components)
        return max(mean, components.get("repetition", 0.0))

    @staticmethod
    def _is_diverged(components: dict[str, float], score: float,
                     threshold: float) -> bool:
        """A single sub-score may never be the whole verdict.

        Goal alignment is lexical and it is the term that produced 0.96 on a
        compliant turn. It may CONTRIBUTE to a divergence verdict; it may not
        constitute one. At least one term describing what the agent DID has to
        agree.

        This also neutralises the checkpoint-clearing interaction on its own:
        when the tool window was emptied, goal_alignment was the only term
        present, and a lone term can no longer declare anything.
        """
        if score < threshold:
            return False
        return any(components.get(t, 0.0) > 0.0 for t in BEHAVIOURAL_TERMS)

    def _calculate_score(
        self,
        state: _TaskState,
        messages: list[dict],
        tool_results: list[dict],
    ) -> float:
        """Back-compat shim: the aggregate alone."""
        return self._aggregate(self._components(state, messages, tool_results))

    def _build_reason(self, score: float,
                      components: dict[str, float] | None = None,
                      diverged: bool | None = None) -> str:
        """Human-readable reason, naming the terms that produced it.

        The old text was the number and nothing else, so a WARNING line gave a
        reader no way to tell a genuine divergence from lexical dissimilarity
        on a compliant turn — and that is precisely the discrimination the
        first live observation needed.
        """
        detail = ""
        if components:
            detail = " [" + " ".join(
                f"{k}={v:.2f}" for k, v in sorted(components.items())) + "]"
        if diverged is False and score >= self.threshold:
            return (f"high_score_but_no_behavioural_signal "
                    f"(score={score:.2f}); goal alignment is lexical and "
                    f"cannot declare divergence alone{detail}")
        if score < 0.3:
            return f"on_track (score={score:.2f}){detail}"
        elif score < 0.5:
            return f"minor_drift (score={score:.2f}){detail}"
        elif score < self.threshold:
            return f"moderate_drift (score={score:.2f}){detail}"
        return f"diverged (score={score:.2f}){detail}"

    def end_task(self, task_id: str | None = None) -> None:
        """Drop a task's record. Idempotent — safe to call for a task that
        was never started (disabled detector, or an early-exit turn)."""
        with self._lock:
            state = self._tasks.pop(task_id or DEFAULT_TASK_KEY, None)
        if state is not None:
            logger.info(
                "Task ended: %s, steps=%d", state.task_id, state.step_count,
            )
