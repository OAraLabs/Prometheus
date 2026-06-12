"""Coding session — drives the agent loop in episodes under the policy.

One :class:`CodingSession` = one task against one sandbox. The existing
``run_loop`` is reused unmodified (Phase 0 finding: knobs, not surgery):
each *episode* is one ``run_loop`` invocation that ends when the model
stops requesting tools — its "I believe I'm done" signal. The session then
applies :class:`~prometheus.coding.policy.IterateToGreenPolicy`:

1. model evidence missing → inject the no-evidence rejection, next episode;
2. model evidence present → the session runs the ACCEPTANCE COMMAND ITSELF
   (ground truth; bakeoff F3) — exit 0 accepts, anything else injects the
   real failure output and continues;
3. caps (rounds / wall) → honest abandonment, never a success claim.

Terminal artifact either way: a branch in the sandbox clone with the work
committed (repo-local identity), a diff stat, and the final acceptance
output. Never merges, never pushes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from prometheus.coding.policy import IterateToGreenPolicy
from prometheus.coding.sandbox import Sandbox
from prometheus.coding.tools import build_coding_registry
from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage
from prometheus.engine.stream_events import (
    AssistantTurnComplete,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)

log = logging.getLogger(__name__)

_GIT_IDENTITY = ("-c", "user.name=prometheus-coding",
                 "-c", "user.email=coding@prometheus.local")


def coding_system_prompt(task_description: str, acceptance_command: str) -> str:
    """The native coding contract. Appended to nothing — coding runs are
    fresh sessions; the daemon's chat system prompt does not apply here."""
    return (
        "You are Prometheus running a CODING TASK inside a sandboxed clone "
        "of the target repository. Work only through your tools.\n"
        "\n"
        f"TASK:\n{task_description}\n"
        "\n"
        f"ACCEPTANCE COMMAND (the task is done ONLY when this exits 0):\n"
        f"    {acceptance_command}\n"
        "\n"
        "RULES:\n"
        "1. View before you edit: code_str_replace requires the exact "
        "current text — use code_view first, always.\n"
        "2. Run tests with code_run. A failing run is information: read the "
        "failure output and act on it. Never re-run an identical command "
        "expecting a different result without editing in between.\n"
        "3. DONE IS A VERDICT, NOT A CLAIM: before you finish, execute the "
        "acceptance command via code_run and confirm exit 0. Turns that "
        "claim success without a passing acceptance run are rejected.\n"
        "4. Keep edits minimal and within the repo's existing style. Do not "
        "fix unrelated issues.\n"
    )


@dataclass(frozen=True)
class CodingTask:
    task_id: str
    description: str
    acceptance_command: str


@dataclass
class CodingRunReport:
    """Honest terminal report — the managed task's output payload."""

    task_id: str
    status: str  # "success" | "failed_abandoned" — never success without green
    reason: str
    rounds_used: int
    episodes: int
    wall_seconds: float
    acceptance_exit: int | None
    acceptance_output_tail: str
    branch: str
    diff_stat: str
    model_usage_note: str = (
        "per-round tokens + thinking flag are in telemetry.subsystem_runs "
        "(subsystem='agent_loop', session_id=coding:<task_id>)"
    )

    def to_summary(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "reason": self.reason,
            "rounds": self.rounds_used,
            "episodes": self.episodes,
            "wall_seconds": round(self.wall_seconds, 1),
            "acceptance_exit": self.acceptance_exit,
            "branch": self.branch,
        }


class CodingSession:
    """One coding run. See module docstring."""

    def __init__(
        self,
        *,
        provider: Any,
        model: str,
        sandbox: Sandbox,
        task: CodingTask,
        adapter: Any | None = None,
        telemetry: Any | None = None,
        max_tokens: int = 4096,
        max_rounds: int = 30,
        max_wall_seconds: float = 1_200.0,
        suppress_thinking: bool | None = False,  # coding turns THINK by default
        acceptance_timeout_seconds: float = 240.0,
    ) -> None:
        self._provider = provider
        self._model = model
        self._sandbox = sandbox
        self._task = task
        self._adapter = adapter
        self._telemetry = telemetry
        self._max_tokens = max_tokens
        self._suppress_thinking = suppress_thinking
        self._acceptance_timeout = acceptance_timeout_seconds
        self._policy = IterateToGreenPolicy(
            acceptance_command=task.acceptance_command,
            max_rounds=max_rounds,
            max_wall_seconds=max_wall_seconds,
        )
        self._registry = build_coding_registry(sandbox)
        self._branch = f"coding/{task.task_id}"

    # ------------------------------------------------------------------
    # Git plumbing (inside the sandbox, repo-local identity, never pushes)
    # ------------------------------------------------------------------

    async def _git(self, *args: str) -> str:
        cmd = "git " + " ".join(args)
        result = await self._sandbox.run(cmd, timeout_seconds=60)
        return result.output

    async def _prepare_branch(self) -> None:
        await self._git("checkout", "-q", "-b", self._branch)

    async def _commit_artifact(self, status: str) -> str:
        await self._git("add", "-A")
        await self._git(
            *_GIT_IDENTITY,
            "commit", "-q", "--allow-empty",
            "-m", f"'coding task {self._task.task_id}: {status}'",
        )
        return await self._git("diff", "--stat", "HEAD~1..HEAD")

    # ------------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------------

    async def _run_acceptance(self) -> tuple[int | None, str]:
        result = await self._sandbox.run(
            self._task.acceptance_command,
            timeout_seconds=self._acceptance_timeout,
        )
        exit_code = None if result.timed_out else result.exit_code
        return exit_code, result.output

    # ------------------------------------------------------------------
    # The run
    # ------------------------------------------------------------------

    async def run(self) -> CodingRunReport:
        started = time.monotonic()
        await self._prepare_branch()

        context = LoopContext(
            provider=self._provider,
            model=self._model,
            system_prompt=coding_system_prompt(
                self._task.description, self._task.acceptance_command
            ),
            max_tokens=self._max_tokens,
            tool_registry=self._registry,
            adapter=self._adapter,
            telemetry=self._telemetry,
            cwd=self._sandbox.root,
            session_id=f"coding:{self._task.task_id}",
            suppress_thinking=self._suppress_thinking,
            max_turns=self._policy.max_rounds,         # re-tightened per episode
            max_tool_iterations=self._policy.max_rounds * 4,
        )
        messages = [ConversationMessage.from_user_text(
            f"Begin the coding task. Repository root: {self._sandbox.root}. "
            "Start by exploring the relevant files."
        )]

        episodes = 0
        # Track in-flight code_run calls so completions can be paired with
        # the command that produced them.
        pending_runs: dict[str, str] = {}

        while True:
            episodes += 1
            # An episode may use at most the remaining round budget.
            context.max_turns = max(
                1, self._policy.max_rounds - self._policy.rounds_used
            )

            rounds_before = self._policy.rounds_used
            try:
                async for event, _usage in run_loop(context, messages):
                    if isinstance(event, AssistantTurnComplete):
                        self._policy.observe_round()
                    elif isinstance(event, ToolExecutionStarted):
                        if event.tool_name == "code_run":
                            pending_runs[event.tool_use_id] = str(
                                (event.tool_input or {}).get("command", "")
                            )
                    elif isinstance(event, ToolExecutionCompleted):
                        if event.tool_name == "code_run":
                            command = pending_runs.pop(event.tool_use_id, "")
                            self._policy.observe_code_run(command, event.output)
            except RuntimeError as exc:
                # run_loop raises this when an episode consumes its whole
                # per-episode turn allowance without the model stopping — the
                # model kept tool-calling past the budget. Treat it as a hard
                # episode end (rounds are already counted from the yielded
                # AssistantTurnComplete events) and fall through to the cap
                # check, which abandons honestly or accepts if green. Any
                # OTHER RuntimeError is a real fault and must propagate.
                if "maximum turn limit" not in str(exc):
                    raise
                log.info(
                    "coding task %s: episode %d hit its turn allowance — "
                    "evaluating", self._task.task_id, episodes,
                )

            wall = time.monotonic() - started

            # Cap check — honest abandonment, with ground truth attached.
            cap_reason = self._policy.over_cap(wall)
            stalled = self._policy.rounds_used == rounds_before
            if cap_reason or stalled:
                reason = cap_reason or (
                    "episode made no model progress (0 rounds) — aborting "
                    "to avoid an injection loop"
                )
                exit_code, output = await self._run_acceptance()
                if exit_code == 0:
                    # The work is actually green even though we hit a cap —
                    # report success honestly (the evidence is the run).
                    return await self._finish(
                        "success", f"green at cap ({reason})",
                        episodes, wall, exit_code, output,
                    )
                return await self._finish(
                    "failed_abandoned", reason, episodes, wall, exit_code, output,
                )

            # Done-is-a-verdict, layer 1: the model's own evidence.
            if not self._policy.has_recent_green_evidence():
                messages.append(ConversationMessage.from_injected(
                    self._policy.no_evidence_rejection(),
                    provenance="orchestrator",
                    is_trusted=True,
                ))
                continue

            # Layer 2: ground truth — the session runs acceptance itself.
            exit_code, output = await self._run_acceptance()
            if exit_code == 0:
                self._policy.record_ground_truth_success()
                wall = time.monotonic() - started
                return await self._finish(
                    "success", "acceptance command exited 0",
                    episodes, wall, exit_code, output,
                )
            self._policy.record_ground_truth_failure(output)
            messages.append(ConversationMessage.from_injected(
                self._policy.ground_truth_rejection(output),
                provenance="orchestrator",
                is_trusted=True,
            ))

    async def _finish(
        self,
        status: str,
        reason: str,
        episodes: int,
        wall: float,
        acceptance_exit: int | None,
        acceptance_output: str,
    ) -> CodingRunReport:
        diff_stat = await self._commit_artifact(status)
        report = CodingRunReport(
            task_id=self._task.task_id,
            status=status,
            reason=reason,
            rounds_used=self._policy.rounds_used,
            episodes=episodes,
            wall_seconds=wall,
            acceptance_exit=acceptance_exit,
            acceptance_output_tail=acceptance_output[-3_000:],
            branch=self._branch,
            diff_stat=diff_stat.strip(),
        )
        if self._telemetry is not None and hasattr(self._telemetry, "record_run"):
            try:
                self._telemetry.record_run(
                    subsystem="coding_mode",
                    operation="run",
                    outcome="success" if status == "success" else "failed",
                    duration_ms=wall * 1000.0,
                    summary=report.to_summary(),
                    session_id=f"coding:{self._task.task_id}",
                    model=self._model,
                )
            except Exception:
                log.exception("coding_mode: terminal telemetry write failed")
        log.info(
            "coding task %s: %s (%s) — %d rounds, %d episodes, %.0fs",
            self._task.task_id, status, reason,
            self._policy.rounds_used, episodes, wall,
        )
        return report
