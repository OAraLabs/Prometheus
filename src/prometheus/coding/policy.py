"""Iterate-to-green policy (SPRINT-coding-mode v2, scope item 4 — THE CORE).

The addendum localized the irreducible gap: with thinking on, the loop
matches openhands on edit tasks but loses T3 1/10 vs 9/10 because it does
not *systematically run tests and re-edit on failure*. This module is that
discipline, as pure logic (no model, no I/O) so it is unit-testable:

- **Done is a verdict, not a claim.** A turn that ends without a qualifying
  exit-0 ``code_run`` in the last ``evidence_window`` rounds is rejected
  back to the model with the policy stated.
- **Ground truth outranks the model's evidence.** The session re-runs the
  task's acceptance command itself before accepting (bakeoff F3: only test
  execution catches confident-but-wrong). A model-passed/-ground-truth-
  failed turn is rejected with the real failure output.
- **Repeated identical failures trigger a step-back.** Two consecutive
  ground-truth failures with the same normalized signature → the rejection
  tells the model to STOP editing and re-view the failing code first.

All prompt text in this module is NATIVE — written for this policy from the
bakeoff evidence, not adapted from OpenHands (the provenance exception was
therefore never needed).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CodeRunObservation:
    """One ``code_run`` the model executed, as seen in the event stream."""

    round_index: int
    command: str
    exit_code: int | None  # None = timed out / unparseable
    output_tail: str


_EXIT_RE = re.compile(r"^exit code: (\d+)", re.MULTILINE)


def parse_exit_code(code_run_output: str) -> int | None:
    """Extract the exit code from code_run's deterministic output header."""
    m = _EXIT_RE.search(code_run_output)
    return int(m.group(1)) if m else None


def failure_signature(output: str, tail_chars: int = 2_000) -> str:
    """Stable signature of a failure output — durations/addresses normalized."""
    tail = output[-tail_chars:]
    tail = re.sub(r"\d+\.\d+s", "Ts", tail)          # pytest timings
    tail = re.sub(r"0x[0-9a-fA-F]+", "ADDR", tail)   # object addresses
    tail = re.sub(r"in \d+\.\d+", "in T", tail)
    tail = re.sub(r"\s+", " ", tail).strip()
    return hashlib.sha1(tail.encode("utf-8", "replace")).hexdigest()


@dataclass
class IterateToGreenPolicy:
    """Verdict + feedback machine for one coding run."""

    acceptance_command: str
    max_rounds: int = 30
    max_wall_seconds: float = 1_200.0
    evidence_window: int = 2  # acceptance evidence must be this recent (rounds)

    rounds_used: int = 0
    observations: list[CodeRunObservation] = field(default_factory=list)
    _last_failure_sig: str | None = None
    repeat_failures: int = 0

    # ------------------------------------------------------------------
    # Stream bookkeeping
    # ------------------------------------------------------------------

    def observe_round(self) -> None:
        self.rounds_used += 1

    def observe_code_run(self, command: str, output: str) -> None:
        self.observations.append(
            CodeRunObservation(
                round_index=self.rounds_used,
                command=command,
                exit_code=parse_exit_code(output),
                output_tail=output[-2_000:],
            )
        )

    # ------------------------------------------------------------------
    # The verdict
    # ------------------------------------------------------------------

    def _qualifies_as_evidence(self, obs: CodeRunObservation) -> bool:
        """A green run of the acceptance command — or a repo test invocation.

        The spec allows "the task's acceptance command (or repo test
        invocation)" as the model's evidence; ground truth (the session
        re-running the real acceptance command) is what actually accepts.
        """
        if obs.exit_code != 0:
            return False
        cmd = " ".join(obs.command.split())
        accept = " ".join(self.acceptance_command.split())
        return accept in cmd or "pytest" in cmd

    def has_recent_green_evidence(self) -> bool:
        floor = self.rounds_used - self.evidence_window
        return any(
            o.round_index >= floor and self._qualifies_as_evidence(o)
            for o in self.observations
        )

    def over_cap(self, wall_seconds: float) -> str | None:
        if self.rounds_used >= self.max_rounds:
            return f"round cap reached ({self.rounds_used}/{self.max_rounds})"
        if wall_seconds >= self.max_wall_seconds:
            return (
                f"wall-clock cap reached "
                f"({wall_seconds:.0f}s/{self.max_wall_seconds:.0f}s)"
            )
        return None

    # ------------------------------------------------------------------
    # Ground-truth failure tracking (fed by the session's own acceptance run)
    # ------------------------------------------------------------------

    def record_ground_truth_failure(self, output: str) -> None:
        sig = failure_signature(output)
        if sig == self._last_failure_sig:
            self.repeat_failures += 1
        else:
            self.repeat_failures = 0
        self._last_failure_sig = sig

    def record_ground_truth_success(self) -> None:
        self._last_failure_sig = None
        self.repeat_failures = 0

    # ------------------------------------------------------------------
    # Native policy prompt text
    # ------------------------------------------------------------------

    def no_evidence_rejection(self) -> str:
        return (
            "[CODING POLICY] Your turn ended without proof the task is done. "
            "A coding task is finished only when its acceptance command has "
            "been executed via code_run, in this round or the previous one, "
            "and exited 0. Claims without a passing run are rejected — run "
            f"the acceptance command now:\n\n    {self.acceptance_command}\n\n"
            "If it fails, read the failure output, fix the code, and run it "
            "again. Do not declare success without it."
        )

    def ground_truth_rejection(self, acceptance_output: str) -> str:
        msg = (
            "[CODING POLICY] Independent verification ran the task's "
            "acceptance command and it FAILED — whatever you ran does not "
            "cover the task. The real failure output is below. Fix the code, "
            "then re-run the acceptance command via code_run until it exits 0."
            f"\n\nAcceptance command:\n    {self.acceptance_command}\n\n"
            f"Failure output (tail):\n{acceptance_output[-3_000:]}"
        )
        if self.repeat_failures == 1:
            msg += (
                "\n\n[STEP BACK] This is the SAME failure as your previous "
                "attempt. Stop editing. Re-view the failing code and the "
                "test that exercises it (code_view) before changing anything "
                "else — your current theory of the bug is wrong."
            )
        elif self.repeat_failures >= 2:
            msg += (
                "\n\n[STEP BACK — FINAL] The identical failure has now "
                "occurred three times. Discard your approach entirely: "
                "re-read the task, re-view every file involved, and form a "
                "new hypothesis before your next edit. Repeating the same "
                "edit again will exhaust the round budget."
            )
        return msg
