"""One computer-use step, through the whole path.

    observe -> build candidates -> select -> validate -> GATE -> execute -> verify

Every stage is a real one. The gate is the actual ``SecurityGate``, the schema
it reads is the actual wrapped model's, the staleness check is the actual
snapshot binding. What is substitutable is the DRIVER (fixtures in milestone 1,
Cua later) and the CHOOSER (rules now, Jev later) — the two places where
substituting a fake proves something rather than hiding something.

WHY THE GATE CALL IS BUILT HERE THE SAME WAY ``agent_loop`` BUILDS IT
----------------------------------------------------------------------
Two call sites deriving a gate subject differently is how the four-month
``file_path`` defect survived: every test supplied the argument the caller
never did. So this module derives the extent through the SAME
``computer_extent_for`` the agent loop uses, from the SAME declared schema,
and passes it through the SAME ``evaluate`` kwargs. If the two ever diverge,
the divergence is in one function and one test pins it
(``test_computer_use_gate_path.py``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from prometheus.computer.actions import ACTION_MODELS, schema_for
from prometheus.computer.candidates import (
    InvalidChoice,
    UnusableObservation,
    action_arguments,
    build_candidates,
    build_choice_request,
    validate_choice,
)
from prometheus.computer.driver import Driver, StaleSnapshot, check_preconditions
from prometheus.computer.types import Candidate
from prometheus.permissions.computer_extent import computer_extent_for

log = logging.getLogger(__name__)

#: Verb -> the wrapped tool name the gate and the audit trail see. Derived
#: from ACTION_MODELS so a new action cannot be added without a name.
TOOL_NAME_FOR_VERB: dict[str, str] = {
    verb: f"computer_{verb}" for verb in ACTION_MODELS
}
VERB_FOR_TOOL_NAME: dict[str, str] = {v: k for k, v in TOOL_NAME_FOR_VERB.items()}


@dataclass
class StepResult:
    """What happened, in terms a caller and an audit row can both use."""

    status: str  # "executed" | "refused" | "abstained" | "reobserve" | "blocked"
    reason: str = ""
    candidate: Candidate | None = None
    verified: bool | None = None
    driver_result: dict[str, Any] | None = None
    candidates_offered: int = 0
    extent: str = ""
    history: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "executed"


class ComputerUseLoop:
    """Runs bounded computer-use steps against a driver and a chooser."""

    def __init__(
        self,
        driver: Driver,
        chooser: Any,
        gate: Any,
        *,
        approve: Callable[..., Awaitable[bool]] | None = None,
        origin: str = "system",
        skip_preconditions: bool = False,
    ) -> None:
        self._driver = driver
        self._chooser = chooser
        self._gate = gate
        self._approve = approve
        self._origin = origin
        # Only a FixtureDriver legitimately skips the substrate check — it has
        # no substrate. A real driver that skipped it is the silent-failure
        # shape this whole check exists to refuse, so the flag is explicit
        # rather than inferred from the driver's type.
        self._skip_preconditions = skip_preconditions

    async def step(
        self,
        goal: str,
        app: str,
        pid: int,
        window_id: int,
        *,
        text_to_type: str | None = None,
        history: list[str] | None = None,
    ) -> StepResult:
        if not self._skip_preconditions:
            pre = check_preconditions()
            if not pre:
                # REFUSE, do not degrade. See driver.check_preconditions: the
                # observe and act halves fail independently on this box, and
                # the combination that reports success while doing nothing is
                # exactly the one being refused here.
                return StepResult(status="blocked", reason=pre.reason)

        # 1. OBSERVE ---------------------------------------------------------
        observation = self._driver.observe(app, pid, window_id)

        # 2. BUILD -----------------------------------------------------------
        try:
            candidates = build_candidates(observation, text_to_type=text_to_type)
        except UnusableObservation as exc:
            return StepResult(status="blocked", reason=str(exc))
        if not candidates:
            return StepResult(
                status="abstained",
                reason="no bounded action was available in this window",
            )

        # 3. SELECT ----------------------------------------------------------
        request = build_choice_request(goal, observation, candidates, history)
        choice = self._chooser.choose(request)

        # 4. VALIDATE — fail closed, never coerce --------------------------
        try:
            candidate = validate_choice(
                choice.candidate_id, candidates, observation
            )
        except InvalidChoice as exc:
            log.warning("computer-use: rejecting chooser answer — %s", exc)
            return StepResult(
                status="refused", reason=str(exc),
                candidates_offered=len(candidates),
            )
        if candidate is None:
            status = (
                "reobserve" if choice.candidate_id == "reobserve" else "abstained"
            )
            return StepResult(
                status=status,
                reason=f"chooser returned {choice.candidate_id!r}",
                candidates_offered=len(candidates),
            )

        # 5. GATE ------------------------------------------------------------
        verb = VERB_FOR_TOOL_NAME[candidate.tool_name]
        arguments = action_arguments(candidate)
        schema = schema_for(verb)
        # Parse through the declared model. This is not ceremony: it is what
        # proves the arguments we are about to gate are the arguments the tool
        # accepts, so the gate cannot rule on a shape the driver would reject
        # or, worse, reinterpret.
        ACTION_MODELS[verb](**arguments)

        extent, unknown = computer_extent_for(
            candidate.tool_name, arguments, schema=schema
        )
        decision = self._gate.evaluate(
            candidate.tool_name,
            is_read_only=False,
            file_path=None,
            command=None,
            origin=self._origin,
            computer_action=extent,
            computer_unknown=unknown,
        )
        extent_value = extent.value if extent else ""
        if not decision.allowed:
            if decision.requires_confirmation and self._approve is not None:
                confirmed = await self._call_approve(
                    candidate.tool_name, decision.reason, arguments
                )
                if not confirmed:
                    return StepResult(
                        status="refused",
                        reason=f"operator declined: {decision.reason}",
                        candidate=candidate, extent=extent_value,
                        candidates_offered=len(candidates),
                    )
            else:
                # No approver reachable. REFUSE rather than auto-approve —
                # the unattended case must not be the permissive one.
                return StepResult(
                    status="refused",
                    reason=(
                        decision.reason
                        or f"permission denied for {candidate.tool_name}"
                    ),
                    candidate=candidate, extent=extent_value,
                    candidates_offered=len(candidates),
                )

        # 6. EXECUTE ---------------------------------------------------------
        try:
            result = self._driver.act(verb, arguments)
        except StaleSnapshot as exc:
            # Never retried here. A stale snapshot means the window moved
            # under us; the correct response is to observe again and rebuild
            # the table, which is the caller's next step, not a silent redo.
            return StepResult(
                status="refused", reason=str(exc),
                candidate=candidate, extent=extent_value,
                candidates_offered=len(candidates),
            )

        # 7. VERIFY — against FRESH state, not against the return value ------
        verified = self._verify(candidate, app, pid, window_id)
        return StepResult(
            status="executed",
            candidate=candidate,
            verified=verified,
            driver_result=result,
            candidates_offered=len(candidates),
            extent=extent_value,
            reason=f"{candidate.description} ({extent_value})",
            history=[*(history or []), candidate.description],
        )

    def _verify(
        self, candidate: Candidate, app: str, pid: int, window_id: int
    ) -> bool | None:
        """Did the action land? Asked of a NEW observation.

        ⚠ NOT asked of the driver's return value. "The call returned ok" is
        the success message, and trusting it is the shape this codebase has
        been bitten by repeatedly — a dispatched action and a landed one are
        different facts. Returns None when the question cannot be answered,
        which is honest and distinguishable from False.
        """
        try:
            after = self._driver.observe(app, pid, window_id)
        except Exception:
            log.warning("computer-use: post-action observe failed", exc_info=True)
            return None
        if after.snapshot_id == candidate.snapshot_id:
            # The window did not change at all. For a click that usually means
            # nothing happened — reported as unverified rather than as failure,
            # because some actions legitimately leave the tree identical.
            return None
        return True

    async def _call_approve(
        self, tool_name: str, reason: str, arguments: dict[str, Any]
    ) -> bool:
        """Ask the operator, showing the arguments. Falls back if unsupported.

        Returns False when there is no approver. That is not defensive
        padding: "nobody could be asked" and "somebody said no" must reach the
        same outcome, because the alternative — proceeding when unattended —
        is the one failure mode a consent system cannot have.
        """
        approve = self._approve
        if approve is None:
            return False
        try:
            return await approve(tool_name, reason, arguments=arguments)
        except TypeError:
            # A caller with the older two-arg prompt shape. It shows less; it
            # must not refuse everything (see checker.request_approval's
            # matching branch, where swallowing this denied every approval).
            return await approve(tool_name, reason)
