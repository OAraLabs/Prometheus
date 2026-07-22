"""Step verifier — second-pass model review of extracted action sequences.

Uses a text-only model (no vision) to review extracted steps for missing
steps, duplicates, illogical ordering, parameter consistency, and action
type misclassification. Advisory: the deterministic quality gate decides
pass/fail; the verifier's findings are recorded alongside the skill so a
human (or the curator) can judge them later.

Ported from skillforge-engine core/refinement/step_verifier.py; the
OpenAI/Anthropic client plumbing was replaced with Prometheus's provider
layer via LLMCallEnvelope, so the verifying model is whatever the daemon
is configured with — no separate config surface.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from prometheus.learning.llm_envelope import LLMCallEnvelope

if TYPE_CHECKING:
    from prometheus.providers.base import ModelProvider

log = logging.getLogger(__name__)


VERIFICATION_PROMPT = """You are reviewing an automatically extracted workflow from a browser recording. Your job is to quality-check the action sequence for errors.

APPLICATION: {app_name}
WORKFLOW: {workflow_name}
DESCRIPTION: {workflow_description}

EXTRACTED STEPS:
{steps_json}

PARAMETERS DETECTED:
{params_json}

Review these steps and respond with a JSON object (no markdown fencing):

{{
  "overall_quality": "good|needs_fixes|poor",
  "confidence": 0.85,
  "issues": [
    {{
      "type": "missing_step|duplicate_step|wrong_order|wrong_action_type|parameter_error|logic_error",
      "severity": "critical|warning|info",
      "step_number": 3,
      "description": "What's wrong and why",
      "suggestion": "How to fix it"
    }}
  ],
  "missing_steps": [
    {{
      "should_be_after_step": 2,
      "description": "What step is missing",
      "action_type": "click|type|navigate|etc",
      "reasoning": "Why this step should exist"
    }}
  ],
  "summary": "Brief overall assessment"
}}

Check for:
1. MISSING STEPS: Are there logical gaps? (e.g., form filled but never submitted, page navigated but never loaded)
2. DUPLICATES: Any steps that do the same thing?
3. ORDERING: Does the sequence make logical sense for this application?
4. PARAMETERS: Are detected parameters actually used in steps? Are there values that should be parameters but aren't?
5. ACTION TYPES: Are clicks labeled as types, or navigations labeled as clicks?

Be specific and constructive. Only flag real issues, not style preferences."""


@dataclass
class VerificationResult:
    """Result of step verification."""

    overall_quality: str = "unknown"  # good, needs_fixes, poor
    confidence: float = 0.0
    issues: list[dict] = field(default_factory=list)
    missing_steps: list[dict] = field(default_factory=list)
    summary: str = ""
    critical_count: int = 0
    warning_count: int = 0
    verified_by: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    if raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    return raw.strip()


class StepVerifier:
    """Reviews extracted action sequences for quality issues.

    Args:
        provider: ModelProvider for the review call.
        model: Model name for the review call.
        telemetry: Optional telemetry store; failures land in
            ``telemetry.silent_failures`` via the shared envelope.
    """

    def __init__(
        self,
        provider: ModelProvider,
        *,
        model: str = "default",
        telemetry: object | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._envelope = LLMCallEnvelope(
            subsystem="live_recorder",
            telemetry=telemetry,
            on_failure="return_none",
        )

    async def verify(
        self,
        actions: list[dict[str, Any]],
        parameters: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> VerificationResult:
        """Run verification on live recorder actions. Never raises.

        Returns a :class:`VerificationResult`; on model failure the result
        carries ``overall_quality="unknown"`` with an explanatory summary
        (and the failure is visible in telemetry).
        """
        steps_for_prompt = []
        for i, a in enumerate(actions, 1):
            value = a.get("value", "")
            if a.get("is_parameter") and a.get("parameter_name"):
                value = f"{{{a['parameter_name']}}}"
            steps_for_prompt.append({
                "step": i,
                "action": a.get("action_type", ""),
                "description": a.get("description", ""),
                "target": a.get("target", ""),
                "value": value,
                "url": a.get("url", ""),
            })

        prompt = VERIFICATION_PROMPT.format(
            app_name=metadata.get("app", "") or metadata.get("start_url", "unknown"),
            workflow_name=metadata.get("title", "recorded workflow"),
            workflow_description=metadata.get("description", ""),
            steps_json=json.dumps(steps_for_prompt, indent=2),
            params_json=json.dumps(parameters, indent=2),
        )

        raw = await self._envelope.call(
            provider=self._provider,
            model=self._model,
            prompt=prompt,
            max_tokens=4096,
            operation="verify_steps",
        )
        if raw is None or not raw.strip():
            return VerificationResult(summary="Verifier unavailable (model call failed)")

        try:
            data = json.loads(_strip_fences(raw))
        except json.JSONDecodeError as exc:
            log.warning("StepVerifier: unparseable model response: %s", exc)
            return VerificationResult(summary=f"Verification response unparseable: {exc}")

        issues = data.get("issues", []) or []
        return VerificationResult(
            overall_quality=data.get("overall_quality", "unknown"),
            confidence=data.get("confidence", 0.0),
            issues=issues,
            missing_steps=data.get("missing_steps", []) or [],
            summary=data.get("summary", ""),
            critical_count=sum(1 for i in issues if i.get("severity") == "critical"),
            warning_count=sum(1 for i in issues if i.get("severity") == "warning"),
            verified_by=self._model,
        )
