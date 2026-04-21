"""SkillRefiner — compare actual tool traces to skill steps and refine.

After a task uses a skill, compare what actually happened to what the
skill prescribed. If the deviation led to a better outcome, update the skill.

Usage:
    refiner = SkillRefiner(provider)
    updated = await refiner.maybe_refine(skill_path, tool_trace, outcome)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from prometheus.providers.base import ModelProvider

log = logging.getLogger(__name__)

_REFINEMENT_PROMPT = """\
You are a skill refinement engine. A skill was used to guide a task, but the
actual execution deviated from the prescribed steps. Analyze whether the
deviation improved the outcome and, if so, update the skill.

Current skill content:
```
{skill_content}
```

Actual tool trace (what happened):
{trace}

Outcome: {outcome}

Rules:
- If the deviation was beneficial, update the skill steps to match.
- If the deviation was neutral or harmful, keep the original steps.
- Preserve the YAML frontmatter (name, description).
- Keep the same markdown structure.
- Output the FULL updated SKILL.md content, or output "NO_CHANGE" if no update needed.
"""


class SkillRefiner:
    """Refine skills based on actual execution traces.

    Args:
        provider: ModelProvider for refinement analysis.
        model: Model name for the refinement call.
    """

    def __init__(
        self,
        provider: ModelProvider,
        *,
        model: str = "default",
    ) -> None:
        self._provider = provider
        self._model = model

    async def maybe_refine(
        self,
        skill_path: Path,
        tool_trace: list[dict[str, Any]],
        outcome: str,
    ) -> bool:
        """Refine a skill if the execution deviated beneficially.

        Args:
            skill_path: Path to the SKILL.md file.
            tool_trace: Actual tool calls executed.
            outcome: Description of the task outcome (success/failure + details).

        Returns:
            True if the skill was updated, False otherwise.
        """
        if not skill_path.exists():
            log.warning("SkillRefiner: skill not found at %s", skill_path)
            return False

        skill_content = skill_path.read_text(encoding="utf-8")
        trace_text = self._format_trace(tool_trace)

        prompt = _REFINEMENT_PROMPT.format(
            skill_content=skill_content,
            trace=trace_text,
            outcome=outcome,
        )

        try:
            response = await self._call_model(prompt)
        except Exception:
            log.exception("SkillRefiner: model call failed")
            return False

        response = response.strip()
        if not response or response == "NO_CHANGE":
            log.debug("SkillRefiner: no changes needed for %s", skill_path.name)
            return False

        # Validate the response looks like a skill file
        if not response.startswith("---"):
            log.warning("SkillRefiner: response doesn't look like SKILL.md, skipping")
            return False

        # Back up the original
        backup = skill_path.with_suffix(f".bak-{int(time.time())}.md")
        backup.write_text(skill_content, encoding="utf-8")

        # Write the refined version
        skill_path.write_text(response + "\n", encoding="utf-8")
        log.info("SkillRefiner: updated %s (backup at %s)", skill_path.name, backup.name)
        return True

    async def _call_model(self, prompt: str) -> str:
        """Call the ModelProvider and return the full text response."""
        from prometheus.engine.messages import ConversationMessage
        from prometheus.providers.base import ApiMessageRequest, ApiTextDeltaEvent

        request = ApiMessageRequest(
            model=self._model,
            messages=[ConversationMessage(role="user", content=prompt)],
            max_tokens=2048,
        )
        text_parts: list[str] = []
        async for event in self._provider.stream_message(request):
            if isinstance(event, ApiTextDeltaEvent):
                text_parts.append(event.text)
        return "".join(text_parts)

    @staticmethod
    def _format_trace(trace: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for i, call in enumerate(trace, 1):
            tool = call.get("tool_name", "unknown")
            args = call.get("arguments", {})
            result = str(call.get("result", ""))[:200]
            lines.append(f"{i}. {tool}({args}) → {result}")
        return "\n".join(lines)
