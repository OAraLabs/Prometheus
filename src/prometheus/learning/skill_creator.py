"""SkillCreator — auto-generate SKILL.md files from successful tool-call traces.

PostTaskHook: after a task completes with >3 tool calls, analyse the trace
and produce a reusable skill file under ~/.prometheus/skills/auto/.

Usage:
    creator = SkillCreator(provider)
    skill_path = await creator.maybe_create(task_record, tool_trace)
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prometheus.config.paths import get_config_dir

if TYPE_CHECKING:
    from prometheus.providers.base import ModelProvider

log = logging.getLogger(__name__)

_MIN_TOOL_CALLS = 3
_AUTO_SKILLS_DIR_NAME = "skills/auto"

_GENERATION_PROMPT = """\
You are a skill generator. Given a sequence of tool calls that accomplished a task,
produce a SKILL.md file that codifies the approach for reuse.

Format:
---
name: <short-kebab-case-name>
description: <one-line description of what the skill does>
---

# <Skill Name>

## When to use
<one sentence>

## Steps
1. <step>
2. <step>
...

## Notes
- <any caveats or variations>

Task description: {task_description}

Tool call trace:
{trace}

Output ONLY the SKILL.md content. No commentary.
"""


def _get_auto_skills_dir() -> Path:
    path = get_config_dir() / _AUTO_SKILLS_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _slugify(text: str) -> str:
    """Convert text to a kebab-case filename slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower().strip())
    return slug.strip("-")[:60]


class SkillCreator:
    """Generate SKILL.md files from successful task tool-call traces.

    Args:
        provider: ModelProvider for generating skill content.
        model: Model name for the generation call.
        min_tool_calls: Minimum tool calls to trigger skill creation.
        auto_dir: Override the auto-skills output directory.
    """

    def __init__(
        self,
        provider: ModelProvider,
        *,
        model: str = "default",
        min_tool_calls: int = _MIN_TOOL_CALLS,
        auto_dir: Path | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._min_tool_calls = min_tool_calls
        self._auto_dir = auto_dir or _get_auto_skills_dir()

    @classmethod
    def from_config(
        cls,
        provider: ModelProvider,
        config_path: str | None = None,
    ) -> SkillCreator:
        """Build from prometheus.yaml learning section."""
        import yaml

        if config_path is None:
            from prometheus.config.defaults import DEFAULTS_PATH
            config_path = str(DEFAULTS_PATH)

        try:
            with open(Path(config_path).expanduser()) as fh:
                data = yaml.safe_load(fh)
            learning = data.get("learning", {})
            min_calls = learning.get("skill_min_tool_calls", _MIN_TOOL_CALLS)
        except (OSError, Exception):
            min_calls = _MIN_TOOL_CALLS

        return cls(provider, min_tool_calls=min_calls)

    async def maybe_create(
        self,
        task_description: str,
        tool_trace: list[dict[str, Any]],
    ) -> Path | None:
        """Create a skill if the trace meets the threshold.

        Args:
            task_description: What the task accomplished.
            tool_trace: List of dicts with keys: tool_name, arguments, result.

        Returns:
            Path to the created SKILL.md, or None if skipped.
        """
        if len(tool_trace) < self._min_tool_calls:
            log.debug(
                "SkillCreator: only %d tool calls (need %d), skipping",
                len(tool_trace),
                self._min_tool_calls,
            )
            return None

        trace_text = self._format_trace(tool_trace)
        prompt = _GENERATION_PROMPT.format(
            task_description=task_description,
            trace=trace_text,
        )

        try:
            content = await self._call_model(prompt)
        except Exception:
            log.exception("SkillCreator: model call failed")
            return None

        if not content.strip():
            return None

        slug = _slugify(task_description) or f"skill-{int(time.time())}"
        path = self._auto_dir / f"{slug}.md"

        # Don't overwrite existing skills
        if path.exists():
            path = self._auto_dir / f"{slug}-{int(time.time())}.md"

        path.write_text(content.strip() + "\n", encoding="utf-8")
        log.info("SkillCreator: created skill at %s", path)
        return path

    async def _call_model(self, prompt: str) -> str:
        """Call the ModelProvider and return the full text response."""
        from prometheus.engine.messages import ConversationMessage
        from prometheus.providers.base import ApiMessageRequest, ApiTextDeltaEvent

        request = ApiMessageRequest(
            model=self._model,
            messages=[ConversationMessage(role="user", content=prompt)],
            max_tokens=1024,
        )
        text_parts: list[str] = []
        async for event in self._provider.stream_message(request):
            if isinstance(event, ApiTextDeltaEvent):
                text_parts.append(event.text)
        return "".join(text_parts)

    @staticmethod
    def _format_trace(trace: list[dict[str, Any]]) -> str:
        """Format a tool trace into readable text."""
        lines: list[str] = []
        for i, call in enumerate(trace, 1):
            tool = call.get("tool_name", "unknown")
            args = call.get("arguments", {})
            result = str(call.get("result", ""))[:200]
            lines.append(f"{i}. {tool}({args}) → {result}")
        return "\n".join(lines)
