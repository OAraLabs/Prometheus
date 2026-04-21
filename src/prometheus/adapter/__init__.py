"""adapter package — Model Adapter Layer (Sprint 3).

ModelAdapter bundles the validator, formatter, retry engine, and enforcer.
Pass an instance to AgentLoop to activate model-specific formatting,
tool-call validation, auto-repair, and structured output extraction.

Usage:
    from prometheus.adapter import ModelAdapter
    from prometheus.adapter.formatter import QwenFormatter

    adapter = ModelAdapter(formatter=QwenFormatter(), strictness="MEDIUM")
    loop = AgentLoop(provider=provider, tool_registry=registry, adapter=adapter)
"""

from __future__ import annotations

from typing import Any

from prometheus.adapter.enforcer import StructuredOutputEnforcer
from prometheus.adapter.formatter import (
    AnthropicFormatter,
    GemmaFormatter,
    ModelPromptFormatter,
    QwenFormatter,
)
from prometheus.adapter.retry import RetryAction, RetryEngine
from prometheus.adapter.router import ModelRouter, TaskClassifier, TaskType, ProviderConfig
from prometheus.adapter.validator import RepairResult, Strictness, ToolCallValidator, ValidationResult

__all__ = [
    "ModelAdapter",
    "ModelRouter",
    "TaskClassifier",
    "TaskType",
    "ProviderConfig",
    "AnthropicFormatter",
    "GemmaFormatter",
    "ModelPromptFormatter",
    "QwenFormatter",
    "RepairResult",
    "RetryAction",
    "RetryEngine",
    "Strictness",
    "StructuredOutputEnforcer",
    "ToolCallValidator",
    "ValidationResult",
]


class ModelAdapter:
    """High-level adapter that wires together all Sprint 3 components.

    Three adapter tiers:
      - "off"   — API enforces structure (Anthropic, OpenAI). Skip everything.
      - "light" — Model has native tool calling but server doesn't guarantee
                  structure (Gemma 4/Qwen on llama.cpp). GBNF on, validator
                  at NONE, enforcer ON (model may emit tool calls as text),
                  max_retries=1.
      - "full"  — Model lacks tool calling training. Full adapter pipeline.

    Args:
        formatter:   Model-specific prompt formatter.
        strictness:  Validation strictness: "NONE" | "MEDIUM" | "STRICT".
        max_retries: Max tool-call retries before giving up.
        tier:        Adapter tier: "off", "light", or "full". Overrides
                     strictness/max_retries if set.
    """

    TIER_OFF = "off"
    TIER_LIGHT = "light"
    TIER_FULL = "full"

    def __init__(
        self,
        formatter: ModelPromptFormatter | None = None,
        strictness: str | Strictness = Strictness.NONE,
        max_retries: int = 3,
        adaptive_strictness: bool = False,
        strictness_threshold: float = 0.8,
        strictness_window: int = 100,
        tier: str | None = None,
    ) -> None:
        # If tier is explicitly set, override strictness and max_retries
        if tier == self.TIER_OFF:
            strictness = Strictness.NONE
            max_retries = 0
            adaptive_strictness = False
        elif tier == self.TIER_LIGHT:
            strictness = Strictness.NONE
            max_retries = 1
            adaptive_strictness = True

        self.tier = tier or self.TIER_FULL
        self.formatter = formatter or AnthropicFormatter()
        self.validator = ToolCallValidator(strictness=strictness)
        self.retry = RetryEngine(max_retries=max_retries)
        self.enforcer = StructuredOutputEnforcer()
        self._base_strictness = Strictness(strictness) if isinstance(strictness, str) else strictness
        self._adaptive_strictness = adaptive_strictness
        self._strictness_threshold = strictness_threshold
        self._strictness_window = strictness_window
        self._tool_strictness: dict[str, Strictness] = {}  # per-tool overrides
        self._tool_call_history: dict[str, list[bool]] = {}  # per-tool success history

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_request(
        self,
        system_prompt: str,
        tools: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Format system prompt and tools for the current model.

        Returns (formatted_system_prompt, formatted_tools).
        """
        # Tier off/light: server handles tool format natively (--jinja / peg-gemma4).
        # Only rewrite tool schemas for tier full where they go in the prompt.
        if self.tier == self.TIER_FULL:
            formatted_tools = self.formatter.format_tools(tools)
        else:
            formatted_tools = tools
        formatted_system = self.formatter.format_system_prompt(
            system_prompt, tools  # pass original tools for description extraction
        )
        return formatted_system, formatted_tools

    # ------------------------------------------------------------------
    # Validation + repair
    # ------------------------------------------------------------------

    def validate_and_repair(
        self,
        tool_name: str,
        tool_input: Any,
        tool_registry: Any,
    ) -> tuple[str, dict[str, Any], list[str]]:
        """Validate a tool call and auto-repair if needed.

        Returns (final_tool_name, final_tool_input, repairs_made).
        Raises ValueError if validation fails and repair also fails.
        """
        if self.tier == self.TIER_OFF:
            return tool_name, tool_input, []

        # Use per-tool strictness if adaptive mode is on
        if self._adaptive_strictness and tool_name:
            effective = self.get_effective_strictness(tool_name)
            if effective != self._base_strictness:
                orig_strictness = self.validator.strictness
                self.validator.strictness = effective
                try:
                    return self._do_validate_and_repair(tool_name, tool_input, tool_registry)
                finally:
                    self.validator.strictness = orig_strictness
        return self._do_validate_and_repair(tool_name, tool_input, tool_registry)

    def _do_validate_and_repair(
        self,
        tool_name: str,
        tool_input: Any,
        tool_registry: Any,
    ) -> tuple[str, dict[str, Any], list[str]]:
        """Internal validate + repair logic."""
        result = self.validator.validate(tool_name, tool_input, tool_registry)
        if result.valid:
            self.record_tool_call(tool_name, success=True)
            return tool_name, tool_input, []

        repair = self.validator.repair(tool_name, tool_input, result.error, tool_registry)
        if repair.repaired:
            self.record_tool_call(tool_name, success=True)
            return repair.tool_name, repair.tool_input, repair.repairs_made

        self.record_tool_call(tool_name, success=False)
        raise ValueError(
            f"Tool call validation failed and could not be repaired: {result.error}"
        )

    # ------------------------------------------------------------------
    # Grammar generation (GBNF for llama.cpp constrained decoding)
    # ------------------------------------------------------------------

    def generate_grammar(self, tool_registry: Any) -> str | None:
        """Generate GBNF grammar from the current tool registry schemas."""
        if self.tier == self.TIER_OFF:
            return None  # API enforces structure, no grammar needed
        if tool_registry is None:
            return None
        schemas = tool_registry.to_api_schema()
        if not schemas:
            return None
        return self.enforcer.generate_grammar(schemas)

    # ------------------------------------------------------------------
    # Text-based tool call extraction
    # ------------------------------------------------------------------

    def extract_tool_calls(
        self,
        text: str,
        tool_registry: Any = None,
    ):
        """Extract tool calls from raw model text (for models that embed JSON in prose)."""
        # Tier off: API guarantees structured output, no text extraction needed
        if self.tier == self.TIER_OFF:
            return []
        # Tier light + full: model may emit tool calls as <tool_call> XML or
        # JSON in response text — the enforcer extracts them
        return self.enforcer.extract_tool_calls(text, tool_registry)

    # ------------------------------------------------------------------
    # Retry
    # ------------------------------------------------------------------

    def handle_retry(
        self,
        tool_name: str,
        error: str,
        tool_registry: Any,
    ) -> tuple[RetryAction, str]:
        """Decide whether to retry and build the retry prompt."""
        if self.tier == self.TIER_OFF:
            return RetryAction.ABORT, f"Adapter off (tier={self.tier}): {error}"
        return self.retry.handle_failure(tool_name, error, tool_registry)

    # ------------------------------------------------------------------
    # Adaptive per-tool strictness
    # ------------------------------------------------------------------

    def record_tool_call(self, tool_name: str, success: bool) -> None:
        """Record a tool call outcome for adaptive strictness tuning."""
        if not self._adaptive_strictness:
            return
        history = self._tool_call_history.setdefault(tool_name, [])
        history.append(success)
        # Keep only the last N calls
        if len(history) > self._strictness_window:
            self._tool_call_history[tool_name] = history[-self._strictness_window:]
        # Check if strictness needs bumping
        if len(history) >= 10:  # need at least 10 calls to judge
            rate = sum(history) / len(history)
            if rate < self._strictness_threshold:
                self._bump_tool_strictness(tool_name, rate)

    def _bump_tool_strictness(self, tool_name: str, rate: float) -> None:
        """Increase strictness for a specific tool based on failure rate.

        For tier "light": NONE→MEDIUM escalates the tool to tier "full"
        behavior (full validation + 3 retries) for that tool only.
        """
        current = self._tool_strictness.get(tool_name, self._base_strictness)
        if current == Strictness.NONE:
            new = Strictness.MEDIUM
        elif current == Strictness.MEDIUM:
            new = Strictness.STRICT
        else:
            return  # Already at max
        self._tool_strictness[tool_name] = new
        import logging
        escalation = ""
        if self.tier == self.TIER_LIGHT and new in (Strictness.MEDIUM, Strictness.STRICT):
            escalation = " [tier light→full for this tool]"
        logging.getLogger(__name__).info(
            "Adaptive strictness: %s bumped %s → %s (success rate %.1f%%)%s",
            tool_name, current.value, new.value, rate * 100, escalation,
        )

    def get_effective_strictness(self, tool_name: str) -> Strictness:
        """Get the effective strictness for a tool.

        Priority: manual per-tool override > adaptive > base model default.
        """
        return self._tool_strictness.get(tool_name, self._base_strictness)

    def set_tool_strictness(self, tool_name: str, strictness: Strictness) -> None:
        """Manually override strictness for a specific tool."""
        self._tool_strictness[tool_name] = strictness
