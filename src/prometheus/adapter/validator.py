"""ToolCallValidator — validates and auto-repairs tool calls from open models.

Strictness levels:
  NONE   — skip all validation (Claude-level, already structured)
  MEDIUM — validate + auto-repair (Qwen, Mistral)
  STRICT — validate + repair + aggressive coercion (weaker models)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import ValidationError


def _build_structured_error(
    error: str,
    tool_name: str,
    tool_registry: Any,
    error_type: str,
) -> str:
    """Build a rich error message with context to help the model self-correct."""
    lines: list[str] = [f"Tool call failed: {error}"]

    # Available tool names
    tools = tool_registry.list_tools() if tool_registry else []
    if tools:
        names = ", ".join(t.name for t in tools)
        lines.append(f"Available tools: {names}")

    # Expected format
    lines.append('Expected format: {"name": "tool_name", "arguments": {...}}')

    # Example from the first tool that has example_call set
    for t in tools:
        ex = getattr(t, "example_call", None)
        if ex is not None:
            example = json.dumps({"name": t.name, "arguments": ex})
            lines.append(f"Example: {example}")
            break

    return "\n".join(lines)


class Strictness(str, Enum):
    NONE = "NONE"
    MEDIUM = "MEDIUM"
    STRICT = "STRICT"


@dataclass
class ValidationResult:
    valid: bool
    error: str = ""
    error_type: str = ""   # unknown_tool | invalid_json | missing_param | wrong_type | extra_param


@dataclass
class RepairResult:
    repaired: bool
    tool_name: str
    tool_input: dict[str, Any]
    repairs_made: list[str] = field(default_factory=list)
    error: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def _find_json_in_text(text: str) -> dict[str, Any] | None:
    """Extract JSON object from fenced code blocks or raw text."""
    # Try ```json ... ``` blocks first
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try bare JSON object anywhere in the text
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _coerce_value(value: Any, target_type: str) -> Any:
    """Coerce a value to the target JSON schema type."""
    if target_type == "integer":
        try:
            return int(float(str(value)))
        except (ValueError, TypeError):
            return value
    if target_type == "number":
        try:
            return float(str(value))
        except (ValueError, TypeError):
            return value
    if target_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)
    if target_type == "string":
        return str(value)
    if target_type == "array" and isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    return value


# ---------------------------------------------------------------------------
# ToolCallValidator
# ---------------------------------------------------------------------------

class ToolCallValidator:
    """Validates and auto-repairs tool calls before execution.

    Usage:
        validator = ToolCallValidator(strictness=Strictness.MEDIUM)
        result = validator.validate("bash", {"command": "ls"}, registry)
        if not result.valid:
            repair = validator.repair("bash", {"command": "ls"}, result.error, registry)
    """

    def __init__(self, strictness: Strictness | str = Strictness.NONE) -> None:
        self.strictness = Strictness(strictness) if isinstance(strictness, str) else strictness

    def validate(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_registry: Any,
    ) -> ValidationResult:
        """Validate a tool call against the registry.

        Returns ValidationResult with valid=True on success, or valid=False
        with error + error_type describing the first failure found.
        """
        if self.strictness == Strictness.NONE:
            return ValidationResult(valid=True)

        # 0. Reject empty/whitespace tool names immediately
        if not tool_name or not tool_name.strip():
            return ValidationResult(
                valid=False,
                error=_build_structured_error(
                    "Model produced empty tool name — GBNF grammar enforcement may not be active",
                    tool_name,
                    tool_registry,
                    "unknown_tool",
                ),
                error_type="unknown_tool",
            )

        # 1. Tool name must exist
        tool = tool_registry.get(tool_name)
        if tool is None:
            return ValidationResult(
                valid=False,
                error=_build_structured_error(
                    f"Unknown tool: {tool_name!r}",
                    tool_name,
                    tool_registry,
                    "unknown_tool",
                ),
                error_type="unknown_tool",
            )

        # 2. Input must be a dict
        if not isinstance(tool_input, dict):
            return ValidationResult(
                valid=False,
                error=f"Tool input must be a JSON object, got {type(tool_input).__name__}",
                error_type="invalid_json",
            )

        # 3. Validate against pydantic model
        try:
            tool.input_model.model_validate(tool_input)
        except ValidationError as exc:
            # Classify the first error
            errors = exc.errors()
            first = errors[0] if errors else {}
            etype = first.get("type", "")
            if "missing" in etype:
                error_type = "missing_param"
            elif "type" in etype or "value_error" in etype:
                error_type = "wrong_type"
            else:
                error_type = "invalid_json"
            return ValidationResult(valid=False, error=str(exc), error_type=error_type)

        # 4. STRICT: also reject unknown parameters
        if self.strictness == Strictness.STRICT:
            schema = tool.input_model.model_json_schema()
            known = set(schema.get("properties", {}).keys())
            extra = set(tool_input.keys()) - known
            if extra:
                return ValidationResult(
                    valid=False,
                    error=f"Unknown parameters for {tool_name}: {sorted(extra)}",
                    error_type="extra_param",
                )

        return ValidationResult(valid=True)

    def repair(
        self,
        tool_name: str,
        tool_input: Any,
        error: str,
        tool_registry: Any,
    ) -> RepairResult:
        """Attempt to repair a malformed tool call.

        Strategies applied in order:
        1. Fuzzy-match tool name (Levenshtein ≤ 3)
        2. Extract JSON from markdown code blocks or mixed text
        3. Coerce types per schema (string "5" → int 5)
        4. Strip unknown parameters (MEDIUM+)
        """
        repairs: list[str] = []
        repaired_name = tool_name

        # --- 0. Empty tool names cannot be fuzzy-matched ---
        if not tool_name or not tool_name.strip():
            return RepairResult(
                repaired=False,
                tool_name=tool_name,
                tool_input={} if not isinstance(tool_input, dict) else tool_input,
                error="Empty tool name cannot be repaired — enable grammar enforcement",
            )

        # --- 1. Fuzzy tool name ---
        tool = tool_registry.get(tool_name)
        if tool is None:
            best_name, best_dist = self._fuzzy_match_tool_name(tool_name, tool_registry)
            if best_name and best_dist <= 3:
                repaired_name = best_name
                tool = tool_registry.get(repaired_name)
                repairs.append(f"fuzzy-matched tool name {tool_name!r} → {repaired_name!r} (distance {best_dist})")

        if tool is None:
            return RepairResult(
                repaired=False,
                tool_name=repaired_name,
                tool_input={} if not isinstance(tool_input, dict) else tool_input,
                error=_build_structured_error(
                    f"Could not find a matching tool for {tool_name!r}",
                    tool_name,
                    tool_registry,
                    "unknown_tool",
                ),
            )

        # --- 2. Extract JSON if input is a string ---
        if isinstance(tool_input, str):
            extracted = _find_json_in_text(tool_input)
            if extracted is not None:
                tool_input = extracted
                repairs.append("extracted JSON from text/markdown")
            else:
                return RepairResult(
                    repaired=False,
                    tool_name=repaired_name,
                    tool_input={},
                    error="Could not extract JSON from tool input string",
                )

        if not isinstance(tool_input, dict):
            return RepairResult(
                repaired=False,
                tool_name=repaired_name,
                tool_input={},
                error=f"Tool input is not a dict: {type(tool_input).__name__}",
            )

        schema = tool.input_model.model_json_schema()
        properties = schema.get("properties", {})

        # --- 3. Type coercion ---
        if self.strictness in (Strictness.MEDIUM, Strictness.STRICT):
            coerced: dict[str, Any] = dict(tool_input)
            for param_name, param_schema in properties.items():
                if param_name in coerced:
                    target_type = param_schema.get("type", "")
                    if target_type:
                        original = coerced[param_name]
                        coerced_val = _coerce_value(original, target_type)
                        if coerced_val != original:
                            coerced[param_name] = coerced_val
                            repairs.append(
                                f"coerced {param_name}: {type(original).__name__} → {target_type}"
                            )
            tool_input = coerced

        # --- 4. Strip unknown parameters ---
        if self.strictness in (Strictness.MEDIUM, Strictness.STRICT):
            known = set(properties.keys())
            extra = set(tool_input.keys()) - known
            if extra:
                tool_input = {k: v for k, v in tool_input.items() if k in known}
                repairs.append(f"stripped unknown params: {sorted(extra)}")

        # --- Final validation ---
        try:
            tool.input_model.model_validate(tool_input)
            return RepairResult(
                repaired=True,
                tool_name=repaired_name,
                tool_input=tool_input,
                repairs_made=repairs,
            )
        except ValidationError as exc:
            return RepairResult(
                repaired=False,
                tool_name=repaired_name,
                tool_input=tool_input,
                repairs_made=repairs,
                error=_build_structured_error(
                    str(exc),
                    repaired_name,
                    tool_registry,
                    "invalid_json",
                ),
            )

    def _fuzzy_match_tool_name(
        self, name: str, tool_registry: Any
    ) -> tuple[str | None, int]:
        """Return (best_match_name, distance) for the closest tool name."""
        tools = tool_registry.list_tools()
        if not tools:
            return None, 999
        best_name = None
        best_dist = 999
        for tool in tools:
            d = _levenshtein(name.lower(), tool.name.lower())
            if d < best_dist:
                best_dist = d
                best_name = tool.name
        return best_name, best_dist
