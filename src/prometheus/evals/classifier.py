"""Failure classification — model issue vs harness issue.

Analyzes tool traces and eval results to determine whether a failure
is caused by the LLM model or the Prometheus harness, so you can
tell what to fix vs what requires a better model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class FailureSource(str, Enum):
    """Where the failure originated."""

    PASS = "pass"          # No failure
    MODEL = "model"        # LLM model issue (can't fix in harness code)
    HARNESS = "harness"    # Harness bug (tool crash, permission block, adapter error)
    UNCLEAR = "unclear"    # Can't determine from available data


class FailureCategory(str, Enum):
    """Specific failure type."""

    # Pass
    NONE = "none"

    # Model failures
    NO_TOOL_CALL = "model:no_tool_call"         # Answered without using required tool
    WRONG_TOOL = "model:wrong_tool"             # Called a different tool than expected
    BAD_ARGS = "model:bad_args"                 # Tool called but with invalid arguments
    HALLUCINATED_OUTPUT = "model:hallucinated"  # Made up results not from tools
    INCOMPLETE = "model:incomplete"             # Started but didn't finish multi-step

    # Harness failures
    TOOL_ERROR = "harness:tool_error"           # Tool execution returned is_error=True
    TOOL_CRASH = "harness:tool_crash"           # Task crashed during tool execution
    VALIDATION_FAIL = "harness:validation"      # Adapter rejected the tool call
    PERMISSION_DENIED = "harness:permission"    # Security gate blocked the call

    # Unclear
    UNKNOWN = "unclear:unknown"


@dataclass
class FailureClassification:
    """Classification result for a single eval task."""

    source: FailureSource
    category: FailureCategory
    detail: str


def classify_failure(
    task_id: str,
    expected_tools: list[str],
    tool_trace: list[dict[str, Any]],
    agent_output: str,
    error: str | None,
    metric_scores: dict[str, float],
) -> FailureClassification:
    """Classify a task result as model issue, harness issue, or pass.

    Pure logic on trace data — no LLM call.

    Args:
        task_id: Task identifier.
        expected_tools: Tools the task expects to be called.
        tool_trace: List of {"tool_name", "result", "is_error"} dicts.
        agent_output: The agent's final text output.
        error: Exception message if the task crashed, None otherwise.
        metric_scores: Dict of metric_name -> score (0.0-1.0).
    """
    tools_used = [t.get("tool_name", "") for t in tool_trace]
    tools_used_set = set(tools_used)
    tool_errors = [t for t in tool_trace if t.get("is_error")]

    # --- Check for crash ---
    if error:
        # If error message mentions tool/permission/validation keywords → harness
        error_lower = error.lower()
        harness_keywords = [
            "unsupported", "permission", "denied", "blocked",
            "validation", "invalid input", "unknown tool",
            "no tool registry", "non-relative",
        ]
        if any(kw in error_lower for kw in harness_keywords):
            return FailureClassification(
                source=FailureSource.HARNESS,
                category=FailureCategory.TOOL_CRASH,
                detail=f"Crash with harness keyword: {error[:150]}",
            )
        # Otherwise could be model causing the crash (bad args, etc.)
        return FailureClassification(
            source=FailureSource.UNCLEAR,
            category=FailureCategory.UNKNOWN,
            detail=f"Crash: {error[:150]}",
        )

    # --- Check if all metrics passed ---
    all_passed = all(s >= 0.5 for s in metric_scores.values()) if metric_scores else True
    if all_passed and not tool_errors:
        return FailureClassification(
            source=FailureSource.PASS,
            category=FailureCategory.NONE,
            detail="All metrics passed",
        )

    # --- Check for harness-side tool errors ---
    if tool_errors:
        error_details = []
        for te in tool_errors:
            result = te.get("result", "")
            result_lower = result.lower()

            if "permission denied" in result_lower or "blocked" in result_lower:
                return FailureClassification(
                    source=FailureSource.HARNESS,
                    category=FailureCategory.PERMISSION_DENIED,
                    detail=f"{te.get('tool_name')}: {result[:150]}",
                )
            if "invalid input" in result_lower or "validation" in result_lower:
                # Could be model sending bad args or harness schema issue
                return FailureClassification(
                    source=FailureSource.MODEL,
                    category=FailureCategory.BAD_ARGS,
                    detail=f"{te.get('tool_name')}: {result[:150]}",
                )
            if "unknown tool" in result_lower:
                return FailureClassification(
                    source=FailureSource.MODEL,
                    category=FailureCategory.WRONG_TOOL,
                    detail=f"Called unknown tool: {result[:150]}",
                )
            error_details.append(f"{te.get('tool_name')}: {result[:80]}")

        return FailureClassification(
            source=FailureSource.HARNESS,
            category=FailureCategory.TOOL_ERROR,
            detail="; ".join(error_details),
        )

    # --- Model-side classification ---

    # No tools used but tools were expected
    if expected_tools and not tools_used:
        return FailureClassification(
            source=FailureSource.MODEL,
            category=FailureCategory.NO_TOOL_CALL,
            detail=f"Expected {expected_tools} but no tools called",
        )

    # Wrong tools used
    if expected_tools and tools_used_set:
        overlap = tools_used_set & set(expected_tools)
        if not overlap:
            return FailureClassification(
                source=FailureSource.MODEL,
                category=FailureCategory.WRONG_TOOL,
                detail=f"Used {sorted(tools_used_set)}, expected {expected_tools}",
            )

    # Hallucination detected by metric
    halluc_score = metric_scores.get("No Hallucination", 1.0)
    if halluc_score < 0.5:
        return FailureClassification(
            source=FailureSource.MODEL,
            category=FailureCategory.HALLUCINATED_OUTPUT,
            detail=f"No Hallucination score: {halluc_score:.2f}",
        )

    # Task completion failed but tools were used correctly
    completion_score = metric_scores.get("Task Completion", 1.0)
    tool_score = metric_scores.get("Tool Usage", 1.0)
    if completion_score < 0.5 and tool_score >= 0.5:
        return FailureClassification(
            source=FailureSource.MODEL,
            category=FailureCategory.INCOMPLETE,
            detail=f"Tools OK ({tool_score:.2f}) but completion low ({completion_score:.2f})",
        )

    # Some metric failed but can't pinpoint why
    if not all_passed:
        lowest = min(metric_scores.items(), key=lambda x: x[1])
        return FailureClassification(
            source=FailureSource.UNCLEAR,
            category=FailureCategory.UNKNOWN,
            detail=f"Lowest metric: {lowest[0]}={lowest[1]:.2f}",
        )

    return FailureClassification(
        source=FailureSource.PASS,
        category=FailureCategory.NONE,
        detail="All checks passed",
    )
