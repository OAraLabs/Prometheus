"""ToolCallValidator — validates and auto-repairs tool calls from open models.

Strictness levels (policy — how aggressively to validate/repair):
  NONE   — invariants only: non-empty name, name in registry, input is a
           dict. No schema validation, no coercion.
  MEDIUM — invariants + schema validation + auto-repair (Qwen, Mistral)
  STRICT — MEDIUM + aggressive coercion + unknown-param rejection

Invariants run at every level: strictness governs repair aggressiveness,
never whether structural sanity is checked. So does one repair: a ``null``
for an optional parameter the schema refuses is dropped, so its default
applies (``null_drop``). It guesses nothing — absent and null both mean
"not given" — so no level has a reason to refuse it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
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


# The kind of each adapter repair (WP-X.54 T-2). Until now a repair was known
# only by its free-text ``repair_log`` line; training and telemetry need to
# count and filter by kind without parsing prose. ``other`` is any entry the
# adapter did not mint (a plain string from elsewhere).
REPAIR_KINDS: tuple[str, ...] = (
    "fuzzy_name",     # tool name fuzzy-matched to a registered one
    "json_extract",   # arguments pulled out of a text/markdown string
    "null_drop",      # a null for an optional param dropped, so its default applies
    "type_coerce",    # an argument coerced to its schema type
    "strip_params",   # unknown parameters dropped
    "dict_unwrap",    # phantom dict nesting removed (adapter/unwrap.py)
    "other",
)


class RepairNote(str):
    """One ``repair_log`` entry: the same string it always was, plus its kind.

    A ``str`` subclass so that what the adapter returns does not change:
    ``validate_and_repair`` still returns a ``list[str]`` that compares,
    joins, counts and JSON-encodes exactly as before. The kind rides beside
    the text, and ``repair_kind`` reads it back (``other`` for a plain str).
    """

    kind: str

    def __new__(cls, text: str, kind: str) -> RepairNote:
        if kind not in REPAIR_KINDS:
            raise ValueError(f"unknown repair kind {kind!r}")
        note = super().__new__(cls, text)
        note.kind = kind
        return note

    def __getnewargs__(self) -> tuple[str, str]:  # type: ignore[override]
        # copy/pickle rebuild through __new__, which needs the kind too.
        return str(self), self.kind


def repair_kind(entry: str) -> str:
    """The kind of one ``repair_log`` entry; ``other`` when it carries none."""
    kind = getattr(entry, "kind", None)
    return kind if kind in REPAIR_KINDS else "other"


def repair_kinds(log: Iterable[str]) -> list[str]:
    """The kinds of a ``repair_log``, one per entry, in order."""
    return [repair_kind(entry) for entry in log]


@dataclass
class RepairResult:
    repaired: bool
    tool_name: str
    tool_input: dict[str, Any]
    repairs_made: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def repair_kinds(self) -> list[str]:
        return repair_kinds(self.repairs_made)


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


def _drop_optional_nulls(
    input_model: Any, tool_input: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Drop each optional parameter sent as ``null`` that the schema refuses.

    Small local models write ``"limit": null`` for a parameter they mean to
    leave unset. qwen2.5:7b did it on read_file (2026-10-05): two
    input_validation failures, and then a README it had never read, invented
    in its reply. The key's absence already says "unset", so the null is
    dropped and the field's default applies.

    Only a null that pydantic itself refuses, on a parameter the schema does
    not require. A null that an ``Optional`` field accepts is a value, so it
    stays. A null for a required parameter also stays, to be refused:
    dropping it would only turn "wrong type" into "missing".
    """
    if not any(value is None for value in tool_input.values()):
        return tool_input, []
    try:
        input_model.model_validate(tool_input)
        return tool_input, []
    except ValidationError as exc:
        refused = {err["loc"][0] for err in exc.errors() if err.get("loc")}
    except Exception:
        # A tool's own validator raising something else is not this step's
        # business: leave the call as it came, for the loop's own check.
        return tool_input, []
    try:
        schema = input_model.model_json_schema()
    except Exception:
        return tool_input, []
    optional = set(schema.get("properties", {})) - set(schema.get("required", []))
    dropped = [
        name for name, value in tool_input.items()
        if value is None and name in refused and name in optional
    ]
    if not dropped:
        return tool_input, []
    kept = {name: value for name, value in tool_input.items() if name not in dropped}
    return kept, [
        RepairNote(f"dropped null {name}: optional, its default applies", "null_drop")
        for name in dropped
    ]


def _coerce_value(value: Any, target_type: str) -> Any:
    """Coerce a value to the target JSON schema type."""
    # A null is never coerced. An optional param's null was already dropped
    # (null_drop), and a required one's stays, to be refused. Before this,
    # str(None) made a null path the file "None", and bool(None) made False.
    if value is None:
        return value
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

        Two layers (invariants-vs-policy split):

        **Invariants** — structural sanity, checked at EVERY strictness:
        non-empty name, name exists in registry, input is a dict. A call
        failing these is unexecutable garbage no tier should pass through.
        (Pre-split, ``Strictness.NONE`` short-circuited above these checks,
        which let 232 empty-name calls flow past a guard written for
        exactly that failure — strictness must govern how aggressively we
        repair, never whether we sanity-check.)

        **Policy** — strictness-gated: pydantic schema validation (MEDIUM+),
        unknown-parameter rejection (STRICT).
        """
        # ── Invariants: run unconditionally ────────────────────────────
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

        # ── Policy: strictness-gated from here down ────────────────────
        if self.strictness == Strictness.NONE:
            return ValidationResult(valid=True)

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

    def drop_optional_nulls(
        self,
        tool_name: str,
        tool_input: Any,
        tool_registry: Any,
    ) -> tuple[Any, list[str]]:
        """Drop the nulls an optional parameter's schema refuses, at every strictness.

        The adapter runs this BEFORE ``validate``. At ``NONE`` (tier light)
        ``validate`` never reads the schema, so a null there would never reach
        ``repair``; the loop's own pydantic check would refuse it instead.
        Returns the input unchanged, and no notes, when the tool is unknown or
        the input is not a dict: ``repair`` runs this step again once it has
        a tool and a dict.
        """
        tool = tool_registry.get(tool_name) if tool_name else None
        if tool is None or not isinstance(tool_input, dict):
            return tool_input, []
        return _drop_optional_nulls(tool.input_model, tool_input)

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
        2b. Drop a null for an optional param, so its default applies (every level)
        3. Coerce types per schema (string "5" → int 5) (MEDIUM+)
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
                repairs.append(RepairNote(
                    f"fuzzy-matched tool name {tool_name!r} → {repaired_name!r} (distance {best_dist})",
                    "fuzzy_name",
                ))

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
                repairs.append(RepairNote("extracted JSON from text/markdown", "json_extract"))
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

        # --- 2b. Null for an optional param → its default (every strictness) ---
        # A no-op after the adapter's own pre-pass, except where this call
        # only now has a tool (a fuzzy-matched name) or a dict (extracted JSON).
        tool_input, null_notes = _drop_optional_nulls(tool.input_model, tool_input)
        repairs.extend(null_notes)

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
                            repairs.append(RepairNote(
                                f"coerced {param_name}: {type(original).__name__} → {target_type}",
                                "type_coerce",
                            ))
            tool_input = coerced

        # --- 4. Strip unknown parameters ---
        if self.strictness in (Strictness.MEDIUM, Strictness.STRICT):
            known = set(properties.keys())
            extra = set(tool_input.keys()) - known
            if extra:
                tool_input = {k: v for k, v in tool_input.items() if k in known}
                repairs.append(RepairNote(
                    f"stripped unknown params: {sorted(extra)}", "strip_params",
                ))

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
