"""StructuredOutputEnforcer — extract tool calls from raw model output.

Handles the messy reality of open models that don't always return clean
structured tool calls. Supports:
  - Clean JSON tool call objects
  - JSON wrapped in markdown code blocks
  - JSON mixed with prose text
  - Multiple tool calls in one response
  - Partial / truncated JSON (best-effort)

Also generates GBNF grammars for llama.cpp constrained decoding.
"""

from __future__ import annotations

import json
import re
from typing import Any
from uuid import uuid4

from prometheus.engine.messages import ToolUseBlock


# ---------------------------------------------------------------------------
# StructuredOutputEnforcer
# ---------------------------------------------------------------------------

class StructuredOutputEnforcer:
    """Extract tool calls from raw LLM text and generate GBNF grammars.

    Usage:
        enforcer = StructuredOutputEnforcer()
        calls = enforcer.extract_tool_calls(response_text, tool_registry)
        grammar = enforcer.generate_grammar(tool_schemas)
    """

    def extract_tool_calls(
        self,
        raw_response: str,
        tool_registry: Any = None,
    ) -> list[ToolUseBlock]:
        """Extract all tool calls from raw model text output.

        Tries in order:
        1. JSON in ```json ... ``` fenced blocks
        2. JSON in ``` ... ``` generic fenced blocks
        3. JSON objects on their own line / at start of response
        4. Any JSON object in the text (greedy last resort)
        """
        if not raw_response or not raw_response.strip():
            return []

        results: list[ToolUseBlock] = []
        seen_ids: set[str] = set()

        def _add(block: ToolUseBlock | None) -> None:
            if block is None:
                return
            key = f"{block.name}:{json.dumps(block.input, sort_keys=True)}"
            if key not in seen_ids:
                seen_ids.add(key)
                results.append(block)

        # --- Strategy 1: ```json ... ``` blocks ---
        for m in re.finditer(r"```json\s*(.*?)\s*```", raw_response, re.DOTALL | re.IGNORECASE):
            _add(_try_parse_tool_call(m.group(1)))

        # --- Strategy 2: ``` ... ``` blocks (any language tag) ---
        if not results:
            for m in re.finditer(r"```\w*\s*(\{.*?\})\s*```", raw_response, re.DOTALL):
                _add(_try_parse_tool_call(m.group(1)))

        # --- Strategy 3: JSON on its own line ---
        if not results:
            for m in re.finditer(r"^\s*(\{[^\n]+\})\s*$", raw_response, re.MULTILINE):
                _add(_try_parse_tool_call(m.group(1)))

        # --- Strategy 4: Any JSON object (greedy, last resort) ---
        if not results:
            # Find all {...} blocks, try longest first
            candidates = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", raw_response, re.DOTALL)
            for candidate in candidates:
                _add(_try_parse_tool_call(candidate))

        # Filter against registry if provided — always apply filter when registry given
        if tool_registry is not None:
            return [b for b in results if tool_registry.get(b.name) is not None]

        return results

    def generate_grammar(
        self,
        tool_schemas: list[dict[str, Any]],
        *,
        require_tool_use: bool = False,
        only_tool: str | None = None,
    ) -> str:
        """Generate a GBNF grammar string for llama.cpp constrained decoding.

        force-search (per-call tool_choice -> grammar selection):
          * default (require_tool_use=False, only_tool=None): root ::= tool-call | prose
            — today's "auto" grammar, byte-identical.
          * require_tool_use=True: root ::= tool-call (prose branch dropped) — the model
            MUST emit a tool call ("required").
          * only_tool="X": as required, AND tool-call restricted to X's single alternative.

        The grammar constrains the model's output to valid JSON tool calls
        matching the union of all provided tool schemas.

        Args:
            tool_schemas: List of tool schemas in Anthropic format
                         (with "name" and "input_schema" keys).

        Returns:
            GBNF grammar string suitable for the llama.cpp `grammar` parameter.
        """
        if not tool_schemas:
            return _JSON_OBJECT_GRAMMAR

        # force-search: {"tool": X} restricts the grammar to X's single alternative.
        if only_tool is not None:
            tool_schemas = [t for t in tool_schemas if t.get("name") == only_tool]
            if not tool_schemas:
                raise ValueError(f"only_tool={only_tool!r} is not among the provided tool schemas")

        # Build per-tool argument schemas + the tool-call alternatives.
        tool_arg_rules: list[str] = []
        tool_alternatives: list[str] = []

        for tool in tool_schemas:
            rule_name = _make_rule_name(tool["name"])
            schema = tool.get("input_schema", {})
            arg_rule = _schema_to_grammar_rule(schema, rule_name)
            tool_arg_rules.append(arg_rule)
            tool_alternatives.append(
                f'"{{" ws "\\\"name\\\"" ws ":" ws "\\"{tool["name"]}\\"" ws ","'
                f' ws "\\\"arguments\\\"" ws ":" ws {rule_name}-args ws "}}"'
            )

        tool_choice = "\n  | ".join(tool_alternatives)

        # Tool-OR-text root: a constrained-decoding agent must still be able to
        # answer in prose, so the root permits either a valid tool-call object
        # OR free text that doesn't start with '{'. Without the prose branch the
        # grammar would force a tool call on every turn and break plain answers.
        grammar_parts = [
            "# Prometheus tool-call grammar — generated by StructuredOutputEnforcer",
            "",
            ("root ::= tool-call" if (require_tool_use or only_tool is not None) else "root ::= tool-call | prose"),
            "",
            f"tool-call ::= {tool_choice}",
            "",
            "# Prose = any text that doesn't begin with '{' (so it can't be",
            "# mistaken for a tool-call object).",
            "prose ::= [^{] anychar*",
            "anychar ::= [^\\x00]",
            "",
        ]
        grammar_parts.extend(tool_arg_rules)
        grammar_parts.append("")
        grammar_parts.extend(_BASE_JSON_RULES.splitlines())

        return "\n".join(grammar_parts)

    @staticmethod
    def grammar_admits_tool(grammar: str, tool_name: str) -> bool:
        """True if `grammar` permits a tool_use of `tool_name` — its tool-call alternation
        contains that tool's quoted name production. A forced-tool grammar fixes the name,
        so a different tool is not a permissible production (returns False for it)."""
        if not grammar:
            return False
        return ('\\"' + tool_name + '\\"') in grammar


# ---------------------------------------------------------------------------
# GBNF helpers
# ---------------------------------------------------------------------------

def _make_rule_name(tool_name: str) -> str:
    """Convert a tool name to a safe GBNF rule name."""
    return re.sub(r"[^a-zA-Z0-9-]", "-", tool_name).strip("-")


def _schema_to_grammar_rule(schema: dict[str, Any], rule_prefix: str) -> str:
    """Generate a GBNF rule for a JSON-object arguments schema.

    Produces ALWAYS-VALID JSON (the prior version emitted forced trailing and
    double commas):
      - Required props are comma-separated.
      - Each optional prop is an independently-omittable ``(ws "," ws pair)?``
        clause appended AFTER the required ones — so a leading comma only ever
        appears when a required prop precedes it.
      - With no required props, the object is ``{ (member (, member)*)? }`` over
        the known keys — no member forces a trailing comma.
    """
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    if not properties:
        return f"{rule_prefix}-args ::= object"

    def _pair(name: str) -> str:
        return f'"\\"{name}\\"" ws ":" ws {_type_to_grammar(properties[name])}'

    req_props = [p for p in properties if p in required]
    opt_props = [p for p in properties if p not in required]

    if req_props:
        body = ' ws "," ws '.join(_pair(p) for p in req_props)
        for p in opt_props:
            body += f' (ws "," ws {_pair(p)})?'
        return f'{rule_prefix}-args ::= "{{" ws {body} ws "}}"'

    # All-optional: any subset of the known keys, comma-separated. (Order/dups
    # aren't schema-enforced here — pydantic validation downstream handles that;
    # the grammar's job is valid JSON with known keys + typed values.)
    member_rule = f"{rule_prefix}-member"
    alts = " | ".join(_pair(p) for p in opt_props)
    return (
        f'{rule_prefix}-args ::= "{{" ws ( {member_rule} '
        f'(ws "," ws {member_rule})* )? ws "}}"\n'
        f"{member_rule} ::= {alts}"
    )


def _type_to_grammar(schema: dict[str, Any]) -> str:
    """Map a JSON schema type to a GBNF terminal."""
    t = schema.get("type", "")
    if t == "string":
        return "string"
    if t == "integer":
        return "integer"
    if t == "number":
        return "number"
    if t == "boolean":
        return "boolean"
    if t == "array":
        return "array"
    if t == "null":
        return '"null"'
    if "anyOf" in schema or "oneOf" in schema:
        return "value"
    return "value"


_BASE_JSON_RULES = """\
# Base JSON grammar rules
string  ::= "\\"" ([^"\\\\] | "\\\\" .)* "\\""
integer ::= ("-")? [0-9]+
number  ::= ("-")? [0-9]+ ("." [0-9]+)? (("e" | "E") ("-" | "+")? [0-9]+)?
boolean ::= "true" | "false"
null    ::= "null"
value   ::= object | array | string | number | boolean | null
array   ::= "[" ws (value (ws "," ws value)*)? ws "]"
object  ::= "{" ws (string ws ":" ws value (ws "," ws string ws ":" ws value)*)? ws "}"
ws      ::= ([ \\t\\n\\r])*
"""

_JSON_OBJECT_GRAMMAR = """\
root   ::= object
value  ::= object | array | string | number | boolean | null
object ::= "{" ws (string ws ":" ws value (ws "," ws string ws ":" ws value)*)? ws "}"
array  ::= "[" ws (value (ws "," ws value)*)? ws "]"
string ::= "\\"" ([^"\\\\] | "\\\\" .)* "\\""
number ::= ("-")? [0-9]+ ("." [0-9]+)? (("e" | "E") ("-" | "+")? [0-9]+)?
boolean ::= "true" | "false"
null   ::= "null"
ws     ::= ([ \\t\\n\\r])*
"""


# ---------------------------------------------------------------------------
# Parse helper
# ---------------------------------------------------------------------------

def _try_parse_tool_call(text: str) -> ToolUseBlock | None:
    """Try to parse text as a tool call JSON object."""
    text = text.strip()
    if not text.startswith("{"):
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to repair truncated JSON by closing open braces
        repaired = _repair_truncated_json(text)
        if repaired is None:
            return None
        data = repaired

    if not isinstance(data, dict):
        return None

    name = (
        data.get("name")
        or data.get("function")
        or data.get("tool_name")
        or data.get("tool")
    )
    if not name or not isinstance(name, str):
        return None

    args = (
        data.get("arguments")
        or data.get("parameters")
        or data.get("args")
        or data.get("input")
        or {}
    )
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}

    return ToolUseBlock(
        id=f"toolu_{uuid4().hex[:12]}",
        name=name,
        input=args,
    )


def _repair_truncated_json(text: str) -> dict[str, Any] | None:
    """Try to repair truncated JSON by appending closing characters."""
    opens = text.count("{") - text.count("}")
    closes = text.count("[") - text.count("]")
    if opens <= 0 and closes <= 0:
        return None
    candidate = text + "]" * closes + "}" * opens
    try:
        result = json.loads(candidate)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        return None
