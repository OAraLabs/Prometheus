"""StructuredOutputEnforcer — extract tool calls from raw model output.

Handles the messy reality of open models that don't always return clean
structured tool calls. Supports:
  - Qwen's XML tool calls: <function=NAME><parameter=K>V</parameter></function>
    (see :func:`parse_xml_tool_calls`)
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
        """Extract all tool calls from raw model text output, in document order.

        Two families, read together:

        * Qwen's XML calls, ``<function=NAME><parameter=K>V</parameter>``
          (:func:`parse_xml_tool_calls`) — the format the Qwen3.5 / Qwen3.8
          chat template teaches, and Bonsai 2, the same checkpoint. At tier
          full the tools are withheld from the request, so the server parses
          nothing and the model's XML reaches this extractor as prose. It used
          to read JSON only: the markup stripper then deleted the call — a
          PARSE DISAGREEMENT retry when the call was the whole reply, a
          silently lost call when prose preceded it. Measured on the Bonsai
          tier sweep: 21 of 204 tier-full runs ended in that disagreement, and
          full scored 25 points under light.
        * JSON calls, tried in order on the text OUTSIDE the XML function
          spans — so a JSON object inside a ``<parameter>`` value is the
          value, never a second call:

          1. JSON in ```json ... ``` fenced blocks
          2. JSON in ``` ... ``` generic fenced blocks
          3. JSON objects on their own line / at start of response
          4. Any JSON object in the text (greedy last resort)

        A reply carrying both formats keeps every call, in the order the model
        wrote them (the production 27B, told to write JSON and trained to write
        XML, writes two different calls in one reply); a call written twice —
        the same name and arguments in JSON and in XML — is one call. A reply
        with no ``<function=`` takes exactly the path it always took.
        """
        if not raw_response or not raw_response.strip():
            return []

        found: list[tuple[int, ToolUseBlock]] = []   # (offset in the reply, block)
        text = raw_response
        if "<function=" in raw_response:
            spans = _xml_tool_call_spans(raw_response, tool_registry)
            found.extend((start, block) for start, _end, block in spans)
            text = _blank_spans(raw_response, [(start, end) for start, end, _block in spans])

        from_json: list[tuple[int, ToolUseBlock]] = []

        def _add(offset: int, block: ToolUseBlock | None) -> None:
            if block is not None:
                from_json.append((offset, block))

        # --- Strategy 1: ```json ... ``` blocks ---
        for m in re.finditer(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE):
            _add(m.start(), _try_parse_tool_call(m.group(1)))

        # --- Strategy 2: ``` ... ``` blocks (any language tag) ---
        if not from_json:
            for m in re.finditer(r"```\w*\s*(\{.*?\})\s*```", text, re.DOTALL):
                _add(m.start(), _try_parse_tool_call(m.group(1)))

        # --- Strategy 3: JSON on its own line ---
        if not from_json:
            for m in re.finditer(r"^\s*(\{[^\n]+\})\s*$", text, re.MULTILINE):
                _add(m.start(), _try_parse_tool_call(m.group(1)))

        # --- Strategy 4: Any JSON object (greedy, last resort) ---
        if not from_json:
            # Find all {...} blocks, try longest first
            for m in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", text, re.DOTALL):
                _add(m.start(), _try_parse_tool_call(m.group(0)))

        found.extend(from_json)
        found.sort(key=lambda item: item[0])
        results: list[ToolUseBlock] = []
        seen_ids: set[str] = set()
        for _offset, block in found:
            key = f"{block.name}:{json.dumps(block.input, sort_keys=True)}"
            if key not in seen_ids:
                seen_ids.add(key)
                results.append(block)

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

        # SINGLE-LINE alternates: llama.cpp's GBNF parser silently REJECTS a
        # grammar whose alternates continue on the next line ("\n  | ..."), and
        # the server then runs UNCONSTRAINED with no error (live-bisected
        # 2026-07-02: `root ::= "MOO"` forces, the multi-line tool-call rule is
        # ignored; joining the alternates onto one line forces). Every
        # multi-tool grammar this generator ever emitted was llama-invalid —
        # tier-full "worked" only in unit tests that never parsed the GBNF.
        tool_choice = " | ".join(tool_alternatives)

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


# ---------------------------------------------------------------------------
# Qwen's XML tool calls
# ---------------------------------------------------------------------------

# The call format the Qwen3.5 / Qwen3.8 chat template teaches — recorded from
# the production server's /props in tests/fixtures/parity/tool_calls.trace.json,
# and pinned against it by tests/test_enforcer_xml.py:
#
#     <tool_call>
#     <function=example_function_name>
#     <parameter=example_parameter_1>
#     value_1
#     </parameter>
#     </function>
#     </tool_call>
#
# The template renders a string argument as its text and any other argument as
# JSON (``args_value | tojson``), with one newline on each side. So a value is
# text unless the tool's schema says the parameter is not a string, in which
# case the reader tries JSON and keeps the text when that fails (the validator
# then reports it). Without a registry every value is text.
_XML_FUNCTION_RE = re.compile(r"<function=([A-Za-z0-9_.\-]+)>")
_XML_PARAMETER_RE = re.compile(r"<parameter=([A-Za-z0-9_.\-]+)>")
_XML_FUNCTION_CLOSE = "</function>"
_XML_PARAMETER_CLOSE = "</parameter>"
_XML_ENVELOPE_CLOSE = "</tool_call>"


def parse_xml_tool_calls(raw_response: str, tool_registry: Any = None) -> list[ToolUseBlock]:
    """Read every ``<function=NAME>`` call in ``raw_response``, in order.

    Tolerant the way the JSON reader is tolerant of truncation: a block that
    lacks its closing tag runs to the next ``<function=``, to the envelope's
    ``</tool_call>``, or to the end of the text — the grammar's prose branch
    never constrains the XML and ``max_tokens`` can cut it. The ``<tool_call>``
    envelope is not required; the stripper removes it either way. The caller
    dedups and filters against its registry; this reads.
    """
    return [block for _start, _end, block in _xml_tool_call_spans(raw_response, tool_registry)]


def _xml_tool_call_spans(
    raw_response: str, tool_registry: Any = None,
) -> list[tuple[int, int, ToolUseBlock]]:
    """Every XML call with the span ``[start, end)`` of the text it occupies —
    from ``<function=`` through ``</function>`` when that closer is there —
    so the JSON reader can be kept off the text inside a call."""
    if not raw_response or "<function=" not in raw_response:
        return []
    results: list[tuple[int, int, ToolUseBlock]] = []
    functions = list(_XML_FUNCTION_RE.finditer(raw_response))
    for index, fn in enumerate(functions):
        body_start = fn.end()
        body_end = (
            functions[index + 1].start() if index + 1 < len(functions) else len(raw_response)
        )
        for marker in (_XML_FUNCTION_CLOSE, _XML_ENVELOPE_CLOSE):
            close = raw_response.find(marker, body_start, body_end)
            if close != -1:
                body_end = close
        body = raw_response[body_start:body_end]
        args: dict[str, Any] = {}
        params = list(_XML_PARAMETER_RE.finditer(body))
        for p_index, param in enumerate(params):
            value_start = param.end()
            value_end = params[p_index + 1].start() if p_index + 1 < len(params) else len(body)
            close = body.find(_XML_PARAMETER_CLOSE, value_start, value_end)
            if close != -1:
                value_end = close
            value = _strip_one_newline_each_side(body[value_start:value_end])
            args[param.group(1)] = _xml_value(value, fn.group(1), param.group(1), tool_registry)
        span_end = body_end
        if raw_response.startswith(_XML_FUNCTION_CLOSE, body_end):
            span_end = body_end + len(_XML_FUNCTION_CLOSE)
        results.append((fn.start(), span_end,
                        ToolUseBlock(id=f"toolu_{uuid4().hex[:12]}", name=fn.group(1), input=args)))
    return results


def _blank_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """``text`` with each span replaced by spaces of the same length, so
    offsets in the result are offsets in the original."""
    out = list(text)
    for start, end in spans:
        out[start:end] = " " * (end - start)
    return "".join(out)


def _strip_one_newline_each_side(value: str) -> str:
    """The template puts exactly one newline after ``<parameter=K>`` and one
    before ``</parameter>``; a value that itself ends in a newline keeps it."""
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


def _xml_value(text: str, tool_name: str, key: str, tool_registry: Any) -> Any:
    """The text as the schema types it: JSON for a non-string parameter, else text."""
    kind = _parameter_type(tool_registry, tool_name, key)
    if kind is None or kind == "string":
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _parameter_type(tool_registry: Any, tool_name: str, key: str) -> str | None:
    """The JSON-schema type of ``key`` on ``tool_name``, or None when unknown."""
    if tool_registry is None:
        return None
    try:
        tool = tool_registry.get(tool_name)
        schema = tool.input_model.model_json_schema()
        prop = (schema.get("properties") or {}).get(key) or {}
    except Exception:  # noqa: BLE001 — a duck-typed registry without schemas: keep the text
        return None
    kind = prop.get("type")
    if kind is None:
        # ``int | None`` renders as anyOf; the first non-null branch types it.
        for alternative in prop.get("anyOf") or prop.get("oneOf") or []:
            alt_kind = alternative.get("type") if isinstance(alternative, dict) else None
            if alt_kind and alt_kind != "null":
                kind = alt_kind
                break
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), None)
    return kind if isinstance(kind, str) else None
