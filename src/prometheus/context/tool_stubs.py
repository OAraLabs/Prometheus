"""How a local run offers the tools it does not describe in full (audit 2026-10-16, fix 3).

Deferred tools were unreachable on a local llama.cpp turn: the model does not
call a function that is not in ``tools``, and the server's grammar (Qwen's XML
format) admits only names in ``tools`` and, for each, only its declared
property names. So every tool is in ``tools``:

* the FULL_SCHEMA_TOOLS (the most-used, plus ``agent`` and ``ask_user``, which a
  model must never call with guessed arguments) with their whole schema;
* every other tool as a SKELETON: its name, one line of description, and its
  top-level property names with primitive types (and the allowed values of a
  short enum), all optional, sorted.

The block is sorted by name and built once per run, so it is byte-identical
across rounds (the template renders it ahead of the system text: one changed
byte re-prefills the whole prompt). A call's arguments are validated against
the real schema when it runs; a refusal carries the full schema and a minimal
valid call, at most twice per tool per turn (:func:`refusal`).

An A/B on the 4090's 27B (12 plain requests, 2026-10-16) chose skeletons over
one ``call_tool`` dispatcher: the right tool 11/12 either way, the right
arguments first time 11/12 against 9/12, for ~2,400 more prompt tokens.

:func:`sanitize` makes every schema safe for llama.cpp's JSON-schema-to-grammar
converter: local ``$ref`` inlined, ``Optional`` collapsed, and the keywords it
cannot compile or that only cost tokens dropped.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable
from typing import Any

#: Described in full on a local run: the most-used tools, plus the two a
#: model must never call with guessed arguments (agent, ask_user).
FULL_SCHEMA_TOOLS: tuple[str, ...] = (
    "bash", "read_file", "write_file", "edit_file", "grep", "glob",
    "web_search", "web_fetch", "agent", "ask_user",
)

#: ``tools.deferred_loading.local_catalog``: "stubs" (the default) or "names",
#: the 0.9.12 catalog (only the always-loaded tools, the rest named nowhere).
CATALOG_MODES = ("stubs", "names")
DEFAULT_CATALOG = "stubs"


def catalog_mode(deferred_cfg: Any) -> str:
    """The local catalog a ``tools.deferred_loading`` section selects: its
    ``local_catalog``, or the default when that is absent or not a known mode.
    One rule for the loader and the doctor rows, so they cannot disagree."""
    raw = deferred_cfg.get("local_catalog") if isinstance(deferred_cfg, dict) else None
    mode = str(raw or DEFAULT_CATALOG)
    return mode if mode in CATALOG_MODES else DEFAULT_CATALOG
#: A stub lists an enum's values only when there are at most this many; a
#: longer one is left to the refusal, which carries the whole schema.
MAX_STUB_ENUM = 8
#: Refusals with the full schema per tool per turn; the next one says stop.
MAX_SCHEMA_RETRIES = 2

_PRIMITIVES = ("string", "integer", "number", "boolean", "array", "object")
#: Keywords the grammar converter cannot compile (regex features beyond its
#: subset) or that only cost prompt tokens.
_DROP = frozenset({"title", "examples", "pattern", "format", "$schema", "discriminator"})


def first_line(description: str, limit: int = 120) -> str:
    """The first sentence or line of *description*, at most *limit* characters."""
    text = " ".join((description or "").split())
    for end in (". ", "; ", " — "):
        i = text.find(end)
        if 0 < i < limit:
            return text[: i + (1 if end == ". " else 0)].rstrip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _resolve(node: Any, defs: dict[str, Any], seen: frozenset[str] = frozenset()) -> Any:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            key = ref.rsplit("/", 1)[-1]
            if key in seen or key not in defs:
                return {"type": "object"}
            merged = {**defs[key], **{k: v for k, v in node.items() if k != "$ref"}}
            return _resolve(merged, defs, seen | {key})
        return {k: _resolve(v, defs, seen) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [_resolve(v, defs, seen) for v in node]
    return node


def _collapse(node: Any) -> Any:
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for k, v in node.items():
            if k in _DROP:
                continue
            if k == "properties" and isinstance(v, dict):
                # Property NAMES are the tool's arguments ("pattern", "title",
                # "format" among them): never dropped, only their schemas cleaned.
                out[k] = {name: _collapse(sub) for name, sub in v.items()}
            else:
                out[k] = _collapse(v)
        node = out
        for key in ("anyOf", "oneOf"):
            options = node.get(key)
            if isinstance(options, list):
                real = [o for o in options if not (isinstance(o, dict) and o.get("type") == "null")]
                if len(real) == 1 and isinstance(real[0], dict):
                    rest = {k: v for k, v in node.items() if k != key}
                    node = {**real[0], **rest}
        if isinstance(node.get("type"), list):
            types = [t for t in node["type"] if t != "null"]
            node["type"] = types[0] if len(types) == 1 else types
        return node
    if isinstance(node, list):
        return [_collapse(v) for v in node]
    return node


def sanitize(input_schema: dict[str, Any]) -> dict[str, Any]:
    """*input_schema* made safe for llama.cpp's grammar converter (see the module docstring)."""
    schema = copy.deepcopy(input_schema or {})
    defs = schema.get("$defs") or {}
    return _collapse(_resolve(schema, defs))


def primitive_type(prop: Any) -> str:
    """One JSON type for a (sanitized) property: its own, the first non-null of a
    union, ``string`` for an enum or anything unreadable."""
    if not isinstance(prop, dict):
        return "string"
    t = prop.get("type")
    if isinstance(t, str) and t in _PRIMITIVES:
        return t
    if isinstance(t, list):
        for x in t:
            if x in _PRIMITIVES:
                return x
    for key in ("anyOf", "oneOf"):
        for option in prop.get(key) or []:
            pt = primitive_type(option)
            if pt != "string" or (isinstance(option, dict) and option.get("type") == "string"):
                return pt
    return "string"


def _input_schema(schema: dict[str, Any]) -> dict[str, Any]:
    return schema.get("input_schema") or schema.get("parameters") or (schema.get("function") or {}).get(
        "parameters") or {}


def full(schema: dict[str, Any]) -> dict[str, Any]:
    """A tool's whole schema, sanitized, in the registry's shape."""
    return {"name": schema["name"], "description": schema.get("description", ""),
            "input_schema": sanitize(_input_schema(schema))}


def _stub_property(prop: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"type": primitive_type(prop)}
    enum = prop.get("enum") if isinstance(prop, dict) else None
    if isinstance(enum, list) and 0 < len(enum) <= MAX_STUB_ENUM:
        out["enum"] = list(enum)
    return out


def skeleton(schema: dict[str, Any]) -> dict[str, Any]:
    """A tool's skeleton (see the module docstring)."""
    props = (sanitize(_input_schema(schema)).get("properties") or {})
    return {
        "name": schema["name"],
        "description": first_line(schema.get("description", "")),
        "input_schema": {
            "type": "object",
            "properties": {k: _stub_property(props[k]) for k in sorted(props)},
        },
    }


def catalog(schemas: list[dict[str, Any]], full_names: Iterable[str] = FULL_SCHEMA_TOOLS
            ) -> tuple[list[dict[str, Any]], frozenset[str]]:
    """The run's ``tools``, sorted by name, and the names NOT described in full."""
    keep = set(full_names)
    block = [full(s) if s["name"] in keep else skeleton(s) for s in schemas]
    block.sort(key=lambda s: s["name"])
    return block, frozenset(s["name"] for s in schemas if s["name"] not in keep)


def minimal_example(input_schema: dict[str, Any]) -> dict[str, Any]:
    """The smallest call the (sanitized) schema accepts: its required fields, each a placeholder."""
    schema = sanitize(input_schema)
    props = schema.get("properties") or {}
    out: dict[str, Any] = {}
    for name in schema.get("required") or []:
        prop = props.get(name) or {}
        if prop.get("enum"):
            out[name] = prop["enum"][0]
            continue
        t = primitive_type(prop)
        out[name] = {"string": f"<{name}>", "integer": 1, "number": 1, "boolean": True,
                     "array": [], "object": {}}[t]
    return out


def error_summary(exc: BaseException) -> str:
    """A validation error in one short line per field (pydantic's own text
    repeats the input and links its docs: tokens the model does not need)."""
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            rows = errors()
            return "; ".join(f"{'.'.join(str(p) for p in e.get('loc') or ()) or '(input)'}: {e.get('msg')}"
                             for e in rows)
        except Exception:  # noqa: BLE001 — fall back to the text
            pass
    return str(exc).split("\n    For further information", 1)[0]


def refusal(tool_name: str, input_schema: dict[str, Any], error: str, attempt: int) -> str:
    """The answer to an invalid call of a tool the run did not describe in full."""
    if attempt > MAX_SCHEMA_RETRIES:
        return (f"Invalid input for {tool_name}: {error}\nNot retried again this turn "
                f"({MAX_SCHEMA_RETRIES} refusals already): answer without {tool_name}, or ask the user.")
    return (f"Invalid input for {tool_name}: {error}\n"
            f"Its full schema: {json.dumps(sanitize(input_schema), ensure_ascii=False, separators=(',', ':'))}\n"
            f"A minimal valid call: {json.dumps(minimal_example(input_schema), ensure_ascii=False)}\n"
            f"Call {tool_name} again with arguments that match (attempt {attempt} of {MAX_SCHEMA_RETRIES}).")
