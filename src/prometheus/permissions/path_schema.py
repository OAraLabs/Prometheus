"""How a tool DECLARES that a parameter carries a filesystem path.

WHY THIS EXISTS — the third name-pattern enumeration
-----------------------------------------------------
Three times now, a control has decided "is this a path?" by looking at the
parameter's NAME, and three times the answer has been wrong:

1. ``agent_loop`` read ``tool_input["file_path"]`` — a key no registered tool
   declares. The gate got None on every call and ``denied_paths`` was inert
   for four months (#176).
2. The fix's own guard grepped parameter names for ``*path*``, which finds
   nine tools and misses ``download_file.destination`` and
   ``youtube_transcript.save_to``. ``tool_paths.py``'s docstring records that
   lesson: *enumerate from the schema, never from a name pattern.*
3. ``_PATH_SHAPED = ("path", "file", "dir", ...)`` — the runtime fallback
   inside the fix for (2) — has no ``"root"``, so ``grep.root`` and
   ``glob.root`` resolved to "this tool targets no path" and the gate never
   ruled on them. ``read_file ~/.ssh/id_rsa`` was refused while
   ``grep --root ~/.ssh`` was not, and grep prints matching LINES.

A fourth enumeration would fail the same way. The fix is for the schema to
carry the fact, so the question is ANSWERED by the tool rather than guessed
about it: a param is a path because its author said so, next to the field.

USAGE
-----
::

    path: str = Field(..., json_schema_extra=PATH_FIELD)        # a file
    root: str | None = Field(None, json_schema_extra=DIR_FIELD) # a directory

KIND MATTERS, and it is not cosmetic. A *file* param keeps the deliberate
"relative → UNKNOWN → prompt" rule: for a write, resolving against the
process's cwd would let the caller's working directory decide whether a write
is allowed, which is the defect fixed in ``denied_paths`` on 2026-08-13. A
*directory* param is resolved against the tool's base instead, because the
tool itself already does exactly that (``grep`` computes
``_resolve_path(context.cwd, arguments.root)``), so the gate ruling on a
different path than the tool will read would be the real unsoundness. 100 of
124 real ``grep``/``glob`` roots in telemetry are relative — treating them as
UNKNOWN would prompt on ~81% of rooted calls.
"""

from __future__ import annotations

from typing import Any

#: Schema key carrying the declaration. Namespaced so it cannot collide with
#: a JSON-Schema keyword, and prefixed ``x-`` per the extension convention.
PATH_KIND_KEY = "x-prometheus-path"

#: The two kinds. FILE keeps relative→UNKNOWN; DIR is base-resolved.
PATH_KIND_FILE = "file"
PATH_KIND_DIR = "dir"

#: Drop-in values for ``Field(json_schema_extra=...)``.
PATH_FIELD: dict[str, Any] = {PATH_KIND_KEY: PATH_KIND_FILE}
DIR_FIELD: dict[str, Any] = {PATH_KIND_KEY: PATH_KIND_DIR}


def declared_path_params(schema: dict[str, Any] | None) -> dict[str, str]:
    """Map ``param name -> kind`` for every param the schema calls a path.

    Reads a tool's JSON schema (``input_model.model_json_schema()`` or the
    registry's advertised ``input_schema``). Returns ``{}`` for a schema that
    declares none — which is the common and correct case.
    """
    if not schema:
        return {}
    props = (schema.get("input_schema") or schema.get("parameters") or schema
             ).get("properties", {})
    if not isinstance(props, dict):
        return {}
    out: dict[str, str] = {}
    for name, spec in props.items():
        if not isinstance(spec, dict):
            continue
        kind = spec.get(PATH_KIND_KEY)
        if kind in (PATH_KIND_FILE, PATH_KIND_DIR):
            out[name] = kind
    return out
