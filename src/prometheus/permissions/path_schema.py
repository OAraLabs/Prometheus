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

    path: str = Field(..., json_schema_extra=PATH_FIELD_WRITE)   # written
    src:  str = Field(..., json_schema_extra=PATH_FIELD_READ)    # only read
    root: str | None = Field(None, json_schema_extra=DIR_FIELD_READ)

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

#: Schema key carrying whether the tool READS this path or WRITES to it.
#: Separate from the kind: "is it a path" and "does the tool write there" are
#: different questions, and the workspace boundary only cares about the second.
PATH_ACCESS_KEY = "x-prometheus-path-access"

PATH_ACCESS_READ = "read"
PATH_ACCESS_WRITE = "write"

#: Drop-in values for ``Field(json_schema_extra=...)``. Say which you mean.
PATH_FIELD_WRITE: dict[str, Any] = {
    PATH_KIND_KEY: PATH_KIND_FILE, PATH_ACCESS_KEY: PATH_ACCESS_WRITE,
}
PATH_FIELD_READ: dict[str, Any] = {
    PATH_KIND_KEY: PATH_KIND_FILE, PATH_ACCESS_KEY: PATH_ACCESS_READ,
}
DIR_FIELD_WRITE: dict[str, Any] = {
    PATH_KIND_KEY: PATH_KIND_DIR, PATH_ACCESS_KEY: PATH_ACCESS_WRITE,
}
DIR_FIELD_READ: dict[str, Any] = {
    PATH_KIND_KEY: PATH_KIND_DIR, PATH_ACCESS_KEY: PATH_ACCESS_READ,
}

#: Kind without an access declaration. Kept so a field that predates the
#: access key still classifies as a path — but it resolves to WRITE (see
#: ``declared_path_access``), which prompts rather than passes. Every tool in
#: this repo declares access explicitly; ``test_write_boundary_is_a_property``
#: fails the build if one stops.
PATH_FIELD: dict[str, Any] = {PATH_KIND_KEY: PATH_KIND_FILE}
DIR_FIELD: dict[str, Any] = {PATH_KIND_KEY: PATH_KIND_DIR}


def declared_path_access(schema: dict[str, Any] | None) -> dict[str, str]:
    """Map ``param name -> "read" | "write"`` for every declared path param.

    WHY ACCESS IS DECLARED PER PARAMETER, NOT INFERRED FROM THE TOOL
    ----------------------------------------------------------------
    The workspace boundary used to be ``tool_name in {"write_file",
    "edit_file"}`` — a list of names, and four tools that write to arbitrary
    paths were not on it (``notebook_edit``, ``download_file``, ``tts``,
    ``youtube_transcript``). This module's own docstring already records three
    earlier name enumerations that failed the same way; that was the fourth.

    The obvious repair — "apply the boundary whenever the call is not
    read-only" — is a genuine property rather than a list, and it is still
    wrong, in the over-refusing direction. Two params are paths the tool only
    READS while the tool itself is not read-only:

      * ``video_generate.image_path`` — the source image it animates.
      * ``task_create.watch_dir``     — a directory a file_watch task watches.

    Keying on the tool would prompt for both. Keying on the PARAMETER does
    not, because the author states which one it is next to the field — the
    same bargain this module already made for "is it a path".

    UNSTATED RESOLVES TO WRITE, deliberately. A path whose access nobody
    declared is unknown, and unknown must prompt rather than pass: that is the
    direction every previous failure here went the wrong way.
    """
    out: dict[str, str] = {}
    props = _properties(schema)
    for name, kind in declared_path_params(schema).items():
        spec = props.get(name)
        access = spec.get(PATH_ACCESS_KEY) if isinstance(spec, dict) else None
        out[name] = (
            PATH_ACCESS_READ if access == PATH_ACCESS_READ else PATH_ACCESS_WRITE
        )
        del kind
    return out


def _properties(schema: dict[str, Any] | None) -> dict[str, Any]:
    if not schema:
        return {}
    props = (schema.get("input_schema") or schema.get("parameters") or schema
             ).get("properties", {})
    return props if isinstance(props, dict) else {}


def declared_path_params(schema: dict[str, Any] | None) -> dict[str, str]:
    """Map ``param name -> kind`` for every param the schema calls a path.

    Reads a tool's JSON schema (``input_model.model_json_schema()`` or the
    registry's advertised ``input_schema``). Returns ``{}`` for a schema that
    declares none — which is the common and correct case.
    """
    out: dict[str, str] = {}
    for name, spec in _properties(schema).items():
        if not isinstance(spec, dict):
            continue
        kind = spec.get(PATH_KIND_KEY)
        if kind in (PATH_KIND_FILE, PATH_KIND_DIR):
            out[name] = kind
    return out
