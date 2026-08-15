"""Which parameter of each tool carries the filesystem path the gate must see.

THE DEFECT THIS EXISTS TO CLOSE
--------------------------------
``agent_loop`` used to build the gate's subject as::

    _file_path = str(tool_input.get("file_path", "")) or None

**Not one of the 51 registered tools declares a parameter called
``file_path``.** ``write_file``, ``edit_file``, ``read_file``,
``notebook_edit``, ``todo_write`` and ``vault_read`` declare ``path``;
``image_generate`` and ``tts`` declare ``output_path``; ``video_generate``
declares ``image_path``; ``download_file`` declares ``destination``; and
``youtube_transcript`` declares ``save_to``.

So the gate got ``None`` on every call, and both ``_check_denied_path`` and
``_within_workspace`` — each guarded by ``if file_path`` — were skipped.
``denied_paths`` and ``workspace_root`` were inert for every file tool from
the initial commit (2026-04-20) until 2026-08-13. Four months, live, on a box
reachable from Telegram. A write to ``~/.gnupg/test`` succeeded through the
real dispatch path with no prompt.

It survived because every test of the control called
``gate.evaluate(..., file_path=p)`` **by hand** — supplying the argument the
caller never supplied. A 7-of-7 verification passed against it an hour before
it was found: true about the gate's logic, and about a world where the caller
passes the path. See ``tests/test_gate_sees_the_path.py``, which reaches the
gate only through ``_execute_tool_call``.

⚠ TWO LESSONS THAT SHAPE THIS FILE

1. **Enumerate from the schema, never from a name pattern.** Grepping tool
   parameters for ``*path*`` finds nine tools and MISSES
   ``download_file.destination`` and ``youtube_transcript.save_to``, both of
   which write to arbitrary local paths. Matching a name instead of reading
   the schema is the original defect one level up.
2. **An unmapped file tool must be LOUD, not exempt.** Silence is what made
   this cost four months. A tool with a path-shaped argument that nobody
   mapped resolves to UNKNOWN, which prompts — see :func:`gate_path_for`.
   ``tests/test_gate_sees_the_path.py`` also fails the build for any
   registered tool whose schema advertises a path this file does not account
   for, so the runtime fallback should never be the thing that catches it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from prometheus.permissions.path_schema import (
    PATH_KIND_DIR,
    declared_path_params,
)

log = logging.getLogger(__name__)

#: tool name -> the parameter carrying a real filesystem path.
#: Derived by reading every registered tool's schema, not by name matching.
TOOL_PATH_PARAM: dict[str, str] = {
    "write_file": "path",
    "edit_file": "path",
    "read_file": "path",
    "notebook_edit": "path",
    "image_generate": "output_path",
    "tts": "output_path",
    "video_generate": "image_path",
    "download_file": "destination",
    "youtube_transcript": "save_to",
    # Directory roots. Read-only tools, so they never reach the
    # workspace prompt (_APPROVE_TOOLS is write_file/edit_file only)
    # — mapping them buys denied_paths enforcement and nothing else,
    # which is exactly the intent: grep must not be stricter than
    # read_file, only as strict.
    "grep": "root",
    "glob": "root",
    # A file_watch task WATCHES this directory, and the task is persisted
    # and resumed across restarts. Read-only like grep/glob (it reports a
    # matched filename, never file contents), so mapping it buys
    # denied_paths and no workspace prompt — task_create is not in
    # _APPROVE_TOOLS.
    "task_create": "watch_dir",
}

#: Tools with a path-SHAPED parameter that is not a filesystem path, each with
#: the confinement that stands in the gate's place. An exemption without its
#: own guard is a hole, so every entry here is pinned by a test asserting that
#: confinement still holds — the same bargain the Documents gate won on.
PATH_PARAM_EXEMPT: dict[str, str] = {
    "vault_read":
        "`path` is a namespaced identifier relative to the brain-vault root, "
        "not a filesystem path — resolving it would produce a verdict about a "
        "path that does not exist. The tool confines itself: it resolves "
        "under the vault root and refuses to escape it, and it cannot write "
        "at all. Pinned by test_vault_read_confines_itself.",
    "todo_write":
        "`path` is relative to a runtime scratchpad the tool owns, not a "
        "user-supplied filesystem path. Pinned by "
        "test_todo_write_confines_itself.",
}

# _PATH_SHAPED (a substring tuple over parameter NAMES) lived here and was
# the THIRD name-pattern enumeration to get this wrong: it had no "root",
# so grep.root/glob.root resolved to "this tool targets no path" and the
# gate never ruled on them. Replaced by permissions/path_schema.py — the
# tool's own schema declares which params are paths, so the question is
# answered rather than guessed.


def gate_path_for(
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    schema: dict[str, Any] | None = None,
    base: Path | str | None = None,
) -> tuple[str | None, str | None]:
    """The absolute path this call targets, for the SecurityGate.

    Returns ``(path, unknown_reason)``:

    * ``(abs_path, None)`` — a real target the gate can rule on.
    * ``(None, None)``     — this tool targets no path (the common case).
    * ``(None, reason)``   — a path exists but cannot be resolved to an
      absolute one. The caller MUST treat this as requiring approval; it must
      never fall through to "allowed".

    A relative FILE path is UNKNOWN, DELIBERATELY — not resolved against the
    process's working directory. That is the same defect fixed in
    ``denied_paths`` on 2026-08-13 — a control whose target is chosen by where
    the process happens to be running — and reinstating it on the input side
    would be worse, because here it decides whether a write is allowed rather
    than merely which file is protected. A relative file path therefore
    prompts.

    A relative DIRECTORY path resolves against *base* instead. See the
    asymmetry note at the resolution site: for a read-root, declining to
    resolve makes the gate rule on a different path than the tool will read,
    and would prompt on ~81% of real ``grep``/``glob`` calls.

    Args:
        schema: the tool's advertised JSON schema. Supplies which params the
            tool DECLARES to be paths (see ``permissions.path_schema``); the
            loud fallback for unmapped tools reads this rather than guessing
            from parameter names.
        base: the directory a relative DIRECTORY param resolves against —
            the same ``context.cwd`` the tool itself resolves against. Omit
            it and directory params fall back to UNKNOWN.
    """
    if tool_name in PATH_PARAM_EXEMPT:
        return None, None

    declared = declared_path_params(schema)

    param = TOOL_PATH_PARAM.get(tool_name)
    if param is None:
        # Loud fallback: a tool nobody mapped, carrying a param its own
        # SCHEMA declares to be a path. Silence here is exactly what cost
        # four months — and guessing from the NAME is what let grep.root
        # through, so the question is now answered by the tool.
        stray = [k for k in declared
                 if isinstance((tool_input or {}).get(k), str)
                 and (tool_input or {}).get(k)]
        if stray:
            log.warning(
                "SecurityGate: tool %r has unmapped path-shaped argument(s) %s "
                "— treating as UNKNOWN (requires approval). Add it to "
                "TOOL_PATH_PARAM or PATH_PARAM_EXEMPT in "
                "prometheus/permissions/tool_paths.py.",
                tool_name, stray,
            )
            return None, (
                f"{tool_name} carries an unrecognised path argument "
                f"({', '.join(stray)}) — the security gate cannot rule on it"
            )
        return None, None

    raw = (tool_input or {}).get(param)
    if not isinstance(raw, str) or not raw.strip():
        # The parameter is optional and unset (tts with no output_path writes
        # a temp file). Nothing to rule on.
        return None, None

    raw = raw.strip()
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        # DIRECTORY params resolve against the tool's base; FILE params do
        # not. This asymmetry is deliberate and load-bearing in both
        # directions:
        #
        # * For a FILE target the rule stays "relative → UNKNOWN". Resolving
        #   against the working directory would let where the process happens
        #   to be running decide whether a WRITE is allowed — the defect fixed
        #   in denied_paths on 2026-08-13, and worse on the input side.
        # * For a DIRECTORY root, refusing to resolve is the unsound option.
        #   The tool itself computes `_resolve_path(context.cwd, root)`, so a
        #   gate that declines to resolve rules on a different path than the
        #   one the tool will actually read. And it is not theoretical: 100 of
        #   124 real grep/glob roots in telemetry are relative, so UNKNOWN
        #   here would prompt on ~81% of rooted calls and train the model to
        #   route around the tool.
        if declared.get(param) == PATH_KIND_DIR and base is not None:
            candidate = (Path(base).expanduser() / candidate)
        else:
            return None, (
                f"{tool_name} target {raw!r} is a relative path; the gate "
                f"resolves no path against the working directory, so the "
                f"target is unknown"
            )
    return str(candidate.resolve()), None
