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

#: Substrings that make an argument NAME look like it could carry a path.
#: Used only by the loud fallback and the build-time guard — never as the
#: source of truth for the mapping above.
_PATH_SHAPED = ("path", "file", "dir", "destination", "save_to", "output")


def _looks_path_shaped(name: str) -> bool:
    n = name.lower()
    return any(tok in n for tok in _PATH_SHAPED)


def gate_path_for(
    tool_name: str, tool_input: dict[str, Any],
) -> tuple[str | None, str | None]:
    """The absolute path this call targets, for the SecurityGate.

    Returns ``(path, unknown_reason)``:

    * ``(abs_path, None)`` — a real target the gate can rule on.
    * ``(None, None)``     — this tool targets no path (the common case).
    * ``(None, reason)``   — a path exists but cannot be resolved to an
      absolute one. The caller MUST treat this as requiring approval; it must
      never fall through to "allowed".

    RELATIVE PATHS ARE UNKNOWN, DELIBERATELY. They are not resolved against
    the process's working directory. That is the same defect fixed in
    ``denied_paths`` on 2026-08-13 — a control whose target is chosen by where
    the process happens to be running — and reinstating it on the input side
    would be worse, because here it decides whether a write is allowed rather
    than merely which file is protected. A relative path therefore prompts.
    The right long-term answer is threading each tool's real base directory
    (a coding run knows its repo); that is a separate change.
    """
    if tool_name in PATH_PARAM_EXEMPT:
        return None, None

    param = TOOL_PATH_PARAM.get(tool_name)
    if param is None:
        # Loud fallback: a tool nobody mapped, carrying a path-shaped
        # argument. Silence here is exactly what cost four months.
        stray = [k for k, v in (tool_input or {}).items()
                 if _looks_path_shaped(k) and isinstance(v, str) and v]
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
        return None, (
            f"{tool_name} target {raw!r} is a relative path; the gate resolves "
            f"no path against the working directory, so the target is unknown"
        )
    return str(candidate.resolve()), None
