"""The descriptive/behavioural split, converted from a comment into a failure.

Piece 5 of the 2026-08-17 arc. ``context.session_id`` is the real conversation
id on four of five caller paths, but the WEB path uses one shared, pre-built
LoopContext whose ``session_id`` is the literal ``"web"`` — a model-routing
namespace, not a session. Two TELEMETRY writers read it raw and recorded "web"
for every web turn.

Correcting them to the per-call id is right. Correcting the BEHAVIOURAL readers
the same way would be a permissions change:

    origin_from_session_id("web")                 -> user     (a literal)
    origin_from_session_id("telegram:8139…")      -> user     (a prefix)
    origin_from_session_id("web:abc")             -> SYSTEM   ("web:" is no prefix)

and that classification decides whether a human is treated as present to
sanction the next tool call.

Nothing today passes a real ``web:*`` id into that function, so the whole suite
stays green either way — the defect's enabling condition IS "no path exercises
it". A test that must drive the path therefore inherits the same blindness, so
this guard is STRUCTURAL: it asserts over the call sites themselves.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "prometheus"

# The name given to the corrected per-call value in _run_loop. If it appears
# inside an origin_from_session_id() argument, a descriptive fix has leaked
# into a behavioural site.
EFFECTIVE = "effective_session_id"

# Exact argument source permitted at a behavioural call site. Anything else —
# including the precedence expression, a local alias, or a threaded parameter —
# fails, because from source alone we cannot prove an alias is not the
# effective id.
# Compared against ``ast.unparse`` output, which normalises quoting — so
# these are written in ITS form, not the source's.
ALLOWED_ARGS = {
    "context.session_id",
    "context.get('session_id') if isinstance(context, dict) else None",
    # REVIEWED 2026-09-08 (message tool destination perimeter). A TOOL has no
    # LoopContext — ToolExecutionContext carries only cwd and metadata — so it
    # cannot use the first form. This reads the value the agent loop puts there
    # at agent_loop.py:4062, whose source is the literal `context.session_id`:
    #
    #     metadata={..., "session_id": context.session_id,
    #               **(context.tool_metadata or {})}
    #
    # so it is the reviewed form one hop away, not a new alias. Two properties
    # were checked before admitting it, and both are asserted below:
    #   1. tool_metadata spreads LAST and could shadow the key. It is populated
    #      only by daemon.py with "printing_press", never with a session id and
    #      never from tool arguments — so a model cannot forge an origin.
    #   2. A tool reached with no metadata (jobs, cron) yields None, which
    #      classifies as SYSTEM. Absence must fail closed, not open.
    "(context.metadata or {}).get('session_id')",
}


def _origin_call_args() -> list[tuple[str, int, str]]:
    """Every origin_from_session_id(...) call in src/, as (file, line, argsrc)."""
    out: list[tuple[str, int, str]] = []
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "origin_from_session_id" not in text:
            continue
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = getattr(fn, "id", None) or getattr(fn, "attr", None)
            if name != "origin_from_session_id":
                continue
            if not node.args:
                continue
            out.append((
                str(path.relative_to(SRC.parent.parent)),
                node.lineno,
                ast.unparse(node.args[0]),
            ))
    return out


def test_the_behavioural_sites_exist_at_all():
    """Guard the guard: if the calls are renamed away, this test must not
    silently pass by finding nothing to check."""
    calls = _origin_call_args()
    assert len(calls) >= 3, (
        "expected at least 3 origin_from_session_id call sites, found "
        f"{len(calls)} — if the function was renamed, update this guard rather "
        "than deleting it"
    )


@pytest.mark.parametrize("case", _origin_call_args(), ids=lambda c: f"{c[0]}:{c[1]}")
def test_origin_classification_never_receives_the_effective_session_id(case):
    """THE GUARD. A descriptive fix must not reach a behavioural reader."""
    file, line, argsrc = case
    assert EFFECTIVE not in argsrc, (
        f"{file}:{line} passes {argsrc!r} to origin_from_session_id.\n\n"
        f"{EFFECTIVE} is the per-call conversation id. Feeding it here is a "
        "PERMISSIONS CHANGE, not a telemetry fix: 'web' is a member of "
        "_USER_SESSION_LITERALS (-> user) while 'web:' is absent from "
        "_USER_SESSION_PREFIXES (-> system), so every Beacon turn would be "
        "demoted to the stricter class that governs whether a human can "
        "sanction the next tool call.\n\n"
        "If web sessions SHOULD classify differently, change "
        "_USER_SESSION_PREFIXES deliberately and update this guard — do not "
        "arrive there by threading an id."
    )
    assert argsrc in ALLOWED_ARGS, (
        f"{file}:{line} passes {argsrc!r} to origin_from_session_id, which is "
        "not one of the reviewed forms:\n  "
        + "\n  ".join(sorted(ALLOWED_ARGS))
        + "\n\nThis site is BEHAVIOURAL. An alias cannot be proven safe from "
        "source, so new argument forms must be reviewed and added here."
    )


def test_the_literal_web_is_still_load_bearing():
    """The premise the guard rests on. If this flips, the guard's reasoning —
    and the daemon.py comment — are stale and must be revisited."""
    from prometheus.permissions.checker import (
        ORIGIN_SYSTEM,
        ORIGIN_USER,
        origin_from_session_id,
    )

    assert origin_from_session_id("web") == ORIGIN_USER
    assert origin_from_session_id("telegram:8139235390") == ORIGIN_USER
    # The trap, asserted so it cannot change unnoticed.
    assert origin_from_session_id("web:abc") == ORIGIN_SYSTEM, (
        "'web:' now classifies as USER — the piece-5 split may no longer be "
        "necessary, and the web:-prefix question should be closed explicitly"
    )


def test_tool_metadata_cannot_shadow_the_trusted_session_id():
    """Premise 1 of the message-tool admission in ALLOWED_ARGS.

    The agent loop spreads ``context.tool_metadata`` AFTER "session_id", so a
    "session_id" key there would win. Nothing sets one today, and nothing may:
    a tool that reads origin from metadata is only as trustworthy as the
    dict, and tool_metadata must stay daemon-populated.
    """
    keys: set[str] = set()
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "_tool_metadata[" not in text:
            continue
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "_tool_metadata"
                and isinstance(node.slice, ast.Constant)
            ):
                keys.add(str(node.slice.value))

    # Guard the guard: if the write moves or is renamed, find nothing and say so
    # rather than passing vacuously.
    assert keys, "found no tool_metadata key writes — has the write site moved?"
    assert "session_id" not in keys, (
        f"tool_metadata now carries a 'session_id' key ({sorted(keys)}). It "
        "spreads after the loop's trusted value at agent_loop.py:4062, so it "
        "would OVERRIDE the origin that the message tool's destination "
        "perimeter depends on. Either stop writing that key, or move the "
        "trusted assignment after the spread."
    )


def test_absent_session_metadata_classifies_as_system():
    """Premise 2. Contexts built without metadata must fail CLOSED."""
    from prometheus.permissions.checker import ORIGIN_SYSTEM, origin_from_session_id

    assert origin_from_session_id(({} or {}).get("session_id")) == ORIGIN_SYSTEM
    assert origin_from_session_id(None) == ORIGIN_SYSTEM
