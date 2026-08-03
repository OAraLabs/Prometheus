"""The web-bridge LoopContext must not drift from the AgentLoop one.

``run_daemon()`` builds the agent loop TWICE:

* ``AgentLoop(...)`` (daemon.py ~421) — telegram / CLI / bakeoff paths, which
  threads config through ``AgentLoop.run_async``.
* ``LoopContext(...)`` (daemon.py ~1325) — a PRE-BUILT context handed to the
  web launcher, used by every web / Beacon / Bridge turn. It bypasses
  ``run_async`` entirely, so anything not passed here falls back to the
  dataclass default.

That asymmetry has now silently broken six separate features:

1. the context compactor (Sprint 2) — compaction.enabled=true did nothing on web
2. passive memory recall — web turns never recalled
3. the iteration caps — Beacon was pinned to the LOCAL 25 even on cloud models,
   the same bug #124 fixed at the *other* call site
4. ``tool_loader`` — deferred loading (#121) was inert on the web path, and
   ``GET /api/tools/deferred`` could only answer "daemon loop not wired"
5. ``post_result_hooks`` — with ``lsp.enabled: true`` live, telegram/CLI turns
   got LSP diagnostics appended to every write and Beacon, which does the most
   file writing of any surface, got none
6. ``file_mutation_verifier`` — web turns got no claimed-vs-actual disk audit.
   Unlike 1-5 this one was NOT just a missing line: the hook held turn-global
   state and had to be made turn-scoped before the shared web context could
   safely hold it (see KNOWN_UNVERIFIED_DRIFT below for the full account).

Each was invisible because the fallback is a plausible default, not an error.
This test makes the drift itself fail, so the seventh one gets caught here.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from prometheus.engine.agent_loop import LoopContext

DAEMON = Path(__file__).resolve().parents[1] / "src" / "prometheus" / "daemon.py"

# Names the AgentLoop path passes that the web bridge intentionally does not,
# each of which must stay justified. Anything else showing up in the drift is a
# bug, not a style choice.
#
# The allowlist is EMPTY, and the goal is to keep it that way. Its last two
# entries were both real bugs wearing a "deliberate" label:
#
# * ``post_result_hooks`` — the fifth instance of the same wiring bug
#   (``lsp.enabled: true`` is live, so telegram/CLI got LSP diagnostics
#   appended to every write and Beacon, the primary coding surface, never did).
#   Fixed by passing it; see (5) above.
#
# * ``file_mutation_verifier`` — the one field here that was mutable STATE
#   rather than config, and the only carve-out that needed the hook itself
#   fixed before wiring was safe. ``run_daemon`` builds ONE verifier;
#   ``_TurnRecord.mutations`` was a flat list and ``post_turn()`` drained AND
#   reset it globally with no session or turn key, so concurrent turns shared
#   one accumulator — the first to finish reported the other's writes as its
#   own and the second reported nothing, inverting a feature whose whole job
#   is checking that the writes YOU claimed actually landed. (That was already
#   live for telegram-vs-cron; web was merely the path that would have made it
#   routine.) Its summary also went in via ``ConversationMessage.from_user_text``
#   — provenance="user", is_trusted=True — so LCM stored it as a turn the human
#   typed and ``GET /api/sessions/{id}/messages`` replayed it as ``role:
#   "user"``: a Beacon chat bubble nobody wrote, and user-provenance rows for
#   the MemoryExtractor to mine as facts. Both fixed: state is keyed by a
#   per-``run_loop`` turn key, and the summary is emitted with
#   ``provenance="file_mutation_verifier"`` (exposed on the REST route so a UI
#   can filter it). See tests/test_file_mutation_verifier.py.
#
# If you add an entry, give it the same treatment: a reason a reader can check,
# and a test that fails when the reason stops being true.
KNOWN_UNVERIFIED_DRIFT: set[str] = set()


def _kwargs_by_callee() -> dict[str, set[str]]:
    tree = ast.parse(DAEMON.read_text(encoding="utf-8"))
    out: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name in ("LoopContext", "AgentLoop"):
            out.setdefault(name, set()).update(k.arg for k in node.keywords if k.arg)
    return out


def _loop_context_fields() -> set[str]:
    return {f.name for f in dataclasses.fields(LoopContext)}


def test_both_constructions_still_exist():
    """If the daemon stops building two loops, this whole guard is moot and
    should be deleted rather than left passing vacuously."""
    kwargs = _kwargs_by_callee()
    assert "AgentLoop" in kwargs, "daemon.py no longer builds an AgentLoop"
    assert "LoopContext" in kwargs, "daemon.py no longer builds a web LoopContext"


def test_web_bridge_gets_the_iteration_caps():
    """Beacon turns were capped at the local 25 regardless of provider."""
    web = _kwargs_by_callee()["LoopContext"]
    assert "max_tool_iterations" in web
    assert "max_tool_iterations_cloud" in web, (
        "without this the web LoopContext keeps the None default and "
        "_effective_max_tool_iterations applies the LOCAL cap to cloud models"
    )


def test_web_bridge_gets_the_tool_loader():
    """Deferred loading (#121) is inert on the web path without it, and
    /api/tools/deferred cannot resolve the effective state."""
    assert "tool_loader" in _kwargs_by_callee()["LoopContext"]


def test_web_bridge_gets_the_post_result_hooks():
    """With `lsp.enabled: true` this list carries the LSPDiagnosticsHook, which
    appends type errors to write_file/edit_file results. Without it Beacon —
    the surface that does the most file writing — is the only path that never
    sees a diagnostic."""
    assert "post_result_hooks" in _kwargs_by_callee()["LoopContext"]


def test_web_bridge_gets_the_file_mutation_verifier():
    """The inverse of the test this replaces.

    ``test_the_verifier_is_a_turn_global_accumulator`` used to live here and
    asserted the BROKEN semantics on purpose: two interleaved flows landing in
    one summary, and the first drain leaving the second turn with nothing. That
    was the stated blocker for wiring the verifier on the shared web context,
    so the fix deletes it rather than updating it.

    What replaced it: ``run_loop`` mints a turn key per invocation and the
    verifier keys its state on it — proven in
    tests/test_file_mutation_verifier.py::TestTurnScoping, which asserts the
    isolation directly. This test only guards the wiring.
    """
    assert "file_mutation_verifier" in _kwargs_by_callee()["LoopContext"]


def test_no_new_drift_between_the_two_loops():
    """The actual guard: every LoopContext field the AgentLoop path configures
    must also be configured on the web path."""
    kwargs = _kwargs_by_callee()
    fields = _loop_context_fields()
    agent_side = kwargs["AgentLoop"] & fields
    web_side = kwargs["LoopContext"] & fields

    drift = agent_side - web_side - KNOWN_UNVERIFIED_DRIFT
    assert not drift, (
        f"these LoopContext fields are configured for telegram/CLI but NOT for "
        f"web/Beacon turns: {sorted(drift)}. Web turns will silently use the "
        f"dataclass default. Pass them at the web-bridge LoopContext in "
        f"daemon.py, or add them to KNOWN_UNVERIFIED_DRIFT with a reason."
    )


def test_carveouts_are_still_real():
    """Keep the allowlist honest — a stale entry would mask real drift."""
    kwargs = _kwargs_by_callee()
    fields = _loop_context_fields()
    still_drifting = (kwargs["AgentLoop"] & fields) - (kwargs["LoopContext"] & fields)
    stale = KNOWN_UNVERIFIED_DRIFT - still_drifting
    assert not stale, (
        f"{sorted(stale)} are in KNOWN_UNVERIFIED_DRIFT but no longer drift — "
        f"remove them from the allowlist"
    )
    assert KNOWN_UNVERIFIED_DRIFT <= fields, "allowlist names must be real fields"


def test_the_defaults_that_made_this_silent():
    """Documents WHY each omission was invisible: the fallback is a plausible
    value, so a misconfigured web turn looks like a working one."""
    defaults = {f.name: f.default for f in dataclasses.fields(LoopContext)}
    assert defaults["max_tool_iterations"] == 25
    assert defaults["max_tool_iterations_cloud"] is None
    assert defaults["tool_loader"] is None


# ---------------------------------------------------------------------------
# The ephemeral flag — the same trap, solved by NOT having a field
#
# ``/ephemeral on`` (prometheus.config.ephemeral) is per-SESSION, which makes it
# the seventh candidate for the drift above — except that passing it at both
# construction sites would be the WRONG fix, and the guards below pin the right
# one.
#
# The web LoopContext is built ONCE at daemon start and shared by every
# Beacon/Bridge session. A per-chat privacy flag stored on it would not merely
# drift; it would CROSS-TALK — one ephemeral chat suppressing persistence for
# every other web session, or an ordinary chat inheriting a stale False and
# persisting a turn the user had flagged. That is why ``ws_server._run_agent``
# already passes ``mode`` and ``session_id`` as per-call ``run_loop``
# arguments and explicitly never stores them on the shared context.
#
# So the flag is resolved INSIDE ``run_loop``, from the effective session id,
# once per turn. Both construction sites get it because there is nothing to
# pass — the CROSS-CUTTING §5 shape: prefer a property that cannot be violated
# over a check that must remember to run.
# ---------------------------------------------------------------------------

AGENT_LOOP = (
    Path(__file__).resolve().parents[1]
    / "src" / "prometheus" / "engine" / "agent_loop.py"
)

# The chain that carries the per-turn flag from run_loop down to the telemetry
# write. Every hop must keep the parameter: drop it anywhere and the tail of
# the chain silently falls back to ``ephemeral=False``, i.e. retention, which
# is the plausible-default failure this whole file exists to catch.
_EPHEMERAL_CHAIN = (
    "_run_loop",
    "_dispatch_tool_calls",
    "_safe_execute",
    "_execute_tool_call",
)


def _func_node(name: str) -> ast.AST:
    tree = ast.parse(AGENT_LOOP.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in agent_loop.py")


def _param_names(fn: ast.AST) -> set[str]:
    args = fn.args
    return {a.arg for a in (*args.args, *args.posonlyargs, *args.kwonlyargs)}


def test_ephemeral_is_not_a_loop_context_field():
    """The guard that keeps the fix correct rather than merely present.

    If ``ephemeral`` ever becomes a LoopContext field, the field-level parity
    test above starts demanding it be passed at both sites — and passing a
    per-session value on the ONE shared web context is precisely the
    cross-talk bug. Adding it here would make this file's other guard enforce
    the wrong thing, which is worse than no guard."""
    assert "ephemeral" not in _loop_context_fields(), (
        "ephemeral became a LoopContext field. The web context is built once "
        "and shared by every session, so a per-chat privacy flag parked on it "
        "leaks across chats. Resolve it per-call in run_loop instead."
    )


def test_run_loop_resolves_the_flag_from_the_effective_session_id():
    """``session_id or context.session_id`` — the per-call argument must win.

    Reading ``context.session_id`` alone would resolve every web turn against
    the shared context's id, so Beacon would get one answer for all sessions.
    Same idiom the file-mutation-verifier turn key and the router's override
    lookup already use, and for the same reason."""
    src = AGENT_LOOP.read_text(encoding="utf-8")
    assert "is_session_ephemeral(session_id or context.session_id)" in src, (
        "run_loop no longer resolves the ephemeral flag from the EFFECTIVE "
        "session id. On the web path context.session_id is not this turn's "
        "session — see ws_server._run_agent."
    )


@pytest.mark.parametrize("fn_name", _EPHEMERAL_CHAIN)
def test_the_ephemeral_flag_survives_every_hop_to_the_telemetry_write(fn_name):
    """Drop the parameter at any hop and the tail defaults to retention."""
    assert "ephemeral" in _param_names(_func_node(fn_name)), (
        f"{fn_name} no longer accepts `ephemeral`; everything downstream of it "
        f"falls back to ephemeral=False and starts writing content again"
    )


def test_run_async_gates_the_post_task_hooks_on_the_flag():
    """The hooks are the one suppression that CANNOT move into run_loop — they
    live in the AgentLoop wrapper by design (see RUN_ASYNC_ONLY in
    test_run_async_web_parity.py), so they need their own explicit check."""
    fn = _func_node("run_async")
    calls = {
        node.func.id
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "is_session_ephemeral" in calls, (
        "run_async no longer checks the ephemeral flag before firing post-task "
        "hooks — SkillCreator would again be handed the raw user message of an "
        "ephemeral turn and write it into skills/auto/ plus a skill_created "
        "signal payload"
    )
