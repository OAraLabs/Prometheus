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
