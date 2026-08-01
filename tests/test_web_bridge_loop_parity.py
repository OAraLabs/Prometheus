"""The web-bridge LoopContext must not drift from the AgentLoop one.

``run_daemon()`` builds the agent loop TWICE:

* ``AgentLoop(...)`` (daemon.py ~421) — telegram / CLI / bakeoff paths, which
  threads config through ``AgentLoop.run_async``.
* ``LoopContext(...)`` (daemon.py ~1325) — a PRE-BUILT context handed to the
  web launcher, used by every web / Beacon / Bridge turn. It bypasses
  ``run_async`` entirely, so anything not passed here falls back to the
  dataclass default.

That asymmetry has now silently broken five separate features:

1. the context compactor (Sprint 2) — compaction.enabled=true did nothing on web
2. passive memory recall — web turns never recalled
3. the iteration caps — Beacon was pinned to the LOCAL 25 even on cloud models,
   the same bug #124 fixed at the *other* call site
4. ``tool_loader`` — deferred loading (#121) was inert on the web path, and
   ``GET /api/tools/deferred`` could only answer "daemon loop not wired"
5. ``post_result_hooks`` — with ``lsp.enabled: true`` live, telegram/CLI turns
   got LSP diagnostics appended to every write and Beacon, which does the most
   file writing of any surface, got none

Each was invisible because the fallback is a plausible default, not an error.
This test makes the drift itself fail, so the sixth one gets caught here.
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
# ``post_result_hooks`` USED to be here as unverified drift; it was the fifth
# instance of the same bug (``lsp.enabled: true`` is live, so telegram/CLI got
# LSP diagnostics appended to every write and Beacon — the primary coding
# surface — never did) and is now passed on both paths.
#
# ``file_mutation_verifier`` stays out, verified, for two reasons:
#
# 1. SHARED MUTABLE STATE. Unlike every other field here, it is not config —
#    it is a per-turn accumulator, and ``run_daemon`` builds exactly ONE
#    instance. ``_TurnRecord.mutations`` is a flat list and ``post_turn()``
#    drains AND resets it globally, with no session or turn key. This
#    LoopContext is a single object shared by every Beacon session and every
#    concurrent turn (see ws_server._run_agent, which threads ``mode`` and
#    ``session_id`` as per-call args precisely so concurrent turns cannot
#    cross-talk through it). Wiring the verifier here would hand concurrent
#    turns one accumulator: the turn that finishes first reports the OTHER
#    turn's file writes as its own and the second reports nothing — which
#    inverts the feature, whose entire job is checking that the writes YOU
#    claimed actually landed. Proven below in
#    ``test_the_verifier_is_a_turn_global_accumulator``.
#
# 2. IT WRITES A SYNTHETIC USER MESSAGE INTO THE TRANSCRIPT. The summary is
#    appended with ``ConversationMessage.from_user_text`` (provenance="user",
#    is_trusted=True — not ``from_injected``), so ``persist_loop_result``
#    stores it in LCM and ``GET /api/sessions/{id}/messages`` returns it as
#    ``role: "user"`` with nothing to distinguish it from something the user
#    typed. On telegram that message is model-facing only (the gateway renders
#    just the assistant reply), which is why the omission went unnoticed; on
#    Beacon it would surface as a chat bubble Will never wrote. The
#    ``show_in_telegram`` config knob that was meant to govern this is defined
#    in hooks/file_mutation_verifier.py and never read by anything.
#
# Both are fixable — scope the accumulator per turn, and emit the summary via
# ``from_injected`` so the UI can filter it — but that is a change to the
# verifier and to the AgentLoop path too (telegram/CLI/cron already share the
# one instance), not a line at this call site.
KNOWN_UNVERIFIED_DRIFT = {"file_mutation_verifier"}


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


def test_the_verifier_is_a_turn_global_accumulator():
    """Evidence for reason (1) of the file_mutation_verifier carve-out.

    One instance, no session/turn key: mutations from separate tool-call flows
    pile into the same list, and the first ``post_turn()`` drains all of them
    and leaves the next caller with nothing. Harmless while ONE loop owns the
    instance; wrong the moment the shared web LoopContext hands it to every
    concurrent Beacon turn.

    If this ever fails, the verifier has been made turn-scoped — revisit the
    carve-out above and wire it at the web bridge.
    """
    from prometheus.hooks.file_mutation_verifier import FileMutationVerifier

    v = FileMutationVerifier()
    # Two independent "turns", interleaved the way concurrent turns would be.
    v.pre_tool_use("write_file", {"path": "/tmp/turn-a.txt"}, "call-a")
    v.pre_tool_use("write_file", {"path": "/tmp/turn-b.txt"}, "call-b")
    v.post_tool_use("write_file", {"path": "/tmp/turn-a.txt"}, "call-a")
    v.post_tool_use("write_file", {"path": "/tmp/turn-b.txt"}, "call-b")

    first = v.post_turn()
    assert first is not None
    assert "/tmp/turn-a.txt" in first and "/tmp/turn-b.txt" in first, (
        "both flows landed in ONE summary — the accumulator has no turn key"
    )
    assert v.post_turn() is None, (
        "the first drain reset the shared state — a concurrent second turn "
        "would report no mutations at all"
    )


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
