"""The web-bridge LoopContext must not drift from the AgentLoop one.

``run_daemon()`` builds the agent loop TWICE:

* ``AgentLoop(...)`` (daemon.py ~421) — telegram / CLI / bakeoff paths, which
  threads config through ``AgentLoop.run_async``.
* ``LoopContext(...)`` (daemon.py ~1325) — a PRE-BUILT context handed to the
  web launcher, used by every web / Beacon / Bridge turn. It bypasses
  ``run_async`` entirely, so anything not passed here falls back to the
  dataclass default.

That asymmetry has now silently broken four separate features:

1. the context compactor (Sprint 2) — compaction.enabled=true did nothing on web
2. passive memory recall — web turns never recalled
3. the iteration caps — Beacon was pinned to the LOCAL 25 even on cloud models,
   the same bug #124 fixed at the *other* call site
4. ``tool_loader`` — deferred loading (#121) was inert on the web path, and
   ``GET /api/tools/deferred`` could only answer "daemon loop not wired"

Each was invisible because the fallback is a plausible default, not an error.
This test makes the drift itself fail, so the fifth one gets caught here.
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
# These two are NOT verified-intentional — they are pre-existing drift found
# while fixing (3) and (4) above, carved out so this guard can land without
# silently changing web behaviour that hasn't been smoke-tested. Verifying and
# either wiring or documenting them is tracked as follow-up work.
KNOWN_UNVERIFIED_DRIFT = {"file_mutation_verifier", "post_result_hooks"}


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
