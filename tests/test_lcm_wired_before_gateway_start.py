"""The LCM engine must be wired BEFORE any gateway adapter starts.

Observed live in production 2026-08-29: a Telegram update was PENDING at boot
(start_polling runs with drop_pending_updates=False, so it is fetched the
instant the adapter starts) and #326's block=False data-plane handlers let the
turn detach immediately. Journal receipts: adapter "started (polling)"
16:13:14.033 → turn start 16:13:14.116 → "LCM engine initialised" 16:13:14.533.
The turn's ChatSession was created via get_or_create with
_effective_lcm_engine() → None, so the whole conversation ran unpersisted —
the model replied fine, zero rows landed in lcm.db, and the session-title hook
no-op'd (store=None). Persistence is best-effort by design, so the operator
saw a working bot and quietly lost the conversation.

daemon.py's old comment claimed "sessions are created lazily by the gateway on
first message, which happens after this point, so this ordering is safe" —
a wiring invariant asserted only in prose. This file pins it mechanically, the
same way tests/test_fallback_is_actually_wired.py pins its producer sites:
statically over the daemon source, so a refactor that drifts the LCM block
back below an adapter start fails a test instead of a live conversation.
"""

from __future__ import annotations

import ast
from pathlib import Path

DAEMON = Path(__file__).resolve().parent.parent / "src" / "prometheus" / "daemon.py"


def _run_daemon_body() -> ast.AsyncFunctionDef:
    tree = ast.parse(DAEMON.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run_daemon":
            return node
    raise AssertionError("run_daemon not found in daemon.py — did the entrypoint move?")


def _attr_assign_lines(fn: ast.AsyncFunctionDef, attr: str) -> list[int]:
    """Lines of `<something>.<attr> = lcm_engine` assignments."""
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Attribute) and tgt.attr == attr:
                    if isinstance(node.value, ast.Name) and node.value.id == "lcm_engine":
                        out.append(node.lineno)
    return out


def _registered_adapter_names(fn: ast.AsyncFunctionDef) -> set[str]:
    """Names passed to gateway_registry.register_adapter(...) — every gateway."""
    names: set[str] = set()
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "register_adapter"
            and node.args
            and isinstance(node.args[0], ast.Name)
        ):
            names.add(node.args[0].id)
    return names


def _adapter_start_lines(fn: ast.AsyncFunctionDef, adapter_names: set[str]) -> dict[str, int]:
    """Line of `await <adapter>.start()` for each registered adapter name."""
    starts: dict[str, int] = {}
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Await)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "start"
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id in adapter_names
        ):
            starts[node.value.func.value.id] = node.lineno
    return starts


def test_lcm_attach_precedes_every_gateway_adapter_start():
    fn = _run_daemon_body()

    session_attach = _attr_assign_lines(fn, "lcm_engine")
    assert session_attach, (
        "no `<x>.lcm_engine = lcm_engine` assignment found in run_daemon — the "
        "session manager and agent loop would run every turn unpersisted"
    )
    # Both session_manager.lcm_engine and agent_loop.lcm_engine must be wired.
    assert len(session_attach) >= 2, (
        f"expected both session_manager.lcm_engine and agent_loop.lcm_engine "
        f"assignments, found {len(session_attach)} at lines {session_attach}"
    )
    last_attach = max(session_attach)

    adapter_names = _registered_adapter_names(fn)
    assert adapter_names, (
        "no gateway_registry.register_adapter(<name>) calls found — the sweep "
        "below would vacuously pass; did the registration idiom change?"
    )
    starts = _adapter_start_lines(fn, adapter_names)
    assert starts, (
        f"registered adapters {sorted(adapter_names)} but found no "
        f"`await <adapter>.start()` calls — did the start idiom change?"
    )

    late = {name: line for name, line in starts.items() if line < last_attach}
    assert not late, (
        f"gateway adapter(s) start BEFORE the LCM engine is attached "
        f"(attach completes at line {last_attach}): {late}. A message pending "
        f"at boot is fetched the instant the adapter starts and its "
        f"ChatSession is created with lcm_engine=None — the whole turn runs "
        f"unpersisted while the bot looks healthy (observed live 2026-08-29)."
    )


def test_lcm_engine_constructed_before_every_gateway_adapter_start():
    """The construction site, not just the attach: LCMEngine(...) must also
    precede adapter start, or the attach the previous test checks would be
    assigning a name that is still None."""
    fn = _run_daemon_body()

    ctor_lines = [
        node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and (getattr(node.func, "id", None) or getattr(node.func, "attr", None)) == "LCMEngine"
    ]
    assert ctor_lines, "no LCMEngine(...) construction found in run_daemon"

    adapter_names = _registered_adapter_names(fn)
    starts = _adapter_start_lines(fn, adapter_names)
    late = {name: line for name, line in starts.items() if line < min(ctor_lines)}
    assert not late, (
        f"LCMEngine is constructed at line {min(ctor_lines)}, after these "
        f"adapters already started: {late}"
    )


def test_session_created_at_adapter_start_time_carries_the_engine():
    """The behavioural face of the same invariant: a gateway turn arriving the
    instant an adapter starts calls get_or_create on a manager whose
    lcm_engine is already wired, and the resulting ChatSession must carry it.
    """
    from prometheus.engine.session import SessionManager

    manager = SessionManager()
    engine = object()  # any non-None engine handle; the manager only stores it
    manager.lcm_engine = engine  # daemon wiring, now BEFORE adapter start

    session = manager.get_or_create("telegram:8139235390")
    assert session.lcm_engine is engine, (
        "a session created by a gateway turn at adapter-start time must carry "
        "the wired LCM engine — a None here is exactly the silent-loss race"
    )
