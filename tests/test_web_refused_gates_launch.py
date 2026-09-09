"""The web bridge must not bind when the API token could not be bootstrapped.

P1.2 of the audit: ensure_api_token raising (an unwritable env-file dir) used
to log one ERROR and launch the bridge OPEN anyway. The fix sets ``_web_refused``
and wraps ``create_task(launch_web(...))`` so the port never binds in that case.

``run_daemon`` is a single 2000-line async function with no seam to call
directly, so — like ``test_web_bridge_loop_parity`` — this guards the WIRING
structurally via AST: the launch_web call must sit inside a branch gated on
``_web_refused``. A future edit that moves the launch back out of the guard
fails here instead of silently re-opening the door.

The DECISION logic itself (when _web_refused is True) is unit-tested in
test_env_file.py::TestRefuseToServeOnBootstrapFailure against the pure helper.
"""

from __future__ import annotations

import ast
from pathlib import Path

DAEMON = Path(__file__).resolve().parents[1] / "src" / "prometheus" / "daemon.py"


def _parse() -> ast.Module:
    return ast.parse(DAEMON.read_text(encoding="utf-8"))


def _find_launch_web(tree: ast.Module) -> ast.Call | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = getattr(fn, "id", None) or getattr(fn, "attr", None)
            if name == "launch_web":
                return node
    return None


def _set_parents(tree: ast.Module) -> None:
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child._parent = parent  # type: ignore[attr-defined]


def _gated_by_web_refused(target: ast.AST) -> bool:
    """True if any ancestor If tests `_web_refused` and `target` is in a branch
    of it (body or orelse). One tree, parent pointers — never cross-parse
    identity, which compares nodes from two separate parses and is always False.
    """
    node = target
    while hasattr(node, "_parent"):
        node = node._parent  # type: ignore[attr-defined]
        if isinstance(node, ast.If):
            test = node.test
            if any(
                isinstance(n, ast.Name) and n.id == "_web_refused"
                for n in ast.walk(test)
            ):
                return True
    return False


def test_launch_web_call_exists():
    """If launch_web is gone from the daemon, this guard is moot — delete it."""
    assert _find_launch_web(_parse()) is not None, (
        "daemon.py no longer calls launch_web; this wiring guard should be "
        "updated or removed rather than left passing vacuously"
    )


def test_launch_web_is_gated_by_web_refused():
    """The whole point: the bridge must not launch when the token bootstrap
    failed and nothing is configured."""
    tree = _parse()
    _set_parents(tree)
    call = _find_launch_web(tree)
    assert call is not None
    assert _gated_by_web_refused(call), (
        "create_task(launch_web(...)) is NOT inside an `if _web_refused` "
        "guard — a bootstrap failure would serve the control plane open again "
        "(the P1.2 regression). The launch must sit in the else/skip branch."
    )


def test_web_refused_comes_from_the_pure_helper():
    """The decision must be the tested helper, not a re-inlined boolean — a
    second copy of the condition is the next place for it to drift."""
    src = DAEMON.read_text(encoding="utf-8")
    assert "web_refused_on_bootstrap_failure(" in src, (
        "daemon must derive _web_refused from the unit-tested helper in "
        "config/api_token.py, not re-inline the condition"
    )
