"""THE INVARIANT: no bare integer literal is used as a context limit or token
budget anywhere in ``src/prometheus/web/``.

WHY THIS FILE EXISTS
--------------------
``create_app()`` had no resolved context budget in scope, so the one route
that needed one typed the old default. ``/api/lcm/{session_id}`` reported
``"limit": 24000`` four times over, on a daemon whose config said 72000 and
whose llama.cpp server reported 32768 — three numbers, none of them each
other.

The mislabelled denominator was the visible half. The damaging half was
``lcm_engine.assemble(session_id, token_budget=24000)``: the endpoint ran a
real assembly against a fabricated window, so ``total_tokens``,
``fresh_count``, ``summary_count`` and ``compression_ratio`` were all computed
against a context that does not exist. Beacon was not showing a stale view of
the real state — it was showing a parallel one. 9888/32768 (30%) rendered
as 41%.

This is the same failure class ``daemon.py`` documents having already fixed on
the agent-loop path ("a config effective_limit that outlived a model swap
silently won... n_ctx=32768 came to be budgeted at 72000"). Review did not
catch it there either; a passing build is what caught it. Hence a test, not a
convention.

WHAT IS AND IS NOT FLAGGED
--------------------------
Flagged: an int literal bound to a name that MEANS a context window
(``token_budget``, ``effective_limit``, ``n_ctx``, …) as a keyword argument, a
dict entry, or an assignment. Also flagged: a ``"limit"`` dict entry sitting
next to context-shaped siblings (``total_tokens``, ``compression_ratio``, …),
which is the exact shape of the bug.

Not flagged: pagination and feed limits (``recent(limit=100)``,
``_SEARCH_MAX_LIMIT = 50``). Those are genuinely constants of this layer and
have nothing to do with a model's context window. A guard that failed on them
would be turned off within a week, which is worse than no guard.

BOTH DIRECTIONS
---------------
A guard that flags nothing because its AST walk silently matches nothing looks
identical to a clean tree. ``test_detector_catches_the_original_bug`` replays
the four literals that shipped and asserts every one is caught, so a green
result from the real scan means the detector is looking, not blind.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import prometheus.web as web_pkg

WEB_ROOT = Path(web_pkg.__file__).resolve().parent

# Names that mean "a model's context window" or "how much of it may be spent".
# A bare integer bound to any of these is a fabricated budget by definition:
# the real value is resolved at runtime from the server or the config.
CONTEXT_NAMES = frozenset({
    "token_budget",
    "effective_limit",
    "context_limit",
    "detected_limit",
    "detected_context_size",
    "reserved_output",
    "max_context_tokens",
    "n_ctx",
    "cloud_default_limit",
})

# Siblings that identify a dict as a context-window payload, so a plain
# ``"limit"`` key inside it is a context limit rather than a page size.
CONTEXT_SIBLINGS = frozenset({
    "total_tokens",
    "compression_ratio",
    "fresh_count",
    "summary_count",
    "limit_source",
})


def _is_int_literal(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(
        node.value, bool
    )


class _ContextLiteralVisitor(ast.NodeVisitor):
    """Collect (line, name, value) for every bare int used as a budget."""

    def __init__(self) -> None:
        self.hits: list[tuple[int, str, int]] = []

    def visit_Call(self, node: ast.Call) -> None:
        for kw in node.keywords:
            if kw.arg in CONTEXT_NAMES and _is_int_literal(kw.value):
                self.hits.append((kw.value.lineno, f"{kw.arg}=", kw.value.value))
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        keys = {
            k.value for k in node.keys
            if isinstance(k, ast.Constant) and isinstance(k.value, str)
        }
        contextual = bool(keys & CONTEXT_SIBLINGS)
        for key, value in zip(node.keys, node.values):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                continue
            name = key.value
            flagged = name in CONTEXT_NAMES or (name == "limit" and contextual)
            if flagged and _is_int_literal(value):
                self.hits.append((value.lineno, f'"{name}":', value.value))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if _is_int_literal(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.lower() in CONTEXT_NAMES:
                    self.hits.append((node.lineno, f"{target.id} =", node.value.value))
                elif (
                    isinstance(target, ast.Attribute)
                    and target.attr.lower() in CONTEXT_NAMES
                ):
                    self.hits.append((node.lineno, f".{target.attr} =", node.value.value))
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if (
            node.value is not None
            and _is_int_literal(node.value)
            and isinstance(node.target, ast.Name)
            and node.target.id.lower() in CONTEXT_NAMES
        ):
            self.hits.append((node.lineno, f"{node.target.id} =", node.value.value))
        self.generic_visit(node)


def _scan(source: str) -> list[tuple[int, str, int]]:
    visitor = _ContextLiteralVisitor()
    visitor.visit(ast.parse(source))
    return visitor.hits


def _python_files() -> list[Path]:
    return sorted(
        p for p in WEB_ROOT.rglob("*.py") if "__pycache__" not in p.parts
    )


def test_web_package_is_actually_scanned() -> None:
    """The walk finds files. A guard over an empty set is not a guard."""
    files = _python_files()
    assert len(files) >= 3, f"expected the web package, found {files}"
    assert any(p.name == "server.py" for p in files)


@pytest.mark.parametrize(
    "path", _python_files(), ids=lambda p: p.name
)
def test_no_hardcoded_context_budget(path: Path) -> None:
    hits = _scan(path.read_text(encoding="utf-8"))
    assert not hits, (
        f"{path.relative_to(WEB_ROOT.parent.parent.parent)} hardcodes a context "
        f"budget: " + ", ".join(f"line {ln}: {name} {val}" for ln, name, val in hits)
        + ". Resolve it through prometheus.context.budget.resolve_effective_limit "
        "instead — a literal here is a window that exists nowhere in the system, "
        "and assemble() will build a context against it."
    )


def test_detector_catches_the_original_bug() -> None:
    """Replay the four literals that shipped; every one must be caught.

    Without this, a detector that silently matched nothing would produce the
    same green as a clean tree.
    """
    shipped = '''
def get_lcm_state(session_id, lcm_engine):
    if not lcm_engine:
        return {
            "session_id": session_id,
            "total_tokens": 0,
            "limit": 24000,
            "compression_ratio": 0,
            "fresh_count": 0,
            "summary_count": 0,
        }
    result = lcm_engine.assemble(session_id, token_budget=24000)
    return {
        "session_id": session_id,
        "total_tokens": result.total_tokens,
        "limit": 24000,
        "compression_ratio": result.compression_ratio,
        "fresh_count": 0,
        "summary_count": 0,
    }
'''
    hits = _scan(shipped)
    assert len(hits) == 3, f"expected all three literals, got {hits}"
    assert {v for _, _, v in hits} == {24000}


def test_detector_ignores_pagination_limits() -> None:
    """Page sizes are not context windows. Flagging them would kill the guard."""
    benign = '''
_SEARCH_MAX_LIMIT = 50
recent = signal_bus.recent(limit=100)
page = {"items": [], "limit": 20, "offset": 0}
'''
    assert _scan(benign) == []


def test_detector_catches_a_context_limit_assignment() -> None:
    """The other shape the same mistake takes: a module-level constant."""
    variant = '''
effective_limit = 72000
DEFAULTS = {"token_budget": 32768}
'''
    hits = _scan(variant)
    assert {v for _, _, v in hits} == {72000, 32768}
