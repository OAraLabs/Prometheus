"""A test module that needs `fastapi` must SKIP without it, not abort the run.

WHAT HAPPENED
-------------
CONTRIBUTING states that the web/WebSocket tests "call `pytest.importorskip(...)`,
so they are silently skipped" under `uv run pytest`, whose environment has no
`fastapi`. For eleven modules that was not true: they imported `fastapi` at
module scope with no guard.

A missing import at module scope is a COLLECTION error, and pytest aborts the
whole session on one. So the single command CONTRIBUTING gives every
contributor ran **zero tests**. Measured on main at 7a48060:

    66 skipped, 2 warnings, 11 errors in 6.54s
    !!!! Interrupted: 11 errors during collection !!!!

Not "the web tests were skipped" — nothing ran, and the exit code was 2.

WHY A GUARD AND NOT JUST A DOC FIX
----------------------------------
Correcting the prose would have described the breakage accurately and left the
documented command useless. The design in CONTRIBUTING is the right one — skip
what the environment cannot support, and say plainly that a green `uv` run has
not exercised the web surface. So the code was changed to match the doc, and
this test keeps them matched: the claim is now checkable rather than merely
written down.

WHAT THIS ASSERTS
-----------------
For every module under tests/: if it imports `fastapi`/`starlette`/`uvicorn`,
or anything under `prometheus.web`, at module scope, then a
`pytest.importorskip(...)` must appear BEFORE that import. Position matters —
a guard placed after the import it is meant to guard does nothing, which is a
mistake easy to make and invisible in review.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
SRC = TESTS.parent / "src"

# The third-party roots that live only in the `web` extra.
WEB_DISTS = ("fastapi", "starlette", "uvicorn", "multipart")


def _module_imports(tree: ast.Module) -> set[str]:
    """Module-scope imports that are NOT inside a try/except.

    Scope is the whole point. An import inside a function runs on call, and an
    import inside `try: ... except ImportError:` is already handled — neither
    can abort collection. Counting them over-reports: an earlier draft walked
    every node and claimed `prometheus.web.launcher` needs the extra, when it
    imports fastapi lazily and loads fine without it.
    """
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


def _prometheus_modules_needing_web() -> frozenset[str]:
    """First-party modules that cannot be imported without the `web` extra.

    DERIVED, NOT LISTED. An earlier draft of this guard hardcoded the prefix
    `prometheus.web`, which over-fired on sixteen `test_ws_*.py` files:
    `prometheus.web.ws_server` does NOT import fastapi and collects fine
    without the extra. A hand-maintained list would have been wrong on the day
    it was written and wrong again whenever an import moved — the same
    list-of-names failure the permission checker had.

    So the set is computed as a transitive closure over the real source: a
    module needs the extra if it imports one of WEB_DISTS, or imports another
    first-party module that does.
    """
    files: dict[str, Path] = {}
    for path in (SRC / "prometheus").rglob("*.py"):
        rel = path.relative_to(SRC).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts.pop()
        files[".".join(parts)] = path

    direct: dict[str, set[str]] = {}
    needs: set[str] = set()
    for mod, path in files.items():
        try:
            imported = _module_imports(ast.parse(path.read_text(encoding="utf-8")))
        except SyntaxError:  # pragma: no cover - would fail elsewhere first
            continue
        if any(i.split(".")[0] in WEB_DISTS for i in imported):
            needs.add(mod)
        direct[mod] = {i for i in imported if i.split(".")[0] == "prometheus"}

    # Propagate until stable — an importer of a needing module also needs it.
    changed = True
    while changed:
        changed = False
        for mod, deps in direct.items():
            if mod in needs:
                continue
            if any(d in needs or any(d.startswith(n + ".") for n in needs)
                   for d in deps):
                needs.add(mod)
                changed = True
    return frozenset(needs)


NEEDS_WEB: tuple[str, ...] = WEB_DISTS + tuple(sorted(_prometheus_modules_needing_web()))


def _needs_web(name: str) -> bool:
    """Exact match or a submodule of one — never a bare prefix.

    `prometheus.web` as a prefix would sweep in `prometheus.web.ws_server`,
    which does not need the extra.
    """
    return any(name == n or name.startswith(n + ".") for n in NEEDS_WEB)


def _module_level_nodes(tree: ast.Module):
    """Only module scope — an import inside a function or a try/except ImportError
    block is already deferred or already handled."""
    return tree.body


def _first_web_import_line(tree: ast.Module) -> tuple[int, str] | None:
    for node in _module_level_nodes(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if _needs_web(a.name):
                    return node.lineno, a.name
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if _needs_web(mod):
                return node.lineno, mod
    return None


def _importorskip_lines(tree: ast.Module) -> list[int]:
    lines: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "importorskip"
        ):
            lines.append(node.lineno)
    return lines


def _test_modules() -> list[Path]:
    return sorted(p for p in TESTS.glob("test_*.py"))


@pytest.mark.parametrize("path", _test_modules(), ids=lambda p: p.name)
def test_web_importing_module_skips_instead_of_erroring(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    web = _first_web_import_line(tree)
    if web is None:
        pytest.skip("does not import the web stack at module scope")

    line, name = web
    guards = _importorskip_lines(tree)
    assert guards, (
        f"{path.name} imports {name!r} at module scope (line {line}) with no "
        f"pytest.importorskip(). Without fastapi installed this is a COLLECTION "
        f"error, and pytest aborts the entire session on one — so this module "
        f"does not skip, it stops the whole suite from running. Add "
        f'`pytest.importorskip("fastapi")` above the import.'
    )
    assert min(guards) < line, (
        f"{path.name} calls pytest.importorskip() at line {min(guards)}, which "
        f"is AFTER it imports {name!r} at line {line}. The import raises first, "
        f"so the guard never executes and collection still aborts. Move it above."
    )


def test_the_guard_can_actually_fail(tmp_path):
    """Mutation check — the assertions above must reject the known-bad shapes.

    Without this, a broken AST walk that found no web imports anywhere would
    make every case above report `skipped` and look like a clean suite.
    """
    unguarded = ast.parse("import pytest\nfrom fastapi import FastAPI\n")
    assert _first_web_import_line(unguarded) == (2, "fastapi")
    assert _importorskip_lines(unguarded) == []

    guarded_too_late = ast.parse(
        "import pytest\nfrom fastapi import FastAPI\n"
        'pytest.importorskip("fastapi")\n'
    )
    line, _ = _first_web_import_line(guarded_too_late)
    assert min(_importorskip_lines(guarded_too_late)) > line, (
        "the ordering check cannot tell a guard placed after its import from "
        "one placed before it"
    )

    correct = ast.parse(
        "import pytest\n"
        'pytest.importorskip("fastapi")\n'
        "from fastapi import FastAPI\n"
    )
    line, _ = _first_web_import_line(correct)
    assert min(_importorskip_lines(correct)) < line

    # transitive: prometheus.web pulls fastapi in without naming it
    indirect = ast.parse("from prometheus.web.server import create_app\n")
    assert _first_web_import_line(indirect) == (1, "prometheus.web.server")
