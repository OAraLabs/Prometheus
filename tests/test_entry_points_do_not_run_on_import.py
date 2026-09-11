"""A module that does work when it is IMPORTED (`benchmarks/__main__.py`).

`python -m prometheus.benchmarks` works, so the file looks fine. It works by
accident: `-m` imports the module, and the work happens as an import side
effect rather than behind `if __name__ == "__main__"`. Anything that imports it
for any other reason runs the benchmark suite — a coverage sweep, a docs
generator, an editor's symbol indexer, `pkgutil.walk_packages`, an import audit.

`runner.main()` parses `sys.argv` with argparse, so under a test runner it reads
the RUNNER's arguments, finds them unrecognised, and calls `sys.exit(2)`. An
import that raises SystemExit is not a collection error — pytest reports
INTERNALERROR and the run produces no result at all. That is not hypothetical:
it happened in this repo during this change, from a different file with the same
shape (`scripts/smoke_test_tool_calling.py`, noted separately).

An AST scan of `src/prometheus` found exactly one module with an unguarded
top-level call. The scan is kept as the third test so the next one cannot be
added quietly.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import runpy
import sys

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "prometheus"


@pytest.fixture
def watched_main(monkeypatch):
    """Replace `runner.main` and record calls, without ever running benchmarks."""
    calls: list[tuple] = []
    runner = importlib.import_module("prometheus.benchmarks.runner")
    monkeypatch.setattr(runner, "main", lambda *a, **k: calls.append((a, k)))
    # Force a fresh import of the entry point so the module body actually runs.
    monkeypatch.delitem(sys.modules, "prometheus.benchmarks.__main__", raising=False)
    return calls


def test_importing_the_entry_point_does_not_run_the_benchmarks(watched_main):
    """THE DEFECT. Importing a module must not be the same as invoking it."""
    importlib.import_module("prometheus.benchmarks.__main__")
    assert watched_main == [], (
        "importing prometheus.benchmarks.__main__ ran the benchmark suite. "
        "Any tool that walks the package tree now runs benchmarks as a side "
        "effect of looking at it"
    )


def test_python_dash_m_still_runs_them(watched_main):
    """And the guard must not break the thing the file exists for.

    A fix that made the import inert by deleting the call would pass the test
    above and silently turn `python -m prometheus.benchmarks` into a no-op.
    """
    runpy.run_module("prometheus.benchmarks", run_name="__main__")
    assert len(watched_main) == 1, (
        "python -m prometheus.benchmarks no longer invokes the runner"
    )


def _unguarded_top_level_calls(path: pathlib.Path) -> list[str]:
    """Top-level expression-statement calls, ignoring anything inside an `if`.

    Deliberately narrow: decorators, assignments (`log = getLogger(...)`) and
    calls inside any `if` block are all fine. What is not fine is a bare call
    at module scope that does the module's actual work.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            fn = node.value.func
            name = getattr(fn, "id", None) or getattr(fn, "attr", None) or "<call>"
            out.append(f"{path.name}:{node.lineno} {name}()")
    return out


def test_no_entry_point_module_does_its_work_at_import_time():
    """The general guard, so the next one cannot be added quietly."""
    mains = sorted(SRC.rglob("__main__.py"))
    assert mains, "found no __main__.py at all — the scan is looking in the wrong place"
    offenders = [c for p in mains for c in _unguarded_top_level_calls(p)]
    assert offenders == [], (
        f"these entry points do their work when merely imported: {offenders}"
    )
