"""Coding-mode tools — registration + side-effect tests (standing rule).

Every tool gets a registration test AND a side-effect output test (the
MemoryTool orphan must not recur). The big one for the sprint:
``code_run`` reports a FAILING command as is_error=False with the exit code
in the output — that is what keeps iterate-to-green out of the circuit
breaker's jaws.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from prometheus.coding.sandbox import ProcessSandbox
from prometheus.coding.tools import (
    CODING_TOOL_CLASSES,
    build_coding_registry,
)
from prometheus.tools.base import ToolExecutionContext


@pytest.fixture
def box(tmp_path: Path) -> ProcessSandbox:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "calc.py").write_text(
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "\n"
        "def sub(a, b):\n"
        "    return a - b\n"
    )
    (root / "README.md").write_text("# demo\n")
    return ProcessSandbox(root=root)


@pytest.fixture
def registry(box):
    return build_coding_registry(box)


def _ctx(box: ProcessSandbox) -> ToolExecutionContext:
    return ToolExecutionContext(cwd=box.root)


def _run(registry, box, name: str, **args):
    tool = registry.get(name)
    assert tool is not None, f"{name} not registered"
    parsed = tool.input_model(**args)
    return asyncio.run(tool.execute(parsed, _ctx(box)))


# --------------------------------------------------------------------------- #
# Registration — all six, schemas valid, session-scoped by design
# --------------------------------------------------------------------------- #


class TestRegistration:

    def test_all_six_tools_registered(self, registry):
        names = {t.name for t in registry.list_tools()}
        assert names == {
            "code_view", "code_str_replace", "code_create",
            "code_grep", "code_glob", "code_run",
        }
        assert len(CODING_TOOL_CLASSES) == 6

    def test_schemas_are_complete(self, registry):
        for schema in registry.to_api_schema():
            assert schema["name"].startswith("code_")
            assert schema["description"]
            assert schema["input_schema"]["properties"]

    def test_read_only_flags_are_truthful(self, registry):
        ro = {
            t.name: t.is_read_only(t.input_model(**(t.example_call or {"path": "x", "pattern": "x", "command": "x"})))
            for t in registry.list_tools()
        }
        assert ro["code_view"] is True
        assert ro["code_grep"] is True
        assert ro["code_glob"] is True
        assert ro["code_str_replace"] is False
        assert ro["code_create"] is False
        assert ro["code_run"] is False

    def test_not_in_global_builtin_registry(self):
        # Session-scoped by design — the daemon's global registry is built
        # from prometheus/tools/builtin/*; coding tools must never appear
        # there (their jail is per-task). Asserted at the package level so
        # the test doesn't drag the whole daemon module in.
        import prometheus.tools.builtin as builtin_pkg

        builtin_modules = {
            p.stem for p in Path(builtin_pkg.__path__[0]).glob("*.py")
        }
        assert not any(m.startswith("code_") for m in builtin_modules)


# --------------------------------------------------------------------------- #
# code_view
# --------------------------------------------------------------------------- #


class TestCodeView:

    def test_file_view_is_numbered(self, registry, box):
        r = _run(registry, box, "code_view", path="src/calc.py")
        assert not r.is_error
        assert "     1\tdef add(a, b):" in r.output

    def test_view_range(self, registry, box):
        r = _run(registry, box, "code_view", path="src/calc.py", view_range=[5, 6])
        assert "     5\tdef sub(a, b):" in r.output
        assert "def add" not in r.output

    def test_directory_listing(self, registry, box):
        r = _run(registry, box, "code_view", path="src")
        assert "calc.py" in r.output

    def test_missing_is_loud(self, registry, box):
        r = _run(registry, box, "code_view", path="src/nope.py")
        assert r.is_error
        assert "NOT FOUND" in r.output

    def test_escape_is_violation(self, registry, box):
        r = _run(registry, box, "code_view", path="../../etc/passwd")
        assert r.is_error
        assert "SANDBOX VIOLATION" in r.output


# --------------------------------------------------------------------------- #
# code_str_replace — exactly-once semantics, loud distinct errors
# --------------------------------------------------------------------------- #


class TestStrReplace:

    def test_unique_match_edits_file_on_disk(self, registry, box):
        r = _run(
            registry, box, "code_str_replace",
            path="src/calc.py",
            old_str="    return a + b",
            new_str="    return a + b + 0",
        )
        assert not r.is_error
        assert "OK — replaced 1 occurrence" in r.output
        # THE side effect: disk changed.
        assert "a + b + 0" in (box.root / "src" / "calc.py").read_text()

    def test_no_match_distinct_error_and_unchanged(self, registry, box):
        before = (box.root / "src" / "calc.py").read_text()
        r = _run(
            registry, box, "code_str_replace",
            path="src/calc.py", old_str="return a * b", new_str="x",
        )
        assert r.is_error
        assert "NO MATCH" in r.output
        assert "view the file" in r.output.lower() or "code_view" in r.output
        assert (box.root / "src" / "calc.py").read_text() == before

    def test_multi_match_distinct_error_and_unchanged(self, registry, box):
        before = (box.root / "src" / "calc.py").read_text()
        r = _run(
            registry, box, "code_str_replace",
            path="src/calc.py", old_str="(a, b):", new_str="(a, b, c):",
        )
        assert r.is_error
        assert "2 MATCHES" in r.output
        assert "unique" in r.output
        assert (box.root / "src" / "calc.py").read_text() == before

    def test_missing_file_is_loud(self, registry, box):
        r = _run(
            registry, box, "code_str_replace",
            path="src/ghost.py", old_str="x", new_str="y",
        )
        assert r.is_error
        assert "NOT FOUND" in r.output


# --------------------------------------------------------------------------- #
# code_create
# --------------------------------------------------------------------------- #


class TestCreate:

    def test_creates_file_with_parents(self, registry, box):
        r = _run(
            registry, box, "code_create",
            path="tests/unit/test_calc.py", content="def test_ok():\n    assert True\n",
        )
        assert not r.is_error
        assert (box.root / "tests" / "unit" / "test_calc.py").exists()

    def test_existing_file_refused(self, registry, box):
        before = (box.root / "README.md").read_text()
        r = _run(registry, box, "code_create", path="README.md", content="clobber")
        assert r.is_error
        assert "ALREADY EXISTS" in r.output
        assert (box.root / "README.md").read_text() == before

    def test_escape_is_violation(self, registry, box, tmp_path):
        r = _run(registry, box, "code_create", path="../evil.py", content="x")
        assert r.is_error
        assert "SANDBOX VIOLATION" in r.output
        assert not (tmp_path / "evil.py").exists()


# --------------------------------------------------------------------------- #
# code_grep / code_glob
# --------------------------------------------------------------------------- #


class TestSearch:

    def test_grep_finds_line(self, registry, box):
        r = _run(registry, box, "code_grep", pattern=r"def add\(")
        assert "src/calc.py:1:" in r.output

    def test_grep_respects_glob(self, registry, box):
        r = _run(registry, box, "code_grep", pattern="demo", path_glob="src/**/*.py")
        assert "NO MATCHES" in r.output

    def test_grep_bad_regex_is_loud(self, registry, box):
        r = _run(registry, box, "code_grep", pattern="(unclosed")
        assert r.is_error
        assert "BAD REGEX" in r.output

    def test_glob_lists_sorted(self, registry, box):
        r = _run(registry, box, "code_glob", pattern="**/*.py")
        assert r.output.splitlines() == ["src/calc.py"]

    def test_glob_no_match(self, registry, box):
        r = _run(registry, box, "code_glob", pattern="**/*.rs")
        assert "NO FILES" in r.output


# --------------------------------------------------------------------------- #
# code_run — THE iterate-to-green semantics
# --------------------------------------------------------------------------- #


class TestCodeRun:

    def test_success_reports_exit_zero(self, registry, box):
        r = _run(registry, box, "code_run", command="echo done")
        assert not r.is_error
        assert "exit code: 0" in r.output
        assert "done" in r.output
        assert r.metadata["exit_code"] == 0

    def test_failing_command_is_not_a_tool_error(self, registry, box):
        # The load-bearing assertion: red tests come back as REPORTS,
        # so the circuit breaker (3-identical/5-any ERRORS) never sees them.
        r = _run(registry, box, "code_run", command="echo boom >&2; exit 1")
        assert r.is_error is False
        assert "exit code: 1" in r.output
        assert "boom" in r.output
        assert r.metadata["exit_code"] == 1

    def test_timeout_is_a_tool_error(self, registry, box):
        r = _run(registry, box, "code_run", command="sleep 30", timeout_seconds=1.0)
        assert r.is_error
        assert "TIMED OUT" in r.output

    def test_runs_in_sandbox_root(self, registry, box):
        r = _run(registry, box, "code_run", command="cat README.md")
        assert "# demo" in r.output

    def test_loop_timeout_allowance_raised(self, registry):
        # code_run must outlive the loop's 300 s default tool timeout;
        # the sandbox enforces the real wall inside this allowance.
        tool = registry.get("code_run")
        assert tool.execution_timeout_seconds > 300.0
