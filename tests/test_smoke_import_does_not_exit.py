"""#467 — importing smoke_test_tool_calling must not kill the interpreter.

The module wrapped its top-level imports in ``except ImportError: ... sys.exit(2)``.
Run as a script that is the correct loud exit. But three things IMPORT it
(two tests via import/spec_from_file_location, one script that a fourth test
imports), and ``SystemExit`` does not inherit from ``Exception`` — so pytest
does not attribute it to a file. It escapes the collector as INTERNALERROR and
a whole-suite run reports "N errors" with NO VERDICT, pointing at the sys.exit
rather than at the import that actually failed. Same class as #465.

The fix keeps ``sys.exit(2)`` for the script path (``__name__ == "__main__"``)
and re-raises the ImportError otherwise. These tests drive BOTH paths in real
subprocesses, because the defect is about what the interpreter does on import
and cannot be observed by importing the module into a healthy test process
(where the import succeeds and the handler never runs).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "smoke_test_tool_calling.py"


@pytest.fixture
def broken_prometheus(tmp_path: Path) -> Path:
    """A package dir where `prometheus` exists but `prometheus.__main__` does
    not define `load_config` — so `from prometheus.__main__ import load_config`
    raises ImportError, exactly the shape an out-of-sync worktree venv produces.

    Put FIRST on sys.path so it shadows the real package.
    """
    pkg = tmp_path / "shadow" / "prometheus"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "__main__.py").write_text(
        "# deliberately missing load_config / create_adapter / create_security_gate\n",
        encoding="utf-8",
    )
    # A telemetry package so the FIRST failing import is __main__, deterministically.
    return tmp_path / "shadow"


def _run(code: str, shadow: Path) -> subprocess.CompletedProcess:
    env_path = f"{shadow}{os.pathsep}{REPO / 'scripts'}"
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(env_path), "PATH": os.environ.get("PATH", "")},
        cwd=str(REPO),
        timeout=120,
    )


def test_importing_with_broken_prometheus_raises_ImportError_not_SystemExit(broken_prometheus):
    """The import path — what pytest does when it collects the three importers.
    Must surface a normal, attributable ImportError; must NOT exit the process."""
    proc = _run(
        """
        import sys
        sys.argv = ["importer"]   # not __main__ for the smoke module
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "smoke_test_tool_calling", %r
        )
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
            print("RESULT: imported-cleanly")
        except SystemExit as e:
            print("RESULT: SystemExit", e.code)
        except ImportError as e:
            print("RESULT: ImportError")
        """ % str(SCRIPT),
        broken_prometheus,
    )
    assert "RESULT: SystemExit" not in proc.stdout, (
        "importing the module called sys.exit() — this is the INTERNALERROR "
        "that turns a suite run into N errors with no verdict:\n"
        + proc.stdout + proc.stderr
    )
    assert "RESULT: ImportError" in proc.stdout, (
        "the import path must re-raise the ImportError so the importer sees an "
        "attributable collection error:\n" + proc.stdout + proc.stderr
    )


def test_running_as_script_with_broken_prometheus_exits_2(broken_prometheus):
    """The script path is UNCHANGED — still a loud exit(2) with the provenance
    report, because that is the correct behaviour when a human runs it."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        env={
            "PYTHONPATH": f"{broken_prometheus}{os.pathsep}{REPO / 'scripts'}",
            "PATH": os.environ.get("PATH", ""),
        },
        cwd=str(REPO),
        timeout=120,
    )
    # --help still parses args before imports? No — imports are at module top,
    # so a broken prometheus hits the handler first and exits 2.
    assert proc.returncode == 2, (
        "run as a script with a broken import, it must still exit 2 (the loud "
        "operator-facing failure). stdout:\n" + proc.stdout + "\nstderr:\n" + proc.stderr
    )


def test_healthy_import_still_works():
    """Control: with the real prometheus importable, importing the module
    succeeds and exposes the names its importers rely on. Guards against a fix
    that broke the success path."""
    proc = _run(
        """
        import sys
        sys.path.insert(0, %r)
        import importlib.util
        spec = importlib.util.spec_from_file_location("smoke_test_tool_calling", %r)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ok = hasattr(mod, "SmokeTestRunner") and hasattr(mod, "TestResult")
        print("RESULT:", "ok" if ok else "missing-names")
        """ % (str(REPO / "src"), str(SCRIPT)),
        Path("/nonexistent-shadow"),  # no shadow; real package on path via src
    )
    assert "RESULT: ok" in proc.stdout, proc.stdout + proc.stderr
