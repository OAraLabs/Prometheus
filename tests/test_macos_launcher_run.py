"""packaging/macos/launcher/main.swift ``--run``: what the launcher hands the bundled Python.

``--run`` (what launchd starts) changes to the person's home directory, because the daemon's tools start
there, and replaces itself with ``python -m prometheus daemon``. Under ``-m`` Python puts the WORKING
DIRECTORY first on ``sys.path``, so a ``~/secrets.py``, ``~/json.py`` or ``~/prometheus/`` would be imported
INSIDE the signed app in place of the real module: code nobody signed, running with everything the daemon
holds. An independent review found it; these tests keep it found.

The launcher passes ``-I`` (isolated: no working directory or script directory on ``sys.path``, no
``PYTHON*`` variables, no user site) and ``-B`` (no bytecode written into the signed bundle).

Swift is not compiled here (the build does that on a Mac), so the argv is read from the source; the second
test runs THIS interpreter with exactly those flags against a shadowing module, so the flags are proven to do
the job and not only to be present.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

LAUNCHER = Path(__file__).resolve().parents[1] / "packaging" / "macos" / "launcher" / "main.swift"


def run_arguments() -> list[str]:
    """The ``arguments`` array ``run()`` passes to execv: identifiers as written, string literals unquoted."""
    text = LAUNCHER.read_text(encoding="utf-8")
    body = text[text.index("func run() -> Never"):]
    body = body[: body.index("\n}\n")]
    match = re.search(r"let arguments = \[([^\]]*)\]", body)
    assert match, "run() no longer builds `let arguments = [...]`; update this test with it"
    items = [item.strip() for item in match.group(1).split(",")]
    return [item[1:-1] if item.startswith('"') and item.endswith('"') else item for item in items]


def interpreter_flags() -> list[str]:
    """The options between the interpreter and ``-m``."""
    arguments = run_arguments()
    return arguments[1: arguments.index("-m")]


def test_the_daemon_is_started_isolated_and_writes_no_bytecode():
    arguments = run_arguments()
    assert arguments[0] == "python", arguments
    assert arguments[arguments.index("-m") + 1] == "prometheus", arguments
    flags = interpreter_flags()
    isolated = "-I" in flags or {"-P", "-s", "-E"} <= set(flags)
    assert isolated, f"python is started without -I (or -P -s -E): {arguments}"
    assert "-B" in flags, f"python may write bytecode into the signed bundle: {arguments}"


def _shadow_imported(tmp_path: Path, flags: list[str]) -> bool:
    """Run ``python <flags> -m json.tool`` in a directory holding a ``json.py`` that leaves a mark."""
    marker = tmp_path / "shadow-ran"
    (tmp_path / "json.py").write_text(
        f"open({str(marker)!r}, 'w').close()\n", encoding="utf-8")
    env = {key: value for key, value in os.environ.items() if not key.startswith("PYTHON")}
    subprocess.run([sys.executable, *flags, "-m", "json.tool"], cwd=tmp_path, env=env, input="{}",
                   capture_output=True, text=True, timeout=60)
    return marker.exists()


def test_with_the_launchers_flags_a_module_in_the_working_directory_is_not_imported(tmp_path):
    control = tmp_path / "control"
    control.mkdir()
    assert _shadow_imported(control, []), "the control must show the shadowing, or this test proves nothing"
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    assert not _shadow_imported(isolated, interpreter_flags()), (
        f"{interpreter_flags()} still imports ./json.py from the working directory")
