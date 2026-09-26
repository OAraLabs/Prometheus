"""The ladder's acceptance runner — run as a SCRIPT, never imported by a task.

    <python> -I accept.py <workspace> <result.json> <test_module> [...]

A task's acceptance command is ``{unittest} <modules>``; the harness expands
``{unittest}`` to the line above. Three properties make an exit code of 0
mean nothing on its own, so this runner does not rely on one:

* ``-I`` keeps the working directory, ``PYTHON*`` variables and the user site
  off ``sys.path``, and the workspace is APPENDED after the standard library
  has been imported. A ``unittest.py`` (or ``json.py``, or a ``unittest/``
  package) the agent leaves in the workspace cannot replace the real one.
* Loading and running happen inside ``except BaseException``: code under test
  that calls ``sys.exit(0)`` at import time — a module-level ``main()`` with no
  ``__main__`` guard is the classic — is a failure, not a pass.
* The verdict is the JSON written to ``result.json`` (outside the workspace),
  and the harness requires at least one test to have RUN and none to have
  failed. Code that kills the interpreter outright leaves no result file,
  which the harness reads as a failure too.

Standard library only: this file runs under ``-I``, where the prometheus
package may not be importable.
"""

import json
import sys
import unittest


def main(argv):
    workspace, result_path, modules = argv[1], argv[2], argv[3:]
    out = {"tests_run": 0, "failures": 0, "errors": 0, "skipped": 0,
           "ok": False, "error": None}
    if not modules:
        out["error"] = "no test modules named"
    else:
        sys.path.append(workspace)
        try:
            suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
            result = unittest.TextTestRunner(stream=sys.stderr, verbosity=1).run(suite)
            out.update(
                tests_run=result.testsRun,
                failures=len(result.failures),
                errors=len(result.errors),
                skipped=len(result.skipped),
                ok=result.wasSuccessful(),
            )
        except BaseException as exc:  # noqa: BLE001 — SystemExit at import is a failure
            out["ok"] = False
            out["error"] = f"{type(exc).__name__}: {exc}"
    with open(result_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    return 0 if out["ok"] and out["tests_run"] - out["skipped"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
