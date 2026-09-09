#!/usr/bin/env python3
"""Type gate — mypy on everything that already passes, and only that.

WHY A SCRIPT AND NOT A FLAG
---------------------------
Turning mypy on wholesale is not an option: 447 errors across 122 modules on
the day this landed. A gate that fails on day one is a gate someone disables,
so the useful shape is a RATCHET rather than a cliff.

`mypy-debt.txt` lists the modules that do not pass yet. This runs mypy ONCE
over all of src/prometheus and partitions the output:

  * an error in a module NOT on the list  -> FAIL. New and already-clean code
    is gated by default, so nobody has to remember to opt a file in.
  * a module ON the list with zero errors -> FAIL, asking for the line to be
    deleted. Without this half the list only ever grows stale, which is how a
    debt register quietly becomes an allowlist.

One mypy invocation answers both questions, so the gate costs one run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "prometheus"
DEBT_FILE = REPO / "mypy-debt.txt"


def _read_debt() -> set[str]:
    out: set[str] = set()
    for raw in DEBT_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            out.add(line)
    return out


def main() -> int:
    debt = _read_debt()

    proc = subprocess.run(
        [sys.executable, "-m", "mypy", str(SRC),
         "--ignore-missing-imports", "--no-error-summary", "--no-color-output"],
        capture_output=True, text=True, cwd=REPO,
    )

    errors_by_file: dict[str, list[str]] = {}
    for line in proc.stdout.splitlines():
        if ": error:" not in line:
            continue
        path = line.split(":", 1)[0]
        try:
            rel = str(Path(path).resolve().relative_to(SRC))
        except ValueError:
            rel = path
        errors_by_file.setdefault(rel, []).append(line)

    # CROSS-CUTTING §8: a detector that broke must not fall through into the
    # permissive branch. mypy exits 0 with EMPTY stdout when it is genuinely
    # clean and 1 when it found errors — but `python -m mypy` with mypy absent
    # also exits 1 with empty stdout, and that must not read as "every module
    # is clean" (it would surface as 122 modules "graduating" at once, which
    # is a confident wrong answer rather than a loud one).
    if not proc.stdout.strip() and proc.returncode != 0:
        print(
            "TYPE GATE DID NOT RUN — mypy produced no output and exited "
            f"{proc.returncode}. This is UNKNOWN, not clean.\n"
            f"  stderr: {proc.stderr.strip()[:500] or '(empty)'}\n"
            "  Is mypy installed in this environment?",
            file=sys.stderr,
        )
        return 2
    if proc.returncode not in (0, 1):
        print("mypy exited abnormally:", proc.stderr.strip()[:800], file=sys.stderr)
        return 2

    regressions = {f: e for f, e in errors_by_file.items() if f not in debt}
    graduated = sorted(debt - set(errors_by_file))

    if regressions:
        print("TYPE GATE FAILED — errors in modules that are not on the debt list:\n")
        for f in sorted(regressions):
            for line in regressions[f][:10]:
                print(" ", line)
        print(
            f"\n{len(regressions)} module(s), {sum(map(len, regressions.values()))} error(s).\n"
            "These modules type-check today, so this is a regression rather than\n"
            "pre-existing debt. Fix them, or — if the module genuinely cannot be\n"
            "typed yet — say so in review before adding it to mypy-debt.txt."
        )

    if graduated:
        print("\nTYPE GATE FAILED — mypy-debt.txt is STALE:\n")
        for f in graduated:
            print("  now clean:", f)
        print(
            f"\n{len(graduated)} module(s) on the debt list have no errors.\n"
            "Delete those lines. The list only shrinks — leaving a fixed module\n"
            "registered as broken is how it stops meaning anything."
        )

    if regressions or graduated:
        return 1

    clean = len(list(SRC.rglob("*.py"))) - len(debt)
    print(f"type gate OK — {clean} module(s) clean, {len(debt)} on the debt list.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
