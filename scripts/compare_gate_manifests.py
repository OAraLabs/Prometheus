#!/usr/bin/env python3
"""Compare two test runs — and REFUSE to compare runs that were not comparable.

WHY THIS EXISTS
---------------
"Diff the failure sets" was the baseline discipline for deciding whether a
change regressed anything. It turns out to be necessary and NOT sufficient: two
full-suite runs on one tree gave 7781 passed / 404 skipped and 7742 passed /
441 skipped, with the collected total constant at 8186. Nothing was added or
dropped — 39 tests changed category because the bwrap write-floor gate probes
live host state (the root filesystem's mount options) at collection time, and
that state differed between the two runs.

Diffing those two failure sets and reporting "no new failures" would have been
an unearned claim, and the dangerous direction: the security-floor tests had
SKIPPED in one run, so their absence from the failure set meant nothing about
whether they still passed.

THE RULE
--------
Same gate manifest FIRST, then same failure sets. When the manifests differ the
comparison is **VOID** — not passed, not failed, UNMEASURED — and this script
exits non-zero saying so, rather than producing a verdict it cannot support.

This is recurring-failures §4f one level up: don't force the value to be
stable, make the provenance visible so an incomparable comparison refuses
instead of lying.

USAGE
-----
    # record manifests while running each tree
    pytest tests/ --gate-manifest=/tmp/before.json
    pytest tests/ --gate-manifest=/tmp/after.json

    # compare
    python3 scripts/compare_gate_manifests.py /tmp/before.json /tmp/after.json

Exit codes: 0 = manifests match, comparison is admissible. 1 = manifests
differ, comparison is VOID. 2 = could not read/parse the inputs.

The gate manifest is produced by tests/support/gate_manifest.py; this script
only consumes it, so the gate set is defined in exactly one place.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VOID_EXIT = 1
IO_EXIT = 2


def _load(path: str) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read manifest {path}: {exc}", file=sys.stderr)
        raise SystemExit(IO_EXIT) from exc
    if not isinstance(data, dict) or "gates" not in data or "hash" not in data:
        print(
            f"{path} is not a gate manifest (expected 'hash' and 'gates' keys). "
            f"Produce one with: pytest tests/ --gate-manifest={path}",
            file=sys.stderr,
        )
        raise SystemExit(IO_EXIT)
    return data


def compare(before_path: str, after_path: str) -> int:
    before = _load(before_path)
    after = _load(after_path)

    print(f"before: {before_path}  hash={before['hash']}  schema={before.get('schema')}")
    print(f"after : {after_path}  hash={after['hash']}  schema={after.get('schema')}")
    print()

    # Different schema versions collected different gate sets: not a difference
    # in host state, but a difference in what was MEASURED. Incomparable.
    if before.get("schema") != after.get("schema"):
        print(
            "VOID — the two runs used DIFFERENT manifest schemas "
            f"({before.get('schema')} vs {after.get('schema')}). They did not "
            "record the same set of gates, so no statement about the difference "
            "in their results is supported.",
            file=sys.stderr,
        )
        return VOID_EXIT

    b_gates = before["gates"]
    a_gates = after["gates"]

    only_before = sorted(set(b_gates) - set(a_gates))
    only_after = sorted(set(a_gates) - set(b_gates))
    if only_before or only_after:
        # Should not happen within one schema, but a manifest can be edited by
        # hand and silently trusting it is the failure this exists to prevent.
        print("VOID — the gate SETS differ despite matching schemas.", file=sys.stderr)
        if only_before:
            print(f"  only in before: {', '.join(only_before)}", file=sys.stderr)
        if only_after:
            print(f"  only in after : {', '.join(only_after)}", file=sys.stderr)
        return VOID_EXIT

    changed = sorted(
        (name, b_gates[name]["value"], a_gates[name]["value"])
        for name in b_gates
        if b_gates[name]["value"] != a_gates[name]["value"]
    )

    if not changed:
        print(
            f"MATCH — all {len(b_gates)} gates agree (hash {before['hash']}).\n"
            "\n"
            "These two runs measured the same thing, so their failure sets ARE\n"
            "comparable: diff them, and a new failure is a regression.\n"
        )
        return 0

    print("VOID — the two runs were gated differently:", file=sys.stderr)
    for name, was, is_now in changed:
        print(f"  {name}: {was} → {is_now}", file=sys.stderr)
        for side, gates in (("before", b_gates), ("after", a_gates)):
            detail = gates[name].get("detail") or ""
            if detail:
                print(f"      {side}: {detail[:160]}", file=sys.stderr)
    print(
        "\n"
        "The comparison is VOID — not passed, not failed, UNMEASURED.\n"
        "\n"
        "A gate that flipped reclassifies every test it guards, so the two\n"
        "runs do not have the same composition and their failure sets cannot\n"
        "be diffed. Whatever changed between them is confounded by the gate:\n"
        "a test absent from one run's failures may have skipped, not passed.\n"
        "\n"
        "To get an admissible comparison, re-run both trees under the same\n"
        "host state. For the bwrap/AppArmor floor gates that means the same\n"
        "root-filesystem mount state (see root_mount_ro) — it is not a\n"
        "property of either tree, and it is not something to fix by\n"
        "stabilizing the probe: a floor test that ran where the floor cannot\n"
        "work would be worse than one that skips.\n",
        file=sys.stderr,
    )
    return VOID_EXIT


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("before", help="gate manifest from the baseline run")
    parser.add_argument("after", help="gate manifest from the changed run")
    args = parser.parse_args()
    return compare(args.before, args.after)


if __name__ == "__main__":
    raise SystemExit(main())
