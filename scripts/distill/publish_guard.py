#!/usr/bin/env python3
"""Publish guard: refuse to ship a Max-derived model or dataset (DISTILL-OPEN-QUESTIONS Q6b).

    python3 scripts/distill/publish_guard.py check PATH [PATH ...] [--json]

Exit 0 when every PATH may be published, 1 when any may not (the reasons go
to stderr), 2 on a usage error. A PATH may not be published when it is
Max-derived (trained on Max-derived rows, or its name or manifest carries the
``max-derived`` tag), holds Max-derived rows (a dataset's eval-tools set is
built from Max traces), or nothing shows where it came from (fail closed). A
directory is checked whole: its own manifest chain and everything inside it.
The rule is ``artifact_policy.py`` (:func:`assert_publishable`).

No step in this repo publishes model or dataset artifacts today. Any future
one (a Hugging Face upload such as ``push_to_hub`` / ``upload_folder`` /
``hf upload``, a ``gh release upload`` of a model, a copy to a public bucket)
must run this check, or call ``artifact_policy.assert_publishable``, on every
artifact it ships, and stop on refusal. There is no override flag.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _load_policy() -> Any:
    name = "distill_artifact_policy"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name("artifact_policy.py"))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


AP = _load_policy()


def cmd_check(paths: list[str], *, as_json: bool) -> int:
    results = []
    refused = 0
    for raw in paths:
        prov, reasons = AP.check_publishable(raw)
        refused += bool(reasons)
        results.append({"path": raw, "publishable": not reasons, "reasons": reasons,
                        "provenance": prov.as_dict()})
        if not as_json:
            if reasons:
                print(f"REFUSED: {raw}: not publishable ({AP.RULING})", file=sys.stderr)
                for r in reasons:
                    print(f"  - {r}", file=sys.stderr)
            else:
                print(f"ok: {raw}")
    if as_json:
        print(json.dumps({"ruling": AP.RULING, "results": results}, indent=1))
    return 1 if refused else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="publish_guard", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ck = sub.add_parser("check", help="refuse unless every path may be published")
    ck.add_argument("paths", nargs="+", metavar="PATH")
    ck.add_argument("--json", action="store_true", help="print the verdicts as JSON")
    a = ap.parse_args(argv)
    return cmd_check(a.paths, as_json=a.json)


if __name__ == "__main__":
    sys.exit(main())
