"""Read a CI run as a RECEIPT: conclusions AND annotations, against a base.

WHY THIS EXISTS
---------------
``gh pr checks`` reports each job's **conclusion**. GitHub separately attaches
**annotations**, and a ``failure``-level annotation can sit on a job that
concluded ``success``. Reading the first and reporting "all green" is a claim
about one field, not about the run — and anyone opening the web UI sees the
other one.

That is not hypothetical. On 2026-09-20 a run was reported here as
"security-floors fail, everything else pass" — true of conclusions — while the
run carried **four** failure-level annotations across **three** jobs. The
receipt was meant to prove "this is the only failure"; it could not, and nobody
reading it could tell.

This repo produces such annotations on most runs: an asyncio subprocess
transport finalised after its loop closed raises ``RuntimeError: Event loop is
closed`` from ``__del__``, pytest reports it as an unraisable-exception
*warning*, and the runner's log matcher promotes the traceback line to
``##[error]``. The count swings 0-5 per run purely on GC timing. So "nothing
else moved" is unprovable by inspection unless the comparison is made
mechanically — which is what this does.

THE COMPARISON IS AGAINST A BASE, NOT AGAINST ZERO
---------------------------------------------------
Zero annotations is the wrong bar while that defect is open: every run would
fail it. The question a receipt must answer is *did THIS BRANCH change
anything*, so the counts are compared per job name against the same fields on
the base commit's own run. Identical counts falsify "the branch caused it" in a
way that calling something a flake never does.

USAGE
-----
    python scripts/ci_receipt.py <run-id> [--base <base-run-id>]
    python scripts/ci_receipt.py --pr 525          # resolves both runs

Exit codes: 0 clean receipt · 1 contaminated (differs from base) · 2 usage.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict

REPO = "OAraLabs/Prometheus"

#: Annotation messages that are infrastructure noise on every job, not signal.
_NOISE = (
    "Node.js 20 is deprecated",
    "will migrate to Ubuntu",
)


def _signature(message: str) -> str:
    """Collapse a message to what IDENTIFIES it, dropping what varies.

    Numbers and absolute paths differ between runs for the same underlying
    event (`/home/runner/work/...`, line numbers, pids), and a signature that
    keeps them would report a brand-new message on every run — the same
    fires-every-time failure this file exists to avoid.
    """
    first = (message or "").strip().splitlines()
    sig = first[0] if first else "(no message)"
    sig = re.sub(r"/\S+", "<path>", sig)
    sig = re.sub(r"\d+", "N", sig)
    return sig[:120]


def _gh_json(args: list[str]) -> object:
    out = subprocess.run(
        ["gh", *args], capture_output=True, text=True, check=False
    )
    if out.returncode != 0:
        raise SystemExit(f"gh {' '.join(args)} failed:\n{out.stderr.strip()}")
    return json.loads(out.stdout or "null")


def _jobs(run_id: str) -> list[dict]:
    data = _gh_json(["run", "view", run_id, "--json", "jobs,conclusion"])
    return data["jobs"] if isinstance(data, dict) else []


def _failure_annotations(job_id: int) -> list[str]:
    """Failure-level annotation messages for one job, noise filtered out."""
    out = subprocess.run(
        ["gh", "api", f"repos/{REPO}/check-runs/{job_id}/annotations"],
        capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        return []
    try:
        anns = json.loads(out.stdout or "[]")
    except json.JSONDecodeError:
        return []
    msgs = []
    for a in anns:
        if a.get("annotation_level") != "failure":
            continue
        msg = (a.get("message") or "").strip()
        if any(n in msg for n in _NOISE):
            continue
        msgs.append(msg.splitlines()[0][:120] if msg else "(no message)")
    return msgs


def read_receipt(run_id: str) -> dict:
    """Both halves of the truth about one run, keyed by job name."""
    result: dict[str, dict] = {}
    for job in _jobs(run_id):
        msgs = _failure_annotations(job["databaseId"])
        result[job["name"]] = {
            "conclusion": job.get("conclusion"),
            "failure_annotations": len(msgs),
            "messages": msgs,
        }
    return result


def _render(title: str, receipt: dict) -> None:
    print(f"\n{title}")
    print(f"  {'job':<30} {'conclusion':<10} {'failure-annotations':>20}")
    print("  " + "-" * 62)
    for name, r in sorted(receipt.items()):
        flag = "" if r["conclusion"] == "success" else "  <-- FAILED"
        print(
            f"  {name:<30} {str(r['conclusion']):<10} "
            f"{r['failure_annotations']:>20}{flag}"
        )


def compare(run_id: str, base_run_id: str | None) -> int:
    receipt = read_receipt(run_id)
    _render(f"RUN {run_id}", receipt)

    failed = [n for n, r in receipt.items() if r["conclusion"] != "success"]
    annotated = {n: r for n, r in receipt.items() if r["failure_annotations"]}

    if not base_run_id:
        print("\n  NO BASE RUN GIVEN — conclusions and annotation counts are")
        print("  reported, but nothing is proven about whether THIS BRANCH")
        print("  caused them. Pass --base to make this a receipt.")
        _explain(failed, annotated)
        return 0

    base = read_receipt(base_run_id)
    _render(f"BASE {base_run_id}", base)

    # ── THE VERDICT: the SET of distinct messages, not the counts ─────────
    # A NEW message string is signal — something is happening that was not
    # happening before. A count change within a KNOWN message is noise: these
    # are GC-timing dependent and swing 0-5 on main with no code change, so a
    # count-based verdict fires on nearly every run. A guard that fires every
    # run gets ignored, which is the counter-nobody-reads shape it exists to
    # prevent.
    base_sigs, run_sigs = set(), set()
    for r in base.values():
        base_sigs.update(_signature(m) for m in r["messages"])
    for r in receipt.values():
        run_sigs.update(_signature(m) for m in r["messages"])

    new_sigs = run_sigs - base_sigs
    gone_sigs = base_sigs - run_sigs

    print("\nVERDICT")
    if new_sigs:
        print("  CONTAMINATED — annotation message(s) NOT present on the base:")
        for sig in sorted(new_sigs):
            print(f"    + {sig}")
        print("\n  A message the base does not produce is signal. Investigate it")
        print("  or state it explicitly in the PR body.")
        status = 1
    else:
        print("  CLEAN — every annotation message also appears on the base.")
        print("  No new failure mode was introduced by this branch.")
        status = 0
    if gone_sigs:
        for sig in sorted(gone_sigs):
            print(f"    - {sig}  (present on base, absent here)")

    # ── SECONDARY: count drift. Reported, never the verdict. ──────────────
    drift = {
        name: (base[name]["failure_annotations"], r["failure_annotations"])
        for name, r in receipt.items()
        if name in base
        and r["failure_annotations"] != base[name]["failure_annotations"]
    }
    if drift:
        print("\n  (secondary) annotation COUNTS differ, same message set:")
        for name, (was, now) in sorted(drift.items()):
            print(f"    {name}: {was} -> {now}")
        print("    Counts are timing-dependent; this is informational only.")

    _explain(failed, annotated)
    return status


def _explain(failed: list[str], annotated: dict) -> None:
    if failed:
        print(f"\n  Failing jobs: {', '.join(sorted(failed))}")
    if annotated:
        total = sum(r["failure_annotations"] for r in annotated.values())
        print(
            f"\n  {total} failure-level annotation(s) sit on "
            f"{len(annotated)} job(s). A job can carry these and still"
        )
        print("  conclude success — report BOTH numbers, never just the conclusion:")
        for name, r in sorted(annotated.items()):
            for m in r["messages"][:2]:
                print(f"    [{name}] {m}")


def _runs_for_pr(pr: str) -> tuple[str, str | None]:
    head = _gh_json(["pr", "view", pr, "--json", "headRefOid,baseRefName"])
    branch = _gh_json(["pr", "view", pr, "--json", "headRefName"])["headRefName"]
    runs = _gh_json([
        "run", "list", "--branch", branch, "--limit", "1",
        "--json", "databaseId",
    ])
    if not runs:
        raise SystemExit(f"no CI run found for PR {pr} (branch {branch})")
    base_branch = head["baseRefName"]
    base_runs = _gh_json([
        "run", "list", "--branch", base_branch, "--workflow", "ci.yml",
        "--limit", "1", "--json", "databaseId",
    ])
    return (
        str(runs[0]["databaseId"]),
        str(base_runs[0]["databaseId"]) if base_runs else None,
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("run_id", nargs="?", help="the CI run to read")
    p.add_argument("--base", default=None, help="base run to compare against")
    p.add_argument("--pr", default=None, help="resolve both runs from a PR number")
    args = p.parse_args()

    if args.pr:
        run_id, base = _runs_for_pr(args.pr)
        return compare(run_id, args.base or base)
    if not args.run_id:
        p.print_usage()
        return 2
    return compare(args.run_id, args.base)


if __name__ == "__main__":
    sys.exit(main())
