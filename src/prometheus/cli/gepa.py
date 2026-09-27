"""``oara gepa`` — review what GEPA proposes. Nothing changes a skill until you promote.

GEPA stages proposals; it never writes ``skills/auto/``. These commands are
the explicit human step, and they only call the core in
:mod:`prometheus.learning.gepa_proposals` (and, for ``dry-run``,
:mod:`prometheus.learning.gepa`), so any other surface reuses the same rules.

    oara gepa proposals [--status pending|promoted|rejected]
    oara gepa show <id>
    oara gepa promote <id>
    oara gepa reject <id> [--reason TEXT]
    oara gepa dry-run [--json] [--telemetry-db P] [--lcm-db P] [--skills-dir P] [--proposals-dir P]

Exit codes: 0 done, 1 refused (the message says why), 2 usage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prometheus.learning.gepa_proposals import (
    STATUS_PENDING,
    STATUS_PROMOTED,
    STATUS_REJECTED,
    ProposalError,
    ProposalStore,
    render_list,
    render_promotion,
    render_proposal,
)


def add_gepa_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "gepa",
        help="Review GEPA's skill proposals (no skill changes until you promote one)",
    )
    # On every action, so `oara gepa show <id> --skills-dir …` reads naturally.
    dirs = argparse.ArgumentParser(add_help=False)
    dirs.add_argument("--proposals-dir", default=None,
                      help="Proposal staging directory (default ~/.prometheus/skills/proposals)")
    dirs.add_argument("--skills-dir", default=None,
                      help="Auto skills directory (default ~/.prometheus/skills/auto)")
    sub = p.add_subparsers(dest="gepa_action")

    lst = sub.add_parser("proposals", parents=[dirs], help="List proposals (pending by default)")
    lst.add_argument("--status", choices=(STATUS_PENDING, STATUS_PROMOTED, STATUS_REJECTED),
                     default=STATUS_PENDING)

    show = sub.add_parser("show", parents=[dirs],
                          help="Scores, evidence counts and the diff of one proposal")
    show.add_argument("proposal_id")

    promote = sub.add_parser(
        "promote", parents=[dirs],
        help="Replace the live skill with the proposal (archives the old version)")
    promote.add_argument("proposal_id")

    reject = sub.add_parser("reject", parents=[dirs],
                            help="Close a proposal without changing any skill")
    reject.add_argument("proposal_id")
    reject.add_argument("--reason", default=None)

    dry = sub.add_parser(
        "dry-run", parents=[dirs],
        help="What a GEPA cycle would do now, as counts (no model calls, no writes)")
    dry.add_argument("--json", action="store_true", help="Print the counts as JSON")
    dry.add_argument("--telemetry-db", default=None,
                     help="Telemetry database holding the load counter (default: the configured one)")
    dry.add_argument("--lcm-db", default=None,
                     help="Conversation store (default: the configured one)")


def _store(args: argparse.Namespace) -> ProposalStore:
    return ProposalStore(
        proposals_dir=Path(args.proposals_dir).expanduser() if args.proposals_dir else None,
        skills_auto_dir=Path(args.skills_dir).expanduser() if args.skills_dir else None,
    )


def _dry_run(args: argparse.Namespace) -> int:
    from prometheus.config.defaults import resolve_config_path
    from prometheus.learning.gepa import GEPAOptimizer, warn_renamed_keys

    import yaml

    config_path = Path(args.config).expanduser() if getattr(args, "config", None) else resolve_config_path()
    try:
        data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        print(f"gepa dry-run: cannot read the config ({type(exc).__name__}); "
              "reporting with an empty one")
        data = {}
    warn_renamed_keys(data)

    def _path(value: str | None) -> Path | None:
        return Path(value).expanduser() if value else None

    report = GEPAOptimizer.from_config_dict(
        data,
        telemetry_db=_path(args.telemetry_db),
        lcm_db=_path(args.lcm_db),
        skills_auto_dir=_path(args.skills_dir),
        proposals_dir=_path(args.proposals_dir),
    ).dry_run()
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(report.to_text())
    return 0


def run_gepa_command(args: argparse.Namespace) -> int:
    action = getattr(args, "gepa_action", None)
    if action is None:
        print("usage: oara gepa {proposals,show,promote,reject,dry-run} …")
        return 2
    if action == "dry-run":
        return _dry_run(args)
    store = _store(args)
    try:
        if action == "proposals":
            print(render_list(store.entries(args.status), status=args.status))
        elif action == "show":
            print(render_proposal(store.get(args.proposal_id)))
        elif action == "promote":
            print(render_promotion(store.promote(args.proposal_id, actor="cli")))
        elif action == "reject":
            store.reject(args.proposal_id, reason=args.reason, actor="cli")
            print(f"Rejected {args.proposal_id}. No skill was changed.")
        else:
            return 2
    except ProposalError as exc:
        print(f"gepa {action}: refused ({exc.code}): {exc.message}")
        return 1
    return 0
