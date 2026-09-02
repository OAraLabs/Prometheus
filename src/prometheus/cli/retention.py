"""``oara retention`` — collect tombstoned sessions that nothing will ever show again.

DRY RUN BY DEFAULT. This deletes conversation data irreversibly and is reachable from cron, so
``--apply`` is required to destroy anything. Printing the plan first is the point: the population
it acts on contained real conversations when it was last inspected by hand.
"""

from __future__ import annotations

import argparse

from prometheus.memory.lcm_conversation_store import LCMConversationStore
from prometheus.memory.retention import (
    DEFAULT_CONVERSATION_DAYS,
    DEFAULT_MACHINE_DAYS,
    apply_retention,
    plan_retention,
)


def add_retention_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "retention",
        help="Purge tombstoned sessions past their retention window (dry run unless --apply)",
    )
    p.add_argument("--apply", action="store_true",
                   help="Actually purge. Without this, prints the plan and exits.")
    p.add_argument("--machine-days", type=int, default=DEFAULT_MACHINE_DAYS,
                   help=f"Window for smoke:/bakeoff:/eval:/gym:/coding: (default {DEFAULT_MACHINE_DAYS})")
    p.add_argument("--conversation-days", type=int, default=DEFAULT_CONVERSATION_DAYS,
                   help=f"Window for forgotten user chats (default {DEFAULT_CONVERSATION_DAYS})")
    p.add_argument("--db", help="LCM database path (defaults to the configured one)")


def run_retention(args: argparse.Namespace) -> bool:
    store = LCMConversationStore(args.db) if args.db else LCMConversationStore()
    plan = plan_retention(
        store,
        machine_days=args.machine_days,
        conversation_days=args.conversation_days,
    )
    machine, convo = plan.machine, plan.conversations
    print(f"retention plan  (machine >{args.machine_days}d, conversations >{args.conversation_days}d)")
    print(f"  machine traffic   {len(machine):4d} sessions  {sum(c.messages for c in machine):6d} messages")
    print(f"  forgotten chats   {len(convo):4d} sessions  {sum(c.messages for c in convo):6d} messages")
    print(f"  left alone        {plan.skipped_too_recent} inside their window, "
          f"{plan.skipped_revived} spoke after being forgotten")
    # Conversations are listed INDIVIDUALLY. Machine traffic is fungible; a chat the user chose to
    # forget is not, and if it is about to go the operator should see which one.
    for c in convo:
        print(f"    conversation  {c.session_id:44s} {c.messages:5d} msgs  {c.age_days:.0f}d")
    if not args.apply:
        print("\nDRY RUN — nothing purged. Re-run with --apply.")
        return True
    result = apply_retention(store, plan)
    print(f"\npurged {result['purged']}, skipped {result['skipped']} (spoke since planning)")
    return True
