"""Harvest real candidate tables into the corpus. OBSERVE ONLY — never acts.

WHY IT DOES NOT ACT
-------------------
Milestone 2 authorised ONE real click. Nothing authorises a hundred. And a
corpus row is a TABLE plus a JUDGMENT about which candidate was right —
execution is no part of the label, so acting would add risk and no data.

The five action-outcome columns are therefore empty by construction, which is
why `check --mode capture` exists and why that exclusion is scoped to this mode
rather than added to INTENTIONALLY_ABSENT. A corpus of real runs is still held
to every column.

GOALS ARE WRITTEN BEFORE THE TABLE IS SEEN
-------------------------------------------
docs/computer-use-corpus.md §2.2 measured that goals derived from a candidate's
own description make 93% of rows trivial — RuleChooser recovers its own goal by
substring overlap, and the corpus measures nothing. So the goals below are
declared per app as literals in this file, in task language, and the harvester
reads the window afterwards. It is structurally unable to look first.

Goals are deliberately phrased the way a person states an intent ("undo what I
just did"), never the way a control is labelled ("click Undo").
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import gi  # noqa: E402

gi.require_version("Atspi", "2.0")
from gi.repository import Atspi  # noqa: E402

from prometheus.computer.candidates import (  # noqa: E402
    UnusableObservation,
    build_candidates,
)
from prometheus.computer.corpus import (  # noqa: E402
    CorpusStore,
    TableRecord,
    assert_corpus_complete,
    IncompleteCorpus,
)
from prometheus.computer.chooser import RuleChooser  # noqa: E402
from prometheus.computer.candidates import build_choice_request  # noqa: E402
from prometheus.computer.types import Element, Observation  # noqa: E402
from prometheus.config.paths import get_computer_corpus_db_path  # noqa: E402

#: App -> goals, in TASK LANGUAGE, written before any window was opened.
#: Never the label of a control; always the intent a person would state.
GOALS: dict[str, list[str]] = {
    "gnome-calculator": [
        "add two numbers together",
        "clear what I have entered so far",
        "undo my last keystroke",
        "work out a percentage",
        "switch to a different kind of calculation",
        "copy the result so I can paste it elsewhere",
    ],
    # A "main menu open" capture makes the menu's own items one-step
    # reachable, which is the fix for the container-opening signal: at 3/3 the
    # pilot was measuring "open the menu" rather than task selection.
    "gnome-text-editor (main menu open)": [
        "undo what I just did",
        "save this somewhere",
        "find a word in this document",
    ],
    "org.gnome.Nautilus (main menu open)": [
        "make a new folder",
        "change how these are displayed",
        "show hidden files",
    ],
    "gnome-text-editor": [
        "undo what I just did",
        "save this somewhere",
        "find a word in this document",
        "start a new document",
        "make the text bigger",
        "close this without losing anything",
    ],
    # ⚠ NO PROCESS-LIST GOALS. See STRUCTURALLY_UNREACHABLE: the rows are not
    # in the tree at all, so "stop a program that has hung" and "sort the list"
    # can never be answered from any table this app produces. The tab switcher
    # is reachable and those goals stay.
    "gnome-system-monitor": [
        "look at a different category of information",
        "check how busy the processor is",
        "switch to the view that shows disk usage",
    ],
    "baobab": [
        "find out what is using my disk space",
        "look inside a folder that is taking up room",
        "scan a different drive",
        "go back to where I was",
    ],
    "org.gnome.Nautilus": [
        "open my documents",
        "make a new folder",
        "go back to the previous place",
        "change how these are displayed",
        "search for a file by name",
    ],
    "gnome-control-center": [
        "change the desktop background",
        "check what version of the system I am running",
        "adjust the sound volume",
        "look at my network connection",
        "find a setting I cannot remember the name of",
    ],
    "gnome-disks": [
        "see how much space is left on this drive",
        "check whether the disk is healthy",
        "unmount a volume before unplugging it",
        "look at a different drive",
        "find out what filesystem this is",
    ],
    "evince": [
        "jump to a particular page",
        "make the page bigger so I can read it",
        "search this document for a phrase",
        "go to the next page",
        "print what I am looking at",
    ],
}

#: Never harvested. seahorse's accessible tree is key names and fingerprints;
#: firefox is outside this milestone and is where scraped text is most likely
#: to be personal. Enforced here, not left to the operator to remember.
NEVER_HARVEST = {"seahorse", "firefox", "thunderbird", "evolution", "keepassxc"}

#: App views the loop STRUCTURALLY CANNOT operate — the widget never exposes
#: its content to AT-SPI at all, so no role set and no chooser reaches it.
#: Recorded so harvest states are not spent on them.
#:
#: gnome-system-monitor's PROCESS LIST: zero row-like nodes (table row, table
#: cell, tree table, table, tree item, list item) anywhere in its 118-node full
#: tree, showing or not. Its TAB SWITCHER is fine and worth capturing — that
#: was a real role gap (`page tab`) and is now fixed.
STRUCTURALLY_UNREACHABLE: dict[str, str] = {
    "gnome-system-monitor: process list":
        "the process rows never reach the accessibility tree; not a role gap",
}

MIN_CANDIDATES = 10


def _walk(node, out, depth=0):
    if depth > 14 or len(out) > 4000:
        return
    try:
        st = node.get_state_set()
        if st.contains(Atspi.StateType.SHOWING):
            out.append((
                node.get_role_name(),
                node.get_name() or "",
                st.contains(Atspi.StateType.EDITABLE),
            ))
        n = node.get_child_count()
    except Exception:
        return
    for i in range(min(n, 200)):
        try:
            _walk(node.get_child_at_index(i), out, depth + 1)
        except Exception:
            pass


def _observe(app_name: str) -> Observation | None:
    """Read one live window into an Observation. No driver, no actions."""
    desktop = Atspi.get_desktop(0)
    for i in range(desktop.get_child_count()):
        try:
            app = desktop.get_child_at_index(i)
            if (app.get_name() or "") != app_name or app.get_child_count() == 0:
                continue
            win = app.get_child_at_index(0)
        except Exception:
            continue
        nodes: list = []
        _walk(win, nodes)
        if not nodes:
            return None
        els = tuple(
            Element(element_index=k, element_token=f"{app_name}-{k}",
                    role=r, label=nm, editable=ed)
            for k, (r, nm, ed) in enumerate(nodes)
        )
        return Observation(
            target="mini", app=app_name, pid=0, window_id=0,
            snapshot_id=f"{app_name}-{time.time():.3f}", elements=els,
        )
    return None


def harvest(store: CorpusStore, session: str, apps: list[str]) -> dict:
    chooser = RuleChooser()
    stats = {"captured": 0, "too_small": 0, "no_window": 0, "unusable": 0}

    for app in apps:
        if app in NEVER_HARVEST:
            print(f"  REFUSED {app}: on the never-harvest list")
            continue
        obs = _observe(app)
        if obs is None:
            print(f"  skip    {app}: no window on the accessibility bus")
            stats["no_window"] += 1
            continue
        try:
            candidates = build_candidates(obs)
        except UnusableObservation as exc:
            print(f"  skip    {app}: unusable observation — {exc}")
            stats["unusable"] += 1
            continue
        if len(candidates) < MIN_CANDIDATES:
            print(f"  skip    {app}: {len(candidates)} candidates "
                  f"(< {MIN_CANDIDATES}, chance floor too high to discriminate)")
            stats["too_small"] += 1
            continue

        for goal in GOALS.get(app, []):
            rec = TableRecord(
                goal=goal, target="mini", app=app, window_id=0, pid=0,
                # OBSERVE-ONLY: no driver ran, and the row says so rather than
                # borrowing a name that implies one did.
                driver_kind="atspi-capture",
                goal_source="human", harvest_session=session,
            )
            rec.note_observation(obs)
            rec.note_candidates(candidates)
            choice = chooser.choose(
                build_choice_request(goal, obs, candidates)
            )
            rec.note_choice(choice)
            rec.status = "captured"
            rec.reason = "observe-only harvest; nothing executed"
            if store.capture(rec):
                stats["captured"] += 1
                mark = "ABSTAIN" if choice.candidate_id == "abstain" else choice.candidate_id
                print(f"  ok      {app:<22} N={len(candidates):<3} "
                      f"rule={mark:<10} {goal}")
    return stats


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", required=True,
                   help="groups rows collected under one discipline")
    p.add_argument("--db", default=None)
    p.add_argument("--apps", nargs="*", default=sorted(GOALS))
    args = p.parse_args()

    store = CorpusStore(args.db or get_computer_corpus_db_path())
    print(f"harvest session {args.session!r} -> {store.db_path}")
    print(f"apps: {', '.join(args.apps)}\n")

    stats = harvest(store, args.session, args.apps)
    print(f"\n{stats}")

    print("\nEXIT CRITERIA — completeness guard (capture mode)")
    try:
        assert_corpus_complete(store, mode="capture")
    except IncompleteCorpus as exc:
        print(f"  FAILED\n{exc}", file=sys.stderr)
        return 1
    print("  passed — every declared column is populated in at least one row")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
