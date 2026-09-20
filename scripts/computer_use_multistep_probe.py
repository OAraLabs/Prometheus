#!/usr/bin/env python3
"""Two REAL steps against a live desktop, with history carrying between them.

WHAT WAS UNPROVEN
-----------------
One click is not a loop. ``tests/test_computer_use_candidate_loop.py`` has
exactly one whole-path test and it calls ``step()`` ONCE; no test in the tree
calls it twice, and no test mentions ``history`` at all. So the second
observation, the second candidate table, and the hand-off between them had
never run.

WHAT THIS FOUND, AND IT IS THE POINT OF RUN A
---------------------------------------------
``history`` is assembled by ``loop.step`` and handed to the chooser inside
``ChoiceRequest`` -- and NOTHING READS IT. ``RuleChooser.choose`` uses
``goal`` and ``candidates`` only. ``ScriptedChooser.choose`` opens with
``del request``. A grep for ``.history`` across src, tests and scripts finds
no other consumer. It is carried, not consumed: written-but-not-wired, on the
consuming side.

So this probe runs the two steps TWICE:

  RUN A -- the shipped RuleChooser. History arrives populated and correct,
           the chooser ignores it, and step 2 picks THE SAME ACTION AGAIN.
           That repeat is the orphan, made visible.
  RUN B -- a chooser that actually reads ``request.history`` and will not
           repeat what is in it. Same wire, same data, different decision --
           which is what proves the hand-off carries real, usable data rather
           than a list nobody looks at.

The chooser is WRAPPED rather than trusted: every ``ChoiceRequest`` it is
handed is recorded, so both candidate tables and the history contents are
reported from what the chooser ACTUALLY RECEIVED. Reporting what the caller
passed in would prove the caller, not the wire. The recorded
``snapshot_id`` is how step 2's observation is shown to be genuinely FRESH
rather than the first one reused.

Registers nothing: ``register_computer_tools`` is never called.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prometheus.computer.chooser import RuleChooser  # noqa: E402
from prometheus.computer.loop import ComputerUseLoop  # noqa: E402
from prometheus.computer.types import (  # noqa: E402
    CANDIDATE_ABSTAIN, Choice, ChoiceRequest,
)
from prometheus.gateway.commands import cmd_approve  # noqa: E402
from prometheus.permissions.approval_queue import ApprovalQueue  # noqa: E402
from prometheus.permissions.checker import PermissionMode, SecurityGate  # noqa: E402

THROWAWAY_CONFIG = "/tmp/trackA/multistep-config.yaml"


class _RecordingChooser:
    """Keeps every ChoiceRequest handed to the inner chooser.

    Reporting the tables and the history from what the chooser RECEIVED,
    rather than from what the caller passed, is the difference between
    proving the wire and assuming it.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.requests: list[ChoiceRequest] = []

    @property
    def name(self) -> str:
        return getattr(self._inner, "name", "wrapped")

    def choose(self, request: ChoiceRequest) -> Choice:
        self.requests.append(copy.deepcopy(request))
        return self._inner.choose(request)


class HistoryAwareChooser:
    """Reads ``request.history``. No shipped chooser does.

    Deliberately thin: it drops candidates whose description already appears
    in history and delegates the actual scoring to RuleChooser, so the only
    difference between run A and run B is whether history is consulted.
    """

    name = "history-aware"

    def __init__(self, prefer: tuple[str, ...] = ()) -> None:
        self._rule = RuleChooser(prefer=prefer)

    def choose(self, request: ChoiceRequest) -> Choice:
        done = set(request.history)
        remaining = [c for c in request.candidates
                     if c.get("description") not in done]
        if not remaining:
            return Choice(CANDIDATE_ABSTAIN, confidence=1.0, source=self.name)
        narrowed = ChoiceRequest(
            goal=request.goal, snapshot_id=request.snapshot_id,
            candidates=remaining, history=list(request.history),
        )
        return replace(self._rule.choose(narrowed), source=self.name)


class _Telegram:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send(self, chat_id, text, parse_mode=None):  # noqa: ANN001
        self.messages.append(text)


def _banner(t: str) -> None:
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


def _table(request: ChoiceRequest) -> str:
    rows = [f"    snapshot_id : {request.snapshot_id}",
            f"    goal        : {request.goal!r}",
            f"    history     : {request.history or '(empty)'}",
            f"    candidates  : {len(request.candidates)}"]
    for c in request.candidates:
        rows.append(f"      - {c.get('id')}  {c.get('description')!r}")
    return "\n".join(rows)


async def _two_steps(args, chooser_inner, label):
    from prometheus.computer.cua import CuaDriverAdapter

    telegram = _Telegram()
    gate = SecurityGate(mode=PermissionMode.DEFAULT, audit_logger=None,
                        config_path=THROWAWAY_CONFIG)
    queue = ApprovalQueue(security_gate=gate, telegram_adapter=telegram,
                          default_chat_id=1, timeout_seconds=20)
    gate._approval_queue = queue

    async def approve(tool_name, reason, arguments=None):
        """Answer through the REAL /approve command, always-scope.

        Always-scope on purpose: two steps under ONE consent is the shape a
        real run has, and it reuses the grant path proved in item 1.
        """
        task = asyncio.ensure_future(
            gate.request_approval(tool_name, reason, arguments=arguments))
        for _ in range(200):
            await asyncio.sleep(0.05)
            pending = queue.list_pending()
            if pending:
                await cmd_approve(queue, f"always {pending[0].request_id}")
                break
        return await task

    driver = CuaDriverAdapter(target=args.target)
    driver.start()
    recorder = _RecordingChooser(chooser_inner)
    loop = ComputerUseLoop(
        driver=driver, chooser=recorder, gate=gate, approve=approve,
        origin="system", skip_preconditions=False,
    )

    _banner(f"{label} -- STEP 1")
    r1 = await loop.step(goal=args.goal, target=args.target, app=args.app,
                         pid=args.pid, window_id=args.window_id)
    print(f"  status   : {r1.status}   verified={r1.verified}")
    print(f"  chose    : {r1.candidate.description if r1.candidate else None!r}")
    print(f"  extent   : {r1.extent or '(none)'}")
    print(f"  history OUT: {r1.history}")

    _banner(f"{label} -- STEP 2   (history carried in from step 1)")
    r2 = await loop.step(goal=args.goal, target=args.target, app=args.app,
                         pid=args.pid, window_id=args.window_id,
                         history=r1.history)
    print(f"  status   : {r2.status}   verified={r2.verified}")
    print(f"  chose    : {r2.candidate.description if r2.candidate else None!r}")
    print(f"  history OUT: {r2.history}")

    driver.shutdown()
    return r1, r2, recorder, telegram


async def _run(args) -> int:
    results = {}
    for label, chooser in (
        ("RUN A  (shipped RuleChooser -- ignores history)",
         RuleChooser(prefer=tuple(args.prefer))),
        ("RUN B  (HistoryAwareChooser -- reads history)",
         HistoryAwareChooser(prefer=tuple(args.prefer))),
    ):
        r1, r2, rec, tg = await _two_steps(args, chooser, label)
        results[label] = (r1, r2, rec, tg)

        _banner(f"{label} -- THE TWO CANDIDATE TABLES, AS THE CHOOSER SAW THEM")
        for i, req in enumerate(rec.requests, start=1):
            print(f"  STEP {i}:")
            print(_table(req))

        if len(rec.requests) == 2:
            a, b = rec.requests
            print(f"\n  fresh observation : {a.snapshot_id} -> {b.snapshot_id}"
                  f"   {'DIFFERENT (fresh)' if a.snapshot_id != b.snapshot_id else 'SAME (REUSED!)'}")
            print(f"  history into st.1 : {a.history or '(empty)'}")
            print(f"  history into st.2 : {b.history or '(empty)'}")
            same = (r1.candidate and r2.candidate
                    and r1.candidate.candidate_id == r2.candidate.candidate_id)
            print(f"  step2 repeated st.1: {bool(same)}")

    _banner("VERDICT")
    ok = True
    for label, (r1, r2, rec, _tg) in results.items():
        two = len(rec.requests) == 2
        fresh = two and rec.requests[0].snapshot_id != rec.requests[1].snapshot_id
        carried = two and rec.requests[1].history == r1.history and bool(r1.history)
        both_ran = r1.status == "executed" and r2.status == "executed"
        good = two and fresh and carried and both_ran
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  {label}")
        print(f"        two steps={two} fresh_reobserve={fresh} "
              f"history_carried={carried} both_executed={both_ran}")

    a1, a2 = results[list(results)[0]][0], results[list(results)[0]][1]
    b1, b2 = results[list(results)[1]][0], results[list(results)[1]][1]
    a_same = a1.candidate and a2.candidate and a1.candidate.candidate_id == a2.candidate.candidate_id
    b_same = b1.candidate and b2.candidate and b1.candidate.candidate_id == b2.candidate.candidate_id
    print("\n  THE ORPHAN, STATED AS AN OUTCOME:")
    print(f"    run A repeated its own step-1 action : {bool(a_same)}"
          "   <- history was present and ignored")
    print(f"    run B repeated its own step-1 action : {bool(b_same)}"
          "   <- same history, consulted")
    if not (a_same and not b_same):
        print("    (NOTE: the two runs did not differ as expected -- read the "
              "tables above before trusting this line)")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", default="mini")
    p.add_argument("--app", default="gnome-text-editor")
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--window-id", type=int, required=True)
    p.add_argument("--goal", default="press the button")
    p.add_argument("--prefer", nargs="*", default=["button"])
    return asyncio.run(_run(p.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
