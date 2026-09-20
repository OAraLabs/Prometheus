#!/usr/bin/env python3
"""Show the event loop STALLING across ComputerUseLoop.step, and then not.

THE #416 SHAPE
--------------
``step`` is ``async def``, and four of the things it does are plain blocking
calls made directly on the event loop thread:

    check_preconditions()          real socket.connect + subprocess.run(gdbus)
    driver.observe(...)            blocks the caller
    driver.act(...)                blocks the caller
    _verify(...) -> observe(...)   blocks the caller again, post-action

While any of them runs, NOTHING else on that loop runs: not the heartbeat
below, not a gateway poll, not an approval timeout. This probe makes that
visible as a number instead of an argument.

WHAT IS REAL HERE AND WHAT IS A STAND-IN -- stated, not implied
----------------------------------------------------------------
``check_preconditions`` is the REAL function, unmodified. It is the one of the
four that needs no desktop, so its stall is measured, not simulated (~3 ms on
this box: a unix-socket connect to the X display plus a gdbus subprocess).

``observe`` and ``act`` are a STAND-IN, because the desktop is released to
another session. The stand-in is not ``time.sleep`` dressed up: it reproduces
``CuaDriverAdapter``'s actual threading model -- a private event loop on its
own thread, with the caller blocking in
``asyncio.run_coroutine_threadsafe(...).result()``. That matters because the
question being asked of ``asyncio.to_thread`` is precisely whether wrapping a
call that ALREADY owns a loop nests loops or serialises through that loop, and
a ``sleep``-based fake could not answer it. The DURATION is a parameter and
makes no claim about the real driver's latency; the MECHANISM is the real one.

WHAT THE PROOF IS
-----------------
Not that the call sites changed -- that the heartbeat stops stalling. Run this
before the change and after it, with the same --tick and --cost.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prometheus.computer.chooser import RuleChooser  # noqa: E402
from prometheus.computer.loop import ComputerUseLoop  # noqa: E402
from prometheus.computer.types import Element, Observation  # noqa: E402
from prometheus.permissions.checker import (  # noqa: E402
    Grant, PermissionMode, SecurityGate,
)
from prometheus.permissions.computer_extent import COMPUTER_ACTION_KIND  # noqa: E402

TARGET, APP, PID, WID = "probe", "fakeapp", 4242, 7


class _LoopOwningDriver:
    """Mirrors CuaDriverAdapter's threading model, which is the point.

    A private event loop on its own thread; every driver call submits a
    coroutine to it and BLOCKS the caller in ``.result()``. Swapping in a
    ``time.sleep`` fake here would still show a stall, but it could not show
    whether ``to_thread`` re-enters or serialises through a loop the driver
    already owns -- which is the actual risk in wrapping this particular
    driver.
    """

    def __init__(self, cost: float) -> None:
        self._cost = cost
        self._n = 0
        self.calls: list[str] = []
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="fake-cua-loop")
        self._thread.start()

    def _await(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=30)

    async def _work(self) -> None:
        await asyncio.sleep(self._cost)

    def observe(self, target, app, pid, window_id) -> Observation:
        self.calls.append("observe")
        self._await(self._work())
        self._n += 1
        return Observation(
            target=target, app=app, pid=pid, window_id=window_id,
            snapshot_id=f"snap-{self._n}",
            elements=(Element(0, f"tok-{self._n}", "push button", "Send"),),
        )

    def act(self, verb, arguments) -> dict:
        self.calls.append("act")
        self._await(self._work())
        return {"verb": verb, "effect": "STATE_CHANGED",
                "confirmed_by_driver": True, "landed": True}

    def shutdown(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


class Heartbeat:
    """Ticks on a fixed interval and records how late each tick actually was.

    A stalled loop cannot run this task, so the lateness IS the stall. Measured
    from the scheduled time, not from the previous tick, so one long block
    shows up as one large gap rather than being smeared.
    """

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.gaps: list[float] = []
        self._stop = False

    async def run(self) -> None:
        nxt = time.perf_counter() + self.interval
        while not self._stop:
            await asyncio.sleep(max(0.0, nxt - time.perf_counter()))
            now = time.perf_counter()
            self.gaps.append((now - nxt) * 1000.0)   # ms LATE
            nxt += self.interval

    def stop(self) -> None:
        self._stop = True

    def report(self, label: str, budget_ms: float) -> tuple[float, int]:
        worst = max(self.gaps) if self.gaps else 0.0
        stalls = [g for g in self.gaps if g > budget_ms]
        med = statistics.median(self.gaps) if self.gaps else 0.0
        print(f"  {label}")
        print(f"    ticks            : {len(self.gaps)}")
        print(f"    median lateness  : {med:7.2f} ms")
        print(f"    WORST lateness   : {worst:7.2f} ms")
        print(f"    ticks late > {budget_ms:.0f}ms : {len(stalls)}")
        if stalls:
            print(f"    the stalls       : "
                  f"{', '.join(f'{s:.1f}' for s in sorted(stalls, reverse=True)[:6])} ms")
        return worst, len(stalls)


def _gate() -> SecurityGate:
    """Pre-granted, so the measurement is of BLOCKING and not of consent.

    Uses the real grant path proved in item 1 rather than a permissive mode,
    so the gate still runs its full evaluate() on every step.
    """
    gate = SecurityGate(mode=PermissionMode.DEFAULT, audit_logger=None)
    gate.add_grant(Grant(
        kind=COMPUTER_ACTION_KIND,
        value=f"{TARGET}:{APP}:click:background",
        tool_name="computer_click",
    ))
    return gate


async def _measure(args, label: str, concurrent: int = 1):
    driver = _LoopOwningDriver(cost=args.cost)
    loop = ComputerUseLoop(
        driver=driver, chooser=RuleChooser(prefer=("button",)),
        gate=_gate(), approve=None, origin="system",
        skip_preconditions=False,          # the REAL check runs
    )
    hb = Heartbeat(args.tick)
    task = asyncio.ensure_future(hb.run())
    await asyncio.sleep(args.tick * 5)     # settle

    t0 = time.perf_counter()
    steps = [loop.step(goal="press the button", target=TARGET, app=APP,
                       pid=PID, window_id=WID)
             for _ in range(concurrent)]
    results = await asyncio.gather(*steps)
    wall = (time.perf_counter() - t0) * 1000.0

    await asyncio.sleep(args.tick * 5)
    hb.stop()
    task.cancel()
    driver.shutdown()

    print(f"\n  step wall time   : {wall:7.1f} ms   "
          f"({concurrent} concurrent step(s))")
    print(f"  statuses         : {[r.status for r in results]}")
    print(f"  driver calls     : {driver.calls}")
    worst, stalls = hb.report(label, args.budget)
    return worst, stalls, wall


async def _run(args) -> int:
    print("=" * 74)
    print(f"tick={args.tick*1000:.0f}ms  stand-in driver cost={args.cost*1000:.0f}ms "
          f"per call  budget={args.budget:.0f}ms")
    print("check_preconditions is the REAL function (socket + gdbus subprocess)")
    print("=" * 74)

    print("\n--- ONE STEP -------------------------------------------------")
    worst, stalls, _ = await _measure(args, "heartbeat, one step")

    print("\n--- TWO CONCURRENT STEPS -------------------------------------")
    print("  (does wrapping a driver that OWNS A LOOP serialise through it?)")
    worst2, stalls2, wall2 = await _measure(args, "heartbeat, two steps",
                                            concurrent=2)

    print("\n" + "=" * 74)
    print("READING THIS")
    print("=" * 74)
    print(f"  worst lateness, 1 step : {worst:7.2f} ms")
    print(f"  worst lateness, 2 steps: {worst2:7.2f} ms")
    print("  Before the fix: expect worst >> tick -- the loop is blocked for")
    print("  roughly (3 x driver cost + check_preconditions) per step, and two")
    print("  concurrent steps simply queue, because neither yields.")
    print("  After the fix: expect worst to fall to near the tick interval.")
    print("  Two-step wall time is the serialisation check: if it stays at")
    print("  ~2x the one-step time AFTER the fix, the driver's private loop is")
    print("  the new bottleneck and to_thread bought concurrency in name only.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tick", type=float, default=0.010,
                   help="heartbeat interval, seconds")
    p.add_argument("--cost", type=float, default=0.150,
                   help="stand-in driver cost per call, seconds")
    p.add_argument("--budget", type=float, default=25.0,
                   help="lateness above this counts as a stall, ms")
    return asyncio.run(_run(p.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
