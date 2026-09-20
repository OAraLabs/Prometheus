#!/usr/bin/env python3
"""Drive ONE computer-use step through the whole path, and print the evidence.

    observe -> build candidates -> choose -> validate -> GATE -> execute -> verify

⚠ THIS SCRIPT REGISTERS NOTHING. ``ComputerUseLoop`` has no ToolRegistry
reference, so the loop runs without ``register_computer_tools`` ever being
called and ``/api/status``'s ``computer.registered`` stays 0. That is
deliberate, not an oversight: registering would put ``computer_*`` into the
registry the agent loop reads, which would let a chat model click. Milestone 2
does not need it, so it does not happen.

⚠ THE APPROVAL TEXT PRINTED HERE IS THE REAL ONE. It goes through the actual
``ApprovalQueue``, which is what formats what an operator sees on Telegram —
a rendering of my own would prove that I can format a string, not that the
operator would be told what they are approving.

Modes:
  --cold     expect the substrate check to REFUSE. Run this in a process with
             no reachable display; it asserts nothing executed.
  --act      the successful path. Needs a live display and a target window.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prometheus.computer.chooser import RuleChooser  # noqa: E402
from prometheus.computer.driver import check_preconditions  # noqa: E402
from prometheus.computer.loop import ComputerUseLoop  # noqa: E402
from prometheus.computer.status import render  # noqa: E402
from prometheus.computer.targets import (  # noqa: E402
    KIND_LOCAL, Target, TargetRegistry,
)
from prometheus.permissions.approval_queue import ApprovalQueue  # noqa: E402
from prometheus.permissions.checker import PermissionMode, SecurityGate  # noqa: E402


class _CapturingTelegram:
    """Stands in for the gateway so the REAL prompt text can be read."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send(self, chat_id, text, parse_mode=None):  # noqa: ANN001
        self.messages.append(text)


def _banner(title: str) -> None:
    print("\n" + "═" * 72)
    print(title)
    print("═" * 72)


async def _run(args: argparse.Namespace) -> int:
    telegram = _CapturingTelegram()
    queue = ApprovalQueue(telegram_adapter=telegram, default_chat_id=1,
                          timeout_seconds=30)
    gate = SecurityGate(mode=PermissionMode.DEFAULT, audit_logger=None)
    gate._approval_queue = queue  # the hop SecurityGate.request_approval uses

    async def approve(tool_name, reason, arguments=None):
        """Auto-answer, but only AFTER the real prompt has been rendered."""
        task = asyncio.ensure_future(
            gate.request_approval(tool_name, reason, arguments=arguments))
        for _ in range(100):
            await asyncio.sleep(0.05)
            if queue.list_pending():
                break
        pending = queue.list_pending()
        if pending:
            await queue.approve(pending[0].request_id, scope="once")
        return await task

    _banner("SUBSTRATE (the same probe /api/status reports)")
    pre = check_preconditions()
    print(json.dumps(render(pre), indent=2))

    registry = TargetRegistry()
    registry.declare(Target(name=args.target, kind=KIND_LOCAL,
                            description="this machine's desktop"))

    if args.cold:
        driver = None
    else:
        from prometheus.computer.cua import CuaDriverAdapter
        driver = CuaDriverAdapter(target=args.target)
        driver.start()
        registry.bind(args.target, driver)

    loop = ComputerUseLoop(
        driver=driver or _NoDriver(),
        chooser=RuleChooser(prefer=tuple(args.prefer)),
        gate=gate,
        approve=approve,
        origin="system",
        skip_preconditions=False,   # ⚠ never skipped on a real run
    )

    _banner("THE STEP")
    result = await loop.step(
        goal=args.goal, target=args.target, app=args.app,
        pid=args.pid, window_id=args.window_id,
    )
    print(f"status            : {result.status}")
    print(f"reason            : {result.reason}")
    print(f"extent            : {result.extent or '(none)'}")
    print(f"candidates offered: {result.candidates_offered}")
    print(f"verified          : {result.verified}")
    print(f"driver result     : {result.driver_result}")

    _banner("THE APPROVAL PROMPT, AS AN OPERATOR WOULD SEE IT")
    if telegram.messages:
        for m in telegram.messages:
            print(m)
    else:
        print("(no approval was raised)")

    if driver is not None:
        driver.shutdown()

    if args.cold:
        ok = result.status == "blocked" and not result.driver_result
        print(f"\nCOLD PATH: {'REFUSED as required' if ok else 'DID NOT REFUSE'}")
        return 0 if ok else 1
    print(f"\nRESULT: {result.status}")
    return 0 if result.ok else 1


class _NoDriver:
    """Used only in --cold, where the substrate check must refuse first.

    If the refusal ever stops firing, these raise instead of silently doing
    nothing — the cold run must not be able to pass by accident.
    """

    def observe(self, *a, **k):
        raise AssertionError("observe reached in --cold: the refusal did not fire")

    def act(self, *a, **k):
        raise AssertionError("act reached in --cold: the refusal did not fire")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", default="mini")
    p.add_argument("--app", default="scratch")
    p.add_argument("--pid", type=int, default=0)
    p.add_argument("--window-id", type=int, default=0)
    p.add_argument("--goal", default="press the button")
    p.add_argument("--prefer", nargs="*", default=["button"])
    p.add_argument("--cold", action="store_true",
                   help="expect the substrate check to refuse")
    p.add_argument("--act", action="store_true", help="the successful path")
    args = p.parse_args()
    if not args.cold and not args.act:
        p.error("choose --cold or --act")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
