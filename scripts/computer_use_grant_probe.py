#!/usr/bin/env python3
"""Prove the REMEMBERED GRANT, live, through the real dispatch and approval path.

WHAT WAS ALREADY PROVEN, AND WHAT WAS NOT
-----------------------------------------
Milestone 2 proved refuse-then-approve for ONE action.
tests/test_computer_use_gate_path.py proves the MATCHING half of a
remembered grant — but it HAND-BUILDS Grant(...), calls add_grant
directly, and runs against a FixtureDriver. The MINTING half — a grant
DERIVED from a real operator approval by derive_grant and stored by
cmd_approve — had no coverage and had never run against a real desktop.
This script runs that half.

THE TWO SILENT FAILURE MODES THIS EXISTS TO SEPARATE
----------------------------------------------------
Both look identical from outside a running system:

  1. The grant NEVER MATCHES: every call prompts forever. Safe, useless, and
     indistinguishable from a gate that is simply working. Its real cause was
     that ApprovalQueue.approve does NOT call add_grant — the only call site
     is cmd_approve, which reaches the gate through queue._security_gate, a
     field daemon.py set at one line and nothing else on any surface set at
     all. That field is a REQUIRED CONSTRUCTOR ARGUMENT now, so --unwired can
     no longer build the broken queue: it asserts the refusal instead.
  2. The grant MATCHES WIDER than the operator was shown. Probes C and D are
     the ones that would catch it.

WHY PROBE D USES scroll, NOT type_text
----------------------------------------------
type_text and invoke_menu carry payloads and are UNREMEMBERABLE by
construction, so they prompt no matter what is stored. Using one for the
different-verb probe would prove nothing about matching WIDTH — it would pass
on the payload rule alone. scroll is rememberable, so the only reason it
can prompt is that the stored click grant does not cover it.

REGISTERS NOTHING. Tools go into a ToolRegistry this script owns;
register_computer_tools is never called, so computer.registered
stays 0. Grants persist to a THROWAWAY config, never the daemon's.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prometheus.computer.tools import build_computer_tools  # noqa: E402
from prometheus.engine.agent_loop import LoopContext, _execute_tool_call  # noqa: E402
from prometheus.gateway.commands import cmd_approve  # noqa: E402
from prometheus.permissions.approval_queue import ApprovalQueue  # noqa: E402
from prometheus.permissions.checker import PermissionMode, SecurityGate  # noqa: E402
from prometheus.tools.base import ToolRegistry  # noqa: E402

THROWAWAY_CONFIG = "/tmp/trackA/probe-config.yaml"


class _CapturingAudit:
    """The one audit store. The gate writes here, and so does the QUEUE —
    _audit_resolution reaches the logger through gate._audit by
    design, so request and resolution rows land in the same list."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def log(self, *, tool_name, decision, trust_level, reason, tool_input=None):
        self.rows.append({
            "tool": tool_name,
            "decision": getattr(decision, "value", str(decision)),
            "reason": reason,
        })


class _CapturingTelegram:
    """Stands in for the gateway so the REAL prompt text can be read."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send(self, chat_id, text, parse_mode=None):  # noqa: ANN001
        self.messages.append(text)


def _banner(t: str) -> None:
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


async def _wait_pending(queue, timeout: float = 4.0) -> str | None:
    """The request id of a raised prompt, or None if none was raised.

    None is a RESULT, not a failure: probe B asserts it."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.05)
        pending = queue.list_pending()
        if pending:
            return pending[0].request_id
    return None


async def _probe(ctx, queue, audit, label, tool, args, *, scope):
    """Run ONE gated call through the real dispatch. Report whether it asked.

    scope is the operator's answer if asked: "always" mints a persistent
    grant, "once" mints nothing. Answered through the real cmd_approve,
    so derive_grant and add_grant run exactly as on Telegram."""
    first_row = len(audit.rows)
    call = asyncio.ensure_future(_execute_tool_call(ctx, tool, "probe", args))
    rid = await _wait_pending(queue)
    reply = None
    if rid is not None:
        arg_text = rid if scope == "once" else f"{scope} {rid}"
        reply = await cmd_approve(queue, arg_text)
    result = await call
    return {
        "label": label,
        "tool": tool,
        "app": args.get("app"),
        "prompted": rid is not None,
        "approve_reply": reply,
        "rows": audit.rows[first_row:],
        "is_error": getattr(result, "is_error", None),
    }


def _observe_real(driver, target, app, pid, window_id):
    """A real snapshot, and a real element token from it.

    Element tokens are snapshot-bound, so every probe re-observes. That the
    grant still matches across a NEW snapshot is itself the point: the extent
    deliberately carries no element term."""
    obs = driver.observe(target, app, pid, window_id)
    clickable = [e for e in obs.elements if not e.editable]
    chosen = clickable[0] if clickable else obs.elements[0]
    return obs, chosen


async def _run(args) -> int:
    from prometheus.computer.cua import CuaDriverAdapter

    audit = _CapturingAudit()
    telegram = _CapturingTelegram()
    gate = SecurityGate(mode=PermissionMode.DEFAULT, audit_logger=audit,
                        config_path=THROWAWAY_CONFIG)

    # --unwired NO LONGER REPRODUCES THE DEFECT -- it demonstrates that the
    # defect has no constructor. The gateless queue this flag was written to
    # build is refused at construction now, so the flag asserts the refusal
    # instead of showing the silence it used to.
    if args.unwired:
        try:
            ApprovalQueue(security_gate=None, telegram_adapter=telegram)
        except ValueError as exc:
            _banner("--unwired: THE CONSTRUCTOR REFUSES, WHICH IS THE FIX")
            print(exc)
            return 0
        raise AssertionError(
            "--unwired built a gateless queue: the constructor stopped "
            "refusing and the silent-grant defect is reachable again"
        )

    queue = ApprovalQueue(security_gate=gate, telegram_adapter=telegram,
                          default_chat_id=1, timeout_seconds=20)
    gate._approval_queue = queue          # the late-bound, fail-CLOSED hop

    driver = CuaDriverAdapter(target=args.target)
    driver.start()

    registry = ToolRegistry()
    for tool in build_computer_tools(driver):
        registry.register(tool)

    # permission_prompt stays None ON PURPOSE: _execute_tool_call then falls
    # back to gate.request_approval, which is the route a registered tool
    # takes in the daemon. Supplying a prompt here would test my callable.
    ctx = LoopContext(provider=None, model="probe", system_prompt="",
                      max_tokens=512, tool_registry=registry,
                      permission_checker=gate, permission_prompt=None)

    def click_args(app, pid, wid):
        obs, el = _observe_real(driver, args.target, app, pid, wid)
        return {
            "target": args.target, "app": app, "pid": pid, "window_id": wid,
            "snapshot_id": obs.snapshot_id, "element_token": el.element_token,
            "delivery_mode": "background",
        }, el

    results = []

    _banner("PROBE A — first click in app A. MUST prompt. Answered 'always'.")
    a_args, a_el = click_args(args.app_a, args.pid_a, args.wid_a)
    print(f"element: {a_el.element_token} {a_el.role!r} {a_el.label!r}")
    results.append(await _probe(ctx, queue, audit, "A first click",
                                "computer_click", a_args, scope="always"))
    print(f"prompted: {results[-1]['prompted']}")
    print(f"/approve always -> {results[-1]['approve_reply']}")
    print(f"grants now: {[g.describe() for g in gate.list_grants()]}")

    _banner("PROBE B — IDENTICAL click, fresh snapshot. MUST NOT prompt.")
    b_args, _ = click_args(args.app_a, args.pid_a, args.wid_a)
    results.append(await _probe(ctx, queue, audit, "B identical click",
                                "computer_click", b_args, scope="always"))
    print(f"prompted: {results[-1]['prompted']}   (False is the pass)")

    _banner("PROBE C — same verb, DIFFERENT app. MUST prompt.")
    c_args, _ = click_args(args.app_b, args.pid_b, args.wid_b)
    results.append(await _probe(ctx, queue, audit, "C click other app",
                                "computer_click", c_args, scope="once"))
    print(f"prompted: {results[-1]['prompted']}   (True is the pass)")

    _banner("PROBE D — DIFFERENT verb (scroll, rememberable), same app. MUST prompt.")
    d_args, _ = click_args(args.app_a, args.pid_a, args.wid_a)
    d_args = {**d_args, "direction": "down", "amount": 1}
    results.append(await _probe(ctx, queue, audit, "D scroll same app",
                                "computer_scroll", d_args, scope="once"))
    print(f"prompted: {results[-1]['prompted']}   (True is the pass)")

    _banner("THE APPROVAL PROMPT, AS AN OPERATOR WOULD SEE IT")
    print(telegram.messages[0] if telegram.messages else "(none raised)")

    _banner("AUDIT ROWS, IN ORDER")
    for r in audit.rows:
        print(f"  {r['decision']:<18} {r['tool']:<18} {r['reason']}")

    _banner("VERDICT")
    expect = {"A first click": True, "B identical click": False,
              "C click other app": True, "D scroll same app": True}
    ok = True
    for r in results:
        want = expect[r["label"]]
        good = r["prompted"] == want
        ok = ok and good
        print(f"  {'PASS' if good else 'FAIL'}  {r['label']:<20} "
              f"prompted={r['prompted']} expected={want} "
              f"executed={r['is_error'] is False}")
    if args.unwired:
        print("\n  --unwired: B prompting is the DEFECT being demonstrated, "
              "not a pass.")
    driver.shutdown()
    print(json.dumps({"grants": [g.describe() for g in gate.list_grants()]},
                     indent=2))
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", default="mini")
    p.add_argument("--app-a", default="gnome-text-editor")
    p.add_argument("--pid-a", type=int, required=True)
    p.add_argument("--wid-a", type=int, required=True)
    p.add_argument("--app-b", default="gnome-calculator")
    p.add_argument("--pid-b", type=int, required=True)
    p.add_argument("--wid-b", type=int, required=True)
    p.add_argument("--unwired", action="store_true",
                   help="omit queue._security_gate: reproduce failure mode 1")
    return asyncio.run(_run(p.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
