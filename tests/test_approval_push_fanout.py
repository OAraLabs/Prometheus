"""The APNs approval body is a deliberate narrow subset, and stays one.

WHY THIS FILE EXISTS
--------------------
``ApprovalQueue.serialize_pending`` has THREE consumers, not the two its
docstring claimed until 2026-09-19: ``GET /api/approvals``, the
``approval_pending`` WS signal, and ``push.PushDispatcher`` — which the web
launcher subscribes to ``"*"`` on the SignalBus whenever ``push.enabled``, and
which fans out to **every registered device, unconditionally and off-box**.

The first two answer someone already holding a bearer token or a socket. The
third is unsolicited delivery through Apple to every phone that ever
registered. That asymmetry is invisible in the code: ``on_signal`` takes the
same dict the other two take, and ``_push_approval`` simply happens to read
four keys out of it.

"Happens to read four keys" is not a control. Milestone 1 added ``arguments``
to that dict — the literal text a desktop action is about to type — and the
only reason it does not reach Apple is that nobody added the line. This file
turns that into a decision someone has to make on purpose.

⚠ IF YOU ARE HERE BECAUSE THIS TEST FAILED: adding a key to the push body
ships it to every registered device. That may be right! But it is a consent
and privacy decision, not a formatting one — say so in the PR, and update the
docstrings in ``approval_queue.serialize_pending`` and ``_push_approval``
which both describe this set.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest

from prometheus.permissions.approval_queue import ApprovalQueue, PendingAction
from prometheus.permissions.checker import SecurityGate

#: What the APNs body is allowed to carry, top level. Reviewed 2026-09-19.
ALLOWED_TOP_LEVEL = {"aps", "request_id", "tool_name", "expires_at"}

#: Which keys of the serialized approval the push path may READ. `arguments`
#: and `extents` are deliberately absent: the first is the text a tool was
#: about to type, the second describes lasting grants an operator has not yet
#: been offered.
ALLOWED_PAYLOAD_READS = {"tool_name", "description", "request_id", "expires_at"}


class _Store:
    def __init__(self) -> None:
        self.delivered: list[dict] = []

    def push_targets(self):
        return ["device-1", "device-2"]


class _Sender:
    pass


def _dispatcher(store):
    from prometheus.push.dispatcher import PushDispatcher

    d = PushDispatcher(store, _Sender(), bridge=None)

    async def _deliver(target, body):
        store.delivered.append({"target": target, "body": body})

    d._deliver = _deliver  # type: ignore[method-assign]
    return d


def _serialized_approval() -> dict:
    """A real serialize_pending dict, not a hand-built one.

    Hand-building it would prove the dispatcher filters a shape the caller
    never produces — the defect `tool_paths.py` records at length.
    """
    queue = ApprovalQueue(security_gate=SecurityGate(), telegram_adapter=None, default_chat_id=None,
                          timeout_seconds=60)
    action = PendingAction(
        request_id="r1",
        tool_name="computer_type_text",
        description="computer_type_text acts on the desktop",
        arguments={"text": "transfer 500 to account 9", "app": "mail"},
    )
    return queue.serialize_pending(action)


# ── THE EFFECT: what actually reaches a device ──────────────────────────────

def test_the_typed_text_never_reaches_a_device():
    """The whole point. `arguments` carries what a tool is about to type."""
    store = _Store()
    payload = _serialized_approval()
    assert "transfer 500" in str(payload["arguments"]), (
        "the fixture is not exercising the risk — serialize_pending did not "
        "carry the arguments at all"
    )

    asyncio.run(_dispatcher(store).on_signal(
        type("S", (), {"kind": "approval_pending", "payload": payload})()
    ))

    assert store.delivered, "nothing was delivered; the test proves nothing"
    for row in store.delivered:
        blob = str(row["body"])
        assert "transfer 500" not in blob, (
            f"the text a tool was about to type reached a registered device "
            f"via APNs: {blob}"
        )


def test_the_push_body_carries_only_reviewed_keys():
    store = _Store()
    asyncio.run(_dispatcher(store).on_signal(
        type("S", (), {"kind": "approval_pending",
                       "payload": _serialized_approval()})()
    ))
    for row in store.delivered:
        extra = set(row["body"]) - ALLOWED_TOP_LEVEL
        assert not extra, (
            f"new key(s) {sorted(extra)} in the APNs approval body — this "
            f"ships to EVERY registered device. See this module's docstring."
        )


def test_extents_never_reach_a_device():
    """`extents` describes lasting grants the operator has not been offered."""
    store = _Store()
    payload = _serialized_approval()
    payload["extents"] = {"always": "CANARY-EXTENT-VALUE"}
    asyncio.run(_dispatcher(store).on_signal(
        type("S", (), {"kind": "approval_pending", "payload": payload})()
    ))
    for row in store.delivered:
        assert "CANARY-EXTENT-VALUE" not in str(row["body"])


# ── THE STRUCTURE: which keys the code reads at all ─────────────────────────

def test_push_approval_reads_only_the_reviewed_payload_keys():
    """Read by AST, so a key read into a local still counts.

    The effect tests above catch a value that LANDS in the body. This catches
    the earlier move — reading a key at all — because the natural next step
    after reading one is putting it somewhere.
    """
    import prometheus.push.dispatcher as mod

    src = Path(mod.__file__).read_text()
    tree = ast.parse(src)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == "_push_approval"
    )
    read: set[str] = set()
    for node in ast.walk(fn):
        # payload.get("x")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
           and node.func.attr == "get" and node.args:
            a = node.args[0]
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                read.add(a.value)
        # payload["x"]
        elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
                and isinstance(node.slice.value, str):
            read.add(node.slice.value)

    extra = read - ALLOWED_PAYLOAD_READS
    assert not extra, (
        f"_push_approval now reads {sorted(extra)} from the approval payload. "
        f"Anything it reads is one line away from every registered device."
    )


def test_the_docstrings_name_three_consumers_not_two():
    """The claim that was wrong, pinned in both places it was made.

    'One false claim does not audit the sentence' — it appeared in
    serialize_pending AND in the REST route's comment, and only the first was
    noticed initially.
    """
    import prometheus.permissions.approval_queue as aq
    import prometheus.web.server as srv

    for mod in (aq, srv):
        text = Path(mod.__file__).read_text()
        assert "two transports" not in text.lower(), (
            f"{Path(mod.__file__).name} still claims two transports; the "
            f"approval wire shape has three consumers, the third being APNs "
            f"fan-out to every registered device"
        )

    assert "APNs" in Path(aq.__file__).read_text(), (
        "serialize_pending's docstring does not name the push consumer"
    )


def test_the_dispatcher_subscribes_to_everything():
    """Pins WHY the third consumer is easy to miss.

    The launcher subscribes it to "*", so it receives every signal kind
    rather than an enumerated list — there is no per-kind registration a
    reader would grep for and find.
    """
    from prometheus.web import launcher

    text = Path(launcher.__file__).read_text()
    assert 'subscribe("*", dispatcher.on_signal)' in text, (
        "the push dispatcher's bus subscription changed shape; re-check "
        "which signals reach every registered device"
    )
