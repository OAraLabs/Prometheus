"""`arguments_cut` on a served approval request (Beacon contract approval-arguments-cut, 2026-11-13).

Beacon desktop and iOS show a request's `arguments` and warn when the person is not seeing the whole call. They find a
cut value or an overflow from the view's own text, but cannot find an argument the daemon left out unless it says so.
The daemon sends one more fact on the serialized request (`GET /api/approvals`, the `approval_pending` frame, the
by-id route): `"arguments_cut": true`, only when something a person would want to see was NOT sent:

  * an argument named `_x` was left out (the view names it in its leading note, but a client reading flags wants one);
  * a value was cut (the same condition as the ` (+N more chars)` marker);
  * more arguments than fit (the `…` overflow row);
  * a worker's request carries fewer arguments than the worker had (`bound_for_wire`'s count of those not sent).

And ABSENT otherwise, never `false` (an older client reads absence as nothing, and a call with nothing left out is
byte-identical to before). Not a cut: one of the three validated desktop handles (`session`, `snapshot_id`,
`element_token`) left out, an empty value (the view shows it), and a value that was only MASKED (it is there, and
says `***`). The existing `critical` flag is unchanged: a masked value still makes a request critical.
"""

from __future__ import annotations

import asyncio

import pytest

from prometheus.permissions.approval_queue import ApprovalQueue
from prometheus.permissions.argument_view import (
    NOTE_KEY,
    OVERFLOW_KEY,
    WIRE_DROPPED_KEY,
    redact_arguments,
    render_arguments,
)


class _Gate:
    pass


class _Bus:
    def __init__(self) -> None:
        self.signals: list = []

    async def emit(self, signal) -> None:
        self.signals.append(signal)


def _queue() -> tuple[ApprovalQueue, _Bus]:
    q = ApprovalQueue(security_gate=_Gate(), telegram_adapter=None)
    bus = _Bus()
    q.signal_bus = bus
    return q, bus


async def _ask(q: ApprovalQueue, arguments, **kw) -> dict:
    before = len(q.signal_bus.signals)
    waiter = asyncio.create_task(q.request_approval("bash", "run it", arguments=arguments, **kw))
    for _ in range(50):
        await asyncio.sleep(0)
        if len(q.signal_bus.signals) > before:
            break
    payload = q.signal_bus.signals[-1].payload
    waiter.cancel()
    return payload


def _served(arguments, **kw) -> dict:
    q, _bus = _queue()
    return asyncio.run(_ask(q, arguments, **kw))


FAKE_KEY = "sk-" + "q7" * 16                                  # built here so no key-shaped literal is committed


# ── the contract's acceptance list ───────────────────────────────────────────

def test_a_left_out_underscore_argument_lists_the_rest_and_carries_the_flag():
    served = _served({"_hidden": "x", "command": "ls"})
    assert served["arguments"]["command"] == "ls" and served["arguments_cut"] is True


def test_a_validated_desktop_handle_left_out_is_plumbing_not_a_cut():
    served = _served({"element_token": "t", "text": "hi"})
    assert "arguments_cut" not in served and served["arguments"] == {"text": "hi"}


def test_a_call_whose_only_argument_is_left_out_still_carries_the_flag():
    """The shipped view puts its own note where the contract expected a null: the flag is true either way."""
    served = _served({"_hidden": "x"})
    assert served["arguments_cut"] is True
    assert list(served["arguments"]) == [NOTE_KEY]


def test_a_value_over_the_limit_is_cut_and_carries_the_flag_and_its_marker():
    served = _served({"command": "x" * 500})
    assert served["arguments_cut"] is True and "more char" in served["arguments"]["command"]


def test_a_block_over_the_total_is_cut_and_carries_the_flag():
    served = _served({f"k{i}": "v" * 150 for i in range(10)})
    assert served["arguments_cut"] is True
    assert any("more char" in text for key, text in served["arguments"].items() if key != NOTE_KEY)


def test_a_call_with_nothing_left_out_has_no_arguments_cut_key_at_all():
    served = _served({"command": "ls -la"})
    assert "arguments_cut" not in served
    assert _served(None).get("arguments_cut") is None and "arguments_cut" not in _served(None)


# ── what is not a cut ────────────────────────────────────────────────────────

def test_a_masked_value_is_critical_but_not_a_cut():
    served = _served({"api" + "_key": FAKE_KEY})
    assert served["critical"] is True                         # unchanged: masked still means "answer in the app"
    assert "arguments_cut" not in served
    assert NOTE_KEY in served["arguments"]                    # and the view still says it masked something


def test_an_empty_value_is_shown_not_left_out():
    served = _served({"command": "ls", "note": "", "extra": None, "items": []})
    assert "arguments_cut" not in served and "note" in served["arguments"]


def test_an_underscore_argument_that_is_not_a_plain_handle_is_shown_so_it_is_not_a_cut():
    served = _served({"_x": "rm -rf /tmp/build"})
    assert "arguments_cut" not in served and served["arguments"]["_x"] == "rm -rf /tmp/build"


def test_a_value_both_masked_and_cut_is_a_cut():
    served = _served({"command": "echo " + FAKE_KEY + " " + "z" * 400})
    assert served["arguments_cut"] is True


# ── the other ways a request is made ─────────────────────────────────────────

def test_a_workers_unsent_arguments_are_a_cut():
    served = _served({WIRE_DROPPED_KEY: 3, "command": "ls"}, worker_task_id="w1")
    assert served["arguments_cut"] is True
    assert "arguments_cut" not in _served({"command": "ls"}, worker_task_id="w1")


def test_a_desktop_prompt_says_so_too():
    from prometheus.computer.approvals import ComputerApprovalChannel

    channel = ComputerApprovalChannel(security_gate=_Gate(), people=object())
    bus = _Bus()
    channel.signal_bus = bus

    async def ask(arguments) -> dict:
        task = asyncio.ensure_future(channel.request_for_task(
            tool_name="computer_set_value", description="type it", extent=None, arguments=arguments,
            task_id="t1", session_id="web:abc"))
        await asyncio.sleep(0)
        payload = bus.signals[-1].payload
        task.cancel()
        return payload

    long_text = asyncio.run(ask({"session": "s1", "element_token": "tok", "text": "t" * 500}))
    assert long_text["arguments_cut"] is True
    plain = asyncio.run(ask({"session": "s2", "element_token": "tok", "text": "hi"}))
    assert "arguments_cut" not in plain


def test_the_list_and_the_frame_say_the_same_thing():
    q, bus = _queue()
    frame = asyncio.run(_ask(q, {"command": "x" * 500}))

    async def listed() -> list:
        waiter = asyncio.create_task(q.request_approval("bash", "again", arguments={"command": "x" * 500}))
        await asyncio.sleep(0)
        rows = [q.serialize_pending(a) for a in q.list_pending()]
        waiter.cancel()
        return rows

    rows = asyncio.run(listed())
    assert frame["arguments_cut"] is True and all(row["arguments_cut"] is True for row in rows)


def test_the_push_does_not_carry_it():
    import types

    from prometheus.push.apns import SendResult
    from prometheus.push.dispatcher import PushDispatcher

    served = _served({"_hidden": "x", "command": "ls"})
    sent: list[dict] = []

    class _Sender:
        async def send(self, **kwargs):
            sent.append(kwargs["payload"])
            return SendResult(outcome="ok", status=200)

    class _Store:
        def push_targets(self):
            return [types.SimpleNamespace(id="d1", apns_token="t", environment="sandbox", bundle_id="", push_failures=0)]

        def approval_push_targets(self, *, desktop):
            return self.push_targets()

    asyncio.run(PushDispatcher(_Store(), _Sender(), bridge=None).on_signal(
        types.SimpleNamespace(kind="approval_pending", payload=served, source="test")))
    assert sent and all("arguments_cut" not in body and "arguments" not in body for body in sent)


# ── the view's own account of what it left out ───────────────────────────────

@pytest.mark.parametrize(("arguments", "kwargs", "left_out"), [
    (None, {}, False),
    ({}, {}, False),
    ({"command": "ls"}, {}, False),
    ({"session": "s", "snapshot_id": "n", "element_token": "e"}, {}, False),
    ({"_x": "1", "command": "ls"}, {}, True),
    ({"_x": "rm -rf /tmp/x"}, {}, False),
    ({"command": "x" * 500}, {}, True),
    ({"api" + "_key": FAKE_KEY}, {}, False),
    ({f"a{i}": "v" for i in range(41)}, {}, True),
    ({WIRE_DROPPED_KEY: 2, "command": "ls"}, {"wire": True}, True),
    ({WIRE_DROPPED_KEY: 2, "command": "ls"}, {}, False),   # not a worker's request: just an argument
])
def test_what_the_view_reports_it_left_out(arguments, kwargs, left_out):
    rendered = render_arguments(arguments, **kwargs)
    assert rendered.left_out is left_out
    assert rendered.view == redact_arguments(arguments, **kwargs)           # one rendering, two ways to ask


def test_an_overflow_row_is_a_cut():
    rendered = render_arguments({f"a{i}": "v" for i in range(41)})
    assert rendered.view is not None and OVERFLOW_KEY in rendered.view and rendered.left_out is True
