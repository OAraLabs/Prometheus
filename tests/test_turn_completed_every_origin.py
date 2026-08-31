"""turn_completed must be raised by every origin, not just Telegram (#346).

The push dispatcher maps `turn_completed → TURN` and delivers it to devices with NO live socket —
exactly the backgrounded-phone case. But the signal was emitted in ONE place, gateway/telegram.py,
so a turn started from Beacon or REST completed and raised nothing. The phone stayed silent and
no error appeared anywhere, because nothing had failed: the inert-but-green shape.
"""

from __future__ import annotations

import pytest

from prometheus.web.ws_server import WebSocketBridge


class _Bus:
    def __init__(self):
        self.emitted = []

    async def emit(self, signal):
        self.emitted.append(signal)

    def subscribe(self, *a, **k):
        pass


def _bridge(bus):
    b = WebSocketBridge.__new__(WebSocketBridge)   # no full construction needed for this seam
    b.signal_bus = bus
    return b


@pytest.mark.asyncio
async def test_a_completed_turn_raises_the_signal(tmp_path):
    bus = _Bus()
    await _bridge(bus)._emit_turn_completed("beacon:s1", "All done — the file is written.")
    assert len(bus.emitted) == 1, "a web turn must raise turn_completed, like Telegram's does"
    sig = bus.emitted[0]
    assert sig.kind == "turn_completed"
    assert sig.payload["session_id"] == "beacon:s1"


@pytest.mark.asyncio
async def test_the_summary_is_the_replys_first_line(tmp_path):
    """_push_quiet prefers `summary` over the session id, so the phone says WHAT finished."""
    bus = _Bus()
    await _bridge(bus)._emit_turn_completed("s", "First line here.\nSecond line ignored.")
    assert bus.emitted[0].payload["summary"] == "First line here."


@pytest.mark.asyncio
async def test_a_long_reply_is_clipped_not_sent_whole(tmp_path):
    bus = _Bus()
    await _bridge(bus)._emit_turn_completed("s", "x" * 500)
    assert len(bus.emitted[0].payload["summary"]) == 200


@pytest.mark.asyncio
async def test_an_empty_reply_still_raises_with_an_empty_summary(tmp_path):
    """The push falls back to the session id — a turn that said nothing still finished."""
    bus = _Bus()
    await _bridge(bus)._emit_turn_completed("s", "   ")
    assert bus.emitted[0].payload["summary"] == ""
    assert bus.emitted[0].payload["session_id"] == "s"


@pytest.mark.asyncio
async def test_no_bus_is_a_no_op_not_a_crash(tmp_path):
    await _bridge(None)._emit_turn_completed("s", "hi")   # must not raise


@pytest.mark.asyncio
async def test_a_broken_bus_never_breaks_the_turn(tmp_path):
    """The Telegram emitter swallows bus failures for this reason; so does this one."""
    class _Broken:
        async def emit(self, signal):
            raise RuntimeError("bus down")
    await _bridge(_Broken())._emit_turn_completed("s", "hi")   # must not raise


@pytest.mark.asyncio
async def test_the_dispatcher_turns_that_signal_into_a_TURN_push(tmp_path):
    """End of the chain: the signal the bridge now raises is one the dispatcher acts on."""
    from prometheus.push.dispatcher import PushDispatcher

    sent = []

    class _Sender:
        async def send(self, **kw):
            sent.append(kw)
            class R: outcome, status = "ok", 200
            return R()

    class _Store:
        def push_targets(self):
            class T: id, apns_token, environment, bundle_id, push_failures = "d1", "tok", "production", "b", 0
            return [T()]
        def reset_push_failures(self, *a): pass
        def record_push_success(self, *a): pass

    class _Bridge:
        _ws_identity: dict = {}

    d = PushDispatcher(_Store(), _Sender(), _Bridge())
    bus = _Bus()
    await _bridge(bus)._emit_turn_completed("beacon:s1", "the reply")
    await d.on_signal(bus.emitted[0])

    assert len(sent) == 1, "the emitted signal produced a push"
    alert = sent[0]["payload"]["aps"]
    assert alert["category"] == "TURN"
    assert alert["alert"]["body"] == "the reply", "and it says what finished, not the session id"


# ── the wiring guard ────────────────────────────────────────────────────────────────────────
# Every test above calls _emit_turn_completed directly, so all of them pass even if the turn
# path never calls it — which is the ORIGINAL DEFECT. Mutation testing caught that: deleting the
# call site left 7/7 green. These pin the wiring, the way test_profile_wiring pins the resolver.

def _bridge_source_tree():
    import ast
    import inspect

    import prometheus.web.ws_server as mod
    return ast.parse(inspect.getsource(mod)), mod


def test_the_turn_completion_path_actually_calls_the_emitter():
    import ast

    tree, _ = _bridge_source_tree()
    callers = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        names = {
            n.func.attr for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        if "_emit_turn_completed" in names and fn.name != "_emit_turn_completed":
            callers.append((fn.name, names))
    assert callers, "nothing calls _emit_turn_completed — the signal is defined and never raised"
    # It belongs on the completion path, beside the Live Activity end — not somewhere incidental.
    assert any("_end_live_activity" in names for _, names in callers), (
        "the emitter is called, but not from the turn-completion path"
    )


def test_the_interrupt_path_does_not_claim_completion():
    """"Turn complete" is false for a turn the user stopped. The asymmetry is deliberate."""
    import inspect

    import prometheus.web.ws_server as mod

    src = inspect.getsource(mod)
    # exactly one call site: the completion path. A second would mean the interrupt path
    # regained it, which would push a completion notice for something that did not complete.
    assert src.count("await self._emit_turn_completed(") == 1, (
        "expected exactly one emit call site (the completion path)"
    )
