"""Web command parity + the command-reply terminal frame (SPRINT-WEB-PARITY).

Piece 3 of the web-parity arc. #255 measured the gap and landed it red; this
closes it. Three distinct defects shared one root — the web surface was never a
platform any guard watched:

  * /reset was absent entirely, so an operator whose turn died with a poisoned
    context could not clear the session from the surface he was on;
  * six commands were registered on Telegram but in NEITHER the shared dispatch
    tables NOR WEB_NATIVE_ONLY, so route_slash fell through and the agent ate
    them as chat text — no reply, no effect, no boundary message;
  * a command reply emitted no terminal frame, so every replying command
    false-stalled a client watching for one.

Tested by OUTCOME: what the surface DOES with each command, not which table it
happens to be in.
"""

from __future__ import annotations

import asyncio
import json

from prometheus.engine.session import SessionManager
from prometheus.web.slash_router import (
    WEB_NATIVE_ONLY,
    build_command_context,
    route_slash,
)
from prometheus.web.ws_server import WebSocketBridge


class _FakeGate:
    def __init__(self) -> None:
        self.mode = "ask"

    def list_grants(self):
        return []

    def current_mode(self):
        return self.mode

    def set_mode(self, mode):
        self.mode = mode


class _FakeQueue:
    def __init__(self) -> None:
        self._security_gate = _FakeGate()

    def pending(self):
        return []


def _ctx(session=None, session_id="web:t", queue=None):
    return build_command_context(
        None, {}, session=session, session_id=session_id,
        approval_queue=queue if queue is not None else _FakeQueue(),
    )


# ---------------------------------------------------------------------------
# The six
# ---------------------------------------------------------------------------

NEWLY_WIRED = ["ephemeral", "gate", "grants", "remember", "revoke"]


def test_the_five_dispatch_instead_of_being_eaten_by_the_agent():
    """handled=True with a real reply — NOT a fall-through to the agent.

    ``handled=False`` is the defect: route_slash returns it for an unknown
    token, and the caller then runs the message through the agent as chat text.
    """
    mgr = SessionManager()
    session = mgr.get_or_create("web:t")
    for name in NEWLY_WIRED:
        outcome = asyncio.run(route_slash(f"/{name}", _ctx(session)))
        assert outcome.handled, f"/{name} still falls through to the agent"
        assert outcome.reply, f"/{name} dispatched but replied with nothing"
        assert "isn't available on the web chat surface" not in outcome.reply


def test_qwen_gets_its_siblings_boundary_reply():
    """/qwen is a provider override; all ten siblings are deferred.

    Wiring it alone would have split one class across two behaviours. It fell
    through only because #183-#186 added the command without listing it.
    """
    assert "qwen" in WEB_NATIVE_ONLY
    outcome = asyncio.run(route_slash("/qwen", _ctx()))
    assert outcome.handled
    assert "isn't available on the web chat surface" in outcome.reply
    for sibling in ("claude", "gpt", "deepseek", "kimi", "glm", "mimo"):
        assert sibling in WEB_NATIVE_ONLY


def test_every_telegram_command_is_handled_or_explicitly_refused():
    """No command may fall through. The chart asserts this too; here it is
    stated as behaviour rather than as a manifest property."""
    import re
    import pathlib
    from prometheus.gateway import commands as C

    src = pathlib.Path("src/prometheus/gateway/telegram.py").read_text()
    telegram = {n for n, _ in re.findall(
        r'CommandHandler\(\s*"([\w-]+)",\s*self\.(\w+)\s*\)', src)}
    fell = sorted(
        n for n in telegram
        if not (C.is_formatter_command(n) or C.is_session_command(n))
        and n not in WEB_NATIVE_ONLY
    )
    assert not fell, "silently swallowed on web: " + ", ".join("/" + c for c in fell)


# ---------------------------------------------------------------------------
# /reset — the ruling, as behaviour
# ---------------------------------------------------------------------------


def test_reset_clears_history_and_keeps_the_session_listed():
    """/reset is NOT Beacon's Forget session, and the difference is the point.

    Forget (DELETE /api/sessions/{id}) pops the entry so the thread vanishes
    from GET /api/sessions. /reset empties the thread and leaves it there. Both
    clear a poisoned in-memory history; only one is what a user who typed
    /reset asked for, so routing /reset at Forget would silently destroy a
    thread the user meant to keep.
    """
    mgr = SessionManager()
    session = mgr.get_or_create("web:t")
    session.add_user_message("one")
    session.add_user_message("two")
    assert len(session.messages) == 2

    outcome = asyncio.run(route_slash("/reset", _ctx(session)))

    assert outcome.handled
    assert session.messages == []          # history cleared
    assert "web:t" in mgr._sessions        # session SURVIVES — not a Forget
    # The watermark must retreat with it, or post-reset messages look
    # already-persisted and are dropped.
    assert session._lcm_persisted_len == 0


def test_clear_is_an_alias_of_reset():
    mgr = SessionManager()
    session = mgr.get_or_create("web:t")
    session.add_user_message("one")
    asyncio.run(route_slash("/clear", _ctx(session)))
    assert session.messages == []
    assert "web:t" in mgr._sessions


# ---------------------------------------------------------------------------
# The terminal frame
# ---------------------------------------------------------------------------


def _frames_for(content: str) -> list[dict]:
    bridge = WebSocketBridge(
        session_mgr=SessionManager(), loop_context=None, config={},
        approval_queue=_FakeQueue(),
    )
    frames: list[dict] = []

    async def cap(ev):
        frames.append(ev)

    bridge.broadcast = cap
    asyncio.run(bridge._handle_send_message("web:t", content))
    return frames


def test_command_reply_emits_a_terminal_frame():
    frames = _frames_for("/help")
    types = [f["type"] for f in frames]
    assert types == ["chat_message", "chat_message", "command_done"], types

    done = frames[-1]
    assert done["payload"]["session_id"] == "web:t"
    assert done["payload"]["command_name"] == "help"
    # The bool lives on the assistant chat_message; the NAME lives here.
    assert frames[1]["payload"]["command"] is True
    assert "command" not in done["payload"], (
        "command_done must not carry a key that is a bool one frame earlier"
    )
    assert done["payload"]["transient"] is True
    # Ties the terminator to the assistant frame it closes.
    assert done["payload"]["message_id"] == frames[1]["payload"]["message_id"]
    assert json.dumps(done)  # serialisable as sent


def test_boundary_replies_terminate_too():
    """The 25 deferred commands reply and must close their exchange as well —
    the stall was never about which commands were WIRED."""
    frames = _frames_for("/route")
    assert [f["type"] for f in frames][-1] == "command_done"
    assert frames[-1]["payload"]["command_name"] == "route"


def test_command_reply_never_emits_chat_done():
    """The ruling, guarded.

    chat_done means "the assistant turn identified by message_id finished".
    /steer and /queue are dispatched WHILE a turn streams, so emitting one here
    would stand a client's watchdog down for a turn still running — strictly
    worse than the defect being fixed.
    """
    for content in ("/help", "/route", "/steer focus on X", "/reset"):
        types = [f["type"] for f in _frames_for(content)]
        assert "chat_done" not in types, f"{content} emitted chat_done"
        assert types[-1] == "command_done", f"{content} did not terminate"
