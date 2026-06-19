"""Web slash-command router — parse, dispatch, /note side-effect, boundaries.

Mirrors the Telegram command tests at the router level: a formatter command
runs and returns text, /note writes a manual fact, a mutating command gets a
boundary reply (not a silent drop), and unknown/non-commands fall through to
the agent.
"""

from __future__ import annotations

import sys
import types

# Bypass the prometheus.memory circular-import chain (same shim as test_wiki /
# test_manual_layer) — the /note test touches prometheus.memory.store.
if "prometheus.memory" not in sys.modules:
    _pkg = types.ModuleType("prometheus.memory")
    _pkg.__path__ = ["src/prometheus/memory"]
    _pkg.__package__ = "prometheus.memory"
    sys.modules["prometheus.memory"] = _pkg

import pytest  # noqa: E402

from prometheus.gateway.commands import (  # noqa: E402
    CommandContext,
    formatter_command_names,
)
from prometheus.web.slash_router import (  # noqa: E402
    WEB_NATIVE_ONLY,
    build_command_context,
    parse_slash,
    route_slash,
)


# --- parse_slash -----------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("/help", ("help", "")),
    ("/note @Pham started a clinic", ("note", "@Pham started a clinic")),
    ("/STATUS", ("status", "")),                  # command name lowercased
    ("  /help  ", ("help", "")),                  # outer whitespace tolerated
    ("/skills show foo", ("skills", "show foo")),
    ("/note   @X   spaced", ("note", "@X   spaced")),  # inner spacing kept
])
def test_parse_slash_commands(text, expected):
    assert parse_slash(text) == expected


@pytest.mark.parametrize("text", [
    "hello world",          # no slash
    "/",                    # bare slash
    "/ leading-space",      # space right after the slash
    "/home/will/file.txt",  # file path, not a command
    "/3/4",                 # fraction, not a command
    "",                     # empty
])
def test_parse_slash_non_commands(text):
    assert parse_slash(text) is None


# --- routing: formatter / boundary / fall-through --------------------------

@pytest.mark.asyncio
async def test_help_is_handled():
    out = await route_slash("/help", CommandContext())
    assert out.handled
    assert "Prometheus" in out.reply


@pytest.mark.asyncio
async def test_status_handled_with_injected_registry():
    class _Reg:
        def list_tools(self):
            return ["a", "b", "c"]

    ctx = CommandContext(
        model_name="gemma-4", model_provider="llama_cpp", tool_registry=_Reg()
    )
    out = await route_slash("/status", ctx)
    assert out.handled
    assert "Prometheus Status" in out.reply
    assert "gemma-4" in out.reply


@pytest.mark.asyncio
async def test_mutating_command_gets_boundary_not_silent():
    out = await route_slash("/steer go left after the next tool call", CommandContext())
    assert out.handled                       # handled => NOT dropped into the agent
    assert "isn't available on the web" in out.reply
    assert "steer" in out.reply


@pytest.mark.asyncio
async def test_unknown_command_falls_through_to_agent():
    out = await route_slash("/wibble frobnicate", CommandContext())
    assert out.handled is False
    assert out.reply is None


@pytest.mark.asyncio
async def test_plain_message_falls_through():
    out = await route_slash("what is the capital of France?", CommandContext())
    assert out.handled is False


@pytest.mark.asyncio
async def test_note_writes_manual_fact_via_router(tmp_path, monkeypatch):
    """/note through the web router writes ONE manual fact (manual=1)."""
    import prometheus.tools.builtin.wiki_compile as wc
    from prometheus.memory.store import MemoryStore

    store = MemoryStore(db_path=tmp_path / "memory.db")
    monkeypatch.setattr(wc, "_memory_store", store, raising=False)

    out = await route_slash(
        "/note @Web a manual fact from the web surface", CommandContext()
    )
    assert out.handled
    assert "Web" in out.reply

    rows = store._conn.execute(
        "SELECT entity_name, fact, manual FROM memories"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["manual"] == 1
    assert rows[0]["entity_name"] == "Web"
    store.close()


# --- build_command_context -------------------------------------------------

def test_build_command_context_extracts_from_loop_context():
    class _Prov:
        value = "anthropic"

    class _LC:
        model = "claude-x"
        provider = _Prov()
        system_prompt = "you are..."
        tool_registry = object()

    ctx = build_command_context(_LC(), {"web": {"enabled": True}})
    assert ctx.model_name == "claude-x"
    assert ctx.model_provider == "anthropic"
    assert ctx.system_prompt == "you are..."
    assert ctx.config == {"web": {"enabled": True}}
    assert ctx.tool_registry is not None


def test_build_command_context_handles_none_loop_context():
    ctx = build_command_context(None, None)
    assert ctx.model_name == ""
    assert ctx.tool_registry is None


def test_native_only_disjoint_from_formatters():
    """A command is either a formatter or native-only — never both."""
    assert not (WEB_NATIVE_ONLY & formatter_command_names())


# --- bridge seam: interception happens before session + agent --------------

@pytest.mark.asyncio
async def test_bridge_intercepts_slash_before_session_and_agent():
    """_handle_send_message handles a slash command WITHOUT touching the session
    manager or running the agent, and broadcasts the echo + reply."""
    from unittest.mock import MagicMock

    from prometheus.web.ws_server import WebSocketBridge

    session_mgr = MagicMock()
    bridge = WebSocketBridge(session_mgr=session_mgr, loop_context=None, config={})

    frames: list[dict] = []

    async def _capture(frame):
        frames.append(frame)

    bridge.broadcast = _capture

    await bridge._handle_send_message("sess-1", "/help", client_msg_id="c1")

    # The session manager (and therefore the agent) was never engaged.
    session_mgr.get_or_create.assert_not_called()

    # A user echo then an assistant reply carrying the help text were broadcast.
    roles = [f["payload"].get("role") for f in frames if f["type"] == "chat_message"]
    assert roles == ["user", "assistant"]
    asst = next(f for f in frames if f["payload"].get("role") == "assistant")
    assert "Prometheus" in asst["payload"]["content"]
    assert asst["payload"].get("command") is True


@pytest.mark.asyncio
async def test_bridge_passes_plain_message_through_to_session():
    """A non-command message is NOT intercepted — it reaches the session path."""
    from unittest.mock import MagicMock

    from prometheus.web.ws_server import WebSocketBridge

    session = MagicMock()
    session.add_user_message.return_value = 0
    session.last_persisted_row_id.return_value = 1
    session_mgr = MagicMock()
    session_mgr.get_or_create.return_value = session
    bridge = WebSocketBridge(session_mgr=session_mgr, loop_context=None, config={})

    async def _capture(frame):
        pass

    bridge.broadcast = _capture

    await bridge._handle_send_message("sess-1", "hello there", client_msg_id="c1")

    # Not a command => the normal path ran and engaged the session.
    session_mgr.get_or_create.assert_called_once()
