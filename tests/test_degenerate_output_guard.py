"""#772 — the degenerate-output guard.

Fixtures are the REAL incident text from 2026-10-13 (qwen3.8-max on DashScope
at ~241k tokens), shortened, plus the legitimate shapes the letter test must
never drop (tables, number lists, CSV, hex dumps, code, non-Latin prose, a
short "OK").
"""

from __future__ import annotations

import pytest

from prometheus.engine.degeneration import (
    CONTROL_BYTES,
    LETTER_RATIO,
    check,
    detect,
)

# Breakdown 1 (240,772 tokens): visible text, verbatim prefix incl. its C0 bytes.
BREAKDOWN_1 = (
    "- Button0:35.6983.3898\n\n0.\n[16.1199/11,13.2092-21/22/5\n\n[/12/14/21,\n"
    "[3/2-1.14/4/8/\n\n[3.13/3.321/8/4/2/11/10/2/3/4/6/1/1/6]\n[1/.  5]   ,    / ] "
    "[ 2  2 — \n  -- -]/   /]:   / \n  [[ \x00  ]     [ [\x00  .26 2:( [8 [2 [[78\x00"
    " ...0 00    - **20 //5 / []:2[[]0 ]0  5 :[ Z0 0[ - (/  - (  :  -]02]}[-],] 222[ "
    "]]]]02][]:]_[ ]\x00]-][:]\x00][ ]_13:\x0019,```\n[1]:\x01\x01\x05 \x00 \x00"
)
# Breakdown 2 (244,459 tokens): the THINKING flood; visible text was "Report".
BREAKDOWN_2_THINKING = "- Button-4-8:21\n\n-01 " + " ".join(
    "0" * (1 + (i * 7) % 9) for i in range(900)
)
BREAKDOWN_2_TEXT = "Report"


def _table() -> str:
    head = "| run | failed | passed | skipped | errors | secs |\n|---|---|---|---|---|---|\n"
    return head + "".join(
        f"| {i} | {i * 3 % 29} | {13521 + i} | {710 + i} | {i % 5} | {987 + i * 2} |\n"
        for i in range(1, 15)
    )


LEGITIMATE = {
    "results table": _table(),
    "number list": ", ".join(str(i) for i in range(1, 201)),
    "csv": "\n".join(f"{i},{i * 1.5:.2f},{i * i},{(i * 7) % 13}" for i in range(60)),
    "hex dump": "\n".join(
        f"{i * 16:08x}: " + " ".join(f"{(i * 31 + j * 7) % 256:02x}" for j in range(16))
        for i in range(15)
    ),
    "code": "```python\n" + "\n".join(
        f"def f{i}(x):\n    return [y * {i} for y in range(x) if y % 3]" for i in range(12)
    ) + "\n```",
    "non-Latin prose": "これは日本語の文章です。モデルの出力は正常です。" * 12,
    "short ok": "OK",
    "short number": "200",
    "tabs and newlines": "col1\tcol2\r\nvalue\tvalue\n" * 20,
    "divider line": "Results\n" + "─" * 60 + "\nAll 4 checks passed; nothing to fix.\n",
    # One character IS dominant here (~87%), but letters are ~12%: the letter
    # floor is what keeps a banner like this from being dropped.
    "banner": "═" * 300 + "\nRelease notes for version nine point nine\n" + "═" * 10,
}


def test_breakdown_1_is_caught_by_its_control_bytes():
    found = detect(BREAKDOWN_1, "Button: \n")
    assert found is not None
    assert (found.test, found.where) == (CONTROL_BYTES, "text")


def test_breakdown_2_is_caught_in_the_thinking_block():
    # The visible text alone is a perfectly fine word; only the thinking shows it.
    assert detect(BREAKDOWN_2_TEXT, "") is None
    found = detect(BREAKDOWN_2_TEXT, BREAKDOWN_2_THINKING)
    assert found is not None
    assert (found.test, found.where) == (LETTER_RATIO, "thinking")


def test_breakdown_1_without_control_bytes_is_not_a_letter_ratio_hit():
    """Pins WHY the letter test needs its dominance condition: with its control
    bytes stripped, breakdown 1 has the same letter share as a results table, so
    any threshold that caught it on letters would also drop the table."""
    stripped = "".join(ch for ch in BREAKDOWN_1 if ch >= " " or ch in "\t\n\r")
    assert check(stripped) is None
    assert check(LEGITIMATE["results table"]) is None


@pytest.mark.parametrize("name", sorted(LEGITIMATE))
def test_legitimate_output_passes(name):
    content = LEGITIMATE[name]
    assert detect(content, "") is None, name
    assert detect("fine", content) is None, name


@pytest.mark.parametrize("byte", ["\x00", "\x01", "\x07", "\x0b", "\x0c", "\x1b", "\x1f"])
def test_any_c0_byte_except_tab_newline_cr_is_caught(byte):
    assert check(f"normal words {byte} more words")[0] == CONTROL_BYTES


def test_a_short_flood_is_below_the_floor():
    assert check("0 " * 50) is None          # 50 non-space chars
    assert check("0 " * 250)[0] == LETTER_RATIO


# ---------------------------------------------------------------------------
# The guard in the loop: drop, retry once, else end the turn — and record why
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402
import json  # noqa: E402
import sqlite3  # noqa: E402

from pydantic import BaseModel  # noqa: E402

from prometheus.engine.agent_loop import LoopContext, run_loop  # noqa: E402
from prometheus.engine.messages import (  # noqa: E402
    ConversationMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)
from prometheus.engine.usage import UsageSnapshot  # noqa: E402
from prometheus.providers.base import (  # noqa: E402
    ApiMessageCompleteEvent,
    ApiTextDeltaEvent,
    ModelProvider,
)
from prometheus.telemetry.tracker import ToolCallTelemetry  # noqa: E402
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult  # noqa: E402
from tests.support.doubles import register_double  # noqa: E402

SID = "beacon:degen"


class _In(BaseModel):
    n: int


class _CountTool(BaseTool):
    name = "count_tool"
    description = "counts"
    input_model = _In
    runs = 0

    async def execute(self, arguments, context):  # noqa: ANN001
        type(self).runs += 1
        return ToolResult(output=f"counted {arguments.n}")


def _msg(text: str = "", thinking: str = "", tool: bool = False) -> ConversationMessage:
    content = []
    if thinking:
        content.append(ThinkingBlock(thinking=thinking))
    if text:
        content.append(TextBlock(text=text))
    if tool:
        content.append(ToolUseBlock(id="t1", name="count_tool", input={"n": 1}))
    return ConversationMessage(role="assistant", content=content)


@register_double(
    "degenerate.scripted_provider",
    replaces="prometheus.providers.base.ModelProvider",
)
class _Script(ModelProvider):
    """Streams each scripted message's text as a delta (as a real provider
    does), then completes it with incident-sized usage."""

    def __init__(self, script: list[ConversationMessage]) -> None:
        self.script = list(script)
        self.requests: list = []

    async def stream_message(self, request):  # noqa: ANN001
        self.requests.append(request)
        msg = self.script.pop(0) if self.script else _msg("done")
        if msg.text:
            yield ApiTextDeltaEvent(text=msg.text)
        yield ApiMessageCompleteEvent(
            message=msg,
            usage=UsageSnapshot(input_tokens=240772, output_tokens=394,
                                cached_input_tokens=239616),
            stop_reason="stop", served_model="qwen3.8-max",
        )


def _ctx(provider, tel=None) -> LoopContext:
    reg = ToolRegistry()
    reg.register(_CountTool())
    return LoopContext(provider=provider, model="qwen3.8-max", system_prompt="sys",
                       max_tokens=128, tool_registry=reg, telemetry=tel)


def _drive(ctx, messages):
    events = []

    async def go():
        async for ev, _u in run_loop(ctx, messages, session_id=SID, surface="beacon"):
            events.append(ev)

    asyncio.run(go())
    return events


def _evidence(db) -> list[dict]:
    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT outcome, input_tokens, cached_input_tokens, model, round_index, "
        "session_id, summary_json FROM subsystem_runs "
        "WHERE subsystem='agent_loop' AND operation='degenerate_output' ORDER BY rowid"
    ).fetchall()
    conn.close()
    out = []
    for outcome, inp, cached, model, rnd, sid, summary in rows:
        out.append({"row_outcome": outcome, "input_tokens": inp, "cached": cached,
                    "model": model, "round_index": rnd, "session_id": sid,
                    **json.loads(summary or "{}")})
    return out


def _no_garbage(messages) -> bool:
    blob = json.dumps([m.model_dump() for m in messages])
    return "Button" not in blob and "\\u0000" not in blob


def test_a_degenerate_round_is_dropped_and_retried_once(tmp_path):
    from prometheus.engine.stream_events import RoundDiscarded

    tel = ToolCallTelemetry(tmp_path / "telemetry.db")
    provider = _Script([_msg(BREAKDOWN_1, "Button: \n"), _msg("All three checks pass.")])
    messages = [ConversationMessage.from_user_text("run the checks")]

    events = _drive(_ctx(provider, tel), messages)
    tel.close()

    # Nothing of the garbage reached history; the retry's answer did.
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[-1].text == "All three checks pass."
    assert _no_garbage(messages)
    # Exactly one retry, and the retry's request did not carry the garbage.
    assert len(provider.requests) == 2
    assert _no_garbage(provider.requests[1].messages)
    # The retry request differs from the first (a request-only nudge), so a
    # bad prompt-cache entry cannot simply be served again.
    assert provider.requests[1].system_prompt != provider.requests[0].system_prompt
    # Streaming clients are told to drop what they already showed.
    assert any(isinstance(e, RoundDiscarded) for e in events)
    # Evidence: one row for the drop, one for the recovery.
    ev = _evidence(tmp_path / "telemetry.db")
    assert [e["outcome"] for e in ev] == ["retried", "retry_ok"]
    first = ev[0]
    assert first["session_id"] == SID and first["round_index"] == 0
    assert (first["input_tokens"], first["cached"]) == (240772, 239616)
    assert first["model"] == "qwen3.8-max"
    assert first["served_model"] == "qwen3.8-max"
    assert first["provider"]
    assert (first["test"], first["where"]) == ("control_bytes", "text")
    assert "ctx_window" in first and "turn_id" in first


def test_two_degenerate_rounds_end_the_turn_with_a_clear_error(tmp_path):
    tel = ToolCallTelemetry(tmp_path / "telemetry.db")
    provider = _Script([
        _msg(BREAKDOWN_1),
        _msg(BREAKDOWN_2_TEXT, BREAKDOWN_2_THINKING),
    ])
    messages = [ConversationMessage.from_user_text("run the checks")]

    events = _drive(_ctx(provider, tel), messages)
    tel.close()

    assert len(provider.requests) == 2, "retried once, not more"
    assert _no_garbage(messages)
    last = messages[-1]
    assert last.role == "assistant"
    assert "unreadable" in last.text and "240,772" in last.text
    assert "qwen3.8-max" in last.text
    final = [e for e in events if type(e).__name__ == "AssistantTurnComplete"][-1]
    assert final.stop_reason == "degenerate_output"
    ev = _evidence(tmp_path / "telemetry.db")
    assert [e["outcome"] for e in ev] == ["retried", "turn_ended"]
    assert (ev[1]["test"], ev[1]["where"]) == ("letter_ratio", "thinking")
    conn = sqlite3.connect(tmp_path / "telemetry.db")
    [(reason,)] = conn.execute("SELECT forced_stop_reason FROM turns").fetchall()
    prose = " ".join(p or "" for (p,) in conn.execute("SELECT prose FROM responses"))
    conn.close()
    assert reason == "degenerate_output"
    # The dropped rounds' text is not kept in telemetry either.
    assert "Button" not in prose and "\x00" not in prose


def test_a_degenerate_round_never_dispatches_its_tool_calls(tmp_path):
    _CountTool.runs = 0
    provider = _Script([_msg(BREAKDOWN_1, tool=True), _msg("done, no tools")])
    messages = [ConversationMessage.from_user_text("go")]

    _drive(_ctx(provider), messages)

    assert _CountTool.runs == 0
    assert not any(m.tool_uses for m in messages)


def test_the_web_bridge_drops_streamed_garbage(tmp_path):
    from prometheus.engine.session import SessionManager
    from prometheus.web.ws_server import WebSocketBridge

    provider = _Script([_msg(BREAKDOWN_1), _msg("clean answer")])
    bridge = WebSocketBridge(session_mgr=SessionManager(), loop_context=_ctx(provider),
                             agent_state_ref={"state": "idle"})
    frames: list[dict] = []

    async def cap(ev):
        frames.append(ev)

    bridge.broadcast = cap
    session = bridge.session_mgr.get_or_create(SID)
    session.add_user_message("go")

    text, _ = asyncio.run(bridge._run_agent(SID, session))

    assert "Button" not in text and "\x00" not in text
    assert "clean answer" in text
    discard = next(f for f in frames if f.get("type") == "chat_discard")
    assert discard["payload"]["session_id"] == SID
    # Delivered only to clients bound to that session, like chat_delta.
    assert "chat_discard" in WebSocketBridge._SESSION_SCOPED_TYPES
    assert _no_garbage(session.messages)


# ---------------------------------------------------------------------------
# Readback, and background wakes
# ---------------------------------------------------------------------------


def _record(tel, sid, outcome, tokens, model="qwen3.8-max", provider="qwen"):
    tel.record_run(
        subsystem="agent_loop", operation="degenerate_output",
        outcome="failed" if outcome == "turn_ended" else "partial",
        session_id=sid, model=model, input_tokens=tokens, round_index=0,
        summary={"outcome": outcome, "provider": provider, "served_model": model,
                 "test": "control_bytes", "where": "text"},
    )


def test_degeneration_summary_by_model_and_provider(tmp_path):
    tel = ToolCallTelemetry(tmp_path / "telemetry.db")
    _record(tel, "a", "retried", 244459)
    _record(tel, "a", "turn_ended", 240772)
    _record(tel, "b", "retried", 30000, model="local-27b", provider="llama_cpp")
    # A recovery is not a breakdown: it must not count, nor lower the minimum.
    _record(tel, "a", "retry_ok", 1000)

    rows = {(r["model"], r["provider"]): r for r in tel.degeneration_summary()}
    tel.close()

    assert rows[("qwen3.8-max", "qwen")]["count"] == 2
    assert rows[("qwen3.8-max", "qwen")]["min_input_tokens"] == 240772
    assert rows[("local-27b", "llama_cpp")]["min_input_tokens"] == 30000
    assert all(r["last_at"] for r in rows.values())


def test_health_shows_degeneration_by_model(tmp_path, monkeypatch):
    from prometheus.gateway.commands import cmd_health
    from prometheus.telemetry import tracker

    tel = ToolCallTelemetry(tmp_path / "telemetry.db")
    _record(tel, "a", "turn_ended", 240772)
    monkeypatch.setattr(tracker, "_telemetry_singleton", tel)

    out = cmd_health()
    tel.close()

    assert "Degenerate output" in out
    assert "qwen3.8-max" in out and "240,772" in out


def test_session_last_round_dropped(tmp_path):
    tel = ToolCallTelemetry(tmp_path / "telemetry.db")
    assert tel.session_last_round_dropped("s") is False
    _record(tel, "s", "turn_ended", 240772)
    assert tel.session_last_round_dropped("s") is True
    # A later healthy round on the session clears it.
    tel.record_run(subsystem="agent_loop", operation="loop_round", outcome="success",
                   session_id="s", model="qwen3.8-max", input_tokens=1000)
    assert tel.session_last_round_dropped("s") is False
    # A drop that was retried successfully never blocked anything.
    _record(tel, "s2", "retried", 240772)
    _record(tel, "s2", "retry_ok", 240800)
    assert tel.session_last_round_dropped("s2") is False
    tel.close()


def test_a_background_wake_skips_a_session_whose_last_round_was_dropped(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from prometheus.tasks.completion_handler import TaskCompletionHandler
    from prometheus.telemetry import tracker

    tel = ToolCallTelemetry(tmp_path / "telemetry.db")
    monkeypatch.setattr(tracker, "_telemetry_singleton", tel)
    injected: list[str] = []

    async def inject(session_id, content, **kw):  # noqa: ANN001, ANN003
        injected.append(session_id)
        return "ok"

    handler = TaskCompletionHandler(signal_bus=SimpleNamespace(subscribe=lambda *a: None),
                                    inject_turn=inject)

    def signal(sid):
        return SimpleNamespace(payload={"on_complete": "reengage", "session_id": sid,
                                        "task_id": "t1", "status": "completed"})

    _record(tel, "beacon:bad", "turn_ended", 240772)
    asyncio.run(handler._on_signal(signal("beacon:bad")))
    asyncio.run(handler._on_signal(signal("beacon:fine")))
    tel.close()

    assert injected == ["beacon:fine"]


def test_evidence_uses_the_compactors_own_threshold_resolver(tmp_path):
    """When the compactor resolves thresholds itself (#773), the evidence row
    records ITS answer and source — not a re-derivation that can drift."""

    class _Compactor:
        def limit_for(self, model=None, backend=None):  # noqa: ANN001
            return 1_000_000

        def is_measured(self, model=None, backend=None):  # noqa: ANN001
            return False

        def threshold_for(self, model=None, backend=None):  # noqa: ANN001
            return 200_000, "model tokens"

        async def apply(self, messages, **kw):  # noqa: ANN001, ANN003
            return messages

    tel = ToolCallTelemetry(tmp_path / "telemetry.db")
    ctx = _ctx(_Script([_msg(BREAKDOWN_1), _msg("fine")]), tel)
    ctx.compactor = _Compactor()
    _drive(ctx, [ConversationMessage.from_user_text("go")])
    tel.close()

    first = _evidence(tmp_path / "telemetry.db")[0]
    assert (first["compaction_threshold"], first["compaction_threshold_source"]) == (
        200_000, "model tokens")
    assert first["ctx_window"] == 1_000_000
