"""A restart restores what fits by the SAME estimate the compactor uses.

engine/session.py sized the restore window with its own chars/4-of-content_json
copies: ``_estimated_tokens`` (the newest-turn restore) and the per-row cost in
``rehydrate_if_cold``. Since #595 the shared estimate
(``context.token_estimation.estimate_message_tokens``) leaves out thinking no
request builder sends back — an unsigned ThinkingBlock, which Ollama (#592),
openai_compat and stub keep on assistant messages. The session copies still
charged it, so a thinking-heavy session restored fewer turns after a restart
than the 8,000-token window holds, for tokens the model never sees.

The tests marked "was short" fail before the change; the controls pass before
and after.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from prometheus.context.token_estimation import estimate_message_tokens
from prometheus.engine.messages import (
    ConversationMessage,
    ImageBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from prometheus.engine.session import (
    _REHYDRATE_TOKEN_BUDGET,
    _REHYDRATE_WINDOW,
    SessionManager,
    _estimated_tokens,
    _message_from_part,
)
from prometheus.memory.lcm_engine import LCMEngine

SID = "desktop:restore-shared-estimate"
THOUGHT = "Let me reason about this carefully before answering.\n" * 60     # ~3.2k chars


def _engine(tmp_path: Path) -> LCMEngine:
    return LCMEngine(MagicMock(), db_path=tmp_path / "lcm.db")


def _manager(engine: LCMEngine, *, rehydrate: bool = True) -> SessionManager:
    mgr = SessionManager()
    mgr.lcm_engine = engine
    mgr.rehydrate_enabled = rehydrate
    return mgr


def _seed(engine: LCMEngine, messages: list[ConversationMessage], sid: str = SID) -> None:
    """One daemon lifetime writing ``messages`` through the real persist path."""
    session = _manager(engine, rehydrate=False).get_or_create(sid)
    session.messages = list(messages)
    session.persist_loop_result(0)


def _restore(tmp_path: Path, messages: list[ConversationMessage]) -> list[ConversationMessage]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    engine = _engine(tmp_path)
    _seed(engine, messages)
    mgr = _manager(engine)
    mgr.rehydrate_if_cold(SID)
    return mgr.get_or_create(SID).messages


def _turns(n: int, *, thought: str = "", filler: int = 600) -> list[ConversationMessage]:
    out: list[ConversationMessage] = []
    for i in range(n):
        out.append(ConversationMessage.from_user_text(f"q{i}: " + "x" * filler))
        blocks = [ThinkingBlock(thinking=thought)] if thought else []
        out.append(ConversationMessage(
            role="assistant", content=[*blocks, TextBlock(text=f"a{i}: " + "y" * filler)]))
    return out


def _stored_rows(tmp_path: Path, messages: list[ConversationMessage]):
    tmp_path.mkdir(parents=True, exist_ok=True)
    engine = _engine(tmp_path)
    _seed(engine, messages)
    rows, _ = engine.conversation_store.messages_page(limit=10_000, session_id=SID)
    return engine, rows


def _old_window(costs: list[int]) -> int:
    """How many of the newest rows the ORIGINAL budget loop kept, given each
    row's cost under the original formula (oldest first)."""
    budget, kept = _REHYDRATE_TOKEN_BUDGET, 0
    for cost in reversed(costs[-_REHYDRATE_WINDOW:]):
        if kept and budget - cost < 0:
            break
        budget -= cost
        kept += 1
    return kept


# -- the shared estimate -----------------------------------------------------

def test_a_message_is_costed_by_the_shared_estimate():
    """Was short: the copy charged the unsent thought."""
    with_thought = ConversationMessage(role="assistant", content=[
        ThinkingBlock(thinking=THOUGHT), TextBlock(text="Done.")])
    assert _estimated_tokens(with_thought) == max(1, estimate_message_tokens(with_thought))
    assert _estimated_tokens(with_thought) == _estimated_tokens(
        ConversationMessage(role="assistant", content=[TextBlock(text="Done.")]))


def test_control_a_message_without_unsent_thinking_costs_what_it_did():
    for msg in (
        ConversationMessage.from_user_text("hello"),
        ConversationMessage(role="assistant", content=[
            ThinkingBlock(thinking=THOUGHT, signature="sig"), TextBlock(text="Done.")]),
        ConversationMessage(role="user", content=[ToolResultBlock(tool_use_id="t", content="z" * 900)]),
        ConversationMessage(role="assistant", content=[]),
    ):
        assert _estimated_tokens(msg) == max(1, len(msg.content_json) // 4)


# -- the restore window --------------------------------------------------------

def test_a_thinking_session_restores_as_many_turns_as_the_same_session_without_it(tmp_path: Path):
    """Was short: each unsent thought cost ~800 tokens of the 8,000-token window,
    so the newest-first budget stopped several turns early."""
    plain = _restore(tmp_path / "plain", _turns(10))
    thinking = _restore(tmp_path / "thinking", _turns(10, thought=THOUGHT))
    assert len(plain) == 20                                     # the window holds all ten turns
    assert [m.text for m in thinking] == [m.text for m in plain]
    # The restored messages keep their thoughts: only the COST leaves them out.
    assert sum(isinstance(b, ThinkingBlock) for m in thinking for b in m.content) == 10


def test_a_long_thinking_turn_is_not_cut_for_thinking_the_model_never_sees(tmp_path: Path):
    """Was short: the newest-turn restore (a turn longer than the 40-row window)
    left out rounds to make room for unsent thoughts, and said so in a note."""
    rounds: list[ConversationMessage] = []
    for i in range(30):
        rounds += [
            ConversationMessage(role="assistant", content=[
                ThinkingBlock(thinking=THOUGHT[:1200]),
                ToolUseBlock(id=f"t{i}", name="bash", input={"command": f"step {i}"})]),
            ConversationMessage(role="user", content=[
                ToolResultBlock(tool_use_id=f"t{i}", content=f"ok {i}")]),
        ]
    turn = [ConversationMessage.from_user_text("migrate the schema"), *rounds,
            ConversationMessage(role="assistant", content=[TextBlock(text="migration done")])]
    restored = _restore(tmp_path, turn)
    assert restored[0].content[0].text == "migrate the schema"
    assert not any("left out" in getattr(b, "text", "") for b in restored[0].content[1:])
    assert len(restored) == len(turn)
    assert sum(_estimated_tokens(m) for m in restored) <= _REHYDRATE_TOKEN_BUDGET


def test_control_a_session_without_thinking_restores_exactly_as_before(tmp_path: Path):
    # Rows of ~270 tokens: the 40-row window holds ~10,800, so the 8,000-token
    # budget cuts it — the restored set is decided by every row's cost, and must
    # be exactly what the original formula decided.
    messages = _turns(20, filler=1_050)
    _, rows = _stored_rows(tmp_path / "rows", messages)
    kept = _old_window([max(1, len(r.content_json) // 4) for r in rows])
    assert kept < _REHYDRATE_WINDOW                              # the budget really cut it
    expected = [_message_from_part(r) for r in rows[-kept:]]
    expected = expected[next(i for i, m in enumerate(expected) if m.role == "user"):]
    restored = _restore(tmp_path / "restore", messages)
    assert [m.text for m in restored] == [m.text for m in expected]


def test_control_a_legacy_row_costs_its_text_as_before(tmp_path: Path):
    # A row stored before content_json existed holds only flat text: no thought
    # to leave out, so it keeps the original cost (costing it as the rebuilt
    # message would add the JSON framing and restore fewer rows).
    import sqlite3

    messages = _turns(20, filler=1_050)                           # 40 rows of ~260 tokens
    tmp_path.mkdir(parents=True, exist_ok=True)
    engine = _engine(tmp_path)
    _seed(engine, messages)
    con = sqlite3.connect(tmp_path / "lcm.db")
    con.execute("UPDATE lcm_messages SET content_json = NULL")
    con.commit()
    con.close()
    rows, _ = engine.conversation_store.messages_page(limit=10_000, session_id=SID)
    assert all(not r.content_json for r in rows)
    kept = _old_window([max(1, len(r.content or "") // 4) for r in rows])
    assert kept < _REHYDRATE_WINDOW                              # the budget really cut it
    mgr = _manager(engine)
    mgr.rehydrate_if_cold(SID)
    restored = mgr.get_or_create(SID).messages
    expected = rows[-kept:]
    expected = expected[next(i for i, r in enumerate(expected) if r.role == "user"):]
    assert [m.text for m in restored] == [r.content for r in expected]


def test_a_legacy_row_is_costed_as_its_flat_text(tmp_path: Path):
    import sqlite3

    from prometheus.engine.session import _row_tokens

    engine, _ = _stored_rows(tmp_path, _turns(2, filler=333))
    con = sqlite3.connect(tmp_path / "lcm.db")
    con.execute("UPDATE lcm_messages SET content_json = NULL")
    con.commit()
    con.close()
    rows, _ = engine.conversation_store.messages_page(limit=10, session_id=SID)
    assert [_row_tokens(r, _message_from_part(r)) for r in rows] == [
        max(1, len(r.content) // 4) for r in rows]
    # An empty legacy row still costs 1: every restored message costs something.
    from types import SimpleNamespace

    empty = SimpleNamespace(role="assistant", content="", content_json=None)
    assert _row_tokens(empty, _message_from_part(empty)) == 1


def test_a_thinking_row_costs_exactly_what_the_same_row_costs_without_it(tmp_path: Path):
    from prometheus.engine.session import _row_tokens

    _, plain = _stored_rows(tmp_path / "plain", _turns(3))
    _, thinking = _stored_rows(tmp_path / "thinking", _turns(3, thought=THOUGHT))
    assert [_row_tokens(r, _message_from_part(r)) for r in thinking] == [
        _row_tokens(r, _message_from_part(r)) for r in plain]


def test_a_degraded_row_costs_what_it_restores_to(tmp_path: Path):
    # An upload whose cached file was evicted restores as a placeholder; it is
    # costed as that placeholder — what the model is actually sent — not as the
    # stored reference.
    picture = tmp_path / "x.png"
    picture.write_bytes(b"\x89PNG" + b"0" * 2_000)
    upload = ConversationMessage(role="user", content=[
        TextBlock(text="what is this?"),
        ImageBlock(media_type="image/png", data="iVBORw0KGgo=", source_path=str(picture))])
    _, rows = _stored_rows(tmp_path / "db", [upload, ConversationMessage(
        role="assistant", content=[TextBlock(text="a chart")])])
    picture.unlink()
    from prometheus.engine.session import _row_tokens

    row = next(r for r in rows if r.role == "user")
    restored = _message_from_part(row)
    assert "[Image: unavailable]" in restored.content_json
    assert _row_tokens(row, restored) == _estimated_tokens(restored)
