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
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from prometheus.engine.session import _REHYDRATE_TOKEN_BUDGET, SessionManager, _estimated_tokens
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


def _turns(n: int, *, thought: str = "") -> list[ConversationMessage]:
    out: list[ConversationMessage] = []
    for i in range(n):
        out.append(ConversationMessage.from_user_text(f"q{i}: " + "x" * 600))
        blocks = [ThinkingBlock(thinking=thought)] if thought else []
        out.append(ConversationMessage(
            role="assistant", content=[*blocks, TextBlock(text=f"a{i}: " + "y" * 600)]))
    return out


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
    # 14 turns of ~330 tokens: the window keeps the newest ~12, as it always did.
    restored = _restore(tmp_path, _turns(14))
    budget, kept = _REHYDRATE_TOKEN_BUDGET, 0
    for m in reversed(_turns(14)):
        cost = max(1, len(m.content_json) // 4)
        if kept and budget - cost < 0:
            break
        budget -= cost
        kept += 1
    expected = _turns(14)[-kept:]
    start = next(i for i, m in enumerate(expected) if m.role == "user")
    assert [m.text for m in restored] == [m.text for m in expected[start:]]
