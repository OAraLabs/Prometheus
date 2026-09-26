"""The context estimate counts what is sent — not thinking no wire sends back.

#592 (WP-X.35) keeps an Ollama thinking model's reasoning as a ThinkingBlock on
the assistant message; openai_compat and the stub parser already did. Those
blocks are unsigned (only Anthropic signs), and no request builder ever sends
an unsigned thinking block back: the OpenAI-shape builder skips all thinking
(``stub._build_openai_messages``), and the Anthropic builder sends one only
when it carries a signature (``anthropic.py``). The estimator still counted it
— ``len(content_json) // 4`` — so the compactor, the #356 context pre-flight
and the fallback window check budgeted tokens that are never in the prompt.

Signed thinking and redacted thinking ARE sent back to Anthropic, so they stay
counted. For a local model, which drops them too, that over-counts — the safe
direction for a compactor.
"""

from __future__ import annotations

import pytest

from prometheus.context.compactor import ContextCompactor
from prometheus.context.token_estimation import (
    estimate_message_tokens,
    estimate_messages,
    estimate_tokens,
)
from prometheus.engine.messages import (
    ConversationMessage,
    ImageBlock,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from prometheus.providers.base import ApiTextDeltaEvent

THOUGHT = "Let me work through this step by step.\n" * 100      # ~4k chars


def assistant(*blocks) -> ConversationMessage:
    return ConversationMessage(role="assistant", content=list(blocks))


@pytest.mark.parametrize("reply", ["Done.", "Voilà — “naïve” café, 30°C. 日本語の答え。" * 20],
                         ids=["ascii", "non-ascii"])
@pytest.mark.parametrize("signature", [None, ""], ids=["no-signature", "empty-signature"])
def test_an_unsigned_thought_adds_nothing(signature, reply):
    # Exactly nothing: what remains is measured by content_json itself, so a
    # non-ASCII reply costs what it costs without the thought.
    with_thought = assistant(ThinkingBlock(thinking=THOUGHT, signature=signature),
                             TextBlock(text=reply))
    assert estimate_message_tokens(with_thought) == estimate_message_tokens(
        assistant(TextBlock(text=reply)))


def test_a_thinking_only_turn_costs_what_an_empty_turn_costs():
    assert estimate_message_tokens(assistant(ThinkingBlock(thinking=THOUGHT))) == (
        estimate_message_tokens(assistant()))


@pytest.mark.parametrize("block", [
    ThinkingBlock(thinking=THOUGHT, signature="sig-abc"),
    RedactedThinkingBlock(data="opaque" * 300),
], ids=["signed", "redacted"])
def test_thinking_anthropic_sends_back_still_counts(block):
    msg = assistant(block, TextBlock(text="Done."))
    assert estimate_message_tokens(msg) == estimate_tokens(msg.content_json)
    assert estimate_message_tokens(msg) > estimate_message_tokens(assistant(TextBlock(text="Done.")))


@pytest.mark.parametrize("msg", [
    ConversationMessage.from_user_text("What is in a.txt? It's “quoted”, naïve, 30°."),
    assistant(TextBlock(text="Reading it."),
              ToolUseBlock(id="toolu_1", name="read_file", input={"path": "a.txt"})),
    ConversationMessage(role="user", content=[
        ToolResultBlock(tool_use_id="toolu_1", content="line one\nline two\n" * 40)]),
    # As the gateway builds an upload: the bytes live on disk (source_path),
    # and content_json stores the reference, not the ~400k-char payload.
    ConversationMessage(role="user", content=[
        TextBlock(text="look"),
        ImageBlock(media_type="image/png", data="iVBORw0KGgo=" * 33_000,
                   source_path="/cache/uploads/x.png")]),
], ids=["user-text", "tool-use", "tool-result", "image"])
def test_a_message_without_unsent_thinking_is_estimated_exactly_as_before(msg):
    # The compaction golden records exact token figures: nothing but the
    # thinking may move them.
    assert estimate_message_tokens(msg) == estimate_tokens(msg.content_json)


def test_an_upload_beside_an_unsent_thought_still_costs_its_reference_not_its_bytes():
    image = ImageBlock(media_type="image/png", data="iVBORw0KGgo=" * 33_000,
                       source_path="/cache/uploads/x.png")
    with_thought = ConversationMessage(role="user", content=[
        ThinkingBlock(thinking=THOUGHT), TextBlock(text="look"), image])
    without = ConversationMessage(role="user", content=[TextBlock(text="look"), image])
    assert estimate_message_tokens(with_thought) == estimate_message_tokens(without) < 200


def test_the_message_itself_keeps_its_thought():
    # Only the ESTIMATE leaves it out: the stored message, the client frame
    # and the compactor's span key all still carry it.
    msg = assistant(ThinkingBlock(thinking=THOUGHT), TextBlock(text="Done."))
    estimate_message_tokens(msg)
    assert isinstance(msg.content[0], ThinkingBlock)
    assert "Let me work through this step" in msg.content_json


def test_an_ollama_thinking_turn_is_estimated_by_what_goes_back_on_the_wire():
    # The shape #592 produces: reasoning normalized through the shared parser.
    from prometheus.providers.base import ApiMessageRequest
    from prometheus.providers.stub import _build_openai_messages, _parse_assistant_message

    msg, _ = _parse_assistant_message(
        {"message": {"content": "391", "reasoning_content": THOUGHT}}, model="qwen3.5:9b")
    assert isinstance(msg.content[0], ThinkingBlock) and msg.content[0].signature is None
    sent = _build_openai_messages(ApiMessageRequest(model="qwen3.5:9b", messages=[msg]))
    assert "step by step" not in str(sent)                        # never sent back...
    assert estimate_message_tokens(msg) == estimate_message_tokens(   # ...and not counted
        assistant(TextBlock(text="391")))


def _history(thought: str = "") -> list[ConversationMessage]:
    msgs: list[ConversationMessage] = []
    for i in range(12):
        msgs.append(ConversationMessage.from_user_text(f"q{i}: " + "x" * 200))
        blocks = [ThinkingBlock(thinking=thought)] if thought else []
        msgs.append(assistant(*blocks, TextBlock(text=f"a{i}: " + "y" * 200)))
    return msgs


class _CountingSummarizer:
    def __init__(self) -> None:
        self.calls = 0

    async def stream_message(self, request):  # noqa: ANN001
        self.calls += 1
        yield ApiTextDeltaEvent(text="summary.")


@pytest.mark.asyncio
async def test_compaction_does_not_fire_on_thinking_the_model_never_sees():
    summarizer = _CountingSummarizer()
    compactor = ContextCompactor(provider=summarizer, model="qwen3.5:9b", effective_limit=4000,
                                 reserve_tokens=500, threshold_pct=0.9)
    plain, thinking = _history(), _history(THOUGHT)
    threshold = compactor._threshold_tokens()
    assert compactor.estimate_total("sys", plain) < threshold      # what is actually sent fits
    assert estimate_tokens("".join(m.content_json for m in thinking)) > threshold  # the old count did not

    out = await compactor.apply(thinking, session_id="s", system_prompt="sys")
    assert out is thinking and summarizer.calls == 0               # nothing compacted
    assert compactor.estimate_total("sys", thinking) == compactor.estimate_total("sys", plain)


def test_the_meter_and_the_compactor_still_agree_with_thinking_present():
    from prometheus.gateway.commands import cmd_context

    messages = _history(THOUGHT)
    compactor = ContextCompactor(provider=_CountingSummarizer(), model="m", effective_limit=32768)
    text = cmd_context("sys", "m", local_model="m", detected_limit=32768,
                       config={"model": {"model": "m", "provider": "llama_cpp"}}, messages=messages)
    import re
    in_use = int(re.search(r"In use:\s+([\d,]+) tokens", text).group(1).replace(",", ""))
    assert in_use == compactor.estimate_total("sys", messages)
    assert estimate_messages(messages) == estimate_messages(_history())
