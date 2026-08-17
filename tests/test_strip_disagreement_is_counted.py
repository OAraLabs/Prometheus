"""A tool-call envelope we delete is a DROP, not an empty model response.

THE DEFECT
----------
Two components disagree about what a ``<tool_call>`` envelope is. The
enforcer's EXTRACTOR could not parse it; the markup STRIPPER could delete it.
The stripper runs second, so the stripper won — and cleaning is destructive,
so the evidence was gone before anything could count it.

The whole of the instrumentation at that step used to be a comment::

    # empty after strip → drop the block

No counter, no log, no telemetry. The loop's empty guard then fired on *no
text AND no tool_uses AND not dropped_malformed* and told the operator **the
model** returned an empty response. On 2026-08-17 that ran four consecutive
turns across seven hours — "please rephrase and try again" — while the model
was emitting tool calls the entire time. Thirteen events were filed as
``empty_response`` and read as a broken rig.

Every other instrument was correct in isolation, which is why none of them
caught it: the provider had non-empty text when IT looked, nothing was
dropped at parse time, and ``silent_failures`` only covers the provider path.
The one step that destroyed the content is the only one with no instrument,
and it sits BETWEEN the two that have them.

THE FOUR CASES
--------------
Taken from the reproduction, not invented: unknown tool, invalid JSON args,
wrong key names, and truncated mid-envelope. All four give extracted=0 and
post-strip length 0. The truncated case is the one that matches the live
07:28 turn, which generated 60 tokens — about one envelope.

THE MUTATION
------------
Restore the silent drop (delete the counter and let the block vanish) and
every test here must go red. If they survive it, this file is decoration.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiTextDeltaEvent,
    ModelProvider,
)

# The four reproduced cases. Each is text the EXTRACTOR cannot parse and the
# STRIPPER can delete entirely.
CASES = {
    "unknown tool":
        '<tool_call>\n{"name": "not_a_registered_tool", "arguments": {"x": 1}}\n</tool_call>',
    "invalid JSON args":
        '<tool_call>\n{"name": "bash", "arguments": {oops}}\n</tool_call>',
    "wrong key names":
        '<tool_call>\n{"tool": "bash", "args": {"command": "ls"}}\n</tool_call>',
    "truncated mid-envelope":
        '<tool_call>\n{"name": "bash", "arguments": {"command": "ls -la /ho',
}

RECOVERED = "Here is the answer in plain text."


class _EnvelopeThenRecoverProvider(ModelProvider):
    """Emits an unparseable envelope, then recovers on the retry.

    The recovery is what proves the retry actually happened AND that the turn
    was not abandoned — a failure turn would never reach the second response.
    """

    def __init__(self, envelope: str) -> None:
        self.envelope = envelope
        self.calls = 0
        self.seen_feedback: list[str] = []

    async def stream_message(self, request):  # noqa: ANN001
        self.calls += 1
        # Capture what the loop fed back to us on the retry.
        for m in request.messages:
            text = getattr(m, "text", "") or ""
            if "could not be parsed" in text:
                self.seen_feedback.append(text)
        if self.calls == 1:
            yield ApiTextDeltaEvent(text=self.envelope)
            msg = ConversationMessage(
                role="assistant", content=[TextBlock(text=self.envelope)]
            )
        else:
            yield ApiTextDeltaEvent(text=RECOVERED)
            msg = ConversationMessage(
                role="assistant", content=[TextBlock(text=RECOVERED)]
            )
        yield ApiMessageCompleteEvent(
            message=msg,
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            stop_reason="stop",
        )


def _ctx(provider: ModelProvider) -> LoopContext:
    return LoopContext(
        provider=provider,
        model="test",
        system_prompt="- Model: test (provider: test)",
        max_tokens=256,
        # tier full + an extractor that MISSES: the disagreement itself.
        adapter=SimpleNamespace(
            tier="full",
            extract_tool_calls=lambda text, reg=None: [],
        ),
    )


def _final_text(ctx: LoopContext) -> str:
    messages = [ConversationMessage.from_user_text("do the thing")]

    async def _run() -> str:
        last = ""
        async for event, _usage in run_loop(ctx, messages):
            if type(event).__name__ == "AssistantTurnComplete":
                last = event.message.text or ""
        return last

    return asyncio.run(_run())


@pytest.mark.parametrize("label", list(CASES))
def test_stripped_envelope_is_counted_and_logged_not_reported_empty(label, caplog):
    """The core assertion, once per reproduced case."""
    provider = _EnvelopeThenRecoverProvider(CASES[label])
    with caplog.at_level(logging.WARNING):
        final = _final_text(_ctx(provider))

    warnings = [r.getMessage() for r in caplog.records
                if r.levelno >= logging.WARNING]
    disagreement = [w for w in warnings if "PARSE DISAGREEMENT" in w]

    assert disagreement, (
        f"[{label}] the strip was silent. A block that strips to nothing is a "
        f"DROP and must be counted and logged — this is the line that ate "
        f"four of Will's turns.\nwarnings seen: {warnings}"
    )
    # The pre-strip bytes are recorded. Not recovering them was part of the
    # defect, so the fix has to close that too.
    assert "tool_call" in disagreement[0], (
        f"[{label}] the log records the drop but not what was dropped; the "
        f"bytes are the evidence and they are still gone"
    )
    assert "empty response" not in final.lower(), (
        f"[{label}] a deleted tool call was reported to the operator as an "
        f"empty model response. The model produced output; we removed it."
    )
    assert "rephrase" not in final.lower(), (
        f"[{label}] the operator was told to rephrase for a disagreement "
        f"between two of our own components"
    )


@pytest.mark.parametrize("label", list(CASES))
def test_the_turn_is_retried_with_the_models_own_text_fed_back(label):
    """RULING: retry with the text preserved, do not surface a failure.

    The model DID produce output, so a failure turn discards work and blames
    the operator. The retry is INFORMED — it quotes back what the model
    emitted, which is the difference between "try again" and "this exact
    string did not parse" — and it is breaker-bounded exactly like the
    dropped_malformed path it mirrors.
    """
    provider = _EnvelopeThenRecoverProvider(CASES[label])
    final = _final_text(_ctx(provider))

    assert provider.calls >= 2, (
        f"[{label}] the loop did not retry — it gave up on a turn where the "
        f"model had produced output"
    )
    assert provider.seen_feedback, (
        f"[{label}] the retry carried no feedback; a blind retry repeats the "
        f"same failure and risks the stuck-turn class"
    )
    fed = provider.seen_feedback[0]
    head = CASES[label][:30].replace("\n", " ")
    assert head.split()[0] in fed, (
        f"[{label}] the feedback did not quote the model's own text back. "
        f"That text is the one thing it needs to self-correct.\nfed: {fed!r}"
    )
    assert final == RECOVERED, (
        f"[{label}] the recovered answer did not reach the operator: {final!r}"
    )


def test_a_genuinely_empty_turn_is_still_reported_as_empty(caplog):
    """The guard must not swallow the real empty case it was written for."""

    class _TrulyEmpty(ModelProvider):
        async def stream_message(self, request):  # noqa: ANN001
            yield ApiTextDeltaEvent(text="")
            yield ApiMessageCompleteEvent(
                message=ConversationMessage(role="assistant", content=[]),
                usage=UsageSnapshot(input_tokens=1, output_tokens=0),
                stop_reason="stop",
            )

    final = _final_text(_ctx(_TrulyEmpty()))
    assert "empty response" in final.lower(), (
        "a genuinely empty turn stopped being reported as one — the fix "
        "widened past its case"
    )
