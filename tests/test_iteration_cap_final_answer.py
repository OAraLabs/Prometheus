"""The tool-iteration cap must end the turn with an ANSWER, not a bare notice.

Before this, hitting ``max_tool_iterations`` yielded only

    Tool iteration limit reached (26/25). Stopping to prevent runaway loops.

and returned — discarding every tool result the turn had already gathered.
The turn that prompted this work was 25 serial ``bash`` calls in 64 seconds
with no circuit-breaker trip, no repeat-guard blocks and zero tool errors:
correct, non-looping investigation, guillotined one call from the end. Raising
the cap only moves that wall; it does not stop the loss.

These tests pin the new contract:

* the turn ends with the model's own account of what it found;
* the extra call is STRUCTURALLY tool-free, so the runaway guard cannot
  itself be a way to run away;
* the directive rides the per-call system prompt, never conversation history;
* every failure mode falls back to the old notice, so the change can only
  improve on the previous outcome, never worsen it.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from prometheus.engine.agent_loop import AgentLoop
from prometheus.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiTextDeltaEvent,
    ModelProvider,
)
from prometheus.tools.base import BaseTool, ToolRegistry, ToolResult

CAP = 2
ANSWER = "I checked the llama-server props: vision is off because no mmproj is loaded."


class _In(BaseModel):
    pass


class _Noop(BaseTool):
    name = "noop"
    description = "does nothing, successfully"
    input_model = _In

    async def execute(self, arguments, context):
        return ToolResult(output="ok")


class _Provider(ModelProvider):
    """Asks for a tool forever — until the round is tool-free, then answers.

    Modelling the *productive* runaway on purpose: every tool call succeeds,
    so nothing but the iteration cap can end this turn.
    """

    def __init__(self, *, final: str | None = ANSWER, raise_on_final: bool = False):
        self.requests: list[ApiMessageRequest] = []
        self._final = final
        self._raise_on_final = raise_on_final
        self._n = 0

    @property
    def final_calls(self) -> list[ApiMessageRequest]:
        return [r for r in self.requests if r.suppress_tools]

    async def stream_message(self, request: ApiMessageRequest):
        self.requests.append(request)

        if request.suppress_tools:
            if self._raise_on_final:
                raise RuntimeError("provider exploded on the final-answer pass")
            if not self._final:
                yield ApiMessageCompleteEvent(
                    message=ConversationMessage(role="assistant", content=[]),
                    usage=UsageSnapshot(),
                    stop_reason="stop",
                )
                return
            yield ApiTextDeltaEvent(text=self._final)
            yield ApiMessageCompleteEvent(
                message=ConversationMessage(
                    role="assistant", content=[TextBlock(text=self._final)]
                ),
                usage=UsageSnapshot(input_tokens=9, output_tokens=4),
                stop_reason="stop",
            )
            return

        self._n += 1
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(
                role="assistant",
                content=[ToolUseBlock(id=f"t{self._n}", name="noop", input={})],
            ),
            usage=UsageSnapshot(input_tokens=10, output_tokens=10),
            stop_reason="tool_calls",
        )


def _loop(provider: _Provider) -> AgentLoop:
    reg = ToolRegistry()
    reg.register(_Noop())
    return AgentLoop(
        provider=provider,
        tool_registry=reg,
        max_turns=12,
        max_tool_iterations=CAP,
    )


async def _run(provider: _Provider):
    return await _loop(provider).run_async("You are helpful.", "why is vision off?")


# ---------------------------------------------------------------------------
# The behaviour the user actually feels
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_exhausted_budget_returns_the_findings_not_a_bare_notice():
    provider = _Provider()
    result = await _run(provider)

    assert ANSWER in result.text, (
        "the turn must end with the model's account of what it found; "
        f"got {result.text!r}"
    )
    assert "Tool iteration limit reached" not in result.text, (
        "the bare stop notice must not be what the user is left with"
    )


@pytest.mark.asyncio
async def test_the_cap_still_stops_the_loop():
    """The guard must still FIRE — an answer is not a licence to keep going."""
    provider = _Provider()
    await _run(provider)

    tool_rounds = [r for r in provider.requests if not r.suppress_tools]
    assert len(tool_rounds) == CAP + 1, (
        "the loop must stop at the cap, not run to max_turns; "
        f"took {len(tool_rounds)} tool-bearing rounds"
    )
    assert len(provider.final_calls) == 1, (
        "exactly ONE final answer pass — it must not become a second loop"
    )


# ---------------------------------------------------------------------------
# The extra call cannot itself run away
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_final_call_is_structurally_tool_free():
    provider = _Provider()
    await _run(provider)

    final = provider.final_calls[0]
    assert final.suppress_tools is True
    assert final.tools == [], "no tool schema may be offered on the final pass"
    assert final.tool_choice == "none", (
        "tool_choice must be 'none' so local tiers drop the tool-calling "
        "grammar too — tool-free in structure, not merely in practice"
    )


@pytest.mark.asyncio
async def test_directive_rides_the_system_prompt_never_the_history():
    """Machinery text in `messages` reaches LCM and is mined as a user fact."""
    provider = _Provider()
    result = await _run(provider)

    final = provider.final_calls[0]
    assert "tool budget for this turn is now spent" in (final.system_prompt or "")
    assert "You are helpful." in (final.system_prompt or ""), (
        "the caller's system prompt must be preserved, not replaced"
    )

    for m in result.messages:
        assert "tool budget for this turn is now spent" not in (m.text or ""), (
            "the directive must never enter conversation history"
        )


@pytest.mark.asyncio
async def test_unexecuted_tool_calls_are_closed_with_results():
    """A dangling tool_use with no tool_result is a 400 on strict providers."""
    provider = _Provider()
    result = await _run(provider)

    requested: list[str] = []
    answered: list[str] = []
    for m in result.messages:
        requested.extend(b.id for b in m.content if isinstance(b, ToolUseBlock))
        answered.extend(
            b.tool_use_id for b in m.content if isinstance(b, ToolResultBlock)
        )

    assert requested, "sanity: the run should have requested tools"
    assert set(requested) == set(answered), (
        "every tool_use must have a matching tool_result — including the batch "
        f"the budget refused to run. dangling: {set(requested) - set(answered)}"
    )

    not_executed = [
        b
        for m in result.messages
        for b in m.content
        if isinstance(b, ToolResultBlock) and "NOT EXECUTED" in b.content
    ]
    assert not_executed, "the refused calls must be marked as never having run"
    assert all(b.is_error for b in not_executed)


# ---------------------------------------------------------------------------
# Never worse than before
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_falls_back_to_the_bare_notice_when_the_final_call_raises():
    provider = _Provider(raise_on_final=True)
    result = await _run(provider)

    assert "Tool iteration limit reached" in result.text, (
        "a failed final pass must degrade to the old notice, not to silence"
    )


@pytest.mark.asyncio
async def test_falls_back_to_the_bare_notice_when_the_final_answer_is_empty():
    provider = _Provider(final=None)
    result = await _run(provider)

    assert "Tool iteration limit reached" in result.text
    assert not any(
        m.role == "assistant" and not (m.text or "").strip() and not m.tool_uses
        for m in result.messages
    ), "an empty assistant turn must never be committed (the #65 invariant)"
