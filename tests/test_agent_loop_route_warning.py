"""run_loop() logs LOUDLY when ModelRouter.route() raises.

The handler in ``agent_loop`` says why it exists, in its own words:

    "Any exception in route() means the user's override (or task rule, or
    escalation) was NOT applied and we silently fell through to primary —
    that's not something we should discover by reading source code."

It was elevated DEBUG → WARNING in Phase 4 after a silent DEBUG hid a real
production bug (stale-system-prompt identity). But nothing proved the warning
fires: no test anywhere drove ``route()`` to raise. A guard whose only evidence
is a comment is the exact shape of defect this handler was written to stop.

These tests close that path. They assert the CURRENT behaviour — warn and fall
back — not a redesign:

  * ``route()`` raises            → WARNING, with traceback, naming the fallback
  * an explicit override is set   → still warned; the override is silently
                                    dropped, which is the case that actually
                                    burns a user
  * the turn still completes      → the fallback is real, not a masked crash

Recovered 2026-08-02 from the abandoned ``fix/agent-loop-route-failure`` branch
(2026-06-17, never opened as a PR). That branch also proposed a behaviour
change — raise ``RouteOverrideError`` when an override is present rather than
falling back. That half is deliberately NOT ported: it changes semantics and
deserves its own decision. This file only proves what main already claims.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiStreamEvent,
    ModelProvider,
)

_ROUTE_FAILURE = "provider build failed (key rotated post-switch)"


class _CapturingProvider(ModelProvider):
    """In-memory primary: records requests, answers one text turn."""

    def __init__(self) -> None:
        self.requests: list[ApiMessageRequest] = []
        self._suppress_thinking = True

    async def stream_message(
        self, request: ApiMessageRequest
    ) -> AsyncIterator[ApiStreamEvent]:
        self.requests.append(request)
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(
                role="assistant", content=[TextBlock(text="ok")]
            ),
            usage=UsageSnapshot(input_tokens=10, output_tokens=2),
            stop_reason="stop",
        )


class _Override:
    """Stand-in for ModelRouter's ProviderOverride (only what the loop reads)."""

    def __init__(self, model: str) -> None:
        self.provider_config = {"model": model}


class _RaisingRouter:
    """A ModelRouter whose route() always fails — e.g. the provider build blew
    up because a key rotated after the model switch was accepted."""

    def __init__(self, *, override: _Override | None = None) -> None:
        self._override = override

    def route(self, message, context=None):
        raise RuntimeError(_ROUTE_FAILURE)

    def get_override_for_session(self, session_id):
        return self._override


def _context(router: _RaisingRouter) -> LoopContext:
    return LoopContext(
        provider=_CapturingProvider(),
        model="primary-model",
        system_prompt="s",
        max_tokens=512,
        session_id="web",
        model_router=router,
    )


def _drain(context: LoopContext) -> None:
    async def go() -> None:
        async for _ in run_loop(context, [ConversationMessage.from_user_text("hi")]):
            pass

    asyncio.run(go())


def test_route_failure_logs_a_warning_with_traceback(caplog):
    """The handler's whole purpose: make the fallback visible in the logs."""
    context = _context(_RaisingRouter())

    with caplog.at_level(logging.WARNING, logger="prometheus.engine.agent_loop"):
        _drain(context)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "route() raised and NOTHING was logged at WARNING"

    route_warnings = [r for r in warnings if "route()" in r.getMessage()]
    assert route_warnings, (
        "no WARNING mentions route(); got: "
        f"{[r.getMessage()[:80] for r in warnings]}"
    )

    record = route_warnings[0]
    # The operator has to be able to tell WHAT was lost, not just that something was.
    assert "falling back to primary" in record.getMessage()
    # exc_info=True is the difference between a breadcrumb and a diagnosis.
    assert record.exc_info is not None, "warning logged without the traceback"
    assert _ROUTE_FAILURE in "".join(
        __import__("traceback").format_exception(*record.exc_info)
    )


def test_route_failure_with_an_override_still_warns(caplog):
    """The case that actually burns a user: they explicitly switched models,
    the switch was silently dropped, and the turn ran on the primary anyway."""
    context = _context(_RaisingRouter(override=_Override("claude-haiku-4-5")))

    with caplog.at_level(logging.WARNING, logger="prometheus.engine.agent_loop"):
        _drain(context)

    assert any(
        "route()" in r.getMessage() and r.levelno == logging.WARNING
        for r in caplog.records
    ), "an explicitly-overridden model was dropped with no WARNING"

    # Documents the CURRENT contract: fall back, do not hard-error. If this
    # ever becomes a raise (see fix/agent-loop-route-failure), this assertion
    # is the one that should fail and force the conversation.
    assert context.provider.requests, "the turn did not run at all"
    assert context.provider.requests[0].model == "primary-model"


def test_route_failure_does_not_break_the_turn(caplog):
    """The fallback is real: a raising router must not take the turn down."""
    context = _context(_RaisingRouter())

    with caplog.at_level(logging.WARNING, logger="prometheus.engine.agent_loop"):
        _drain(context)

    assert len(context.provider.requests) == 1
    assert context.provider.requests[0].model == "primary-model"
