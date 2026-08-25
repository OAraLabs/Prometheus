"""The routing decision is auditable AFTER the fact, at the default log level.

WHAT PROMPTED THIS. A user switched a Beacon session to Qwen, the model chip and
``GET /api/sessions/{id}/model`` both reported the override as set, and the question
"did the turn actually run on it?" could not be answered from the logs. The line that
would have said so was ``log.debug``; the daemon runs at INFO. Three days of journal on
a live daemon held ZERO of them, so routing had to be inferred sideways from a
provider-build side effect that only fires on the FIRST turn after a switch — the built
provider is cached on the override, so every later turn is silent.

An override whose effect cannot be observed after it fires is indistinguishable from one
that was dropped. These tests pin the observable part:

  * the decision is logged at INFO — it survives the level the daemon actually runs at
  * it names the session that was ROUTED ON, not the LoopContext's (possibly stale) one
  * it does NOT carry the user's message text — promoting the old excerpt verbatim would
    have written conversation content into the journal at the default level
  * one line per turn, not one per tool iteration
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

_SECRET = "my bank password is hunter2 and the deal closes friday"
_LOGGER = "prometheus.engine.agent_loop"


class _Provider(ModelProvider):
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


class _Decision:
    def __init__(self) -> None:
        self.provider = None
        self.adapter = None
        self.reason = "user_override"
        self.model_name = "qwen3.8-max"
        self.provider_name = "qwen"


class _RecordingRouter:
    """Records the session_id it was routed with, so the log line can be compared
    against the id the override was ACTUALLY looked up under."""

    def __init__(self) -> None:
        self.routed_with: list[str | None] = []

    def route(self, message, context=None):
        self.routed_with.append((context or {}).get("session_id"))
        return _Decision()

    def get_override_for_session(self, session_id):
        return None


def _run(router: _RecordingRouter, *, turn_session: str | None, ctx_session: str):
    context = LoopContext(
        provider=_Provider(),
        model="primary-model",
        system_prompt="s",
        max_tokens=512,
        session_id=ctx_session,
        model_router=router,
    )

    async def go() -> None:
        async for _ in run_loop(
            context,
            [ConversationMessage.from_user_text(_SECRET)],
            session_id=turn_session,
        ):
            pass

    asyncio.run(go())
    return context


def _route_lines(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if "ModelRouter:" in r.getMessage()]


def test_routing_decision_is_logged_at_info(caplog):
    """At DEBUG this record does not exist in a running daemon."""
    router = _RecordingRouter()
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        _run(router, turn_session="beacon:live", ctx_session="stale-context")

    lines = _route_lines(caplog)
    assert lines, "the routing decision was not logged at INFO"
    assert lines[0].levelno == logging.INFO, (
        f"logged at {lines[0].levelname}, which the daemon's default level drops"
    )
    msg = lines[0].getMessage()
    assert "qwen" in msg and "qwen3.8-max" in msg, f"provider/model missing: {msg}"
    assert "user_override" in msg, f"the REASON is the point of the record: {msg}"


def test_it_names_the_session_that_was_routed_on(caplog):
    """The turn's session, not the LoopContext's — those differ for every REST/WS
    turn, and naming the wrong one makes the record worse than none."""
    router = _RecordingRouter()
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        _run(router, turn_session="beacon:live", ctx_session="stale-context")

    assert router.routed_with == ["beacon:live"]
    msg = _route_lines(caplog)[0].getMessage()
    assert "session=beacon:live" in msg, f"wrong or missing session: {msg}"
    assert "stale-context" not in msg


def test_the_users_message_is_not_in_the_record(caplog):
    """The old DEBUG line carried a 60-char excerpt of the prompt. At INFO that is
    conversation content in the daemon journal."""
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        _run(_RecordingRouter(), turn_session="s", ctx_session="s")

    msg = _route_lines(caplog)[0].getMessage()
    assert "hunter2" not in msg, f"message content leaked into the log: {msg}"
    assert _SECRET[:20] not in msg


def test_one_line_per_turn(caplog):
    """The block sits before the tool-iteration loop; if it ever moves inside,
    a long turn would emit one of these per iteration."""
    router = _RecordingRouter()
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        context = _run(router, turn_session="s", ctx_session="s")

    assert len(context.provider.requests) >= 1
    assert len(_route_lines(caplog)) == 1, "expected exactly one routing record per turn"
    assert len(router.routed_with) == 1
