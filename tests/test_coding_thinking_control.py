"""Per-run thinking control (SPRINT-coding-mode v2, scope item 1).

A coding run sets ``LoopContext.suppress_thinking=False`` and ONLY its own
provider calls think — the global default is untouched. Asserted exactly the
way the spec asks: through the F1 envelope's ``subsystem_runs.thinking``
column (1 = thinking on), plus the payload the provider actually receives.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
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
from prometheus.telemetry.tracker import ToolCallTelemetry


class _CapturingProvider(ModelProvider):
    """Records requests; answers one text turn. Carries a provider default."""

    def __init__(self, suppress_default: bool = True) -> None:
        self.requests: list[ApiMessageRequest] = []
        self._suppress_thinking = suppress_default  # what LlamaCppProvider has

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


def _turn(tel: ToolCallTelemetry, suppress: bool | None) -> _CapturingProvider:
    provider = _CapturingProvider()
    context = LoopContext(
        provider=provider,
        model="m",
        system_prompt="s",
        max_tokens=512,
        telemetry=tel,
        session_id="coding:test",
        suppress_thinking=suppress,
    )
    messages = [ConversationMessage.from_user_text("q")]

    async def drain() -> None:
        async for _ in run_loop(context, messages):
            pass

    asyncio.run(drain())
    return provider


def _thinking_flags(tel: ToolCallTelemetry) -> list[int | None]:
    return [
        r[0]
        for r in tel._conn.execute(
            "SELECT thinking FROM subsystem_runs WHERE subsystem='agent_loop'"
            " ORDER BY timestamp"
        ).fetchall()
    ]


def test_coding_context_thinks_and_envelope_records_it(tmp_path: Path) -> None:
    tel = ToolCallTelemetry(db_path=tmp_path / "tel.db")
    provider = _turn(tel, suppress=False)

    # The provider was asked to think on THIS call…
    assert provider.requests[0].suppress_thinking is False
    # …and the F1 row records the effective flag: thinking = 1.
    assert _thinking_flags(tel) == [1]


def test_default_context_keeps_provider_default(tmp_path: Path) -> None:
    tel = ToolCallTelemetry(db_path=tmp_path / "tel.db")
    provider = _turn(tel, suppress=None)

    # No override sent — provider default governs (suppressed → thinking=0).
    assert provider.requests[0].suppress_thinking is None
    assert _thinking_flags(tel) == [0]


def test_explicit_suppress_true_recorded_as_zero(tmp_path: Path) -> None:
    tel = ToolCallTelemetry(db_path=tmp_path / "tel.db")
    provider = _turn(tel, suppress=True)

    assert provider.requests[0].suppress_thinking is True
    assert _thinking_flags(tel) == [0]
