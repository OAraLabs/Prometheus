"""Record the loop's outbound ApiMessageRequest for the F1 payload-equivalence proof.

SPRINT-loop-envelope Phase 1: the envelope wrap must be behavior-preserving —
"identical request payloads (assert via recorded fixture diff)". This script
captures what ``run_loop`` actually hands the provider, serialized canonically,
so the post-wrap test (tests/test_loop_envelope.py) can diff against it.

PROVENANCE RULE: the committed fixture must be recorded at a PRE-WRAP commit
(run_loop calling provider.stream_message directly). The fixture header records
the git SHA it was captured at. Re-record only from a pre-wrap checkout, or the
equivalence proof is circular.

The scenario here is duplicated (deliberately, self-contained) in
tests/test_loop_envelope.py — if the two drift, the equality test fails loudly,
which is the guard doing its job.

Usage:  uv run python scripts/record_loop_envelope_fixture.py
Writes: tests/fixtures/loop_envelope_prewrap_request.json
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import AsyncIterator

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from prometheus.engine.agent_loop import LoopContext, run_loop  # noqa: E402
from prometheus.engine.messages import ConversationMessage, TextBlock  # noqa: E402
from prometheus.engine.usage import UsageSnapshot  # noqa: E402
from prometheus.providers.base import (  # noqa: E402
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiStreamEvent,
    ApiTextDeltaEvent,
    ModelProvider,
)


class CapturingProvider(ModelProvider):
    """Captures the request, then plays one scripted text-only turn."""

    def __init__(self) -> None:
        self.requests: list[ApiMessageRequest] = []

    async def stream_message(
        self, request: ApiMessageRequest
    ) -> AsyncIterator[ApiStreamEvent]:
        self.requests.append(request)
        yield ApiTextDeltaEvent(text="OK")
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(
                role="assistant", content=[TextBlock(text="OK")]
            ),
            usage=UsageSnapshot(input_tokens=123, output_tokens=7),
            stop_reason="stop",
        )


def build_scenario() -> tuple[LoopContext, list[ConversationMessage], CapturingProvider]:
    """The frozen fixture scenario. Mirrored in tests/test_loop_envelope.py."""
    provider = CapturingProvider()
    context = LoopContext(
        provider=provider,
        model="fixture-model",
        system_prompt="You are the loop-envelope fixture agent.",
        max_tokens=512,
        session_id="sess-fixture",
    )
    messages = [
        ConversationMessage.from_user_text("fixture turn — reply without tools")
    ]
    return context, messages, provider


def canonical_request_json(request: ApiMessageRequest) -> str:
    """Stable serialization of everything the provider receives."""
    return json.dumps(
        {
            "model": request.model,
            "system_prompt": request.system_prompt,
            "max_tokens": request.max_tokens,
            "tools": request.tools,
            "suppress_thinking": request.suppress_thinking,
            "messages": [m.model_dump(mode="json") for m in request.messages],
        },
        sort_keys=True,
        indent=1,
    )


async def main() -> None:
    context, messages, provider = build_scenario()
    async for _event, _usage in run_loop(context, messages):
        pass
    assert len(provider.requests) == 1, f"expected 1 request, saw {len(provider.requests)}"

    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()
    payload = canonical_request_json(provider.requests[0])
    out = REPO / "tests" / "fixtures" / "loop_envelope_prewrap_request.json"
    out.write_text(
        json.dumps(
            {"recorded_at_sha": sha, "request_canonical": payload}, indent=1
        )
        + "\n"
    )
    print(f"recorded @ {sha} → {out}")


if __name__ == "__main__":
    asyncio.run(main())
