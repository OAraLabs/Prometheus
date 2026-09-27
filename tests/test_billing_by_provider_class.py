"""A round served by a local provider bills ``local``, whatever its model is called
(WP-X.21 T9, docs/audits/TELEMETRY-GAPS.md).

``billing_mode`` is stamped at write time from the model's NAME: a path is local, a
flat-plan host marker is subscription, a priced name is metered. A blank config name
(the evals) and an Ollama ``name:tag`` are none of those, so a round the 4090 or the
mini's Ollama served came out ``unknown`` — the one real gap ``/api/usage`` reports.
The provider that served is known at write time, and a llama.cpp, Ollama, LM Studio
or vLLM provider serves from a box you own.

Pinned here through a real ``run_loop``, against the rows actually written; the name
still decides for every other provider.
"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
from prometheus.telemetry.tracker import ToolCallTelemetry
from prometheus.tools.base import ToolRegistry


class _Provider(ModelProvider):
    def __init__(self, provider_name: str) -> None:
        self.provider_name = provider_name

    async def stream_message(self, request):  # noqa: ANN001
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[TextBlock(text="hello")]),
            usage=UsageSnapshot(input_tokens=10, output_tokens=2), stop_reason="stop")


def _billing(tmp_path, provider_name: str, model: str) -> str | None:
    db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db)
    ctx = LoopContext(provider=_Provider(provider_name), model=model, system_prompt="",
                      max_tokens=64, tool_registry=ToolRegistry(), telemetry=tel)

    async def drain() -> None:
        async for _ in run_loop(ctx, [ConversationMessage.from_user_text("hi")], session_id="desktop:b1"):
            pass

    asyncio.run(drain())
    tel.close()
    con = sqlite3.connect(db)
    [(mode,)] = con.execute("SELECT billing_mode FROM subsystem_runs WHERE subsystem='agent_loop'"
                            " AND operation='loop_round'").fetchall()
    con.close()
    return mode


@pytest.mark.parametrize("provider_name,model", [
    ("llama_cpp", ""),                      # the evals: a blank config name
    ("ollama", "qwen2.5:7b-instruct"),      # an Ollama name:tag
    ("lm_studio", "some-local-model"),
    ("vllm", "served-model"),
])
def test_a_local_providers_round_bills_local(tmp_path, provider_name, model):
    assert _billing(tmp_path, provider_name, model) == "local", (
        "the provider that served is a box you own; its model's name was billed unknown")


@pytest.mark.parametrize("provider_name,model,mode", [
    ("xai", "grok-4.5", "metered"),                 # a priced name is still metered
    ("openai", "a-model-with-no-price-row-zzz", "unknown"),  # a cloud with no price stays unknown
])
def test_every_other_provider_is_still_decided_by_the_name(tmp_path, provider_name, model, mode):
    assert _billing(tmp_path, provider_name, model) == mode


@pytest.fixture(autouse=True)
def _no_global_handle(monkeypatch):
    import prometheus.telemetry.tracker as tracker

    monkeypatch.setattr(tracker, "_telemetry_singleton", None)
