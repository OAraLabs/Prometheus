"""Which model served a round is recorded, and Ollama says which model served it
(WP-X.21 T13, T12b; docs/audits/TELEMETRY-GAPS.md).

* T13: ``loop_round`` rows keep ``request.model``, the name asked for. The envelope
  saw ``event.served_model`` and dropped it, so a round asked for under a blank
  config name (the evals, coding runs before #585) had no model at all. The served
  model now goes in the round's summary; the row's ``model`` stays the requested
  name, as ``tool_calls`` keeps ``model`` beside ``served_model``.
* T12b: Ollama's parser ignored the model each chunk names (llama.cpp read it; #614
  added the OpenAI-compatible and Anthropic parsers). It now reads it too.

Pinned here through a real ``run_loop``, against the rows actually written, over the
recorded Ollama streams in the committed parity traces.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import httpx
import pytest

from prometheus.engine.agent_loop import LoopContext, run_loop
from prometheus.engine.messages import ConversationMessage, TextBlock
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
from prometheus.providers.ollama import OllamaProvider
from prometheus.telemetry.tracker import ToolCallTelemetry
from prometheus.tools.base import ToolRegistry

TRACES = Path(__file__).resolve().parent / "fixtures" / "parity"


def _round(tmp_path, provider, model: str) -> sqlite3.Row:
    """One turn with no tools: one round. Returns its ``loop_round`` row."""
    db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db)
    ctx = LoopContext(provider=provider, model=model, system_prompt="You are a test.",
                      max_tokens=64, tool_registry=ToolRegistry(), telemetry=tel)

    async def drain() -> None:
        async for _ in run_loop(ctx, [ConversationMessage.from_user_text("hi")], session_id="desktop:s1"):
            pass

    asyncio.run(drain())
    tel.close()
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    [row] = con.execute("SELECT model, summary_json FROM subsystem_runs WHERE subsystem='agent_loop'"
                        " AND operation='loop_round'").fetchall()
    return row


class _Serves(ModelProvider):
    def __init__(self, served: str | None) -> None:
        self.served = served

    async def stream_message(self, request):  # noqa: ANN001
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[TextBlock(text="hello")]),
            usage=UsageSnapshot(input_tokens=10, output_tokens=2), stop_reason="stop",
            served_model=self.served)


def _summary(row) -> dict:
    return json.loads(row["summary_json"] or "{}")


def test_a_round_asked_for_under_a_blank_name_records_the_model_that_served_it(tmp_path):
    row = _round(tmp_path, _Serves("/models-root/models/Qwen3.8-27B-UD-Q4_K_XL.gguf"), "")
    assert row["model"] == "", "the requested name is kept as it was asked for"
    assert _summary(row).get("served_model") == "/models-root/models/Qwen3.8-27B-UD-Q4_K_XL.gguf", (
        "the envelope saw the served model and dropped it")


def test_a_provider_that_names_no_model_adds_nothing(tmp_path):
    row = _round(tmp_path, _Serves(None), "stub-model")
    assert "served_model" not in _summary(row)


def _recorded_ollama(trace: str) -> list[tuple[dict, str]]:
    data = json.loads((TRACES / f"{trace}.trace.json").read_text(encoding="utf-8"))
    return [(ex["request"], ex["body"]) for ex in data["exchanges"]
            if ex["method"] == "POST" and ex["upstream"] == "alt"
            and ex["path"] == "/v1/chat/completions" and "data: " in (ex["body"] or "")]


def _serve_ollama(monkeypatch, body: str) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/chat/completions"):
            return httpx.Response(200, content=body.encode(),
                                  headers={"content-type": "text/event-stream"})
        # /api/show (the thinking-capability question, #592): unknown here.
        return httpx.Response(404, json={"error": "not found"})

    real = httpx.AsyncClient

    def client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)


def test_an_ollama_round_records_the_model_its_stream_names(tmp_path, monkeypatch):
    streams = _recorded_ollama("model_switch")
    assert streams, "premise: the trace holds the alt model's streamed completion"
    request, body = streams[0]
    named = {json.loads(line[6:]).get("model") for line in body.splitlines()
             if line.startswith("data: ") and line[6:].strip() not in ("", "[DONE]")} - {None}
    assert len(named) == 1, "premise: the recorded stream names one model"
    _serve_ollama(monkeypatch, body)
    row = _round(tmp_path, OllamaProvider(base_url="http://unit.test:11434"), request["model"])
    assert _summary(row).get("served_model") == named.pop()


@pytest.fixture(autouse=True)
def _no_global_handle(monkeypatch):
    import prometheus.telemetry.tracker as tracker

    monkeypatch.setattr(tracker, "_telemetry_singleton", None)
