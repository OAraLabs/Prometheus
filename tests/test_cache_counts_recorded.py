"""Prompt-cache counts reach the round's row, and Anthropic's input is the whole prompt
(WP-X.21 T6, T7, T8; docs/audits/TELEMETRY-GAPS.md).

``subsystem_runs`` has carried ``cached_input_tokens`` and ``cache_write_tokens`` since
#119, and no row has ever held one:

* T6: the envelope read both counts off every completion and passed them on its
  failure row only; the success row dropped them;
* T7: llama.cpp's parser never read the count its server sends on every completion
  (``prompt_tokens_details.cached_tokens``: 81% of the recorded prompt tokens);
* T8: Anthropic's ``input_tokens`` leaves out both cache counters, but the usage
  contract is the whole prompt. The recorded hosted turn reported 348 of 15,083.

Pinned here, through a real ``run_loop`` over the recorded streams in the committed
parity traces, against the rows actually written. A provider that reports nothing
about its cache still leaves NULL, never 0.
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
from prometheus.providers.anthropic import AnthropicProvider
from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
from prometheus.providers.llama_cpp import LlamaCppProvider
from prometheus.telemetry.tracker import ToolCallTelemetry
from prometheus.tools.base import ToolRegistry

TRACES = Path(__file__).resolve().parent / "fixtures" / "parity"


def _recorded(trace: str, path: str) -> list[str]:
    """Every streamed completion body in a committed trace."""
    data = json.loads((TRACES / f"{trace}.trace.json").read_text(encoding="utf-8"))
    return [ex["body"] for ex in data["exchanges"]
            if ex["method"] == "POST" and ex["path"] == path and "data: " in (ex["body"] or "")]


def _events(body: str) -> list[dict]:
    return [json.loads(line[6:]) for line in body.splitlines()
            if line.startswith("data: ") and line[6:].strip() not in ("", "[DONE]")]


def _serve(monkeypatch, body: str) -> None:
    """Every POST is answered with *body*; anything else is a 404."""
    def handler(req: httpx.Request) -> httpx.Response:
        if req.method != "POST":
            return httpx.Response(404, json={"error": "not found"})
        return httpx.Response(200, content=body.encode(),
                              headers={"content-type": "text/event-stream"})

    real = httpx.AsyncClient

    def client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)


def _round(tmp_path, provider, model: str) -> sqlite3.Row:
    """One turn with no tools: one round. Returns its ``loop_round`` row."""
    db = tmp_path / "telemetry.db"
    tel = ToolCallTelemetry(db)
    ctx = LoopContext(provider=provider, model=model, system_prompt="You are a test.",
                      max_tokens=64, tool_registry=ToolRegistry(), telemetry=tel)

    async def drain() -> None:
        async for _ in run_loop(ctx, [ConversationMessage.from_user_text("hi")], session_id="desktop:c1"):
            pass

    asyncio.run(drain())
    tel.close()
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    [row] = con.execute("SELECT input_tokens, output_tokens, cached_input_tokens, cache_write_tokens"
                        " FROM subsystem_runs WHERE subsystem='agent_loop' AND operation='loop_round'"
                        ).fetchall()
    return row


# ── T6: the envelope's success row ──────────────────────────────────────────

class _Reports(ModelProvider):
    def __init__(self, usage: UsageSnapshot) -> None:
        self.usage = usage

    async def stream_message(self, request):  # noqa: ANN001
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[TextBlock(text="hello")]),
            usage=self.usage, stop_reason="stop")


def test_a_successful_round_records_the_cache_counts_its_provider_reported(tmp_path):
    row = _round(tmp_path, _Reports(UsageSnapshot(input_tokens=6187, output_tokens=11,
                                                  cached_input_tokens=4172, cache_write_tokens=0)),
                 "stub-model")
    assert (row["cached_input_tokens"], row["cache_write_tokens"]) == (4172, 0), (
        "the envelope had both counts and dropped them on the success row")


def test_a_provider_that_reports_no_cache_leaves_null_not_zero(tmp_path):
    """None is "the provider said nothing"; 0 is "the cache was cold"."""
    row = _round(tmp_path, _Reports(UsageSnapshot(input_tokens=95, output_tokens=6)), "stub-model")
    assert (row["cached_input_tokens"], row["cache_write_tokens"]) == (None, None)


# ── T7: llama.cpp, over its recorded streams ────────────────────────────────

def test_a_llama_cpp_round_records_the_cache_count_its_server_sent(tmp_path, monkeypatch):
    bodies = _recorded("plain_chat", "/v1/chat/completions")
    assert bodies, "premise: the trace holds llama.cpp's streamed completions"
    body = bodies[-1]
    [usage] = [e["usage"] for e in _events(body) if e.get("usage")]
    sent = usage["prompt_tokens_details"]["cached_tokens"]
    _serve(monkeypatch, body)
    row = _round(tmp_path, LlamaCppProvider(base_url="http://unit.test:8080"), "Qwen3.8-27B")
    assert row["input_tokens"] == usage["prompt_tokens"]
    assert row["cached_input_tokens"] == sent, "llama.cpp sends it on every completion"


# ── T8: Anthropic's input is the whole prompt ───────────────────────────────

def test_an_anthropic_round_records_the_whole_prompt_and_its_cache_counts(tmp_path, monkeypatch):
    [body] = _recorded("hosted_route", "/v1/messages")
    [start] = [e for e in _events(body) if e.get("type") == "message_start"]
    usage = start["message"]["usage"]
    read, created = usage["cache_read_input_tokens"], usage["cache_creation_input_tokens"]
    _serve(monkeypatch, body)
    row = _round(tmp_path, AnthropicProvider(api_key="test-key", base_url="http://unit.test/v1"),
                 "claude-haiku-4-5")
    assert row["input_tokens"] == usage["input_tokens"] + read + created, (
        "input_tokens is the whole prompt; Anthropic's leaves out the cached part")
    assert (row["cached_input_tokens"], row["cache_write_tokens"]) == (read, created)


@pytest.fixture(autouse=True)
def _no_global_handle(monkeypatch):
    import prometheus.telemetry.tracker as tracker

    monkeypatch.setattr(tracker, "_telemetry_singleton", None)
