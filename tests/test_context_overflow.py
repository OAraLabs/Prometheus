"""A prompt over the model's window is a named, terminal failure of the TURN.

WHAT HAPPENED (2026-10-09, #704)
--------------------------------
The local box answered an over-window request with an unambiguous body::

    {"error": {"code": 400, "type": "exceed_context_size_error",
               "message": "request (33094 tokens) exceeds the available context size (32768 tokens)",
               "n_prompt_tokens": 33094, "n_ctx": 32768}}

and ``classify_turn_error`` filed it with every other 400 as ``bad_request``.
The turn died with a generic "400 Bad Request", the provider fallback said
``DECLINED ... kind='bad_request'``, and a worker reported nothing anyone
could act on. The #356 pre-flight has the right sentence for this ("too long
for the model ... /reset ... retrying unchanged will fail the same way"),
but only reaches it when the daemon's own estimate catches the overflow
first — and a request the estimate let through got the 400 instead.

THE RULE THESE TESTS PIN
------------------------
* An over-window rejection is ``context_overflow`` — llama.cpp's
  ``exceed_context_size_error``, the OpenAI-compatible
  ``context_length_exceeded``, Anthropic's "prompt is too long" — with the
  server's own numbers when it gives them. Any other 400 stays bad_request.
* It ends the TURN, not the session: the loop answers with the #356-style
  explanation (``stop_reason="context_overflow"``) instead of raising, and
  sends nothing more — no retry of the same prompt.
* It is not a reason to DEGRADE to the fallback target: that target is
  usually the smaller local box.
* A worker reports it by name.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import httpx
import pytest

from tests.fixtures.fake_llama_server import WORKER_MARK, FakeLlamaServer


def _status_error(status: int, body: dict | str, host: str = "127.0.0.1:8080"):
    req = httpx.Request("POST", f"http://{host}/v1/chat/completions")
    text = body if isinstance(body, str) else json.dumps(body)
    resp = httpx.Response(status, text=text, request=req)
    return httpx.HTTPStatusError(f"Client error '{status}'", request=req, response=resp)


LLAMA_BODY = {"error": {
    "code": 400, "type": "exceed_context_size_error",
    "message": "request (33094 tokens) exceeds the available context size (32768 tokens), "
               "try increasing it",
    "n_prompt_tokens": 33094, "n_ctx": 32768}}


def test_llama_cpps_over_window_400_is_context_overflow_with_its_numbers():
    from prometheus.api.turn_errors import classify_turn_error

    out = classify_turn_error(_status_error(400, LLAMA_BODY))
    assert out["kind"] == "context_overflow", out
    assert out["tokens_needed"] == 33094 and out["window"] == 32768, out
    assert "/reset" in out["hint"]


def test_the_openai_compatible_and_anthropic_shapes_are_context_overflow_too():
    from prometheus.api.turn_errors import classify_turn_error

    openai = {"error": {"message": "This model's maximum context length is 128000 tokens. "
                                   "However, your messages resulted in 131072 tokens.",
                        "type": "invalid_request_error", "code": "context_length_exceeded"}}
    out = classify_turn_error(_status_error(400, openai, "api.openai.com"))
    assert out["kind"] == "context_overflow", out
    assert out["window"] == 128000 and out["tokens_needed"] == 131072, out

    anthropic = {"type": "error", "error": {
        "type": "invalid_request_error",
        "message": "prompt is too long: 210012 tokens > 200000 maximum"}}
    out = classify_turn_error(_status_error(400, anthropic, "api.anthropic.com"))
    assert out["kind"] == "context_overflow", out
    assert out["tokens_needed"] == 210012 and out["window"] == 200000, out


def test_any_other_400_is_still_bad_request():
    from prometheus.api.turn_errors import classify_turn_error

    body = {"error": {"type": "invalid_request_error", "message": "tools[3] is malformed"}}
    assert classify_turn_error(_status_error(400, body))["kind"] == "bad_request"


def test_an_overflow_is_not_a_reason_to_degrade_to_the_fallback_target():
    from prometheus.engine.fallback import is_terminal

    assert not is_terminal("context_overflow")


@pytest.mark.asyncio
async def test_the_loop_answers_an_overflow_instead_of_raising_and_sends_it_once(tmp_path):
    from prometheus.engine.agent_loop import LoopContext, run_loop
    from prometheus.engine.messages import ConversationMessage
    from prometheus.engine.stream_events import AssistantTurnComplete
    from prometheus.providers.registry import ProviderRegistry

    with FakeLlamaServer(n_ctx=50) as box:
        ctx = LoopContext(
            provider=ProviderRegistry.create(
                {"provider": "llama_cpp", "base_url": box.url, "model": "fake-local-27b"}),
            model="fake-local-27b", system_prompt="s" * 400, max_tokens=256,
        )
        msgs = [ConversationMessage.from_user_text("hello")]
        done = [ev async for ev, _u in run_loop(ctx, msgs, surface="cli")
                if isinstance(ev, AssistantTurnComplete)]
        sent = box.chat_rows()
    assert done, "the turn ended without an answer"
    final = done[-1]
    assert final.stop_reason == "context_overflow", final.stop_reason
    text = final.message.text or ""
    assert "too long" in text and "/reset" in text, text
    assert len(sent) == 1, f"the over-window prompt was sent {len(sent)} times"


@pytest.fixture
def child_env(tmp_path, monkeypatch):
    site_dir = tmp_path / "child-site"
    site_dir.mkdir()
    (site_dir / "sitecustomize.py").write_text(
        "import prometheus.config.defaults as _defaults\n"
        f"_defaults.REPO_CONFIG_PATH = _defaults.Path({str(tmp_path / 'no-repo' / 'prometheus.yaml')!r})\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(p for p in (str(site_dir), os.environ.get("PYTHONPATH", "")) if p),
    )
    for var in ("ANTHROPIC_API_KEY", "QWEN_API_KEY", "QWEN_BASE_URL", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    work = tmp_path / "work"
    work.mkdir()
    return work


@pytest.mark.asyncio
async def test_a_worker_reports_the_overflow_by_name(child_env):
    from prometheus.tasks.manager import get_task_manager
    from prometheus.tools.base import ToolExecutionContext
    from prometheus.tools.builtin.task_create import TaskCreateTool, TaskCreateToolInput

    work = child_env
    with FakeLlamaServer(work_dir=work, n_ctx=500) as box:
        cfg_dir = Path(os.environ["PROMETHEUS_CONFIG_DIR"])
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "prometheus.yaml").write_text(
            f"model:\n  provider: llama_cpp\n  base_url: {box.url}\n  model: fake-local-27b\n"
            f"security:\n  permission_mode: autonomous\n  workspace_root: {work}\n",
            encoding="utf-8")
        result = await TaskCreateTool().execute(
            TaskCreateToolInput(type="local_agent", description="w",
                                prompt=f"{WORKER_MARK} O.", timeout_seconds=60),
            ToolExecutionContext(cwd=work, metadata={
                "session_id": "web", "effective_session_id": "beacon:p", "ephemeral": False,
                "serving_route": {"key": "local", "provider": "llama_cpp",
                                  "model": "fake-local-27b", "backend": None}}))
        mgr = get_task_manager()
        task_id = result.metadata["task_id"]
        deadline = time.monotonic() + 30
        while mgr.get_task(task_id).status == "running" and time.monotonic() < deadline:
            await asyncio.sleep(0.25)
        rec = mgr.get_task(task_id)
        if rec.status == "running":
            await mgr.stop_task(task_id)
    assert rec.status == "failed" and rec.error == "context_overflow", (rec.status, rec.error)
