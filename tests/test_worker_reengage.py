"""A worker reports back to the conversation that spawned it — after one retry
on that conversation's route when a smaller worker route ran out of context.

WHAT HAPPENED (2026-10-09, #704)
--------------------------------
Both workers in the incident were started from a Beacon conversation with
``task_create(type=local_agent, on_complete=reengage)``. The task recorded
``context.session_id`` as its creator, which on the web path is the shared
``"web"`` namespace, not the conversation. So when the parent finally stopped
each worker, the re-engagement turn ran in a session called ``web`` — on the
local model, through Telegram's ``inject_turn`` — and failed. The Beacon
conversation never heard. ``sessions_spawn`` recorded no session at all and
never re-engaged; and with no Telegram adapter configured, re-engagement was
skipped outright.

THE RULE THESE TESTS PIN (Will's rulings D3, D4)
------------------------------------------------
* A worker's task names the TURN's own conversation (``effective_session_id``)
  and re-engages it by default. An explicit ``on_complete`` is honoured.
* Re-engagement is delivered on the conversation's own surface: a
  ``telegram:*`` chat through Telegram, anything else through the web bridge,
  live — with or without Telegram configured. The turn is UNTRUSTED
  (``task_supervisor``), never the user's.
* A worker that runs out of context on a route the worker-model setting
  chose is retried ONCE on the parent's route. One already on the parent's
  route is not retried — the same prompt cannot fit the same window — and the
  report says so.
* The report is plain: what the worker ran on, whether it finished, why not,
  and its output.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from tests.fixtures.fake_llama_server import WORKER_MARK, FakeLlamaServer


# ---------------------------------------------------------------------------
# The task names the conversation and re-engages it
# ---------------------------------------------------------------------------

class _Captured(Exception):
    pass


@pytest.fixture
def captured(monkeypatch, tmp_path):
    from prometheus.tasks import manager as mgr_mod

    seen: dict = {}

    async def fake(self, **kw):
        seen.update(kw)
        raise _Captured()

    monkeypatch.setattr(mgr_mod.BackgroundTaskManager, "create_agent_task", fake)
    cfg_dir = Path(os.environ["PROMETHEUS_CONFIG_DIR"])
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "prometheus.yaml").write_text(
        "model:\n  provider: llama_cpp\n  base_url: http://127.0.0.1:9\n  model: m\n",
        encoding="utf-8")
    return seen


_BEACON_TURN = {
    "session_id": "web", "effective_session_id": "beacon:abc", "ephemeral": False,
    "serving_route": {"key": "local", "provider": "llama_cpp", "model": "m", "backend": None},
}


async def _call(tool: str, **args):
    from prometheus.tools.base import ToolExecutionContext
    from prometheus.tools.builtin.sessions_spawn import SessionsSpawnInput, SessionsSpawnTool
    from prometheus.tools.builtin.task_create import TaskCreateTool, TaskCreateToolInput

    ctx = ToolExecutionContext(cwd=Path.cwd(), metadata=dict(_BEACON_TURN))
    try:
        if tool == "sessions_spawn":
            await SessionsSpawnTool().execute(SessionsSpawnInput(prompt="p"), ctx)
        else:
            await TaskCreateTool().execute(
                TaskCreateToolInput(type="local_agent", description="d", prompt="p", **args), ctx)
    except _Captured:
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ["sessions_spawn", "task_create"])
async def test_a_workers_task_names_the_turns_conversation_and_reengages_it(tool, captured):
    await _call(tool)
    assert captured.get("session_id") == "beacon:abc", captured.get("session_id")
    assert captured.get("on_complete") == "reengage", captured.get("on_complete")


@pytest.mark.asyncio
async def test_an_explicit_on_complete_is_honoured(captured):
    await _call("task_create", on_complete="notify")
    assert captured.get("on_complete") == "notify"


# ---------------------------------------------------------------------------
# Delivered on the conversation's own surface, as an untrusted turn
# ---------------------------------------------------------------------------

def _payload(session_id: str, **extra) -> dict:
    base = {
        "task_id": "a1", "type": "local_agent", "status": "completed",
        "description": "worker", "exit_code": 0, "return_code": 0, "error": None,
        "session_id": session_id, "on_complete": "reengage", "output_tail": "all done",
        "worker_result": {"status": "ok", "reason": None, "route": "local",
                          "model": "fake-local-27b", "provider": "llama_cpp"},
    }
    base.update(extra)
    return base


class _Signal:
    def __init__(self, payload: dict) -> None:
        self.payload = payload


@pytest.mark.asyncio
async def test_a_beacon_conversation_is_reengaged_through_the_bridge_without_telegram():
    from prometheus.tasks.completion_handler import TaskCompletionHandler
    from prometheus.tasks.reengage import ReengageRouter

    calls: list = []

    async def bridge(session_id, content, *, provenance, is_trusted):
        calls.append(("bridge", session_id, provenance, is_trusted, content))
        return "ok"

    router = ReengageRouter(telegram=None)
    router.set_bridge(bridge)
    handler = TaskCompletionHandler(signal_bus=object(), inject_turn=router)
    await handler._on_signal(_Signal(_payload("beacon:abc")))
    assert calls and calls[0][:4] == ("bridge", "beacon:abc", "task_supervisor", False), calls
    assert "fake-local-27b" in calls[0][4] and "finished" in calls[0][4], calls[0][4]


@pytest.mark.asyncio
async def test_a_telegram_chat_is_reengaged_through_telegram():
    from prometheus.tasks.completion_handler import TaskCompletionHandler
    from prometheus.tasks.reengage import ReengageRouter

    calls: list = []

    async def tg(session_id, content, *, provenance, is_trusted):
        calls.append(("telegram", session_id))
        return "ok"

    async def bridge(session_id, content, *, provenance, is_trusted):
        calls.append(("bridge", session_id))
        return "ok"

    router = ReengageRouter(telegram=tg)
    router.set_bridge(bridge)
    await TaskCompletionHandler(signal_bus=object(), inject_turn=router)._on_signal(
        _Signal(_payload("telegram:42")))
    assert calls == [("telegram", "telegram:42")], calls


@pytest.mark.asyncio
async def test_the_bridge_injects_an_untrusted_turn_and_shows_it_live():
    from prometheus.web.ws_server import WebSocketBridge

    class _Session:
        def __init__(self):
            self.added: list = []

        def add_user_message(self, content, provenance="user", is_trusted=True):
            self.added.append((content, provenance, is_trusted))
            return 0

        def last_persisted_row_id(self):
            return 7

    class _Mgr:
        def __init__(self):
            self.session = _Session()

        def get_or_create(self, session_id):
            return self.session

    mgr = _Mgr()
    bridge = WebSocketBridge(session_mgr=mgr, loop_context=object())
    frames: list = []

    async def broadcast(event):
        frames.append(event)

    async def run_agent(session_id, session, raise_on_error=False, **kw):
        return "worker noted", None

    bridge.broadcast = broadcast  # type: ignore[method-assign]
    bridge._run_agent = run_agent  # type: ignore[method-assign]
    text = await bridge.inject_turn("beacon:abc", "a worker finished",
                                    provenance="task_supervisor", is_trusted=False)
    assert text == "worker noted"
    assert mgr.session.added == [("a worker finished", "task_supervisor", False)]
    shown = [f for f in frames if f.get("type") == "chat_message"]
    assert shown and shown[0]["payload"]["provenance"] == "task_supervisor", frames


# ---------------------------------------------------------------------------
# One retry on the parent's route, then a plain report
# ---------------------------------------------------------------------------

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


def _config(work: Path, big_url: str, small_url: str) -> None:
    cfg_dir = Path(os.environ["PROMETHEUS_CONFIG_DIR"])
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "prometheus.yaml").write_text(
        f"model:\n  provider: llama_cpp\n  base_url: {big_url}\n  model: fake-big\n"
        f"backends:\n  small:\n    provider: llama_cpp\n    base_url: {small_url}\n"
        f"security:\n  permission_mode: autonomous\n  workspace_root: {work}\n",
        encoding="utf-8")


async def _spawn_and_wait(work: Path, seconds: float = 60):
    from prometheus.tasks.manager import get_task_manager
    from prometheus.tools.base import ToolExecutionContext
    from prometheus.tools.builtin.task_create import TaskCreateTool, TaskCreateToolInput

    emitted: list = []

    class _Bus:
        async def emit(self, signal):
            emitted.append(signal.payload)

    mgr = get_task_manager()
    mgr.signal_bus = _Bus()
    result = await TaskCreateTool().execute(
        TaskCreateToolInput(type="local_agent", description="worker",
                            prompt=f"{WORKER_MARK} S.", timeout_seconds=90),
        ToolExecutionContext(cwd=work, metadata={
            **_BEACON_TURN,
            "serving_route": {"key": "local", "provider": "llama_cpp", "model": "fake-big",
                              "backend": None}}))
    assert not result.is_error, result.output
    task_id = result.metadata["task_id"]
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and not emitted:
        await asyncio.sleep(0.25)
    rec = mgr.get_task(task_id)
    if rec.status == "running":
        await mgr.stop_task(task_id)
    return rec, emitted


@pytest.mark.asyncio
async def test_a_worker_the_setting_put_on_a_smaller_box_retries_once_on_the_parents_route(
        child_env):
    from prometheus.tasks.completion_handler import _build_reengage_content
    from prometheus.tasks.worker_settings import get_worker_model_setting

    work = child_env
    with FakeLlamaServer(work_dir=work, model_id="fake-big", writes_per_turn=1) as big, \
            FakeLlamaServer(work_dir=work, model_id="fake-small", n_ctx=500) as small:
        _config(work, big.url, small.url)
        get_worker_model_setting().set("small")
        rec, emitted = await _spawn_and_wait(work)
        on_small = small.chat_rows(worker=True)
        on_big = big.chat_rows(worker=True)
    assert len(on_small) == 1 and on_small[0]["outcome"] == "400", on_small
    assert on_big, "the worker was never retried on the parent's route"
    assert rec.status == "completed", (rec.status, rec.error)
    assert len(emitted) == 1, f"{len(emitted)} completion signals for one worker"
    report = _build_reengage_content(emitted[0])
    assert "small" in report and "retried" in report and "finished" in report, report


@pytest.mark.asyncio
async def test_a_worker_already_on_the_parents_route_is_not_retried_and_the_report_says_so(
        child_env):
    from prometheus.tasks.completion_handler import _build_reengage_content

    work = child_env
    with FakeLlamaServer(work_dir=work, model_id="fake-big", n_ctx=500) as big, \
            FakeLlamaServer(work_dir=work, model_id="fake-small") as small:
        _config(work, big.url, small.url)
        rec, emitted = await _spawn_and_wait(work)
        on_big = big.chat_rows(worker=True)
    assert len(on_big) == 1, f"the parent's route was asked {len(on_big)} times"
    assert rec.status == "failed" and rec.error == "context_overflow", (rec.status, rec.error)
    report = _build_reengage_content(emitted[0]) if emitted else ""
    assert "did not finish" in report and "not retried" in report, report
    assert "context" in report, report
