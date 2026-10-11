"""A worker takes its brief as ONE turn, and exits with a result when it is done.

WHAT HAPPENED (2026-10-09, #704)
--------------------------------
A spawned worker was the interactive REPL. It read its prompt from stdin one
line per turn (``input()``), and the manager never closed that pipe. In the
incident a 59-line brief became 59 turns: the first ran 53 tool rounds and
overflowed the window, and each of the other 58 then failed on the same
over-full history within a minute. After an error the REPL just printed it
and waited for the next line, so the worker sat at ``>>>`` for an hour until
the parent stopped it; the task said ``failed`` with no reason.

THE RULE THESE TESTS PIN
------------------------
* A worker (``--worker``) reads its whole brief to EOF and runs it as one
  turn. The manager closes stdin after writing it.
* It exits when the turn ends: 0 when it finished, 1 when it did not. Its
  last line is a machine-readable result (status, reason, route, model),
  which the manager puts on the task record and the completion signal.
* A failure the LOOP chose (the #356 context pre-flight, the iteration cap,
  ...) is a failure with that reason, not a finished turn.
* A running worker takes no more input: ``sessions_send`` is refused.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from tests.fixtures.fake_llama_server import WORKER_MARK, FakeLlamaServer


@pytest.fixture
def child_env(tmp_path, monkeypatch):
    """Keep every spawned child on THIS test's config (as test_worker_route.py)."""
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


def _write_config(local_url: str, work: Path) -> None:
    cfg_dir = Path(os.environ["PROMETHEUS_CONFIG_DIR"])
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "prometheus.yaml").write_text(
        "model:\n  provider: llama_cpp\n"
        f"  base_url: {local_url}\n  model: fake-local-27b\n"
        "security:\n  permission_mode: autonomous\n"
        f"  workspace_root: {work}\n",
        encoding="utf-8",
    )


_LOCAL_TURN = {
    "session_id": "web", "effective_session_id": "beacon:parent", "ephemeral": False,
    "serving_route": {"key": "local", "provider": "llama_cpp", "model": "fake-local-27b",
                      "backend": None},
}

BRIEF = "\n".join([
    f"{WORKER_MARK} M. Implement the step files.",
    "Step 1: module one.",
    "",
    "Step 2: module two.",
    "Step 3: module three.",
    "When done, reply with a summary.",
])


async def _spawn_worker(work: Path, *, timeout_seconds: int = 60):
    from prometheus.tasks.manager import get_task_manager
    from prometheus.tools.base import ToolExecutionContext
    from prometheus.tools.builtin.task_create import TaskCreateTool, TaskCreateToolInput

    result = await TaskCreateTool().execute(
        TaskCreateToolInput(type="local_agent", description="worker", prompt=BRIEF,
                            timeout_seconds=timeout_seconds),
        ToolExecutionContext(cwd=work, metadata=dict(_LOCAL_TURN)),
    )
    assert not result.is_error, result.output
    return get_task_manager(), result.metadata["task_id"]


async def _wait_terminal(mgr, task_id: str, seconds: float):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        rec = mgr.get_task(task_id)
        if rec.status not in ("running", "pending"):
            return rec
        await asyncio.sleep(0.25)
    return mgr.get_task(task_id)


@pytest.mark.asyncio
async def test_a_multiline_brief_is_one_turn_and_the_worker_exits_when_done(child_env):
    work = child_env
    with FakeLlamaServer(work_dir=work, writes_per_turn=1) as local:
        _write_config(local.url, work)
        mgr, task_id = await _spawn_worker(work)
        t0 = time.monotonic()
        rec = await _wait_terminal(mgr, task_id, 30)
        waited = time.monotonic() - t0
        turns = max((r["user_text_msgs"] for r in local.chat_rows(worker=True)), default=0)
        if rec.status == "running":
            await mgr.stop_task(task_id)
    assert turns == 1, f"a {len(BRIEF.splitlines())}-line brief became {turns} user turns"
    assert rec.status == "completed" and rec.return_code == 0, (
        f"worker still {rec.status} after {waited:.0f}s (rc={rec.return_code}, "
        f"error={rec.error})")


@pytest.mark.asyncio
async def test_the_worker_result_is_on_the_task_record_and_the_signal(child_env):
    from prometheus.tasks.manager import get_task_manager

    work = child_env
    emitted: list = []

    class _Bus:
        async def emit(self, signal):
            emitted.append(signal)

    with FakeLlamaServer(work_dir=work, writes_per_turn=1) as local:
        _write_config(local.url, work)
        get_task_manager().signal_bus = _Bus()
        mgr, task_id = await _spawn_worker(work)
        rec = await _wait_terminal(mgr, task_id, 30)
        if rec.status == "running":
            await mgr.stop_task(task_id)
    result = json.loads(rec.metadata.get("worker_result") or "{}")
    assert result.get("status") == "ok", rec.metadata
    assert result.get("route") == "local" and result.get("model") == "fake-local-27b", result
    payload = emitted[-1].payload if emitted else {}
    assert payload.get("worker_result", {}).get("status") == "ok", payload


@pytest.mark.asyncio
async def test_a_worker_whose_turn_fails_exits_failed_with_the_reason(child_env):
    work = child_env
    # Every request is over this window: the worker's first round is refused.
    with FakeLlamaServer(work_dir=work, n_ctx=500) as local:
        _write_config(local.url, work)
        mgr, task_id = await _spawn_worker(work)
        t0 = time.monotonic()
        rec = await _wait_terminal(mgr, task_id, 30)
        waited = time.monotonic() - t0
        overs = sum(1 for r in local.chat_rows(worker=True) if r["outcome"] == "400")
        if rec.status == "running":
            await mgr.stop_task(task_id)
    assert rec.status == "failed" and rec.error not in (None, "", "timeout"), (
        f"worker {rec.status} after {waited:.0f}s with error={rec.error!r}")
    assert overs == 1, f"the failed turn was followed by {overs - 1} more on the same history"
    result = json.loads(rec.metadata.get("worker_result") or "{}")
    assert result.get("status") == "failed" and result.get("reason") == rec.error


@pytest.mark.asyncio
async def test_a_running_worker_takes_no_more_input(child_env):
    from prometheus.tools.base import ToolExecutionContext
    from prometheus.tools.builtin.sessions_send import SessionsSendInput, SessionsSendTool

    work = child_env
    with FakeLlamaServer(work_dir=work, writes_per_turn=40) as local:
        _write_config(local.url, work)
        mgr, task_id = await _spawn_worker(work)
        out = await SessionsSendTool().execute(
            SessionsSendInput(task_id=task_id, message="also do module four"),
            ToolExecutionContext(cwd=work, metadata={}))
        await mgr.stop_task(task_id)
    assert out.is_error and "one brief" in out.output, out.output


@pytest.mark.asyncio
async def test_a_loop_refusal_is_the_workers_failure_not_a_finished_turn():
    """The #356 pre-flight ends the turn with an explanation instead of
    sending it. For a worker that is a failure with the reason, not success."""
    from prometheus.cli.worker import run_worker_turn
    from prometheus.engine.agent_loop import LoopContext

    class _NeverCalled:
        provider_name = "llama_cpp"

        async def stream_message(self, request):  # pragma: no cover - must not run
            raise AssertionError("the pre-flight should refuse before any request")
            yield

    class _OverCompactor:
        def limit_for(self, model=None, backend=None):
            return 1000

        def estimate_total(self, system_prompt, messages, tools_chars=0):
            return 5000

        async def apply(self, messages, **kw):
            return messages

    ctx = LoopContext(provider=_NeverCalled(), model="fake-local-27b", system_prompt="w",
                      max_tokens=256, compactor=_OverCompactor(), backend="local")
    result = await run_worker_turn(ctx, f"{WORKER_MARK} N.", route="local")
    assert result["status"] == "failed", result
    assert result["reason"] == "context_preflight_refusal", result
