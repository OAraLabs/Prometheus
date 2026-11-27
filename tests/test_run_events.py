"""Run-level events (hooks/1.1, D17) — the off-by-default gate and the
terminal coding-run translation. Targeted tests; the full suite is the
builder's single slot."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from prometheus.hooks.events import HookEvent
from prometheus.hooks.run_events import RunEventEmitter, run_events_enabled


# ── the config gate: literal true only ──────────────────────────────────────


@pytest.mark.parametrize("config,expected", [
    (None, False),
    ({}, False),
    ({"blackboard": {}}, False),
    ({"blackboard": {"run_events": {}}}, False),
    ({"blackboard": {"run_events": {"enabled": False}}}, False),
    ({"blackboard": {"run_events": {"enabled": "true"}}}, False),
    ({"blackboard": {"run_events": {"enabled": 1}}}, False),
    ({"blackboard": {"run_events": None}}, False),
    ({"blackboard": None}, False),
    ({"blackboard": {"run_events": {"enabled": True}}}, True),
])
def test_the_gate_is_literal_true_only(config, expected):
    assert run_events_enabled(config) is expected


def test_shipped_template_keeps_run_events_off():
    from pathlib import Path

    import yaml
    template = Path(__file__).parent.parent / "config" / "prometheus.yaml.default"
    cfg = yaml.safe_load(template.read_text())
    assert run_events_enabled(cfg) is False


# ── the seven names exist (additive minor version) ──────────────────────────


def test_the_seven_run_event_names_exist():
    assert HookEvent("run_started") is HookEvent.RUN_STARTED
    assert HookEvent("run_finished") is HookEvent.RUN_FINISHED
    assert HookEvent("run_failed") is HookEvent.RUN_FAILED
    assert HookEvent("worker_escalated") is HookEvent.WORKER_ESCALATED
    assert HookEvent("worker_context_pressure") is HookEvent.WORKER_CONTEXT_PRESSURE
    assert HookEvent("merge_result") is HookEvent.MERGE_RESULT
    assert HookEvent("mission_amended") is HookEvent.MISSION_AMENDED
    # the 1.0 names are untouched
    assert HookEvent("pre_tool_use") is HookEvent.PRE_TOOL_USE


def test_the_registry_loads_a_run_event_subscription():
    from prometheus.hooks.loader import load_hook_registry
    reg = load_hook_registry({
        "run_finished": [{"type": "command", "command": "true"}],
    })
    assert len(reg.get(HookEvent.RUN_FINISHED)) == 1
    # an unknown name still skips with a warning, as before
    reg2 = load_hook_registry({"run.bogus": [{"type": "command", "command": "true"}]})
    assert all(len(reg2.get(ev)) == 0 for ev in HookEvent)


# ── the terminal translation ────────────────────────────────────────────────


def _signal(session_id: str, status: str, return_code: int | None) -> SimpleNamespace:
    return SimpleNamespace(payload={
        "session_id": session_id, "status": status, "return_code": return_code,
        "error": None, "worker_result": None, "artifact_path": None,
    })


@pytest.mark.asyncio
async def test_a_green_coding_run_emits_run_finished():
    ex = AsyncMock()
    em = RunEventEmitter(ex)
    await em._on_terminal(_signal("coding:c1", "completed", 0))
    ex.execute.assert_awaited_once()
    event, payload = ex.execute.await_args.args
    assert event is HookEvent.RUN_FINISHED
    assert payload["facts"]["run_id"] == "c1"
    assert payload["event"] == "run_finished"


@pytest.mark.asyncio
async def test_a_failed_coding_run_emits_run_failed():
    ex = AsyncMock()
    em = RunEventEmitter(ex)
    await em._on_terminal(_signal("coding:c2", "failed", 1))
    event, _ = ex.execute.await_args.args
    assert event is HookEvent.RUN_FAILED
    # completed status with a nonzero exit is still a failure
    ex2 = AsyncMock()
    em2 = RunEventEmitter(ex2)
    await em2._on_terminal(_signal("coding:c3", "completed", 2))
    event2, _ = ex2.execute.await_args.args
    assert event2 is HookEvent.RUN_FAILED


@pytest.mark.asyncio
async def test_other_tasks_are_ignored():
    ex = AsyncMock()
    em = RunEventEmitter(ex)
    await em._on_terminal(_signal("telegram:123", "completed", 0))
    await em._on_terminal(_signal("", "completed", 0))
    await em._on_terminal(SimpleNamespace(payload=None))
    ex.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_executor_means_no_emission():
    em = RunEventEmitter(None)
    await em._on_terminal(_signal("coding:c1", "completed", 0))  # must not raise


@pytest.mark.asyncio
async def test_a_broken_executor_never_raises_into_the_bus():
    ex = AsyncMock()
    ex.execute.side_effect = RuntimeError("hook exploded")
    em = RunEventEmitter(ex)
    await em._on_terminal(_signal("coding:c1", "completed", 0))  # logged, dropped


@pytest.mark.asyncio
async def test_subscribe_is_idempotent_per_bus():
    bus = SimpleNamespace(subscribe=AsyncMock())
    calls: list = []
    bus.subscribe = lambda kind, cb: calls.append(kind)
    em = RunEventEmitter(AsyncMock())
    em.subscribe(bus)
    em.subscribe(bus)
    assert calls == ["task_completed", "task_failed"]


@pytest.mark.asyncio
async def test_started_is_explicit_and_not_called_by_the_terminal_path():
    ex = AsyncMock()
    em = RunEventEmitter(ex)
    await em._on_terminal(_signal("coding:c1", "completed", 0))
    assert ex.execute.await_args.args[0] is HookEvent.RUN_FINISHED
    await em.started(run_id="c9", facts={"item_id": "m.1"})
    assert ex.execute.await_args.args[0] is HookEvent.RUN_STARTED
    assert ex.execute.await_args.args[1]["facts"]["item_id"] == "m.1"


@pytest.mark.asyncio
async def test_an_operator_hook_receives_the_payload_through_the_real_executor(tmp_path):
    """The whole path with the REAL executor: a run_finished command hook
    writes the payload it got; the env-var contract is the same one every
    operator hook uses ($ARGUMENTS / PROMETHEUS_HOOK_PAYLOAD)."""
    import json

    from prometheus.hooks.executor import HookExecutionContext, HookExecutor
    from prometheus.hooks.loader import load_hook_registry

    out = tmp_path / "payload.json"
    reg = load_hook_registry({
        "run_finished": [{
            "type": "command",
            "command": f'printf "%s" "$PROMETHEUS_HOOK_PAYLOAD" > {out}',
        }],
    })
    ex = HookExecutor(registry=reg, context=HookExecutionContext(
        cwd=tmp_path, provider=None, default_model="test"))
    em = RunEventEmitter(ex)
    await em._on_terminal(_signal("coding:c7", "completed", 0))
    # the command hook runs as a task; give it a moment
    for _ in range(50):
        if out.exists() and out.read_text():
            break
        await asyncio.sleep(0.1)
    data = json.loads(out.read_text())
    assert data["event"] == "run_finished"
    assert data["facts"]["run_id"] == "c7"