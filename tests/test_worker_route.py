"""A worker runs on the route of the conversation that spawned it — or on the
one route the operator chose for workers — and never on a substitute.

WHAT HAPPENED (2026-10-09, #704)
--------------------------------
A Beacon conversation served by qwen3.8-max called ``task_create(type=
local_agent)`` twice. Both workers were served by the local 27B with a 32k
window — the second one launched with ``--model qwen3.8-max`` — filled it,
and failed. Three things put them there:

1. The loop told tools nothing about the model, provider or backend serving
   the turn, so a spawn tool had nothing to inherit. The worker was a fresh
   ``python -m prometheus`` that took the config's ``model:`` (the box the
   registry calls ``local``).
2. On a llama_cpp config, ``--model X`` was overwritten by whatever the box
   reported at ``/v1/models``, while the task record still said ``X``.
3. The obvious repair, passing ``--provider`` too, would have kept the
   config's local ``base_url`` (an explicit ``base_url`` wins outright in the
   provider registry): the cloud request — key included — sent to the box.

THE RULE THESE TESTS PIN
------------------------
* By default a worker inherits the spawning turn's full route: the picker
  key the session chose (``qwen``, ``qwen:qwen3.8-flash``, ``4090``) or
  ``local`` for the primary, resolved again in the child through the same
  catalog the model picker uses, so the endpoint and key come from the route,
  never from the command line.
* One operator setting, ``workers.model`` (Beacon Settings → Worker model),
  replaces that with a single fixed route. Same as chat is the default.
* The agent may not choose. An agent-supplied ``model`` is refused unless it
  names the route the setting already chose.
* A worker the setting puts on a SMALLER window than its parent's is allowed,
  and the spawn result says so with both numbers.
* A route that cannot be served is refused at spawn, with the reason.
  Nothing is ever substituted.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from pathlib import Path

import pytest

from prometheus.utils.child_python import prometheus_argv
from tests.fixtures.fake_llama_server import SPAWN_MARK, WORKER_MARK, FakeLlamaServer


def _fake_key(name: str) -> str:
    # Letters only, built by concatenation: obviously synthetic to every scanner.
    return "fake" + name + "key" + "forworkertests"


@pytest.fixture
def child_env(tmp_path, monkeypatch):
    """Keep every spawned child on THIS test's config, and off real credentials.

    conftest already points PROMETHEUS_CONFIG_DIR / DATA_DIR at tmp for this
    process; a child inherits them. The sitecustomize (as in
    tests/test_spawn_key_not_persisted.py) points the child's REPO_CONFIG_PATH
    at a missing file so a checkout's own config/prometheus.yaml is never read.
    """
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


def _write_config(local_url: str, work: Path, extra: str = "") -> Path:
    cfg_dir = Path(os.environ["PROMETHEUS_CONFIG_DIR"])
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / "prometheus.yaml"
    path.write_text(
        "model:\n"
        "  provider: llama_cpp\n"
        f"  base_url: {local_url}\n"
        "  model: fake-local-27b\n"
        "security:\n"
        "  permission_mode: autonomous\n"
        f"  workspace_root: {work}\n"
        + extra,
        encoding="utf-8",
    )
    return path


def _registry(*tools):
    from prometheus.tools.base import ToolRegistry

    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


async def _wait_for_worker_request(*fakes: FakeLlamaServer, seconds: float = 40) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if any(f.chat_rows(worker=True) for f in fakes):
            return
        await asyncio.sleep(0.25)


async def _stop_all(mgr) -> None:
    for t in mgr.list_tasks():
        if t.status == "running":
            await mgr.stop_task(t.id)


def _ctx(metadata: dict, work: Path):
    from prometheus.tools.base import ToolExecutionContext

    return ToolExecutionContext(cwd=work, metadata=metadata)


def _cloud_route(window: int | None = 1_000_000) -> dict:
    """The parent-turn metadata the loop supplies: a session on /qwen."""
    meta: dict = {
        "session_id": "web",
        "effective_session_id": "beacon:parent",
        "ephemeral": False,
        "serving_route": {"key": "qwen", "provider": "qwen", "model": "qwen3.8-max",
                          "backend": None},
    }
    if window is not None:
        meta["context_window_for"] = (
            lambda model, backend=None: window if backend is None else 32768)
    return meta


# ---------------------------------------------------------------------------
# End to end: a cloud turn spawns a worker
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_worker_is_served_by_the_route_of_the_turn_that_spawned_it(child_env, monkeypatch):
    """The incident, end to end. A turn served by qwen3.8-max calls task_create;
    the worker must be served by qwen3.8-max, not by the config's local box."""
    from prometheus.engine.agent_loop import LoopContext, run_loop
    from prometheus.engine.messages import ConversationMessage
    from prometheus.providers.registry import ProviderRegistry
    from prometheus.tasks.manager import get_task_manager
    from prometheus.tools.builtin.task_create import TaskCreateTool

    work = child_env
    args = {"type": "local_agent", "description": "worker",
            "prompt": f"{WORKER_MARK} A. Write the step files.", "timeout_seconds": 60}
    with FakeLlamaServer(work_dir=work, writes_per_turn=1) as local, FakeLlamaServer(
        n_ctx=1_000_000, model_id="qwen3.8-max", work_dir=work, writes_per_turn=1,
        parent_tool="task_create", parent_args=args,
    ) as cloud:
        _write_config(local.url, work)
        monkeypatch.setenv("QWEN_BASE_URL", cloud.url)
        monkeypatch.setenv("QWEN_API_KEY", _fake_key("qwen"))
        # Only so the old spawn path starts a worker at all (it refused without
        # one); test_a_worker_no_longer_needs_an_anthropic_key covers dropping it.
        monkeypatch.setenv("ANTHROPIC_API_KEY", _fake_key("anthropic"))
        ctx = LoopContext(
            provider=ProviderRegistry.create({"provider": "qwen", "model": "qwen3.8-max"}),
            model="qwen3.8-max", system_prompt="parent", max_tokens=1024,
            tool_registry=_registry(TaskCreateTool()), cwd=work,
        )
        msgs = [ConversationMessage.from_user_text(f"{SPAWN_MARK} now")]
        async for _ in run_loop(ctx, msgs, surface="cli"):
            pass
        mgr = get_task_manager()
        await _wait_for_worker_request(local, cloud)
        await _stop_all(mgr)
        assert any(r["outcome"].startswith("parent") for r in cloud.chat_rows()), \
            "the parent turn never ran on the cloud route"
        on_local = local.chat_rows(worker=True)
        on_cloud = cloud.chat_rows(worker=True)
        assert on_cloud and not on_local, (
            f"worker served by the config primary: {len(on_local)} request(s) on the local box "
            f"(model {on_local[0]['model'] if on_local else None}), "
            f"{len(on_cloud)} on the parent's route")
        assert {r["model"] for r in on_cloud} == {"qwen3.8-max"}


# ---------------------------------------------------------------------------
# The child CLI never substitutes a model
# ---------------------------------------------------------------------------

def test_an_explicit_model_the_local_box_does_not_serve_is_refused(child_env):
    """`--model qwen3.8-max` on a llama_cpp config used to run the box's own GGUF
    under that name. It must refuse instead — the box cannot serve it."""
    work = child_env
    with FakeLlamaServer(work_dir=work) as local:
        cfg = _write_config(local.url, work)
        r = subprocess.run(
            prometheus_argv("--config", str(cfg), "--model", "qwen3.8-max",
                            "--once", f"{WORKER_MARK} B."),
            cwd=work, capture_output=True, text=True, timeout=90,
        )
        served = {row["model"] for row in local.chat_rows()}
    assert not served, f"--model qwen3.8-max was served by the local box as {served}"
    assert r.returncode != 0
    assert "qwen3.8-max" in r.stderr


def test_switching_provider_never_keeps_the_configs_local_base_url(child_env, monkeypatch):
    """`--provider qwen` on a llama_cpp config kept model.base_url, so the cloud
    request and its Authorization header went to the local box."""
    work = child_env
    with FakeLlamaServer(work_dir=work) as local, FakeLlamaServer(
        n_ctx=1_000_000, model_id="qwen3.8-max", work_dir=work,
    ) as cloud:
        cfg = _write_config(local.url, work)
        monkeypatch.setenv("QWEN_BASE_URL", cloud.url)
        monkeypatch.setenv("QWEN_API_KEY", _fake_key("qwen"))
        subprocess.run(
            prometheus_argv("--config", str(cfg), "--provider", "qwen", "--model",
                            "qwen3.8-max", "--once", f"{WORKER_MARK} C."),
            cwd=work, capture_output=True, text=True, timeout=90,
        )
        to_local = local.chat_rows()
        to_cloud = cloud.chat_rows()
    assert not to_local, (
        f"{len(to_local)} cloud request(s) went to the local box "
        f"(Authorization header present: {any(r['auth_header_present'] for r in to_local)})")
    assert to_cloud


def test_a_route_the_child_cannot_serve_is_refused_with_the_reason(child_env):
    work = child_env
    with FakeLlamaServer(work_dir=work) as local:
        cfg = _write_config(local.url, work)
        r = subprocess.run(
            prometheus_argv("--config", str(cfg), "--route", "qwen", "--once", "x"),
            cwd=work, capture_output=True, text=True, timeout=90,
        )
        rows = local.chat_rows()
    assert not rows
    assert r.returncode != 0
    # No QWEN_API_KEY in this environment: the refusal names what is missing.
    assert "qwen" in r.stderr and "QWEN_API_KEY" in r.stderr, r.stderr[-400:]


# ---------------------------------------------------------------------------
# The spawn tools: inherit, refuse an agent's choice, follow the setting
# ---------------------------------------------------------------------------

class _Captured(Exception):
    pass


@pytest.fixture
def captured_spawn(monkeypatch):
    """Capture what the spawn tools ask the manager for; spawn nothing."""
    from prometheus.tasks import manager as mgr_mod

    seen: dict = {}

    async def fake_create_agent_task(self, **kw):
        seen.update(kw)
        raise _Captured()

    monkeypatch.setattr(mgr_mod.BackgroundTaskManager, "create_agent_task",
                        fake_create_agent_task)
    return seen


async def _spawn(tool_name: str, meta: dict, work: Path, **args):
    from prometheus.tools.builtin.sessions_spawn import SessionsSpawnInput, SessionsSpawnTool
    from prometheus.tools.builtin.task_create import TaskCreateTool, TaskCreateToolInput

    try:
        if tool_name == "sessions_spawn":
            return await SessionsSpawnTool().execute(
                SessionsSpawnInput(prompt=f"{WORKER_MARK} D.", **args), _ctx(meta, work))
        return await TaskCreateTool().execute(TaskCreateToolInput(
            type="local_agent", description="w", prompt=f"{WORKER_MARK} D.", **args),
            _ctx(meta, work))
    except _Captured:
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["sessions_spawn", "task_create"])
async def test_the_spawn_tools_hand_the_worker_the_parents_route(
        tool_name, child_env, captured_spawn, monkeypatch):
    _write_config("http://127.0.0.1:9", child_env)
    monkeypatch.setenv("QWEN_API_KEY", _fake_key("qwen"))
    await _spawn(tool_name, _cloud_route(), child_env)
    assert captured_spawn.get("route") == "qwen", captured_spawn
    assert not captured_spawn.get("model"), captured_spawn


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["sessions_spawn", "task_create"])
async def test_an_agent_cannot_choose_a_workers_model(
        tool_name, child_env, captured_spawn, monkeypatch):
    _write_config("http://127.0.0.1:9", child_env)
    monkeypatch.setenv("QWEN_API_KEY", _fake_key("qwen"))
    result = await _spawn(tool_name, _cloud_route(), child_env, model="qwen3.8-flash")
    assert not captured_spawn, f"spawned anyway: {captured_spawn}"
    assert result is not None and result.is_error
    assert "worker model" in result.output.lower()


@pytest.mark.asyncio
async def test_the_worker_model_setting_routes_the_worker_and_says_the_window_is_smaller(
        child_env, monkeypatch):
    from prometheus.tasks import manager as mgr_mod
    from prometheus.tasks.worker_settings import get_worker_model_setting

    seen: dict = {}

    async def fake_create_agent_task(self, **kw):
        seen.update(kw)
        from prometheus.tasks.types import TaskRecord
        return TaskRecord(id="a0", type="local_agent", status="running", description="w",
                          cwd=str(child_env), output_file=child_env / "a0.log")

    monkeypatch.setattr(mgr_mod.BackgroundTaskManager, "create_agent_task",
                        fake_create_agent_task)
    _write_config("http://127.0.0.1:9", child_env)
    get_worker_model_setting().set("local")
    # The agent may name the model the setting already chose; it changes nothing.
    result = await _spawn("sessions_spawn", _cloud_route(window=1_000_000), child_env,
                          model="local")
    assert seen.get("route") == "local", seen
    assert result is not None and not result.is_error, result.output
    text = result.output
    assert "smaller" in text and "32,768" in text and "1,000,000" in text, text


@pytest.mark.asyncio
async def test_a_parent_route_that_cannot_be_handed_on_is_refused(
        child_env, captured_spawn):
    _write_config("http://127.0.0.1:9", child_env)
    meta = _cloud_route()
    meta["serving_route"] = {"key": None, "provider": "openai", "model": "gpt-unlisted",
                             "backend": None}
    result = await _spawn("sessions_spawn", meta, child_env)
    assert not captured_spawn
    assert result is not None and result.is_error
    assert "gpt-unlisted" in result.output


@pytest.mark.asyncio
async def test_a_worker_no_longer_needs_an_anthropic_key(child_env, monkeypatch):
    from prometheus.tasks.manager import BackgroundTaskManager

    seen: dict = {}

    async def fake_shell(self, **kw):
        seen.update(kw)
        raise _Captured()

    monkeypatch.setattr(BackgroundTaskManager, "create_shell_task", fake_shell)
    mgr = BackgroundTaskManager()
    with pytest.raises(_Captured):
        await mgr.create_agent_task(prompt="p", description="d", cwd=str(child_env),
                                    route="local")
    assert "--route" in seen["command"] and "local" in seen["command"]


# ---------------------------------------------------------------------------
# The routing stage records the picker key
# ---------------------------------------------------------------------------

def test_a_route_decision_carries_the_picker_key():
    from prometheus.router.model_router import ModelRouter, RouterConfig

    primary = object()
    router = ModelRouter(RouterConfig(), primary, None, primary_model="fake-local-27b")
    assert router.route("hi", context={"session_id": "s1"}).key == "local"
    router.set_override("s1", {"provider": "llama_cpp", "model": "fake-local-27b",
                               "backend": "local"}, key="local", persist=False)
    router._overrides["s1"].provider = primary  # built; no network
    assert router.route("hi", context={"session_id": "s1"}).key == "local"


# ---------------------------------------------------------------------------
# The setting itself
# ---------------------------------------------------------------------------

def test_the_worker_model_setting_validates_persists_and_defaults_to_same_as_chat(
        child_env, monkeypatch):
    from prometheus.tasks import worker_settings as ws

    _write_config("http://127.0.0.1:9", child_env)
    s = ws.get_worker_model_setting()
    assert s.current() is None and s.source == "default"
    with pytest.raises(ValueError):
        s.set("not-a-model-anyone-offers")
    s.set("local")
    fresh = ws.WorkerModelSetting(config=ws.load_worker_config())
    assert fresh.current() == "local" and fresh.source == "settings"
    s.set(None)
    assert ws.WorkerModelSetting(config=ws.load_worker_config()).current() is None
