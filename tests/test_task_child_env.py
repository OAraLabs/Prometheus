"""A daemon-built task child starts without the daemon's API token or gateway tokens.

A worker (and every other ``floored=False`` child: spawned agents, coding runs)
started with ``{**os.environ, **overlay}``, or inherited ``os.environ`` whole when
there was no overlay. The daemon's environment holds ``PROMETHEUS_API_TOKEN``
(config/api_token.py), and Linux shows a process's starting environment at
``/proc/<pid>/environ``, so the model's shell inside a worker could read the
master token from ``/proc/$PPID/environ`` and approve its own requests.

These start REAL children (the daemon's interpreter, never a crashing one) that
write the environment they were given to a file, and read it back.
"""

from __future__ import annotations

import asyncio
import json
import shlex
import sys
from pathlib import Path

import pytest

from prometheus.tasks.manager import BackgroundTaskManager
from prometheus.tasks.store import TaskStore

# Synthetic. The scrub decides by NAME, so the values need no real shape.
API_TOKEN = "child-env-test-api-token-not-real"
GATEWAY_TOKEN = "child-env-test-telegram-token-not-real"
PROVIDER_KEY = "fake-anthropic-key-child-env"
OPENROUTER_KEY = "fake-openrouter-key-child-env"

_DUMP = (
    "import json, os, sys\n"
    "with open(sys.argv[1], 'w', encoding='utf-8') as fh:\n"
    "    json.dump(dict(os.environ), fh)\n"
)


@pytest.fixture
def daemon_env(monkeypatch, tmp_path):
    """The daemon's environment as api_token.py and the env file leave it."""
    monkeypatch.setenv("PROMETHEUS_API_TOKEN", API_TOKEN)
    monkeypatch.setenv("PROMETHEUS_API_TOKEN_FILE", str(tmp_path / "token-file"))
    monkeypatch.setenv("PROMETHEUS_TELEGRAM_TOKEN", GATEWAY_TOKEN)
    monkeypatch.setenv("PROMETHEUS_SLACK_BOT_TOKEN", GATEWAY_TOKEN)
    monkeypatch.setenv("ANTHROPIC_API_KEY", PROVIDER_KEY)
    monkeypatch.setenv("OPENROUTER_API_KEY", OPENROUTER_KEY)


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMETHEUS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("prometheus.tasks.manager.get_tasks_dir", lambda: tmp_path, raising=False)
    mgr = BackgroundTaskManager()
    mgr.store = TaskStore(tmp_path / "tasks.db")

    async def _prompt_withheld(task_id: str, data: str) -> None:
        return None

    monkeypatch.setattr(mgr, "write_to_task", _prompt_withheld)
    return mgr


@pytest.fixture
def dump(tmp_path) -> tuple[Path, Path]:
    script = tmp_path / "dump_env.py"
    script.write_text(_DUMP, encoding="utf-8")
    return script, tmp_path / "child-env.json"


async def _read_child_env(out: Path) -> dict[str, str]:
    for _ in range(300):
        if out.exists() and out.stat().st_size:
            try:
                return json.loads(out.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass  # still being written
        await asyncio.sleep(0.05)
    raise AssertionError("the child never wrote its environment")


def _assert_no_daemon_secret(env: dict[str, str]) -> None:
    # NAMES only in every message: a failing assert on the dict itself would print
    # the test runner's whole environment, real tokens included.
    leaked = sorted(n for n in ("PROMETHEUS_API_TOKEN", "PROMETHEUS_API_TOKEN_FILE",
                                "PROMETHEUS_TELEGRAM_TOKEN", "PROMETHEUS_SLACK_BOT_TOKEN")
                    if n in env)
    assert leaked == [], "reached the child's environment"
    carrying = sorted(k for k, v in env.items() if API_TOKEN in v or GATEWAY_TOKEN in v)
    assert carrying == [], "these variables carry a daemon token's value"


@pytest.mark.asyncio
async def test_a_worker_child_has_no_api_token_but_keeps_its_provider_keys(
        manager, daemon_env, dump, tmp_path, monkeypatch):
    script, out = dump
    monkeypatch.setattr("prometheus.tasks.manager.prometheus_argv",
                        lambda: [sys.executable, str(script), str(out)])
    manager.worker_approval_url = "http://127.0.0.1:9/api/workers/approvals"
    record = await manager.create_agent_task(
        prompt="brief", description="worker", cwd=str(tmp_path), worker=True)
    assert record.status not in ("blocked", "failed"), record.error
    env = await _read_child_env(out)
    _assert_no_daemon_secret(env)
    assert env.get("ANTHROPIC_API_KEY") == PROVIDER_KEY
    assert env.get("OPENROUTER_API_KEY") == OPENROUTER_KEY
    # The overlay still arrives: who the worker is and where it asks.
    assert env.get("PROMETHEUS_WORKER_TASK_ID") == record.id
    assert env.get("PROMETHEUS_WORKER_APPROVAL_URL") == manager.worker_approval_url
    assert env.get("PATH")


@pytest.mark.asyncio
async def test_a_daemon_built_child_with_no_overlay_does_not_inherit_the_token(
        manager, daemon_env, dump, tmp_path):
    """The ``env=None`` path: no overlay used to mean "inherit os.environ" whole."""
    script, out = dump
    record = await manager.create_shell_task(
        command=shlex.join([sys.executable, str(script), str(out)]),
        description="daemon-built child", cwd=str(tmp_path), task_type="local_agent",
        floored=False)
    assert record.status not in ("blocked", "failed"), record.error
    env = await _read_child_env(out)
    _assert_no_daemon_secret(env)
    assert env.get("ANTHROPIC_API_KEY") == PROVIDER_KEY
    assert env.get("PATH")


def test_a_route_key_the_config_names_is_kept_but_never_a_daemon_token(tmp_path):
    from prometheus.tasks.child_env import provider_key_names, task_child_env

    config = tmp_path / "prometheus.yaml"
    config.write_text(
        "router:\n"
        "  rules:\n"
        "    - match: {task_type: code}\n"
        "      provider: openai\n"
        "      api_key_env: GEMMA_TUNNEL_KEY\n"
        "  fallbacks:\n"
        "    - provider: openai\n"
        "      api_key_env: PROMETHEUS_API_TOKEN\n",
        encoding="utf-8")
    names = provider_key_names(str(config))
    assert "GEMMA_TUNNEL_KEY" in names
    assert "GEMINI_API_KEY" in names and "KLING_SECRET_KEY" in names
    assert "PROMETHEUS_API_TOKEN" not in names
    assert "PROMETHEUS_TELEGRAM_TOKEN" not in names
    env = task_child_env(config_path=str(config), base={
        "PATH": "/usr/bin", "GEMMA_TUNNEL_KEY": "k", "PROMETHEUS_API_TOKEN": "t",
        "PROMETHEUS_DISCORD_TOKEN": "d", "PROMETHEUS_GITHUB_TOKEN": "g"})
    assert env == {"PATH": "/usr/bin", "GEMMA_TUNNEL_KEY": "k"}
