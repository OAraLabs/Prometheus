"""C2's ``skill`` tool reaches the configs setup wrote.

Option C2 (#593) added ``skill`` to the shipped ``always_loaded`` set, and that
reached only configs WITHOUT the key. From FL-2 (2026-08-12) until 0.9.5, setup
wrote the shipped set into every config it made: ``oara setup --fast``,
``--noninteractive`` and ``--provider``, ``prometheus-init`` and Beacon's
first-run page. A copied template carries it too. Every one of those configs
pinned the set of its day, so ``skill`` stayed deferred and each boot logged
"config OVERRIDES a shipped default".

What these tests hold:

* a list equal to ANY set Prometheus shipped as the default (all four, from git
  history) follows the CURRENT one, with one INFO line at boot and no warning;
* a customised list is used exactly as written and still warns (these pass on
  main by design: they pin the half that must not change);
* setup, on every path, writes no ``always_loaded``;
* the doctor rows say which case applies.

In-process only: no child process, and every config is a dict or a tmp file.
"""

from __future__ import annotations

import pytest
import yaml

from tests.support.advertisement import build_registry
from tests.test_setup_api_phase2 import (  # noqa: F401 — fixtures, used by name
    config_dir,
    env_file,
)

KEY = "tools.deferred_loading.always_loaded"

#: Every set the template or the shipped constant ever held, oldest first, read
#: from git history on 2026-09-28 (see ALWAYS_LOADED_DEFAULTS_SHIPPED). Data, so
#: an edit of the history in the source is loud here.
FROM_GIT = (
    ("bash", "read_file", "write_file", "edit_file", "grep", "glob", "tool_search"),
    ("bash", "task_create", "read_file", "write_file", "edit_file", "grep", "glob",
     "tool_search"),
    ("bash", "task_create", "read_file", "write_file", "edit_file", "grep", "glob",
     "tool_search", "web_search", "web_fetch", "memory"),
    ("bash", "task_create", "read_file", "write_file", "edit_file", "grep", "glob",
     "tool_search", "skill", "web_search", "web_fetch", "memory"),
)
V094 = FROM_GIT[2]  # what 0.9.4's setup wrote


class _Log:
    """Captures lazy %-formatted lines the way logging would render them."""

    def __init__(self) -> None:
        self.infos: list[str] = []
        self.warnings: list[str] = []

    def info(self, msg: str, *args: object) -> None:
        self.infos.append(msg % args if args else msg)

    def warning(self, msg: str, *args: object) -> None:
        self.warnings.append(msg % args if args else msg)


def _config(always_loaded) -> dict:
    return {"tools": {"deferred_loading": {"enabled": "auto",
                                           "always_loaded": list(always_loaded)}}}


def _advertised(config: dict) -> set[str]:
    """What the model is offered under deferred loading, through the real loader."""
    from prometheus.context.dynamic_tools import DynamicToolLoader

    deferred = config.get("tools", {}).get("deferred_loading")
    loader = DynamicToolLoader(build_registry(), deferred)
    return {s.get("name") for s in loader.schemas_for_run(True)}


def _described(config: dict):
    from prometheus.config.divergence import describe

    [value] = [v for v in describe(config) if v.key == KEY]
    return value


def _boot_lines(config: dict) -> _Log:
    from prometheus.config.divergence import warn_on_divergence

    log = _Log()
    warn_on_divergence(config, log=log)
    return log


# ---------------------------------------------------------------------------
# The history
# ---------------------------------------------------------------------------


def test_the_shipped_history_is_the_one_git_shows():
    from prometheus.config import shipped_defaults as sd

    assert sd.ALWAYS_LOADED_DEFAULTS_SHIPPED == FROM_GIT
    assert sd.SHIPPED_ALWAYS_LOADED == FROM_GIT[-1]


# ---------------------------------------------------------------------------
# Every shipped default follows the current one
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shipped", FROM_GIT, ids=lambda s: f"{len(s)}-tools")
def test_every_shipped_default_follows_the_current_one(shipped):
    current = _advertised({"tools": {"deferred_loading": {"enabled": "auto"}}})
    assert "skill" in current

    assert _advertised(_config(shipped)) == current

    value = _described(_config(shipped))
    assert value.source == "shipped_value"
    assert list(value.resolved) == list(FROM_GIT[-1])

    log = _boot_lines(_config(shipped))
    assert [w for w in log.warnings if KEY in w] == []
    [line] = [i for i in log.infos if KEY in i]
    assert "follows a shipped default" in line and "To pin a list of your own" in line


# ---------------------------------------------------------------------------
# A customised list is the operator's: unchanged, and it still warns
# ---------------------------------------------------------------------------

CUSTOMISED = {
    "0.9.4 default minus memory": V094[:-1],
    "0.9.4 default reordered": tuple(reversed(V094)),
    "current default plus a tool": FROM_GIT[-1] + ("vault_search",),
    "two tools": ("bash", "grep"),
    "empty": (),
}


@pytest.mark.parametrize("custom", list(CUSTOMISED.values()), ids=list(CUSTOMISED))
def test_a_customised_list_is_used_as_written_and_still_warns(custom):
    registered = {t.name for t in build_registry().list_tools()}
    assert _advertised(_config(custom)) == set(custom) & registered

    value = _described(_config(custom))
    assert value.source == "config_override"
    assert list(value.resolved) == list(custom)

    log = _boot_lines(_config(custom))
    assert [i for i in log.infos if KEY in i] == []
    [warning] = [w for w in log.warnings if KEY in w]
    assert "THE CONFIG WINS" in warning


# ---------------------------------------------------------------------------
# Setup writes no always_loaded, on every path
# ---------------------------------------------------------------------------


def _deferred_written(cfg: dict) -> dict:
    return (cfg.get("tools") or {}).get("deferred_loading") or {}


def test_the_setup_writer_leaves_always_loaded_out():
    from prometheus.cli.init import _cloud_default_config, _default_config

    for cfg in (_default_config(None, "some-model"),
                _cloud_default_config("anthropic", "ANTHROPIC_API_KEY", "some-model")):
        deferred = _deferred_written(cfg)
        assert "always_loaded" not in deferred
        assert deferred.get("enabled") == "auto"
        assert "skill" in _advertised(cfg)


def test_the_fast_setup_writes_a_config_with_no_always_loaded(tmp_path):
    """``oara setup --fast/--noninteractive`` and ``prometheus-init``: run_init."""
    from prometheus.cli.init import run_init
    from tests.test_setup_api_phase2 import _FakeLlamaCppHandler, _serve

    with _serve(_FakeLlamaCppHandler) as url:
        config = run_init(noninteractive=True, target_dir=tmp_path, timeout=2.0,
                          probe_url=url)
    assert config is not None
    written = yaml.safe_load((tmp_path / "prometheus.yaml").read_text(encoding="utf-8"))
    assert "always_loaded" not in _deferred_written(written)
    assert "skill" in _advertised(written)


def test_the_fast_setup_with_a_provider_writes_no_always_loaded(tmp_path, monkeypatch):
    """``oara setup --provider``: the cloud fast path, no probe."""
    from prometheus.cli.init import run_init

    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key-for-the-setup-test")
    config = run_init(noninteractive=True, target_dir=tmp_path, provider="anthropic")
    assert config is not None
    written = yaml.safe_load((tmp_path / "prometheus.yaml").read_text(encoding="utf-8"))
    assert "always_loaded" not in _deferred_written(written)


@pytest.mark.parametrize("path", ["local", "cloud"])
def test_beacons_first_run_setup_writes_no_always_loaded(path, env_file, config_dir,
                                                        monkeypatch):
    """The daemon's setup mode (``POST /api/setup/configure``), both paths."""
    pytest.importorskip("fastapi")
    from tests.test_setup_api_phase2 import _FakeLlamaCppHandler, _serve, make_client, pair

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client, _state = make_client()
    headers = pair(client)
    if path == "local":
        with _serve(_FakeLlamaCppHandler) as url:
            resp = client.post("/api/setup/configure", headers=headers,
                               json={"provider": "llama_cpp", "base_url": url,
                                     "model": "gemma4-26b"})
    else:
        resp = client.post("/api/setup/configure", headers=headers,
                           json={"provider": "anthropic",
                                 "api_key": "dummy-key-for-the-setup-test"})
    assert resp.status_code == 200, resp.text
    written = yaml.safe_load((config_dir / "prometheus.yaml").read_text(encoding="utf-8"))
    assert "always_loaded" not in _deferred_written(written)


# ---------------------------------------------------------------------------
# The doctors say which case applies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config, status, says", [
    ({}, "ok", "follows the shipped default"),
    (_config(FROM_GIT[0]), "ok", "follows the shipped default"),
    (_config(V094), "ok", "follows the shipped default"),
    (_config(FROM_GIT[-1] + ("vault_search",)), "ok", "pinned by"),
    (_config(V094[:-1]), "warning", "not offered from the shipped default: skill, memory"),
    (_config(()), "warning", "not offered from the shipped default: bash"),
    ({"tools": {"deferred_loading": {"always_loaded": "bash, grep"}}}, "warning", "not a list"),
], ids=["absent", "7-tools", "0.9.4", "superset", "missing-two", "empty", "not-a-list"])
def test_the_chat_doctor_row_says_which_case_applies(config, status, says):
    from prometheus.infra.doctor import check_always_loaded

    row = check_always_loaded(config)
    assert row.name == "Advertised tools"
    assert row.status == status
    assert says in row.message
    assert (row.fix is not None) == (status == "warning")


def test_oara_doctors_row_carries_the_same_case():
    from prometheus.cli.doctor import check_advertised_tools

    following = check_advertised_tools(_config(V094))
    assert following.status == "ok"
    assert "offered to the model" in following.message
    assert "follows the shipped default" in following.message

    pinned = check_advertised_tools(_config(V094[:-1]))
    assert pinned.status == "warning"
    assert "not offered from the shipped default: skill, memory" in pinned.message


async def test_the_chat_doctor_shows_the_row(tmp_path):
    from unittest.mock import AsyncMock, MagicMock, patch

    from prometheus.infra.doctor import Doctor
    from tests.test_doctor import SAMPLE_REGISTRY, _sample_state

    doctor = Doctor.__new__(Doctor)
    doctor.config = _config(V094)
    doctor.registry = SAMPLE_REGISTRY
    doctor.repo_root = tmp_path
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=MagicMock())
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    with patch("prometheus.infra.doctor.httpx.AsyncClient", return_value=mock_client), \
         patch("prometheus.infra.doctor.get_config_dir", return_value=tmp_path):
        report = await doctor.diagnose(_sample_state())
    [row] = [c for c in report.checks if c.name == "Advertised tools"]
    assert "follows the shipped default" in row.message
