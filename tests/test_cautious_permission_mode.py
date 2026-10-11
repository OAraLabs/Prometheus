"""``cautious``: the permission mode a NEW Prometheus.app install starts in.

A release gate of the app (docs/design/macos-app-installer.md): on macOS the daemon has neither shell floor (no
bwrap, no AppArmor), so a shell command the model runs can read whatever this user can, and the app is for people
who never chose a terminal. A new app install therefore asks before EVERY shell command the model runs and before
every file write. Reads are not asked about, and what the person types themselves (user origin) is theirs.

What changes, and what does not:

* a new mode, ``cautious`` = ``strict`` (every file write asks) + every model-run ``bash`` call asks. ``strict``
  is unchanged (only network/install commands ask among shell commands), as are ``default`` and ``autonomous``:
  tests/test_gate_computer_floor.py's 840-decision golden pins all three;
* a NEW config written on an app install (``PROMETHEUS_INSTALL_KIND=app``) says ``permission_mode: cautious``,
  from every writer: setup mode's configure, ``oara setup`` (the fast path) and the interactive wizard. Any other
  install writes ``default`` as before;
* an EXISTING config is never changed: a setup re-run carries its mode over, and a config with no mode still
  means ``default``;
* ``/gate on`` restores the CONFIGURED mode, as its own docstring always said, instead of ``default``: otherwise
  ``/gate off`` then ``/gate on`` would quietly turn a cautious install into a default one.
"""

from __future__ import annotations

import pytest
import yaml

import prometheus.setup_wizard as wizard_mod
from prometheus.cli.init import run_init
from prometheus.gateway.commands import cmd_gate
from prometheus.permissions.checker import SecurityGate
from prometheus.permissions.modes import PermissionMode, TrustLevel
from tests.test_setup_api_phase2 import config_dir, env_file  # noqa: F401 — fixtures used below


@pytest.fixture
def ws(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    return root


def _gate(mode, ws):
    return SecurityGate(mode=mode, audit_logger=None, workspace_root=str(ws))


# ── the mode ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("command", ["ls", "echo hi", "cat notes.txt", "python3 script.py"])
def test_cautious_asks_before_every_shell_command_the_model_runs(ws, command):
    d = _gate("cautious", ws).evaluate("bash", command=command, origin="system")
    assert d.requires_confirmation and not d.allowed, (command, d)
    assert "cautious" in d.reason


@pytest.mark.parametrize("tool", ["write_file", "edit_file"])
def test_cautious_asks_before_every_file_write_even_inside_the_workspace(ws, tool):
    d = _gate("cautious", ws).evaluate(tool, file_path=str(ws / "a.txt"), path_is_write=True, origin="system")
    assert d.requires_confirmation and not d.allowed, d


def test_cautious_does_not_ask_for_reads_or_for_what_the_person_types(ws):
    gate = _gate("cautious", ws)
    assert gate.evaluate("read_file", file_path=str(ws / "a.txt"), is_read_only=True, origin="system").allowed
    assert gate.evaluate("bash", command="ls", origin="user").allowed


@pytest.mark.parametrize("mode", ["default", "strict"])
def test_default_and_strict_still_run_an_ordinary_shell_command_unasked(ws, mode):
    assert _gate(mode, ws).evaluate("bash", command="ls", origin="system").allowed


def test_cautious_is_a_named_mode_at_the_approve_tier(ws):
    assert PermissionMode("cautious") is PermissionMode.CAUTIOUS
    assert _gate("cautious", ws)._mode_trust_level() == TrustLevel.APPROVE


# ── through the real dispatch path (agent_loop._execute_tool_call) ───────────
#
# The decision-table tests above call the gate by hand; these drive a tool call the way the agent loop does, with a
# prompt that answers NO, so a write or a command that runs anyway is visible on disk.

def _loop(tmp_path, mode, prompted):
    from prometheus.__main__ import create_tool_registry
    from prometheus.engine.agent_loop import LoopContext

    gate = SecurityGate(mode=mode, workspace_root=[str(tmp_path / "ws")], audit_logger=None)

    async def prompt(tool_name, reason):
        prompted.append(reason)
        return False

    return LoopContext(provider=None, model="t", system_prompt="", max_tokens=512,
                       tool_registry=create_tool_registry({}, gate), permission_checker=gate,
                       permission_prompt=prompt, cwd=str(tmp_path / "ws"))


def _call(ctx, tool, args):
    import asyncio

    from prometheus.engine.agent_loop import _execute_tool_call

    return asyncio.run(_execute_tool_call(ctx, tool, "t1", args))


def test_a_cautious_install_asks_before_a_write_inside_the_workspace_and_a_no_stops_it(tmp_path):
    (tmp_path / "ws").mkdir()
    target = tmp_path / "ws" / "notes.txt"
    prompted: list = []
    _call(_loop(tmp_path, "cautious", prompted), "write_file", {"path": str(target), "content": "x\n"})
    assert prompted and not target.exists(), (prompted, target.exists())
    prompted.clear()
    _call(_loop(tmp_path, "default", prompted), "write_file", {"path": str(target), "content": "x\n"})
    assert not prompted and target.read_text() == "x\n", "default mode changed: an in-workspace write prompted"


def test_a_cautious_install_asks_before_a_shell_command_and_a_no_stops_it(tmp_path):
    (tmp_path / "ws").mkdir()
    marker = tmp_path / "ws" / "ran"
    prompted: list = []
    _call(_loop(tmp_path, "cautious", prompted), "bash", {"command": f"touch {marker}"})
    assert prompted and "cautious" in prompted[0] and not marker.exists(), (prompted, marker.exists())


# ── /gate ────────────────────────────────────────────────────────────────────

def test_gate_on_restores_the_configured_mode_not_default(ws):
    gate = _gate("cautious", ws)
    cmd_gate(gate, "off")
    assert gate.current_mode() == PermissionMode.AUTONOMOUS
    cmd_gate(gate, "on")
    assert gate.current_mode() == PermissionMode.CAUTIOUS, "/gate on downgraded a cautious install"


def test_gate_cautious_and_gate_default_set_those_modes(ws):
    gate = _gate("default", ws)
    assert "cautious" in cmd_gate(gate, "cautious").lower()
    assert gate.current_mode() == PermissionMode.CAUTIOUS
    cmd_gate(gate, "default")
    assert gate.current_mode() == PermissionMode.DEFAULT


# ── what a NEW config says ───────────────────────────────────────────────────

@pytest.fixture
def app_install(monkeypatch):
    monkeypatch.setenv("PROMETHEUS_INSTALL_KIND", "app")


@pytest.fixture
def other_install(monkeypatch):
    monkeypatch.delenv("PROMETHEUS_INSTALL_KIND", raising=False)


@pytest.fixture(autouse=True)
def _cloud_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key-for-tests")


def _fast_setup(target):
    result = run_init(noninteractive=True, target_dir=target, timeout=0.1, candidates=[], provider="openai")
    assert result is not None, "setup wrote nothing"
    return yaml.safe_load((target / "prometheus.yaml").read_text(encoding="utf-8"))


def _mode_of(cfg):
    return (cfg.get("security") or {}).get("permission_mode")


def test_oara_setup_writes_cautious_into_a_new_app_install(tmp_path, app_install):
    assert _mode_of(_fast_setup(tmp_path / "new")) == "cautious"


def test_oara_setup_writes_default_into_a_new_install_that_is_not_the_app(tmp_path, other_install):
    assert _mode_of(_fast_setup(tmp_path / "new")) == "default"


@pytest.mark.parametrize("prior", ["default", "strict", "autonomous", None])
def test_rerunning_oara_setup_on_an_app_install_never_changes_its_mode(tmp_path, app_install, prior):
    tmp_path.mkdir(exist_ok=True)
    security = {} if prior is None else {"permission_mode": prior}
    (tmp_path / "prometheus.yaml").write_text(yaml.safe_dump({"security": security, "system": {"name": "Old"}}),
                                              encoding="utf-8")
    cfg = _fast_setup(tmp_path)
    assert cfg["system"]["name"] == "Prometheus", "the rerun really did rewrite the file"
    assert _mode_of(cfg) == (prior or "default"), "an existing install's permission mode changed"


def test_the_wizard_writes_cautious_into_a_new_app_install(tmp_path, monkeypatch, app_install):
    path = tmp_path / "prometheus.yaml"
    monkeypatch.setattr(wizard_mod, "_config_target", lambda: path)
    wizard_mod.SetupWizard()._write_config()
    assert _mode_of(yaml.safe_load(path.read_text(encoding="utf-8"))) == "cautious"


def test_the_wizard_leaves_an_existing_app_installs_mode_alone(tmp_path, monkeypatch, app_install):
    path = tmp_path / "prometheus.yaml"
    path.write_text(yaml.safe_dump({"security": {"permission_mode": "default"}}), encoding="utf-8")
    monkeypatch.setattr(wizard_mod, "_config_target", lambda: path)
    wizard_mod.SetupWizard()._write_config()
    assert _mode_of(yaml.safe_load(path.read_text(encoding="utf-8"))) == "default"


class TestSetupModeConfigure:
    """The app's own first-run path: setup mode, driven by Beacon."""

    @pytest.fixture(autouse=True)
    def _web(self):
        pytest.importorskip("fastapi")

    def _configure(self, config_dir):
        from tests.test_setup_api_phase2 import make_client, pair

        client, _state = make_client(api_port=8005, ws_port=8010)
        resp = client.post("/api/setup/configure", json={"provider": "anthropic", "api_key": "dummy-key-for-tests"},
                           headers=pair(client))
        assert resp.status_code == 200, resp.text
        return yaml.safe_load((config_dir / "prometheus.yaml").read_text(encoding="utf-8"))

    def test_setup_mode_writes_cautious_on_the_app_install(self, env_file, config_dir, app_install):
        assert _mode_of(self._configure(config_dir)) == "cautious"

    def test_setup_mode_writes_default_elsewhere(self, env_file, config_dir, other_install):
        assert _mode_of(self._configure(config_dir)) == "default"

