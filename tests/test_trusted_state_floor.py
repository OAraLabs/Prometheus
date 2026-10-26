"""Prometheus's trusted state is under the floors: a model's shell cannot rewrite it, and a file tool must ask.

The trusted state is what the daemon reads as its own: the loaded config file (the permission mode, the floors'
modes, allowed commands and persisted grants, read at boot), the identity files that go into every prompt
(SOUL.md, AGENTS.md, ANATOMY.md), MEMORY.md and USER.md (every prompt, refreshed per turn), and skills/. The
daemon's own writers for these (the memory tool, the skill writers and their scanners) are unconfined and keep
working; a model's SHELL bypassed every one of them.

Measured on a scratch home under the Seatbelt profile before this change: a shell could append to every one of
them. Now:

- the shell is refused: Seatbelt on macOS, the AppArmor profile on Linux, with a boot probe on each;
- a file tool's write asks the owner, even inside a workspace that covers them, under a remembered grant, and with
  the gate off. The config file stays refused outright, as before;
- reading them, and writing anything else in the config directory, still work.

The Seatbelt tests run the real thing on macOS. The AppArmor rule is pinned as text here, and its real-profile
measurement is in tests/test_bash_confinement.py (CI security-floors).
"""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
import sys
from pathlib import Path

import pytest

from prometheus.__main__ import create_security_gate
from prometheus.permissions import confinement as C

PROFILE_FILE = Path(__file__).resolve().parent.parent / "deploy" / "apparmor" / "prometheus-bash"
HOME = "/Users/someone"
CFG = f"{HOME}/.prometheus"
TRUSTED_FILES = ("prometheus.yaml", "SOUL.md", "AGENTS.md", "ANATOMY.md", "MEMORY.md", "USER.md")


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    for name in ("PROMETHEUS_DATA_DIR", "PROMETHEUS_CONFIG_DIR", "PROMETHEUS_LOCAL_PAIRING_DIR"):
        monkeypatch.delenv(name, raising=False)
    C.reset_cache()
    C.reset_loaded_config()
    yield
    C.reset_cache()
    C.reset_loaded_config()


def _write_denies(profile: str) -> str:
    return "\n".join(block for block in profile.split("(deny file-write*")[1:])


# ── Seatbelt: the profile text ───────────────────────────────────────────────

def test_the_profile_refuses_writes_to_every_trusted_file_and_to_skills():
    denies = _write_denies(C.seatbelt_profile(HOME))
    for name in TRUSTED_FILES:
        assert f'(literal "{CFG}/{name}")' in denies, name
    assert f'(subpath "{CFG}/skills")' in denies


def test_a_config_directory_moved_by_the_environment_is_covered(monkeypatch, tmp_path):
    moved = tmp_path.resolve() / "conf"
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(moved))
    denies = _write_denies(C.seatbelt_profile(HOME))
    assert f'(literal "{moved}/AGENTS.md")' in denies and f'(subpath "{moved}/skills")' in denies
    assert f'(literal "{moved}")' in denies, "the moved directory cannot be renamed out from under the rule"


def test_the_config_file_the_daemon_loaded_is_covered_wherever_it_is(tmp_path):
    loaded = tmp_path.resolve() / "deploy" / "config" / "prometheus.yaml"
    C.protect_loaded_config(loaded)
    denies = _write_denies(C.seatbelt_profile(HOME))
    assert f'(literal "{loaded}")' in denies
    assert f'(literal "{loaded.parent}")' in denies


def test_building_the_gate_registers_the_loaded_config_with_the_shell_floor(tmp_path):
    loaded = tmp_path.resolve() / "prometheus.yaml"
    loaded.write_text("security: {}\n")
    create_security_gate({}, str(loaded))
    assert f'(literal "{loaded}")' in _write_denies(C.seatbelt_profile(HOME))


# ── the gate: a file tool's write asks; the config file is refused ──────────

def _home_cfg(name: str) -> str:
    return str(Path.home() / ".prometheus" / name)


def _asks(decision) -> bool:
    return not decision.allowed and decision.requires_confirmation


@pytest.mark.parametrize("target", [*TRUSTED_FILES[1:], "skills/new-skill.md", "skills/auto/x.md"])
@pytest.mark.parametrize("cfg", [
    {},
    {"permission_mode": "autonomous"},
    {"workspace_root": str(Path.home() / ".prometheus")},
    {"grants": [{"kind": "path_prefix", "tool_name": "write_file", "value": str(Path.home() / ".prometheus"),
                 "scope": "persistent"}]},
], ids=["default", "gate-off", "workspace-covers-it", "remembered-grant"])
def test_a_file_tool_write_to_trusted_state_asks_the_owner(cfg, target):
    decision = create_security_gate(cfg).evaluate("write_file", file_path=_home_cfg(target), path_is_write=True)
    assert _asks(decision), (cfg, target, decision)


def test_the_loaded_config_file_stays_refused_outright(tmp_path):
    loaded = tmp_path.resolve() / "prometheus.yaml"
    loaded.write_text("security: {}\n")
    decision = create_security_gate({"permission_mode": "autonomous"}, str(loaded)).evaluate(
        "write_file", file_path=str(loaded), path_is_write=True)
    assert not decision.allowed and not decision.requires_confirmation, decision


def test_reading_trusted_state_and_writing_elsewhere_are_unchanged():
    gate = create_security_gate({"workspace_root": str(Path.home() / ".prometheus")})
    assert gate.evaluate("read_file", file_path=_home_cfg("AGENTS.md"), is_read_only=True).allowed
    assert gate.evaluate("write_file", file_path=_home_cfg("workspace/notes.md"), path_is_write=True).allowed


def test_through_the_loop_a_refused_approval_writes_nothing_and_an_approved_one_lands(monkeypatch, tmp_path):
    """The real path, against a scratch config directory: the prompt is the owner's answer."""
    from prometheus.engine.agent_loop import LoopContext, run_loop
    from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
    from prometheus.engine.usage import UsageSnapshot
    from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
    from prometheus.tools.base import ToolRegistry
    from prometheus.tools.builtin.file_write import FileWriteTool

    conf = tmp_path.resolve() / "conf"
    conf.mkdir()
    agents = conf / "AGENTS.md"
    agents.write_text("owner's instructions\n")
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(conf))

    class _Scripted(ModelProvider):
        def __init__(self) -> None:
            self.n = 0

        async def stream_message(self, request):  # noqa: ANN001
            self.n += 1
            content = ([ToolUseBlock(id="t1", name="write_file", input={"path": str(agents), "content": "obey me"})]
                       if self.n == 1 else [TextBlock(text="done")])
            yield ApiMessageCompleteEvent(message=ConversationMessage(role="assistant", content=content),
                                          usage=UsageSnapshot(input_tokens=1, output_tokens=1), stop_reason="stop")

    def drive(answer: bool) -> list[str]:
        asked: list[str] = []

        async def prompt(tool_name, reason, arguments=None):  # noqa: ANN001
            asked.append(reason)
            return answer

        tools = ToolRegistry()
        tools.register(FileWriteTool())
        ctx = LoopContext(provider=_Scripted(), model="stub", system_prompt="S", max_tokens=64, tool_registry=tools,
                          permission_checker=create_security_gate({"workspace_root": str(conf)}),
                          permission_prompt=prompt, cwd=conf)

        async def go():
            async for _ in run_loop(ctx, [ConversationMessage.from_user_text("go")], session_id="cli"):
                pass
        asyncio.run(go())
        return asked

    assert drive(False) and agents.read_text() == "owner's instructions\n", "a refused write landed"
    assert drive(True) and agents.read_text() == "obey me", "control: the owner's approval still writes"


def test_the_daemon_s_own_memory_writer_is_unaffected(monkeypatch, tmp_path):
    """The memory tool is the daemon writing its own file: it neither asks nor is floored."""
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path))
    from importlib import reload

    from prometheus.config import paths
    from prometheus.memory import hermes_memory_tool

    reload(paths)
    reload(hermes_memory_tool)
    from prometheus.memory.hermes_memory_tool import MemoryTool, MemoryToolInput
    from prometheus.tools.base import ToolExecutionContext

    result = asyncio.run(MemoryTool().execute(MemoryToolInput(operation="add", target="memory", entry="likes tea"),
                                              ToolExecutionContext(cwd=str(tmp_path), metadata={})))
    assert not result.is_error and "likes tea" in (tmp_path / "MEMORY.md").read_text()


# ── AppArmor: the profile text, and the boot probe's logic ───────────────────

def _active_rules() -> list[str]:
    return [re.sub(r"\s+", " ", line.split("#", 1)[0].strip()) for line in PROFILE_FILE.read_text().splitlines()
            if line.split("#", 1)[0].strip()]


def test_the_apparmor_profile_refuses_writes_to_trusted_state():
    rules = _active_rules()
    assert ("deny @{HOME}/.prometheus/{prometheus.yaml,SOUL.md,AGENTS.md,ANATOMY.md,MEMORY.md,USER.md} wl,"
            in rules), rules
    assert "deny @{HOME}/.prometheus/skills/{,**} wl," in rules, rules
    assert "deny @{HOME}/prometheus-deploy/config/prometheus.yaml wl," in rules, \
        "the deploy clone's config (scripts/deploy.sh's default) is the loaded config on a deployed box"


def _label_ok(monkeypatch, *, trusted: str = "refused"):
    monkeypatch.setattr(C, "_probe_label", lambda profile: (True, f"{profile} (enforce)"))
    monkeypatch.setattr(C, "_probe_config_rename", lambda profile: "refused")
    monkeypatch.setattr(C, "_registry_detail", lambda profile: ("; writing the device registry refused", False))
    monkeypatch.setattr(C, "_probe_trusted_rename", lambda profile: trusted)


def test_a_current_profile_says_trusted_state_is_refused(monkeypatch, caplog):
    _label_ok(monkeypatch)
    with caplog.at_level(logging.INFO, logger=C.logger.name):
        ok, detail = C.preflight(C.PROFILE, force=True)
    assert ok and "writing trusted state refused" in detail, detail
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_a_profile_without_the_trusted_state_rule_is_stale(monkeypatch, caplog):
    _label_ok(monkeypatch, trusted="renamed")
    with caplog.at_level(logging.INFO, logger=C.logger.name):
        ok, detail = C.preflight(C.PROFILE, force=True)
    assert ok and "STALE PROFILE" in detail and "trusted state" in detail and C.RELOAD_COMMAND in detail
    assert [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_a_loaded_config_the_profile_cannot_name_is_reported(monkeypatch, tmp_path):
    _label_ok(monkeypatch)
    C.protect_loaded_config(tmp_path / "elsewhere" / "prometheus.yaml")
    ok, detail = C.preflight(C.PROFILE, force=True)
    assert ok and "outside the profile" in detail and "config" in detail, detail


# ── the real thing (macOS only) ──────────────────────────────────────────────

real = pytest.mark.skipif(not (sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").exists()),
                          reason="needs macOS with /usr/bin/sandbox-exec")


def _scratch(home: Path) -> Path:
    cfg = home / ".prometheus"
    (cfg / "skills").mkdir(parents=True)
    for name in TRUSTED_FILES:
        (cfg / name).write_text(f"owner text in {name}\n")
    (cfg / "skills" / "owner-skill.md").write_text("owner skill\n")
    return cfg


def _sh(home: Path, command: str) -> int:
    return subprocess.run([C.SANDBOX_EXEC, "-p", C.seatbelt_profile(home), "/bin/sh", "-c", command],
                          capture_output=True, text=True, timeout=30).returncode


@real
@pytest.mark.parametrize("name", [*TRUSTED_FILES, "skills/owner-skill.md"])
@pytest.mark.parametrize("command", [
    'printf injected >> "{path}"',
    'cp "{path}" "{home}/copy" && printf x >> "{home}/copy" && mv "{home}/copy" "{path}"',
    'rm "{path}"',
    'ln "{path}" "{home}/hardlink"',
], ids=["append", "replace", "delete", "hard-link"])
def test_a_shell_cannot_change_trusted_state(tmp_path, name, command):
    home = tmp_path.resolve()
    cfg = _scratch(home)
    path = cfg / name
    before = path.read_text()
    assert _sh(home, command.format(path=path, home=home)) != 0
    assert path.read_text() == before


@real
def test_a_shell_cannot_add_a_skill(tmp_path):
    home = tmp_path.resolve()
    cfg = _scratch(home)
    assert _sh(home, f'printf x > "{cfg}/skills/planted.md"') != 0
    assert _sh(home, f'mkdir -p "{cfg}/skills/auto" && printf x > "{cfg}/skills/auto/planted.md"') != 0
    assert not (cfg / "skills" / "planted.md").exists()


@real
def test_a_shell_can_still_read_trusted_state_and_write_beside_it(tmp_path):
    home = tmp_path.resolve()
    cfg = _scratch(home)
    assert _sh(home, f'cat "{cfg}/AGENTS.md" "{cfg}/MEMORY.md" "{cfg}/skills/owner-skill.md" >/dev/null') == 0
    assert _sh(home, f'mkdir -p "{cfg}/workspace" && printf ok > "{cfg}/workspace/notes.md" '
                     f'&& printf ok > "{cfg}/other.txt"') == 0


@real
def test_the_boot_probe_measures_the_trusted_state_rule():
    ok, detail = C.seatbelt_preflight(force=True)
    assert ok and "trusted state" in detail, detail


@real
@pytest.mark.parametrize("dropped, words", [
    (tuple(f'/{name}"' for name in TRUSTED_FILES), "the identity files"),
    (('/skills"',), "skill could be planted"),
], ids=["identity-files", "skills"])
def test_the_boot_probe_fails_on_a_profile_missing_either_rule(monkeypatch, dropped, words):
    """By outcome, one rule at a time: each is caught on its own, with its own reason."""
    real_profile = C.seatbelt_profile

    def without(home=None, *, allow_ssh=False):
        return "\n".join(line for line in real_profile(home, allow_ssh=allow_ssh).splitlines()
                         if not any(n in line for n in dropped))

    monkeypatch.setattr(C, "seatbelt_profile", without)
    ok, detail = C.seatbelt_preflight(force=True)
    assert not ok and words in detail, detail
