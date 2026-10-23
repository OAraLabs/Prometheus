"""The device registry (devices.db) is under the shell floor: a model's shell cannot write it, on a Mac or on Linux.

devices.db (config/paths.py) holds who may call the API (``api_devices``: a sha256 of each token), who is an owner
(``owner_devices``) and which device may run desktop tasks (``computer_devices``). The daemon reads it live, so a
row a shell inserted is trusted at once. Measured on a scratch home under the Seatbelt profile before this change:
a shell could insert a device row, append to the file, replace it, delete it, plant a ``-wal`` file, hard-link it,
and rename the data directory out from under any rule. Every one is refused now; reading it, and writing anything
else in the data directory, still work, and the daemon itself (unconfined) still mints and marks.

The same rule is in the AppArmor profile, which the boot probe tests by outcome (rename the registry onto itself:
refused under the rule, a no-op without it) and reports a stale profile at ERROR, as #766 does for ~/.config. The
gate refuses the file tools on it too.

The Seatbelt tests at the bottom run the real thing on macOS. The AppArmor rule is pinned here as text and its
probe logic with stubs; the real-profile measurement needs Linux with the profile loaded.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from prometheus.__main__ import create_security_gate
from prometheus.permissions import confinement as C

PROFILE_FILE = Path(__file__).resolve().parent.parent / "deploy" / "apparmor" / "prometheus-bash"
HOME = "/Users/someone"
DATA = f"{HOME}/.prometheus/data"
REGISTRY_RULE = r'(regex #"/devices\.db[^/]*$")'


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    for name in ("PROMETHEUS_DATA_DIR", "PROMETHEUS_CONFIG_DIR", "PROMETHEUS_LOCAL_PAIRING_DIR"):
        monkeypatch.delenv(name, raising=False)
    C.reset_cache()
    yield
    C.reset_cache()


def _write_denies(profile: str) -> str:
    """Every ``(deny file-write* ...)`` block of the profile, joined (the write-only ones, not read+write)."""
    return "\n".join(block for block in profile.split("(deny file-write*")[1:])


# ── Seatbelt: the profile text ───────────────────────────────────────────────

def test_the_profile_refuses_writes_to_the_device_registry():
    profile = C.seatbelt_profile(HOME)
    assert f'(require-all (subpath "{DATA}") {REGISTRY_RULE})' in _write_denies(profile), profile


def test_the_profile_refuses_renaming_the_data_directory_and_its_parent():
    denies = _write_denies(C.seatbelt_profile(HOME))
    for directory in (DATA, f"{HOME}/.prometheus"):
        assert f'(literal "{directory}")' in denies, directory
    assert f'(literal "{HOME}")' not in denies, "the home itself is never one of them"


@pytest.mark.parametrize("variable, data_dir", [
    ("PROMETHEUS_DATA_DIR", "wired/data"),
    ("PROMETHEUS_CONFIG_DIR", "conf/data"),   # data dir = <config dir>/data
], ids=["data-dir", "config-dir"])
def test_a_data_directory_moved_by_the_environment_is_covered(monkeypatch, tmp_path, variable, data_dir):
    root = tmp_path.resolve()
    monkeypatch.setenv(variable, str(root / data_dir.split("/")[0]) if variable == "PROMETHEUS_CONFIG_DIR"
                       else str(root / data_dir))
    moved = root / data_dir
    denies = _write_denies(C.seatbelt_profile(HOME))
    assert f'(require-all (subpath "{moved}") {REGISTRY_RULE})' in denies
    assert f'(literal "{moved}")' in denies


# ── the gate: the file tools ─────────────────────────────────────────────────

def _home_registry(suffix: str = "") -> str:
    return str(Path.home() / ".prometheus" / "data" / f"devices.db{suffix}")


@pytest.mark.parametrize("cfg", [{}, {"permission_mode": "default", "denied_paths": []}],
                         ids=["shipped", "explicit-empty-list"])
@pytest.mark.parametrize("tool, args", [
    ("write_file", {"file_path": _home_registry(), "path_is_write": True}),
    ("write_file", {"file_path": _home_registry("-wal"), "path_is_write": True}),
    ("edit_file", {"file_path": _home_registry(), "path_is_write": True}),
    ("read_file", {"file_path": _home_registry(), "is_read_only": True}),
    ("write_file", {"file_path": "/home/someone/.prometheus/data/devices.db", "path_is_write": True}),
], ids=["write", "write-wal", "edit", "read", "write-linux-home"])
def test_the_gate_refuses_the_registry_on_every_config_shape(cfg, tool, args):
    """Refused outright by the floor: not "ask", which an approval (or a workspace covering it) would turn into yes."""
    decision = create_security_gate(cfg).evaluate(tool, **args)
    assert not decision.allowed and not decision.requires_confirmation, (tool, args, decision)


def test_the_floor_holds_inside_a_workspace_that_covers_the_data_directory():
    data = str(Path.home() / ".prometheus")
    gate = create_security_gate({"workspace_root": data})
    decision = gate.evaluate("write_file", file_path=_home_registry(), path_is_write=True)
    assert not decision.allowed and not decision.requires_confirmation, decision
    beside = gate.evaluate("write_file", file_path=str(Path(data) / "data" / "notes.txt"), path_is_write=True)
    assert beside.allowed, ("control: the rest of the workspace is writable", beside)


def test_the_gate_follows_a_moved_data_directory(monkeypatch, tmp_path):
    moved = tmp_path.resolve() / "elsewhere"
    monkeypatch.setenv("PROMETHEUS_DATA_DIR", str(moved))
    decision = create_security_gate({}).evaluate("write_file", file_path=str(moved / "devices.db"), path_is_write=True)
    assert not decision.allowed and not decision.requires_confirmation, decision


def test_a_model_s_write_file_on_the_registry_is_refused_through_the_loop(monkeypatch, tmp_path):
    """The real path: a scripted model asks write_file for the registry, the loop asks the gate, nothing lands.

    The data directory AND the workspace are a temporary folder, so even a failing floor writes nothing real, and
    the workspace covering it is the case the floor exists for (an ordinary write there is allowed)."""
    import asyncio

    from prometheus.engine.agent_loop import LoopContext, run_loop
    from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
    from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
    from prometheus.engine.usage import UsageSnapshot
    from prometheus.tools.base import ToolRegistry
    from prometheus.tools.builtin.file_write import FileWriteTool

    data = tmp_path.resolve() / "data"
    data.mkdir()
    registry_file = data / "devices.db"
    registry_file.write_bytes(b"REAL REGISTRY")
    monkeypatch.setenv("PROMETHEUS_DATA_DIR", str(data))

    class _Scripted(ModelProvider):
        def __init__(self, path: Path) -> None:
            self.path = path
            self.n = 0

        async def stream_message(self, request):  # noqa: ANN001
            self.n += 1
            content = ([ToolUseBlock(id="t1", name="write_file", input={"path": str(self.path), "content": "evil"})]
                       if self.n == 1 else [TextBlock(text="done")])
            yield ApiMessageCompleteEvent(message=ConversationMessage(role="assistant", content=content),
                                          usage=UsageSnapshot(input_tokens=1, output_tokens=1), stop_reason="stop")

    def drive(path: Path) -> None:
        tools = ToolRegistry()
        tools.register(FileWriteTool())
        gate = create_security_gate({"workspace_root": str(tmp_path.resolve())})
        ctx = LoopContext(provider=_Scripted(path), model="stub", system_prompt="S", max_tokens=64,
                          tool_registry=tools, permission_checker=gate, cwd=tmp_path.resolve())

        async def go():
            async for _ in run_loop(ctx, [ConversationMessage.from_user_text("go")], session_id="cli"):
                pass
        asyncio.run(go())

    drive(registry_file)
    assert registry_file.read_bytes() == b"REAL REGISTRY", "write_file changed the registry through the loop"
    drive(data / "notes.txt")
    assert (data / "notes.txt").read_text() == "evil", "control: an ordinary write in the workspace still lands"


# ── AppArmor: the profile text, and the boot probe's logic ───────────────────

def _active_rules() -> list[str]:
    return [re.sub(r"\s+", " ", line.split("#", 1)[0].strip()) for line in PROFILE_FILE.read_text().splitlines()
            if line.split("#", 1)[0].strip()]


def test_the_apparmor_profile_refuses_writes_to_the_registry_and_renaming_above_it():
    rules = _active_rules()
    assert "deny @{HOME}/.prometheus/data/devices.db* wl," in rules, rules
    assert "deny @{HOME}/.prometheus/data/ w," in rules, rules
    assert "deny @{HOME}/.prometheus/ w," in rules, rules
    assert "deny @{HOME}/.config/ w," in rules, "control: the earlier floor is still there"


def _label_ok(monkeypatch, *, config: str = "refused", registry: str = "refused"):
    monkeypatch.setattr(C, "_probe_label", lambda profile: (True, f"{profile} (enforce)"))
    monkeypatch.setattr(C, "_probe_config_rename", lambda profile: config)
    monkeypatch.setattr(C, "_probe_registry_rename", lambda profile: registry)


def test_a_current_profile_says_the_registry_is_refused(monkeypatch, caplog):
    _label_ok(monkeypatch)
    with caplog.at_level(logging.INFO, logger=C.logger.name):
        ok, detail = C.preflight(C.PROFILE, force=True)
    assert ok and "writing the device registry refused" in detail, detail
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_a_profile_without_the_registry_rule_is_stale_and_says_how_to_reload(monkeypatch, caplog):
    _label_ok(monkeypatch, registry="renamed")
    with caplog.at_level(logging.INFO, logger=C.logger.name):
        ok, detail = C.preflight(C.PROFILE, force=True)
    assert ok, "a stale profile keeps the floor it has"
    assert "STALE PROFILE" in detail and "device registry" in detail and C.RELOAD_COMMAND in detail
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors and C.RELOAD_COMMAND in errors[0], errors


def test_a_data_directory_the_profile_cannot_name_is_reported(monkeypatch, tmp_path, caplog):
    """The profile names @{HOME}/.prometheus/data; a data directory moved elsewhere is outside it, and says so."""
    monkeypatch.setenv("PROMETHEUS_DATA_DIR", str(tmp_path / "elsewhere"))
    _label_ok(monkeypatch, registry="renamed")
    with caplog.at_level(logging.INFO, logger=C.logger.name):
        ok, detail = C.preflight(C.PROFILE, force=True)
    assert ok and "outside" in detail and "PROMETHEUS_DATA_DIR" in detail, detail


@pytest.mark.parametrize("registry, words", [("absent", "not measured"), ("unknown", "could not run")])
def test_a_registry_check_that_cannot_measure_says_so_without_failing(monkeypatch, registry, words):
    _label_ok(monkeypatch, registry=registry)
    ok, detail = C.preflight(C.PROFILE, force=True)
    assert ok and words in detail, detail


def test_renaming_the_registry_onto_itself_changes_nothing(tmp_path):
    """Why the boot probe is safe on the real file: allowed, the rename is a no-op."""
    db = tmp_path / "devices.db"
    conn = sqlite3.connect(db)
    conn.execute("create table t (x)")
    conn.execute("insert into t values (1)")
    conn.commit()
    conn.close()
    os.rename(db, db)
    assert sqlite3.connect(db).execute("select x from t").fetchall() == [(1,)]


# ── the real thing (macOS only) ──────────────────────────────────────────────

real = pytest.mark.skipif(not (sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").exists()),
                          reason="needs macOS with /usr/bin/sandbox-exec")


def _registry(home: Path) -> Path:
    """A real devices.db made by the daemon's own store (unconfined), with an owner device and a desktop mark."""
    from prometheus.config.device_store import DeviceStore

    data = home / ".prometheus" / "data"
    data.mkdir(parents=True)
    store = DeviceStore(db_path=data / "devices.db")
    owner = store.mint_owner("Owner's Mac", "macos", by="test")
    assert store.set_computer(owner["id"], True, by="test"), "control: the daemon still marks"
    return data / "devices.db"


def _sh(home: Path, command: str) -> int:
    return subprocess.run([C.SANDBOX_EXEC, "-p", C.seatbelt_profile(home), "/bin/sh", "-c", command],
                          capture_output=True, text=True, timeout=30).returncode


def _sql(db: Path, statement: str) -> str:
    return (f'{sys.executable} -c "import sqlite3; c = sqlite3.connect(\'{db}\'); '
            f"c.execute(\\\"{statement}\\\"); c.commit()\"")


@real
@pytest.mark.parametrize("statement", [
    "insert into api_devices (id, name, platform, token_sha256, created_at) values ('evil', 'x', 'x', 'h', 0)",
    "insert into owner_devices (device_id, marked_at, marked_by) values ('evil', 0, 'shell')",
    "insert into computer_devices (device_id, marked_at, marked_by) values ('evil', 0, 'shell')",
], ids=["api-device", "owner-device", "desktop-mark"])
def test_a_shell_cannot_insert_a_row(tmp_path, statement):
    home = tmp_path.resolve()
    db = _registry(home)
    assert _sh(home, _sql(db, statement)) != 0
    conn = sqlite3.connect(db)
    for table in ("api_devices", "owner_devices", "computer_devices"):
        ids = {row[0] for row in conn.execute(f"select {'id' if table == 'api_devices' else 'device_id'} from {table}")}
        assert "evil" not in ids, table


@real
@pytest.mark.parametrize("command", [
    'printf x >> "{db}"',
    'cp "{db}" "{home}/copy" && mv "{home}/copy" "{db}"',
    'rm "{db}"',
    'printf x > "{db}-wal"',
    'ln "{db}" "{home}/hardlink"',
    'mv "{data}" "{home}/moved"',
    'mv "{home}/.prometheus" "{home}/moved"',
], ids=["append", "replace", "delete", "plant-wal", "hard-link", "rename-data-dir", "rename-parent"])
def test_a_shell_cannot_change_the_registry_by_any_route(tmp_path, command):
    home = tmp_path.resolve()
    db = _registry(home)
    before = db.read_bytes()
    assert _sh(home, command.format(db=db, home=home, data=db.parent)) != 0
    assert db.read_bytes() == before and not Path(f"{db}-wal").exists() or Path(f"{db}-wal").stat().st_size == 0


@real
@pytest.mark.parametrize("route", [
    "{py} -c \"import sqlite3; print(sqlite3.connect('file:{db}?mode=ro', uri=True)"
    ".execute('select count(*) from api_devices').fetchone())\"",
    "{py} -c \"import sqlite3; print(sqlite3.connect('{db}').execute('select count(*) from owner_devices').fetchone())\"",
    "{py} -c \"import sqlite3; sqlite3.connect('file:{db}?mode=ro', uri=True).backup(sqlite3.connect('{out}/b.db'))\"",
    'cp "{db}" "{out}/copy.db"',
    'tar -cf "{out}/data.tar" -C "{home}/.prometheus" data',
], ids=["read-only-select", "plain-select", "sqlite-backup", "cp", "tar"])
def test_a_shell_can_still_read_it_and_back_it_up(tmp_path, route):
    """Backups, health checks and cron jobs read the registry; the floor takes writes only."""
    home = tmp_path.resolve()
    db = _registry(home)
    out = home / "backups"
    out.mkdir()
    assert _sh(home, route.format(py=sys.executable, db=db, out=out, home=home)) == 0


@real
def test_a_shell_can_still_write_beside_it(tmp_path):
    home = tmp_path.resolve()
    db = _registry(home)
    assert _sh(home, f'printf ok > "{db.parent}/notes.txt" && printf ok > "{home}/.prometheus/notes.txt"') == 0


def test_the_registry_stays_in_rollback_journal_mode(tmp_path):
    """Why reading needs no write: in WAL mode every reader writes the -shm file, which the floor refuses, so a
    backup or cron job reading the registry would break. Switching the registry to WAL must change the floor too."""
    from prometheus.config.device_store import DeviceStore

    store = DeviceStore(db_path=tmp_path / "devices.db")
    store.mint("Phone", "ios")
    mode = sqlite3.connect(tmp_path / "devices.db").execute("pragma journal_mode").fetchone()[0]
    assert mode == "delete", mode


@real
def test_the_boot_probe_measures_the_registry_rule():
    ok, detail = C.seatbelt_preflight(force=True)
    assert ok and "device registry" in detail, detail


@real
def test_the_boot_probe_fails_on_a_profile_without_the_registry_rule(monkeypatch):
    """By outcome, not by text: a profile missing the rule is caught, with the reason."""
    real_profile = C.seatbelt_profile

    def without_registry_rule(home=None, *, allow_ssh=False):
        return "\n".join(line for line in real_profile(home, allow_ssh=allow_ssh).splitlines()
                         if "devices" not in line and ".prometheus" not in line)

    monkeypatch.setattr(C, "seatbelt_profile", without_registry_rule)
    ok, detail = C.seatbelt_preflight(force=True)
    assert not ok and "device registry" in detail, detail


@real
def test_the_boot_probe_fails_when_the_data_directory_can_be_renamed(monkeypatch):
    """The registry rule alone is escaped by moving its folder: a profile that kept it but lost the folder rules is
    caught too."""
    real_profile = C.seatbelt_profile

    def without_folder_rules(home=None, *, allow_ssh=False):
        return "\n".join(line for line in real_profile(home, allow_ssh=allow_ssh).splitlines()
                         if not ("(literal" in line and ".prometheus" in line))

    monkeypatch.setattr(C, "seatbelt_profile", without_folder_rules)
    ok, detail = C.seatbelt_preflight(force=True)
    assert not ok and "data directory could be renamed" in detail, detail
