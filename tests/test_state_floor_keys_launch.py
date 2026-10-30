"""The machine's private keys and the daemon's launch configuration are under the floor (GHSA-9ff8-vhh2-r2cw).

The trusted-state floor (GHSA-wq93-7fvg-53pj) left out:

* ``node/`` — ``node.key`` (the identity-challenge key a paired client pins) and ``instance.key`` (the TLS key a paired
  client pins). A copy of both answers as this Prometheus. Floored for READS and writes, like the pairing secret,
  wherever ``PROMETHEUS_NODE_DIR`` puts it.
* ``data/mcp_servers.json`` — MCP servers the daemon STARTS at boot, outside the floor, and their own credentials.
  Floored for reads and writes; the boot merge now runs the REST door's own check on what it reads, too.
* ``data/job-python/`` — first on PATH for the owner's cron jobs, which run outside the floor: a planted ``curl``
  there runs as the owner.
* ``data/lcm.db*`` — each conversation's workspace binding (the bwrap write area) and profile binding (its tool
  limits), and the summaries fed back into prompts.
* ``config_pins.yaml`` — re-applied over the config at every boot, any key: one write could turn the floors off.
* ``checkpoints/`` — an owner's restore writes what a checkpoint names; restore now trusts nothing in it: the
  workspace is the conversation's binding, a path that leaves it is refused, links are not followed, and
  Prometheus's own state is never written.
* ``profiles/`` — custom agent profiles, which limit a conversation's tools. Trusted state: a file tool's write asks.

Every surface: the gate (file tools), Seatbelt (macOS shells, by outcome below), AppArmor (Linux shells: the shipped
profile's rules, and the boot probe that calls a profile without them stale). Reads stay the shell's where nothing
secret is in the file (lcm.db for backups, config_pins.yaml, checkpoints, profiles).
"""

from __future__ import annotations

import json
import logging
import os
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
DATA = f"{CFG}/data"
NODE = f"{CFG}/node"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    for name in ("PROMETHEUS_DATA_DIR", "PROMETHEUS_CONFIG_DIR", "PROMETHEUS_NODE_DIR"):
        monkeypatch.delenv(name, raising=False)
    C.reset_cache()
    yield
    C.reset_cache()


def _home(*parts: str) -> str:
    return str(Path.home().joinpath(".prometheus", *parts))


GATE_CFGS = [{}, {"permission_mode": "autonomous"}, {"workspace_root": str(Path.home() / ".prometheus")}]
GATE_IDS = ["default", "autonomous", "workspace-covers-it"]

# (path parts under ~/.prometheus, is a read refused too)
SECRET = [("node", "node.key"), ("node", "instance.key"), ("node", "instance.crt"),
          ("data", "mcp_servers.json"), ("data", "mcp_servers.json.tmp"), ("xai_oauth.json",),
          ("xai_oauth.json.tmp",)]
LAUNCH = [("data", "job-python", "curl"), ("data", "job-python", "python3"), ("data", "lcm.db"),
          ("data", "lcm.db-wal"), ("config_pins.yaml",), ("checkpoints", "checkpoints.db"),
          ("checkpoints", "blobs", "ab", "abcdef")]


# ── the gate: the file tools ─────────────────────────────────────────────────

@pytest.mark.parametrize("cfg", GATE_CFGS, ids=GATE_IDS)
@pytest.mark.parametrize("parts", SECRET, ids=lambda p: "/".join(p))
def test_the_gate_refuses_reading_or_writing_a_key_or_the_mcp_store(cfg, parts):
    gate = create_security_gate(cfg)
    read = gate.evaluate("read_file", file_path=_home(*parts), is_read_only=True)
    write = gate.evaluate("write_file", file_path=_home(*parts), path_is_write=True)
    for d in (read, write):
        assert not d.allowed and not d.requires_confirmation, (cfg, parts, d)


@pytest.mark.parametrize("cfg", GATE_CFGS, ids=GATE_IDS)
@pytest.mark.parametrize("parts", LAUNCH, ids=lambda p: "/".join(p))
def test_the_gate_refuses_writing_launch_configuration(cfg, parts):
    d = create_security_gate(cfg).evaluate("write_file", file_path=_home(*parts), path_is_write=True)
    assert not d.allowed and not d.requires_confirmation, (cfg, parts, d)


@pytest.mark.parametrize("cfg", GATE_CFGS, ids=GATE_IDS)
def test_a_file_tool_write_to_a_custom_profile_asks_even_in_autonomous(cfg):
    d = create_security_gate(cfg).evaluate("write_file", file_path=_home("profiles", "research.yaml"),
                                           path_is_write=True)
    assert d.requires_confirmation and "trusted state" in d.reason, (cfg, d)


def test_the_gate_covers_any_home():
    other = "/home/someone-else"
    gate = create_security_gate({"workspace_root": other})
    assert gate.evaluate("write_file", file_path=f"{other}/notes.txt", path_is_write=True).allowed, "control"
    for rel in ("node/node.key", "data/mcp_servers.json", "data/job-python/curl", "data/lcm.db",
                "config_pins.yaml", "checkpoints/checkpoints.db"):
        d = gate.evaluate("write_file", file_path=f"{other}/.prometheus/{rel}", path_is_write=True)
        assert not d.allowed and not d.requires_confirmation, (rel, d)
    d = gate.evaluate("read_file", file_path=f"{other}/.prometheus/node/node.key", is_read_only=True)
    assert not d.allowed, d


def test_the_gate_follows_a_moved_config_directory(monkeypatch, tmp_path):
    moved = tmp_path.resolve() / "conf"
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(moved))
    gate = create_security_gate({})
    for rel in ("node/node.key", "data/mcp_servers.json", "data/job-python/curl", "data/lcm.db",
                "config_pins.yaml", "checkpoints/x"):
        d = gate.evaluate("write_file", file_path=str(moved / rel), path_is_write=True)
        assert not d.allowed and not d.requires_confirmation, (rel, d)
    assert not gate.evaluate("read_file", file_path=str(moved / "node" / "instance.key"), is_read_only=True).allowed


def test_the_controls_still_hold():
    gate = create_security_gate({"permission_mode": "autonomous"})
    assert not gate.evaluate("read_file", file_path=_home("data", "devices.db"), is_read_only=True).allowed
    d = gate.evaluate("write_file", file_path=_home("prometheus.yaml"), path_is_write=True)
    assert d.requires_confirmation and "trusted state" in d.reason


# ── Seatbelt profile text ────────────────────────────────────────────────────

def _blocks(profile: str) -> tuple[str, str]:
    read_and_write = profile.split("(deny file-read* file-write*", 1)[1].split("(deny file-write*", 1)[0]
    write_only = "\n".join(profile.split("(deny file-write*")[1:])
    return read_and_write, write_only


def test_the_profile_refuses_reading_and_writing_the_keys_and_the_mcp_store():
    rw, _ = _blocks(C.seatbelt_profile(HOME))
    assert f'(subpath "{NODE}")' in rw, rw
    assert f'(require-all (subpath "{DATA}") {C._MCP_STORE_REGEX})' in rw, rw
    assert f'(require-all (subpath "{CFG}") {C._XAI_OAUTH_REGEX})' in rw, rw


def test_the_profile_refuses_writing_launch_configuration():
    _, w = _blocks(C.seatbelt_profile(HOME))
    assert f'(subpath "{DATA}/job-python")' in w, w
    assert f'(require-all (subpath "{DATA}") {C._LCM_REGEX})' in w, w
    assert f'(literal "{CFG}/config_pins.yaml")' in w, w
    assert f'(subpath "{CFG}/checkpoints")' in w, w
    assert f'(subpath "{CFG}/profiles")' in w, w


def test_the_new_regexes_are_narrow():
    for regex, hits, misses in (
        (C._MCP_STORE_REGEX, ("mcp_servers.json", "mcp_servers.json.tmp"), ("mcp_notes.json", "my_mcp_servers.json")),
        (C._LCM_REGEX, ("lcm.db", "lcm.db-wal", "lcm.db-shm"), ("lcm.txt", "old_lcm.db", "lcm")),
    ):
        rx = re.compile(re.search(r'regex #"(.*)"\)$', regex).group(1))
        for name in hits:
            assert rx.search(f"{DATA}/{name}"), (regex, name)
        for name in misses:
            assert not rx.search(f"{DATA}/{name}"), (regex, name)


def test_a_moved_config_directory_is_covered_in_the_profile(monkeypatch, tmp_path):
    moved = tmp_path.resolve() / "conf"
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(moved))
    rw, w = _blocks(C.seatbelt_profile(HOME))
    assert f'(subpath "{moved}/node")' in rw, rw
    assert f'(literal "{moved}/config_pins.yaml")' in w and f'(subpath "{moved}/data/job-python")' in w, w


# ── AppArmor profile text ────────────────────────────────────────────────────

def _active_rules() -> list[str]:
    return [re.sub(r"\s+", " ", line.split("#", 1)[0].strip()) for line in PROFILE_FILE.read_text().splitlines()
            if line.split("#", 1)[0].strip()]


@pytest.mark.parametrize("rule", [
    "deny @{HOME}/.prometheus/node/{,**} rwlkmx,",
    "deny @{HOME}/.prometheus/data/mcp_servers.json* rwlk,",
    "deny @{HOME}/.prometheus/xai_oauth.json* rwlk,",
    "deny @{HOME}/.prometheus/data/job-python/{,**} wl,",
    "deny @{HOME}/.prometheus/data/lcm.db* wl,",
    "deny @{HOME}/.prometheus/config_pins.yaml wl,",
    "deny @{HOME}/.prometheus/{checkpoints,profiles}/{,**} wl,",
])
def test_the_apparmor_profile_has_the_rule(rule):
    assert rule in _active_rules(), rule


# ── the boot probe (AppArmor) ────────────────────────────────────────────────

def _label_ok(monkeypatch, *, keys: str = "refused", launch: str = "refused"):
    monkeypatch.setattr(C, "_probe_label", lambda profile: (True, f"{profile} (enforce)"))
    monkeypatch.setattr(C, "_probe_config_rename", lambda profile: "refused")
    monkeypatch.setattr(C, "_registry_detail", lambda profile: ("; writing the device registry refused", False))
    monkeypatch.setattr(C, "_trusted_detail", lambda profile: ("; writing trusted state refused", False))
    monkeypatch.setattr(C, "_cron_detail", lambda profile: ("; writing the scheduled-jobs files refused", False))
    monkeypatch.setattr(C, "_probe_node_read", lambda profile: keys)
    monkeypatch.setattr(C, "_probe_launch_write", lambda profile: launch)
    # The what-starts-Prometheus half is tests/test_startup_floor.py's.
    monkeypatch.setattr(C, "_startup_detail", lambda profile: ("; writing what starts Prometheus refused", False))


def test_a_current_profile_says_the_keys_and_launch_configuration_are_refused(monkeypatch, caplog):
    _label_ok(monkeypatch)
    with caplog.at_level(logging.INFO, logger=C.logger.name):
        ok, detail = C.preflight(C.PROFILE, force=True)
    assert ok and "reading the node keys refused" in detail and "writing launch configuration refused" in detail
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


@pytest.mark.parametrize("keys, launch", [("readable", "refused"), ("refused", "renamed")])
def test_a_profile_without_the_new_rules_is_stale_and_says_how_to_reload(monkeypatch, caplog, keys, launch):
    _label_ok(monkeypatch, keys=keys, launch=launch)
    with caplog.at_level(logging.INFO, logger=C.logger.name):
        ok, detail = C.preflight(C.PROFILE, force=True)
    assert ok, "a stale profile keeps the floor it has"
    assert "STALE PROFILE" in detail and C.RELOAD_COMMAND in detail, detail
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors and C.RELOAD_COMMAND in errors[0], errors


# ── the real thing (macOS only) ──────────────────────────────────────────────

real = pytest.mark.skipif(not (sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").exists()),
                          reason="needs macOS with /usr/bin/sandbox-exec")


def _seed(home: Path) -> Path:
    cfg = home / ".prometheus"
    for d in ("node", "data/job-python", "checkpoints", "profiles"):
        (cfg / d).mkdir(parents=True)
    (cfg / "node" / "node.key").write_text("private\n")
    (cfg / "node" / "instance.key").write_text("private\n")
    (cfg / "data" / "mcp_servers.json").write_text("{}\n")
    (cfg / "xai_oauth.json").write_text("{}\n")
    (cfg / "data" / "lcm.db").write_text("x\n")
    (cfg / "data" / "job-python" / "python3").write_text("#!/bin/sh\n")
    (cfg / "config_pins.yaml").write_text("model.provider: llama_cpp\n")
    (cfg / "checkpoints" / "checkpoints.db").write_text("x\n")
    (cfg / "profiles" / "research.yaml").write_text("name: research\n")
    return cfg


def _sh(home: Path, command: str) -> int:
    return subprocess.run([C.SANDBOX_EXEC, "-p", C.seatbelt_profile(home), "/bin/sh", "-c", command],
                          capture_output=True, text=True, timeout=30).returncode


@real
@pytest.mark.parametrize("rel", ["node/node.key", "node/instance.key", "data/mcp_servers.json", "xai_oauth.json"])
def test_a_shell_cannot_read_a_key_or_the_mcp_store(tmp_path, rel):
    home = tmp_path.resolve()
    cfg = _seed(home)
    assert _sh(home, f'cat "{cfg / rel}"') != 0
    assert _sh(home, f'cp "{cfg / rel}" "{home}/copy"') != 0


@real
def test_a_shell_cannot_move_the_node_directory_out_from_under_its_rule(tmp_path):
    home = tmp_path.resolve()
    cfg = _seed(home)
    assert _sh(home, f'mv "{cfg}/node" "{home}/moved"') != 0
    assert (cfg / "node" / "node.key").exists()


@real
@pytest.mark.parametrize("rel", ["data/mcp_servers.json", "data/job-python/python3", "data/job-python/curl",
                                 "data/lcm.db", "data/lcm.db-wal", "config_pins.yaml", "checkpoints/checkpoints.db",
                                 "profiles/research.yaml", "profiles/full.yaml"])
def test_a_shell_cannot_write_launch_configuration(tmp_path, rel):
    home = tmp_path.resolve()
    cfg = _seed(home)
    before = (cfg / rel).read_text() if (cfg / rel).exists() else None
    assert _sh(home, f'printf x >> "{cfg / rel}"') != 0
    after = (cfg / rel).read_text() if (cfg / rel).exists() else None
    assert after == before, rel


@real
def test_a_shell_cannot_plant_a_symlink_as_the_mcp_store(tmp_path):
    home = tmp_path.resolve()
    cfg = _seed(home)
    (cfg / "data" / "mcp_servers.json").unlink()
    assert _sh(home, f'ln -s "{home}/elsewhere.json" "{cfg}/data/mcp_servers.json"') != 0
    assert not (cfg / "data" / "mcp_servers.json").exists()


@real
@pytest.mark.parametrize("rel", ["data/lcm.db", "config_pins.yaml", "profiles/research.yaml"])
def test_a_shell_can_still_read_what_holds_no_secret(tmp_path, rel):
    home = tmp_path.resolve()
    cfg = _seed(home)
    assert _sh(home, f'cat "{cfg / rel}" >/dev/null') == 0


@real
def test_a_shell_can_still_write_beside_them(tmp_path):
    home = tmp_path.resolve()
    cfg = _seed(home)
    assert _sh(home, f'printf x > "{cfg}/data/notes.txt" && printf x > "{cfg}/workspace-note.txt"') == 0


@real
def test_the_seatbelt_preflight_measures_the_new_rules():
    ok, detail = C.seatbelt_preflight(force=True)
    assert ok, detail
    assert "reading the node keys refused" in detail and "writing launch configuration refused" in detail, detail


# ── defence in depth: the MCP store and checkpoint restore ──────────────────

def test_the_mcp_store_skips_a_definition_that_rest_would_refuse(tmp_path, caplog):
    from prometheus.mcp.store import McpServerStore, merged_server_configs

    path = tmp_path / "mcp_servers.json"
    path.write_text(json.dumps({
        "good": {"command": "uvx", "args": ["docs-mcp"]},
        "hijack": {"command": "/bin/sh", "args": ["-c", "true"], "env": {"LD_PRELOAD": "/tmp/x.so"}},
    }))
    with caplog.at_level(logging.WARNING):
        merged = merged_server_configs({}, McpServerStore(path=path))
    assert set(merged) == {"good"}, merged
    assert any("hijack" in r.getMessage() for r in caplog.records)


def _checkpoint(tmp_path):
    from prometheus.checkpoints.store import FileCheckpointStore

    ws = (tmp_path / "ws").resolve()
    (ws / "sub").mkdir(parents=True)
    (ws / "a.txt").write_text("one")
    (ws / "sub" / "b.txt").write_text("one")
    store = FileCheckpointStore(tmp_path / "cp")
    return ws, store, store.create("s", ws)


def _tamper(store, sql: str, *args) -> None:
    import sqlite3

    with sqlite3.connect(store.root / "checkpoints.db") as conn:
        conn.execute(sql, args)


@pytest.mark.parametrize("rel", ["../outside.txt", "sub/../../outside.txt", "/tmp/outside.txt", "sub//b.txt", "."])
def test_a_checkpoint_restore_refuses_a_manifest_that_leaves_its_workspace(tmp_path, rel):
    from prometheus.checkpoints.store import CheckpointRefused

    ws, store, rec = _checkpoint(tmp_path)
    _tamper(store, "UPDATE checkpoint_files SET rel_path = ? WHERE checkpoint_id = ? AND rel_path = 'a.txt'",
            rel, rec.id)
    (ws / "sub" / "b.txt").write_text("two")
    with pytest.raises(CheckpointRefused, match="nothing was restored"):
        store.restore(rec.id, workspace=ws)
    assert not (tmp_path / "outside.txt").exists()
    assert (ws / "sub" / "b.txt").read_text() == "two", "refused whole: nothing else restored either"


def test_a_checkpoint_restore_refuses_a_workspace_the_manifest_moved(tmp_path):
    from prometheus.checkpoints.store import CheckpointRefused

    ws, store, rec = _checkpoint(tmp_path)
    elsewhere = (tmp_path / "elsewhere").resolve()
    elsewhere.mkdir()
    _tamper(store, "UPDATE checkpoints SET workspace = ? WHERE id = ?", str(elsewhere), rec.id)
    with pytest.raises(CheckpointRefused, match="point the conversation back"):
        store.restore(rec.id, workspace=ws)
    assert list(elsewhere.iterdir()) == []


def test_a_checkpoint_restore_never_writes_prometheus_state(tmp_path, monkeypatch):
    from prometheus.checkpoints.store import CheckpointRefused

    ws, store, rec = _checkpoint(tmp_path)
    # A workspace that IS Prometheus's own state (a manifest and a binding both pointed there) is refused whole.
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(ws))
    with pytest.raises(CheckpointRefused, match="own state"):
        store.restore(rec.id, workspace=ws)
    monkeypatch.delenv("PROMETHEUS_CONFIG_DIR")
    # A workspace that CONTAINS it (a home) restores the rest and names what it refused.
    cfg = ws / "sub"
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(cfg))
    (cfg / "b.txt").write_text("two")
    (ws / "a.txt").write_text("two")
    res = store.restore(rec.id, workspace=ws)
    assert (ws / "a.txt").read_text() == "one"
    assert (cfg / "b.txt").read_text() == "two"
    assert [e["path"] for e in res["errors"]] == ["sub/b.txt"] and "own state" in res["errors"][0]["error"]


def test_a_checkpoint_restore_does_not_follow_a_planted_symlink(tmp_path):
    ws, store, rec = _checkpoint(tmp_path)
    target = (tmp_path / "target").resolve()
    target.mkdir()
    (ws / "sub" / "b.txt").unlink()
    (ws / "sub").rmdir()
    (ws / "sub").symlink_to(target, target_is_directory=True)
    res = store.restore(rec.id, workspace=ws)
    assert list(target.iterdir()) == [], "nothing written through the link"
    assert [e["path"] for e in res["errors"]] == ["sub/b.txt"], res


def test_a_checkpoint_restore_replaces_a_symlinked_file_rather_than_writing_through_it(tmp_path):
    ws, store, rec = _checkpoint(tmp_path)
    victim = tmp_path / "victim.txt"
    victim.write_text("keep")
    (ws / "a.txt").unlink()
    (ws / "a.txt").symlink_to(victim)
    store.restore(rec.id, workspace=ws)
    assert victim.read_text() == "keep"
    assert not (ws / "a.txt").is_symlink() and (ws / "a.txt").read_text() == "one"


@pytest.mark.parametrize("sha", ["../../../outside", "/etc/hosts", "AB" * 32, "ab" * 31])
def test_a_checkpoint_restore_refuses_a_blob_id_that_is_not_a_sha256(tmp_path, sha):
    from prometheus.checkpoints.store import CheckpointRefused

    ws, store, rec = _checkpoint(tmp_path)
    (tmp_path / "outside").write_text("not the workspace's")
    _tamper(store, "UPDATE checkpoint_files SET sha256 = ? WHERE checkpoint_id = ? AND rel_path = 'a.txt'",
            sha, rec.id)
    (ws / "a.txt").unlink()
    with pytest.raises(CheckpointRefused, match="not a sha256"):
        store.restore(rec.id, workspace=ws)
    assert not (ws / "a.txt").exists()


def test_a_checkpoint_restore_never_sets_setuid_setgid_or_sticky(tmp_path):
    ws, store, rec = _checkpoint(tmp_path)
    _tamper(store, "UPDATE checkpoint_files SET mode = ? WHERE checkpoint_id = ? AND rel_path = 'a.txt'",
            0o107755, rec.id)
    (ws / "a.txt").unlink()
    store.restore(rec.id, workspace=ws)
    assert (ws / "a.txt").stat().st_mode & 0o7777 == 0o755


def test_a_checkpoint_never_captures_a_link_or_prometheus_state(tmp_path, monkeypatch):
    from prometheus.checkpoints.store import FileCheckpointStore

    ws = (tmp_path / "ws").resolve()
    (ws / "cfg" / "node").mkdir(parents=True)
    (ws / "cfg" / "node" / "node.key").write_text("PRIVATE")
    (ws / "ordinary.txt").write_text("ok")
    secret = tmp_path / "secret.txt"
    secret.write_text("PRIVATE")
    (ws / "link.txt").symlink_to(secret)
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(ws / "cfg"))
    store = FileCheckpointStore(tmp_path / "cp", config={"skip_dirs": []})
    rec = store.create("s", ws)
    assert [f["rel_path"] for f in rec.files_detail] == ["ordinary.txt"], rec.files_detail
    assert ["cfg/node/node.key"] == [s["rel_path"] for s in rec.skipped_detail if "own state" in s["reason"]]
    blobs = [b.read_bytes() for b in (tmp_path / "cp" / "blobs").rglob("*") if b.is_file()]
    assert b"PRIVATE" not in blobs and blobs == [b"ok"]


def test_a_checkpoint_reads_the_file_it_checked_not_one_swapped_in(tmp_path, monkeypatch):
    """The window the walk used to leave: a name checked as a regular file, then swapped for a link to a secret
    before it was read. Captured from a descriptor opened without following links, the swap is never read."""
    from prometheus.checkpoints import store as S

    ws = (tmp_path / "ws").resolve()
    ws.mkdir()
    (ws / "a.txt").write_text("ok")
    secret = tmp_path / "secret.txt"
    secret.write_text("PRIVATE")
    real_open = S.FileCheckpointStore._open_regular

    def swap_then_open(dir_fd, name):
        os.unlink(name, dir_fd=dir_fd)
        os.symlink(str(secret), name, dir_fd=dir_fd)
        return real_open(dir_fd, name)

    monkeypatch.setattr(S.FileCheckpointStore, "_open_regular", staticmethod(swap_then_open))
    store = S.FileCheckpointStore(tmp_path / "cp")
    rec = store.create("s", ws)
    assert rec.files_detail == []
    assert not [b for b in (tmp_path / "cp" / "blobs").rglob("*") if b.is_file()]


# ── PROMETHEUS_NODE_DIR ──────────────────────────────────────────────────────

def test_the_gate_follows_a_moved_node_directory(monkeypatch, tmp_path):
    moved = tmp_path.resolve() / "keys"
    monkeypatch.setenv("PROMETHEUS_NODE_DIR", str(moved))
    gate = create_security_gate({"permission_mode": "autonomous"})
    for name in ("node.key", "instance.key"):
        for d in (gate.evaluate("read_file", file_path=str(moved / name), is_read_only=True),
                  gate.evaluate("write_file", file_path=str(moved / name), path_is_write=True)):
            assert not d.allowed and not d.requires_confirmation, (name, d)


def test_the_prune_layer_follows_a_moved_node_directory(monkeypatch, tmp_path):
    from prometheus.tools.denied_prune import resolve_denied

    moved = tmp_path.resolve() / "keys"
    monkeypatch.setenv("PROMETHEUS_NODE_DIR", str(moved))
    assert str(moved) in resolve_denied([])


def test_the_profile_follows_a_moved_node_directory(monkeypatch, tmp_path):
    moved = tmp_path.resolve() / "keys"
    monkeypatch.setenv("PROMETHEUS_NODE_DIR", str(moved))
    rw, w = _blocks(C.seatbelt_profile(HOME))
    assert f'(subpath "{moved}")' in rw and f'(subpath "{NODE}")' in rw, rw
    assert f'(literal "{moved.parent}")' in w, "the folder above it cannot be renamed either"


def test_apparmor_says_a_moved_node_directory_is_outside_its_rule(monkeypatch, tmp_path, caplog):
    _label_ok(monkeypatch)
    monkeypatch.setenv("PROMETHEUS_NODE_DIR", str(tmp_path / "keys"))
    with caplog.at_level(logging.INFO, logger=C.logger.name):
        ok, detail = C.preflight(C.PROFILE, force=True)
    assert ok and "node keys are outside the profile's rule" in detail, detail
    assert [r for r in caplog.records if r.levelno >= logging.ERROR]


@real
def test_a_shell_cannot_read_a_key_in_a_moved_node_directory(monkeypatch, tmp_path):
    home = tmp_path.resolve() / "home"
    home.mkdir()
    moved = tmp_path.resolve() / "keys"
    moved.mkdir()
    (moved / "node.key").write_text("private\n")
    monkeypatch.setenv("PROMETHEUS_NODE_DIR", str(moved))
    assert _sh(home, f'cat "{moved}/node.key"') != 0
    assert _sh(home, f'mv "{moved}" "{home}/stolen"') != 0


# ── through the loop ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("rel", ["node/node.key", "data/mcp_servers.json", "data/job-python/curl", "data/lcm.db",
                                 "config_pins.yaml", "checkpoints/checkpoints.db"])
def test_a_model_s_write_file_on_launch_state_is_refused_through_the_loop(monkeypatch, tmp_path, rel):
    """The real path: a scripted model asks write_file for the file, the loop asks the gate, nothing lands. The
    config directory and the workspace are one temporary folder, so the workspace covers it (the case the floor
    exists for) and a failing floor writes nothing real. A write beside it is the control."""
    import asyncio

    from prometheus.engine.agent_loop import LoopContext, run_loop
    from prometheus.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
    from prometheus.engine.usage import UsageSnapshot
    from prometheus.providers.base import ApiMessageCompleteEvent, ModelProvider
    from prometheus.tools.base import ToolRegistry
    from prometheus.tools.builtin.file_write import FileWriteTool

    cfg = tmp_path.resolve() / "cfg"
    target = cfg / rel
    target.parent.mkdir(parents=True)
    target.write_bytes(b"REAL")
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(cfg))

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
        gate = create_security_gate({"workspace_root": str(cfg), "permission_mode": "autonomous"})
        ctx = LoopContext(provider=_Scripted(path), model="stub", system_prompt="S", max_tokens=64,
                          tool_registry=tools, permission_checker=gate, cwd=cfg)

        async def go():
            async for _ in run_loop(ctx, [ConversationMessage.from_user_text("go")], session_id="cli"):
                pass
        asyncio.run(go())

    drive(target)
    assert target.read_bytes() == b"REAL", f"write_file changed {rel} through the loop"
    drive(cfg / "notes.txt")
    assert (cfg / "notes.txt").read_text() == "evil", "control: an ordinary write in the workspace lands"
