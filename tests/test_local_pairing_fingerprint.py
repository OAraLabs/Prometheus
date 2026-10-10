"""``pair.fp``: the daemon's instance fingerprint, beside ``pair.secret`` (src/prometheus/config/local_pairing.py).

Beacon, on the same Mac, reads ``pair.secret`` and sends it to whatever answers on the daemon's port. Before it
sends a secret, it can now ask ``GET /api/hello`` for ``fp`` and compare it with ``pair.fp``: a file only this
user can read, written by the daemon that owns the pairing directory. So ``pair.fp`` always says exactly what
this daemon's hello says:

* the full daemon writes it at boot, right after it makes or loads the instance key;
* setup mode writes it only when a key ALREADY exists, because setup mode creates no ``~/.prometheus`` state and
  a key is state (tests/test_hello_setup_mode.py pins ``fp == ""`` there). With no key, hello's ``fp`` is empty,
  and an earlier ``pair.fp`` is REMOVED rather than left to vouch for an identity this daemon does not have;
* it gets the secret's protections: a 0700 directory that is ours and never a symlink, a 0600 file written via a
  private temp file and a rename, never followed through a link.

What it is not: proof. ``fp`` is public (hello is unauthenticated), so another local account that once read it
could serve the same value. It tells a client it is talking to THIS machine's daemon and not something else on
the port; key-backed proof is the TLS change's.
"""

from __future__ import annotations

import ast
import logging
import os
import stat
from pathlib import Path

import pytest

from prometheus.config import local_pairing as lp

FP = "0123456789abcdef"


@pytest.fixture
def directory(tmp_path) -> Path:
    return tmp_path / "Application Support" / "Prometheus" / "pairing"


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


# ── the file ─────────────────────────────────────────────────────────────────

def test_it_is_written_beside_the_secret_private_and_whole(directory):
    assert lp.write_fingerprint(FP, directory) is True
    path = directory / "pair.fp"
    assert path == lp.fingerprint_path(directory) and path.parent == lp.secret_path(directory).parent
    assert path.read_text() == FP + "\n"
    assert _mode(path) == 0o600 and _mode(directory) == 0o700
    assert sorted(p.name for p in directory.iterdir()) == ["pair.fp"], "no temp file left behind"
    assert lp.read_fingerprint(directory) == FP


def test_a_new_fingerprint_replaces_the_old_one(directory):
    lp.write_fingerprint(FP, directory)
    lp.write_fingerprint("fedcba9876543210", directory)
    assert lp.read_fingerprint(directory) == "fedcba9876543210"


def test_no_fingerprint_removes_a_stale_one_and_creates_nothing(directory, tmp_path):
    lp.write_fingerprint(FP, directory)
    assert lp.write_fingerprint("", directory) is False
    assert not (directory / "pair.fp").exists()
    elsewhere = tmp_path / "never-made"
    assert lp.write_fingerprint("", elsewhere) is False
    assert not elsewhere.exists(), "an empty fingerprint makes no directory"


@pytest.mark.parametrize("value", ["0123456789ABCDEF", "0123456789abcde", "0123456789abcdef0", "../../etc", "x" * 16])
def test_only_a_fingerprint_shaped_value_is_written(directory, value):
    with pytest.raises(ValueError):
        lp.write_fingerprint(value, directory)
    assert not (directory / "pair.fp").exists()


def test_a_symlinked_directory_is_refused(tmp_path):
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    link = tmp_path / "pairing"
    link.symlink_to(real)
    with pytest.raises(lp.PairingFileError):
        lp.write_fingerprint(FP, link)
    assert list(real.iterdir()) == []


def test_a_symlinked_fp_file_is_replaced_never_followed(directory, tmp_path):
    lp.write_fingerprint(FP, directory)
    victim = tmp_path / "victim"
    victim.write_text("keep me")
    (directory / "pair.fp").unlink()
    (directory / "pair.fp").symlink_to(victim)
    assert lp.read_fingerprint(directory) is None, "a link is never read"
    lp.write_fingerprint(FP, directory)
    assert victim.read_text() == "keep me"
    assert not (directory / "pair.fp").is_symlink() and lp.read_fingerprint(directory) == FP


def test_a_file_others_could_write_is_not_trusted(directory):
    lp.write_fingerprint(FP, directory)
    os.chmod(directory / "pair.fp", 0o666)
    assert lp.read_fingerprint(directory) is None


# ── the boot step ────────────────────────────────────────────────────────────

def test_publishing_is_for_the_app_install_only(directory, monkeypatch, tmp_path):
    monkeypatch.delenv("PROMETHEUS_INSTALL_KIND", raising=False)
    monkeypatch.delenv("PROMETHEUS_LOCAL_PAIRING_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    lp.publish_fingerprint(FP)
    assert not (tmp_path / "home").exists()
    monkeypatch.setenv("PROMETHEUS_LOCAL_PAIRING_DIR", str(directory))
    lp.publish_fingerprint(FP)
    assert lp.read_fingerprint(directory) == FP


def test_publishing_never_raises_and_says_what_failed(tmp_path, monkeypatch, caplog):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "pairing"
    link.symlink_to(real)
    monkeypatch.setenv("PROMETHEUS_LOCAL_PAIRING_DIR", str(link))
    caplog.set_level(logging.WARNING)
    lp.publish_fingerprint(FP)
    assert any("pair.fp" in r.getMessage() for r in caplog.records)


def test_the_daemon_publishes_the_fingerprint_of_the_key_it_just_made():
    """In run_daemon, after ensure_instance_key, with the fingerprint of that key ("" when it failed)."""
    import prometheus.daemon as daemon

    tree = ast.parse(Path(daemon.__file__).read_text(encoding="utf-8"))
    run_daemon = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "run_daemon")

    def lines(name: str) -> list[int]:
        return [n.lineno for n in ast.walk(run_daemon) if isinstance(n, ast.Call)
                and (n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", None)) == name]

    assert lines("ensure_instance_key") and lines("publish_fingerprint"), "run_daemon must do both"
    assert min(lines("publish_fingerprint")) > min(lines("ensure_instance_key"))


# ── setup mode ───────────────────────────────────────────────────────────────

class TestSetupMode:
    @pytest.fixture(autouse=True)
    def _isolated(self, tmp_path, monkeypatch, directory):
        monkeypatch.setenv("PROMETHEUS_LOCAL_PAIRING_DIR", str(directory))
        monkeypatch.setenv("PROMETHEUS_NODE_DIR", str(tmp_path / "node"))
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path / "home" / ".prometheus"))

    def _run(self, monkeypatch):
        from prometheus.web import setup_server

        async def _noop(*args, **kwargs):
            return None

        monkeypatch.setattr(setup_server, "_serve_setup_mode", _noop)
        monkeypatch.setattr(setup_server, "missing_web_stack", lambda: [])
        return setup_server.run_setup_mode()

    def test_with_no_key_there_is_no_fp_and_no_key_is_made(self, monkeypatch, directory, tmp_path):
        lp.write_fingerprint(FP, directory)                    # left by an earlier identity
        self._run(monkeypatch)
        assert (directory / "pair.secret").exists()
        assert not (directory / "pair.fp").exists(), "hello's fp is empty here, so pair.fp must be too"
        assert not (tmp_path / "node").exists(), "setup mode creates no ~/.prometheus state"

    def test_with_a_key_already_there_pair_fp_is_hellos_fp(self, monkeypatch, directory):
        from prometheus.config.instance_key import ensure_instance_key, fingerprint_of, instance_fingerprint

        expected = fingerprint_of(ensure_instance_key())
        self._run(monkeypatch)
        assert lp.read_fingerprint(directory) == expected == instance_fingerprint()


def test_the_shape_is_exactly_the_instance_keys_fingerprint():
    from prometheus.config.instance_key import FINGERPRINT_HEX_CHARS, fingerprint_of

    assert FINGERPRINT_HEX_CHARS == 16
    assert lp._FP_SHAPE.fullmatch(fingerprint_of(b"any SubjectPublicKeyInfo DER"))
