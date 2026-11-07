"""A model-written shell that can reach the Docker control socket is warned about at boot.

The bash floor confines the process tree it starts; it cannot confine a separate daemon the shell delegates
to. On Linux a daemon user in the ``docker`` group can write ``/run/docker.sock``, and `docker run -v /:/host`
mounts the host root into a container it controls as uid 0 — passwordless root, outside every floor. AppArmor
cannot close this (the profile is broad by design, and denying the socket neither covers the group member's
other roads to dockerd nor is a privilege boundary once they are root). So the floor's doctrine — "verify by
outcome, fail loud" — says the operator must be told. This is the Linux cousin of the ``systemd-run --user``
escape already documented in docs/design/master-key-out-of-agent-shell.md.
"""

from __future__ import annotations

import logging
import os

import pytest

from prometheus.permissions import confinement as C


@pytest.fixture(autouse=True)
def _fresh():
    C.reset_cache()
    C.reset_write_cache()
    yield
    C.reset_cache()
    C.reset_write_cache()


@pytest.fixture()
def linux(monkeypatch):
    monkeypatch.setattr(C, "apparmor_possible", lambda: True, raising=False)


def _sock(tmp_path, mode):
    p = tmp_path / "docker.sock"
    p.write_bytes(b"")
    os.chmod(p, mode)
    return str(p)


# ── the probe ────────────────────────────────────────────────────────────────

def test_a_writable_socket_on_linux_is_reachable(linux, tmp_path):
    path = _sock(tmp_path, 0o660)
    reachable, where = C.docker_socket_reachable([path])
    assert reachable and where == path


def test_an_unwritable_socket_is_not_reachable(linux, tmp_path):
    reachable, _ = C.docker_socket_reachable([_sock(tmp_path, 0o000)])
    assert not reachable


def test_an_absent_socket_is_not_reachable(linux, tmp_path):
    reachable, _ = C.docker_socket_reachable([str(tmp_path / "nope.sock")])
    assert not reachable


def test_the_first_reachable_socket_wins(linux, tmp_path):
    absent = str(tmp_path / "nope.sock")
    present = _sock(tmp_path, 0o660)
    reachable, where = C.docker_socket_reachable([absent, present])
    assert reachable and where == present


def test_not_probed_off_linux(monkeypatch, tmp_path):
    # A Mac: Docker Desktop's socket reaches a VM, not the host, so the escape does not apply.
    monkeypatch.setattr(C, "apparmor_possible", lambda: False, raising=False)
    reachable, detail = C.docker_socket_reachable([_sock(tmp_path, 0o660)])
    assert not reachable and "Linux" in detail


# ── the boot warning ─────────────────────────────────────────────────────────

def _announce(caplog):
    from prometheus.security.shell_floor import ShellFloor, announce

    with caplog.at_level(logging.WARNING):
        announce(ShellFloor(read_mode="off", write_mode="off"))
    return [r for r in caplog.records if r.levelno == logging.WARNING]


def test_boot_warns_when_the_socket_is_reachable(linux, tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(C, "DOCKER_SOCKETS", (_sock(tmp_path, 0o660),))
    warnings = [r.getMessage() for r in _announce(caplog)]
    hit = [m for m in warnings if "docker" in m.lower()]
    assert hit, warnings
    msg = hit[0]
    assert "root" in msg.lower()
    assert "docker group" in msg.lower()
    assert "docs/guide/docker.md" in msg


def test_boot_is_silent_when_no_socket_is_reachable(linux, tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(C, "DOCKER_SOCKETS", (str(tmp_path / "nope.sock"),))
    assert not [m for m in (r.getMessage() for r in _announce(caplog)) if "docker" in m.lower()]
