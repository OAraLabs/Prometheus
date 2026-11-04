"""``GET /api/soul`` — the owner reads the persona (SOUL.md) the daemon puts into every prompt.

THE RULE
--------
* The operator reads it: the global token or an owner device. A scoped device (an approved phone, a
  server-to-server caller) is refused ``403 operator_only`` and sees none of it; no credential is a 401.
* It is the file the prompt reads, ``<config dir>/SOUL.md``, and only that file: a SOUL.md that is a link is
  refused rather than followed, so it cannot be pointed at a key or another file the floors protect.
* No SOUL.md is a 404. One over the size cap is a 413 and nothing is read into the answer.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("fastapi")

from tests.support.device_world import World  # noqa: E402

PERSONA = "You are Prometheus. Be direct; never invent facts.\n"


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    d = tmp_path / "prom-config"
    d.mkdir()
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(d))
    return d


@pytest.fixture
def w(tmp_path, config_dir) -> World:
    return World(tmp_path)


def _owner_header(w: World) -> dict:
    owner = w.devices.mint_owner("Beacon on this Mac", "macos", by="test")
    return {"Authorization": f"Bearer {owner['token']}"}


def test_the_operator_reads_the_persona(w, config_dir):
    (config_dir / "SOUL.md").write_text(PERSONA)
    r = w.call("op", "GET", "/api/soul")
    assert r.status_code == 200, r.text
    assert r.json()["content"] == PERSONA


def test_an_owner_device_reads_the_persona(w, config_dir):
    (config_dir / "SOUL.md").write_text(PERSONA)
    r = w.client.get("/api/soul", headers=_owner_header(w))
    assert r.status_code == 200, r.text
    assert r.json()["content"] == PERSONA


def test_a_scoped_device_is_refused_and_sees_none_of_it(w, config_dir):
    (config_dir / "SOUL.md").write_text(PERSONA)
    r = w.call("a", "GET", "/api/soul")
    assert r.status_code == 403
    assert r.json().get("error") == "operator_only"
    assert "Prometheus" not in r.text


def test_no_credential_is_refused(w, config_dir):
    (config_dir / "SOUL.md").write_text(PERSONA)
    r = w.client.get("/api/soul")
    assert r.status_code == 401
    assert "Prometheus" not in r.text


def test_no_persona_is_a_404(w):
    r = w.call("op", "GET", "/api/soul")
    assert r.status_code == 404
    assert r.json()["error"] == "not_found"


def test_a_persona_that_is_a_link_is_not_followed(w, config_dir, tmp_path):
    secret = tmp_path / "node.key"
    secret.write_text("PRIVATE KEY MATERIAL")
    os.symlink(secret, config_dir / "SOUL.md")
    r = w.call("op", "GET", "/api/soul")
    assert r.status_code == 409, r.text
    assert "PRIVATE KEY MATERIAL" not in r.text


def test_a_persona_over_the_cap_is_refused_unread(w, config_dir):
    from prometheus.web import server

    (config_dir / "SOUL.md").write_text("x" * (server.SOUL_MAX_BYTES + 1))
    r = w.call("op", "GET", "/api/soul")
    assert r.status_code == 413
    assert "xxxx" not in r.text


def test_the_route_is_classed_operator_only():
    from prometheus.web.route_access import OPERATOR_ONLY, SCOPED_ALLOWED

    assert ("GET", "/api/soul") in OPERATOR_ONLY
    assert ("GET", "/api/soul") not in SCOPED_ALLOWED
