"""Adversarial: /api/files + /api/files/read must confine every read to the
workspace root. Each attack must be REJECTED (403/404) and must NOT leak the
out-of-workspace secret. Guard = realpath(root) + realpath(candidate) with the
containment check AFTER .resolve() (symlinks followed before the check).
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.web.server import create_app  # noqa: E402

SECRET = "TOPSECRET-do-not-leak"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "ok.txt").write_text("safe content", encoding="utf-8")
    (ws / "sub").mkdir()
    (ws / "sub" / "deep.txt").write_text("deep", encoding="utf-8")
    # A secret file OUTSIDE the workspace (sibling of the ws root).
    (tmp_path / "secret.txt").write_text(SECRET, encoding="utf-8")
    monkeypatch.setenv("PROMETHEUS_WORKSPACE_DIR", str(ws))
    return TestClient(create_app({})), ws, tmp_path


def test_traversal_variants_rejected(client):
    c, _ws, _tmp = client
    # Classic, doubled, nested, and the ....// filter-bypass shape. None may escape.
    for p in [
        "../secret.txt",
        "../../etc/passwd",
        "../../../../../../etc/passwd",
        "sub/../../secret.txt",
        "....//secret.txt",
        "....//....//secret.txt",
        "%2e%2e/secret.txt",
        "..%2f..%2fsecret.txt",
    ]:
        r = c.get("/api/files/read", params={"path": p})
        assert r.status_code in (403, 404), f"{p!r} → {r.status_code} (expected 403/404)"
        assert SECRET not in r.text, f"{p!r} LEAKED the secret"


def test_absolute_path_is_contained_not_escaped(client):
    c, _ws, _tmp = client
    # Leading slash is stripped → treated as workspace-relative → never the real /etc/passwd.
    r = c.get("/api/files/read", params={"path": "/etc/passwd"})
    assert r.status_code == 404
    assert "root:" not in r.text


def test_symlink_file_escape_not_followed(client):
    c, ws, tmp = client
    link = ws / "out"
    try:
        link.symlink_to(tmp / "secret.txt")
    except OSError:
        pytest.skip("symlinks unsupported on this platform")
    r = c.get("/api/files/read", params={"path": "out"})
    assert r.status_code == 403 and SECRET not in r.text


def test_symlink_dir_escape_not_followed(client):
    c, ws, tmp = client
    # An in-workspace directory symlink pointing OUT; reading THROUGH it must fail.
    linkdir = ws / "outdir"
    try:
        linkdir.symlink_to(tmp)  # → parent of the workspace
    except OSError:
        pytest.skip("symlinks unsupported on this platform")
    r = c.get("/api/files/read", params={"path": "outdir/secret.txt"})
    assert r.status_code == 403 and SECRET not in r.text
    # …and listing through the escaping symlink dir is rejected too.
    assert c.get("/api/files", params={"path": "outdir"}).status_code == 403


def test_size_cap_enforced(client):
    c, ws, _tmp = client
    (ws / "big.txt").write_text("A" * (300 * 1024), encoding="utf-8")
    body = c.get("/api/files/read", params={"path": "big.txt"}).json()
    assert body["truncated"] is True
    assert body["size"] == 300 * 1024
    assert len(body["content"]) == 256 * 1024  # only the cap is read into memory


def test_files_require_bearer_when_token_set(tmp_path, monkeypatch):
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "ok.txt").write_text("x", encoding="utf-8")
    monkeypatch.setenv("PROMETHEUS_WORKSPACE_DIR", str(ws))
    c = TestClient(create_app({"web": {"api_token": "tok"}}))
    assert c.get("/api/files").status_code == 401
    assert c.get("/api/files/read", params={"path": "ok.txt"}).status_code == 401
    assert c.get("/api/files", headers={"Authorization": "Bearer wrong"}).status_code == 401
    h = {"Authorization": "Bearer tok"}
    assert c.get("/api/files", headers=h).status_code == 200
    assert c.get("/api/files/read", params={"path": "ok.txt"}, headers=h).status_code == 200
