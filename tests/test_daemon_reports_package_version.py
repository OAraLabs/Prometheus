"""The daemon reports the version it is, not "0.1.0".

``GET /``, the FastAPI app (``/openapi.json``) and the websocket ``connected``
welcome all said "0.1.0", a literal from the first commit, while the package
was on 0.9.x. They now read ``prometheus.version.package_version()``: the
installed distribution's metadata, or ``prometheus.__version__`` when there is
none.

There is none in production. deploy.sh builds the venv with
``--no-install-project`` and the unit imports from ``PYTHONPATH``, so the
fallback is the live path, and ``__version__`` must agree with pyproject.
"""

from __future__ import annotations

import asyncio
import json
import tomllib
from importlib import metadata
from pathlib import Path

import pytest

import prometheus
from prometheus import version as version_mod

REPO = Path(__file__).resolve().parents[1]
STALE = "0.1.0"


def _pyproject_version() -> str:
    return tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["version"]


@pytest.fixture(autouse=True)
def _fresh_cache():
    version_mod.package_version.cache_clear()
    yield
    version_mod.package_version.cache_clear()


def test_init_version_matches_pyproject():
    """The fallback is what a deployed daemon reports, so it must be right."""
    assert prometheus.__version__ == _pyproject_version()


def test_package_version_reads_the_distribution_metadata(monkeypatch):
    def fake_version(name: str) -> str:
        assert name == "oara-prometheus"
        return "9.8.7"

    monkeypatch.setattr(version_mod.metadata, "version", fake_version)
    assert version_mod.package_version() == "9.8.7"


def test_package_version_falls_back_to_init_without_metadata(monkeypatch):
    """A source checkout, or deploy.sh's venv: no installed distribution."""

    def missing(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(version_mod.metadata, "version", missing)
    assert version_mod.package_version() == prometheus.__version__


def test_root_and_openapi_report_the_package_version(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from prometheus.web.server import create_app

    monkeypatch.delenv("PROMETHEUS_API_TOKEN", raising=False)
    app = create_app({})
    expected = version_mod.package_version()
    assert expected != STALE

    assert app.version == expected
    body = TestClient(app).get("/").json()
    assert body["version"] == expected


@pytest.mark.asyncio
async def test_websocket_connected_frame_reports_the_package_version():
    websockets = pytest.importorskip("websockets")
    from prometheus.web.ws_server import WebSocketBridge

    bridge = WebSocketBridge(api_token=None)
    await bridge.start(host="127.0.0.1", port=0)
    port = bridge._server.sockets[0].getsockname()[1]
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            frame = json.loads(await asyncio.wait_for(ws.recv(), timeout=6.0))
    finally:
        await bridge.stop()

    assert frame["type"] == "connected"
    # Same shape the clients parse: a string at payload.version.
    assert frame["payload"] == {"version": version_mod.package_version()}
    assert frame["payload"]["version"] != STALE
