"""The bearer gate decides on the ROUTED path, never on ``request.url``.

WHY THIS EXISTS
---------------
The middleware in web/server.py enforced the token only
``if request.url.path.startswith(("/api/", "/v1/"))``. ``request.url`` is not
the path the router dispatches on: Starlette rebuilds it from the client's
``Host`` header. Before starlette 1.0.1 (CVE-2026-48710) a Host such as
``x/abc?`` moved the path boundary in that rebuild — ``request.url.path`` read
``/abc`` while the router still ran ``/api/status`` — so the gate skipped the
token check for every ``/api`` and ``/v1`` route. Reproduced against the
deployed daemon's code on starlette 0.52.1: no token, crafted Host, 200.

0.9.2's starlette floor (>=1.3.1) fixes the rebuild. This file pins the other
half: the gate must not depend on it. The body-size cap had the same shape
(the recorder upload's larger cap was granted by ``request.url.path``).

HOW THE BYPASS IS REPLAYED ON A PATCHED STARLETTE
-------------------------------------------------
On starlette >=1.0.1 a crafted Host no longer poisons ``request.url``, so the
literal crafted-Host test below passes whatever the middleware reads — it
documents the contract but cannot catch a regression on its own. The replay
tests make ``request.url`` lie exactly the way 0.52.1 did (the path it reports
differs from the routed path) and assert the gate still refuses. Those are the
tests that fail against the old middleware.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402
from starlette.datastructures import URL  # noqa: E402
from starlette.requests import HTTPConnection  # noqa: E402

from prometheus.web.server import create_app  # noqa: E402

TOKEN = "routed-path-test-token"
CRAFTED_HOST = "x/abc?"


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app({"web": {"api_token": TOKEN}}))


@pytest.fixture
def lying_url(monkeypatch):
    """Make ``request.url`` report a different path than the one routed —
    what starlette <1.0.1 produced for ``Host: x/abc?``."""
    def _set(reported_path: str) -> None:
        monkeypatch.setattr(
            HTTPConnection, "url",
            property(lambda self: URL(f"http://x{reported_path}")),
        )
    return _set


@pytest.mark.parametrize("path", ["/api/status", "/v1/models"])
def test_a_crafted_host_header_gets_401(client, path):
    assert client.get(path, headers={"Host": CRAFTED_HOST}).status_code == 401


def test_the_token_still_works(client):
    """Guard the guard: a gate that refused everything would pass the tests
    above and below."""
    assert client.get("/api/status", headers={
        "Authorization": f"Bearer {TOKEN}"}).status_code == 200


@pytest.mark.parametrize("path", ["/api/status", "/v1/models"])
def test_the_gate_ignores_a_request_url_that_lies(client, lying_url, path):
    lying_url("/abc")
    assert client.get(path).status_code == 401, (
        "the bearer gate followed request.url instead of the routed path — "
        "CVE-2026-48710's bypass is open again")


def test_the_body_cap_keys_on_the_routed_path(client, lying_url):
    """A request whose request.url claims the recorder upload path must not
    get that route's 32 MiB cap: 3 MiB to /api/status is over the 2 MiB one."""
    lying_url("/api/learning/live-upload")
    resp = client.post("/api/status", content=b"x" * (3 * 1024 * 1024),
                       headers={"Content-Type": "application/octet-stream"})
    assert resp.status_code == 413


async def _status_of(app, path: str, root_path: str, headers: list) -> int:
    """One request through the raw ASGI app, with the scope a server behind a
    proxy prefix builds: ``path`` INCLUDES ``root_path`` (TestClient's
    root_path does not prefix, so it cannot model this)."""
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "GET", "scheme": "http", "path": path,
        "raw_path": path.encode(), "root_path": root_path, "query_string": b"",
        "headers": [(b"host", b"testserver"), *headers],
        "client": ("127.0.0.1", 50000), "server": ("testserver", 80),
    }
    sent: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    return next(m["status"] for m in sent if m["type"] == "http.response.start")


async def test_the_gate_holds_under_a_root_path():
    """Behind a proxy prefix the router strips root_path before matching; the
    gate must see that same path, or `/prom/api/...` would skip the token."""
    app = create_app({"web": {"api_token": TOKEN}})
    bearer = [(b"authorization", f"Bearer {TOKEN}".encode())]
    # the prefixed path routes — so a 401 below is the gate, not a 404
    assert await _status_of(app, "/prom/api/status", "/prom", bearer) == 200
    assert await _status_of(app, "/prom/api/status", "/prom", []) == 401
