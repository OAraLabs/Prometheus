"""Unknown query parameters are refused, not ignored.

The bug this closes is not a crash — it is a RIGHT-LOOKING ANSWER. `GET /api/usage?window=7d`
returned the all-time rollup, identical to `?window=30d` and `?window=all`, because Starlette
drops a parameter no route declares. Three identical responses read as a broken window feature
rather than as three malformed requests; the real parameter is `days`.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi import Depends, FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.web.server import create_app  # noqa: E402
from prometheus.web.strict_query import StrictQueryRoute, known_query_names  # noqa: E402


@pytest.fixture
def client():
    return TestClient(create_app({}))


# ── the regression, on the real route it happened to ────────────────────────────────────────

def test_the_mistyped_window_param_is_now_an_error(client):
    r = client.get("/api/usage?window=7d")
    assert r.status_code == 400, "?window=7d silently returned the ALL-TIME rollup before this"
    assert "window" in r.json()["unknown"]


def test_the_real_param_still_works(client):
    assert client.get("/api/usage?days=7").status_code == 200
    assert client.get("/api/usage").status_code == 200, "no params at all is still valid"


def test_the_error_says_what_IS_accepted(client):
    """"unknown parameter: window" alone leaves the caller to guess a second time."""
    body = client.get("/api/usage?window=7d").json()
    assert "days" in body["accepted"], "the reply names the parameter the caller wanted"
    assert "window" in body["error"], "and names the one it rejected"


def test_every_unknown_param_is_listed_not_just_the_first(client):
    body = client.get("/api/usage?window=7d&scope=all").json()
    assert body["unknown"] == ["scope", "window"], "one round trip should fix ALL of them"


# ── it must not become a new way to break working callers ───────────────────────────────────

def test_a_path_param_is_not_mistaken_for_an_unknown_query_param(client):
    """`/api/sessions/{id}/messages` takes session_id from the PATH — rejecting it would 400
    every history fetch Beacon makes."""
    r = client.get("/api/sessions/abc/messages?since=0")
    assert r.status_code != 400, f"path+known query rejected: {r.text[:200]}"


def test_a_repeated_KNOWN_key_is_still_accepted(client):
    """Replaces a test that could not fail. It asserted that a repeated unknown key is listed
    once — true, but Starlette's multidict collapses duplicate keys before we ever see them, so
    no change to this module could have made it red. Mutation testing caught it, not review.

    What is worth pinning is that repeating a VALID parameter is not turned into a rejection."""
    assert client.get("/api/usage?days=7&days=30").status_code != 400


@pytest.mark.parametrize(
    "path",
    [
        "/api/wiki/pages?path=",
        "/api/wiki/search?q=kling&limit=5",
        "/api/tools/recent?limit=5",
        "/api/events/recent?limit=25",
        "/api/files?path=",
        "/api/documents?path=",
    ],
)
def test_every_query_string_beacon_actually_sends_still_works(client, path):
    """Captured from Beacon's source. A 400 here means this guard broke the client."""
    assert client.get(path).status_code != 400, f"{path} was refused"


def test_docs_and_openapi_are_untouched():
    """FastAPI registers those during construction, before the route class is swapped in."""
    app = create_app({})
    strict = [r for r in app.routes if isinstance(r, StrictQueryRoute)]
    assert len(strict) > 50, f"the route class did not take effect (got {len(strict)})"
    assert all(not r.path.startswith("/openapi") for r in strict)


# ── the helper: a name reachable only through Depends is still accepted ──────────────────────

def test_a_sub_dependency_query_param_is_accepted():
    """Rejecting a name declared by a Depends(...) would break a working caller, and the top-level
    dependant does not list it — the walk has to recurse."""
    app = FastAPI()
    app.router.route_class = StrictQueryRoute

    def paging(offset: int = 0):
        return offset

    @app.get("/thing")
    async def thing(name: str = "", page=Depends(paging)):
        return {"name": name, "page": page}

    c = TestClient(app)
    assert c.get("/thing?name=x").status_code == 200
    assert c.get("/thing?offset=5").status_code == 200, "a Depends-declared param must be accepted"
    assert c.get("/thing?nope=1").status_code == 400

    route = next(r for r in app.routes if getattr(r, "path", None) == "/thing")
    assert known_query_names(route.dependant) == {"name", "offset"}


def test_an_aliased_param_is_matched_by_its_WIRE_name():
    from fastapi import Query

    app = FastAPI()
    app.router.route_class = StrictQueryRoute

    @app.get("/aliased")
    async def aliased(internal: str = Query("", alias="on-the-wire")):
        return {"v": internal}

    c = TestClient(app)
    assert c.get("/aliased?on-the-wire=1").status_code == 200, "the alias is what callers send"
    assert c.get("/aliased?internal=1").status_code == 400, "the python name is NOT the wire name"
