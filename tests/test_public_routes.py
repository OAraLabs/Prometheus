"""``web/public_routes.py`` — the one exact list of routes that answer without a bearer token.

The bearer middleware gates every path under ``/api/`` and ``/v1/`` (its own comment says: "widen here,
never elsewhere"). A route that must answer without a token is an exception to that, and an exception
written as a PREFIX is how a neighbouring path ends up open by accident. So the list is exact: a method and a
path pattern, matched whole, where ``{name}`` stands for exactly one path segment and nothing else.

Two things are pinned:

* the matcher, so ``/api/pair/requests/{id}`` can never admit ``/api/pair/requests/{id}/approve``;
* the whole route table of the real app, so a new route that answers without a token, or a list entry that
  matches no route, fails here and has to be argued for in review. Routes outside the middleware's prefixes
  (the dashboard page and the health probe) are listed by name, because their being open is a decision too.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from prometheus.web.public_routes import PUBLIC_ROUTES, is_public_route  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402

# Open by design today, and not under the bearer prefixes at all.
OUTSIDE_THE_PREFIXES = {("GET", "/"), ("GET", "/health")}


# ── the matcher ──────────────────────────────────────────────────────────────

def test_the_installers_route_is_public_and_only_as_a_post():
    assert is_public_route("POST", "/api/pair/local") is True
    assert is_public_route("GET", "/api/pair/local") is False
    assert is_public_route("DELETE", "/api/pair/local") is False


def test_hello_is_public_and_only_as_a_get():
    assert ("GET", "/api/hello") in PUBLIC_ROUTES
    assert is_public_route("GET", "/api/hello") is True
    for method in ("POST", "PUT", "PATCH", "DELETE"):
        assert is_public_route(method, "/api/hello") is False, method
    for neighbour in ("/api/hello/", "/api/hello/x", "/api/hellos", "/api/hello?x=1", "/api/hello%2Fx"):
        assert is_public_route("GET", neighbour) is False, neighbour


def test_a_method_is_matched_whole_and_case_insensitively():
    assert is_public_route("post", "/api/pair/local") is True
    assert is_public_route("POSTX", "/api/pair/local") is False
    assert is_public_route("", "/api/pair/local") is False


def test_a_path_is_matched_whole_never_as_a_prefix():
    for neighbour in ("/api/pair/local/", "/api/pair/local/x", "/api/pair/locals", "/api/pair", "/api/pair/",
                      "/api/pair/local?x=1", "//api/pair/local", "/api/pair/local%2Fx", "/x/api/pair/local"):
        assert is_public_route("POST", neighbour) is False, neighbour


def test_a_parameter_is_exactly_one_segment(monkeypatch):
    from prometheus.web import public_routes

    monkeypatch.setattr(public_routes, "PUBLIC_ROUTES", frozenset({("GET", "/api/things/{thing_id}")}))
    assert public_routes.is_public_route("GET", "/api/things/abc") is True
    for no in ("/api/things/", "/api/things", "/api/things/abc/approve", "/api/things/a/b", "/api/things//"):
        assert public_routes.is_public_route("GET", no) is False, no
    assert public_routes.is_public_route("POST", "/api/things/abc") is False


def test_every_entry_is_exact_and_under_api():
    assert ("POST", "/api/pair/local") in PUBLIC_ROUTES
    for method, pattern in PUBLIC_ROUTES:
        assert method == method.upper() and method.isalpha(), method
        assert pattern.startswith("/api/") and not pattern.endswith("/"), pattern
        assert "*" not in pattern and "?" not in pattern, f"{pattern}: patterns are exact, not wildcards"


def test_it_is_a_frozenset_so_nothing_edits_it_at_runtime():
    assert isinstance(PUBLIC_ROUTES, frozenset)


# ── the real route table ─────────────────────────────────────────────────────

def _route_pairs(app):
    pairs = set()
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if path is None or not methods:
            continue
        for method in methods:
            if method not in ("HEAD", "OPTIONS"):
                pairs.add((method, path))
    return pairs


def _matches(pattern: str, path: str) -> bool:
    a, b = pattern.split("/"), path.split("/")
    return len(a) == len(b) and all(x == y or (x.startswith("{") and x.endswith("}")) for x, y in zip(a, b))


def test_only_the_listed_routes_answer_without_a_bearer():
    """Every route is either under the bearer prefixes and not public, or it is named here on purpose."""
    app = create_app({"web": {"api_token": "t"}})
    open_routes = set()
    for method, path in _route_pairs(app):
        gated = path.startswith(("/api/", "/v1/"))
        public = any(method == m and _matches(p, path) for m, p in PUBLIC_ROUTES)
        if not gated or public:
            open_routes.add((method, path))
    expected = OUTSIDE_THE_PREFIXES | {(m, p) for m, p in PUBLIC_ROUTES}
    assert open_routes == expected, (
        f"routes that answer without a token: {sorted(open_routes - expected)}; "
        f"listed but not registered: {sorted(expected - open_routes)}"
    )


def test_every_public_entry_is_a_real_route():
    """A stale entry is an exemption nobody remembers granting."""
    app = create_app({"web": {"api_token": "t"}})
    registered = _route_pairs(app)
    for method, pattern in PUBLIC_ROUTES:
        assert any(m == method and p == pattern for m, p in registered), f"{method} {pattern} matches no route"


def test_the_middleware_and_the_list_agree_on_what_is_open(monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.delenv("PROMETHEUS_LOCAL_PAIRING_DIR", raising=False)
    monkeypatch.delenv("PROMETHEUS_INSTALL_KIND", raising=False)
    client = TestClient(create_app({"web": {"api_token": "t"}}))
    gate = client.get("/api/status")
    assert gate.status_code == 401 and "Bearer" in gate.json()["error"]
    assert client.get("/health").status_code != 401
    # A public route gets past the token gate and answers for itself: here "this install has no such
    # feature" (404), which is not the gate's 401 and does not carry the gate's wording.
    reached = client.post("/api/pair/local", json={"code": "x" * 43})
    assert reached.status_code == 404
    assert "Bearer" not in reached.text
    # Hello is the other public route: it answers with no token, and its neighbours are still gated.
    assert client.get("/api/hello").status_code == 200
    assert client.get("/api/hello/extra").status_code == 401
