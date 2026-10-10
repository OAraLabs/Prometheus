"""The one exact list of routes that answer without a bearer token.

``_check_bearer_token`` (web/server.py) gates every path under ``/api/`` and ``/v1/``, and its own comment
says to widen the exemption "here, never elsewhere". This is "here". A route that must answer without a
token is an exception to that rule, and an exception written as a PREFIX is how a neighbouring path ends up
open by accident (``/api/pair/requests`` opening ``/api/pair/requests/{id}/approve``).

So the list is exact: a method and a path pattern, matched whole. ``{name}`` stands for exactly one
non-empty path segment and nothing else. There is no wildcard and no trailing-slash tolerance.

Adding a route here is a review decision, not a side effect: ``tests/test_public_routes.py`` walks the
app's real route table and fails if any route outside this list (and the two named non-API ones) answers
without a token, and fails if an entry here matches no route.

Every public route must do its own checking, because the gate no longer does it. The same-Mac pairing
route requires a loopback peer AND a loopback Host header, refuses a request from a browser, and compares a
one-time secret in constant time.
"""

from __future__ import annotations

# (METHOD, "/exact/path/{param}"). Add the other pairing routes here as they land; do not turn any of these
# into a prefix.
PUBLIC_ROUTES: frozenset[tuple[str, str]] = frozenset({
    ("POST", "/api/pair/local"),   # same-Mac pairing with the one-time file secret (config/local_pairing.py)
    ("GET", "/api/hello"),         # "is there a Prometheus here": six fields, no credentials (web/hello.py)
})


def _matches(pattern: str, path: str) -> bool:
    expected = pattern.split("/")
    actual = path.split("/")
    if len(expected) != len(actual):
        return False
    for want, got in zip(expected, actual, strict=True):
        if want.startswith("{") and want.endswith("}"):
            if not got or "{" in got:
                return False
        elif want != got:
            return False
    return True


def is_public_route(method: str, path: str) -> bool:
    """Whether ``method path`` may be served without a bearer token.

    ``path`` is the ROUTED path (``scope["path"]``), never ``request.url.path``: the latter is rebuilt from
    the Host header, which is how CVE-2026-48710 once opened every /api route.
    """
    verb = (method or "").upper()
    return any(m == verb and _matches(pattern, path) for m, pattern in PUBLIC_ROUTES)
