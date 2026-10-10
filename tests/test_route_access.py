"""``web/route_access.py`` — which routes a SCOPED device may use, and the guard that keeps the table honest.

Enforcement is default-deny and lives in the bearer middleware: a scoped token on a route that is not in
``SCOPED_ALLOWED`` is a ``403 operator_only``, whether or not anyone remembered to classify it. That makes a
forgotten route safe. It does not make it *visible*, so this file adds the ratchet: every route the real app
registers is classified exactly once, as public (``web/public_routes.py``), scoped-allowed, or operator-only,
and a route that is none of those fails here until someone says which. The same goes for an entry that matches
no route and for a route in two classes.

``SCOPED_ALLOWED`` is also written out below, in full. Widening what a phone you approved can do is a security
decision; it should show up as a changed line in a test, not as a quiet addition to a set.
"""

from __future__ import annotations

import re
import secrets

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.config.device_store import DeviceStore  # noqa: E402
from prometheus.web import route_access  # noqa: E402
from prometheus.web.public_routes import PUBLIC_ROUTES  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402

GLOBAL = "route-access-global-" + secrets.token_hex(8)

#: Pinned here, not just defined there: this is the whole of what an approved device may do.
EXPECTED_SCOPED_ALLOWED = {
    # its own device
    ("GET", "/api/devices"),
    ("DELETE", "/api/devices/{device_id}"),
    ("PUT", "/api/devices/{device_id}/push"),
    ("DELETE", "/api/devices/{device_id}/push"),
    ("POST", "/api/devices/{device_id}/activity"),
    ("DELETE", "/api/devices/{device_id}/activity"),
    # chat
    ("POST", "/api/chat"),
    ("POST", "/api/chat/send"),
    ("POST", "/api/chat/interrupt"),
    # its own sessions
    ("GET", "/api/sessions"),
    ("POST", "/api/sessions"),
    ("DELETE", "/api/sessions/{session_id}"),
    ("GET", "/api/sessions/{session_id}/messages"),
    ("PUT", "/api/sessions/{session_id}/title"),
    ("PUT", "/api/sessions/{session_id}/pin"),
    ("GET", "/api/sessions/{session_id}/fork"),
    ("POST", "/api/sessions/{session_id}/fork"),
    ("GET", "/api/events/recent"),
    ("GET", "/api/activity/recent"),
    ("POST", "/api/search"),
    # the one desktop route open to every valid token: a stop can only end a task
    ("POST", "/api/computer/tasks/{task_id}/stop"),
    # the tool calls of its own sessions: once or until-restart, enforced in the handlers
    ("GET", "/api/approvals"),
    ("POST", "/api/approvals/{request_id}/approve"),
    ("POST", "/api/approvals/{request_id}/deny"),
}


#: What a device an operator MARKED for computer use may additionally do (the door's W3 ruling).
EXPECTED_MARKED_DEVICE_ALLOWED = {
    ("GET", "/api/computer/apps"),
    ("POST", "/api/computer/tasks"),
    ("GET", "/api/computer/tasks/{task_id}"),
    ("GET", "/api/sessions/{session_id}/computer"),
    ("PUT", "/api/sessions/{session_id}/computer"),
    ("DELETE", "/api/sessions/{session_id}/computer"),
    ("PUT", "/api/devices/{device_id}/computer"),
}


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMETHEUS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path / "config"))


def _route_pairs(app) -> set[tuple[str, str]]:
    pairs = set()
    for route in app.routes:
        path, methods = getattr(route, "path", None), getattr(route, "methods", None)
        if path is None or not methods:
            continue
        for method in methods:
            if method not in ("HEAD", "OPTIONS"):
                pairs.add((method, path))
    return pairs


# ── the matcher ──────────────────────────────────────────────────────────────

def test_an_allowed_route_is_matched_by_method_and_whole_path():
    assert route_access.is_scoped_allowed("POST", "/api/chat/send") is True
    assert route_access.is_scoped_allowed("GET", "/api/chat/send") is False
    assert route_access.is_scoped_allowed("post", "/api/chat/send") is True
    for neighbour in ("/api/chat/send/x", "/api/chat/sends", "/api/chat", "//api/chat/send", "/x/api/chat/send"):
        assert route_access.is_scoped_allowed("POST", neighbour) is (neighbour == "/api/chat"), neighbour


def test_a_parameter_is_exactly_one_segment():
    assert route_access.is_scoped_allowed("PUT", "/api/sessions/abc/title") is True
    for no in ("/api/sessions//title", "/api/sessions/a/b/title", "/api/sessions/title"):
        assert route_access.is_scoped_allowed("PUT", no) is False, no


def test_head_is_judged_as_the_get_it_would_run():
    assert route_access.is_scoped_allowed("HEAD", "/api/sessions") is True
    assert route_access.is_scoped_allowed("HEAD", "/api/cron") is False


def test_one_trailing_slash_is_the_route_it_redirects_to_and_no_more():
    assert route_access.is_scoped_allowed("GET", "/api/sessions/") is True
    assert route_access.is_scoped_allowed("GET", "/api/sessions//") is False
    assert route_access.is_scoped_allowed("GET", "/api/cron/") is False


def test_nothing_unknown_is_allowed():
    for method, path in (("GET", ""), ("", "/api/sessions"), ("GET", "/"), ("GET", "/api"), ("TRACE", "/api/sessions")):
        assert route_access.is_scoped_allowed(method, path) is False, (method, path)


def test_the_allowlist_is_exactly_what_this_file_says():
    assert route_access.SCOPED_ALLOWED == EXPECTED_SCOPED_ALLOWED
    assert route_access.MARKED_DEVICE_ALLOWED == EXPECTED_MARKED_DEVICE_ALLOWED


# ── the marked tier ──────────────────────────────────────────────────────────

def test_a_marked_route_needs_the_mark_and_an_unmarked_device_does_not_have_it():
    for method, pattern in route_access.MARKED_DEVICE_ALLOWED:
        path = _filled(pattern)
        assert route_access.is_scoped_allowed(method, path) is False, (method, pattern)
        assert route_access.is_scoped_allowed(method, path, marked=lambda: False) is False, (method, pattern)
        assert route_access.is_scoped_allowed(method, path, marked=lambda: True) is True, (method, pattern)


def test_the_mark_is_asked_only_for_a_marked_route():
    asked = []

    def marked() -> bool:
        asked.append(1)
        return True

    assert route_access.is_scoped_allowed("GET", "/api/sessions", marked=marked) is True      # allowed outright
    assert route_access.is_scoped_allowed("GET", "/api/cron", marked=marked) is False         # operator-only
    assert asked == [], "no registry lookup for a route the mark cannot change"
    assert route_access.is_scoped_allowed("POST", "/api/computer/tasks", marked=marked) is True
    assert asked == [1]


def test_a_mark_that_cannot_be_read_is_no_mark():
    def broken() -> bool:
        raise RuntimeError("the registry is unavailable")

    assert route_access.is_scoped_allowed("POST", "/api/computer/tasks", marked=broken) is False


def test_the_mark_does_not_widen_anything_outside_its_list():
    assert route_access.is_scoped_allowed("POST", "/api/cron", marked=lambda: True) is False
    assert route_access.is_scoped_allowed("PUT", "/api/providers/keys/x", marked=lambda: True) is False
    assert route_access.is_scoped_allowed("GET", "/api/approvals/grants", marked=lambda: True) is False


def test_the_tables_are_frozen_and_exact():
    for table in (route_access.SCOPED_ALLOWED, route_access.MARKED_DEVICE_ALLOWED, route_access.OPERATOR_ONLY,
                  route_access.OUTSIDE_THE_GATE):
        assert isinstance(table, frozenset)
        for method, pattern in table:
            assert method == method.upper() and method.isalpha(), method
            assert not pattern.endswith("/") or pattern == "/", pattern
            assert "*" not in pattern and "?" not in pattern, pattern


# ── the real route table ─────────────────────────────────────────────────────

def test_every_registered_route_is_classified_exactly_once():
    registered = _route_pairs(create_app({"web": {"api_token": "t"}}))
    classes = {
        "public": set(PUBLIC_ROUTES) | set(route_access.OUTSIDE_THE_GATE),
        "scoped-allowed": set(route_access.SCOPED_ALLOWED),
        "marked-device": set(route_access.MARKED_DEVICE_ALLOWED),
        "operator-only": set(route_access.OPERATOR_ONLY),
    }
    everything = set().union(*classes.values())
    assert registered - everything == set(), (
        "routes with no classification (a scoped device is denied them either way; say so on purpose by adding "
        f"each to OPERATOR_ONLY, or to SCOPED_ALLOWED if a phone really needs it): {sorted(registered - everything)}")
    assert everything - registered == set(), f"classified but not registered: {sorted(everything - registered)}"
    names = list(classes)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert classes[a] & classes[b] == set(), f"in both {a} and {b}: {sorted(classes[a] & classes[b])}"


def test_the_gate_s_own_exemptions_are_the_two_named_ones():
    assert route_access.OUTSIDE_THE_GATE == frozenset({("GET", "/"), ("GET", "/health")})


# ── the table, through the real middleware ───────────────────────────────────

def _filled(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "x", path)


@pytest.fixture
def world(tmp_path):
    devices = DeviceStore(tmp_path / "devices.db")
    app = create_app({"web": {"api_token": GLOBAL}}, device_store=devices)
    scoped = devices.mint("a scoped phone", "ios")
    return TestClient(app, client=("127.0.0.1", 50123)), {"Authorization": f"Bearer {scoped['token']}"}


@pytest.fixture
def marked_world(tmp_path):
    """The same, with an operator's mark on the device: it is a person (the computer-use door)."""
    devices = DeviceStore(tmp_path / "devices.db")
    app = create_app({"web": {"api_token": GLOBAL}}, device_store=devices)
    person = devices.mint("a marked phone", "ios")
    assert devices.set_computer(person["id"], True, by="test")
    return TestClient(app, client=("127.0.0.1", 50123)), {"Authorization": f"Bearer {person['token']}"}


@pytest.mark.parametrize("method, pattern", sorted(route_access.OPERATOR_ONLY))
def test_a_scoped_device_is_denied_every_operator_only_route(world, method, pattern):
    client, headers = world
    response = client.request(method, _filled(pattern), headers=headers, json={} if method != "GET" else None)
    assert response.status_code == 403 and response.json().get("error") == "operator_only", (
        method, pattern, response.status_code, response.text[:100])


@pytest.mark.parametrize("method, pattern", sorted(route_access.SCOPED_ALLOWED))
def test_a_scoped_device_gets_past_the_gate_on_every_allowed_route(world, method, pattern):
    client, headers = world
    response = client.request(method, _filled(pattern), headers=headers, json={} if method != "GET" else None)
    denied = response.status_code == 403 and response.json().get("error") == "operator_only"
    assert not denied, (method, pattern, response.status_code, response.text[:100])


@pytest.mark.parametrize("method, pattern", sorted(route_access.MARKED_DEVICE_ALLOWED))
def test_an_unmarked_scoped_device_is_denied_every_marked_route(world, method, pattern):
    client, headers = world
    response = client.request(method, _filled(pattern), headers=headers, json={} if method != "GET" else None)
    assert response.status_code == 403 and response.json().get("error") == "operator_only", (
        method, pattern, response.status_code, response.text[:100])


@pytest.mark.parametrize("method, pattern", sorted(route_access.MARKED_DEVICE_ALLOWED))
def test_a_marked_device_gets_past_the_gate_on_every_marked_route(marked_world, method, pattern):
    client, headers = marked_world
    response = client.request(method, _filled(pattern), headers=headers, json={} if method != "GET" else None)
    denied = response.status_code == 403 and response.json().get("error") == "operator_only"
    assert not denied, (method, pattern, response.status_code, response.text[:100])


def test_a_marked_device_is_still_denied_everything_the_mark_does_not_cover(marked_world):
    client, headers = marked_world
    for method, path in (("POST", "/api/cron"), ("PUT", "/api/providers/keys/x"), ("GET", "/api/approvals/grants"),
                         ("GET", "/api/config"), ("POST", "/api/devices")):
        response = client.request(method, path, headers=headers, json={})
        assert response.status_code == 403 and response.json().get("error") == "operator_only", (method, path)
