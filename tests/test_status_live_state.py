"""GRAFT-MOBILE-BRIDGE 6 — `/api/status.state` reflects the LIVE turn state.

The field was permanently "idle": the WS bridge drives
``app.state.agent_state_ref["state"]`` to thinking/idle per turn, but the route
read ``app.state.agent_state`` — a different object, a string set once at boot
and never updated. A client (Beacon's Status screen) could not render it and was
told not to. These pin the route onto the live ref, with a safe fallback when no
bridge is wired.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.web.server import create_app  # noqa: E402


def _client():
    app = create_app({})
    return TestClient(app), app


def test_status_reports_the_live_ref_over_the_boot_string():
    client, app = _client()
    # The bridge drives this dict; simulate a turn in flight.
    app.state.agent_state_ref = {"state": "thinking"}
    assert client.get("/api/status").json()["state"] == "thinking"

    # And back to idle when the turn ends.
    app.state.agent_state_ref["state"] = "idle"
    assert client.get("/api/status").json()["state"] == "idle"


def test_status_falls_back_to_the_string_without_a_bridge():
    # The `web` entrypoint wires no bridge, so there is no ref — the route must
    # still answer, from the boot-time string, rather than error.
    client, app = _client()
    if hasattr(app.state, "agent_state_ref"):
        delattr(app.state, "agent_state_ref")
    app.state.agent_state = "idle"
    assert client.get("/api/status").json()["state"] == "idle"


def test_a_malformed_ref_does_not_break_the_route():
    # Defensive: a ref that is not a dict (never expected) falls back rather than
    # raising inside the status route.
    client, app = _client()
    app.state.agent_state_ref = None
    app.state.agent_state = "idle"
    assert client.get("/api/status").json()["state"] == "idle"
