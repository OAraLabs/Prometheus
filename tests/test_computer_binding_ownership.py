"""A device reaches the computer binding of its own sessions only.

``GET``/``PUT``/``DELETE /api/sessions/{session_id}/computer`` read, set and end
the app a person picked for a session's desktop tasks. A device an operator
marked for computer use passes the route allowlist for them
(``MARKED_DEVICE_ALLOWED``), so what keeps it out of SOMEONE ELSE's session is
the router-level ownership guard every ``{session_id}`` route gets
(``_guard_session_path``, docs/contracts/device-scoping.md section 3). These pin
that for each verb: the same 404 an unknown session gets, and the runner is
never reached, so nothing is read, bound, unbound or stopped.

A review once listed these routes as unguarded (2026-10-11); they were not, and
this file is the proof that stays.
"""

from __future__ import annotations

import secrets
from typing import Any

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from prometheus.computer.door import PersonCheck  # noqa: E402
from prometheus.config.device_store import DeviceStore  # noqa: E402
from prometheus.web.server import create_app  # noqa: E402

GLOBAL = "binding-owner-global-" + secrets.token_hex(8)
ALICE_SESSION = "web:alice-1"
BOB_SESSION = "web:bob-1"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMETHEUS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PROMETHEUS_CONFIG_DIR", str(tmp_path / "config"))


class _Binding:
    def __init__(self, session_id: str, app: str) -> None:
        self.session_id, self.app = session_id, app

    def as_dict(self) -> dict[str, Any]:
        return {"state": "on", "session_id": self.session_id, "app": self.app}


class _Runner:
    """The routes' runner, recording every call. A real PersonCheck: a marked device is a person."""

    def __init__(self, devices: DeviceStore) -> None:
        self.people = PersonCheck(device_store=devices)
        self.calls: list[tuple[str, str]] = []

    def binding_for(self, session_id: str):
        self.calls.append(("binding_for", session_id))
        return None

    async def bind(self, session_id: str, app_name: str, **_kw):
        self.calls.append(("bind", session_id))
        return _Binding(session_id, app_name)

    def unbind(self, session_id: str) -> None:
        self.calls.append(("unbind", session_id))

    def stop_session(self, session_id: str) -> None:
        self.calls.append(("stop_session", session_id))


class Daemon:
    def __init__(self, tmp_path) -> None:
        self.devices = DeviceStore(tmp_path / "devices.db")
        self.app = create_app({"web": {"api_token": GLOBAL}}, device_store=self.devices)
        self.runner = _Runner(self.devices)
        self.app.state.computer_runner = self.runner
        self.alice = self.devices.mint("alice's phone", "ios")
        self.bob = self.devices.mint("bob's phone", "ios")
        assert self.devices.set_computer(self.alice["id"], True, by="test")
        self.devices.claim_session(ALICE_SESSION, self.alice["id"])
        self.devices.claim_session(BOB_SESSION, self.bob["id"])
        self.client = TestClient(self.app, client=("127.0.0.1", 50123))

    def as_alice(self, method: str, session_id: str):
        return self.client.request(
            method, f"/api/sessions/{session_id}/computer",
            headers={"Authorization": f"Bearer {self.alice['token']}"},
            json={"app": "gedit"} if method == "PUT" else None)


@pytest.fixture
def d(tmp_path) -> Daemon:
    return Daemon(tmp_path)


def _unknown_session(response) -> bool:
    return response.status_code == 404 and response.json() == {"error": "unknown session"}


def test_a_marked_device_cannot_read_another_devices_binding(d):
    assert _unknown_session(d.as_alice("GET", BOB_SESSION))
    assert d.runner.calls == []


def test_a_marked_device_cannot_bind_another_devices_session(d):
    assert _unknown_session(d.as_alice("PUT", BOB_SESSION))
    assert d.runner.calls == []
    assert d.devices.session_owner(BOB_SESSION) == d.bob["id"], "the PUT claimed bob's session"


def test_a_marked_device_cannot_unbind_another_devices_session(d):
    assert _unknown_session(d.as_alice("DELETE", BOB_SESSION))
    assert d.runner.calls == [], "bob's binding was ended or his task stopped"


def test_the_refusal_is_the_answer_an_unknown_session_gets(d):
    """No oracle: someone else's session reads exactly like one that does not exist."""
    theirs = d.as_alice("GET", BOB_SESSION)
    nobodys = d.as_alice("GET", "web:no-such-session")
    assert (theirs.status_code, theirs.json()) == (nobodys.status_code, nobodys.json())


@pytest.mark.parametrize("method", ["GET", "PUT", "DELETE"])
def test_the_same_device_reaches_its_own_sessions_binding(d, method):
    """The control: the 404s above are the ownership guard, not the route refusing everyone."""
    r = d.as_alice(method, ALICE_SESSION)
    assert r.status_code == 200, r.text
    assert d.runner.calls and all(sid == ALICE_SESSION for _, sid in d.runner.calls)
