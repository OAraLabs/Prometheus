"""A dropped WebSocket frame must leave a record.

`broadcast()` swallowed send failures with a bare `except` and NO log at any
level, then silently discarded the client. So the daemon could drop every
frame of a turn — an operator's entire view of a live turn — and leave nothing
behind. That is one level worse than the `agent_progress` emitter, which at
least logs at debug, and debug is precisely why 42 client disconnects produced
zero visible evidence.

WARNING and counted. The counters answer "is this daemon losing frames?"
without attaching a WebSocket client, which was the only way anything about
this seam was established.

⚠ These tests cover the case where `send` RAISES. The half-open case — a peer
that called `terminate()`, whose socket still accepts writes — raises nothing
and is invisible here by construction. The library's keepalive
(ping_interval=20, ping_timeout=20) fails the connection within ~20-40s, after
which sends raise and are counted; the blind window is bounded, not closed.
`test_a_successful_send_is_silent` is the guard that keeps this suite honest
about that: it asserts we do NOT invent a drop we cannot see.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from prometheus.web.ws_server import WebSocketBridge, _client_label

pytestmark = pytest.mark.integration


class _Client:
    """A stand-in socket. `fail` decides whether send raises."""

    def __init__(self, fail: bool = False, label: str = "1.2.3.4:5678"):
        self.fail = fail
        self.sent: list[str] = []
        host, port = label.split(":")
        self.remote_address = (host, int(port))

    async def send(self, raw: str) -> None:
        if self.fail:
            raise ConnectionResetError("peer went away")
        self.sent.append(raw)


def _bridge(*clients) -> WebSocketBridge:
    b = WebSocketBridge(None, None, None)
    for c in clients:
        b._clients.add(c)
    return b


def _broadcast(b) -> None:
    asyncio.run(b.broadcast({"type": "agent_progress", "payload": {"phase": "thinking"}}))


class TestAFailedSendIsLoudAndCounted:
    def test_it_logs_at_WARNING_not_debug(self, caplog):
        b = _bridge(_Client(fail=True))
        with caplog.at_level(logging.WARNING, logger="prometheus.web.ws_server"):
            _broadcast(b)
        assert "DROPPED" in caplog.text
        recs = [r for r in caplog.records if "DROPPED" in r.getMessage()]
        assert recs and recs[0].levelno == logging.WARNING, (
            "debug is what made 42 disconnects invisible; this must be WARNING"
        )

    def test_the_log_names_the_client_and_the_frame_type(self, caplog):
        b = _bridge(_Client(fail=True, label="10.0.0.9:4242"))
        with caplog.at_level(logging.WARNING, logger="prometheus.web.ws_server"):
            _broadcast(b)
        assert "10.0.0.9:4242" in caplog.text
        assert "agent_progress" in caplog.text

    def test_the_discard_is_logged_separately(self, caplog):
        b = _bridge(_Client(fail=True))
        with caplog.at_level(logging.WARNING, logger="prometheus.web.ws_server"):
            _broadcast(b)
        assert "DISCARDED" in caplog.text

    def test_both_counters_increment(self):
        b = _bridge(_Client(fail=True))
        _broadcast(b)
        s = b.delivery_stats()
        assert s["frames_dropped"] == 1
        assert s["clients_discarded"] == 1
        assert s["clients"] == 0, "the dead client must be removed"

    def test_counters_are_monotonic_across_broadcasts(self):
        b = _bridge(_Client(fail=True), _Client(fail=True))
        _broadcast(b)
        first = b.delivery_stats()["frames_dropped"]
        assert first == 2
        b._clients.add(_Client(fail=True))
        _broadcast(b)
        assert b.delivery_stats()["frames_dropped"] == 3, "counter reset or missed"

    def test_a_healthy_client_still_receives_when_a_sibling_fails(self):
        good, bad = _Client(), _Client(fail=True)
        b = _bridge(good, bad)
        _broadcast(b)
        assert len(good.sent) == 1, "one dead client must not starve the others"
        assert b.delivery_stats()["frames_dropped"] == 1


class TestASuccessfulSendIsSilent:
    """The other direction. A drop we cannot see must not be invented."""

    def test_no_warning_when_every_send_succeeds(self, caplog):
        b = _bridge(_Client(), _Client())
        with caplog.at_level(logging.WARNING, logger="prometheus.web.ws_server"):
            _broadcast(b)
        assert "DROPPED" not in caplog.text
        assert "DISCARDED" not in caplog.text

    def test_counters_stay_zero(self):
        b = _bridge(_Client(), _Client())
        _broadcast(b)
        assert b.delivery_stats() == {
            "clients": 2, "frames_dropped": 0, "clients_discarded": 0}

    def test_no_clients_is_not_a_drop(self):
        b = _bridge()
        _broadcast(b)
        assert b.delivery_stats()["frames_dropped"] == 0


class TestTheLabelHelperNeverRaises:
    def test_a_socket_with_no_remote_address(self):
        assert _client_label(object()).startswith("id=")

    def test_a_socket_whose_attribute_explodes(self):
        class Hostile:
            @property
            def remote_address(self):
                raise RuntimeError("nope")
        assert _client_label(Hostile()).startswith("id="), (
            "the label runs inside an error path; it must not become the "
            "second failure"
        )
