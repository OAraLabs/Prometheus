"""``is_this_host``: did this request come from a process on the machine the daemon runs on?

``is_same_machine`` (tests/test_same_machine.py) answers that for a loopback peer. A daemon bound to ONE
non-loopback address (a tailnet or LAN address, how an owner narrows a headless box) has no loopback listener,
so a terminal on that same machine reaches it at that address, and the kernel makes that address the source:
the TCP peer IS the local address the connection arrived on. A remote host cannot complete a TCP handshake with
the daemon's own address as its source (the reply goes to the daemon's own stack), so "peer == the local address"
is "this host", the same way a loopback peer is.

Neither form believes a relay: a request that carries ``X-Forwarded-For`` or ``Forwarded`` was relayed by
something and is not this host, whatever its peer says. Host names (Starlette's ``testclient`` / ``testserver``)
and anything unparseable are not addresses and so never equal.

``oara pair --owner`` is the first user: minting an owner code is the person at the machine's act, from the
machine (tests/test_owner_code_routes.py).
"""

from __future__ import annotations

import pytest

from prometheus.web.loopback import is_this_host


def scope(peer: str | None, server: str | None = "192.0.2.5", *headers: tuple[str, str]) -> dict:
    return {"type": "http",
            "client": (peer, 50123) if peer is not None else None,
            "server": (server, 8005) if server is not None else None,
            "headers": [(k.lower().encode(), v.encode()) for k, v in headers]}


@pytest.mark.parametrize("peer", ["127.0.0.1", "::1", "::ffff:127.0.0.1", "127.0.0.2"])
def test_a_loopback_peer_is_this_host(peer):
    assert is_this_host(scope(peer)) is True


@pytest.mark.parametrize("address", ["192.0.2.50", "192.168.1.20", "2001:db8::5", "fe80::1%en0"])
def test_a_peer_at_the_very_address_the_connection_arrived_on_is_this_host(address):
    assert is_this_host(scope(address, address)) is True


def test_the_ipv4_mapped_spelling_of_the_same_address_is_the_same_address():
    assert is_this_host(scope("::ffff:192.168.1.20", "192.168.1.20")) is True
    assert is_this_host(scope("192.168.1.20", "::ffff:192.168.1.20")) is True


@pytest.mark.parametrize("peer, server", [
    ("192.0.2.77", "192.0.2.50"),               # another tailnet node
    ("192.168.1.21", "192.168.1.20"),           # the next machine on the Wi-Fi
    ("2001:db8::6", "2001:db8::5"),
])
def test_any_other_peer_is_not(peer, server):
    assert is_this_host(scope(peer, server)) is False


@pytest.mark.parametrize("peer, server", [
    ("testclient", "testserver"), ("testserver", "testserver"), ("", ""), (None, "192.0.2.5"),
    ("192.0.2.5", None), ("not an address", "not an address"),
])
def test_names_and_missing_slots_are_not_addresses(peer, server):
    assert is_this_host(scope(peer, server)) is False


@pytest.mark.parametrize("header", [("X-Forwarded-For", "203.0.113.9"), ("Forwarded", "for=127.0.0.1")])
@pytest.mark.parametrize("peer, server", [("127.0.0.1", "127.0.0.1"), ("192.168.1.20", "192.168.1.20")])
def test_a_relayed_request_is_not_this_host_whatever_its_peer_says(header, peer, server):
    assert is_this_host(scope(peer, server, header)) is False


def test_it_takes_a_request_as_well_as_a_scope():
    class Request:
        scope = {"type": "http", "client": ("10.0.0.4", 1), "server": ("10.0.0.4", 8005), "headers": []}

    assert is_this_host(Request()) is True
