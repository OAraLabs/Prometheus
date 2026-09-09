"""DNS rebinding: the connection goes where the guard looked.

THE DEFECT, demonstrated rather than argued. ``check_url`` resolved the name,
then httpx resolved it AGAIN at connect time, so a TTL-0 record could answer
public for the first lookup and loopback for the second. Against the code before
this change, with exactly that resolver::

    preflight allowed=True
    HTTP 200 body = b'LOCAL-SECRET-PROVES-CONNECT-TIME-REBIND'
    resolutions: ["'rebind.example'(str)->public", "'rebind.example'(str)->public",
                  "b'rebind.example'(bytes)->LOOPBACK"]

The detail that hides it from a casual patch is in that third entry: connect-time
resolution DOES go through ``socket.getaddrinfo``, but the host arrives as
**bytes**, so an interceptor comparing against a ``str`` never fires.

THE FIX is structural rather than a narrower race: the validated address is
written into the request, so no name survives for a second lookup to disagree
about. ``Host:`` and the TLS server name keep the original name, which is why
pinning costs neither virtual hosting nor certificate verification.
"""

from __future__ import annotations

import ipaddress
import pathlib
import socket

import httpx
import pytest

from prometheus.security import url_guard
from prometheus.security.url_guard import (
    SsrfBlocked,
    display_url,
    guard_request_hop,
    resolve_pinned,
)

# NOT the RFC 5737 documentation ranges. Measured: Python's ``is_private``
# covers 192.0.2.0/24, 198.51.100.0/24 and 203.0.113.0/24, so the addresses a
# test author reaches for first are exactly the ones this guard refuses — the
# test would then pass for the wrong reason, or fail confusingly.
PUBLIC_A = "93.184.216.34"
PUBLIC_B = "8.8.8.8"
MOCK = {"public.example": PUBLIC_A, "other.example": PUBLIC_B}


@pytest.fixture(autouse=True)
def _resolver(monkeypatch):
    """Mock hosts resolve to public addresses; everything else hits real DNS."""
    real = url_guard.socket.getaddrinfo
    calls: list[object] = []

    def fake(host, *args, **kwargs):
        calls.append(host)
        name = host.decode() if isinstance(host, bytes) else host
        if name in MOCK:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (MOCK[name], 0))]
        return real(host, *args, **kwargs)

    monkeypatch.setattr(url_guard.socket, "getaddrinfo", fake)
    return calls


def _record(seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({
            "url": str(request.url),
            "host_header": request.headers.get("Host"),
            "sni": request.extensions.get("sni_hostname"),
        })
        return httpx.Response(200, text="ok")
    return handler


async def _fetch(url: str, handler) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        follow_redirects=True,
        event_hooks={"request": [guard_request_hop]},
    ) as client:
        await client.get(url)


# --------------------------------------------------------------------------- #
# The pin
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_the_request_that_goes_out_names_the_validated_address():
    seen: list = []
    await _fetch("http://public.example/page", _record(seen))

    host = httpx.URL(seen[0]["url"]).host
    assert host == PUBLIC_A
    # No name survives, so there is nothing for a second lookup to disagree
    # about. That is the whole closure argument, asserted rather than described.
    ipaddress.ip_address(host)


@pytest.mark.asyncio
async def test_the_host_header_keeps_the_name_so_virtual_hosting_still_routes():
    seen: list = []
    await _fetch("http://public.example/page", _record(seen))
    assert seen[0]["host_header"] == "public.example"


def test_pinning_sets_the_host_header_even_when_there_is_none():
    """httpx happens to fill Host at construction, so the pin's own assignment
    is invisible in the normal path — this drives it directly, so the line
    cannot be deleted on the grounds that nothing noticed."""
    request = httpx.Request("GET", "http://public.example/x")
    del request.headers["Host"]
    target, reason = resolve_pinned("http://public.example/x")
    assert target is not None, reason

    url_guard.pin_request(request, target)

    assert request.headers["Host"] == "public.example"
    assert request.url.host == PUBLIC_A


@pytest.mark.asyncio
async def test_a_non_default_port_survives_in_the_host_header():
    seen: list = []
    await _fetch("http://public.example:8443/page", _record(seen))
    assert seen[0]["host_header"] == "public.example:8443"
    assert httpx.URL(seen[0]["url"]).port == 8443


@pytest.mark.asyncio
async def test_https_presents_the_real_name_for_certificate_verification():
    """Measured against a live host: with this extension the handshake
    succeeds and the cert verifies; without it, SSLV3_ALERT_HANDSHAKE_FAILURE.
    Pinning does NOT cost verification — this is why."""
    seen: list = []
    await _fetch("https://public.example/page", _record(seen))
    assert seen[0]["sni"] == "public.example"


@pytest.mark.asyncio
async def test_plain_http_sets_no_server_name():
    seen: list = []
    await _fetch("http://public.example/page", _record(seen))
    assert seen[0]["sni"] is None


# --------------------------------------------------------------------------- #
# Redirects — where the stale-state bugs live
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_cross_origin_redirect_does_not_inherit_a_stale_server_name():
    """MEASURED on httpx 0.28: a cross-origin redirect RECOMPUTES ``Host`` but
    carries ``extensions`` forward, so an sni_hostname from hop 0 reaches an
    unrelated host unless something clears it. SNI follows Host, so it cannot."""
    seen: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({
            "url": str(request.url),
            "host_header": request.headers.get("Host"),
            "sni": request.extensions.get("sni_hostname"),
        })
        if len(seen) == 1:
            return httpx.Response(302, headers={"Location": "https://1.1.1.1/next"})
        return httpx.Response(200, text="ok")

    await _fetch("https://public.example/1", handler)

    assert seen[0]["sni"] == "public.example"
    assert seen[1]["sni"] is None, (
        f"hop 0's server name leaked to {seen[1]['url']} — the certificate "
        f"would be checked against the wrong host"
    )


@pytest.mark.asyncio
async def test_a_relative_redirect_keeps_the_name_and_the_server_name():
    """httpx preserves Host on a same-origin redirect, so the name survives even
    though the URL now holds an address — and SNI must survive with it."""
    seen: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({
            "url": str(request.url),
            "host_header": request.headers.get("Host"),
            "sni": request.extensions.get("sni_hostname"),
        })
        if len(seen) == 1:
            return httpx.Response(302, headers={"Location": "/second"})
        return httpx.Response(200, text="ok")

    await _fetch("https://public.example/first", handler)

    assert seen[1]["host_header"] == "public.example"
    assert seen[1]["sni"] == "public.example", (
        "the server name was dropped on a same-origin redirect — TLS would then "
        "verify against the pinned address instead of the site"
    )


@pytest.mark.asyncio
async def test_a_redirect_to_a_blocked_address_is_still_refused():
    """The pin must not weaken what the guard already caught."""
    def handler(request: httpx.Request) -> httpx.Response:
        if len(getattr(handler, "n", [])) == 0:
            handler.n = [1]
            return httpx.Response(302, headers={"Location": "http://127.0.0.1:8005/api/status"})
        return httpx.Response(200, text="SECRET")

    handler.n = []
    with pytest.raises(SsrfBlocked) as exc:
        await _fetch("http://public.example/1", handler)
    assert "non-public" in str(exc.value)


# --------------------------------------------------------------------------- #
# IP literals and address families
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_an_ip_literal_url_is_left_alone_and_never_resolved(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("an IP literal must not need getaddrinfo")

    monkeypatch.setattr(url_guard.socket, "getaddrinfo", boom)
    seen: list = []
    await _fetch("http://93.184.216.34/x", _record(seen))
    assert httpx.URL(seen[0]["url"]).host == "93.184.216.34"


def test_an_ipv6_literal_is_validated_and_reports_its_family(monkeypatch):
    monkeypatch.setattr(
        url_guard.socket, "getaddrinfo",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not resolve")),
    )
    target, reason = resolve_pinned("http://[2606:4700::1111]/x")
    assert target is not None, reason
    assert target.was_literal
    assert target.family == socket.AF_INET6


def test_an_ipv6_answer_is_bracketed_in_the_pinned_url(monkeypatch):
    v6 = "2606:4700::1111"
    monkeypatch.setattr(
        url_guard.socket, "getaddrinfo",
        lambda *a, **k: [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", (v6, 0, 0, 0))],
    )
    request = httpx.Request("GET", "https://v6.example/x")
    target, _ = resolve_pinned("https://v6.example/x")
    url_guard.pin_request(request, target)
    assert str(request.url).startswith(f"https://[{v6}]"), str(request.url)
    assert target.family == socket.AF_INET6


def test_the_pinned_address_is_the_one_the_resolver_returned_first(monkeypatch):
    """getaddrinfo applies RFC 6724 sorting, so the first answer is the one the
    system itself would have chosen — including preferring IPv4 on a host with
    no IPv6 route."""
    monkeypatch.setattr(url_guard.socket, "getaddrinfo", lambda *a, **k: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.35", 0)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.36", 0)),
    ])
    target, _ = resolve_pinned("http://multi.example/")
    assert target.address == "93.184.216.35"


def test_every_answer_is_validated_not_only_the_pinned_one(monkeypatch):
    """A name with one public and one loopback answer must not pass on the
    strength of the public one that happens to sort first."""
    monkeypatch.setattr(url_guard.socket, "getaddrinfo", lambda *a, **k: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.35", 0)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0)),
    ])
    target, reason = resolve_pinned("http://split.example/")
    assert target is None
    assert "non-public" in reason


# --------------------------------------------------------------------------- #
# What the model is told
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_the_reported_url_names_the_site_not_the_pinned_address():
    """Answering "URL: https://93.184.216.34/docs" for a request to a name would
    be a worse answer than the one before pinning."""
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, text="ok")),
        follow_redirects=True,
        event_hooks={"request": [guard_request_hop]},
    ) as client:
        response = await client.get("http://public.example/docs")

    assert httpx.URL(str(response.url)).host == PUBLIC_A, "precondition: it was pinned"
    assert display_url(response) == "http://public.example/docs"


@pytest.mark.asyncio
async def test_an_ip_literal_request_is_reported_unchanged():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, text="ok")),
        event_hooks={"request": [guard_request_hop]},
    ) as client:
        response = await client.get("http://93.184.216.34/x")
    assert display_url(response) == "http://93.184.216.34/x"


def test_reporting_never_fails_a_fetch_over_a_missing_request():
    """httpx raises RuntimeError when no request is attached, and callers pass
    response-shaped doubles. A reporting helper must not be able to crash the
    fetch whose result it is describing."""
    class _NoRequest:
        url = httpx.URL("http://public.example/x")

    assert display_url(_NoRequest()) == "http://public.example/x"

    # The precondition that makes the guard necessary rather than defensive
    # decoration: httpx really does raise here, it does not return None.
    with pytest.raises(RuntimeError):
        _ = httpx.Response(200).request


# --------------------------------------------------------------------------- #
# The corrected claim
# --------------------------------------------------------------------------- #


def test_the_module_no_longer_claims_pinning_costs_verification():
    """The false premise is why the door stayed open. It must not come back as
    a justification without someone re-measuring it."""
    source = pathlib.Path(url_guard.__file__).read_text(encoding="utf-8")
    marker = "WHAT THIS USED TO GET WRONG"
    assert marker in source, "the correction was removed"
    body = source.split(marker)[0]
    for stale in ("giving up SNI", "cannot fully close DNS rebinding"):
        assert stale not in body, (
            f"{stale!r} is asserted again outside the correction — it was "
            "measured false: httpcore honours sni_hostname, so a pinned "
            "connection still verifies against the real hostname"
        )
