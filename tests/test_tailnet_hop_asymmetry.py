"""The tailnet is the operator's to reach, and a remote server's to be refused.

Two destinations that look identical to an address check are different in kind:

* the URL the CALLER supplied — the operator pointing this tool at his own
  services, which is routine and legitimate;
* a REDIRECT target — a destination chosen by whatever server answered the last
  hop, which has no legitimate reason to steer a fetch onto a private fleet.

So the tailnet is allowed on hop 0 (governed by ``ENFORCE_TAILNET_BLOCK``, off
by default) and blocked on every hop after it, unconditionally.

WHAT JUSTIFIES IT, measured on this fleet rather than argued:

* the GPU box's inference port answers an unauthenticated ``GET /api/tags``
  with HTTP 200 and its full model list;
* the daemon's own gate is conditional — ``if _api_token and path.startswith(
  ("/api/", "/v1/"))`` — so an unset token skips it entirely.

NOT justified by the daemon's :8005 as it runs today: with a token set, every
``/api/*`` route answers 401, including ``POST /api/bash``. Only ``/`` returns
200 and leaks the endpoint inventory. That is disclosure, not execution, and
citing it would overstate the finding.

Addresses here are from the 100.64.0.0/24 head of the CGNAT block — the range
the pre-commit hook allowlists for exactly this purpose. No machine on the real
fleet is addressed here.
"""

from __future__ import annotations

import ipaddress

import httpx
import pytest

from prometheus.security import url_guard
from prometheus.security.url_guard import (
    ENFORCE_TAILNET_BLOCK,
    SsrfBlocked,
    check_url,
    guard_request_hop,
    is_blocked_address,
    resolve_pinned,
)

TAILNET = "100.64.0.7"
TAILNET_2 = "100.64.0.9"
PUBLIC = "93.184.216.34"
LOOPBACK = "127.0.0.1"


def _chain(*locations: str):
    """A transport that walks a fixed redirect chain, recording every hop."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        i = len(seen) - 1
        if i < len(locations):
            return httpx.Response(302, headers={"Location": locations[i]})
        return httpx.Response(200, text="ok")

    return handler, seen


async def _walk(start: str, handler):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        follow_redirects=True,
        event_hooks={"request": [guard_request_hop]},
    ) as client:
        return await client.get(start)


# --------------------------------------------------------------------------- #
# Hop 0 — the operator's own choice, unchanged
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_the_caller_may_still_point_the_tool_at_the_tailnet():
    """The capability this asymmetry exists to preserve."""
    handler, seen = _chain()
    response = await _walk(f"http://{TAILNET}:11434/api/tags", handler)

    assert response.status_code == 200
    assert seen == [f"http://{TAILNET}:11434/api/tags"]


def test_hop_zero_follows_the_operator_flag_and_it_is_off():
    assert ENFORCE_TAILNET_BLOCK is False
    ok, _ = check_url(f"http://{TAILNET}/")
    assert ok is True


# --------------------------------------------------------------------------- #
# Redirects — a remote server's choice, refused
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_redirect_onto_the_tailnet_is_refused():
    handler, seen = _chain(f"http://{TAILNET}:11434/api/tags")

    with pytest.raises(SsrfBlocked) as exc:
        await _walk(f"http://{PUBLIC}/start", handler)

    assert "redirect" in str(exc.value)
    assert seen == [f"http://{PUBLIC}/start"], (
        "the tailnet hop was actually sent — the refusal came too late"
    )


@pytest.mark.asyncio
async def test_the_body_of_a_tailnet_redirect_never_reaches_the_caller():
    """What matters is not the raise, it is that nothing comes back."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == PUBLIC:
            return httpx.Response(302, headers={"Location": f"http://{TAILNET}:11434/api/tags"})
        return httpx.Response(200, text="MODEL-LIST-MUST-NOT-BE-RETURNED")

    body = None
    try:
        body = (await _walk(f"http://{PUBLIC}/start", handler)).text
    except SsrfBlocked:
        pass
    assert body is None


@pytest.mark.asyncio
async def test_the_block_holds_deep_in_a_chain_not_just_on_hop_one():
    handler, seen = _chain(f"http://{PUBLIC}/2", f"http://{PUBLIC}/3",
                           f"http://{TAILNET_2}/steal")

    with pytest.raises(SsrfBlocked):
        await _walk(f"http://{TAILNET}/mine", handler)

    # Hop 0 was the operator's own tailnet host and was allowed; the chain then
    # ran through two public hops and was stopped at the tailnet one.
    assert seen == [f"http://{TAILNET}/mine", f"http://{PUBLIC}/2", f"http://{PUBLIC}/3"]


@pytest.mark.asyncio
async def test_a_public_chain_is_untouched():
    """Guard the guard: over-blocking would be invisible without this."""
    handler, seen = _chain(f"http://{PUBLIC}/2", f"http://{PUBLIC}/3")
    response = await _walk(f"http://{PUBLIC}/1", handler)

    assert response.status_code == 200
    assert len(seen) == 3


@pytest.mark.asyncio
async def test_a_response_header_cannot_forge_the_hop_marker():
    """The marker lives in httpx's extensions, which no remote server writes.

    If it were ever read from anything the response controls, a server could
    claim to be hop 0 and unblock itself.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == PUBLIC:
            return httpx.Response(
                302,
                headers={
                    "Location": f"http://{TAILNET}/x",
                    "url_guard_hop_seen": "",
                    "X-Url-Guard-Hop-Seen": "false",
                },
            )
        return httpx.Response(200, text="reached")

    with pytest.raises(SsrfBlocked):
        await _walk(f"http://{PUBLIC}/1", handler)


# --------------------------------------------------------------------------- #
# Everything else the guard blocked, it still blocks
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize("addr", [LOOPBACK, "169.254.169.254", "10.0.0.5"])
async def test_non_tailnet_private_addresses_are_still_refused_on_hop_zero(addr):
    handler, _ = _chain()
    with pytest.raises(SsrfBlocked) as exc:
        await _walk(f"http://{addr}/", handler)
    assert "non-public" in str(exc.value)
    assert "redirect" not in str(exc.value), "hop 0 must not be reported as a redirect"


@pytest.mark.asyncio
async def test_a_redirect_to_loopback_is_still_refused():
    handler, _ = _chain(f"http://{LOOPBACK}:8005/api/status")
    with pytest.raises(SsrfBlocked) as exc:
        await _walk(f"http://{PUBLIC}/1", handler)
    assert "non-public" in str(exc.value)


# --------------------------------------------------------------------------- #
# The per-call decision itself
# --------------------------------------------------------------------------- #


def test_the_address_check_takes_a_per_call_tailnet_decision():
    ip = ipaddress.ip_address(TAILNET)
    assert is_blocked_address(ip, block_tailnet=True) is True
    assert is_blocked_address(ip, block_tailnet=False) is False
    # None means "ask the operator flag", which is what every old caller gets.
    assert is_blocked_address(ip) is ENFORCE_TAILNET_BLOCK
    assert is_blocked_address(ip, block_tailnet=None) is ENFORCE_TAILNET_BLOCK


def test_a_per_call_decision_does_not_loosen_anything_else():
    """block_tailnet=False must not turn off the rest of the check."""
    for addr in (LOOPBACK, "10.0.0.5", "169.254.169.254", "::1"):
        assert is_blocked_address(
            ipaddress.ip_address(addr), block_tailnet=False
        ) is True


def test_resolve_pinned_threads_the_decision_to_ip_literals():
    allowed, _ = resolve_pinned(f"http://{TAILNET}/", block_tailnet=False)
    refused, reason = resolve_pinned(f"http://{TAILNET}/", block_tailnet=True)
    assert allowed is not None
    assert refused is None and "non-public" in reason


@pytest.mark.asyncio
async def test_turning_the_operator_flag_on_tightens_hop_zero_too(monkeypatch):
    """The flag governs hop 0. Redirects were already blocked without it, so
    setting it must not be describable as 'turning the feature on'."""
    monkeypatch.setattr(url_guard, "ENFORCE_TAILNET_BLOCK", True)

    handler, _ = _chain()
    with pytest.raises(SsrfBlocked):
        await _walk(f"http://{TAILNET}/", handler)
