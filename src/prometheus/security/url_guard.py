"""url_guard — one SSRF predicate for every tool that fetches a URL on request.

WHY ONE MODULE
--------------
Three consumers each carried their own copy of a pre-check-only guard:
``tools/builtin/web_fetch.py``, ``tools/builtin/download_file.py`` and (importing
the first) ``web/references.py`` for @url references. Two independent
``_is_safe_url`` definitions with the same four-flag body. Fixing one would have
left two, which is the same shape ``__main__.py`` warns about for config readers
and ``denied_prune`` had for denied-path matchers: a boundary that looks repaired
while most of it stays inert. So the predicate lives here and all three call it.

WHAT WAS WRONG
--------------
1. The guard ran ONCE on the caller-supplied URL, then httpx followed the whole
   30x chain with ``follow_redirects=True`` and nothing looked at where it landed.
   A public URL that 302s to ``http://127.0.0.1:8005/api/…`` — the daemon's own
   REST surface, which includes bash — was fetched and its body handed to the
   model or written to disk. The tool even advertised "Blocked: URL resolves to a
   private or reserved IP address" while the redirect walked straight past it.

2. The four flags (``is_private``, ``is_loopback``, ``is_reserved``,
   ``is_link_local``) miss addresses that are not on a public route but are not
   in any of those classes either. Measured against this checkout's interpreter:

       100.64.0.1        all four flags False   -> NOT caught
       ::ffff:100.64.0.1 all four flags False   -> NOT caught
       ff02::1           all four flags False   -> NOT caught (multicast)

   ``is_private`` does not cover the CGNAT block 100.64.0.0/10, and IPv4-mapped
   IPv6 inherits whatever gap the mapped address has.

3. Any scheme was accepted. ``file://`` has no hostname, so it fell through the
   ``if not hostname: return False`` line only by accident of ordering; ``ftp://``
   and ``gopher://`` passed with a hostname and were fetched.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
* NO domain allowlist and no restriction on WHICH public URLs may be fetched. The
  operator's instruction on this finding was explicit: check the post-redirect
  host, do not add an allowlist. A tool that only fetches blessed domains is not
  worth having.
* It does not decide the TAILNET question. 100.64.0.0/10 is reachable from here
  and is genuinely two-sided: it is how a fetched page could reach the daemon's
  own :8005 or the GPU box's unauthenticated ollama/whisper/ComfyUI ports, and it
  is also how the operator legitimately points this tool at their own services.
  ``TAILNET_RANGE`` is defined and measured but NOT enforced; see
  :func:`is_blocked_address` and the note at :data:`ENFORCE_TAILNET_BLOCK`.
  Turning it on is an operator decision, not a default.
* It cannot fully close DNS rebinding. The check resolves the name, then httpx
  resolves it again at connect time — a TTL-0 record can flip between the two.
  Checking every redirect hop narrows that to milliseconds per hop rather than
  the whole chain, but closing it outright means pinning the resolved address
  into the connection, which for https means giving up SNI and certificate
  verification against the real hostname. That trade is worse than the residual
  window, so the window is documented rather than papered over.

Source: Prometheus (OAra Labs)
License: MIT
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

#: Schemes this guard will clear. httpx speaks these two; anything else (file,
#: ftp, gopher, data) is refused outright rather than left to whatever the
#: transport happens to do with it.
ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})

#: The CGNAT block Tailscale allocates from. Python's ``is_private`` does not
#: cover it — measured, not assumed: ``ipaddress.ip_address('100.64.0.1')``
#: reports ``is_private=False`` and False for all three sibling flags.
TAILNET_RANGE = ipaddress.ip_network("100.64.0.0/10")

#: Whether to treat the tailnet as blocked. DEFAULT FALSE, deliberately: the
#: operator runs a tailnet fleet and may legitimately point web_fetch at his own
#: services on it. See the module docstring — this is his call to make, and
#: flipping it is one assignment plus the tests that pin the choice.
ENFORCE_TAILNET_BLOCK = False


def is_blocked_address(ip: "ipaddress.IPv4Address | ipaddress.IPv6Address") -> bool:
    """True when *ip* is not on a publicly routable path.

    Covers the four classes the old check did, plus the ones it missed:
    unspecified, multicast, and — unwrapped first — the IPv4 inside an
    IPv4-mapped IPv6 address, which otherwise inherits the mapped address's gap.
    """
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped

    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_reserved
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_unspecified
    ):
        return True

    if ENFORCE_TAILNET_BLOCK and ip in TAILNET_RANGE:
        return True

    return False


def is_ip_literal_blocked(host: str) -> bool:
    """True when *host* is already an IP literal and is blocked.

    Returns False for a hostname (nothing to decide without resolving). Brackets
    are stripped first because ``urlparse`` hands back ``::1`` for
    ``http://[::1]:8005/`` but hands back the literal brackets in some shapes.
    """
    text = host.strip("[]")
    try:
        return is_blocked_address(ipaddress.ip_address(text))
    except ValueError:
        return False


def check_url(url: str) -> tuple[bool, str]:
    """Whether *url* may be fetched, and why not.

    Returns ``(True, "")`` when allowed, else ``(False, reason)``. The reason is
    shown to the model, so it names the class of problem without echoing an
    address that may be attacker-chosen.

    Resolves the hostname and checks EVERY address it returns: a name with one
    public and one loopback answer must not pass on the strength of the public
    one. Resolution failure is a refusal — an unresolvable host is not fetchable
    anyway, and treating the error as "safe" would be the inverse of fail-closed.
    """
    if not url or not isinstance(url, str):
        return False, "no URL supplied"

    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return False, "malformed URL"

    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        return False, f"scheme {scheme!r} is not fetchable (http/https only)"

    hostname = parsed.hostname
    if not hostname:
        return False, "URL has no host"

    if is_ip_literal_blocked(hostname):
        return False, "the host resolves to a non-public address"

    try:
        addrs = socket.getaddrinfo(
            hostname.strip("[]"), None, socket.AF_UNSPEC, socket.SOCK_STREAM
        )
    except (socket.gaierror, socket.herror, UnicodeError, OSError):
        return False, "the host does not resolve"
    if not addrs:
        return False, "the host does not resolve"

    for _family, _type, _proto, _canon, sockaddr in addrs:
        try:
            ip = ipaddress.ip_address(sockaddr[0])
        except (ValueError, IndexError):
            return False, "the host resolves to an unparseable address"
        if is_blocked_address(ip):
            return False, "the host resolves to a non-public address"

    return True, ""


def is_safe_url(url: str) -> bool:
    """Boolean form of :func:`check_url`, for call sites that only need the bit."""
    ok, _reason = check_url(url)
    return ok


# ---------------------------------------------------------------------------
# Per-hop enforcement
# ---------------------------------------------------------------------------

#: Raised when a hop resolves to a non-public address. Distinct from
#: ``httpx.HTTPError`` on purpose: a caller must be able to tell "we refused to go
#: there" from "the network said no". Reported as a transport failure, the refusal
#: reads as something worth retrying, and it is not.
class SsrfBlocked(Exception):
    """A redirect hop (or the original URL) resolved to a non-public address."""


async def guard_request_hop(request) -> None:
    """httpx request hook: re-apply the guard on EVERY hop of a redirect chain.

    THE MECHANISM, verified against httpx 0.28 rather than assumed: the client runs
    request event hooks INSIDE ``_send_handling_redirects``'s ``while True`` loop,
    so a hook sees each hop before it is sent and raising here aborts the chain
    with the body unread. Checked with a MockTransport that a 302 from a public
    host to ``http://127.0.0.1:8005/api/…`` was refused and the secret body never
    arrived — and that the hook saw BOTH hops.

    MUST be ``async``. ``httpx.AsyncClient`` raises TypeError on a sync request
    hook (also verified), so a ``def`` here would turn every guarded fetch into a
    confusing TypeError instead of a guard.

    Lives here, in the security module, rather than in one of the tools: both
    ``web_fetch`` and ``download_file`` build their own client and need the same
    hook, and a tool importing a private helper from a sibling tool is how the two
    drift apart again.

    DNS rebinding is NOT fully closed by this, and the residual window is stated
    rather than papered over: the guard resolves the name, then httpx resolves it
    again at connect time, so a TTL-0 record can flip between the two. Checking
    every hop narrows that to milliseconds per hop instead of the whole chain.
    Closing it outright means pinning the resolved address into the connection,
    which for https costs SNI and certificate verification against the real
    hostname — a worse trade than the residual window.
    """
    ok, reason = check_url(str(request.url))
    if not ok:
        raise SsrfBlocked(reason)
