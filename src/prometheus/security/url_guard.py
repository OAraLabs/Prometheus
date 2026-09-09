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
* It does not block the tailnet on the URL THE CALLER SUPPLIED. That stays an
  operator decision (:data:`ENFORCE_TAILNET_BLOCK`, default False) because
  pointing this tool at one's own services is a legitimate, routine use.

  It DOES block the tailnet on every redirect hop, unconditionally. The two
  cases only look alike: hop 0 is a destination the operator chose, and a
  redirect is a destination a remote server chose. Nothing legitimate requires
  a fetched page to be able to steer the next hop onto the private fleet, and
  what sits there is not theoretical — the GPU box answers
  ``GET /api/tags`` on its inference port with HTTP 200 and the full model
  list, no token. See :func:`guard_request_hop`.
* It does NOT keep happy eyeballs. Pinning means one address is chosen, so a
  host whose first resolved address is unreachable now fails instead of falling
  back to its second. The address is the one ``getaddrinfo`` returned first,
  i.e. the one the system would have picked anyway.

WHAT THIS USED TO GET WRONG
---------------------------
This module previously listed DNS rebinding as something it "cannot fully
close", on the reasoning that pinning the resolved address "means giving up SNI
and certificate verification against the real hostname" — a worse trade than a
residual window narrowed to milliseconds per hop.

Both halves were false. The window was an open door, not milliseconds: with a
resolver answering public for the guard and loopback for the connection, a fetch
returned a loopback service's body with HTTP 200. And the trade does not exist —
httpcore honours a ``sni_hostname`` extension, so a connection to a pinned IP
still presents the real server name and still verifies the certificate against
it. :func:`guard_request_hop` carries the measurements.

Source: Prometheus (OAra Labs)
License: MIT
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

#: Schemes this guard will clear. httpx speaks these two; anything else (file,
#: ftp, gopher, data) is refused outright rather than left to whatever the
#: transport happens to do with it.
ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})

#: The CGNAT block Tailscale allocates from. Python's ``is_private`` does not
#: cover it — measured, not assumed: ``ipaddress.ip_address('100.64.0.1')``
#: reports ``is_private=False`` and False for all three sibling flags.
TAILNET_RANGE = ipaddress.ip_network("100.64.0.0/10")

#: Whether to treat the tailnet as blocked ON THE CALLER-SUPPLIED URL. DEFAULT
#: FALSE, deliberately: the operator runs a tailnet fleet and may legitimately
#: point web_fetch at his own services on it. See the module docstring — this is
#: his call to make, and flipping it is one assignment plus the tests that pin
#: the choice.
#:
#: This flag does NOT govern redirect hops. Those block the tailnet regardless,
#: because the destination was chosen by a remote server rather than by the
#: operator; see :func:`guard_request_hop`. Setting this True therefore tightens
#: hop 0 to match, it does not "turn the feature on".
ENFORCE_TAILNET_BLOCK = False


def is_blocked_address(
    ip: "ipaddress.IPv4Address | ipaddress.IPv6Address",
    *,
    block_tailnet: bool | None = None,
) -> bool:
    """True when *ip* is not on a publicly routable path.

    Covers the four classes the old check did, plus the ones it missed:
    unspecified, multicast, and — unwrapped first — the IPv4 inside an
    IPv4-mapped IPv6 address, which otherwise inherits the mapped address's gap.

    *block_tailnet* decides the 100.64.0.0/10 question FOR THIS CALL. ``None``
    means "use :data:`ENFORCE_TAILNET_BLOCK`", which is what every pre-existing
    caller gets. It is a parameter rather than only a module flag because the
    right answer differs by hop: see :func:`guard_request_hop`.
    """
    if block_tailnet is None:
        block_tailnet = ENFORCE_TAILNET_BLOCK
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

    if block_tailnet and ip in TAILNET_RANGE:
        return True

    return False


def is_ip_literal_blocked(host: str, *, block_tailnet: bool | None = None) -> bool:
    """True when *host* is already an IP literal and is blocked.

    Returns False for a hostname (nothing to decide without resolving). Brackets
    are stripped first because ``urlparse`` hands back ``::1`` for
    ``http://[::1]:8005/`` but hands back the literal brackets in some shapes.
    """
    text = host.strip("[]")
    try:
        return is_blocked_address(
            ipaddress.ip_address(text), block_tailnet=block_tailnet
        )
    except ValueError:
        return False


@dataclass(frozen=True)
class PinnedTarget:
    """The ONE address a validated URL is allowed to connect to.

    The whole point of returning an address rather than a bool: the name is
    resolved once, and that answer is what the socket uses. A second lookup at
    connect time is a second chance for the answer to change.
    """

    #: The name to present as ``Host:`` and as the TLS server name. Equals
    #: *address* when the URL already named an IP.
    hostname: str
    #: The validated IP the connection must go to.
    address: str
    #: ``AF_INET`` / ``AF_INET6`` — carried through from the answer that was
    #: validated, never re-selected at connect time.
    family: int
    #: True when the URL already named an IP literal: already validated by
    #: value, and there is nothing to pin.
    was_literal: bool


def resolve_pinned(
    url: str, *, block_tailnet: bool | None = None
) -> tuple[PinnedTarget | None, str]:
    """Validate *url* and return the single address it may connect to.

    ``(None, reason)`` when it may not be fetched at all.

    EVERY resolved address is checked, not just the one returned: a name with
    one public and one loopback answer must not pass on the strength of the
    public one. The address handed back is the FIRST the resolver returned,
    which is the one the system itself would have chosen — ``getaddrinfo``
    applies RFC 6724 sorting, so on a host with no IPv6 route the A record
    already sorts first.

    Resolution failure is a refusal: an unresolvable host is not fetchable
    anyway, and reading the error as "safe" would invert fail-closed.
    """
    if not url or not isinstance(url, str):
        return None, "no URL supplied"

    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return None, "malformed URL"

    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        return None, f"scheme {scheme!r} is not fetchable (http/https only)"

    hostname = parsed.hostname
    if not hostname:
        return None, "URL has no host"

    literal = hostname.strip("[]")
    try:
        ip = ipaddress.ip_address(literal)
    except ValueError:
        ip = None
    if ip is not None:
        if is_blocked_address(ip, block_tailnet=block_tailnet):
            return None, "the host resolves to a non-public address"
        family = socket.AF_INET6 if ip.version == 6 else socket.AF_INET
        return PinnedTarget(literal, literal, family, True), ""

    try:
        addrs = socket.getaddrinfo(
            literal, None, socket.AF_UNSPEC, socket.SOCK_STREAM
        )
    except (socket.gaierror, socket.herror, UnicodeError, OSError):
        return None, "the host does not resolve"
    if not addrs:
        return None, "the host does not resolve"

    for _family, _type, _proto, _canon, sockaddr in addrs:
        try:
            ipaddress.ip_address(sockaddr[0])
        except (ValueError, IndexError):
            return None, "the host resolves to an unparseable address"
        if is_blocked_address(
            ipaddress.ip_address(sockaddr[0]), block_tailnet=block_tailnet
        ):
            return None, "the host resolves to a non-public address"

    family, _t, _p, _c, sockaddr = addrs[0]
    return PinnedTarget(hostname, sockaddr[0], family, False), ""


def check_url(url: str, *, block_tailnet: bool | None = None) -> tuple[bool, str]:
    """Whether *url* may be fetched, and why not.

    Returns ``(True, "")`` when allowed, else ``(False, reason)``. The reason is
    shown to the model, so it names the class of problem without echoing an
    address that may be attacker-chosen.

    The boolean form of :func:`resolve_pinned`, kept for the pre-flight callers
    that only need the bit. The two cannot disagree: this IS that function.
    """
    target, reason = resolve_pinned(url, block_tailnet=block_tailnet)
    return target is not None, reason


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


#: Extensions key marking "this request has already been through the guard", so
#: a later hop can tell it is a REDIRECT rather than the caller's own URL. Not a
#: security boundary — a remote server cannot write here — just the hop counter
#: httpx does not otherwise expose to an event hook.
_HOP_SEEN = "url_guard_hop_seen"


def _host_header_name(request) -> str | None:
    """The NAME in the Host header, or None when there isn't one.

    Returns None for an absent header and for one carrying a bare IP, because
    neither names a server whose certificate could be checked against it.
    """
    raw = request.headers.get("Host")
    if not raw:
        return None
    name = raw.rsplit(":", 1)[0] if raw.count(":") == 1 else raw
    name = name.strip("[]")
    try:
        ipaddress.ip_address(name)
    except ValueError:
        return name or None
    return None


def pin_request(request, target: PinnedTarget) -> None:
    """Point this request at the validated address, keeping its identity intact.

    The URL's host becomes the IP, so the socket connects where the guard
    looked. ``Host:`` and the TLS server name stay the original name, so
    virtual hosting still routes and the certificate is still verified against
    the name the caller asked for — which is why pinning does NOT cost SNI or
    verification, contrary to what this module used to claim.

    ONE RULE FOR SNI: it follows the Host header. Measured on httpx 0.28 —
    a cross-origin redirect RECOMPUTES ``Host`` but carries ``extensions``
    forward, so a ``sni_hostname`` set on an earlier hop reaches a later,
    unrelated host unless something clears it. Deriving it from Host each hop
    means there is no stale value to leak, and it keeps a same-origin relative
    redirect working: httpx preserves ``Host`` there, so the name survives even
    though the URL now holds an IP.
    """
    if target.was_literal:
        # Nothing to pin — the URL named an address and it was validated by
        # value. Still fix SNI, because this hop may be a redirect carrying a
        # previous hop's name.
        name = _host_header_name(request)
        if name:
            request.extensions["sni_hostname"] = name
        else:
            request.extensions.pop("sni_hostname", None)
        return

    port = request.url.port
    authority = target.hostname if port is None else f"{target.hostname}:{port}"
    literal = f"[{target.address}]" if ":" in target.address else target.address

    request.url = request.url.copy_with(host=literal)
    # Set explicitly, not left to httpx. Measured: httpx fills Host at Request
    # CONSTRUCTION and does not recompute it when ``request.url`` changes, so
    # today this re-affirms the value that is already there. It stays because
    # the invariant — the origin server is told the name, not the address — must
    # not rest on that implementation detail, and because a caller that builds
    # its own headers can arrive here without a Host at all.
    request.headers["Host"] = authority
    if request.url.scheme == "https":
        request.extensions["sni_hostname"] = target.hostname
    else:
        request.extensions.pop("sni_hostname", None)


def display_url(response) -> str:
    """The response's URL with the pinned IP swapped back for the real name.

    Pinning puts an address in ``request.url``, which is what the socket needs
    and NOT what the caller asked for. ``web_fetch`` reports this string to the
    model, and answering "URL: https://93.184.216.34/docs" for a request to
    ``example.com`` would be a worse answer than the one before pinning.

    Reconstructed from the ``Host`` header rather than from stashed state,
    because extensions carry across redirect hops and stashed state would go
    stale exactly where redirects make it hardest to notice. Redirects still
    show their real final URL: the Host header tracks each hop.
    """
    url = response.url
    try:
        request = response.request
    except (AttributeError, RuntimeError):
        # httpx raises RuntimeError when no request is attached, and a caller
        # may hand us a response-shaped object that has none. Reporting is not
        # worth failing a fetch over: fall back to the URL as it stands.
        return str(url)

    name = _host_header_name(request)
    if name and name != url.host:
        url = url.copy_with(host=name)
    return str(url)


async def guard_request_hop(request) -> None:
    """httpx request hook: validate EVERY hop, and pin it to what was validated.

    THE MECHANISM, verified against httpx 0.28 rather than assumed: the client
    runs request event hooks INSIDE ``_send_handling_redirects``'s ``while True``
    loop, so a hook sees each hop before it is sent and raising here aborts the
    chain with the body unread. Checked with a MockTransport that a 302 from a
    public host to ``http://127.0.0.1:8005/api/…`` was refused and the secret
    body never arrived — and that the hook saw BOTH hops.

    MUST be ``async``. ``httpx.AsyncClient`` raises TypeError on a sync request
    hook (also verified), so a ``def`` here would turn every guarded fetch into
    a confusing TypeError instead of a guard.

    Lives here, in the security module, rather than in one of the tools: both
    ``web_fetch`` and ``download_file`` build their own client and need the same
    hook, and a tool importing a private helper from a sibling tool is how the
    two drift apart again.

    DNS REBINDING IS CLOSED HERE, and the claim it replaces was wrong. This
    docstring used to say the window was "narrowed to milliseconds per hop but
    not closed", and that closing it "means pinning the resolved address into
    the connection, which for https costs SNI and certificate verification
    against the real hostname — a worse trade than the residual window."

    Both halves were false, and measurement is what showed it:

    * It was not a narrow window, it was an open door. With a resolver that
      answered public for the guard's lookups and loopback for the connection's,
      a fetch returned the body of a loopback service — HTTP 200, secret read.
      The detail that hides this from a casual patch: connect-time resolution
      DOES go through ``socket.getaddrinfo``, but the host arrives as **bytes**,
      so an interceptor comparing against a ``str`` never fires.
    * The trade does not exist. httpcore honours a ``sni_hostname`` request
      extension, so a connection to a pinned IP can still present the real
      server name and verify the certificate against it. Measured against a live
      host with a fresh client per attempt: with the extension, HTTP 200; without
      it, ``SSLV3_ALERT_HANDSHAKE_FAILURE``. Verification is not given up — it is
      the reason the extension is set.

    HOP ASYMMETRY, and what actually justifies it. The tailnet is blocked on
    every redirect hop and left alone on the caller's own URL. Measured, on this
    fleet, today:

    * The real open door is the GPU box's INFERENCE PORT: an unauthenticated
      ``GET /api/tags`` returns HTTP 200 and the full model list. Whisper and
      ComfyUI sit beside it on the same terms. A fetched page that can redirect
      onto the fleet reaches those with no credential at all.
    * The daemon's own token gate is CONDITIONAL: ``server.py`` reads
      ``if _api_token and path.startswith(("/api/", "/v1/"))``. An empty token
      skips the gate entirely, so on a deployment where it is unset — a fresh
      install before the wizard runs — ``/api/bash`` is open on 0.0.0.0.

    NOT justified by the daemon's :8005 as it runs here, and the difference is
    worth keeping straight. It does bind 0.0.0.0, but with a token set every
    ``/api/*`` route answers 401 — verified just now, including
    ``POST /api/bash`` with ``{"command":"id"}``. Only ``/`` returns 200, and it
    leaks the endpoint inventory: information disclosure, not code execution.
    Citing that as the reason would overstate the finding and make the two real
    ones easier to dismiss.

    What this costs, and it is not nothing: a public page that legitimately
    redirects to a tailnet host will now be refused. The operator can still
    fetch that host by naming it directly, which is the case
    ``ENFORCE_TAILNET_BLOCK`` protects.

    WHAT PINNING COSTS, stated plainly: one address is chosen instead of the
    happy-eyeballs walk anyio would do, so a host whose first resolved address is
    unreachable now fails rather than falling back to its second. The address is
    the one ``getaddrinfo`` returned first, which is the one the system would
    have picked anyway.
    """
    # HOP ASYMMETRY. A redirect is a destination chosen by a REMOTE SERVER, and
    # nothing legitimate needs that server to be able to steer a fetch onto the
    # tailnet. The caller-supplied URL is different in kind: the operator points
    # this tool at his own services on purpose, which is why
    # ENFORCE_TAILNET_BLOCK stays False and governs hop 0 only.
    #
    # Measured against the real guard across three hops: a value written into
    # ``extensions`` at hop 0 is visible on every later hop (httpx carries the
    # contents forward when it builds a redirect request), so the marker below
    # is enough to tell "the caller asked for this" from "a server sent us here"
    # with no httpx change and no state of our own to keep in sync.
    from_redirect = request.extensions.get(_HOP_SEEN) is not None

    target, reason = resolve_pinned(
        str(request.url), block_tailnet=True if from_redirect else None
    )
    if target is None:
        if from_redirect:
            raise SsrfBlocked(f"{reason} (redirect target)")
        raise SsrfBlocked(reason)
    pin_request(request, target)
    request.extensions[_HOP_SEEN] = True
