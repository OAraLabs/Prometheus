"""``POST /api/identity/challenge``: the instance signs a client's nonce with its node key (FOUNDATION Part 3, v4).

A paired client sends a fresh 32-byte nonce on every connect, before it sends its token, and checks the answer:

- the signature verifies against the node key it pinned at pairing, over exactly :func:`challenge_message`;
- ``signed_at`` is within a few minutes of its own clock;
- on a TLS listener, ``tls_spki_sha256`` matches the certificate it actually saw (its existing pin).

What this proves, and what it doesn't. On TLS, the signature binds the node key to the channel: a relay would have
to present the real certificate, which it can't. On a plain listener (this machine's loopback, or a network that is
already encrypted) it proves the real instance is reachable through this path, NOT that nobody is in between: an
active relay could pass the challenge through and then read the token. Its job there is to stop a different
Prometheus, or an impostor with no real instance behind it, from ever receiving a token.

VERSION 2 (GHSA-75cg-fxc2-f93j) closes the plain-listener gap above for a client that asks for it. The client sends
the host it dialled; the daemon signs that host into a message under its own domain line
(``prometheus-identity-challenge/v2``) and the client checks the signed host is the one it dialled. Two refusals make the
binding mean something. The body host must equal the ``Host`` header the daemon actually received (a relay that rewrites
the header is caught). And the host must be one of the daemon's OWN names (:func:`is_own_name`): signing whatever host was
asked for would do nothing against a relay that passes the request through unchanged, because the relay's name is then
the host the client dialled. A relay reached at the daemon's real address (ARP or DNS spoofing, a router) answers under
the daemon's own name and still passes; only the certificate pin (``tls_spki_sha256``) stops that. v1 is unchanged.

Unauthenticated (``web/public_routes.py``), like hello: it has to be answerable before the client sends a
credential. It shares hello's per-peer limit and refusal of browsers, sets no cookie, follows no redirect, writes
nothing, and is stateless: replay protection is the client's own fresh nonce.

Source: Prometheus (OAra Labs)
License: MIT
"""

from __future__ import annotations

import base64
import binascii
import ipaddress
import logging
import re
import socket
import time
from collections.abc import Collection, Mapping
from typing import Any, TypeGuard

logger = logging.getLogger(__name__)

#: The first line of every signed challenge. A signature made here can't be replayed as another kind of
#: signature (an audit record's, say), and the reverse, because the domain line differs.
DOMAIN = "prometheus-identity-challenge/v1"
#: v2's domain line: a v1 signature cannot verify as a v2 message, nor the reverse.
DOMAIN_V2 = "prometheus-identity-challenge/v2"
VERSION = 1
#: The highest version this daemon speaks; hello advertises it as ``ic`` so a client can require it before it connects.
HIGHEST_VERSION = 2
NONCE_BYTES = 32
#: Longest nonce text accepted before decoding: 32 bytes are 43 base64url characters (44 padded).
_MAX_NONCE_CHARS = 64


def challenge_message(
    *, nonce: str, instance_id: str | None, node_pub: str, tls_spki_sha256: str | None, signed_at: int,
) -> bytes:
    """The exact bytes the node key signs: six lines, each ending in a newline. *nonce* is as the client sent it."""
    return (
        f"{DOMAIN}\n"
        f"nonce={nonce}\n"
        f"instance_id={instance_id or ''}\n"
        f"node_pub={node_pub}\n"
        f"tls_spki_sha256={tls_spki_sha256 or ''}\n"
        f"signed_at={signed_at}\n"
    ).encode("utf-8")


def challenge_message_v2(
    *, nonce: str, instance_id: str | None, node_pub: str, tls_spki_sha256: str | None, host: str, signed_at: int,
) -> bytes:
    """The exact bytes a v2 answer signs: seven lines, each ending in a newline. *host* is the canonical host the
    client dialled (:func:`canonical_host`): lowercase, no port, no brackets, no trailing dot."""
    return (
        f"{DOMAIN_V2}\n"
        f"nonce={nonce}\n"
        f"instance_id={instance_id or ''}\n"
        f"node_pub={node_pub}\n"
        f"tls_spki_sha256={tls_spki_sha256 or ''}\n"
        f"host={host}\n"
        f"signed_at={signed_at}\n"
    ).encode("utf-8")


_LABEL = r"[a-z0-9](?:[a-z0-9_-]{0,61}[a-z0-9])?"
_NAME = re.compile(rf"^{_LABEL}(?:\.{_LABEL})*$")
_MAX_HOST_CHARS = 253


def canonical_host(value: Any) -> str | None:
    """*value* if it is already a canonical host, else ``None``.

    Canonical: ASCII, lowercase, no port, no brackets, no trailing dot, no zone id, no control characters; a DNS name of
    valid labels, or an IP address in its compressed lowercase form (RFC 5952; an IPv4-mapped IPv6 address is not
    accepted: dial the IPv4 form). The client sends this form, the daemon signs this form,
    and nothing is normalised on the way: what is signed is what was sent, byte for byte.
    """
    if not isinstance(value, str) or not value or len(value) > _MAX_HOST_CHARS or not value.isascii():
        return None
    if "%" in value or not value.isprintable():
        return None
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return value if _NAME.fullmatch(value) else None
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return None          # Python prints these differently from 3.13; a client dials the IPv4 form anyway
    return value if str(ip) == value else None


def host_of_header(value: str | None) -> str | None:
    """The canonical host named by a ``Host`` header (``name``, ``name:port``, ``[v6]``, ``[v6]:port``), or ``None``.

    Lowercased, and a trailing dot on a name dropped, because HTTP stacks send either. Anything that is not a plain
    authority (userinfo, a path, spaces, a second colon outside brackets) is ``None``.
    """
    if not isinstance(value, str) or not value or not value.isascii():
        return None
    text = value.strip()
    if text.startswith("["):
        end = text.find("]")
        if end < 0:
            return None
        name, rest = text[1:end], text[end + 1:]
        if rest and not (rest.startswith(":") and rest[1:].isdigit() and len(rest) <= 6):
            return None
    else:
        if text.count(":") > 1:
            return None
        name, sep, port = text.partition(":")
        if sep and not (port.isdigit() and len(port) <= 5):
            return None
    return canonical_host(name.lower().rstrip(".") if not name.startswith("[") else name.lower())


def identity_hosts(config: Mapping[str, Any] | None) -> frozenset[str]:
    """``web.identity_hosts``: extra names this daemon answers to (a Tailscale name, a reverse proxy's public name).

    A list of canonical hosts (case and a trailing dot are forgiven). Anything else in it is dropped with a warning,
    and a value that is not a list adds nothing: a typo here can only make the daemon refuse a v2 challenge, never
    answer one for a name it was not given.
    """
    web = config.get("web") if isinstance(config, Mapping) else None
    raw = web.get("identity_hosts") if isinstance(web, Mapping) else None
    if raw is None:
        return frozenset()
    if not isinstance(raw, (list, tuple)):
        logger.warning("web.identity_hosts must be a list of host names; ignoring %r", type(raw).__name__)
        return frozenset()
    kept: set[str] = set()
    for item in raw:
        host = canonical_host(item.lower().rstrip(".")) if isinstance(item, str) else None
        if host is None:
            logger.warning("web.identity_hosts: ignoring %r (not a plain host name or IP address)", item)
        else:
            kept.add(host)
    return frozenset(kept)


def _as_ip(text: str | None) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    if not text:
        return None
    try:
        ip = ipaddress.ip_address(text.partition("%")[0])
    except ValueError:
        return None
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    return ip


def _advertised_name() -> str | None:
    """The name this daemon registers its service under on the network (``prometheus-<fp8>.local``,
    ``web/discovery.py``): a phone that dials the SRV target must not be refused. Derived from the public
    fingerprint, read without creating anything."""
    try:
        from prometheus.config.instance_key import instance_fingerprint
        from prometheus.web.discovery import _server_name

        return _server_name({"fp": instance_fingerprint().encode("ascii")}).rstrip(".")
    except Exception:  # noqa: BLE001 - a missing optional piece only narrows what is accepted
        return None


def is_own_name(host: str, *, local_ip: str | None, extra_hosts: Collection[str]) -> bool:
    """Whether *host* is a name or address this daemon answers to.

    * a loopback name or address;
    * an IP address equal to the local address the connection arrived on (*local_ip*, the socket's own end: a client
      that dialled an address reached it, and a TCP relay's address is not it);
    * this machine's host name, its first label, and ``<label>.local``, and the name it advertises over mDNS;
    * a name or address listed in ``web.identity_hosts``.

    The ``.local`` and bare-label names are the weakest of these: anyone on the same network can answer mDNS or LLMNR
    for them, which is how a relay gets in the way in the first place. They are accepted because they are this
    daemon's real names and a client may dial them; a client that wants more dials the IP address discovery gave it
    (the local-address rule needs the attacker on the path of that address), and the certificate pin is stronger still.

    NEVER resolves a name. The caller picks the name, so a DNS answer is the caller's too (split-horizon or rebinding
    makes any name "resolve to this machine"). Port is not part of it: a NAT forward or a published container port
    changes the port and not the name. KNOWN LIMIT: a forwarder on this address at another port gets a signature, so a
    hostile local user on a shared machine, or anything that owns this address, can relay. The pin stops that.
    """
    candidate = _as_ip(host)
    if candidate is not None:
        if candidate.is_unspecified:
            return False
        if candidate.is_loopback:
            return True
        local = _as_ip(local_ip)
        if local is not None and not local.is_unspecified and candidate == local:
            return True
        return host in extra_hosts or str(candidate) in extra_hosts
    if host == "localhost" or host in extra_hosts or host == _advertised_name():
        return True
    try:
        machine = socket.gethostname().strip().lower().rstrip(".")
    except OSError:
        return False
    if not machine:
        return False
    label = machine.partition(".")[0]
    return host in {machine, label, f"{label}.local"}


def valid_nonce(value: Any) -> TypeGuard[str]:
    """Whether *value* is base64url text for exactly :data:`NONCE_BYTES` bytes (padding optional)."""
    if not isinstance(value, str) or not value or len(value) > _MAX_NONCE_CHARS:
        return False
    try:
        raw = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except (binascii.Error, ValueError):
        return False
    return len(raw) == NONCE_BYTES and base64.urlsafe_b64encode(raw).rstrip(b"=") == value.rstrip("=").encode()


def answer(
    body: Any, *, tls: bool, host_header: str | None = None, local_ip: str | None = None,
    extra_hosts: Collection[str] = frozenset(),
) -> tuple[int, dict[str, Any]]:
    """``(status, content)`` for one request body. *tls*: whether the listener it came in on speaks TLS.

    *host_header* is the ``Host`` header the request carried and *local_ip* the local address it arrived on; v2 needs
    both (a missing one is a refusal, never a silent v1). A request that does not ask for ``"v": 2`` gets the v1
    answer, exactly as before.
    """
    from prometheus.config.node_identity import get_instance_id, load_node_signer

    nonce = body.get("nonce") if isinstance(body, dict) else None
    if not valid_nonce(nonce):
        return 400, {"error": "bad_nonce",
                     "detail": f"nonce must be base64url text for exactly {NONCE_BYTES} random bytes"}
    asked = body.get("v") if isinstance(body, dict) else None
    if asked is not None and not (type(asked) is int and asked in (1, 2)):
        return 400, {"error": "bad_version", "detail": "v must be 1 or 2 (a number), or left out for 1"}
    v2 = asked == 2
    host: str | None = None
    if v2:
        host = canonical_host(body.get("host"))
        if host is None:
            return 400, {"error": "bad_host",
                         "detail": "v2 needs \"host\": the host you dialled, lowercase, without port or brackets"}
        if host_of_header(host_header) != host:
            return 400, {"error": "host_mismatch",
                         "detail": "\"host\" must be the host in this request's Host header"}
        if not is_own_name(host, local_ip=local_ip, extra_hosts=extra_hosts):
            return 421, {"error": "host_not_mine",
                         "detail": "this Prometheus does not answer to that name; if it should, list it under "
                                   "web.identity_hosts"}
    signer = load_node_signer()
    if signer is None:
        return 503, {"error": "no_node_key", "detail": "this Prometheus has no node key yet (it is made at first run)"}
    node_pub, sign = signer
    spki: str | None = None
    if tls:
        from prometheus.web.tls import spki_sha256

        spki = spki_sha256()
    instance_id = get_instance_id()
    signed_at = int(time.time())
    if v2:
        assert host is not None
        message = challenge_message_v2(nonce=nonce, instance_id=instance_id, node_pub=node_pub,
                                       tls_spki_sha256=spki, host=host, signed_at=signed_at)
        return 200, {
            "v": 2,
            "instance_id": instance_id,
            "node_pub": node_pub,
            "tls_spki_sha256": spki,
            "host": host,
            "signed_at": signed_at,
            "sig": base64.b64encode(sign(message)).decode("ascii"),
        }
    message = challenge_message(nonce=nonce, instance_id=instance_id, node_pub=node_pub,
                                tls_spki_sha256=spki, signed_at=signed_at)
    return 200, {
        "v": VERSION,
        "instance_id": instance_id,
        "node_pub": node_pub,
        "tls_spki_sha256": spki,
        "signed_at": signed_at,
        "sig": base64.b64encode(sign(message)).decode("ascii"),
    }
