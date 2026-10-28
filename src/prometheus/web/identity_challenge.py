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

Unauthenticated (``web/public_routes.py``), like hello: it has to be answerable before the client sends a
credential. It shares hello's per-peer limit and refusal of browsers, sets no cookie, follows no redirect, writes
nothing, and is stateless: replay protection is the client's own fresh nonce.

Source: Prometheus (OAra Labs)
License: MIT
"""

from __future__ import annotations

import base64
import binascii
import time
from typing import Any, TypeGuard

#: The first line of every signed challenge. A signature made here can't be replayed as another kind of
#: signature (an audit record's, say), and the reverse, because the domain line differs.
DOMAIN = "prometheus-identity-challenge/v1"
VERSION = 1
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


def valid_nonce(value: Any) -> TypeGuard[str]:
    """Whether *value* is base64url text for exactly :data:`NONCE_BYTES` bytes (padding optional)."""
    if not isinstance(value, str) or not value or len(value) > _MAX_NONCE_CHARS:
        return False
    try:
        raw = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except (binascii.Error, ValueError):
        return False
    return len(raw) == NONCE_BYTES and base64.urlsafe_b64encode(raw).rstrip(b"=") == value.rstrip("=").encode()


def answer(body: Any, *, tls: bool) -> tuple[int, dict[str, Any]]:
    """``(status, content)`` for one request body. *tls*: whether the listener it came in on speaks TLS."""
    from prometheus.config.node_identity import get_instance_id, load_node_signer

    nonce = body.get("nonce") if isinstance(body, dict) else None
    if not valid_nonce(nonce):
        return 400, {"error": "bad_nonce",
                     "detail": f"nonce must be base64url text for exactly {NONCE_BYTES} random bytes"}
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
