"""Sealing the pairing token to the requester's key, and the match code.

Pairing hands a new device a bearer token over a network the contract assumes someone else can read, so
the token is SEALED to a public key the device sent in its request: nothing on the wire, and nothing the
daemon stores, holds it in clear. Only the holder of the matching private key can open it. The recipe is
fixed so a client in another language can implement the other half
(docs/PAIRING-APPROVAL-API.md, 4.3 and 4.4), and ``tests/vectors/pairing_seal_v1.json`` holds bytes a
separate implementation produced from the recipe alone:

    shared     = X25519(ephemeral_private, requester_public)
    key        = HKDF-SHA256(shared, salt=ASCII(request_id), info="prometheus-pair-v1/seal", 32 bytes)
    ciphertext = ChaCha20-Poly1305(key).encrypt(nonce12, plaintext_utf8, aad=ASCII(request_id)) + tag

The request id is both the HKDF salt and the associated data, so a sealed blob lifted onto another
request opens nothing. The ephemeral key makes every seal fresh.

The MATCH CODE is a 4-digit hash of both keys and the request id that the requester computes locally and
shows on its own screen, and the daemon shows to the operator, who compares them. It catches a naive relay
that swapped a key. It is a hash an active attacker can grind (10,000 values), so trust on first use is as
strong as SSH's and no stronger; the contract's 7.5 says so.

Source: novel code for Prometheus, 2026-10-08.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from typing import Any

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

ALG = "x25519-hkdf-sha256-chacha20poly1305"

_MATCH_DOMAIN = b"prometheus-pair-v1"
_SEAL_INFO = b"prometheus-pair-v1/seal"
_NONCE_BYTES = 12
_KEY_BYTES = 32
_B64URL = re.compile(r"[A-Za-z0-9_-]*")


def b64url_encode(raw: bytes) -> str:
    """Unpadded base64url, the encoding of every binary field on the wire."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def b64url_decode(text: str) -> bytes:
    """Strict unpadded base64url. Raises ``ValueError`` for anything else (padding, ``+``, ``/``, spaces)."""
    if not isinstance(text, str) or not _B64URL.fullmatch(text) or len(text) % 4 == 1:
        raise ValueError("not unpadded base64url")
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def check_requester_key(raw: bytes) -> None:
    """Raise ``ValueError`` unless *raw* is a usable X25519 public key.

    32 bytes, and not a low-order point: an all-zero key makes the shared secret a constant anyone can
    compute, so a token sealed to it would be sealed to nobody. Caught at the request, before a token exists.
    """
    if not isinstance(raw, (bytes, bytearray)) or len(raw) != _KEY_BYTES:
        raise ValueError("an X25519 public key is 32 bytes")
    X25519PrivateKey.generate().exchange(X25519PublicKey.from_public_bytes(bytes(raw)))   # ValueError if low order


def match_code(device_public_key: bytes, instance_public_key_der: bytes, request_id: str) -> str:
    """The 4-digit code both sides compute: SHA-256 over both keys and the request id, mod 10,000."""
    digest = hashlib.sha256(
        _MATCH_DOMAIN + b"\x00" + bytes(device_public_key) + bytes(instance_public_key_der) + request_id.encode("ascii")
    ).digest()
    return f"{int.from_bytes(digest[:4], 'big') % 10000:04d}"


def _derive_key(shared: bytes, request_id: str) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=_KEY_BYTES, salt=request_id.encode("ascii"),
                info=_SEAL_INFO).derive(shared)


def seal_token(
    *,
    requester_public_key: bytes,
    request_id: str,
    payload: dict[str, Any],
    _ephemeral_private: X25519PrivateKey | None = None,
    _nonce: bytes | None = None,
) -> dict[str, str]:
    """Seal *payload* (JSON) to *requester_public_key*. Returns the four wire fields.

    ``_ephemeral_private`` and ``_nonce`` exist so the test vectors can be reproduced byte for byte;
    production never passes them.
    """
    check_requester_key(requester_public_key)
    ephemeral = _ephemeral_private or X25519PrivateKey.generate()
    shared = ephemeral.exchange(X25519PublicKey.from_public_bytes(bytes(requester_public_key)))
    nonce = _nonce if _nonce is not None else os.urandom(_NONCE_BYTES)
    plaintext = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ciphertext = ChaCha20Poly1305(_derive_key(shared, request_id)).encrypt(nonce, plaintext, request_id.encode("ascii"))
    return {
        "alg": ALG,
        "ephemeral_public_key": b64url_encode(
            ephemeral.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)),
        "nonce": b64url_encode(nonce),
        "ciphertext": b64url_encode(ciphertext),
    }


def unseal_token(*, private_key: X25519PrivateKey, request_id: str, sealed: dict[str, str]) -> dict[str, Any]:
    """Open a sealed blob with the requester's private key (the client's half; tests and tooling use it).

    Raises ``ValueError`` for an unknown algorithm or a malformed field, and
    ``cryptography.exceptions.InvalidTag`` for the wrong key, the wrong request id or any altered byte.
    """
    if sealed.get("alg") != ALG:
        raise ValueError(f"unsupported alg {sealed.get('alg')!r}")
    ephemeral = X25519PublicKey.from_public_bytes(b64url_decode(sealed["ephemeral_public_key"]))
    nonce = b64url_decode(sealed["nonce"])
    shared = private_key.exchange(ephemeral)
    plaintext = ChaCha20Poly1305(_derive_key(shared, request_id)).decrypt(
        nonce, b64url_decode(sealed["ciphertext"]), request_id.encode("ascii"))
    return json.loads(plaintext.decode("utf-8"))
