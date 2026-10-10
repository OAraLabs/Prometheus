"""The key this Prometheus presents to the devices that pair with it.

NOT the node key. ``config/node_identity.py`` holds an Ed25519 keypair that names the machine, under
rules of its own (identity, never encryption; nothing transmitted without opt-in), and nothing here
changes them. The pairing contract (docs/PAIRING-APPROVAL-API.md, section 7.3) needs a key that doubles
as the key of the TLS certificate a later change serves, and an Ed25519 certificate is not accepted by
every client's TLS stack. So this is a P-256 key beside the node key, in the same node directory
(per machine, never copied, untouched by ``reset-data``).

Its public half is what the next change hands a requesting device as ``instance_public_key`` and mixes
into the match code; its SHA-256 fingerprint (first 16 hex characters of the SubjectPublicKeyInfo DER)
is what ``GET /api/hello`` advertises as ``fp``. The fingerprint is a hint for display and for noticing
that a name now points at a different machine, never a trust anchor.

Two rules carry over from the node key, for the same reasons:

* An existing key file is NEVER regenerated. A new key is a new identity, and every paired device's pin
  would break; if the file cannot be loaded, or is not a P-256 key, this refuses and says why.
* The key is made at daemon boot (``ensure_instance_key``), not on a request. Reading it
  (``instance_public_key_der``, ``instance_fingerprint``) is what an unauthenticated GET does, and a GET
  must not write to disk or create a directory.

Source: novel code for Prometheus, 2026-10-08.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from prometheus.config.paths import get_node_dir, node_dir_path

logger = logging.getLogger(__name__)

INSTANCE_KEY_FILENAME = "instance.key"
FINGERPRINT_HEX_CHARS = 16


def _spki_der(key: ec.EllipticCurvePrivateKey) -> bytes:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _load(path: Path) -> ec.EllipticCurvePrivateKey:
    try:
        key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(
            f"{path} exists but cannot be loaded ({exc}) — refusing to regenerate over it. "
            "Regeneration is a NEW identity (every paired device's pin would break) and never "
            "happens implicitly; move the file aside deliberately if this machine should re-key."
        ) from exc
    if not (isinstance(key, ec.EllipticCurvePrivateKey) and isinstance(key.curve, ec.SECP256R1)):
        raise RuntimeError(
            f"{path} exists but is not a P-256 private key — refusing to touch it. An instance key "
            "is never regenerated in place; move the file aside deliberately if it is wrong."
        )
    return key


def fingerprint_of(public_key_der: bytes) -> str:
    """The display fingerprint of a SubjectPublicKeyInfo DER: first 16 hex characters of its SHA-256."""
    return hashlib.sha256(public_key_der).hexdigest()[:FINGERPRINT_HEX_CHARS]


def ensure_instance_key() -> bytes:
    """Load the instance key, generating it on first run, and return its public half (SPKI DER).

    Idempotent. 0600 from the first byte, so the file never exists world-readable, even briefly.
    Raises :class:`RuntimeError` for an existing file that is not a loadable P-256 key.
    """
    path = get_node_dir() / INSTANCE_KEY_FILENAME
    if path.exists():
        return _spki_der(_load(path))
    key = ec.generate_private_key(ec.SECP256R1())
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    # O_EXCL: two processes racing first-run must not both mint an identity; the loser reloads
    # the winner's key.
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return ensure_instance_key()
    with os.fdopen(fd, "wb") as fh:
        fh.write(pem)
    logger.info("Instance key generated (%s)", path)
    return _spki_der(key)


def instance_public_key_der() -> bytes | None:
    """The public half of the instance key, or ``None`` when there is none (or it cannot be read).

    Read path only: creates no file and no directory. An unreadable key is ``None`` here, and
    ``ensure_instance_key`` (daemon boot) is where it is reported.
    """
    path = node_dir_path() / INSTANCE_KEY_FILENAME
    try:
        if not path.exists():
            return None
        return _spki_der(_load(path))
    except (OSError, RuntimeError):
        logger.debug("instance key at %s could not be read", path, exc_info=True)
        return None


def instance_fingerprint() -> str:
    """The advertised ``fp``: 16 hex characters, or ``""`` when there is no readable key."""
    der = instance_public_key_der()
    return fingerprint_of(der) if der else ""
