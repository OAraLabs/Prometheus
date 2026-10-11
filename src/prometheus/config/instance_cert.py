"""The certificate the daemon's home-network TLS listeners present: self-signed, over the instance key.

Pairing PR 5 (docs/design/lan-tls-pr5.md; contract docs/PAIRING-APPROVAL-API.md §7.3). A client pins the SHA-256 of
the certificate's SubjectPublicKeyInfo, all 32 bytes, read from the TLS handshake, and checks that hello's ``fp`` is
its first 16 hex characters. ``fp`` is already the fingerprint of the instance key (``config/instance_key.py``), so
the certificate is made over THAT key: the handshake's SPKI is then the instance key, and the pin, hello's ``fp``
and ``instance_public_key`` in a pairing answer are three views of one key.

Rules:

* The key is never generated or written here. ``ensure_instance_key`` owns it (and never regenerates it), so a
  re-issued certificate keeps the same pin.
* The certificate is public: ``instance.crt`` beside ``instance.key`` in the node directory, mode 0644, written
  through a temp file and a rename so a reader never sees half of it.
* Re-issued over the same key when it is missing, unreadable, made for another key, or within 30 days of expiry.
  Clients ignore names and dates, so a renewal changes nothing for them.
* Made at daemon boot, right after the key, never on a request.

Source: novel code for Prometheus, 2026-10-10.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from prometheus.config.instance_key import INSTANCE_KEY_FILENAME, ensure_instance_key
from prometheus.config.paths import get_node_dir, node_dir_path

logger = logging.getLogger(__name__)

CERT_FILENAME = "instance.crt"
VALIDITY = timedelta(days=5 * 365)
REISSUE_BEFORE = timedelta(days=30)
#: Not-before is set this far back, so a client whose clock runs a little behind still accepts the dates (it should
#: ignore them anyway).
BACKDATE = timedelta(days=1)


def cert_path() -> Path:
    """Where the certificate is. Creates nothing."""
    return node_dir_path() / CERT_FILENAME


def key_path() -> Path:
    return node_dir_path() / INSTANCE_KEY_FILENAME


def _spki_der(public_key: object) -> bytes:
    return public_key.public_bytes(  # type: ignore[attr-defined]
        encoding=serialization.Encoding.DER, format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def instance_spki_sha256() -> str:
    """The full pin: SHA-256 of the instance key's SPKI DER, 64 hex characters. Its first 16 are hello's ``fp``."""
    return hashlib.sha256(ensure_instance_key()).hexdigest()


def _names(fp: str) -> list[x509.GeneralName]:
    host = (socket.gethostname() or "localhost").split(".")[0] or "localhost"
    names = ["localhost", f"{host}.local", f"prometheus-{fp[:8]}.local"]
    return [x509.DNSName(name) for name in dict.fromkeys(names)]


def _issue(key: ec.EllipticCurvePrivateKey, now: datetime) -> x509.Certificate:
    fp = hashlib.sha256(_spki_der(key.public_key())).hexdigest()
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"Prometheus {fp[:16]}")])
    return (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(int.from_bytes(secrets.token_bytes(16), "big") >> 1)
        .not_valid_before(now - BACKDATE)
        .not_valid_after(now + VALIDITY)
        .add_extension(x509.SubjectAlternativeName(_names(fp)), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )


def _current(path: Path, spki: bytes, now: datetime) -> bool:
    """Whether the certificate on disk can stay: readable, over this key, and not within 30 days of expiry."""
    try:
        cert = x509.load_pem_x509_certificate(path.read_bytes())
    except (OSError, ValueError):
        return False
    return _spki_der(cert.public_key()) == spki and cert.not_valid_after_utc - now > REISSUE_BEFORE


def ensure_instance_certificate(*, now: datetime | None = None) -> Path:
    """The path of a current certificate over the instance key, issuing one if needed. Idempotent.

    Raises what ``ensure_instance_key`` raises for a key file that cannot be loaded (it is never regenerated).
    """
    now = now or datetime.now(timezone.utc)
    spki = ensure_instance_key()
    directory = get_node_dir()
    path = directory / CERT_FILENAME
    if _current(path, spki, now):
        return path
    key = serialization.load_pem_private_key((directory / INSTANCE_KEY_FILENAME).read_bytes(), password=None)
    if not isinstance(key, ec.EllipticCurvePrivateKey):
        raise RuntimeError(f"{directory / INSTANCE_KEY_FILENAME} is not an EC key; no certificate was made")
    pem = _issue(key, now).public_bytes(serialization.Encoding.PEM)
    temp = directory / f".{CERT_FILENAME}.{os.getpid()}.{secrets.token_hex(4)}"
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(fd, pem)
    finally:
        os.close(fd)
    os.chmod(temp, 0o644)      # the umask may have narrowed the create mode; a certificate is public
    try:
        os.replace(temp, path)
    except OSError:
        temp.unlink(missing_ok=True)
        raise
    logger.info("Instance certificate issued over the instance key (%s)", path)
    return path
