"""``config/instance_key.py`` — the key this Prometheus presents to the devices that pair with it.

Not the node key. ``config/node_identity.py`` holds an Ed25519 keypair that names the machine, with rules
of its own (identity, never encryption; nothing transmitted without opt-in). The pairing contract needs a
key that doubles as the TLS certificate's key (section 7.3), and an Ed25519 certificate is not accepted
by every client's TLS stack, so this is a P-256 key beside it, in the same node directory.

Pinned here: it is P-256, 0600, never regenerated over an existing file, refused loudly when the file is
not a P-256 key, and its fingerprint is the first 16 hex characters of SHA-256 of the SubjectPublicKeyInfo
DER, which is what hello advertises as ``fp``. Reading the fingerprint NEVER creates the key: hello is
unauthenticated, and a GET must not write to disk.
"""

from __future__ import annotations

import ast
import hashlib
import os
import stat
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519

from prometheus.config import instance_key as ik
from prometheus.config.paths import get_node_dir


def _node_dir() -> Path:
    """The node dir WITHOUT creating it (get_node_dir() mkdirs)."""
    return Path(os.environ["PROMETHEUS_CONFIG_DIR"]) / "node"


def test_nothing_exists_until_it_is_asked_for():
    assert ik.instance_public_key_der() is None
    assert ik.instance_fingerprint() == ""
    assert not _node_dir().exists(), "reading the key must not create the node directory"


def test_ensure_generates_a_p256_key_that_only_its_owner_can_read():
    der = ik.ensure_instance_key()
    path = get_node_dir() / ik.INSTANCE_KEY_FILENAME
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    assert isinstance(key, ec.EllipticCurvePrivateKey)
    assert isinstance(key.curve, ec.SECP256R1)
    assert key.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo) == der


def test_ensure_is_idempotent_and_never_rekeys():
    first = ik.ensure_instance_key()
    path = get_node_dir() / ik.INSTANCE_KEY_FILENAME
    before = path.read_bytes()
    assert ik.ensure_instance_key() == first
    assert path.read_bytes() == before


def test_the_fingerprint_is_16_hex_of_sha256_of_the_spki_der():
    der = ik.ensure_instance_key()
    fp = ik.instance_fingerprint()
    assert fp == hashlib.sha256(der).hexdigest()[:16]
    assert len(fp) == 16 and fp == fp.lower()
    assert ik.fingerprint_of(der) == fp
    assert ik.instance_public_key_der() == der


def test_it_lives_in_the_node_directory_beside_the_node_key():
    ik.ensure_instance_key()
    assert (get_node_dir() / ik.INSTANCE_KEY_FILENAME).is_file()
    assert ik.INSTANCE_KEY_FILENAME != "node.key"


def test_an_unreadable_key_file_is_refused_never_replaced():
    path = get_node_dir() / ik.INSTANCE_KEY_FILENAME
    path.write_bytes(b"not a pem")
    with pytest.raises(RuntimeError, match="refus"):
        ik.ensure_instance_key()
    assert path.read_bytes() == b"not a pem"


def test_a_key_of_another_type_is_refused_never_replaced():
    path = get_node_dir() / ik.INSTANCE_KEY_FILENAME
    pem = ed25519.Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    path.write_bytes(pem)
    with pytest.raises(RuntimeError, match="P-256"):
        ik.ensure_instance_key()
    assert path.read_bytes() == pem


def test_reading_a_broken_key_gives_no_fingerprint_rather_than_a_traceback():
    """Hello calls this on every request; a bad file must read as 'no fingerprint', not a 500."""
    (get_node_dir() / ik.INSTANCE_KEY_FILENAME).write_bytes(b"garbage")
    assert ik.instance_fingerprint() == ""
    assert ik.instance_public_key_der() is None


def test_the_daemon_makes_the_key_at_boot():
    """The key is made where the daemon makes the node key, not on an unauthenticated request."""
    import prometheus.daemon as daemon

    tree = ast.parse(Path(daemon.__file__).read_text(encoding="utf-8"))
    called = {
        (node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None))
        for node in ast.walk(tree) if isinstance(node, ast.Call)
    }
    assert "ensure_instance_key" in called
