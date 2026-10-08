"""``config/pair_seal.py`` — sealing the token to the requester's key, and the match code.

Pairing hands a new device a bearer token over a network the contract assumes someone else can read, so
the token is sealed to a public key the device sent: nothing on the wire or at rest holds it in clear. The
recipe (docs/PAIRING-APPROVAL-API.md, 4.3 and 4.4) is fixed so a client written in another language can
implement the other half, and it is pinned three ways:

* against ``tests/vectors/pairing_seal_v1.json``, bytes produced by a separate implementation written from
  the recipe alone, which Beacon desktop and iOS check their own code against;
* against a reference rebuilt in this file from raw primitives, so the module and the recipe cannot drift
  together;
* by property: the wrong key, the wrong request id (it is both the HKDF salt and the AEAD associated
  data), or any flipped byte opens nothing.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from prometheus.config import pair_seal as ps

VECTORS = json.loads((Path(__file__).parent / "vectors" / "pairing_seal_v1.json").read_text())
REQUEST_ID = VECTORS["request_id"]


def _raw(public_key) -> bytes:
    return public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def _vector_parts():
    requester = X25519PrivateKey.from_private_bytes(bytes.fromhex(VECTORS["requester_private_hex"]))
    ephemeral = X25519PrivateKey.from_private_bytes(bytes.fromhex(VECTORS["ephemeral_private_hex"]))
    return requester, ephemeral, bytes.fromhex(VECTORS["nonce_hex"])


# ── base64url ────────────────────────────────────────────────────────────────

def test_base64url_is_unpadded_and_url_safe():
    raw = bytes(range(256))
    text = ps.b64url_encode(raw)
    assert "=" not in text and "+" not in text and "/" not in text
    assert ps.b64url_decode(text) == raw
    assert ps.b64url_encode(b"") == ""


@pytest.mark.parametrize("bad", ["a+b", "a/b", "a=b", "a b", "a\n", "ü", "====", "a"])
def test_base64url_decoding_is_strict(bad):
    with pytest.raises(ValueError):
        ps.b64url_decode(bad)


# ── the match code ───────────────────────────────────────────────────────────

def test_the_match_code_agrees_with_the_vector():
    requester, _, _ = _vector_parts()
    code = ps.match_code(_raw(requester.public_key()),
                         bytes.fromhex(VECTORS["instance_public_key_der_hex"]), REQUEST_ID)
    assert code == VECTORS["match_code"]


def test_the_match_code_is_the_documented_hash():
    requester, _, _ = _vector_parts()
    device, instance = _raw(requester.public_key()), bytes.fromhex(VECTORS["instance_public_key_der_hex"])
    digest = hashlib.sha256(b"prometheus-pair-v1" + b"\x00" + device + instance + REQUEST_ID.encode()).digest()
    assert ps.match_code(device, instance, REQUEST_ID) == f"{int.from_bytes(digest[:4], 'big') % 10000:04d}"


def test_the_match_code_is_four_digits_and_zero_padded():
    instance = bytes.fromhex(VECTORS["instance_public_key_der_hex"])
    codes = {ps.match_code(bytes([n]) * 32, instance, f"{n:032x}") for n in range(300)}
    assert all(len(code) == 4 and code.isdigit() for code in codes)
    assert any(code.startswith("0") for code in codes), "zero padding is exercised by 300 samples"


def test_the_match_code_depends_on_every_input():
    requester, _, _ = _vector_parts()
    device, instance = _raw(requester.public_key()), bytes.fromhex(VECTORS["instance_public_key_der_hex"])
    base = ps.match_code(device, instance, REQUEST_ID)
    other_device = ps.match_code(bytes(31) + b"\x01", instance, REQUEST_ID)
    other_instance = ps.match_code(device, instance[:-1] + bytes([instance[-1] ^ 1]), REQUEST_ID)
    other_id = ps.match_code(device, instance, "0" * 32)
    assert len({base, other_device, other_instance, other_id}) >= 3   # four values in 10,000: a collision is rare


# ── sealing ──────────────────────────────────────────────────────────────────

def test_the_seal_reproduces_the_vector_byte_for_byte():
    requester, ephemeral, nonce = _vector_parts()
    payload = json.loads(VECTORS["plaintext"])
    sealed = ps.seal_token(
        requester_public_key=_raw(requester.public_key()), request_id=REQUEST_ID, payload=payload,
        _ephemeral_private=ephemeral, _nonce=nonce)
    assert sealed == VECTORS["sealed"]
    assert sealed["alg"] == ps.ALG == "x25519-hkdf-sha256-chacha20poly1305"


def test_the_module_agrees_with_a_reference_rebuilt_from_primitives():
    requester, ephemeral, nonce = _vector_parts()
    shared = ephemeral.exchange(requester.public_key())
    assert shared.hex() == VECTORS["shared_secret_hex"]
    key = HKDF(algorithm=hashes.SHA256(), length=32, salt=REQUEST_ID.encode(),
               info=b"prometheus-pair-v1/seal").derive(shared)
    assert key.hex() == VECTORS["derived_key_hex"]
    expected = ChaCha20Poly1305(key).encrypt(nonce, VECTORS["plaintext"].encode(), REQUEST_ID.encode())
    sealed = ps.seal_token(
        requester_public_key=_raw(requester.public_key()), request_id=REQUEST_ID,
        payload=json.loads(VECTORS["plaintext"]), _ephemeral_private=ephemeral, _nonce=nonce)
    assert ps.b64url_decode(sealed["ciphertext"]) == expected


def test_the_requester_opens_what_the_vector_sealed():
    requester, _, _ = _vector_parts()
    opened = ps.unseal_token(private_key=requester, request_id=REQUEST_ID, sealed=VECTORS["sealed"])
    assert opened == json.loads(VECTORS["plaintext"])


def test_every_seal_is_fresh():
    requester = X25519PrivateKey.generate()
    one = ps.seal_token(requester_public_key=_raw(requester.public_key()), request_id=REQUEST_ID, payload={"a": 1})
    two = ps.seal_token(requester_public_key=_raw(requester.public_key()), request_id=REQUEST_ID, payload={"a": 1})
    assert one["ephemeral_public_key"] != two["ephemeral_public_key"]
    assert one["nonce"] != two["nonce"]
    assert one["ciphertext"] != two["ciphertext"]
    assert len(ps.b64url_decode(one["nonce"])) == 12
    assert len(ps.b64url_decode(one["ephemeral_public_key"])) == 32


def test_the_wrong_key_opens_nothing():
    requester, _, _ = _vector_parts()
    with pytest.raises(InvalidTag):
        ps.unseal_token(private_key=X25519PrivateKey.generate(), request_id=REQUEST_ID, sealed=VECTORS["sealed"])


def test_the_request_id_is_bound_to_the_seal():
    """It is the HKDF salt AND the associated data: a sealed blob lifted onto another request is dead."""
    requester, _, _ = _vector_parts()
    with pytest.raises(InvalidTag):
        ps.unseal_token(private_key=requester, request_id="f" * 32, sealed=VECTORS["sealed"])


@pytest.mark.parametrize("field", ["ephemeral_public_key", "nonce", "ciphertext"])
def test_a_flipped_byte_opens_nothing(field):
    requester, _, _ = _vector_parts()
    raw = bytearray(ps.b64url_decode(VECTORS["sealed"][field]))
    raw[0] ^= 1
    tampered = {**VECTORS["sealed"], field: ps.b64url_encode(bytes(raw))}
    with pytest.raises((InvalidTag, ValueError)):
        ps.unseal_token(private_key=requester, request_id=REQUEST_ID, sealed=tampered)


def test_an_unknown_algorithm_is_refused_not_guessed():
    requester, _, _ = _vector_parts()
    with pytest.raises(ValueError, match="alg"):
        ps.unseal_token(private_key=requester, request_id=REQUEST_ID, sealed={**VECTORS["sealed"], "alg": "rot13"})


# ── the requester's key ──────────────────────────────────────────────────────

def test_a_good_key_is_accepted():
    ps.check_requester_key(_raw(X25519PrivateKey.generate().public_key()))


@pytest.mark.parametrize("raw", [b"", b"\x01" * 31, b"\x01" * 33], ids=["empty", "31 bytes", "33 bytes"])
def test_a_key_of_the_wrong_length_is_refused(raw):
    with pytest.raises(ValueError):
        ps.check_requester_key(raw)


def test_a_low_order_point_is_refused_before_a_token_is_ever_sealed_to_it():
    """An all-zero public key makes the shared secret a constant anyone can compute."""
    for point in (bytes(32), bytes([1]) + bytes(31)):
        with pytest.raises(ValueError):
            ps.check_requester_key(point)
    assert isinstance(X25519PublicKey.from_public_bytes(bytes(32)), X25519PublicKey)  # the library alone allows it
