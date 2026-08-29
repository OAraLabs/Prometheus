"""GRAFT-MOBILE-BRIDGE 2 — the APNs sender: JWT shape, caching, classification.

Needs ``cryptography`` (skips where absent — the dev venv; runs on the mini
and anywhere installed with the ``push`` extra). No network: the HTTP client
is a fake, and the signing key is a throwaway P-256 generated per test.
"""

from __future__ import annotations

import asyncio
import base64
import json
import types

import pytest

cryptography = pytest.importorskip("cryptography")

from prometheus.push.apns import (  # noqa: E402
    JWT_REFRESH_SECONDS,
    APNsSender,
    ApnsConfig,
    es256_jwt,
)


def _key_pem(tmp_path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    key = ec.generate_private_key(ec.SECP256R1())
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    path = tmp_path / "AuthKey_TEST.p8"
    path.write_bytes(pem)
    return path, key


def _config(tmp_path) -> ApnsConfig:
    path, _ = _key_pem(tmp_path)
    return ApnsConfig(key_path=path, key_id="KEYID12345",
                      team_id="53JM8W47RL", topic="com.oaralabs.beacon")


def _b64url_json(part: str) -> dict:
    return json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))


def test_jwt_has_apples_three_fields_and_verifies(tmp_path):
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

    path, key = _key_pem(tmp_path)
    token = es256_jwt(path.read_bytes(), "KEYID12345", "53JM8W47RL", now=1000.0)
    header_b64, claims_b64, sig_b64 = token.split(".")
    assert _b64url_json(header_b64) == {"alg": "ES256", "kid": "KEYID12345"}
    assert _b64url_json(claims_b64) == {"iss": "53JM8W47RL", "iat": 1000}

    raw = base64.urlsafe_b64decode(sig_b64 + "=" * (-len(sig_b64) % 4))
    assert len(raw) == 64, "JOSE signature is raw r||s, not DER"
    der = encode_dss_signature(int.from_bytes(raw[:32], "big"),
                               int.from_bytes(raw[32:], "big"))
    key.public_key().verify(der, f"{header_b64}.{claims_b64}".encode(),
                            ec.ECDSA(hashes.SHA256()))  # raises on mismatch


def test_provider_token_caches_until_the_50_minute_line(tmp_path):
    clock = {"t": 0.0}
    sender = APNsSender(_config(tmp_path), client=object(), now=lambda: clock["t"])
    first = sender.provider_token()
    clock["t"] = JWT_REFRESH_SECONDS - 1
    assert sender.provider_token() == first, "inside the window: cached"
    clock["t"] = JWT_REFRESH_SECONDS
    assert sender.provider_token() != first, "at the line: re-minted"


class _FakeHTTP:
    def __init__(self, status: int, body: dict | None = None) -> None:
        self.status = status
        self.body = body or {}
        self.posts: list[dict] = []

    async def post(self, url, headers=None, json=None):
        self.posts.append({"url": url, "headers": headers, "json": json})
        return types.SimpleNamespace(status_code=self.status,
                                     json=lambda: self.body)


def test_send_headers_and_outcome_classification(tmp_path):
    async def run():
        http = _FakeHTTP(200)
        sender = APNsSender(_config(tmp_path), client=http)
        result = await sender.send(apns_token="dead00beef", environment="sandbox",
                                   payload={"aps": {}})
        assert result.outcome == "ok"
        post = http.posts[0]
        assert post["url"].startswith("https://api.sandbox.push.apple.com/3/device/")
        assert post["headers"]["apns-topic"] == "com.oaralabs.beacon"
        assert post["headers"]["apns-push-type"] == "alert"
        assert post["headers"]["authorization"].startswith("bearer ")

        gone = APNsSender(_config(tmp_path),
                          client=_FakeHTTP(410, {"reason": "Unregistered"}))
        assert (await gone.send(apns_token="x", environment="production",
                                payload={})).outcome == "unregistered"

        flaky = APNsSender(_config(tmp_path), client=_FakeHTTP(503))
        assert (await flaky.send(apns_token="x", environment="production",
                                 payload={})).outcome == "failed"

    asyncio.run(run())


def test_config_validation_names_whats_missing(tmp_path):
    with pytest.raises(ValueError, match="key_id"):
        ApnsConfig.from_config({"apns": {"key_path": str(tmp_path / "nope.p8"),
                                         "team_id": "T", "topic": "b"}})
    with pytest.raises(ValueError, match="does not exist"):
        ApnsConfig.from_config({"apns": {"key_path": str(tmp_path / "nope.p8"),
                                         "key_id": "K", "team_id": "T", "topic": "b"}})
