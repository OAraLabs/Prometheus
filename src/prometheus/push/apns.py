"""APNs sender (GRAFT-MOBILE-BRIDGE 2).

ES256-signed provider JWT over HTTP/2, straight to Apple — no APNs library.
The JWT is hand-rolled on ``cryptography`` (25 lines beats a dependency), and
``httpx`` speaks HTTP/2 with the ``h2`` extra. Both imports are lazy: the
daemon runs fine without them until ``push.enabled`` is true, and the launcher
fails the boot LOUDLY when it is true and they are missing (config-dark law —
an enabled-but-broken feature must not silently do nothing).

Retry policy is NO RETRIES: every notification is one attempt, so a "retry
storm" can only be a signal storm, which the dispatcher owns. The sender only
classifies outcomes: ``unregistered`` (410 — the registration is dead,
permanently), ``failed`` (anything else non-2xx or a transport error), ``ok``.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Apple's provider-token lifetime cap is 60 minutes; refresh at 50 so a token
# is never presented near its edge.
JWT_REFRESH_SECONDS = 50 * 60

_HOSTS = {
    "production": "https://api.push.apple.com",
    "sandbox": "https://api.sandbox.push.apple.com",
}


@dataclass(frozen=True)
class ApnsConfig:
    key_path: Path
    key_id: str
    team_id: str
    topic: str

    @classmethod
    def from_config(cls, push_cfg: dict[str, Any]) -> "ApnsConfig":
        apns = (push_cfg or {}).get("apns") or {}
        key_path = Path(str(apns.get("key_path") or "")).expanduser()
        key_id = str(apns.get("key_id") or "")
        team_id = str(apns.get("team_id") or "")
        topic = str(apns.get("topic") or "")
        missing = [k for k, v in (("key_path", str(key_path) if apns.get("key_path") else ""),
                                  ("key_id", key_id), ("team_id", team_id),
                                  ("topic", topic)) if not v]
        if missing:
            raise ValueError(f"push.apns is missing {', '.join(missing)}")
        if not key_path.is_file():
            raise ValueError(f"push.apns.key_path does not exist: {key_path}")
        return cls(key_path=key_path, key_id=key_id, team_id=team_id, topic=topic)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def es256_jwt(key_pem: bytes, key_id: str, team_id: str,
              now: float | None = None) -> str:
    """A minimal ES256 JWT — exactly the three fields Apple reads."""
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import (
        decode_dss_signature,
    )

    header = _b64url(json.dumps({"alg": "ES256", "kid": key_id}).encode())
    claims = _b64url(json.dumps({"iss": team_id,
                                 "iat": int(now if now is not None else time.time())}).encode())
    signing_input = f"{header}.{claims}".encode()

    key = serialization.load_pem_private_key(key_pem, password=None)
    der = key.sign(signing_input, ec.ECDSA(hashes.SHA256()))
    # JOSE wants raw r||s (32 bytes each), not DER.
    r, s = decode_dss_signature(der)
    signature = r.to_bytes(32, "big") + s.to_bytes(32, "big")
    return f"{header}.{claims}.{_b64url(signature)}"


@dataclass(frozen=True)
class SendResult:
    outcome: str  # "ok" | "unregistered" | "failed"
    status: int | None = None
    reason: str = ""


class APNsSender:
    """One HTTP/2 client, one cached provider token, no retries."""

    def __init__(self, config: ApnsConfig, client: Any | None = None,
                 now: Any = time.time) -> None:
        self.config = config
        self._now = now
        self._jwt = ""
        self._jwt_minted_at = 0.0
        self._client = client  # injected in tests; real one built lazily
        self._key_pem = config.key_path.read_bytes()

    def provider_token(self) -> str:
        now = self._now()
        if not self._jwt or now - self._jwt_minted_at >= JWT_REFRESH_SECONDS:
            self._jwt = es256_jwt(self._key_pem, self.config.key_id,
                                  self.config.team_id, now=now)
            self._jwt_minted_at = now
        return self._jwt

    def _http(self) -> Any:
        if self._client is None:
            import httpx

            # http2=True needs the h2 extra; the launcher verified it at boot.
            self._client = httpx.AsyncClient(http2=True, timeout=10.0)
        return self._client

    async def send(self, *, apns_token: str, environment: str, payload: dict,
                   push_type: str = "alert", priority: int = 10,
                   topic: str | None = None, expiration: int = 0) -> SendResult:
        url = f"{_HOSTS.get(environment, _HOSTS['production'])}/3/device/{apns_token}"
        headers = {
            "authorization": f"bearer {self.provider_token()}",
            "apns-topic": topic or self.config.topic,
            "apns-push-type": push_type,
            "apns-priority": str(priority),
            "apns-expiration": str(expiration),
        }
        try:
            resp = await self._http().post(url, headers=headers, json=payload)
        except Exception as exc:
            return SendResult(outcome="failed", status=None, reason=str(exc))
        if resp.status_code == 200:
            return SendResult(outcome="ok", status=200)
        reason = ""
        try:
            reason = resp.json().get("reason", "")
        except Exception:
            pass
        if resp.status_code == 410 or reason == "Unregistered":
            return SendResult(outcome="unregistered", status=resp.status_code,
                              reason=reason or "Unregistered")
        return SendResult(outcome="failed", status=resp.status_code, reason=reason)

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
