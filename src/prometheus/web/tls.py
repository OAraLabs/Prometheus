"""When the daemon serves TLS on the home network, and where.

Pairing PR 5 (docs/design/lan-tls-pr5.md; contract docs/PAIRING-APPROVAL-API.md §7.3). TLS serves the LAN on its
own ports only in home-network mode: ``network.home_network: true`` and a bind that reaches beyond loopback. Plain
HTTP then stays on loopback (127.0.0.1, the existing ports), so this Mac's own clients (Beacon here, the app's
launcher, ``oara doctor``, curl) are unchanged, and nothing on the LAN is ever plain.

``network.home_network`` is #700's setting, read here through its ``NetworkSettings`` (the one reader).

Source: novel code for Prometheus, 2026-10-10.
"""

from __future__ import annotations

import logging
import ssl
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prometheus.config.instance_cert import cert_path, key_path
from prometheus.web.loopback import is_loopback_address

logger = logging.getLogger(__name__)

DEFAULT_TLS_API_PORT = 8006
DEFAULT_TLS_WS_PORT = 8011
#: Where plain HTTP listens when TLS serves the LAN.
LOOPBACK = "127.0.0.1"
#: The one cipher policy for both TLS listeners: the AEAD part of Python's default list (ECDHE with AES-GCM or
#: ChaCha20-Poly1305), nothing CBC. TLS 1.3's suites are AEAD by definition and not governed by this string.
AEAD_CIPHERS = "ECDH+AESGCM:ECDH+CHACHA20:!aNULL:!eNULL"
MINIMUM_TLS = ssl.TLSVersion.TLSv1_2


@dataclass(frozen=True)
class LanTlsPlan:
    """The TLS listener set: on *host* (the LAN bind), REST on *api_port* and the WebSocket on *ws_port*."""

    host: str
    api_port: int
    ws_port: int
    certfile: Path
    keyfile: Path


def _home_network(config: Mapping[str, Any]) -> bool:
    """``network.home_network``, read by #700's ``NetworkSettings``, the one reader (quiet: the daemon says once at
    boot when the switch is not a boolean)."""
    from prometheus.web.network import NetworkSettings

    return NetworkSettings.from_config(config, quiet=True).home_network


def _port(web: Mapping[str, Any], key: str, default: int) -> int:
    value = web.get(key) if isinstance(web, Mapping) else None
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        logger.warning("web.%s must be a port number (1-65535), got %r: using %d", key, value, default)
        return default
    return value


def lan_tls_plan(config: Mapping[str, Any], address: str) -> LanTlsPlan | None:
    """The TLS listener set for this config and resolved bind *address*, or None when TLS does not run.

    Pure: names the certificate and key files, never makes them (the daemon does at boot).
    """
    if is_loopback_address(address) or not _home_network(config):
        return None
    web = config.get("web") if isinstance(config, Mapping) else None
    web = web if isinstance(web, Mapping) else {}
    return LanTlsPlan(
        host=address,
        api_port=_port(web, "tls_api_port", DEFAULT_TLS_API_PORT),
        ws_port=_port(web, "tls_ws_port", DEFAULT_TLS_WS_PORT),
        certfile=cert_path(),
        keyfile=key_path(),
    )


def listener_hosts(address: str, plan: LanTlsPlan | None) -> tuple[str, str | None]:
    """``(plain host, TLS host)``. With a plan, plain HTTP is loopback only and TLS takes the LAN bind."""
    if plan is None:
        return address, None
    return LOOPBACK, plan.host


def server_ssl_context(plan: LanTlsPlan) -> ssl.SSLContext:
    """The WSS side of the handshake: the instance certificate and key, TLS 1.2 or later, AEAD suites only. The REST
    listener gets the same policy through uvicorn (``web/serving.serve_config``)."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = MINIMUM_TLS
    context.set_ciphers(AEAD_CIPHERS)
    context.load_cert_chain(certfile=str(plan.certfile), keyfile=str(plan.keyfile))
    return context


def advertised_endpoint(plan: LanTlsPlan | None, api_port: int) -> tuple[int, bool]:
    """``(port, tls)`` for the mDNS record: the TLS REST port with ``tls=1`` while TLS serves the LAN."""
    return (plan.api_port, True) if plan is not None else (api_port, False)


def tls_ready() -> bool:
    """Whether this daemon can serve TLS after a restart: an instance key it can read (the certificate is made
    over it at boot). Read-only: creates nothing."""
    from prometheus.config.instance_key import instance_public_key_der

    return instance_public_key_der() is not None


@dataclass(frozen=True)
class LanDecision:
    """What the listeners do with the home network. *loopback_only*: home network asked for and impossible, so
    nothing is served on the LAN. *plaintext_lan*: the owner's explicit opt-out, plain HTTP on the LAN."""

    plan: LanTlsPlan | None
    loopback_only: bool = False
    plaintext_lan: bool = False


def decide_lan(config: Mapping[str, Any], address: str, *, cert_ok: bool) -> LanDecision:
    """TLS on the LAN when it can be had; never plain HTTP in its place unless the owner asked for exactly that."""
    plan = lan_tls_plan(config, address)
    if plan is None:
        return LanDecision(None)
    if cert_ok:
        return LanDecision(plan)
    from prometheus.web.network import NetworkSettings

    if NetworkSettings.from_config(config, quiet=True).allow_plaintext_lan:
        return LanDecision(None, plaintext_lan=True)
    return LanDecision(None, loopback_only=True)

