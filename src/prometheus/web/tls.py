"""When the daemon serves TLS on the home network, and where.

Pairing PR 5 (docs/design/lan-tls-pr5.md; contract docs/PAIRING-APPROVAL-API.md §7.3). TLS serves the LAN on its
own ports only in home-network mode: ``network.home_network: true`` and a bind that reaches beyond loopback. Plain
HTTP then stays on loopback (127.0.0.1, the existing ports), so this Mac's own clients (Beacon here, the app's
launcher, ``oara doctor``, curl) are unchanged, and nothing on the LAN is ever plain.

``network.home_network`` is #700's setting; its ``NetworkSettings`` becomes the one reader when it lands. Until then
this module reads that one key with #700's rule: ``true`` or ``false``, anything else is off.

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


@dataclass(frozen=True)
class LanTlsPlan:
    """The TLS listener set: on *host* (the LAN bind), REST on *api_port* and the WebSocket on *ws_port*."""

    host: str
    api_port: int
    ws_port: int
    certfile: Path
    keyfile: Path


def _home_network(config: Mapping[str, Any]) -> bool:
    network = config.get("network") if isinstance(config, Mapping) else None
    value = network.get("home_network") if isinstance(network, Mapping) else None
    if value is None:
        return False
    if not isinstance(value, bool):
        logger.warning("network.home_network must be true or false, got %r: treated as false (no TLS)", value)
        return False
    return value


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
    """The server side of the handshake: the instance certificate and key, TLS 1.2 or later."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(certfile=str(plan.certfile), keyfile=str(plan.keyfile))
    return context
