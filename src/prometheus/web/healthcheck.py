"""``python -m prometheus.web.healthcheck`` — is the daemon on this machine answering?

The container image's ``HEALTHCHECK``, and usable by any supervisor that runs a
command: exit 0 healthy, 1 unhealthy, one line saying why (``docker inspect``
keeps the last few).

ONE probe for both of the daemon's modes, because a fresh container is in the
first and later falls through into the second in the same process:

* setup mode (no config yet) serves ``GET /api/setup/status`` →
  ``{"setup_mode": true, ...}`` and no ``/health``;
* the configured daemon serves ``GET /health`` → ``{"status": "ok", ...}``
  and no ``/api/setup/*``.

So ``/health`` is asked first and setup status second, on the same address.
(Setup mode answers ``/health`` itself, with its catch-all's ``403``.)
Both are unauthenticated routes, and the probe sends no credentials.

WHERE TO ASK comes from the daemon's own resolvers, never a copy of them (the
config search order has already drifted once, see
:func:`prometheus.web.setup_server.find_config_file`):

* a config is found → its ``web.api_port`` (default 8005) and
  :func:`prometheus.web.bind.resolve_bind`;
* no config → setup mode's :func:`resolve_setup_port` (``PROMETHEUS_WEB_API_PORT``)
  and :func:`resolve_setup_choice`.

A wildcard bind (``0.0.0.0``, ``::``) is asked on the loopback address of the
same family. A ``--bind`` passed on the daemon's command line is not visible
from here; set ``PROMETHEUS_WEB_BIND`` instead when the probe must follow it.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

#: Per request. Docker's own --timeout (5 s in the image) bounds the whole probe.
_TIMEOUT_SECONDS = 2.0

#: Never through a proxy. urllib honours HTTP_PROXY, and a container whose egress goes through one would
#: send this loopback probe to the proxy, which cannot reach the daemon's loopback at all.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@dataclass(frozen=True)
class ProbeTarget:
    host: str
    port: int

    def url(self, path: str) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}{path}"


def _loopback_for(address: str) -> str:
    from prometheus.web.bind import is_all_interfaces

    if is_all_interfaces(address):
        return "::1" if ":" in address else "127.0.0.1"
    return address


def _read_config() -> dict[str, Any] | None:
    """The config the daemon would boot from, or None (setup mode). Creates nothing."""
    from prometheus.web.setup_server import find_config_file

    path = find_config_file()
    if path is None:
        return None
    import yaml

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def probe_target() -> ProbeTarget:
    """The address and port the daemon on this machine is listening on."""
    from prometheus.web.bind import resolve_bind
    from prometheus.web.setup_server import (
        DEFAULT_API_PORT,
        resolve_setup_choice,
        resolve_setup_port,
    )

    config = _read_config()
    if config is None:
        return ProbeTarget(_loopback_for(resolve_setup_choice().address), resolve_setup_port())
    web = config.get("web")
    if not isinstance(web, dict):
        web = {}
    try:
        port = int(web.get("api_port") or DEFAULT_API_PORT)
    except (TypeError, ValueError):
        port = DEFAULT_API_PORT
    return ProbeTarget(_loopback_for(resolve_bind(config).address), port)


def _get_json(url: str) -> tuple[int, Any]:
    """``(status, parsed body)``; the body is None when it is not JSON."""
    try:
        with _OPENER.open(url, timeout=_TIMEOUT_SECONDS) as resp:
            status, raw = resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        status, raw = exc.code, exc.read()
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, None


def check(target: ProbeTarget) -> tuple[bool, str]:
    """``(healthy, one line saying why)``.

    A configured daemon that answers ``/health`` with anything but ``200 ok`` is
    unhealthy, full stop. Anything else on ``/health`` — setup mode's catch-all
    answers every unknown path ``403 {"error": "setup_mode"}`` — sends the probe
    on to ``/api/setup/status``, which only setup mode serves.
    """
    try:
        health_status, body = _get_json(target.url("/health"))
        if health_status == 200:
            if isinstance(body, dict) and body.get("status") == "ok":
                return True, f"ok: daemon answering on {target.host}:{target.port}"
            return False, f"/health answered 200 without status ok: {str(body)[:120]}"
        setup_status, body = _get_json(target.url("/api/setup/status"))
    except (OSError, urllib.error.URLError) as exc:
        reason = getattr(exc, "reason", exc)
        return False, f"nothing answering on {target.host}:{target.port}: {reason}"
    if setup_status == 200 and isinstance(body, dict) and body.get("setup_mode") is True:
        return True, f"ok: setup mode answering on {target.host}:{target.port} (waiting to be paired)"
    return False, (f"neither mode is healthy: /health answered {health_status}, "
                   f"/api/setup/status answered {setup_status}")


def main(argv: list[str] | None = None) -> int:
    del argv  # no options: everything comes from the daemon's own configuration
    try:
        target = probe_target()
    except Exception as exc:  # a bad bind or port refuses the daemon too; say so instead of a traceback
        print(f"unhealthy: cannot work out where the daemon listens: {exc}")
        return 1
    healthy, why = check(target)
    print(why if healthy else f"unhealthy: {why}")
    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
