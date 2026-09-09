"""Web API token bootstrap — mint, persist, and describe auth state.

Onboarding Phase 0: the wizard writes ``web.enabled: true``, so a fresh
install serves the REST/WS control plane on first daemon start. That
surface must not come up open-by-accident: if no token is configured
anywhere, the daemon mints one here, persists it to the env file (the
same ``~/.config/prometheus/env`` the systemd unit loads), and prints it
ONCE loudly in the startup banner.

Deliberately-open remains possible: an explicit empty
``PROMETHEUS_API_TOKEN=`` line in the env file (or an empty value already
in the environment) is respected as "auth OFF" — matching the existing
REST/WS convention that an empty token disables auth.
"""

from __future__ import annotations

from dataclasses import dataclass as _dataclass

import os
import secrets
from typing import Any

from prometheus.config.env_file import (
    get_env_file_path,
    parse_env_file,
    set_env_value,
)

TOKEN_ENV_VAR = "PROMETHEUS_API_TOKEN"

# Where a new user downloads the Beacon desktop client (prebuilt releases —
# Onboarding Phase 3). A public repo URL, not an infra host, so it is safe to
# print anywhere.
BEACON_DOWNLOAD_URL = "https://github.com/OAraLabs/beacon-desktop/releases"


def mint_api_token() -> str:
    """Generate a new secure API token."""
    return secrets.token_urlsafe(32)


def resolve_api_token(config: dict[str, Any] | None = None) -> tuple[str, str]:
    """Resolve the effective API token.

    Returns ``(token, source)`` where source is one of ``"config"``,
    ``"env"``, ``"env-file"``, or ``""`` when no token is set anywhere.
    Mirrors the precedence the web server/WS bridge use:
    ``web.api_token`` (config) first, then the environment.
    """
    cfg_token = ((config or {}).get("web") or {}).get("api_token") or ""
    if cfg_token:
        return str(cfg_token), "config"
    env_token = os.environ.get(TOKEN_ENV_VAR, "")
    if env_token:
        return env_token, "env"
    file_token = parse_env_file().get(TOKEN_ENV_VAR, "")
    if file_token:
        return file_token, "env-file"
    return "", ""


def _deliberately_open(config: dict[str, Any] | None) -> bool:
    """True ONLY when the operator EXPLICITLY chose an empty token.

    The distinction that closes the audit's critical finding: the shipped
    template carries ``web.api_token:`` which YAML parses as present-with-a
    NULL value. ``None`` is "not configured" — the operator copied the
    template verbatim and expects the documented first-start mint. An
    explicit empty STRING (``api_token: ""`` here, or ``PROMETHEUS_API_TOKEN=``
    in the env file / environment) is the deliberate-open channel the module
    docstring and the template comment both name.

    Treating ``None`` as deliberate is what booted an unauthenticated control
    plane on every verbatim template copy: REST on :8005, WS on :8010 and the
    OpenAI-compatible surface — every tool, including bash — on 0.0.0.0 with no
    token. ``None`` now mints; only an explicit ``""`` stays open.
    """
    web_cfg = (config or {}).get("web") or {}
    # Explicit empty STRING in config only. A null/absent value is NOT a
    # choice — it is the template default and must mint.
    if web_cfg.get("api_token", None) == "":
        return True
    if TOKEN_ENV_VAR in os.environ and os.environ[TOKEN_ENV_VAR] == "":
        return True
    file_values = parse_env_file()
    if file_values.get(TOKEN_ENV_VAR, None) == "":
        return True
    return False


def ensure_api_token(config: dict[str, Any] | None = None) -> tuple[str, bool]:
    """Return ``(token, minted)`` — minting + persisting when unset.

    - A token configured anywhere → returned as-is, ``minted=False``.
    - Explicitly blank (see :func:`_deliberately_open`) → ``("", False)``
      — the operator chose an open API; we respect it (but log loudly).
    - Otherwise → mint, persist to the env file, export into
      ``os.environ`` so the web launcher picks it up this same process,
      and return ``(token, True)``.
    """
    token, _source = resolve_api_token(config)
    if token:
        # Make sure the running process env carries it (web launcher +
        # WS bridge read the environment when config.web.api_token is unset).
        os.environ.setdefault(TOKEN_ENV_VAR, token)
        return token, False
    if _deliberately_open(config):
        return "", False
    token = mint_api_token()
    set_env_value(TOKEN_ENV_VAR, token)
    os.environ[TOKEN_ENV_VAR] = token
    return token, True


def web_refused_on_bootstrap_failure(
    config: dict[str, Any] | None,
    *,
    bootstrap_raised: bool,
    environ: dict[str, str] | None = None,
) -> bool:
    """True when the web plane must NOT serve.

    The audit's second open door: when :func:`ensure_api_token` RAISES — an
    unwritable env-file dir (permissions, full disk, read-only HOME) so a
    freshly-minted token cannot persist — the old daemon logged one ERROR line
    and launched the bridge anyway. A web-enabled default install came up
    unauthenticated (bash reachable on 0.0.0.0), the only signal an INFO
    "web auth: OPEN".

    Refusing to serve beats serving open. This is DISTINCT from
    deliberate-open: an explicit empty token resolves cleanly to ``""`` with no
    exception (:func:`_deliberately_open`), and that operator choice still
    serves open. A bootstrap RAISE means we could not establish auth at all, so
    there is nothing to honour — only to refuse.

    ``environ`` is injectable so the decision is testable without a daemon boot.
    """
    if not bootstrap_raised:
        return False
    env = os.environ if environ is None else environ
    cfg_token = ((config or {}).get("web") or {}).get("api_token")
    return not (cfg_token or env.get(TOKEN_ENV_VAR))


class TokenRotationBlocked(RuntimeError):
    """The effective token cannot be rotated where it actually lives (#320).

    Rotation persists to the env file, but :func:`resolve_api_token` — and
    with it ``create_app``/``launcher`` — reads ``web.api_token`` from
    ``prometheus.yaml`` FIRST. When a token is pinned there, writing the env
    file changes nothing the daemon reads: the command used to report success,
    hand back a token that would never authenticate, and leave the old one
    (in the incident that opened #320, a leaked one) valid across restarts.

    Refusing is the honest outcome rather than rotating the YAML in place:
    the env file is where this project keeps secrets BY DESIGN (see
    ``config/env_file.py``), the daemon never writes ``prometheus.yaml``, and
    that file is routinely version-controlled — silently minting a secret
    into it would trade one quiet failure for a worse one.

    Carries the remedy, not just the complaint: ``str(exc)`` names the key,
    the file, and the two steps that make rotation effective.
    """

    def __init__(self, source: str, config_path: str | None = None) -> None:
        self.source = source
        self.config_path = config_path or "prometheus.yaml"
        super().__init__(
            f"the active web API token is pinned in {self.config_path} "
            f"(web.api_token), which the daemon reads BEFORE the env file — "
            f"writing a new token to {get_env_file_path()} would leave the "
            f"current one live. Remove or blank the web.api_token line in "
            f"{self.config_path} (the env file then governs), restart the "
            f"daemon, and rotate again."
        )


def rotate_api_token(config: dict[str, Any] | None = None) -> str:
    """Mint a fresh token, persist it to the env file, and return it.

    Pass the loaded ``config`` so rotation can see a YAML-pinned token and
    refuse instead of no-opping — :class:`TokenRotationBlocked`. Called with
    no config (the pre-#320 signature) it can only see the env layers, so it
    rotates them; every in-tree caller passes config.

    The returned token is always one the daemon will accept after a restart.
    """
    _token, source = resolve_api_token(config)
    if source == "config":
        from prometheus.config.defaults import resolve_config_path

        raise TokenRotationBlocked(source, str(resolve_config_path()))
    token = mint_api_token()
    set_env_value(TOKEN_ENV_VAR, token)
    os.environ[TOKEN_ENV_VAR] = token
    return token


def describe_web_auth(config: dict[str, Any] | None = None) -> str:
    """One-line auth status for the startup log (never includes the token)."""
    token, source = resolve_api_token(config)
    if token:
        return f"web auth: ENABLED (token from {source})"
    return "web auth: OPEN — no token set (anyone who can reach the port has control)"


def format_minted_banner(token: str) -> str:
    """The print-ONCE banner shown when a token was just minted."""
    env_path = get_env_file_path()
    bar = "=" * 68
    return (
        f"\n{bar}\n"
        "  NEW WEB API TOKEN GENERATED (printed once — copy it now)\n"
        f"\n    {token}\n\n"
        f"  Saved to: {env_path}\n"
        "  Clients (Beacon, curl) must send: Authorization: Bearer <token>\n"
        "  Show it again:  oara token show\n"
        "  Rotate it:      oara token rotate\n"
        f"{bar}\n"
    )


def format_connect_client_block(
    config: dict[str, Any] | None = None,
    api_port: int | None = None,
) -> str:
    """The wizard's "Connect a client (Beacon)" end screen (Phase 1, item 1).

    Everything a client needs, in one boxed block: the address (this
    machine's hostname + the web API port — never a hardcoded host) and
    the API token. The token value is printed only when one already
    exists (config/env/env file); otherwise we point at the first-start
    mint + ``oara token show``. The token is never *written*
    anywhere here — printing only.
    """
    import socket

    host = socket.gethostname()
    if api_port is None:
        try:
            api_port = int(((config or {}).get("web") or {}).get("api_port") or 8005)
        except (TypeError, ValueError):
            api_port = 8005
    token, _source = resolve_api_token(config)
    if token:
        token_lines = (
            f"    Token:    {token}\n"
            f"              (stored in {get_env_file_path()})\n"
        )
    else:
        token_lines = (
            "    Token:    minted on first daemon start — re-print with\n"
            "              `oara token show`\n"
        )
    bar = "=" * 68
    return (
        f"\n{bar}\n"
        "  CONNECT A CLIENT (Beacon)\n"
        "\n"
        "    Beacon is the desktop cockpit for this daemon (chat, coding\n"
        "    runs, documents, dashboards). Get it:\n"
        f"      {BEACON_DOWNLOAD_URL}\n"
        "\n"
        f"    Address:  {host}:{api_port}\n"
        f"              (or this machine's Tailscale / LAN address, port {api_port})\n"
        f"{token_lines}"
        f"{bar}\n"
    )


# ---------------------------------------------------------------------------
# Token verification (GRAFT-MOBILE-BRIDGE 1: per-device tokens)
# ---------------------------------------------------------------------------

@_dataclass(frozen=True)
class DeviceIdentity:
    """Who a presented bearer token belongs to. ``id == "global"`` is the
    shared daemon token (and the identity used when auth is disabled)."""

    id: str
    name: str = ""
    platform: str = ""

    @property
    def is_global(self) -> bool:
        return self.id == "global"


GLOBAL_IDENTITY = DeviceIdentity(id="global", name="global", platform="")


def verify_token(presented: str, global_token: str, store=None) -> DeviceIdentity | None:
    """Resolve a presented bearer token to an identity, or None.

    Constant-time compare against the global token first (both REST and WS
    route through here, which is what fixed the REST side's ``!=`` compare).
    Then the device registry: SHA-256 the presented token and look up a live
    row; a hit stamps ``last_seen_at`` (throttled inside the store). A miss —
    including a revoked device — is None.

    ``auth_required`` stays the caller's ``bool(global_token)``: with no
    global token the daemon is deliberately open and device tokens are simply
    unused. This function is only meaningful when a token was presented.
    """
    import hmac as _hmac

    if not isinstance(presented, str) or not presented:
        return None
    if global_token and _hmac.compare_digest(presented, global_token):
        return GLOBAL_IDENTITY
    if store is not None:
        from prometheus.config.device_store import token_digest

        row = store.lookup(token_digest(presented))
        if row is not None:
            store.touch(row.id)
            return DeviceIdentity(id=row.id, name=row.name, platform=row.platform)
    return None
