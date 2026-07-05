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

import os
import secrets
from typing import Any

from prometheus.config.env_file import (
    get_env_file_path,
    parse_env_file,
    set_env_value,
)

TOKEN_ENV_VAR = "PROMETHEUS_API_TOKEN"


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
    """True when the operator explicitly configured an EMPTY token."""
    web_cfg = (config or {}).get("web") or {}
    if "api_token" in web_cfg and not web_cfg.get("api_token"):
        return True
    if TOKEN_ENV_VAR in os.environ and not os.environ[TOKEN_ENV_VAR]:
        return True
    file_values = parse_env_file()
    if TOKEN_ENV_VAR in file_values and not file_values[TOKEN_ENV_VAR]:
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


def rotate_api_token() -> str:
    """Mint a fresh token, persist it to the env file, and return it."""
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
        "  Show it again:  prometheus token show\n"
        "  Rotate it:      prometheus token rotate\n"
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
    mint + ``prometheus token show``. The token is never *written*
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
            "              `prometheus token show`\n"
        )
    bar = "=" * 68
    return (
        f"\n{bar}\n"
        "  CONNECT A CLIENT (Beacon)\n"
        "\n"
        f"    Address:  {host}:{api_port}\n"
        f"              (or this machine's Tailscale / LAN address, port {api_port})\n"
        f"{token_lines}"
        f"{bar}\n"
    )
