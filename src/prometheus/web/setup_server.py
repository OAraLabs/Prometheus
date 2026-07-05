"""Setup-mode web server — the pairing handshake (Onboarding Phase 1).

``prometheus daemon`` with NO config used to be a dead end (nothing to
serve, nothing to pair against). Now it boots THIS minimal server
instead of the full daemon: a dedicated FastAPI app whose only live
surface is ``/api/setup/*`` — the full daemon route surface does not
exist here, so nothing else is reachable even by accident.

Flow:

1. Startup prints a 6-digit pairing code ONCE in a loud banner
   (style-matched to :func:`prometheus.config.api_token.format_minted_banner`).
2. A client (Beacon's "I have a pairing code" first-run path, or curl)
   POSTs ``/api/setup/pair`` with ``{"code": "123456"}``.
3. On success the daemon mints/persists the real API token via the
   Phase 0 machinery (:func:`ensure_api_token` → the env file), and
   returns it with the ports. When the user later finishes
   ``prometheus setup`` and starts the real daemon,
   ``resolve_api_token`` finds the SAME token — the paired client keeps
   working with no re-entry.

Pairing-code policy (documented decisions):

- **One-time use** — a successful pair consumes the code.
- **15-minute TTL** — an expired code is NOT re-minted in place;
  restart the daemon to mint a new one (keeps the "printed ONCE"
  guarantee simple and true).
- **Max 5 failed attempts** — then pairing reports ``"locked"`` until
  restart.
- Codes are compared with :func:`hmac.compare_digest`; neither the code
  nor the token is ever logged after the startup banner.

Setup mode creates NO ``~/.prometheus`` state. The only write it can
perform is the env-file token persist on a successful pair (the same
file ``prometheus token rotate`` writes). Port: ``web.api_port``'s
default (8005), overridable via ``PROMETHEUS_WEB_API_PORT`` (there is
no config to read a port from — that's the point).
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import secrets
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("prometheus.setup_mode")

# fastapi is the optional [web] extra. Imported at module level (not inside
# create_setup_app) because FastAPI resolves the postponed "Request"
# annotation against the endpoint function's __globals__ — a function-local
# import silently degrades it to a required query param. Guarded so that
# `from prometheus.web.setup_server import find_config_file` (the daemon's
# setup-mode gate) still works on a bare install; run_setup_mode() then
# fails loudly with install instructions instead of a traceback.
try:  # pragma: no cover — exercised implicitly by every test below
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover — bare install without [web]
    _FASTAPI_AVAILABLE = False

DEFAULT_API_PORT = 8005  # web.api_port default (config/prometheus.yaml.default)
DEFAULT_WS_PORT = 8010   # web.ws_port default

PAIRING_TTL_SECONDS = 15 * 60
PAIRING_MAX_ATTEMPTS = 5


def mint_pairing_code() -> str:
    """A 6-digit pairing code (leading zeros allowed, crypto-random)."""
    return f"{secrets.randbelow(1_000_000):06d}"


def resolve_setup_port() -> int:
    """The port setup mode serves on (no config exists to read one from)."""
    raw = os.environ.get("PROMETHEUS_WEB_API_PORT", "")
    try:
        return int(raw) if raw else DEFAULT_API_PORT
    except ValueError:
        logger.warning("PROMETHEUS_WEB_API_PORT=%r is not a port — using %d",
                       raw, DEFAULT_API_PORT)
        return DEFAULT_API_PORT


def resolve_setup_ws_port() -> int:
    """The WS port reported to paired clients (real daemon default)."""
    raw = os.environ.get("PROMETHEUS_WEB_WS_PORT", "")
    try:
        return int(raw) if raw else DEFAULT_WS_PORT
    except ValueError:
        return DEFAULT_WS_PORT


# ---------------------------------------------------------------------------
# Config detection — is there anything for the real daemon to run on?
# ---------------------------------------------------------------------------


def find_config_file(explicit: str | None = None) -> Path | None:
    """Locate prometheus.yaml WITHOUT creating any directories.

    Mirrors the daemon's search order (explicit ``--config`` →
    repo-local ``config/prometheus.yaml`` → ``$PROMETHEUS_CONFIG_DIR`` /
    ``~/.prometheus``) but never calls :func:`get_config_dir` — that
    helper ``mkdir``\\ s, and setup mode must not create state.
    """
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_file() else None
    repo_cfg = Path("config/prometheus.yaml")
    if repo_cfg.is_file():
        return repo_cfg
    env_dir = os.environ.get("PROMETHEUS_CONFIG_DIR")
    base = Path(env_dir).expanduser() if env_dir else Path.home() / ".prometheus"
    user_cfg = base / "prometheus.yaml"
    return user_cfg if user_cfg.is_file() else None


# ---------------------------------------------------------------------------
# Pairing state machine
# ---------------------------------------------------------------------------


@dataclass
class PairingState:
    """One pairing window: one code, one use, bounded attempts + TTL."""

    code: str = field(default_factory=mint_pairing_code)
    created_at: float = field(default_factory=time.time)
    ttl_seconds: float = PAIRING_TTL_SECONDS
    max_attempts: int = PAIRING_MAX_ATTEMPTS
    failed_attempts: int = 0
    used: bool = False

    # ``clock`` is injectable for TTL tests; production uses time.time.
    clock: Any = time.time

    def _expired(self) -> bool:
        return (self.clock() - self.created_at) > self.ttl_seconds

    @property
    def attempts_remaining(self) -> int:
        return max(0, self.max_attempts - self.failed_attempts)

    def status(self) -> str:
        """``"available"`` or ``"locked"`` (used/expired/too many attempts)."""
        if self.used or self._expired() or self.attempts_remaining <= 0:
            return "locked"
        return "available"

    def attempt(self, submitted: str) -> tuple[bool, str]:
        """Try a code. Returns ``(success, reason)``.

        reason ∈ {"ok", "invalid_code", "pairing_used", "pairing_expired",
        "pairing_locked"}. Only a WRONG code burns an attempt.
        """
        if self.used:
            return False, "pairing_used"
        if self._expired():
            return False, "pairing_expired"
        if self.attempts_remaining <= 0:
            return False, "pairing_locked"
        if hmac.compare_digest(str(submitted), self.code):
            self.used = True
            return True, "ok"
        self.failed_attempts += 1
        return False, "invalid_code"


# ---------------------------------------------------------------------------
# The FastAPI app
# ---------------------------------------------------------------------------


def create_setup_app(
    pairing: PairingState,
    *,
    api_port: int | None = None,
    ws_port: int | None = None,
):
    """Build the setup-mode FastAPI app.

    Deliberately NOT the full ``prometheus.web.server.create_app`` with a
    flag: the real route surface must be unreachable in setup mode, so
    it simply is not mounted. Routes:

    - ``GET /api/setup/status`` → mode + pairing availability
    - ``POST /api/setup/pair``  → code → real API token
    - anything else            → 403 with an honest JSON body
    """
    from prometheus import __version__

    api_port = api_port if api_port is not None else resolve_setup_port()
    ws_port = ws_port if ws_port is not None else resolve_setup_ws_port()

    app = FastAPI(
        title="Prometheus (setup mode)",
        docs_url=None, redoc_url=None, openapi_url=None,
    )

    @app.get("/api/setup/status")
    async def setup_status() -> dict[str, Any]:
        return {
            "setup_mode": True,
            "configured": False,
            "pairing": pairing.status(),
            "version": __version__,
        }

    @app.post("/api/setup/pair")
    async def setup_pair(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            body = None
        if not isinstance(body, dict) or not str(body.get("code") or "").strip():
            return JSONResponse(status_code=400, content={
                "error": "bad_request",
                "detail": 'expected a JSON body: {"code": "<6 digits>"}',
            })
        ok, reason = pairing.attempt(str(body["code"]).strip())
        if ok:
            from prometheus.config.api_token import ensure_api_token
            from prometheus.config.env_file import get_env_file_path

            token, minted = ensure_api_token(None)
            logger.info(
                "Pairing successful — API token %s and persisted to %s "
                "(token value never logged)",
                "minted" if minted else "reused",
                get_env_file_path(),
            )
            return JSONResponse(status_code=200, content={
                "token": token,
                "api_base_port": api_port,
                "ws_port": ws_port,
            })
        if reason == "invalid_code":
            logger.warning(
                "Pairing attempt failed (wrong code) — %d attempt(s) remaining",
                pairing.attempts_remaining,
            )
            return JSONResponse(status_code=401, content={
                "error": "invalid_code",
                "attempts_remaining": pairing.attempts_remaining,
                "detail": f"wrong pairing code — "
                          f"{pairing.attempts_remaining} attempt(s) remaining",
            })
        detail = {
            "pairing_used": "this pairing code was already used — restart the "
                            "daemon to mint a new one",
            "pairing_expired": "the pairing code expired (15 min) — restart "
                               "the daemon to mint a new one",
            "pairing_locked": "pairing is locked (too many failed attempts) — "
                              "restart the daemon to re-enable it",
        }[reason]
        logger.warning("Pairing attempt rejected: %s", reason)
        return JSONResponse(status_code=403, content={
            "error": reason, "detail": detail,
        })

    # Everything else — the whole real daemon surface — does not exist
    # here. One honest answer for all of it.
    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
    )
    async def setup_mode_catch_all(path: str) -> JSONResponse:
        return JSONResponse(status_code=403, content={
            "error": "setup_mode",
            "detail": "the daemon is in setup mode — only /api/setup/* is "
                      "available. Finish `prometheus setup` on the host, "
                      "then restart `prometheus daemon`.",
        })

    return app


# ---------------------------------------------------------------------------
# Banner + entry point
# ---------------------------------------------------------------------------


def format_pairing_banner(code: str, api_port: int) -> str:
    """The print-ONCE pairing banner (style: format_minted_banner)."""
    import socket

    host = socket.gethostname()
    bar = "=" * 68
    return (
        f"\n{bar}\n"
        "  PROMETHEUS IS IN SETUP MODE — no configuration found\n"
        "\n"
        "  Pairing code (printed once — valid 15 min, one client, 5 tries):\n"
        f"\n    {code}\n\n"
        f"  Pair from Beacon: address {host}:{api_port} (or this machine's\n"
        "  Tailscale / LAN address) + the code above.\n"
        "  Or set up here instead:  prometheus setup\n"
        "  Expired or locked? Restart `prometheus daemon` for a new code.\n"
        f"{bar}\n"
    )


async def _serve_setup_mode(pairing: PairingState, api_port: int) -> None:
    """Run uvicorn until SIGTERM/SIGINT; shuts down cleanly."""
    import uvicorn

    app = create_setup_app(pairing, api_port=api_port)
    config = uvicorn.Config(app, host="0.0.0.0", port=api_port, log_level="info")
    server = uvicorn.Server(config)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        # uvicorn installs its own handlers only when run() owns the loop;
        # serving inside our loop means we wire shutdown explicitly.
        loop.add_signal_handler(
            sig, lambda s=sig: setattr(server, "should_exit", True)
        )
    await server.serve()


def run_setup_mode() -> int:
    """Entry point: `prometheus daemon` found no config. Returns exit code.

    Boots the pairing-only server, prints the pairing code banner ONCE,
    serves until SIGTERM/SIGINT. Creates no ~/.prometheus state; the
    only possible write is the env-file token persist on a successful
    pair.
    """
    if not _FASTAPI_AVAILABLE:
        print(
            "No configuration found, and setup mode needs the web extra to "
            "serve the pairing API.\n"
            "Either install it:   pip install 'oara-prometheus[web]'\n"
            "or set up directly:  prometheus setup",
        )
        return 1

    api_port = resolve_setup_port()
    pairing = PairingState()

    logger.warning(
        "No prometheus.yaml found — starting in SETUP MODE (pairing-only "
        "API on :%d; the full daemon surface is NOT running). Run "
        "`prometheus setup` to configure, then restart the daemon.",
        api_port,
    )
    print(format_pairing_banner(pairing.code, api_port), flush=True)

    try:
        asyncio.run(_serve_setup_mode(pairing, api_port))
    except KeyboardInterrupt:  # pragma: no cover — belt and braces
        pass
    logger.info("Setup-mode server stopped.")
    return 0
