"""Setup-mode web server — pairing + remote-drivable setup (Phases 1–2).

``prometheus daemon`` with NO config used to be a dead end (nothing to
serve, nothing to pair against). Now it boots THIS minimal server
instead of the full daemon: a dedicated FastAPI app whose only live
surface is ``/api/setup/*`` — the full daemon route surface does not
exist here, so nothing else is reachable even by accident.

Phase 1 flow (pairing):

1. Startup prints a 6-digit pairing code ONCE in a loud banner
   (style-matched to :func:`prometheus.config.api_token.format_minted_banner`).
2. A client (Beacon's "I have a pairing code" first-run path, or curl)
   POSTs ``/api/setup/pair`` with ``{"code": "123456"}``.
3. On success the daemon mints/persists the real API token via the
   Phase 0 machinery (:func:`ensure_api_token` → the env file), and
   returns it with the ports. ``resolve_api_token`` later finds the
   SAME token — the paired client keeps working with no re-entry.

Phase 2 flow (the paired client completes the ENTIRE setup remotely):

4. ``GET /api/setup/detect`` (authed) — server-side probe of local
   inference backends (llama.cpp/Ollama/LM Studio/vLLM), reusing the
   CLI wizard's detection (:func:`prometheus.cli.init.detect_local_servers`
   — JSON-models-shape hardening included). ``?base_url=`` probes ONE
   remote/custom URL instead.
5. ``POST /api/setup/configure`` (authed) — validates the chosen
   backend by re-probing it, then writes ``prometheus.yaml`` via the
   SAME writer ``prometheus setup --fast`` uses
   (:func:`prometheus.cli.init._default_config` + ``write_config``),
   generates identity (SOUL.md/AGENTS.md) via the SetupWizard's
   generator when ``agent_name`` is given, and persists an optional
   Telegram token to the env file. Idempotent.
6. ``POST /api/setup/complete`` (authed) — verifies the config exists
   and parses, replies, then exits the serve loop with a "restart"
   sentinel. ``daemon.main()`` re-checks for config and falls through
   into the REAL daemon boot IN THE SAME PROCESS — no systemd needed
   (under systemd a plain exit + ``Restart=`` also works, but the
   in-process fallthrough works standalone). The client polls
   ``/api/status`` with its token to see the real daemon come up.

Auth: after a successful pair, every ``/api/setup/*`` mutation requires
``Authorization: Bearer <token>`` (the minted token, resolved from the
env file via :func:`resolve_api_token`). ``GET /api/setup/status`` stays
open — it is the discovery probe clients use BEFORE they have a token.

Pairing-code policy (documented decisions):

- **One-time use** — a successful pair consumes the code.
- **15-minute TTL** — an expired code is NOT re-minted in place;
  restart the daemon to mint a new one (keeps the "printed ONCE"
  guarantee simple and true).
- **Max 5 failed attempts** — then pairing reports ``"locked"`` until
  restart.
- Codes are compared with :func:`hmac.compare_digest`; neither the code
  nor the token is ever logged after the startup banner.

Setup mode creates NO ``~/.prometheus`` state up front. The only writes
it can perform are the env-file token persist on a successful pair and
— once the operator's paired client explicitly calls ``configure`` —
the config/identity files that call exists to write. Port:
``web.api_port``'s default (8005), overridable via
``PROMETHEUS_WEB_API_PORT`` (there is no config to read a port from —
that's the point).
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
    from starlette.background import BackgroundTask
    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover — bare install without [web]
    _FASTAPI_AVAILABLE = False

DEFAULT_API_PORT = 8005  # web.api_port default (config/prometheus.yaml.default)
DEFAULT_WS_PORT = 8010   # web.ws_port default

PAIRING_TTL_SECONDS = 15 * 60
PAIRING_MAX_ATTEMPTS = 5

# run_setup_mode() return sentinel: setup finished via POST /api/setup/complete
# — daemon.main() re-checks for config and falls through into the real boot.
SETUP_COMPLETE = "setup-complete"


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


@dataclass
class SetupModeState:
    """Mutable per-boot setup progress (in-memory only, by design).

    ``configured`` flips when ``POST /api/setup/configure`` succeeds so a
    client can resume a half-done wizard via ``GET /api/setup/status``.
    ``restart_requested`` flips when ``POST /api/setup/complete``
    verifies the config — the serve loop exits and ``run_setup_mode``
    returns :data:`SETUP_COMPLETE`.
    """

    configured: bool = False
    restart_requested: bool = False


# ---------------------------------------------------------------------------
# The FastAPI app
# ---------------------------------------------------------------------------


def _extract_bearer(request: "Request") -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


def _setup_auth_error(request: "Request") -> "JSONResponse | None":
    """401 for a setup mutation without the paired token; None when OK.

    The expected token is whatever :func:`resolve_api_token` sees — the
    one ``/api/setup/pair`` minted into the env file (or a pre-existing
    env/env-file token on a machine that already had one). No token
    anywhere means nobody has paired yet: mutations are refused until
    the pairing handshake has established a shared secret.
    """
    from prometheus.config.api_token import resolve_api_token

    expected, _source = resolve_api_token(None)
    if not expected:
        return JSONResponse(status_code=401, content={
            "error": "not_paired",
            "detail": "no client is paired yet — POST /api/setup/pair with "
                      "the pairing code first, then send the returned token "
                      "as Authorization: Bearer <token>",
        })
    supplied = _extract_bearer(request)
    if not supplied or not hmac.compare_digest(supplied, expected):
        return JSONResponse(status_code=401, content={
            "error": "unauthorized",
            "detail": "setup mutations require Authorization: Bearer with "
                      "the token returned by /api/setup/pair",
        })
    return None


def _backend_payload(server: Any) -> dict[str, Any]:
    """Wire shape for one detected backend (DetectedServer → JSON)."""
    return {
        "name": server.name,
        "provider": server.provider,
        "base_url": server.url,
        "models": list(server.models),
        "latency_ms": round(server.response_ms, 1),
    }


def _apply_configure(
    body: dict[str, Any],
    *,
    api_port: int,
    ws_port: int,
    state: SetupModeState,
) -> "JSONResponse":
    """The blocking core of ``POST /api/setup/configure``.

    Validates the chosen backend by re-probing it, then writes the
    config with the SAME writer the CLI wizard's fast path uses
    (``prometheus.cli.init._default_config`` + ``write_config`` — shared
    code, not a fork), generates identity via the SetupWizard's
    generator when ``agent_name`` is given, and persists an optional
    Telegram token to the env file. Idempotent: a re-POST overwrites the
    previous answers cleanly. Returns the written summary — never tokens.
    """
    from prometheus.cli.init import _default_config, probe_backend, write_config
    from prometheus.config.paths import get_config_dir

    provider = str(body.get("provider") or "").strip()
    base_url = str(body.get("base_url") or "").strip().rstrip("/")
    model = str(body.get("model") or "").strip()
    agent_name = str(body.get("agent_name") or "").strip()
    persona = str(body.get("persona") or "").strip()
    telegram_token = str(body.get("telegram_token") or "").strip()

    missing = [k for k, v in
               (("provider", provider), ("base_url", base_url), ("model", model))
               if not v]
    if missing:
        return JSONResponse(status_code=400, content={
            "error": "bad_request",
            "detail": f"missing required field(s): {', '.join(missing)}",
        })
    if not base_url.startswith(("http://", "https://")):
        return JSONResponse(status_code=400, content={
            "error": "bad_base_url",
            "detail": "base_url must start with http:// or https://",
        })

    # Dead-end rule (Phase 0): never write a config pointing at a backend
    # that is not there. Re-probe the exact provider+URL the client chose.
    server = probe_backend(provider, base_url)
    if server is None:
        return JSONResponse(status_code=400, content={
            "error": "backend_unreachable",
            "detail": f"{base_url} did not answer like a {provider} "
                      "inference server — configure refused (a config "
                      "written now would be known-broken)",
        })

    config = _default_config(server, model)
    # The paired client reached setup mode on THESE ports — the real
    # daemon must come up on the same ones so the client's address keeps
    # working after the flip. Defaults (8005/8010) are unchanged.
    config["web"]["api_port"] = api_port
    config["web"]["ws_port"] = ws_port
    if agent_name:
        config["system"]["name"] = agent_name

    telegram_token_saved = False
    if telegram_token:
        from prometheus.config.env_file import set_env_value

        set_env_value("PROMETHEUS_TELEGRAM_TOKEN", telegram_token)
        config["gateway"]["telegram_enabled"] = True
        telegram_token_saved = True

    cfg_path = get_config_dir() / "prometheus.yaml"
    # backup_existing=False: setup mode only runs when no config existed
    # at boot, so the only thing a re-POST can overwrite is this same
    # call's earlier output — idempotent, no backup litter.
    write_config(config, cfg_path, backup_existing=False)

    identity: dict[str, str] | None = None
    if agent_name:
        from prometheus.cli.generate_identity import (
            detect_hardware,
            generate_identity_files,
        )

        identity = generate_identity_files(
            owner_name="User",
            hardware=detect_hardware(),
            overwrite=True,
            dest=get_config_dir(),
            agent_name=agent_name,
            persona=persona,
        )

    state.configured = True
    logger.info(
        "Setup configure applied: provider=%s base_url=%s model=%s "
        "agent_name=%s identity=%s telegram_token=%s (config at %s)",
        provider, base_url, model, agent_name or "-",
        "generated" if identity else "skipped",
        "saved-to-env-file" if telegram_token_saved else "none",
        cfg_path,
    )
    return JSONResponse(status_code=200, content={
        "configured": True,
        "config_path": str(cfg_path),
        "provider": provider,
        "base_url": base_url,
        "model": model,
        "agent_name": agent_name or None,
        "identity": identity,
        "telegram_token_saved": telegram_token_saved,
        "web": {
            "enabled": True,
            "api_port": api_port,
            "ws_port": ws_port,
        },
    })


def create_setup_app(
    pairing: PairingState,
    *,
    api_port: int | None = None,
    ws_port: int | None = None,
    state: SetupModeState | None = None,
    on_complete: Any = None,
):
    """Build the setup-mode FastAPI app.

    Deliberately NOT the full ``prometheus.web.server.create_app`` with a
    flag: the real route surface must be unreachable in setup mode, so
    it simply is not mounted. Routes:

    - ``GET  /api/setup/status``    → mode + pairing availability + configured
    - ``POST /api/setup/pair``      → code → real API token
    - ``GET  /api/setup/detect``    → (authed) probe inference backends
    - ``POST /api/setup/configure`` → (authed) write config + identity
    - ``POST /api/setup/complete``  → (authed) verify config, flip to real daemon
    - anything else                 → 403 with an honest JSON body

    ``on_complete`` is called (after the /complete response is sent) to
    stop the serve loop; tests leave it None.
    """
    from prometheus import __version__

    api_port = api_port if api_port is not None else resolve_setup_port()
    ws_port = ws_port if ws_port is not None else resolve_setup_ws_port()
    state = state if state is not None else SetupModeState()

    app = FastAPI(
        title="Prometheus (setup mode)",
        docs_url=None, redoc_url=None, openapi_url=None,
    )

    @app.get("/api/setup/status")
    async def setup_status() -> dict[str, Any]:
        return {
            "setup_mode": True,
            "configured": state.configured,
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

    # ── Phase 2: remotely drivable setup (authed) ───────────────────────

    @app.get("/api/setup/detect")
    def setup_detect(request: Request, base_url: str | None = None):
        # Sync endpoint on purpose: detection is blocking urllib probing;
        # FastAPI runs `def` endpoints in its threadpool.
        denied = _setup_auth_error(request)
        if denied is not None:
            return denied
        from prometheus.cli.init import (
            detect_local_servers,
            remote_server_candidates,
        )

        if base_url is not None:
            url = base_url.strip().rstrip("/")
            if not url.startswith(("http://", "https://")):
                return JSONResponse(status_code=400, content={
                    "error": "bad_base_url",
                    "detail": "base_url must start with http:// or https://",
                })
            servers = detect_local_servers(
                timeout=3.0, candidates=remote_server_candidates(url),
            )
        else:
            servers = detect_local_servers(timeout=1.0)
        return JSONResponse(status_code=200, content={
            "backends": [_backend_payload(s) for s in servers],
        })

    @app.post("/api/setup/configure")
    async def setup_configure(request: Request) -> JSONResponse:
        denied = _setup_auth_error(request)
        if denied is not None:
            return denied
        try:
            body = await request.json()
        except Exception:
            body = None
        if not isinstance(body, dict):
            return JSONResponse(status_code=400, content={
                "error": "bad_request",
                "detail": 'expected a JSON body: {"provider", "base_url", '
                          '"model", "agent_name"?, "persona"?, '
                          '"telegram_token"?}',
            })
        # Blocking work (backend re-probe, file writes) off the event loop.
        return await asyncio.to_thread(
            _apply_configure, body, api_port=api_port, ws_port=ws_port,
            state=state,
        )

    @app.post("/api/setup/complete")
    async def setup_complete(request: Request) -> JSONResponse:
        denied = _setup_auth_error(request)
        if denied is not None:
            return denied
        import yaml

        cfg_file = find_config_file(None)
        parsed: Any = None
        if cfg_file is not None:
            try:
                parsed = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
            except Exception:
                parsed = None
        if not isinstance(parsed, dict) or not parsed.get("model"):
            return JSONResponse(status_code=409, content={
                "error": "not_configured",
                "detail": "no valid prometheus.yaml found — POST "
                          "/api/setup/configure first",
            })

        state.restart_requested = True
        logger.info(
            "Setup complete — config verified at %s; handing off to the "
            "real daemon (this process).", cfg_file,
        )

        async def _shutdown() -> None:
            # Runs AFTER the response body is sent (starlette background
            # task); the short sleep lets the socket flush before the
            # serve loop starts tearing down.
            await asyncio.sleep(0.3)
            if on_complete is not None:
                on_complete()

        return JSONResponse(
            status_code=200,
            content={
                "restarting": True,
                "detail": "setup verified — the real daemon is starting in "
                          "this process. Poll /api/status with your bearer "
                          "token until it answers 200.",
            },
            background=BackgroundTask(_shutdown),
        )

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

    from prometheus.config.api_token import BEACON_DOWNLOAD_URL

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
        f"  Don't have Beacon yet?  {BEACON_DOWNLOAD_URL}\n"
        "  Or set up here instead:  prometheus setup\n"
        "  Expired or locked? Restart `prometheus daemon` for a new code.\n"
        f"{bar}\n"
    )


async def _serve_setup_mode(
    pairing: PairingState, state: SetupModeState, api_port: int,
) -> None:
    """Run uvicorn until SIGTERM/SIGINT or setup-complete; exits cleanly."""
    import uvicorn

    server_box: dict[str, Any] = {}

    def stop_server() -> None:
        srv = server_box.get("server")
        if srv is not None:
            srv.should_exit = True

    app = create_setup_app(
        pairing, api_port=api_port, state=state, on_complete=stop_server,
    )
    config = uvicorn.Config(app, host="0.0.0.0", port=api_port, log_level="info")
    server = uvicorn.Server(config)
    server_box["server"] = server

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        # uvicorn installs its own handlers only when run() owns the loop;
        # serving inside our loop means we wire shutdown explicitly.
        loop.add_signal_handler(
            sig, lambda s=sig: setattr(server, "should_exit", True)
        )
    await server.serve()


def run_setup_mode() -> int | str:
    """Entry point: `prometheus daemon` found no config.

    Boots the pairing-only server, prints the pairing code banner ONCE,
    serves until SIGTERM/SIGINT — or until a paired client finishes
    ``POST /api/setup/complete``, in which case :data:`SETUP_COMPLETE`
    is returned instead of an exit code and ``daemon.main()`` falls
    through into the real daemon boot in the same process.

    Creates no ~/.prometheus state up front; the only writes happen on
    an explicit pair (env-file token) or configure (config/identity).
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
    state = SetupModeState()

    logger.warning(
        "No prometheus.yaml found — starting in SETUP MODE (pairing-only "
        "API on :%d; the full daemon surface is NOT running). Run "
        "`prometheus setup` to configure, then restart the daemon — or "
        "drive the whole setup from a paired client (Beacon).",
        api_port,
    )
    print(format_pairing_banner(pairing.code, api_port), flush=True)

    try:
        asyncio.run(_serve_setup_mode(pairing, state, api_port))
    except KeyboardInterrupt:  # pragma: no cover — belt and braces
        pass
    if state.restart_requested:
        logger.info("Setup-mode server stopped — restarting as the real daemon.")
        return SETUP_COMPLETE
    logger.info("Setup-mode server stopped.")
    return 0
