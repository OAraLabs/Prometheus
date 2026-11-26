"""``GET``/``PUT /api/speech/settings`` — the voice box's configuration, through Prometheus only.

Design: docs/design/voice-settings.md. The logic is ``web/voice_settings.py``; this is the HTTP shape,
deliberately the one ``PUT /api/model`` already has:

* both routes are the OPERATOR's (the master token or an owner device): a scoped device gets 403
  ``operator_only`` on reading as well as on changing (``web/route_access.py``);
* ``GET`` reports the current config, the options DISCOVERED live from the box, and health per component;
* ``PUT`` saves a partial change to the config file (verified text edit), applies it live where possible
  (the speech route reads its config per call), pushes the runtime knobs to the box, and answers the GET
  shape plus ``changed``. Only the box's address waits for a restart (``applied: "on_restart"``);
* every accepted change and every refused attempt writes a security-audit row (``permissions/audit.py``),
  naming the caller and the fields — never a key (no key is a field);
* the box's key never appears in an answer, the yaml or a log line; the address appears only here, to the
  operator.

Source: novel code for Prometheus, 2026-11-26.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from prometheus.web import voice_settings as vs
from prometheus.web.network import NoConfigFile, PersistError
from prometheus.web.pairing_routes import operator_refusal

logger = logging.getLogger("prometheus.voice_settings")

_NO_STORE = {"Cache-Control": "no-store"}


def _json(status: int, content: dict[str, Any]) -> JSONResponse:
    return JSONResponse(status_code=status, content=content, headers=_NO_STORE)


def _audit(request: Request, decision: str, fields: dict[str, Any], reason: str) -> None:
    """One security-audit row for a settings attempt. Values are the settings themselves — no key can be
    among them, because no key is a field. An audit sink must not fail the route."""
    try:
        from prometheus.config.paths import get_data_dir
        from prometheus.permissions.audit import AuditDecision, AuditLogger

        identity = getattr(request.state, "device_identity", None)
        AuditLogger(get_data_dir()).log(
            tool_name="speech.settings",
            decision=AuditDecision.ALLOW if decision == "allow" else AuditDecision.DENY,
            trust_level=0,
            reason=reason,
            tool_input={"fields": fields},
            user_id=getattr(identity, "id", None) or "global-token",
        )
    except Exception:  # noqa: BLE001 — an audit sink must not fail the route
        logger.warning("voice settings: could not write the security audit row", exc_info=True)


def register_voice_settings_routes(app: Any, *, auth_on: Callable[[], bool]) -> None:
    """Mount the two routes on *app* (before the static catch-all).

    Reads ``app.state.config`` (the running config the speech route reads per call) and
    ``app.state.config_path`` (set by the launcher; absent in a process not started from a file).
    """

    def config(request: Request) -> dict[str, Any]:
        cfg = getattr(request.app.state, "config", None)
        return cfg if isinstance(cfg, dict) else {}

    def health(request: Request) -> Any:
        service = getattr(request.app.state, "speech", None)
        return service.voice if service is not None else None

    async def _view(request: Request, *, pending: bool = False,
                    warnings: list[dict[str, str]] | None = None) -> dict[str, Any]:
        cfg = config(request)
        disc = await vs.discover(cfg)
        voice = health(request)
        halves = await voice.status() if voice is not None else {}
        return vs.view(cfg, halves, disc, pending=pending, warnings=warnings)

    @app.get("/api/speech/settings")
    async def get_settings(request: Request) -> JSONResponse:
        refusal = operator_refusal(request, auth_on())
        if refusal is not None:
            return refusal
        return _json(200, await _view(request))

    @app.put("/api/speech/settings")
    async def put_settings(request: Request) -> JSONResponse:
        refusal = operator_refusal(request, auth_on())
        if refusal is not None:
            return refusal
        try:
            body = await request.json()
        except ValueError:
            body = None
        cfg = config(request)
        disc = await vs.discover(cfg)
        try:
            change = vs.clean(body, disc, cfg)
        except vs.VoiceSettingsRefusal as exc:
            _audit(request, "deny", body if isinstance(body, dict) else {}, f"voice settings refused: {exc.code}")
            return _json(exc.status, exc.as_json())
        if not change:
            return _json(400, {"error": vs.INVALID_REQUEST, "detail": "no fields to change"})

        path = getattr(request.app.state, "config_path", None)
        if path is None:
            _audit(request, "deny", change, "voice settings refused: no_config_file")
            return _json(409, {"error": vs.NO_CONFIG_FILE,
                               "detail": "the daemon was not started from a config file; nothing to save into"})
        try:
            plan = vs.plan_change(path, change)
            vs.save(plan)
        except NoConfigFile:
            _audit(request, "deny", change, "voice settings refused: no_config_file")
            return _json(409, {"error": vs.NO_CONFIG_FILE, "detail": f"no config file at {path}"})
        except PersistError as exc:
            _audit(request, "deny", change, "voice settings refused: persist_failed")
            return _json(500, {"error": vs.PERSIST_FAILED, "detail": str(exc)})

        # Live apply: the running config the speech route reads per call. The address waits for a restart.
        vs.apply_running(cfg, change)
        pending = vs.ADDRESS_FIELD in change
        warnings = await vs.push(cfg, change)
        _audit(request, "allow", change,
               "voice settings changed: " + ", ".join(sorted(change)) + (" (address on restart)" if pending else ""))
        result = await _view(request, pending=pending, warnings=warnings)
        result["changed"] = sorted(change)
        return _json(200, result)