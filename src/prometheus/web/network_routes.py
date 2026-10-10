"""``GET /api/network`` and ``PUT /api/network`` — what the daemon is listening for, and the owner's switch.

Contract: docs/PAIRING-APPROVAL-API.md, 2.2. Anyone with a valid token may READ it (a client wants to know
which network mode it is talking to and whether the daemon is discoverable); only an operator
(``identity.is_operator``) may change it, and a scoped device is a 403 ``operator_only``, not a 401.

What the owner's switch can and cannot do is stated, not implied:

* **It is saved, not applied.** The listeners are bound at start, so ``PUT`` writes the choice into the config
  file and answers ``applied: "on_restart"``; ``GET`` then reports the mode it is running AND the one it will
  come back as (``pending_mode``). Rebinding live is not built.
* **It refuses what cannot work.** ``home_network`` without TLS and without the owner's explicit
  ``network.allow_plaintext_lan`` is a 409 ``tls_unavailable`` and writes nothing. A bind pinned by
  ``--bind`` or ``PROMETHEUS_WEB_BIND`` outranks the file, so changing the file would be a lie: 409
  ``bind_overridden``, naming the source.
* **It edits the file's text** (comments and all) and verifies before writing (``web/network.py``); no config
  file is a 409, never a file conjured into existence.
* **It does not let a caller lock itself out.** ``this_mac`` from a caller that is not on this machine is a 409
  ``would_lock_out``: after the restart that caller could no longer reach the daemon. "This machine" is
  ``web.loopback.is_same_machine``: a loopback TCP peer that relayed nothing (no ``X-Forwarded-For`` or
  ``Forwarded``, which a local reverse proxy adds), and anything that does not clearly say so is not.
* **It says whether to show the control** (``can_change``: the caller is an operator AND nothing outside the file
  fixes the bind) and **codes every warning and the advertising reason** (``{"code", "message"}``) so a client
  writes its own copy and keeps the English as the fallback.

Only ``this_mac`` and ``home_network`` can be asked for. ``open`` is a description of what is running (what
every install that predates the setting is), not a choice.

Source: novel code for Prometheus, 2026-10-08.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml
from fastapi import Request
from fastapi.responses import JSONResponse

from prometheus.web.bind import DEFAULT_BIND, BindError, ResolvedBind, parse_bind, resolve_bind
from prometheus.web.loopback import is_loopback_address, is_same_machine
from prometheus.web.network import (
    CHOOSABLE,
    HOME,
    RESTART_REQUIRED,
    THIS_MAC,
    NetworkSettings,
    NetworkState,
    Notice,
    PersistError,
    describe,
    persist_choice,
)
from prometheus.web.pairing_routes import operator_refusal

logger = logging.getLogger("prometheus.network")

_NO_STORE = {"Cache-Control": "no-store"}

#: Bind sources the config file decides. Anything else (a flag, the environment, a caller) outranks it.
_FILE_DECIDES = ("config", "default")

_PINNED_BY = {"flag": "--bind", "env": "PROMETHEUS_WEB_BIND"}

#: The advertising reason when this process has no advertiser to ask (the other codes are web/discovery.py's).
ADVERTISER_ABSENT = "advertiser_not_running"


def _json(status: int, content: dict[str, Any]) -> JSONResponse:
    return JSONResponse(status_code=status, content=content, headers=_NO_STORE)


def _tls_block() -> dict[str, Any]:
    """The TLS listener's state. There is none yet (a later change); the shape is the contract's."""
    return {"enabled": False, "spki_sha256": None}


def _read_file(path: str | None) -> dict[str, Any] | None:
    """The config file as a dict, ``{}`` for an empty one, ``None`` where there is no file to read."""
    if not path or not Path(path).is_file():
        return None
    try:
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise PersistError(f"cannot read {path}: {type(exc).__name__}") from exc
    if document is None:
        return {}
    if not isinstance(document, dict):
        raise PersistError(f"{path} does not hold a mapping at the top level")
    return document


def _state_from_file(resolved: ResolvedBind, document: Mapping[str, Any]) -> NetworkState | None:
    """What the daemon will be after a restart, from the file and whatever outranks it. ``None`` if unknowable."""
    settings = NetworkSettings.from_config(document, quiet=True)
    if resolved.source not in _FILE_DECIDES:
        return describe(resolved.address, resolved.source, settings)
    web = document.get("web")
    raw = web.get("bind") if isinstance(web, Mapping) else None
    if raw is None:
        return describe(DEFAULT_BIND, "default", settings)
    try:
        return describe(parse_bind(raw, "web.bind"), "config", settings)
    except BindError:
        return None                                    # the daemon would refuse to start; nothing to promise


def register_network_routes(app: Any, *, auth_on: Callable[[], bool]) -> None:
    """Mount ``GET`` and ``PUT /api/network`` on *app* (before the static catch-all).

    Reads from ``app.state``: ``config`` (what the daemon booted with), ``resolved_bind`` and ``config_path``
    (set by the launcher) and ``advertiser`` (its status). Any of the last three may be absent, and the
    answer says so rather than guessing.
    """

    def running(request: Request) -> tuple[ResolvedBind, NetworkState]:
        state = request.app.state
        config = state.config if isinstance(getattr(state, "config", None), dict) else {}
        resolved = getattr(state, "resolved_bind", None)
        if resolved is None:
            try:
                resolved = resolve_bind(config)
            except BindError:
                resolved = ResolvedBind(DEFAULT_BIND, "default")
        return resolved, describe(resolved.address, resolved.source, NetworkSettings.from_config(config, quiet=True))

    def advertising(request: Request) -> tuple[bool, dict[str, str] | None]:
        advertiser = getattr(request.app.state, "advertiser", None)
        if advertiser is None:
            return False, Notice(ADVERTISER_ABSENT, "the advertiser is not running in this process").as_json()
        status = advertiser.status
        if status.advertising or status.reason is None:
            return bool(status.advertising), None
        return False, Notice(status.code or "unknown", status.reason).as_json()

    def can_change(request: Request, resolved: ResolvedBind) -> bool:
        """Whether a control that changes the network mode is worth showing THIS caller.

        An operator, and a bind nothing outside the file fixes. (A PUT can still be refused for another reason: no
        config file, no TLS or opt-out for home network, or ``would_lock_out``.)
        """
        return operator_refusal(request, auth_on()) is None and resolved.source in _FILE_DECIDES

    def body(request: Request, shown: NetworkState, *, applied: str, warnings: list[Notice],
             resolved: ResolvedBind, pending: NetworkState | None = None) -> dict[str, Any]:
        on, why = advertising(request)
        out: dict[str, Any] = {
            "mode": shown.mode,
            "bind": shown.bind,
            "bind_source": shown.bind_source,
            "tls": _tls_block(),
            "advertising": on,
            "advertising_reason": why,
            "applied": applied,
            "warnings": [warning.as_json() for warning in warnings],
            "can_change": can_change(request, resolved),
        }
        if pending is not None:
            out["pending_mode"] = pending.mode
        return out

    def who(request: Request) -> str:
        if not auth_on():
            return "an unauthenticated caller (the API token is off)"
        identity = getattr(request.state, "device_identity", None)
        return getattr(identity, "name", None) or getattr(identity, "id", None) or "the master token"

    @app.get("/api/network")
    async def get_network(request: Request):
        resolved, now = running(request)
        pending: NetworkState | None = None
        try:
            document = _read_file(getattr(request.app.state, "config_path", None))
        except PersistError:
            document = None                            # unreadable: report what is running, promise nothing
        if document is not None:
            pending = _state_from_file(resolved, document)
        warnings = list(now.warnings)
        if pending is not None and (pending.mode, pending.bind) != (now.mode, now.bind):
            warnings.append(Notice(
                RESTART_REQUIRED,
                f"A saved change is waiting for a restart: the daemon will come back as {pending.mode} on "
                f"{pending.bind}."))
            return _json(200, body(request, now, applied="on_restart", warnings=warnings, resolved=resolved,
                                   pending=pending))
        return _json(200, body(request, now, applied="live", warnings=warnings, resolved=resolved))

    @app.put("/api/network")
    async def put_network(request: Request):
        if (refusal := operator_refusal(request, auth_on())) is not None:
            return refusal
        try:
            asked = await request.json()
        except ValueError:
            asked = None
        mode = asked.get("mode") if isinstance(asked, dict) and set(asked) == {"mode"} else None
        if not isinstance(mode, str) or mode not in CHOOSABLE:
            return _json(400, {"error": "invalid_request",
                               "detail": 'send {"mode": "this_mac"} or {"mode": "home_network"}, and nothing else'})

        if mode == THIS_MAC and not is_same_machine(request):
            return _json(409, {
                "error": "would_lock_out",
                "detail": "this_mac would stop this daemon answering anywhere but on its own machine after the "
                          "restart, and this request does not come from this machine, so it would cut itself "
                          "off. Do this from the machine the daemon runs on (Beacon there, or set web.bind: "
                          "127.0.0.1 in prometheus.yaml). Nothing was written"})

        resolved, now = running(request)
        if resolved.source not in _FILE_DECIDES and is_loopback_address(resolved.address) != (mode == THIS_MAC):
            pinned = _PINNED_BY.get(resolved.source, "the process that started the daemon")
            return _json(409, {
                "error": "bind_overridden", "source": resolved.source,
                "detail": f"the listen address is set by {pinned}, which outranks the config file, so saving "
                          "this choice would not change it. Change it there, or restart the daemon without it"})

        path = getattr(request.app.state, "config_path", None)
        try:
            document = _read_file(path)
            if path is None or document is None:
                return _json(409, {"error": "no_config_file",
                                   "detail": "the daemon was not started from a config file, so there is nowhere "
                                             "to save this. Nothing was written"})
            if mode == HOME and not NetworkSettings.from_config(document, quiet=True).allow_plaintext_lan:
                return _json(409, {
                    "error": "tls_unavailable",
                    "detail": "home network mode needs TLS, which this daemon does not have yet, or the owner's "
                              "explicit opt-in to plain HTTP (network.allow_plaintext_lan: true in the config "
                              "file). Nothing was written"})
            changed = persist_choice(path, mode, resolved.address)
            saved = _read_file(path)
        except PersistError as exc:
            logger.warning("network: could not save %s: %s", mode, exc)
            return _json(500, {"error": "persist_failed", "detail": str(exc)})

        after = _state_from_file(resolved, saved or {})
        if after is None:                              # cannot happen after our own verified write; fail loudly
            return _json(500, {"error": "persist_failed", "detail": "the saved file does not read back"})
        if changed:
            logger.info("network: %s chose %s (saved to %s; takes effect when the daemon restarts)",
                        who(request), mode, path)
        warnings = list(after.warnings)
        if (after.mode, after.bind) == (now.mode, now.bind):
            return _json(200, body(request, after, applied="live", warnings=warnings, resolved=resolved))
        warnings.append(Notice(
            RESTART_REQUIRED,
            f"Saved. This takes effect when the daemon restarts; until then it is still {now.mode} on "
            f"{now.bind}."))
        return _json(200, body(request, after, applied="on_restart", warnings=warnings, resolved=resolved))
