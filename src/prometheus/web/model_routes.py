"""``GET``/``PUT /api/model`` and ``POST /api/model/check-key`` — change the provider, model and key after setup.

Beacon v0.4 (P8 item 7): a person who picked wrong at setup fixes it from the app instead of reinstalling. The logic
is ``web/model_settings.py``; this is the HTTP shape, deliberately the one ``PUT /api/network`` already has:

* all three are the OPERATOR's (the master token or an owner device): an approved, scoped device gets a 403
  ``operator_only`` on reading as well as on changing (``web/route_access.py``);
* ``PUT`` SAVES and answers ``applied: "on_restart"``; the daemon builds its provider at start, so the change takes
  effect on ``POST /api/restart``. ``GET`` reports what is running, what is saved and whether they differ;
* every refusal is ``{"error": <stable code>, "detail": <English>}`` and writes nothing: the key is checked first, the
  edit is verified before the file is replaced, and the key is written before the file so a failure in between leaves
  a stray key, never a config that names a provider whose key is missing;
* the key is never in an answer, the yaml or a log line.

Source: novel code for Prometheus, 2026-10-11.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from prometheus.web import model_settings as ms
from prometheus.web.network import NoConfigFile, Notice, PersistError, read_config_text, replace_config
from prometheus.web.pairing_routes import operator_refusal

logger = logging.getLogger("prometheus.model")

_NO_STORE = {"Cache-Control": "no-store"}
RESTART_REQUIRED = "restart_required"
MODEL_PINNED = "model_overridden"

#: Set on ``app.state`` when a key was saved that the running provider does not hold: the file may still read the
#: same, but only a restart makes the daemon use the new key.
_KEY_PENDING = "model_key_pending"


def _json(status: int, content: dict[str, Any]) -> JSONResponse:
    return JSONResponse(status_code=status, content=content, headers=_NO_STORE)


def register_model_routes(app: Any, *, auth_on: Callable[[], bool]) -> None:
    """Mount the three routes on *app* (before the static catch-all).

    Reads ``app.state.config`` (what the daemon booted with, so what is running) and ``app.state.config_path`` (set by
    the launcher; absent in a process not started from a file, which the answer says instead of guessing).
    """

    def running_section(request: Request) -> dict[str, Any]:
        config = getattr(request.app.state, "config", None)
        model = config.get("model") if isinstance(config, dict) else None
        return model if isinstance(model, dict) else {}

    def saved_section(request: Request) -> dict[str, Any] | None:
        """The file's ``model:`` section, or ``None`` when there is no readable file to promise anything about."""
        path = getattr(request.app.state, "config_path", None)
        if path is None:
            return None
        try:
            _target, _text, document = read_config_text(path)
        except PersistError:
            return None
        section = document.get("model")
        return section if isinstance(section, dict) else {}

    def who(request: Request) -> str:
        if not auth_on():
            return "an unauthenticated caller (the API token is off)"
        identity = getattr(request.state, "device_identity", None)
        return getattr(identity, "name", None) or getattr(identity, "id", None) or "the master token"

    def providers() -> list[dict[str, Any]]:
        checkable = ms.checkable_providers()
        return [{"provider": name, "api_key_env": env, "default_model": model, "key_present": ms.key_present(env),
                 "checkable": name in checkable}
                for name, (env, model, _limit) in ms.cloud_table().items()]

    def view(request: Request, *, warnings: list[Notice] | None = None, **extra: Any) -> dict[str, Any]:
        """The state both GET and PUT answer with."""
        running = ms.describe(running_section(request))
        section = saved_section(request)
        saved = ms.after_restart(section) if section is not None else running
        pending = saved != running or bool(getattr(request.app.state, _KEY_PENDING, False))
        notices = list(warnings or [])
        if pending:
            notices.insert(0, Notice(
                RESTART_REQUIRED,
                "A saved change is waiting for a restart: "
                + (f"the daemon will come back with {saved['provider']} ({saved['model'] or 'its default model'})."
                   if saved != running else "the daemon will use the new key.")))
        for variable, what in ((ms.PIN_PROVIDER, "provider"), (ms.PIN_MODEL, "model")):
            if what in ms.pins():
                notices.append(Notice(
                    MODEL_PINNED,
                    f"The {what} is set by {variable} in the daemon's environment, which outranks the config file."))
        key_env = saved["api_key_env"]
        return {
            "running": running,
            "saved": saved,
            "pending": pending,
            "applied": "on_restart" if pending else "live",
            "key": ({"env": key_env, "present": bool(ms.held_key(saved["provider"], section or running_section(request)))}
                    if key_env else None),
            "providers": providers(),
            "warnings": [n.as_json() for n in notices],
            "can_change": operator_refusal(request, auth_on()) is None and section is not None,
            **extra,
        }

    def failure(exc: PersistError | ms.ModelRefusal) -> JSONResponse:
        """The coded answer for a request that cannot be saved. Nothing was written."""
        if isinstance(exc, ms.ModelRefusal):
            return _json(exc.status, exc.as_json())
        if isinstance(exc, NoConfigFile):
            return _json(409, {"error": ms.NO_CONFIG_FILE,
                               "detail": "the daemon was not started from a config file, so there is nowhere to "
                                         "save this. Nothing was written"})
        logger.warning("model: could not plan the change: %s", exc)
        return _json(500, {"error": ms.PERSIST_FAILED, "detail": str(exc)})

    async def body_of(request: Request) -> Any:
        try:
            return await request.json()
        except ValueError:
            return None

    @app.get("/api/model")
    async def get_model(request: Request):
        if (refusal := operator_refusal(request, auth_on())) is not None:
            return refusal
        return _json(200, view(request))

    @app.post("/api/model/check-key")
    async def check_model_key(request: Request):
        """Does this key work? With no ``api_key``, the one the daemon already holds for the provider."""
        if (refusal := operator_refusal(request, auth_on())) is not None:
            return refusal
        from prometheus.providers import key_check

        try:
            asked = ms.parse_check(await body_of(request))
            section = saved_section(request) or running_section(request)
            key = asked.api_key or ms.held_key(asked.provider, section)
            if not key:
                raise ms.ModelRefusal(
                    400, ms.CLOUD_KEY_MISSING,
                    f"no {asked.provider} key is saved on this machine and none was sent; send api_key")
        except ms.ModelRefusal as refused:
            return _json(refused.status, refused.as_json())
        verdict = await asyncio.to_thread(key_check.check_provider_key, asked.provider, key)
        return _json(200, verdict.as_dict())

    @app.put("/api/model")
    async def put_model(request: Request):
        if (refusal := operator_refusal(request, auth_on())) is not None:
            return refusal
        from prometheus.config.env_file import set_env_value
        from prometheus.providers import key_check

        path = getattr(request.app.state, "config_path", None)

        def prepare(asked: ms.ChangeRequest) -> tuple[ms.Plan, str]:
            """The verified edit and the key it will use, or a refusal. Reads the file; writes nothing."""
            if path is None:
                raise NoConfigFile("the daemon was not started from a config file")
            plan = ms.plan_change(path, asked)
            if (pinned := ms.pinned_refusal(asked, plan)) is not None:
                raise pinned
            held = ms.stored_key(plan.key_env) if plan.provider_changed else ms.held_key(asked.provider, plan.section)
            key = asked.api_key or held
            if not key:
                raise ms.ModelRefusal(
                    400, ms.CLOUD_KEY_MISSING,
                    f"no {asked.provider} key is saved on this machine and none was sent; send api_key. "
                    "Nothing was written")
            return plan, key

        try:
            asked = ms.parse_change(await body_of(request))
            plan, key = prepare(asked)
        except (PersistError, ms.ModelRefusal) as exc:
            return failure(exc)

        warnings: list[Notice] = []
        verdict: dict[str, Any] | None = None
        if asked.check and asked.provider in ms.checkable_providers() and not plan.custom_endpoint:
            checked = await asyncio.to_thread(key_check.check_provider_key, asked.provider, key)
            verdict = checked.as_dict()
            if checked.result == key_check.BAD_KEY:
                return _json(400, {"error": ms.KEY_REJECTED, "key_check": verdict,
                                   "detail": f"{asked.provider} refused this key, so nothing was saved"})
            if checked.result == key_check.NO_CREDIT:
                warnings.append(Notice(ms.KEY_NO_CREDIT, "The key works, but the account has no credit, so "
                                                         "requests will fail until credit is added."))
            elif checked.result == key_check.UNREACHABLE:
                warnings.append(Notice(ms.KEY_UNCHECKED, f"{asked.provider} could not be reached to check the "
                                                         "key, so it was not confirmed."))
            # The check took seconds; another writer (PUT /api/network, a hand edit) may have changed the file
            # meanwhile. The edit is rebuilt from the file as it is NOW so that change is not written over.
            try:
                plan, key = prepare(asked)
            except (PersistError, ms.ModelRefusal) as exc:
                return failure(exc)
        elif asked.check and asked.api_key is not None:
            why = ("it is set to an address of your own, which the check does not call"
                   if plan.custom_endpoint else f"{asked.provider}'s keys cannot be checked from here")
            warnings.append(Notice(ms.KEY_UNCHECKED, f"The key was saved without being confirmed: {why}."))

        key_saved = False
        try:
            if asked.api_key is not None:
                prior = ms.stored_key(plan.key_env)
                set_env_value(plan.key_env, asked.api_key)         # before the file: see the module docstring
                os.environ[plan.key_env] = asked.api_key
                key_saved = True
                if prior != asked.api_key:
                    setattr(request.app.state, _KEY_PENDING, True)
            if plan.changed:
                replace_config(plan.target, plan.updated)
        except (OSError, PersistError) as exc:
            logger.warning("model: could not save: %s", type(exc).__name__)
            return _json(500, {"error": ms.PERSIST_FAILED, "detail": "could not write the settings"})

        if plan.changed or key_saved:
            logger.info("model: %s set provider=%s model=%s key=%s (saved to %s; takes effect when the daemon "
                        "restarts)", who(request), plan.saved["provider"], plan.saved["model"],
                        "replaced" if key_saved else "kept", plan.target)
        return _json(200, view(request, warnings=warnings, changed=plan.changed, key_saved=key_saved,
                               key_check=verdict))
