"""Change the main model's provider, model and API key AFTER setup (Beacon v0.4, P8 item 7).

Setup picks a provider once. A person who picked wrong (the wrong vendor, a mistyped key, an account with no credit)
has to be able to fix it without reinstalling. This module is the logic behind ``GET``/``PUT /api/model`` and
``POST /api/model/check-key`` (``web/model_routes.py``); it knows nothing of HTTP.

What a change is, and what it is not:

* **Saved, not applied.** The provider is built once, at start, so a change is written to the config file (and the
  key to the env file) and takes effect when the daemon restarts (``POST /api/restart``). The routes say so.
* **Cloud providers only** (the ones setup offers: ``cli/init._CLOUD_FAST_PROVIDERS``, so a provider added there is
  choosable here with no second list). A local server needs an address and a probe of what it serves; that is
  ``oara setup``'s job, and this route takes no ``base_url`` (see below).
* **The file is edited as text**, comments and all (``web/network.py``), and the edit is verified before anything is
  written: the document must read back as the old one plus exactly the intended keys.
* **The key lives in the env file** (0600), never the yaml, an answer or a log line. It is checked first, with the
  same check ``/api/setup/check-key`` runs (``providers/key_check.py``): a key the provider refuses is not saved.

What a provider switch must NOT carry over, because each is a way for "fix my setup" to make it worse. In ``model:``
an inline ``api_key`` outranks the env var, ``base_url`` outranks the provider's own endpoint, and ``base_url_env``
names an environment variable for the OLD vendor's endpoint. Left behind, they would send the old key to the new
vendor, or the new vendor's requests to the old one's server. A switch blanks all three. A new key for the SAME
provider blanks only the inline key (it would otherwise keep outranking the one just saved); the owner's own endpoint
stays.

Source: novel code for Prometheus, 2026-10-11.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prometheus.web.network import (
    PersistError,
    _edit_text,
    _set_nested,
    read_config_text,
    verify_edit,
)

logger = logging.getLogger("prometheus.model")

#: The environment variables that outrank the file for these two settings (``config/env_override.py``).
PIN_PROVIDER = "PROMETHEUS_PROVIDER"
PIN_MODEL = "PROMETHEUS_MODEL"

#: Stable error and warning codes. A code is a promise: it does not change when the wording does, and a new situation
#: gets a new code. A client writes its own copy from the code; the English ``detail`` is for logs and clients that
#: have no copy yet.
INVALID_REQUEST = "invalid_request"
UNEXPECTED_BASE_URL = "unexpected_base_url"
UNSUPPORTED_PROVIDER = "unsupported_provider"
BAD_MODEL = "bad_model"
INVALID_KEY = "invalid_key"
CLOUD_KEY_MISSING = "cloud_key_missing"
KEY_REJECTED = "key_rejected"
MODEL_OVERRIDDEN = "model_overridden"
NO_CONFIG_FILE = "no_config_file"
PERSIST_FAILED = "persist_failed"
KEY_NO_CREDIT = "key_no_credit"
KEY_UNCHECKED = "key_unchecked"

#: A model id as every provider here writes one: ``claude-sonnet-5``, ``anthropic/claude-sonnet-5``,
#: ``qwen3.8-max``, ``llama-4:free``. No whitespace, quote, ``#`` or newline: it goes into a YAML scalar.
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,199}")

#: Sanity cap on a pasted key, as ``PUT /api/providers/keys`` has it; real keys are far shorter.
KEY_MAX_CHARS = 4096


class ModelRefusal(Exception):
    """A request this module will not act on: the HTTP *status*, a stable *code* and an English *detail*."""

    def __init__(self, status: int, code: str, detail: str, **extra: Any) -> None:
        super().__init__(detail)
        self.status, self.code, self.detail, self.extra = status, code, detail, extra

    def as_json(self) -> dict[str, Any]:
        return {"error": self.code, "detail": self.detail, **self.extra}


# ── the providers a person can choose ────────────────────────────────────────

def cloud_table() -> dict[str, tuple[str, str, int]]:
    """provider → (key env var, default model, context budget): setup's table, the one source of truth."""
    from prometheus.cli.init import _CLOUD_FAST_PROVIDERS

    return _CLOUD_FAST_PROVIDERS


def checkable_providers() -> tuple[str, ...]:
    from prometheus.providers.key_check import CHECKABLE_PROVIDERS

    return CHECKABLE_PROVIDERS


def _is_cloud(provider: str) -> bool:
    from prometheus.providers.registry import ProviderRegistry

    return ProviderRegistry.is_cloud(provider)


def _default_env(provider: str) -> str:
    from prometheus.providers.registry import CLOUD_DEFAULTS

    return str(CLOUD_DEFAULTS.get(provider, {}).get("default_env") or "")


# ── describing a ``model:`` section ──────────────────────────────────────────

def describe(section: Mapping[str, Any] | None) -> dict[str, Any]:
    """The four facts a client shows about a ``model:`` section. Names only: no key, no URL."""
    from prometheus.config.defaults import DEFAULT_MODEL_PROVIDER

    section = section if isinstance(section, Mapping) else {}
    provider = str(section.get("provider") or DEFAULT_MODEL_PROVIDER)
    cloud = _is_cloud(provider)
    env = str(section.get("api_key_env") or "") or (_default_env(provider) if cloud else "")
    return {
        "provider": provider,
        "model": str(section.get("model") or ""),
        "api_key_env": env or None,
        "kind": "cloud" if cloud else "local",
    }


def pins() -> dict[str, str]:
    """The settings an environment variable fixes, outranking the file: ``{"provider"|"model": value}``."""
    found = {}
    for name, field in ((PIN_PROVIDER, "provider"), (PIN_MODEL, "model")):
        value = os.environ.get(name, "").strip()
        if value:
            found[field] = value
    return found


def after_restart(section: Mapping[str, Any] | None) -> dict[str, Any]:
    """What a restart with this file comes back as: the file, with whatever the environment pins laid over it."""
    merged = dict(section) if isinstance(section, Mapping) else {}
    merged.update(pins())
    return describe(merged)


# ── keys ─────────────────────────────────────────────────────────────────────

def key_env_for(provider: str, section: Mapping[str, Any] | None) -> str:
    """The environment variable that holds *provider*'s key once the file is saved for that provider.

    The provider the file already names keeps the variable the file names (an owner's own choice); any other
    provider uses setup's standard one.
    """
    section = section if isinstance(section, Mapping) else {}
    if describe(section)["provider"] == provider and section.get("api_key_env"):
        return str(section["api_key_env"])
    row = cloud_table().get(provider)
    return row[0] if row else _default_env(provider)


def stored_key(env: str) -> str:
    """The key under *env* as the process holds it, else as the env file does. ``""`` when there is none."""
    from prometheus.config.env_file import parse_env_file

    value = os.environ.get(env, "")
    if not value:
        try:
            value = parse_env_file().get(env, "")
        except OSError:
            value = ""
    return value.strip()


def held_key(provider: str, section: Mapping[str, Any] | None) -> str:
    """The key the daemon would use for *provider* if the file stayed as it is: an inline one (which outranks the
    variable) for the provider the file names, else whatever the variable holds."""
    section = section if isinstance(section, Mapping) else {}
    if describe(section)["provider"] == provider:
        inline = section.get("api_key")
        if isinstance(inline, str) and inline.strip():
            return inline.strip()
    return stored_key(key_env_for(provider, section))


def key_present(env: str) -> bool:
    return bool(stored_key(env))


# ── reading a request ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ChangeRequest:
    provider: str
    model: str | None
    api_key: str | None
    check: bool


def clean_key(value: Any) -> str:
    """A pasted key, trimmed. Anything that is not one is an ``invalid_key`` refusal: the env file is line-based, so a
    control character (a newline above all) could inject a second assignment."""
    if not isinstance(value, str):
        raise ModelRefusal(400, INVALID_REQUEST, "api_key must be a string")
    key = value.strip()
    if not key:
        raise ModelRefusal(400, INVALID_KEY, "api_key is empty; leave it out to keep the key already saved")
    if len(key) > KEY_MAX_CHARS:
        raise ModelRefusal(400, INVALID_KEY, "that is too long to be an API key")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in key):
        raise ModelRefusal(400, INVALID_KEY, "an API key has no control characters or line breaks")
    return key


def _clean_provider(value: Any, *, among: tuple[str, ...], what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelRefusal(400, INVALID_REQUEST, "provider is required (a string)")
    provider = value.strip()
    if provider not in among:
        raise ModelRefusal(400, UNSUPPORTED_PROVIDER, f"{what} knows {', '.join(among)}")
    return provider


def _object(body: Any, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise ModelRefusal(400, INVALID_REQUEST, "expected a JSON object")
    if "base_url" in body:
        raise ModelRefusal(400, UNEXPECTED_BASE_URL,
                           "each provider is called at its own address; this takes no base_url")
    unknown = sorted(set(body) - allowed)
    if unknown:
        raise ModelRefusal(400, INVALID_REQUEST,
                           f"unknown field(s) {', '.join(unknown)}; send {', '.join(sorted(allowed))}")
    return body


def parse_change(body: Any) -> ChangeRequest:
    """``{"provider", "model"?, "api_key"?, "check"?}``. Raises :class:`ModelRefusal`; touches nothing."""
    body = _object(body, {"provider", "model", "api_key", "check"})
    provider = _clean_provider(body.get("provider"), among=tuple(cloud_table()), what="this route")
    model = body.get("model")
    if "model" in body:
        if not isinstance(model, str):
            raise ModelRefusal(400, INVALID_REQUEST, "model must be a string")
        model = model.strip()
        if not _MODEL_ID.fullmatch(model):
            raise ModelRefusal(400, BAD_MODEL,
                               "a model id has letters, digits and . _ : / @ + - only, and no spaces")
    key = clean_key(body["api_key"]) if "api_key" in body else None
    check = body.get("check", True)
    if not isinstance(check, bool):
        raise ModelRefusal(400, INVALID_REQUEST, "check must be true or false")
    return ChangeRequest(provider, model if "model" in body else None, key, check)


@dataclass(frozen=True)
class CheckRequest:
    provider: str
    api_key: str | None


def parse_check(body: Any) -> CheckRequest:
    """``{"provider", "api_key"?}``: without a key, the one the daemon already holds is checked."""
    body = _object(body, {"provider", "api_key"})
    provider = _clean_provider(body.get("provider"), among=checkable_providers(), what="the key check")
    return CheckRequest(provider, clean_key(body["api_key"]) if "api_key" in body else None)


# ── planning the edit ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Plan:
    """A verified edit of the config file, not yet written."""

    target: Path
    original: str
    updated: str
    section: Mapping[str, Any]        # the file's ``model:`` section BEFORE the edit
    saved: dict[str, Any]             # ``describe`` of the section AFTER it
    key_env: str                      # the variable the key is held under once saved
    provider_changed: bool
    model_written: str | None         # the model id the edit sets, or None when it leaves the model alone
    custom_endpoint: bool             # the provider is kept at an address the owner set: a key for THAT, not the vendor's

    @property
    def changed(self) -> bool:
        return self.updated != self.original


def _literal(value: Any) -> str:
    return str(value) if isinstance(value, int) and not isinstance(value, bool) else json.dumps(value)


def plan_change(path: str | os.PathLike[str], asked: ChangeRequest) -> Plan:
    """The edit *asked* needs, verified against the file. Raises :class:`~prometheus.web.network.PersistError`
    (and, for a missing file, its subclass ``NoConfigFile``); the file is not touched."""
    table = cloud_table()
    target, original, document = read_config_text(path)
    section = document.get("model")
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise PersistError(f"{target} has a model section that is not a mapping; not touching it")

    current = describe(section)["provider"]
    switched = asked.provider != current
    standard_env, default_model, default_limit = table[asked.provider]

    edits: list[tuple[tuple[str, ...], Any]] = []
    if switched:
        edits.append((("model", "provider"), asked.provider))
        edits.append((("model", "api_key_env"), standard_env))
    model = asked.model if asked.model is not None else (default_model if switched else None)
    if model is not None and model != section.get("model"):
        edits.append((("model", "model"), model))

    # What the previous provider left that would be read as the new one's.
    stale = ("api_key", "base_url", "base_url_env") if switched else (("api_key",) if asked.api_key else ())
    for name in stale:
        if section.get(name) not in (None, ""):
            edits.append((("model", name), ""))

    if switched:                                       # the budget follows the provider, unless the owner tuned it
        raw_context = document.get("context")
        old = (raw_context if isinstance(raw_context, dict) else {}).get("effective_limit")
        if _is_cloud(current):
            prior = table.get(current)
            follows = prior is not None and old == prior[2]
        else:
            follows = True                             # a local server's window is not a cloud provider's budget
        if follows and old != default_limit:
            edits.append((("context", "effective_limit"), default_limit))

    expected = copy.deepcopy(document)
    for keys, value in edits:
        _set_nested(expected, list(keys), value)
    updated = _edit_text(original, [(list(keys), _literal(value)) for keys, value in edits]) if edits else original
    verify_edit(original, updated, expected, target, "model settings")

    after = expected.get("model") if isinstance(expected.get("model"), dict) else {}
    return Plan(
        target=target, original=original, updated=updated, section=section, saved=describe(after),
        key_env=key_env_for(asked.provider, after) if not switched else standard_env,
        provider_changed=switched,
        model_written=model if (model is not None and model != section.get("model")) else None,
        custom_endpoint=not switched and bool(section.get("base_url") or section.get("base_url_env")),
    )


def pinned_refusal(asked: ChangeRequest, plan: Plan) -> ModelRefusal | None:
    """A refusal when saving this would be a lie because an environment variable outranks the file."""
    pinned = pins()
    if "provider" in pinned and pinned["provider"] != asked.provider:
        return _overridden(PIN_PROVIDER, "provider")
    if "model" in pinned and plan.model_written is not None and plan.model_written != pinned["model"]:
        return _overridden(PIN_MODEL, "model")
    return None


def _overridden(variable: str, what: str) -> ModelRefusal:
    return ModelRefusal(
        409, MODEL_OVERRIDDEN,
        f"the {what} is set by {variable} in the daemon's environment, which outranks the config file, so saving "
        "this would not change it. Change or remove it there and restart the daemon. Nothing was written",
        source=variable)
