"""Which route a spawned worker runs on — decided in ONE place.

WHY THIS EXISTS (2026-10-09, #704)
----------------------------------
A conversation on qwen3.8-max spawned two ``task_create(type=local_agent)``
workers and both were served by the local 27B with a 32k window. The spawn
tools knew nothing about the turn that called them, so the worker — a fresh
``python -m prometheus`` — took the config's ``model:``. A ``--model`` naming
the cloud model was overwritten by the local box's ``/v1/models``; and the
obvious repair, passing ``--provider`` as well, would have kept the config's
local ``base_url`` and sent the cloud request, key included, to the box.

THE RULE
--------
* A worker inherits the spawning turn's route: the picker key its session
  chose (``qwen``, ``qwen:qwen3.8-flash``, ``4090``) or :data:`PRIMARY_KEY`
  for the configured primary. The KEY is all that crosses into the child; the
  child resolves it again through the same catalog the model picker uses
  (:func:`resolve_route`), so the endpoint and credential come from the route
  and its own environment, never from the command line.
* ``workers.model`` (Beacon Settings → Worker model) is the one way to choose
  a different route; see :mod:`prometheus.tasks.worker_settings`.
* The agent may not choose. A ``model`` argument is refused unless it names
  the route the setting already chose.
* A route that cannot be served is refused with its reason. Nothing is ever
  substituted for it.

Source: novel code for Prometheus, 2026-10-10.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

#: The picker key of the configured primary (``model:``) — the catalog's
#: ``local`` row and the backend registry's reserved name.
PRIMARY_KEY = "local"

#: Providers that serve from a box this install runs (no cloud credential).
_LOCAL_PROVIDERS = frozenset({"llama_cpp", "ollama", "lm_studio", "vllm"})

#: Where the operator chooses a different route for every worker.
SETTING_HINT = "Settings → Worker model (workers.model)"


class WorkerRouteRefused(ValueError):
    """A worker route that cannot be served, with the reason in the message."""


@dataclass(frozen=True)
class WorkerPlan:
    """The route a worker will run on, and the sentence the spawn result says.

    Public facts only — never a URL or a credential: this is what the spawn
    tool tells the model and what the task record keeps.
    """

    key: str
    provider: str
    model: str
    backend: str | None
    source: str                 # "parent" | "setting"
    worker_window: int | None
    parent_window: int | None
    #: The spawning turn's own key, when it has one — what a retry on the
    #: parent's route uses.
    parent_key: str | None = None

    def record(self) -> dict[str, str]:
        """The task-record metadata for this plan (strings, public facts only)."""
        out = {"worker_route": self.key, "worker_route_source": self.source}
        if self.parent_key:
            out["parent_route"] = self.parent_key
        if self.worker_window:
            out["worker_window"] = str(self.worker_window)
        if self.parent_window:
            out["parent_window"] = str(self.parent_window)
        return out

    @property
    def label(self) -> str:
        if self.backend and self.backend != PRIMARY_KEY:
            return f"{self.model} on {self.backend}"
        if self.key == PRIMARY_KEY:
            return f"{self.model} (the configured primary)"
        return f"{self.model} ({self.provider})"

    @property
    def smaller_window(self) -> bool:
        return (
            self.worker_window is not None
            and self.parent_window is not None
            and self.worker_window < self.parent_window
        )

    def note(self) -> str:
        """What the spawn result says about the route, in plain words."""
        if self.source == "setting":
            text = f"Runs on {self.label}, the worker model set in {SETTING_HINT}."
        else:
            text = f"Runs on {self.label}, the same model as this conversation."
        if self.smaller_window:
            text += (
                f" Its context window is {self.worker_window:,} tokens, smaller than "
                f"this conversation's {self.parent_window:,}."
            )
        return text


# ---------------------------------------------------------------------------
# key → provider config (used by the CHILD, and to validate in the parent)
# ---------------------------------------------------------------------------

def resolve_route(
    key: str, config: Mapping[str, Any] | None, *, check_credential: bool = True,
) -> dict[str, Any]:
    """The provider config a picker key names, or :class:`WorkerRouteRefused`.

    The same resolvers the model picker and the boot restore use: the
    configured primary for :data:`PRIMARY_KEY`, the backend registry for a
    box (``4090``, ``mini:qwen2.5:7b-instruct``), the vetted preset list for a
    cloud key, whose credential is checked the way ``/claude`` checks it.

    The result may carry an inline ``api_key`` from the operator's config, so
    it stays in the process that resolved it: never logged, never persisted,
    never put on a command line.
    """
    from prometheus.router.model_router import (
        is_backend_key,
        resolve_backend_target,
        resolve_model_target,
        split_model_key,
    )

    cfg: dict[str, Any] = dict(config or {})
    key = (key or "").strip()
    if not key:
        raise WorkerRouteRefused("no route was named")
    if key == PRIMARY_KEY:
        primary = dict(cfg.get("model") or {})
        primary.setdefault("provider", "llama_cpp")
        return primary
    if is_backend_key(key, cfg):
        name, model = split_model_key(key)
        target = resolve_backend_target(name, cfg)
        if target is None:
            raise WorkerRouteRefused(f"{key!r}: backend {name!r} is not configured")
        if model:
            target["model"] = model
        return target
    target = resolve_model_target(key, cfg)
    if target is None:
        raise WorkerRouteRefused(
            f"{key!r} is not a model this daemon offers (see GET /api/models)")
    api_key_env = str(target.get("api_key_env") or "")
    if api_key_env and check_credential:
        from prometheus.providers.credentials import credential_status

        if credential_status(str(target.get("provider", "")), api_key_env)["mode"] is None:
            raise WorkerRouteRefused(
                f"{key!r} ({target.get('provider')}) needs {api_key_env}, which is not set")
    return dict(target)


def is_offered(key: str, config: Mapping[str, Any] | None) -> bool:
    """True when *key* is a route the catalog offers, credential or not.

    What the setting is validated against: a choice may be made before its
    key is configured, and is refused at spawn (with that reason) until it is.
    """
    try:
        resolve_route(key, config, check_credential=False)
    except WorkerRouteRefused:
        return False
    return True


# ---------------------------------------------------------------------------
# (provider, model, backend) → key — for a turn the router did not label
# ---------------------------------------------------------------------------

def route_key_for(
    provider: str | None,
    model: str | None,
    backend: str | None,
    config: Mapping[str, Any] | None,
) -> str | None:
    """The picker key that serves *model* on *provider*/*backend*, or None.

    For a turn whose route the router did not label with a key (a task rule,
    smart routing, a mid-turn fallback): the catalog row that serves exactly
    this, or nothing. Never a near match.
    """
    from prometheus.router.model_router import (
        OVERRIDE_PRESETS,
        SLASH_COMMAND_NAMES,
        resolve_backend_target,
        resolve_model_choices,
        resolve_slash_command_target,
    )

    cfg: dict[str, Any] = dict(config or {})
    if backend:
        if backend == PRIMARY_KEY:
            return PRIMARY_KEY
        target = resolve_backend_target(backend, cfg)
        if target is None:
            return None
        if not model or model == target.get("model"):
            return backend
        return f"{backend}:{model}"
    primary = cfg.get("model") or {}
    primary_provider = primary.get("provider", "llama_cpp")
    if provider and provider == primary_provider:
        # A local primary serves whatever its box reports, which is the name
        # the turn carries; a cloud primary must match its configured model.
        if provider in _LOCAL_PROVIDERS or not model or model == primary.get("model"):
            return PRIMARY_KEY
    for preset in dict.fromkeys((*SLASH_COMMAND_NAMES, *OVERRIDE_PRESETS)):
        target = resolve_slash_command_target(preset, cfg)
        if not target or target.get("provider") != provider:
            continue
        if not model or model == target.get("model"):
            return preset
        if model in resolve_model_choices(preset, cfg):
            return f"{preset}:{model}"
    return None


# ---------------------------------------------------------------------------
# The decision the spawn tools take
# ---------------------------------------------------------------------------

def _names(requested: str, key: str, target: Mapping[str, Any]) -> bool:
    """Does the agent's *requested* model name the route *key* already chose?"""
    from prometheus.router.model_router import split_model_key

    requested = requested.strip()
    return requested in {key, split_model_key(key)[0], str(target.get("model") or "")}


def plan_worker_route(
    metadata: Mapping[str, Any] | None,
    *,
    requested_model: str | None,
    config: Mapping[str, Any] | None,
    setting_key: str | None,
) -> WorkerPlan:
    """Decide the worker's route, or raise :class:`WorkerRouteRefused`.

    *metadata* is the spawning tool call's: the loop puts the turn's
    ``serving_route`` (key, provider, model, backend) there, and a
    ``context_window_for(model, backend)`` resolver when the turn has a
    compactor.
    """
    meta = metadata or {}
    serving = dict(meta.get("serving_route") or {})
    parent_key: str | None = (serving.get("key") or route_key_for(
        serving.get("provider"), serving.get("model"), serving.get("backend"), config)
    ) if serving else PRIMARY_KEY
    if setting_key:
        key, source = setting_key, "setting"
    else:
        if requested_model:
            raise WorkerRouteRefused(
                f"An agent can't choose a worker's model (asked for {requested_model!r}). "
                f"Workers run on this conversation's model; the operator can pick one "
                f"worker model in {SETTING_HINT}. Spawn again without 'model'.")
        if not parent_key:
            raise WorkerRouteRefused(
                f"This conversation's model ({serving.get('provider') or '?'}/"
                f"{serving.get('model') or '?'}) can't be handed to a worker: it is not "
                f"a model the catalog offers. Nothing was spawned.")
        # With no serving route at all (a direct call, not a loop's), the
        # parent is the configured primary, said as such.
        key = parent_key
        source = "parent"

    target = resolve_route(key, config)
    if setting_key and requested_model and not _names(requested_model, key, target):
        raise WorkerRouteRefused(
            f"An agent can't choose a worker's model (asked for {requested_model!r}); "
            f"workers run on {key!r}, the worker model set in {SETTING_HINT}. "
            f"Spawn again without 'model'.")

    window_for = meta.get("context_window_for")
    provider = str(target.get("provider") or "")
    model = str(target.get("model") or serving.get("model") or "")
    backend = target.get("backend") or (PRIMARY_KEY if key == PRIMARY_KEY else None)
    return WorkerPlan(
        key=key, provider=provider, model=model, backend=backend, source=source,
        worker_window=_window(window_for, model, backend),
        parent_window=_window(window_for, serving.get("model"), serving.get("backend")),
        parent_key=parent_key,
    )


def plan_spawn(
    metadata: Mapping[str, Any] | None, *, requested_model: str | None,
) -> WorkerPlan:
    """:func:`plan_worker_route` with the setting and config this process holds."""
    from prometheus.tasks.worker_settings import get_worker_model_setting

    setting = get_worker_model_setting()
    return plan_worker_route(
        metadata, requested_model=requested_model, config=setting.config,
        setting_key=setting.current(),
    )


def _window(
    window_for: Callable[..., int | None] | None, model: str | None, backend: str | None,
) -> int | None:
    if window_for is None or not model:
        return None
    try:
        value = window_for(model, backend)
    except Exception:  # noqa: BLE001 — a window we cannot size is reported as unknown
        return None
    return int(value) if value else None
