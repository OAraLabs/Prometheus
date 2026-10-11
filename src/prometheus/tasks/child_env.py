"""The environment a daemon-built task child starts with: provider keys, and no daemon secret.

A worker, a spawned agent or a coding run is a ``python -m prometheus`` child the
daemon builds itself (``floored=False`` in tasks/manager.py). It used to start
with ``{**os.environ, **overlay}``, or, with no overlay, inherit ``os.environ``
whole. ``config/api_token.py`` puts ``PROMETHEUS_API_TOKEN`` there, and the
gateway tokens arrive the same way from the env file. On Linux the block a
process STARTED with stays readable at ``/proc/<pid>/environ``, so the model's
shell inside a worker could read the master token from ``/proc/$PPID/environ``
and approve its own requests (tasks/worker_auth.py says the same of the
worker's approval token, which is why that one travels on stdin).

So the child gets the bash tool's scrub (security/env_scrub.py: every
secret-shaped name removed), with one class of secret put back: the provider
keys a model call reads. :func:`provider_key_names` lists them from the tables
that already declare them, plus every ``api_key_env`` the config names.

Source: novel code for Prometheus, 2026-10-11.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: The config sections whose env-mapped values are provider keys
#: (config/env_override.py maps each variable to a config path).
_PROVIDER_SECTIONS = frozenset({"providers", "image_generation", "video_generation"})


def _api_key_envs(node: Any) -> Iterator[str]:
    """Every string value of an ``api_key_env`` key, at any depth of *node*."""
    if isinstance(node, Mapping):
        for key, value in node.items():
            if key == "api_key_env" and isinstance(value, str) and value.strip():
                yield value.strip()
            else:
                yield from _api_key_envs(value)
    elif isinstance(node, list):
        for item in node:
            yield from _api_key_envs(item)


def _read_config(config_path: str | None) -> dict[str, Any]:
    if not config_path:
        return {}
    try:
        import yaml

        data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - an unreadable config names no extra key
        log.debug("child env: could not read %s for api_key_env names", config_path,
                  exc_info=True)
        return {}
    return data if isinstance(data, dict) else {}


def provider_key_names(config_path: str | None = None) -> frozenset[str]:
    """The env-var NAMES of every provider key a model or media call may read.

    From the tables that declare them: each cloud provider's default variable
    (providers/registry.py), the key catalog (presets, with the config's
    ``slash_commands`` merged over them, and the media tools), the provider
    entries of config/env_override.py, and every ``api_key_env`` in the config
    at *config_path* (a route or fallback may name its own).
    """
    from prometheus.config.env_override import ENV_OVERRIDES, SECRET_FILE_VARS
    from prometheus.providers.key_catalog import provider_key_services
    from prometheus.providers.registry import CLOUD_DEFAULTS

    config = _read_config(config_path)
    names: set[str] = set()
    names.update(str(d["default_env"]) for d in CLOUD_DEFAULTS.values() if d.get("default_env"))
    for service in provider_key_services(config):
        names.update(service.env_vars)
    for table in (ENV_OVERRIDES, SECRET_FILE_VARS):
        names.update(var for var, path in table.items() if path[0] in _PROVIDER_SECTIONS)
    names.update(_api_key_envs(config))
    # A config's api_key_env is the operator's text. Whatever it names, the
    # daemon's own credentials stay out: the API token and the gateway tokens.
    from prometheus.config.api_token import TOKEN_ENV_VAR, TOKEN_FILE_ENV_VAR

    names -= {TOKEN_ENV_VAR, TOKEN_FILE_ENV_VAR}
    for table in (ENV_OVERRIDES, SECRET_FILE_VARS):
        names -= {var for var, path in table.items() if path[0] == "gateway"}
    return frozenset(names)


def task_child_env(
    overlay: Mapping[str, str] | None = None,
    *,
    config_path: str | None = None,
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The complete environment for a daemon-built child (``env=`` replaces, never merges).

    *base* defaults to ``os.environ``. *overlay* (the manager's in-memory grant:
    a key the caller passed, the worker's task id and approval URL) is applied
    last.
    """
    from prometheus.security.env_scrub import scrubbed_env

    env = scrubbed_env(
        dict(base) if base is not None else None,
        extra_keep=tuple(sorted(provider_key_names(config_path))),
    )
    if overlay:
        env.update(overlay)
    return env
