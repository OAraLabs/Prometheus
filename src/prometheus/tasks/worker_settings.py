"""The one operator setting for workers: which model they run on.

``workers.model`` in prometheus.yaml seeds it; Beacon Settings → Worker model
(``PUT /api/workers/model``) changes it at runtime and the change is stored in
``<config dir>/worker_settings.json``, which wins over the yaml key — the
daemon never writes prometheus.yaml (the grant-writer rule), so a choice made
in Beacon needs a home of its own that survives a restart.

The value is a picker key from ``GET /api/models`` (``qwen``,
``qwen:qwen3.8-flash``, ``4090``, ``local``), or nothing: **Same as chat**, the
default, where every worker inherits its spawning conversation's route
(:mod:`prometheus.tasks.worker_route`). A key is validated against the catalog
when it is SET, so a typo never becomes the setting; its credential is checked
at SPAWN, where a missing key is refused with that reason.

Source: novel code for Prometheus, 2026-10-10.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

STATE_FILE_NAME = "worker_settings.json"


def load_worker_config() -> dict[str, Any]:
    """The config this process would read, for callers without one in hand."""
    from prometheus.config.load import load_config_resolved

    return load_config_resolved(
        None, subsystem="workers", substituting="built-in defaults", strict=False,
    )


#: What ``workers.timeout_minutes`` is when it is unset or unusable: an hour of WORKING time (a wait for a person's
#: approval does not count). Must equal the template value (config/prometheus.yaml.default).
DEFAULT_WORKER_TIMEOUT_MINUTES = 60


def worker_timeout_seconds(config: Mapping[str, Any] | None) -> int:
    """A worker's time limit, from ``workers.timeout_minutes``, in seconds.

    The owner's setting, not the model's: ``task_create`` no longer passes a model's ``timeout_seconds`` for an agent
    worker. A value that is not a whole number of minutes of at least 1 is a WARNING and the default, never a worker
    with no limit or one killed at once.
    """
    section = (config or {}).get("workers") or {}
    # LITERAL default on purpose: test_config_defaults_equality parses this call to compare it with the template.
    value = section.get("timeout_minutes", 60) if isinstance(section, Mapping) else 60
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        log.warning("workers.timeout_minutes must be a whole number of minutes, at least 1; got %r — using %d",
                    value, DEFAULT_WORKER_TIMEOUT_MINUTES)
        value = DEFAULT_WORKER_TIMEOUT_MINUTES
    return int(value) * 60


def _config_value(config: Mapping[str, Any] | None) -> str | None:
    section = (config or {}).get("workers") or {}
    value = section.get("model") if isinstance(section, Mapping) else None
    value = str(value).strip() if value else ""
    return value or None


class WorkerModelSetting:
    """Same as chat (None) or one picker key, with where the value came from."""

    def __init__(
        self, *, config: Mapping[str, Any] | None, path: Path | None = None,
    ) -> None:
        from prometheus.config.paths import config_dir_path

        self.config: dict[str, Any] = dict(config or {})
        self.path = path or (config_dir_path() / STATE_FILE_NAME)
        self._lock = threading.Lock()

    def _stored(self) -> tuple[bool, str | None]:
        """(a value is stored, the stored value). A broken file counts as none."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return False, None
        except (OSError, ValueError):
            log.warning("worker settings file %s is unreadable; using workers.model",
                        self.path)
            return False, None
        if not isinstance(data, dict) or "model" not in data:
            return False, None
        value = data.get("model")
        return True, (str(value).strip() or None) if value else None

    def current(self) -> str | None:
        stored, value = self._stored()
        return value if stored else _config_value(self.config)

    @property
    def source(self) -> str:
        """``settings`` (Beacon), ``config`` (workers.model) or ``default``."""
        if self._stored()[0]:
            return "settings"
        return "config" if _config_value(self.config) else "default"

    def set(self, key: str | None) -> None:
        """Store a choice; None returns to Same as chat. Unknown keys raise."""
        from prometheus.tasks.worker_route import is_offered

        key = (key or "").strip() or None
        if key is not None and not is_offered(key, self.config):
            raise ValueError(f"{key!r} is not a model this daemon offers (see GET /api/models)")
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"model": key}), encoding="utf-8")
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)


_SETTING: WorkerModelSetting | None = None
_SETTING_LOCK = threading.Lock()


def get_worker_model_setting() -> WorkerModelSetting:
    """The process-wide holder. The daemon installs one built on its own config
    (:func:`set_worker_model_setting`); anything else gets one on the config
    this process would read, rebuilt if the config directory has changed."""
    global _SETTING
    from prometheus.config.paths import config_dir_path

    with _SETTING_LOCK:
        if _SETTING is None or _SETTING.path.parent != config_dir_path():
            _SETTING = WorkerModelSetting(config=load_worker_config())
        return _SETTING


def set_worker_model_setting(setting: WorkerModelSetting | None) -> None:
    global _SETTING
    with _SETTING_LOCK:
        _SETTING = setting
