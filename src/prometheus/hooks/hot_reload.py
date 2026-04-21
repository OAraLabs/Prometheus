"""Hook hot reload — detects config changes and rebuilds the registry.

Donor pattern: HKUDS/OpenHarness src/openharness/hooks/hot_reload.py (MIT).
Adapted for Prometheus: polls prometheus.yaml mtime, rebuilds HookRegistry on change.
Optional background watcher via asyncio task.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from prometheus.hooks.loader import load_hook_registry
from prometheus.hooks.registry import HookRegistry

logger = logging.getLogger(__name__)


class HookReloader:
    """Lazy mtime-based hook reloader.

    Checks the config file's mtime when ``current_registry()`` is called.
    If the file changed, rebuilds the registry from the ``hooks:`` section.
    Optionally runs a background polling loop via ``start_watching()``.

    Usage::

        reloader = HookReloader(Path("config/prometheus.yaml"))
        registry = reloader.current_registry()  # lazy check
        # or
        await reloader.start_watching(interval=5, on_reload=executor.update_registry)
    """

    def __init__(self, config_path: Path) -> None:
        self._config_path = config_path
        self._last_mtime_ns: int = -1
        self._registry = HookRegistry()
        self._watch_task: asyncio.Task | None = None

    def current_registry(self) -> HookRegistry:
        """Return the latest registry, reloading if the config file changed."""
        try:
            mtime = self._config_path.stat().st_mtime_ns
        except FileNotFoundError:
            if self._last_mtime_ns != -1:
                logger.info("Hook config deleted — resetting registry")
                self._registry = HookRegistry()
                self._last_mtime_ns = -1
            return self._registry

        if mtime != self._last_mtime_ns:
            self._reload()
            self._last_mtime_ns = mtime

        return self._registry

    def _reload(self) -> None:
        """Rebuild the registry from the config file."""
        try:
            import yaml

            text = self._config_path.read_text(encoding="utf-8")
            config = yaml.safe_load(text) or {}
            hooks_section = config.get("hooks", {})
            self._registry = load_hook_registry(hooks_section)
            logger.info(
                "Hooks reloaded from %s (%d events configured)",
                self._config_path,
                sum(len(v) for v in hooks_section.values()),
            )
        except Exception as exc:
            logger.warning("Failed to reload hooks: %s", exc)

    async def start_watching(
        self,
        interval: float = 5.0,
        on_reload: callable | None = None,
    ) -> None:
        """Start a background loop that polls for config changes.

        Args:
            interval: seconds between polls.
            on_reload: callback invoked with the new HookRegistry on each reload.
        """
        async def _poll() -> None:
            while True:
                old_mtime = self._last_mtime_ns
                registry = self.current_registry()
                if self._last_mtime_ns != old_mtime and on_reload is not None:
                    try:
                        on_reload(registry)
                    except Exception as exc:
                        logger.warning("on_reload callback failed: %s", exc)
                await asyncio.sleep(interval)

        self._watch_task = asyncio.create_task(_poll())

    def stop_watching(self) -> None:
        """Cancel the background polling task."""
        if self._watch_task and not self._watch_task.done():
            self._watch_task.cancel()
            self._watch_task = None
