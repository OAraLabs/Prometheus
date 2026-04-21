"""Credential pool — multi-key rotation with failover.

Donor pattern: NousResearch/hermes-agent agent/auxiliary_client.py (MIT).
Adapted for Prometheus: round-robin rotation, dead-key cooldown, health tracking.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class KeyStats:
    """Usage statistics for a single API key."""

    successes: int = 0
    failures: int = 0
    last_used: float = 0.0
    last_error: str = ""


class CredentialPool:
    """Manages multiple API keys per provider with rotation and failover.

    Usage::

        pool = CredentialPool(["sk-ant-key1", "sk-ant-key2"])
        key = pool.get_next()       # round-robin
        # ... use key in API call ...
        pool.report_success(key)
        # or
        pool.report_error(key, 429)  # rotate immediately

    Single key still works — pool is a no-op wrapper::

        pool = CredentialPool(["sk-ant-only-key"])
        key = pool.get_next()        # always returns the same key
    """

    def __init__(
        self,
        api_keys: list[str],
        dead_key_cooldown_seconds: int = 300,
    ) -> None:
        if not api_keys:
            raise ValueError("CredentialPool requires at least one API key")

        self._keys = list(api_keys)
        self._current_index = 0
        self._dead_keys: dict[str, float] = {}  # key -> death timestamp
        self._cooldown = dead_key_cooldown_seconds
        self.stats: dict[str, KeyStats] = {k: KeyStats() for k in self._keys}

    @property
    def active_count(self) -> int:
        """Number of currently active (non-dead) keys."""
        self._revive_expired()
        return sum(1 for k in self._keys if k not in self._dead_keys)

    def get_next(self) -> str:
        """Get the next active API key via round-robin. Skips dead keys.

        Raises RuntimeError if all keys are dead.
        """
        self._revive_expired()

        for _ in range(len(self._keys)):
            key = self._keys[self._current_index % len(self._keys)]
            self._current_index += 1
            if key not in self._dead_keys:
                self.stats[key].last_used = time.time()
                return key

        raise RuntimeError(
            f"All {len(self._keys)} API keys are dead. "
            f"Cooldown: {self._cooldown}s. Check your keys."
        )

    def report_success(self, key: str) -> None:
        """Record a successful API call."""
        if key in self.stats:
            self.stats[key].successes += 1

    def report_error(self, key: str, status_code: int) -> None:
        """Handle an API error.

        - 429 (rate limited): key stays active but index advances (next call uses different key)
        - 401/403 (invalid key): mark dead immediately
        - Other: log but keep active
        """
        if key in self.stats:
            self.stats[key].failures += 1
            self.stats[key].last_error = f"HTTP {status_code}"

        if status_code in (401, 403):
            self._dead_keys[key] = time.time()
            logger.warning(
                "API key marked dead (HTTP %d): %s...%s",
                status_code, key[:8], key[-4:],
            )
        elif status_code == 429:
            logger.info(
                "Rate limited (HTTP 429) — rotating to next key: %s...%s",
                key[:8], key[-4:],
            )
        else:
            logger.warning(
                "API error (HTTP %d) on key %s...%s",
                status_code, key[:8], key[-4:],
            )

    def _revive_expired(self) -> None:
        """Revive dead keys whose cooldown has expired."""
        now = time.time()
        revived = [
            k for k, death_time in self._dead_keys.items()
            if now - death_time >= self._cooldown
        ]
        for k in revived:
            del self._dead_keys[k]
            logger.info("Revived API key after cooldown: %s...%s", k[:8], k[-4:])
