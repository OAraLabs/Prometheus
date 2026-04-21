"""PeriodicNudge — inject self-evaluation prompts every N turns.

Invisible to the user. The nudge is an internal system message that asks
the agent to reflect on its approach and adjust if needed.

Usage:
    nudge = PeriodicNudge(interval=15)
    # Called each turn from the agent loop:
    injection = nudge.maybe_inject(turn_count, messages)
    if injection:
        messages.append(injection)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

_DEFAULT_INTERVAL = 15
_MAX_NUDGE_TOKENS = 200

_NUDGE_PROMPT = (
    "Pause and self-evaluate: Are you on the right track? "
    "Consider: (1) Is the current approach efficient? "
    "(2) Are there simpler alternatives you haven't tried? "
    "(3) Have you missed any edge cases? "
    "(4) Should you ask the user for clarification? "
    "Adjust your strategy if needed, then continue."
)


@dataclass
class PeriodicNudge:
    """Inject a self-evaluation nudge every *interval* turns.

    Args:
        interval: Number of user turns between nudges.
        prompt: Custom nudge prompt (must stay under 200 tokens).
        enabled: Set False to disable without removing from the loop.
    """

    interval: int = _DEFAULT_INTERVAL
    prompt: str = _NUDGE_PROMPT
    enabled: bool = True
    _nudge_count: int = field(default=0, init=False, repr=False)

    @classmethod
    def from_config(cls, config_path: str | None = None) -> PeriodicNudge:
        """Build from prometheus.yaml learning.nudge_interval."""
        import yaml
        from pathlib import Path

        if config_path is None:
            from prometheus.config.defaults import DEFAULTS_PATH
            config_path = str(DEFAULTS_PATH)

        try:
            with open(Path(config_path).expanduser()) as fh:
                data = yaml.safe_load(fh)
            learning = data.get("learning", {})
            interval = learning.get("nudge_interval", _DEFAULT_INTERVAL)
            enabled = learning.get("nudge_enabled", True)
        except (OSError, Exception):
            interval = _DEFAULT_INTERVAL
            enabled = True

        return cls(interval=interval, enabled=enabled)

    def maybe_inject(self, turn_count: int) -> dict | None:
        """Return a nudge message dict if it's time, else None.

        Args:
            turn_count: Current number of completed user turns (1-indexed).

        Returns:
            A message dict ``{"role": "user", "content": ..., "_nudge": True}``
            suitable for appending to the message list, or None.
        """
        if not self.enabled:
            return None
        if turn_count <= 0 or turn_count % self.interval != 0:
            return None

        self._nudge_count += 1
        log.debug("PeriodicNudge: injecting nudge #%d at turn %d", self._nudge_count, turn_count)

        return {
            "role": "user",
            "content": f"[system-internal] {self.prompt}",
            "_nudge": True,
            "_nudge_number": self._nudge_count,
        }

    @property
    def nudge_count(self) -> int:
        """Total nudges injected this session."""
        return self._nudge_count

    def reset(self) -> None:
        """Reset the nudge counter (e.g. on session restart)."""
        self._nudge_count = 0
