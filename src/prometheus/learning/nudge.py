"""PeriodicNudge — inject self-evaluation prompts every N rounds.

Invisible to the user. The nudge is an internal system message that asks
the agent to reflect on its approach and adjust if needed.

``interval`` counts COMPLETED ASSISTANT ROUNDS WITHIN ONE RUN, not user
turns — the counter is per ``run_loop`` invocation and resets with it. At
the default 15 that means an ordinary 1-3 round chat reply never nudges;
it fires only inside a long agentic run, which is what it is for.

Wire it through ``LoopContext.nudge`` and let ``run_loop`` drive it. It is
consumed as a request-only system-prompt addendum — the ``dict`` below is
NOT appended to the message list, and must not be: see
``prometheus.engine.agent_loop._maybe_periodic_nudge`` for the three
separate ways the old append-a-user-turn channel was wrong.

Usage:
    context = LoopContext(..., nudge=PeriodicNudge(interval=15))
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
    """Inject a self-evaluation nudge every *interval* rounds.

    Args:
        interval: Completed assistant rounds between nudges, counted within
            a single run (see the module docstring — NOT user turns).
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

        # Narrow the catch — see SkillCreator.from_config for the rationale
        # (the same Tier-1 hotfix shape PR #3 applied to the sibling
        # learning subsystems). Any exception type other than I/O or
        # YAML-parse should propagate.
        try:
            with open(Path(config_path).expanduser()) as fh:
                data = yaml.safe_load(fh) or {}
            learning = data.get("learning", {}) or {}
            interval = learning.get("nudge_interval", _DEFAULT_INTERVAL)
            enabled = learning.get("nudge_enabled", True)
        except (OSError, yaml.YAMLError) as exc:
            log.warning(
                "PeriodicNudge.from_config: failed to load %s (%s: %s); "
                "using default interval=%d, enabled=True",
                config_path, type(exc).__name__, exc, _DEFAULT_INTERVAL,
            )
            interval = _DEFAULT_INTERVAL
            enabled = True

        return cls(interval=interval, enabled=enabled)

    def maybe_inject(self, turn_count: int) -> dict | None:
        """Return a nudge payload if it's time, else None.

        Args:
            turn_count: Completed assistant rounds this run (1-indexed).

        Returns:
            ``{"role": "user", "content": ..., "_nudge": True}``, or None.
            The caller reads ``content`` and folds it into the per-call
            system prompt; the ``role`` key is vestigial from the old
            append-to-messages channel and is deliberately NOT honoured.
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
