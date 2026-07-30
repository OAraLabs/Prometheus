# Source: OpenHarness (HKUDS/OpenHarness)
# Original: src/openharness/api/usage.py
# License: MIT
# Modified: renamed module path only (openharness → prometheus)

"""Usage tracking models."""

from __future__ import annotations

from pydantic import BaseModel


class UsageSnapshot(BaseModel):
    """Token usage returned by the model provider."""

    input_tokens: int = 0
    output_tokens: int = 0
    # Prompt-cache accounting. Providers that do automatic prefix caching report
    # how much of the prompt was served from cache; the agent loop re-sends a
    # near-identical prefix every round, so this is the difference between
    # paying full price for the context 19 times and paying once.
    #
    # ``None`` = the provider said nothing about caching (not "zero cached") —
    # the distinction matters, because 0 is a finding and None is silence.
    # Shapes seen in the wild:
    #   OpenAI-compat / xAI: usage.prompt_tokens_details.cached_tokens
    #   Anthropic:           usage.cache_read_input_tokens / cache_creation_input_tokens
    cached_input_tokens: int | None = None
    cache_write_tokens: int | None = None

    @property
    def total_tokens(self) -> int:
        """Return the total number of accounted tokens."""
        return self.input_tokens + self.output_tokens

    @property
    def uncached_input_tokens(self) -> int | None:
        """Input tokens actually processed (billed at full rate), or None when
        the provider reported no cache information."""
        if self.cached_input_tokens is None:
            return None
        return max(0, self.input_tokens - self.cached_input_tokens)

    @property
    def cache_hit_ratio(self) -> float | None:
        """Fraction of the prompt served from cache, or None when unreported."""
        if self.cached_input_tokens is None or self.input_tokens <= 0:
            return None
        return self.cached_input_tokens / self.input_tokens
