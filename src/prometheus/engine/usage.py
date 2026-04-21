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

    @property
    def total_tokens(self) -> int:
        """Return the total number of accounted tokens."""
        return self.input_tokens + self.output_tokens
