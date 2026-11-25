"""Run a memory-tool call as the agent loop runs it once the owner approved it (GHSA-f68r-xfv4-83fr).

The loop sets two things around an approved call: ``OWNER_APPROVED`` (a yes came from the approval channel) and
``APPROVED_CARD`` (the exact card the owner read). The tool refuses a change without both, and refuses when the card
no longer matches the file. Tests that call ``MemoryTool.execute`` directly use this to stand for that approval.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator


@contextlib.contextmanager
def owner_approved(tool, arguments) -> Iterator[None]:
    """Inside the block, *tool* runs *arguments* as an owner-approved call, with the card built from the file now."""
    from prometheus.engine.tool_context import APPROVED_CARD, OWNER_APPROVED

    t1 = OWNER_APPROVED.set(True)
    t2 = APPROVED_CARD.set(tool.requires_owner_approval(arguments))
    try:
        yield
    finally:
        OWNER_APPROVED.reset(t1)
        APPROVED_CARD.reset(t2)


def approve_every_call(monkeypatch, tool_cls) -> None:
    """Patch *tool_cls*.execute so every call runs as owner-approved, card included (for end-to-end tool tests)."""
    real = tool_cls.execute

    async def approved(self, arguments, context):
        with owner_approved(self, arguments):
            return await real(self, arguments, context)

    monkeypatch.setattr(tool_cls, "execute", approved)
