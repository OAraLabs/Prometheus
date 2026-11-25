"""Version negotiation — v2 only, pinned to one schema tag.

The initialize handshake (schema $defs/InitializeRequest / InitializeResponse):
the client offers its highest version; the agent answers with the highest it
supports AT OR BELOW the offer, or fails with -32022. Prometheus supports
exactly v2 today. A v1-only peer therefore gets -32022 and keeps today's
behavior — no ACP session exists for it, nothing in the daemon changed.

Capabilities are advertised honestly: this core maps sessions and prompts; it
does not yet advertise auth, providers or any unstable (RFD-gated) surface.

Source: novel code for Prometheus, 2026-11-25.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from prometheus.acp import types
from prometheus.acp.framing import UNSUPPORTED_PROTOCOL_VERSION, FrameError

#: The only protocol version this core speaks.
PROTOCOL_VERSION = 2

#: The pinned schema tag the generated types were built from. Moving to a new
#: tag (or to stable) is one PR: regenerate types, bump this, re-run the TCK.
SCHEMA_TAG = "schema-v2.0.0-beta.4"


def negotiate(params: dict[str, Any], *, agent_name: str, agent_version: str,
              agent_capabilities: dict[str, Any] | None = None) -> types.InitializeResponse:
    """Validate an initialize request and build the response.

    Raises FrameError(-32602) for malformed params and
    FrameError(-32022) when the peer cannot speak v2.
    """
    try:
        req = types.InitializeRequest.model_validate(params)
    except ValidationError as exc:  # params are the system boundary
        raise FrameError(-32602, f"invalid initialize params: {exc}") from exc
    if req.protocol_version != PROTOCOL_VERSION:
        # v2 rule: answer with the highest supported version <= the offer.
        # We support only 2, so any offer != 2 has no such version.
        raise FrameError(
            UNSUPPORTED_PROTOCOL_VERSION,
            f"unsupported protocol version {req.protocol_version}; this agent speaks v2 only",
        )
    return types.InitializeResponse(
        protocol_version=PROTOCOL_VERSION,
        info=types.Implementation(name=agent_name, version=agent_version),
        capabilities=types.AgentCapabilities.model_validate(agent_capabilities or {}),
    )