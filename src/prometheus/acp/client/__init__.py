"""The ACP client side — driving an OUTSIDE agent over ACP v2 (experimental).

Prometheus as an ACP CLIENT: it launches an outside agent (Claude Code first)
as a subprocess speaking ACP v2 over stdio, sends prompts, streams the
agent's updates, and answers the agent's permission requests THROUGH THE
SAME GATE as Prometheus's own tool calls. Nothing the outside agent asks
bypasses the gate; anything the gate would escalate goes to Beacon, and the
outside agent runs in its strictest ask-every-time mode (the approval rule,
decided 2026-11-25).

Layers:
- ``connection`` — the client half of the NDJSON JSON-RPC connection:
  requests out, responses matched by id, notifications dispatched, the
  agent's own requests (permission) answered through a handler.
- ``registry``   — the outside agents Prometheus may drive: a named,
  allowlisted command per agent. Nothing outside the registry runs.
- ``driver``     — one driven session: initialize (v2 only), session/new,
  prompt, cancel; permission requests routed to the gate.

OFF by default: nothing here is imported unless ``acp.enabled`` is true,
and the registry ships EMPTY — an operator names each outside agent
explicitly. Hard-disabled for machine/benchmark sessions.

Source: novel code for Prometheus, 2026-11-25.
"""