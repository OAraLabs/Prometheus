"""ACP — the Agent Client Protocol at the edge (experimental, off by default).

Prometheus speaks ACP v2 to editors and outside agents; its internal session
model is NOT ACP (docs/design/acp-core.md). Nothing in this package runs unless
``acp.enabled`` is true in prometheus.yaml, and no existing route, frame or
session behavior changes when it is false.

Layers:
- ``types``      — Pydantic models GENERATED from the pinned schema tag
                   ``schema-v2.0.0-beta.4`` (schema/schema.json, sha256-pinned).
                   Never hand-edit; regenerate per the design note.
- ``framing``    — JSON-RPC 2.0 over newline-delimited JSON (v2: no batching).
- ``negotiation``— initialize handshake; v2 only. A v1-only peer is refused
                   with -32022 and keeps today's behavior (no ACP).
- ``sessions``   — the mapping between ACP sessionIds and Prometheus sessions.

Source: novel code for Prometheus, 2026-11-25.
Vendored Apache-2.0 pieces live in ``_vendor/`` with their licence text.
"""