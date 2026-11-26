"""Vendored runtime support for the generated ACP v2 types (Apache-2.0).

Source: agentclientprotocol/agent-client-protocol, schema-v2.0.0-beta.4 era
(scripts and src/acp runtime helpers the generated schema.py imports).
Licence: Apache-2.0 — full text in LICENSE-APACHE-2.0 beside this file.
Modified: import paths rewritten to prometheus.acp._vendor; nothing else.

These are the upstream deserialization semantics (default-on-error,
skip-invalid-items, reserved-tag guards for open unions). They travel with the
generated types so the models behave exactly as the upstream SDK's would.
"""