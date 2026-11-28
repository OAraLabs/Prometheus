# Prometheus Extension Hooks (Minimal Companion Contract)

Status: implemented on `feat/extension-hooks` (opt-in via `extensions:` in `prometheus.yaml`).

## Why out-of-process Extensions
Extensions like **OAra Gym** (local fine-tuning & gating) and **OAra Voice** (two-way speech) carry heavy GPU/ML/audio dependencies and control optional host resources. Running them as separately installed, separately permissioned companion services keeps `OAraLabs/Prometheus` lean and enforces the **zero in-process third-party code** invariant.

## Configuration (`prometheus.yaml`)
Disabled by default. Each Extension declares its read endpoint and (optionally) a separate write/control endpoint:

```yaml
extensions:
  gym:
    enabled: true
    kind: gym
    url: http://127.0.0.1:18097
    token_env: OARA_GYM_READ_TOKEN
    control_url: http://127.0.0.1:18098
    control_token_env: OARA_GYM_CONTROL_TOKEN
    status_path: /v1/status
    timeout_s: 5.0
```

## Security invariants
1. **Operator-only routes:** Every `/api/extensions/*` route is classified `OPERATOR_ONLY` in `prometheus.web.route_access`. Ordinary chat-scoped phones receive `403 operator_only`.
2. **Loopback by default:** Non-loopback URLs are rejected at load time unless `allow_remote: true` is set explicitly; embedded URL credentials are always forbidden.
3. **Separated read vs. control:** Read calls (`manifest`, `health`, `status`) hit `url` + `token_env` only. Control calls (`control/intent`, `control/confirm`) hit `control_url` + `control_token_env` only and return `403 control_not_configured` when `control_url` is unset.
4. **Zero address/secret leakage:** Public responses never expose backend URLs, bearer tokens, or env var names.
5. **Consent-gated trace export:** `GET /api/extensions/traces/export` refuses when `infrastructure.telemetry_enabled: false`, returns metadata/counts only by default, and restricts `include_text=true` to loopback callers.

## Companion service contract (what an Extension implements)
- `GET /health` → `200 {"status": "ok"}`
- `GET /integration/manifest` → `200 {"schema": "prometheus-extension-manifest/1", "name": "...", "version": "...", "kind": "...", "capabilities": [...], "scopes": [...]}`
- `GET /v1/status` → read-only JSON status object (counts, enums, timestamps — no raw trace text)
- Optional separate control listener:
  - `POST /v1/control/intent` → dry-run plan returning `intent_id`, `plan_sha256`, summary, `expires_at`
  - `POST /v1/control/confirm` → executes only when `intent_id` and `plan_sha256` match an unexpired, unconfirmed intent
