# HTTP & WebSocket API

Prometheus exposes two programmatic surfaces: a FastAPI REST server on **:8005** (`src/prometheus/web/server.py`) and a WebSocket bridge on **:8010** (`src/prometheus/web/ws_server.py`) that streams chat and subsystem events in real time. Both are what Beacon (the web/desktop UI) talks to; anything Beacon can do, you can do with curl or a WebSocket client.

[← README](../../README.md)

## Authentication

Every `/api/*` route requires a bearer token:

```
Authorization: Bearer $PROMETHEUS_API_TOKEN
```

- The token is minted automatically on first daemon start (or by the setup wizard) and stored in `~/.config/prometheus/env`.
- Retrieve or invalidate it with the CLI: `oara token show` | `oara token rotate`.
- Requests with a missing or wrong token get a `401 {"error": "unauthorized — set Authorization: Bearer <token>"}`.
- Two routes under `/api/` answer without a token, by design, and each does its own checking: `GET /api/hello` (below) and, on the Mac app install only, `POST /api/pair/local` (same-Mac pairing; it is a 404 anywhere else). The exact list is `web/public_routes.py`, and a test fails if any other route answers without a token. `GET /health` lives outside `/api/` precisely so external monitors can poll it without credentials, and the JSON API index served at `/` is likewise outside the bearer gate.

Example:

```bash
curl -s http://localhost:8005/api/status \
  -H "Authorization: Bearer $PROMETHEUS_API_TOKEN"
```

## REST reference

All paths below are served on `:8005`. `{id}` placeholders are path parameters.

This section is the **curated** reference — what each endpoint is for and
what its answer means. For the **complete** list, generated from the live
app and kept current by CI, see
[the route reference](../reference/routes.md).

### Status & sessions

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Unauthenticated JSON API index (service name, version, endpoint list) |
| GET | `/health` | Unauthenticated liveness/staleness probe |
| GET | `/api/hello` | Unauthenticated "is there a Prometheus here" for a device that has no address or token yet. Exactly six fields and nothing else: `v` (version), `name` (what this Prometheus calls itself: `pairing.display_name`, else the computer's name), `agent` (the assistant's name), `fp` (first 16 hex characters of the SHA-256 of the instance key's public half; empty until the key exists, a display hint and never a trust anchor), `pair` (how a client can join: `token`, `code` in setup mode, `none` when the daemon has no token) and `tls` — **whether the port this request arrived on speaks TLS**, so the same route answers `true` from `:8006` and `false` from `:8005`. No CORS headers; a request with an `Origin` header (a web page) is refused with `400 browser_not_allowed`; 60 requests a minute per peer address (never `X-Forwarded-For`), then `429` with `Retry-After`; never cached. Served in setup mode too. Contract: `docs/PAIRING-APPROVAL-API.md`. |
| POST | `/api/restart` | Operator only. Answers `202 {"restarting": true}`, then the daemon shuts down gracefully and starts again in the same process (it re-execs its own command line), so a saved setting such as the network mode takes effect without anyone touching the machine. Every socket drops; wait for `GET /api/hello` to answer with the same `fp`. `503 restart_unavailable` from a process that was not started as the daemon. The restart runs with the environment the daemon first started with, so the env file is read again (a rotated token takes effect). Refused to a web page: an `Origin` header is `400 browser_not_allowed`. |
| GET | `/api/status` | Model, uptime, tools, memory, subsystem states — plus `node_pub` (the node's Ed25519 public key) and `instance_id` (the vault's UUID), both `null` until identity exists. Bearer-gated deliberately; `/health` never carries identity |
| GET | `/api/packs` | Discovered packs with load/refuse state, refusal reasons, quarantined-draft ids, and panel declarations. `wired: false` means the pack loader didn't run this boot (the bare `web` entrypoint), distinct from "no packs installed" |
| GET | `/api/mcp/servers` | MCP server cards: transport summary, **probed** health (a dead subprocess reads unhealthy, never as empty success), tool inventory, `allowed_tools`, `env_names`, `header_names`. Credential values (`env`, `headers`) never appear — write-only, the provider-keys stance |
| POST / PATCH / DELETE | `/api/mcp/servers[/{name}]` | Manage REST-owned servers (persisted in `data/mcp_servers.json`, applied to the live runtime — the response's `applies` says `live` or names why not). Servers from `prometheus.yaml` are read-only here (409): the daemon never writes the operator's config file |
| GET | `/api/media?path=` | Stored image bytes by reference (the `source_path` in an image block). Path must RESOLVE under the image cache root — symlinks and `..` included; anything else is a 403. 404 for an evicted file, so clients fall back to the description placeholder |
| GET | `/api/sessions` | List sessions — durable-first, so the list **survives daemon restarts** |
| POST | `/api/sessions` | Create a session (optional `{"gateway": ...}` body) |
| GET | `/api/sessions/{session_id}/messages` | Message history (`?since=<message_id>` for incremental sync) |
| DELETE | `/api/sessions/{session_id}` | Forget a session (durable tombstone — see below) |
| GET | `/api/config` | Effective config (secrets redacted) |
| GET | `/api/sessions/{id}/workspace` | The session's working directory and its source (`session` or `daemon`), plus the daemon's cwd and configured workspace roots |
| PUT | `/api/sessions/{id}/workspace` | Bind a working directory: `{"path": "/abs/dir"}`; blank clears. Refused unless absolute, an existing directory, not `/`, not under `security.denied_paths`. **The gate follows the session:** from the next turn this is the conversation's write boundary (`write_file`/`edit_file`, bash's lock and write floor), where relative paths resolve, and where its instruction files (PROMETHEUS.md, CLAUDE.md, AGENTS.md, …) are read |
| DELETE | `/api/sessions/{id}/workspace` | Clear it — the session follows the daemon again |
| GET | `/api/sessions/{id}/checkpoints` | The session's file checkpoints, newest first — one per turn, taken before any tool runs, only for sessions with a workspace |
| GET | `/api/sessions/{id}/checkpoints/{cid}` | The checkpoint's files, what was skipped (too large), and what a restore would change now (`changed`, `deleted`, `added`, `unchanged`) |
| POST | `/api/sessions/{id}/checkpoints/{cid}/restore` | `{"confirm": "<cid>", "dry_run": false}` — rewrites changed/deleted files from the checkpoint and removes files created since; names every path. `dry_run` reports without touching anything. Naming the checkpoint twice is the confirmation |

Session semantics worth knowing:

- **`GET /api/sessions` enumerates from the durable LCM store first**, then overlays the in-memory working set. Each row carries a `live` field: `live: true` means the session has an in-memory working set right now; `live: false` means it was restored from durable history after a restart — its full history is still servable via the messages route, but the working context starts fresh on the next message.
- **`DELETE` writes a durable tombstone**, not just an in-memory clear: the session disappears from the index and stays hidden across restarts, but the append-only LCM rows are left intact — and **newer activity revives it** (a stable gateway id that speaks again resurfaces).
- **`POST /api/sessions` stamps the origin gateway into the id.** Session ids follow the `<gateway>:<id>` convention (`telegram:123`, `desktop:<uuid>`); this route defaults to `desktop` and accepts an optional `{"gateway": "..."}` body key — 1–32 chars of `[A-Za-z0-9_-]`, no colons. A present-but-empty `gateway` is a 400, not a silent default.
- **A scoped device token (an approved device) is default-deny.** It may use hello, its own device (list itself, sign out, push and live-activity registration), chat, its own sessions, and approve or deny the tool calls of its own sessions with `once` or `until-restart` only. Every other route is `403 operator_only`: cron, provider keys, config, grants, files, MCP, stories, projects, `/v1/*` and the rest. In chat it may run `/help` and the commands on its own session; any other slash command is refused in the reply. The global token and an **owner device** are the operator and keep everything; a device an operator **marked for computer use** additionally reaches the computer routes. Marking (`PUT /api/devices/{id}/computer`, `{"computer": true}`) is a person's act: a marked device, or Telegram's `/computer allow`; the API token never marks. A device may remove only its **own** mark (`{"computer": false}`); removing another device's takes Telegram's `/computer disallow <device id>` or the API token, so a device marked a minute ago cannot strip the owner's (403 `only_own_mark`). The list is `web/route_access.py`, and a test fails if a registered route is in no class.
- **A device token sees only its own sessions.** The global token (`PROMETHEUS_API_TOKEN`) sees and manages every session; a per-device token (`POST /api/devices`) sees and manages the sessions that device created — through `POST /api/sessions`, or by being first to send to, upload into, or switch to an id that exists nowhere — and nothing else. That covers the session list, history, search, `/api/events/recent`, every `/api/sessions/{id}/…` route, and the WebSocket (`switch_session`, `send_message`, and which frames a socket receives, whatever it subscribed to). A session that is not the device's answers `404 {"error": "unknown session"}`, exactly as a missing one does. A device cannot take an id in a namespace the daemon writes into (`telegram:…`, `slack:…`, `discord:…`, `cli:…`, `api:…`, `coding:…`). A device can revoke only itself (`DELETE /api/devices/{its own id}`; another id is a 403) — except an **owner device**, the person's own (see Same-Mac pairing), which is operator-equivalent and sees all sessions. Revoking a device also closes its open WebSocket (4401). Sessions that existed before this rule belong to the operator. What it does not cover is listed in [docs/contracts/device-scoping.md](../contracts/device-scoping.md).

### Pairing a new device

A device with no token asks to join, and the owner approves it on a device they already use. Contract: `docs/PAIRING-APPROVAL-API.md`. The three requester routes are public (see Authentication) and take no bearer token; the three operator routes need `identity.is_operator` (the global token or an **owner device**), and answer a scoped device **403** `operator_only`, not 401: a 401 tells a client its token is dead.

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/pair/requests` | **Public.** `{device_name, platform, public_key}`; the key is 32 raw X25519 bytes as unpadded base64url. Returns `request_id`, `poll_secret`, a 4-digit `match_code` to compare with the one on the new device's screen, `instance_public_key`, `expires_at`, `ttl_seconds`, `poll_interval_seconds` and `notified`. Limits per TCP peer: 1 pending, 3 pending overall, 10 an hour (429 with `reason`, `retry_after_seconds` and `Retry-After`); body at most 4 KiB; `Origin` refused. 403 `pairing_unavailable` in setup mode, with no token set, or when `pairing.requests_enabled` is false; 503 `identity_unavailable` without an instance key. |
| GET | `/api/pair/requests/{id}` | **Public**, with the poll secret in the `X-Pairing-Secret` header (never the URL). `pending` (with `expires_at` and a `notified` that only goes false to true: once a channel took the event, or an operator has been connected at any point while it waited), `denied`, `expired`, `canceled`, `delivered`, or `approved` with `device_id`, `sealed` (`alg`, `ephemeral_public_key`, `nonce`, `ciphertext`), `endpoints`, `tls` and `approved_at`. `endpoints` are where the new device connects back, from the `Host` it used: over the TLS listener (home-network mode) `https` for REST and `wss` on the TLS WebSocket port (`web.tls_ws_port`, default 8011), with `tls` carrying the SPKI pin; a plain request gets `http` / `ws` on the plain ports and `tls: null`. The daemon decides which listener a request came in on, not a header. An unknown id, a wrong secret and another request's secret are one identical 404; five wrong secrets a minute spend that source's budget for wrong guesses (429 `bad_secret`, per TCP peer address; the right secret is never refused for it); polling faster than once a second is 429 `poll_too_fast`. |
| DELETE | `/api/pair/requests/{id}` | **Public**, with the secret. Cancels while pending; once approved it acknowledges receipt and wipes the sealed blob. 204. |
| GET | `/api/pair/requests` | Operator. Who is waiting: `request_id`, `device_name`, `platform`, `source_ip`, `match_code`, `created_at`, `expires_at`, `ttl_seconds`. |
| POST | `/api/pair/requests/{id}/approve` | Operator. Optional `{name, match_code}` and **nothing else**: any other key, `owner` included, is a 400, so no client can believe it granted more than an ordinary scoped device. A retyped `match_code` that differs is 422 `code_mismatch`; 409 `not_pending` names the winner; 410 `expired`. The response carries no token. |
| POST | `/api/pair/requests/{id}/deny` | Operator. |
| POST | `/api/pair/owner` | Any valid token (the API token, an owner device, or a scoped device). Trades a one-time code from `oara pair --owner` for an **owner device**: `{"code", "device_name", "platform", "public_key"}` → 201 `{device_id, name, platform, owner: true, owner_source: "owner-code", sealed}`, the token sealed to `public_key` with `device_id` as the request id. 422 `invalid_code`, 429 `rate_limited` (`reason: bad_code`). See below. |
| GET | `/api/network` | Operator (a scoped device is `403 operator_only`). What the daemon is listening for: `mode` (`this_mac`, `home_network` or `open`: what it is *running* as), `bind`, `bind_source` (`flag`, `env`, `config`, `default`), `tls` (`{enabled, spki_sha256}` — `enabled` while the TLS listener actually serves the LAN, and then `spki_sha256` is the **full** SHA-256 of the instance key's SPKI that the certificate is made over; `/api/hello`'s `fp` is that value's first 16 hex characters, so a client can match what it discovers to what it pinned), `advertising` and `advertising_reason` (mDNS `_prometheus._tcp`: only in `home_network` mode, only with the optional `discovery` extra installed; `{code, message}` or `null`), `applied` (`live`, or `on_restart` when the config file says something else than what is running, with `pending_mode`), `warnings` (a list of `{code, message}`: plain HTTP on the LAN, a switch that cannot work, a saved change waiting; the code is stable, the English is the fallback) and `can_change` (true only for an operator when nothing outside the file fixes the bind: whether to show the control at all). An install that predates the setting is reported `open`, never dressed up as home network. |
| PUT | `/api/network` | Operator (a scoped device is 403 `operator_only`). `{"mode": "this_mac"}` or `{"mode": "home_network"}` and nothing else. It **saves** the choice into the config file (comments kept, edit verified, write atomic) and answers `applied: "on_restart"`: the listeners are bound at start, so it takes effect on the next restart — which `POST /api/restart` (below) can do without anyone touching the machine. 409 `tls_unavailable` for `home_network` when this daemon has no instance key to make the certificate over and `network.allow_plaintext_lan: true` is not in the file (the API never writes that line — plain HTTP on the LAN stays a deliberate opt-out); 409 `would_lock_out` for `this_mac` from a caller that is not on this machine (it would cut that caller off after the restart); 409 `bind_overridden` when `--bind` or `PROMETHEUS_WEB_BIND` fixes the address; 409 `no_config_file`; 500 `persist_failed` with the file untouched. |
| GET | `/api/model` | Operator (a scoped device is `403 operator_only`). The daemon's own provider, model and key, for a "fix my setup" screen. `running` and `saved` are `{provider, model, api_key_env, kind}` (`kind` is `cloud` or `local`; `saved` is what a restart brings back, with `PROMETHEUS_PROVIDER` / `PROMETHEUS_MODEL` laid over the file), `pending` and `applied` (`live`, or `on_restart` when `saved` differs from `running` or a new key was saved that the running provider does not hold), `key` (`{env, present}`: whether a key is held, never its value), `providers` (every provider `PUT` accepts, with its `default_model`, `api_key_env`, `key_present` and whether its key is `checkable`), `warnings` (`{code, message}`: `restart_required`, `model_overridden` when an environment variable outranks the file) and `can_change`. |
| PUT | `/api/model` | Operator. Change the provider, model or API key after setup: `{"provider", "model"?, "api_key"?, "check"?}`. Cloud providers only (the ones setup offers); there is no `base_url` and one is a 400. It **saves** (config file edited as text, comments kept, edit verified, write atomic; the key goes to the 0600 env file, never the yaml) and answers `applied: "on_restart"`: the provider is built at start, so it takes effect on `POST /api/restart`. Switching provider blanks the old vendor's inline `api_key`, `base_url` and `base_url_env` (they would send the old key to the new vendor), sets the key variable and, unless `model` is given, the new provider's default model, and moves `context.effective_limit` to the new provider's unless it was tuned. The key is checked first (same check as `/api/setup/check-key`, unless `"check": false`; with no `api_key` the one already held is checked): a refused key is `400 key_rejected` with `key_check` and nothing is written; `no_credit` and `unreachable` are saved with a `key_no_credit` / `key_unchecked` warning. The answer is `GET`'s plus `changed` (the file changed), `key_saved` and `key_check`. Refusals are `{error, detail}` and write nothing: 400 `invalid_request`, `unexpected_base_url`, `unsupported_provider` (a local server is set up with `oara setup`), `bad_model`, `invalid_key`, `cloud_key_missing`, `key_rejected`; 409 `model_overridden` (`source` names the variable) and `no_config_file`; 500 `persist_failed`. |
| POST | `/api/model/check-key` | Operator. `{"provider", "api_key"?}` → the `/api/setup/check-key` answer (`ok`, `bad_key`, `no_credit` or `unreachable`). With no `api_key`, checks the key the daemon already holds for that provider (400 `cloud_key_missing` if none). Same fixed destinations, no `base_url`, no redirect, and the key appears only in the provider's auth header. |

The owner hears about a request in Beacon (the `pairing_pending` frame below), in the terminal (`oara pair list | approve <id> | deny <id>`, which uses these routes with the global token), and in Telegram only if `pairing.telegram_prompts` is on (off by default; plain text with Approve and Deny buttons, private chats only). The first decision wins on every channel; the others are told.

The token is **sealed** to the key the device sent (X25519, HKDF-SHA256 with the request id as salt, ChaCha20-Poly1305 with the request id as associated data), so it is never in clear on the wire or at rest; `tests/vectors/pairing_seal_v1.json` holds bytes to check a client against. A retried poll returns the same sealed blob until the device acknowledges or five minutes pass, and a device nobody collects within five minutes of approval is revoked. An approved device is an ordinary, **scoped** one: it owns no conversations and sees none of the operator's.

**An owner device on another computer.** Approving a request never makes an owner device, and the API token alone cannot either: the only way for Beacon on another computer (a MacBook reaching the box over a tailnet, say) to become one is a code printed **on the daemon's machine**. `oara pair --owner` writes the code's SHA-256 into the daemon's `devices.db` and prints the code: 12 characters in three groups (`K7QM-2XW9-RTHP`; no 0, 1, I, L, O or U), good for 10 minutes and one use, and running it again replaces it. It needs no network and no token, and no route issues a code, so nothing that only reaches the daemon over the network can make one. The other computer, already connected with any valid token, sends it to `POST /api/pair/owner` with its device name, platform and a fresh X25519 public key, and gets back the new owner device's token **sealed** to that key, exactly as an approved pairing's is (the request id in the recipe is the returned `device_id`). An owner device can approve a worker's request, which the API token cannot (a model can read the API token); the client should keep the new token, drop the API token, and may revoke its old device with `DELETE /api/devices/{id}`. A malformed body is a 400 that spends nothing; each source may guess wrong 3 times a minute (then 429 `rate_limited`, `reason: bad_code`), and the fifth wrong guess from anyone ends the code (422 `invalid_code` from then on, as for a wrong, expired or used one). The device is listed with `owner_source: "owner-code"`, and re-pairing the daemon's own Mac never revokes it.

### Chat

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/chat/send` | Send a chat message. Body: `{"session_id": "my-session", "message": "..."}` (the field is `message`, not `content`); optional `mode` (`"agent"`/`"chat"`), `tool_choice`, and `references` (@-references, see below). Returns `{"run_id", "status": "sent"}`; the reply streams over the WebSocket and lands in `GET /api/sessions/{session_id}/messages` |
| POST | `/api/chat/interrupt` | Stop the running agent turn in a session — the chat Stop button. `{"session_id": ...}`; idempotent (`stopped: false` when nothing is running). Completed rounds persist, a mid-generation partial is kept as an assistant turn, and every client sees the broadcast `chat_done{interrupted:true}`. HTTP twin of the WS `interrupt` frame |
| POST | `/api/chat` | Send a chat message and wait for the reply. Body: `{"session_id": "my-session", "content": "..."}` (the field is `content`, not `message`). The turn runs in session `web:<session_id>`, starts from the same system prompt as `/api/chat/send` (memory files included), and returns `{"text", "turns", "usage"}` when it ends. `503` when the daemon hasn't wired the WebSocket bridge |

### OpenAI-compatible surface — `/v1`

Any client that speaks the OpenAI chat-completions wire (Open WebUI, LobeChat, Continue, Zed, the `openai` SDKs) can point at the daemon with the same bearer token. Beacon stays the surface; this is the second door.

| Method | Path | Description |
|---|---|---|
| GET | `/v1/models` | The keys from `/api/models` that have a credential present, in OpenAI's list shape (`id` = the catalog key: `local`, `claude`, `qwen:qwen3.7-max`, …) |
| POST | `/v1/chat/completions` | One agent turn. `messages` (system/user/assistant, string or text parts), optional `model` (a key from `/v1/models`, default `local`), optional `stream` (SSE chunks ending in `data: [DONE]`), optional Prometheus extension `mode` (`agent` default, `chat` = no tools) |

What to expect, stated plainly:

- **Stateless, like OpenAI.** Send the whole conversation each call; each call runs one turn in a fresh `openai:<id>` session that persists nothing to LCM or memory. The daemon's own sessions are untouched. The response carries `prometheus.session_id` so a telemetry row can be traced.
- **Tools run server-side and are never surfaced.** The model uses Prometheus's tools behind its security gate; the client sees only text. A request that sends `tools`, `functions` or `tool_choice` is refused with `400 tools_unsupported` rather than having them ignored. A non-read-only tool that needs approval blocks the turn exactly as it would for Beacon.
- **`system` messages are appended** to the daemon's own system prompt as client instructions; they never replace the identity, tool and safety text.
- **Ignored on purpose:** `temperature`, `top_p`, `max_tokens`, `n`, `stop`, `logprobs` — the loop owns generation settings. Image and audio content parts are refused (`400 unsupported_content`).
- **Errors** use OpenAI's envelope (`{"error": {"message", "type", "code"}}`): `404 model_not_found`, `400 last_message_not_user`, `502` with the classified turn error when the loop dies, `503` when no loop is wired.

```bash
curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8005/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"What is the Context7 library ID for httpx?"}]}'
```

### Telemetry & events

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/telemetry` | Per-model-per-tool call stats |
| GET | `/api/pairs` | Repair pairs / golden traces |
| GET | `/api/events/recent` | Recent event feed |
| GET | `/api/activity/recent` | Recent activity feed |

### Memory, wiki, LCM & sentinel

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/memory/current` | Current MEMORY.md / USER.md contents with char budgets |
| PUT | `/api/memory/current` | Replace MEMORY.md / USER.md content (Beacon's Memory editor). Body `{"memory": ..., "user": ...}` — null/absent leaves a file untouched. The previous content is **snapshotted to `~/.prometheus/memory-history/` before every write**; over-budget content is a 400 with nothing written. Optional `base_memory`/`base_user` enable optimistic concurrency: if the file changed since the client loaded it, the write is refused with **409 + the current truth**, so the editor can rebase its draft |
| GET | `/api/wiki/stats` | Wiki page/link stats |
| GET | `/api/lcm/{session_id}` | Durable conversation store view for a session |
| GET | `/api/sentinel` | Sentinel subsystem status |

### Skills & profiles

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/skills` | Skills overview |
| GET | `/api/skills/list` | Full skill listing |
| GET | `/api/skills/{name}` | Single skill detail |
| POST | `/api/skills/{name}/pin` | Pin a skill |
| DELETE | `/api/skills/{name}/pin` | Unpin a skill |
| GET | `/api/profiles` | List agent profiles |
| PUT | `/api/profiles/active` | Switch active profile |

### Tools — deferred loading

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/tools/deferred` | Deferred-loading state: the configured tri-state, the **effective** resolution and its source (e.g. `auto → enabled (local provider)`), and advertised/total schema counts |
| PUT | `/api/tools/deferred` | Set the tri-state override. Body `{"enabled": true\|false\|"auto"}` — `"auto"` *is* the cleared state. Applies at the **next run start** (the advertised set is frozen per run); persisted to the on-disk yaml with a surgical single-key write |

### Learning — skill recording & drafts

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/learning/live-upload` | Ingest a live DOM recording (multipart: `events` + `metadata` JSON + screenshots; 32 MB body cap). 503 when `learning.live_recorder.enabled: false` |
| POST | `/api/learning/video-ingest` | Start background video/YouTube ingestion toward a skill **draft** (never auto-persisted). 503 when `learning.video_ingest.enabled: false` — which is the shipped default |
| GET | `/api/learning/skill-drafts` | List pending skill drafts |
| GET | `/api/learning/skill-drafts/{draft_id}` | Draft content + provenance sidecar |
| POST | `/api/learning/skill-drafts/{draft_id}/accept` | Persist a reviewed draft to `skills/auto/` (optionally with edited `content`); goes through the same validated write path DOM recordings use. **409** with `conflict: {skill_name, files, served_elsewhere}` when a skill with the name is already served — auto files, or a builtin or user skill (the draft stays); `replace: true` archives the auto files to `skills/auto/archive/` and writes the draft in their place (builtin and user skills are never modified; the draft is served instead), and the 200 lists the archived files under `replaced` |
| POST | `/api/learning/skill-drafts/{draft_id}/reject` | Archive a draft to `drafts/.rejected/` (never deleted) |

Draft lifecycle events (`skill_draft_created` / `skill_draft_accepted` / `skill_draft_rejected` / `video_ingest_failed`) ride the WebSocket's `sentinel_signal` fan-out. See the [Record a Skill guide](record-a-skill.md).

### Cron

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/cron` | List cron jobs |
| POST | `/api/cron` | Create a cron job |
| PUT | `/api/cron/{name}` | Update a cron job |
| DELETE | `/api/cron/{name}` | Delete a cron job |
| POST | `/api/cron/{name}/run` | Run a job immediately |

### Files & documents

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/files` | List workspace files |
| GET | `/api/files/read` | Read a workspace file |
| GET | `/api/documents` | List editable documents |
| GET | `/api/documents/content` | Read a document |
| PUT | `/api/documents/content` | Save a document |
| POST | `/api/documents/edit` | Apply a span-bounded edit |
| POST | `/api/documents/suggest` | AI redlines — one-shot model call returning JSON suggestions (not an agent loop) |

### Artifacts — the agent's outbox

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/artifacts` | List the outbox manifest — files the agent saved into `~/.prometheus/files` for delivery |
| GET | `/api/artifacts/{id}` | Download an artifact as an attachment (`Cache-Control: no-store`) |

Artifacts are **content-addressed**: ids are a sha256 prefix of the file bytes, so clients never send a path and the whole path-traversal class stays out of the wire contract. Ids survive renames, identical bytes dedup, and symlinks/dotfiles/files over 1 GiB are never indexed.

### Paperclip gateway

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/paperclip/wake` | Wake webhook for a [Paperclip](https://github.com/paperclipai/paperclip) fleet manager — runs one awaited agent turn against a checked-out issue. Returns **503 when `gateway.paperclip.enabled` is false**, which is the shipped default; the feature is off unless you configure it |

### Approvals

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/approvals` | Poll pending approval requests |
| POST | `/api/approvals/{request_id}/approve` | Approve a request |
| POST | `/api/approvals/{request_id}/deny` | Deny a request |

### Benchmarks

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/benchmarks/run` | Run the eval suite |

### Models & per-session overrides

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/backends` | Every local inference backend the daemon knows (the primary as `local` + `backends:` in config) and what each is serving — model, reported window, detected vision, latency, `stale`, `changed_at`. Probes through the registry's TTL; `?refresh=1` forces every backend |
| POST | `/api/backends/{name}/probe` | Force one backend's probe now; 404 for an unknown name |
| GET | `/api/models` | Model catalog (local + backends + cloud providers). Backend rows carry `backend`, `health` (the registry's last probe: `ok`, `probed`, `stale`, `n_ctx`, `latency_ms`, `error`, `changed_at`), `available` = reachable, `vision` = detected. The `local` row carries the same two fields. `vision` on the `local` row is the boot provider's **detected** capability (llama.cpp `/props` modalities, i.e. an mmproj is loaded) — the same value the image-upload gate uses; on cloud rows it is the preset's declared flag. The response also carries `catalog_note` — the catalog is a list of **defaults, not limits**: the model string in `prometheus.yaml` is free-form (only the *provider* name is validated), so a client should not render this as a closed menu |
| GET | `/api/sessions/{session_id}/model` | Current per-session model override |
| POST | `/api/sessions/{session_id}/model` | Set a per-session override (`local` clears back to the primary). A backend key (`4090`, `mini:qwen2.5:7b-instruct`) is **probed first**: 200 with the row on success, 503 with the probe's error and `health` when the box is down — nothing switches |
| DELETE | `/api/sessions/{session_id}/model` | Clear the override |
| GET / PUT | `/api/workers/model` | The **one operator setting for spawned workers** (`task_create type=local_agent`, subagents). By default a worker *inherits the spawning turn's route* — the picker key its session chose (`qwen`, `qwen:qwen3.8-flash`, `4090`) or `local` for the configured primary — so a conversation on a cloud model doesn't silently hand its worker to a local box with a smaller window. Only the picker **key** crosses into the child; the child re-resolves it through the same catalog the model picker uses, so endpoint and credential come from the route, never the command line. `PUT {"model": "<key or null>"}` overrides that for every worker: `null`/empty is **Same as chat** (the default, inherit), or a key from `GET /api/models` pins one route. The value is validated against the catalog when set (a typo can't become the setting) but its credential is checked at *spawn*, where a missing key is refused with the reason. The choice is stored in `<config dir>/worker_settings.json`, which wins over the `workers.model` yaml key — the daemon never rewrites `prometheus.yaml`. **An agent can never choose a worker's model**: a `model` argument on the spawn tools is refused unless it names the route this setting already chose, and a route that can't be served is refused with its reason, never substituted. |

### Provider keys & xAI OAuth

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/providers/keys` | List key-able services. Returns `set: true/false` per env var only — **never key values** |
| PUT | `/api/providers/keys/{service_id}` | Set a provider API key (persisted to `~/.config/prometheus/env`) |
| GET | `/api/providers/xai/oauth` | xAI SuperGrok OAuth status |
| POST | `/api/providers/xai/oauth/login` | Start the device-code OAuth flow |
| DELETE | `/api/providers/xai/oauth` | Remove stored xAI OAuth credentials |

### Coding runs

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/code` | Launch a sandboxed coding run |
| GET | `/api/code/{task_id}` | Run status / round telemetry |
| POST | `/api/code/{task_id}/stop` | Stop a run |
| POST | `/api/code/{task_id}/pause` | Pause between rounds |
| POST | `/api/code/{task_id}/resume` | Resume a paused run |
| POST | `/api/code/{task_id}/inject` | Inject mid-run supervision guidance |
| GET | `/api/code/{task_id}/diff` | Diff produced by the run |

### Project files

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/project-file` | Read a project file (daemon-routed; used by Loop Manager) |
| PUT | `/api/project-file` | Write a project file |

### Kanban — projects & stories

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/projects` | List projects |
| POST | `/api/projects` | Create a project |
| PUT | `/api/projects/{project_id}` | Update a project |
| DELETE | `/api/projects/{project_id}` | Delete a project |
| GET | `/api/stories` | List stories |
| POST | `/api/stories` | Create a story |
| PUT | `/api/stories/{story_pk}` | Update a story |
| DELETE | `/api/stories/{story_pk}` | Delete a story |
| POST | `/api/stories/reorder` | Reorder stories within/between columns |
| POST | `/api/stories/{story_pk}/dispatch` | Dispatch a story to a coding run |
| POST | `/api/stories/{story_pk}/undispatch` | Detach a story from its coding run |

## Setup-mode API

When the daemon starts with **no config file**, it boots a minimal setup server (`src/prometheus/web/setup_server.py`) instead of the full API. Only these routes exist in this mode:

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/hello` | The same six-field answer as the configured daemon; `pair` is `code` here, and `fp` is empty (setup mode creates no state) |
| GET | `/api/setup/status` | Setup progress / pairing window state |
| POST | `/api/setup/pair` | Exchange the 6-digit pairing code for an API token |
| GET | `/api/setup/detect` | Probe for local backends (llama.cpp, Ollama, LM Studio, vLLM); also lists the cloud presets (`cloud_providers`: name, key env var, default model, whether the daemon already holds the key — never the value) |
| POST | `/api/setup/check-key` | Does this cloud key work? `{"provider": "openrouter" \| "anthropic" \| "openai", "api_key": "..."}` → `200 {"result": "ok" \| "bad_key" \| "no_credit" \| "unreachable", "status"?, "detail"?}`, asked before `configure`; master token required. It calls each provider's own fixed address (there is no `base_url`; sending one is a 400), follows no redirect, writes nothing, and the key appears only in the provider's auth header — never in the answer, the URL or a log line, and no provider text is relayed. Classification is by HTTP status: 401/403 `bad_key`, 402 `no_credit`, 2xx `ok`, anything else (no network, a timeout, a redirect, a rate limit, a 5xx) `unreachable`, never `ok`. **OpenRouter** is checked with one free call and then a one-token request on the setup default model, because an account with no credit answers the first with 200 and only a request that spends shows it; a refused key never reaches the second. **Anthropic and OpenAI** use their free model-list call, so a valid key on an empty account still answers `ok` there. |
| POST | `/api/setup/configure` | Write the chosen configuration. Local: `provider` + `base_url` + `model`, re-probed before writing. Cloud: `provider` (a preset name, `openrouter` included) + `api_key` (+ `model`), no `base_url`; the key goes to the env file, and a cloud provider with no key anywhere is refused (`cloud_key_missing`). Every input is checked before anything is written: a provider setup does not offer is `unsupported_provider` (even where its address answers), a key or token with a line break or far too long is `invalid_key` / `invalid_token` (the env file is line-based), a value that is not text is `bad_request`, and a disk that cannot be written is a coded 500 `config_write_failed` |
| POST | `/api/setup/complete` | Finish setup and hand off to the full daemon |

The pairing flow: at startup the daemon prints a crypto-random 6-digit code once in a console banner, and a client (Beacon's first-run screen, or curl) POSTs it to `/api/setup/pair` as `{"code": "123456"}` to receive the bearer token. The code is one-time-use, expires after 15 minutes, and locks after 5 failed attempts (only a wrong code burns an attempt); comparison uses constant-time `hmac.compare_digest`. Once paired, the client uses the returned token for the remaining setup calls and for the full API after `complete`.

**Setup errors are coded.** Every refusal in setup mode is `{"error": <code>, "detail": <English>, "fields"?: [...]}` with one status per code, and nothing is ever a bare-text 500 (an unexpected failure is `internal_error`). A client writes its own plain-English message from the code, and uses `fields` to mark the right input boxes; `detail` is for logs. The full list, with statuses and which routes raise each, is [docs/contracts/setup-errors.md](../contracts/setup-errors.md).

**Same-Mac pairing (the macOS app).** `Prometheus.app` starts under launchd, where nobody sees the banner, so on the app install (`PROMETHEUS_INSTALL_KIND=app`) the daemon also writes a one-time secret to `~/Library/Application Support/Prometheus/pairing/pair.secret` (a `0700` directory and a `0600` file: only this user can read it). An app on the same Mac, Beacon, reads it and sends it as the `code` in the same `POST /api/setup/pair`; the response is the same. The secret is accepted only from a loopback peer with a loopback `Host` header, a request that carries an `Origin` header (a web page) is refused with 400, it never expires but is used exactly once, and a wrong secret neither burns the six-digit code's attempts nor is blocked by its lockout. After setup, `POST /api/pair/local` accepts a fresh secret written by `Prometheus --pair`, so reinstalling Beacon does not strand a person: it is the one route that answers without a bearer, listed exactly in `web/public_routes.py`, and it answers 404 on any install that is not the app. Its `api_base_port` and `ws_port` are the ports the daemon is listening on, read from its own sockets, not what the config file names (which may have been edited since the daemon started, or ask for `0`). **What the credential is.** In **setup mode** (the daemon's first boot) `POST /api/setup/pair` returns the daemon's API token, because setup mutations authenticate with it and setup mode creates no `devices.db`; the client then trades it, once the daemon is configured, for a token of its own with `POST /api/devices` and `{"owner": true, "name": "..."}` (a global-token route, and only from this Mac's loopback address: from anywhere else it is a 403, and another computer becomes an owner device only with a one-time code from `oara pair --owner`, above) and keeps only that. On a **running** daemon `POST /api/pair/local` returns an **owner device token** minted for that device alone — never the daemon's API token — in the same `{token, api_base_port, ws_port}` shape plus `revoked_previous`, an integer. An owner device is listed in `GET /api/devices` (`owner: true`, `owner_source`), is revocable on its own, and is operator-equivalent for sessions (it sees all of them, including ones that predate device scoping) and for managing devices; it is not root (it cannot enrol devices or define MCP servers: those stay the global token's). Pairing again from this Mac **revokes the earlier owner credentials of this Mac** in the same transaction, so a reinstall leaves no standing credential nobody holds; another computer's owner device (made with an owner code) is never touched. **What this does not fix:** the global token is still valid and still sits in `~/.config/prometheus/env`, where the bash tool can read it, and on macOS the daemon has neither shell floor. This stops a paired *device* holding the master key; it does not stop a *model* reading it.

## WebSocket bridge (:8010)

The bridge (`ws_server.py`) forwards live chat streaming and SignalBus subsystem events to all authenticated clients.

**First-frame auth.** The very first frame after connecting must be an auth message, sent within 5 seconds (`AUTH_FRAME_TIMEOUT_SECONDS`), or the server closes the socket with code **4401** (the WebSocket mirror of HTTP 401). No data frames are sent before a successful auth.

```json
{"type": "auth", "token": "<PROMETHEUS_API_TOKEN>"}
```

On success the server replies with a `connected` frame.

### Client → server messages

| Type | Purpose |
|---|---|
| `auth` | First-frame token auth (required) |
| `subscribe` | Subscribe to event fan-out (server acks with `subscribed`) |
| `send_message` | Send a chat turn; accepts optional `tool_choice` (validated against the live tool registry), a `client_msg_id` for echo correlation, and `references` (@-references, resolved server-side before the turn is queued; an unresolvable one is an `error` frame with `kind`) |
| `chat_upload` | Upload an attachment (base64); images get vision captions, documents get text extraction |
| `switch_session` | Point this socket at a different session |
| `interrupt` | Stop the running turn: `{"type": "interrupt", "payload": {"session_id": ...}}`. The requesting socket gets an `interrupt_ack`; every client learns the outcome from the broadcast `chat_done{interrupted:true}` |

Example `send_message`:

```json
{
  "type": "send_message",
  "session_id": "web:default",
  "content": "Summarize today's telemetry",
  "tool_choice": null,
  "references": [{"type": "file", "target": "src/app.py"}]
}
```

### @-references

Both `POST /api/chat/send` and the WS `send_message` command accept an optional
`references` list — the daemon half of the composer's `@` chips. The client
**names** the reference; the daemon **reads** it, on this host, before the turn
is queued. Each resolved reference becomes its own text block persisted with the
user turn (the same blocks path as image uploads), so history shows exactly what
the model was given.

| `type` | `target` | Resolves to |
|--------|----------|-------------|
| `file` | path, relative to the session scope (absolute allowed if inside it) | the file's text, capped at 256 KB (`truncated="true"` past that); binary files refused |
| `diff` | optional git revision or range (`HEAD~1`, `main...HEAD`); empty = working tree vs index | `git diff` in the session scope, capped at 128 KB |
| `url` | `http(s)://…` | the page as compact text via the same fetcher and SSRF guard as `web_fetch`, capped at 64 KB |

Scope follows the session: a session with a workspace bound
(`PUT /api/sessions/{id}/workspace`) resolves `file` and `diff` inside that
workspace and nowhere else; without one, the `/api/files` browse root is used
(plus any configured `security.workspace_root`). Denied paths are the gate's own
list — a path `read_file` could not read cannot be `@`-referenced. At most 16
references per message.

Failures are loud: REST answers `400` (malformed / bad ref / binary), `403`
(outside scope, denied path, private address), `404` (no such file, not a git
repo), `502` (fetch failed) or `503` (workspace lookup unavailable) with
`{"error", "kind"}`; the WS path sends an `error` frame with the same `kind` and
does not queue the turn. References cannot be attached to a slash command.

### Server → client messages

Chat lifecycle:

| Type | Purpose |
|---|---|
| `connected` | Auth accepted; connection metadata |
| `subscribed` | Subscription ack |
| `chat_message` | A complete message (user echo, assistant reply, or slash-command result) |
| `chat_delta` | Streaming token delta |
| `agent_state` | Agent thinking/idle state changes (carries `session_id`) |
| `agent_progress` | Liveness pulse every **3 seconds** while a turn runs: `phase`, `tool_name`, `round`, `chars`, `tool_calls`, `elapsed_s`. Samples what the turn is actually doing, so "still alive" is never a guess — and the pulse is cancelled before the turn finalizes, so a heartbeat can never outlive its turn |
| `tool_call_start` / `tool_call_end` | Live tool-call boundaries |
| `chat_done` | Turn finished. A user-stopped turn broadcasts the `interrupted: true` variant |
| `interrupt_ack` | Reply to an `interrupt` frame (requesting socket only): `{session_id, stopped}` |
| `error` | Turn or frame failure, **structured**: `{session_id, message, kind, provider, status, hint}`. `kind` is a stable machine token (`billing`, `auth`, `rate_limit`, `timeout`, `unreachable`, `provider_error`, …), `hint` is one actionable sentence. Redaction guarantee: only the request URL's **host** is ever echoed — never paths, query strings, or credentials (some providers pass API keys as `?key=`) |

SignalBus fan-out (broadcast to all authed clients; payloads carry `session_id` where relevant):

| Type | Purpose |
|---|---|
| `sentinel_signal` | Sentinel memory-pipeline events |
| `dream_start` / `dream_phase` / `dream_complete` | Dream-cycle progression |
| `skill_created` / `skill_refined` | Learning-system skill events |
| `memory_updated` | Memory file changes |
| `curator_report` | Weekly curator consolidation report |
| `coding_round` / `coding_tool` / `coding_acceptance` / `coding_complete` / `coding_stream_error` | Coding-run live stream: per-round progress (with a run-unique `seq` — `round_index` restarts per episode), per-tool-call detail attributed to its round, the ground-truth acceptance verdict per episode, terminal verdict, non-fatal stream interruption |

Pairing requests, sent **only to operator sockets** (the global token or an owner device; never to a scoped device, and never through the SignalBus), because they carry a source address and a match code. They name no session, so a plain broadcast would reach every device: the bridge sends them with a targeted send instead.

| Type | Purpose |
|---|---|
| `pairing_pending` | A device is asking to join. `payload`: `request_id`, `device_name` (render as text, never markup), `platform`, `source_ip`, `match_code`, `created_at`, `expires_at`, `ttl_seconds`. A socket that connects later is sent one per request still waiting, oldest first, right after `connected`; a frame can arrive twice if a request lands mid-connect, so key on `request_id`. |
| `pairing_resolved` | It was decided, withdrawn or ran out. `payload`: `request_id`, `resolution` (`approved`, `denied`, `expired`, `canceled`), `by` (`beacon`, `telegram`, `cli`, `requester`, `system`), `resolved_at`. Retract the prompt. |

Skill-draft lifecycle events (`skill_draft_created` / `skill_draft_accepted` / `skill_draft_rejected` / `video_ingest_failed`) ride the `sentinel_signal` channel — watch its `kind` field.

## Building a client

The sync contract is deliberately simple: **`message_id` is the durable LCM rowid — monotonic, unique, and restart-stable — and it doubles as the sync cursor.** `GET /api/sessions/{id}/messages?since=<message_id>` returns only rows after your cursor, and every response includes a top-level `watermark` (the session's current max `message_id`) so a client knows it is caught up even when an incremental read comes back empty. Session rows from `GET /api/sessions` carry the same `watermark`, so one list call tells you which sessions have news. A malformed `since` is a **400**, never silently ignored. Key on `message_id` and order by it, not by `timestamp` or `ordinal`. `timestamp` repeats: a whole turn can share one. `ordinal` is the message's prompt position (`turn_index`), unique within its session once the daemon has run its one-time `lcm.db` migration (at its first start on 0.9.5 or later; earlier builds wrote repeats). It is still not the order rows were stored in: a message sent mid-turn can be stored before the rest of its turn, so `ordinal` order and `message_id` order can differ.

## Ports & remote access

- REST: **:8005**. WebSocket: **:8010**. Both listen on the address `web.bind` names. **A new install writes `web.bind: 127.0.0.1`, this machine only.** A config with no `web.bind` listens on **every interface (`0.0.0.0`)**, which is what reaching the daemon over Tailscale or a LAN needs, so an existing install is never changed for you.
- Beacon expects exactly these two ports on the daemon host — they are not currently negotiated.
- Both are reachable over Tailscale, which is the intended remote-access path (e.g. Beacon Desktop on a laptop talking to the daemon box); the bearer token and WS first-frame auth are what stand between the ports and the tailnet.
- **TLS only in home-network mode.** With `network.home_network: true` and a bind beyond loopback, the LAN is served over TLS on **:8006** (REST) and **:8011** (WebSocket), `web.tls_api_port` and `web.tls_ws_port`, with a certificate over the instance key, and plain HTTP stays on 127.0.0.1. Otherwise there is no TLS: the traffic is plain HTTP and `ws://` on whichever interface you name, and on `0.0.0.0` the bearer token is the only access control for everyone who can reach the machine's network address. The daemon's startup log says so (once) when it is listening on every interface, and `oara doctor` warns when `web.bind` is **unset** and the daemon would listen on every interface. A `0.0.0.0` you set yourself is your decision: doctor reports it as plain HTTP on all interfaces and does not warn.

- **Forwarded headers are not believed unless you name the proxy.** The address every rate limit and every same-machine check uses is the TCP peer. `X-Forwarded-For` and `X-Forwarded-Proto` are ignored, so a client cannot choose its own address by sending a header (uvicorn's default would have believed them from anything on 127.0.0.1). If a reverse proxy fronts the daemon (a tunnel, `tailscale serve`, nginx), name it in `web.trusted_proxies` (addresses or networks, e.g. `["127.0.0.1"]`); an entry that would believe everyone, such as `*` or `0.0.0.0/0`, is ignored with a warning. A proxy on this machine that is **not** named makes every request look like it comes from 127.0.0.1. Behind a TLS-terminating proxy, `X-Forwarded-Proto` is also only believed from a named proxy.

### Listening on this machine only

One setting covers all three listeners — the REST API, the WebSocket bridge, and the setup-mode server (whose pairing endpoint is reachable, protected only by the one-time code, for up to 15 minutes):

| Source | Example |
|---|---|
| `oara daemon --bind ADDRESS` | `oara daemon --bind 127.0.0.1` |
| `PROMETHEUS_WEB_BIND` (process environment, or the env file) | `PROMETHEUS_WEB_BIND=127.0.0.1` |
| `web.bind` in `prometheus.yaml` | `web: {bind: "127.0.0.1"}` |
| *(none of the above)* | `0.0.0.0` — what a config with no `web.bind` has always done |

Highest wins, in that order. Setup mode has no config yet, so it honours the flag and the environment; `POST /api/setup/configure` then writes `web.bind` into the new config: the address you chose with `--bind` or `PROMETHEUS_WEB_BIND` (`0.0.0.0` included), so the real daemon — in the same process or after a restart — listens where setup mode did, and `127.0.0.1` when you chose none.

**New installs, existing installs.** `oara setup` (fast path and wizard) and setup mode's `configure` write `web.bind: 127.0.0.1` into a config that is **new**. A rerun on a config that already exists never changes it: an unset `web.bind` stays unset, a chosen one is kept. `oara setup` replaces the file, so it carries your `web.bind` across; the wizard edits the file in place.

**A headless box set up remotely.** Setup mode listens on every interface, so Beacon on another machine can pair and configure it. When nobody chose an address with `--bind` or `PROMETHEUS_WEB_BIND`, `configure` still writes `web.bind: 127.0.0.1`, so once setup finishes the box answers only on itself, until its owner sets the bind on purpose (`web.bind: 0.0.0.0`, or a tailnet address) and restarts the daemon. To keep it reachable straight through setup, start it with `--bind 0.0.0.0` (or the environment variable): a chosen address is written as chosen.

An address is an IPv4 literal, an IPv6 literal (`::1`), or `localhost` (which means `127.0.0.1`; spell `::1` for IPv6 loopback). Anything else — a host name, a port, a mask — makes the daemon **refuse to start** with a message naming the setting; it never falls back to a wider address.

On a loopback address the daemon also refuses any request whose `Host` header is not `localhost`, `127.0.0.1` or `[::1]` (any port), on the REST API, on the WebSocket handshake and on the setup server. That closes DNS rebinding, where a web page in your own browser reaches a loopback-only service under an attacker's host name. Reach it as `http://localhost:8005`, not by the machine's name; a local reverse proxy in front of it must forward `Host: localhost`. A wide or specific-interface bind has no Host restriction.
