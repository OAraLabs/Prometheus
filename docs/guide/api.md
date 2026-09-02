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
- `GET /health` is the only unauthenticated API endpoint — it lives outside `/api/` precisely so external monitors can poll it without credentials. (The JSON API index served at `/` is likewise outside the bearer gate.)

Example:

```bash
curl -s http://localhost:8005/api/status \
  -H "Authorization: Bearer $PROMETHEUS_API_TOKEN"
```

## REST reference

All paths below are served on `:8005`. `{id}` placeholders are path parameters.

### Status & sessions

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Unauthenticated JSON API index (service name, version, endpoint list) |
| GET | `/health` | Unauthenticated liveness/staleness probe |
| GET | `/api/status` | Model, uptime, tools, memory, subsystem states — plus `node_pub` (the node's Ed25519 public key) and `instance_id` (the vault's UUID), both `null` until identity exists. Bearer-gated deliberately; `/health` never carries identity |
| GET | `/api/packs` | Discovered packs with load/refuse state, refusal reasons, quarantined-draft ids, and panel declarations. `wired: false` means the pack loader didn't run this boot (the bare `web` entrypoint), distinct from "no packs installed" |
| GET | `/api/mcp/servers` | MCP server cards: transport summary, **probed** health (a dead subprocess reads unhealthy, never as empty success), tool inventory, `allowed_tools`, `env_names`. Credential values never appear — write-only, the provider-keys stance |
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

### Chat

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/chat/send` | Send a chat message. Body: `{"session_id": "my-session", "message": "..."}` (the field is `message`, not `content`); optional `mode` (`"agent"`/`"chat"`) and `tool_choice`. Returns `{"run_id", "status": "sent"}`; the reply streams over the WebSocket and lands in `GET /api/sessions/{session_id}/messages` |
| POST | `/api/chat/interrupt` | Stop the running agent turn in a session — the chat Stop button. `{"session_id": ...}`; idempotent (`stopped: false` when nothing is running). Completed rounds persist, a mid-generation partial is kept as an assistant turn, and every client sees the broadcast `chat_done{interrupted:true}`. HTTP twin of the WS `interrupt` frame |
| POST | `/api/chat` | Alternate chat send endpoint |

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
| POST | `/api/learning/skill-drafts/{draft_id}/accept` | Persist a reviewed draft to `skills/auto/` (optionally with edited `content`); goes through the same validated write path DOM recordings use |
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
| GET | `/api/models` | Model catalog (local + cloud providers) |
| GET | `/api/sessions/{session_id}/model` | Current per-session model override |
| POST | `/api/sessions/{session_id}/model` | Set a per-session override (`local` clears back to the primary) |
| DELETE | `/api/sessions/{session_id}/model` | Clear the override |

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

When the daemon starts with **no config file**, it boots a minimal setup server (`src/prometheus/web/setup_server.py`) instead of the full API. Only five routes exist in this mode:

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/setup/status` | Setup progress / pairing window state |
| POST | `/api/setup/pair` | Exchange the 6-digit pairing code for an API token |
| GET | `/api/setup/detect` | Probe for local backends (llama.cpp, Ollama, LM Studio, vLLM); also lists the cloud presets (`cloud_providers`: name, key env var, default model, whether the daemon already holds the key — never the value) |
| POST | `/api/setup/configure` | Write the chosen configuration. Local: `provider` + `base_url` + `model`, re-probed before writing. Cloud: `provider` (a preset name) + `api_key` (+ `model`), no `base_url`; the key goes to the env file, and a cloud provider with no key anywhere is refused (`cloud_key_missing`) |
| POST | `/api/setup/complete` | Finish setup and hand off to the full daemon |

The pairing flow: at startup the daemon prints a crypto-random 6-digit code once in a console banner, and a client (Beacon's first-run screen, or curl) POSTs it to `/api/setup/pair` as `{"code": "123456"}` to receive the bearer token. The code is one-time-use, expires after 15 minutes, and locks after 5 failed attempts (only a wrong code burns an attempt); comparison uses constant-time `hmac.compare_digest`. Once paired, the client uses the returned token for the remaining setup calls and for the full API after `complete`.

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
| `send_message` | Send a chat turn; accepts optional `tool_choice` (validated against the live tool registry) and a `client_msg_id` for echo correlation |
| `chat_upload` | Upload an attachment (base64); images get vision captions, documents get text extraction |
| `switch_session` | Point this socket at a different session |
| `interrupt` | Stop the running turn: `{"type": "interrupt", "payload": {"session_id": ...}}`. The requesting socket gets an `interrupt_ack`; every client learns the outcome from the broadcast `chat_done{interrupted:true}` |

Example `send_message`:

```json
{
  "type": "send_message",
  "session_id": "web:default",
  "content": "Summarize today's telemetry",
  "tool_choice": null
}
```

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

Skill-draft lifecycle events (`skill_draft_created` / `skill_draft_accepted` / `skill_draft_rejected` / `video_ingest_failed`) ride the `sentinel_signal` channel — watch its `kind` field.

## Building a client

The sync contract is deliberately simple: **`message_id` is the durable LCM rowid — monotonic, unique, and restart-stable — and it doubles as the sync cursor.** `GET /api/sessions/{id}/messages?since=<message_id>` returns only rows after your cursor, and every response includes a top-level `watermark` (the session's current max `message_id`) so a client knows it is caught up even when an incremental read comes back empty. Session rows from `GET /api/sessions` carry the same `watermark`, so one list call tells you which sessions have news. A malformed `since` is a **400**, never silently ignored. Don't key on `ordinal` or `timestamp` — both repeat; order by `message_id`.

## Ports & remote access

- REST: **:8005** (bound `0.0.0.0`). WebSocket: **:8010** (bound `0.0.0.0`).
- Beacon expects exactly these two ports on the daemon host — they are not currently negotiated.
- Both are reachable over Tailscale, which is the intended remote-access path (e.g. Beacon Desktop on a laptop talking to the daemon box); the bearer token and WS first-frame auth are what stand between the ports and the tailnet.
