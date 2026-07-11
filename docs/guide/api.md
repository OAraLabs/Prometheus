# HTTP & WebSocket API

Prometheus exposes two programmatic surfaces: a FastAPI REST server on **:8005** (`src/prometheus/web/server.py`) and a WebSocket bridge on **:8010** (`src/prometheus/web/ws_server.py`) that streams chat and subsystem events in real time. Both are what Beacon (the web/desktop UI) talks to; anything Beacon can do, you can do with curl or a WebSocket client.

[← README](../../README.md)

## Authentication

Every `/api/*` route requires a bearer token:

```
Authorization: Bearer $PROMETHEUS_API_TOKEN
```

- The token is minted automatically on first daemon start (or by the setup wizard) and stored in `~/.config/prometheus/env`.
- Retrieve or invalidate it with the CLI: `prometheus token show` | `prometheus token rotate`.
- Requests with a missing or wrong token get a `401 {"error": "unauthorized — set Authorization: Bearer <token>"}`.
- `GET /health` is the only unauthenticated API endpoint — it lives outside `/api/` precisely so external monitors can poll it without credentials. (The static Beacon UI served at `/` is likewise outside the bearer gate.)

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
| GET | `/` | Serve Beacon static UI |
| GET | `/health` | Unauthenticated liveness/staleness probe |
| GET | `/api/status` | Model, uptime, tools, memory, subsystem states |
| GET | `/api/sessions` | List sessions |
| POST | `/api/sessions` | Create a session |
| GET | `/api/sessions/{session_id}/messages` | Message history |
| DELETE | `/api/sessions/{session_id}` | Forget a session (clears the in-memory working set) |
| GET | `/api/config` | Effective config (secrets redacted) |

### Chat

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/chat/send` | Send a chat message (supports optional `tool_choice`) |
| POST | `/api/chat` | Alternate chat send endpoint |

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
| GET | `/api/memory/current` | Current MEMORY.md / USER.md contents |
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
| GET | `/api/setup/detect` | Probe for local backends (llama.cpp, Ollama, LM Studio, vLLM) |
| POST | `/api/setup/configure` | Write the chosen configuration |
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
| `agent_state` | Agent thinking/idle state changes |
| `tool_call_start` / `tool_call_end` | Live tool-call boundaries |
| `chat_done` | Turn finished |
| `error` | Malformed frame or handler failure |

SignalBus fan-out (broadcast to all authed clients; payloads carry `session_id` where relevant):

| Type | Purpose |
|---|---|
| `sentinel_signal` | Sentinel memory-pipeline events |
| `dream_start` / `dream_phase` / `dream_complete` | Dream-cycle progression |
| `skill_created` / `skill_refined` | Learning-system skill events |
| `memory_updated` | Memory file changes |
| `curator_report` | Weekly curator consolidation report |
| `coding_round` / `coding_complete` / `coding_stream_error` | Coding-run live stream: per-round progress, terminal verdict, non-fatal stream interruption |

## Ports & remote access

- REST: **:8005** (bound `0.0.0.0`). WebSocket: **:8010** (bound `0.0.0.0`).
- Beacon expects exactly these two ports on the daemon host — they are not currently negotiated.
- Both are reachable over Tailscale, which is the intended remote-access path (e.g. Beacon Desktop on a laptop talking to the daemon box); the bearer token and WS first-frame auth are what stand between the ports and the tailnet.
