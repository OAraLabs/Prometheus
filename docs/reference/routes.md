<!-- GENERATED FILE — DO NOT EDIT BY HAND.
     Regenerate with:  uv run python scripts/gen_reference.py
     Pinned current by tests/test_generated_reference.py. -->

# Route reference (generated)

Every HTTP route the daemon mounts, read out of the live FastAPI
app. For what each one is *for*, see the curated
[API guide](../guide/api.md) — this file is the complete list, that
one is the explanation.

## Daemon app — 100 paths

| Path | Methods |
|---|---|
| `/` | GET |
| `/api/activity/recent` | GET |
| `/api/approvals` | GET |
| `/api/approvals/grants` | GET |
| `/api/approvals/grants/{grant_id}` | DELETE |
| `/api/approvals/{request_id}/approve` | POST |
| `/api/approvals/{request_id}/deny` | POST |
| `/api/artifacts` | GET |
| `/api/artifacts/{artifact_id}` | GET |
| `/api/backends` | GET |
| `/api/backends/{name}/probe` | POST |
| `/api/benchmarks/run` | POST |
| `/api/chat` | POST |
| `/api/chat/interrupt` | POST |
| `/api/chat/send` | POST |
| `/api/code` | POST |
| `/api/code/{task_id}` | GET |
| `/api/code/{task_id}/diff` | GET |
| `/api/code/{task_id}/inject` | POST |
| `/api/code/{task_id}/pause` | POST |
| `/api/code/{task_id}/resume` | POST |
| `/api/code/{task_id}/stop` | POST |
| `/api/config` | GET |
| `/api/cron` | GET, POST |
| `/api/cron/{name}` | DELETE, PUT |
| `/api/cron/{name}/run` | POST |
| `/api/devices` | GET, POST |
| `/api/devices/{device_id}` | DELETE |
| `/api/devices/{device_id}/activity` | DELETE, POST |
| `/api/devices/{device_id}/push` | DELETE, PUT |
| `/api/documents` | GET |
| `/api/documents/content` | GET, PUT |
| `/api/documents/edit` | POST |
| `/api/documents/suggest` | POST |
| `/api/events/recent` | GET |
| `/api/files` | GET |
| `/api/files/read` | GET |
| `/api/lcm/{session_id}` | GET |
| `/api/learning/live-upload` | POST |
| `/api/learning/skill-drafts` | GET |
| `/api/learning/skill-drafts/{draft_id}` | GET |
| `/api/learning/skill-drafts/{draft_id}/accept` | POST |
| `/api/learning/skill-drafts/{draft_id}/reject` | POST |
| `/api/learning/video-ingest` | POST |
| `/api/mcp/servers` | GET, POST |
| `/api/mcp/servers/{name}` | DELETE, PATCH |
| `/api/media` | GET |
| `/api/memory/current` | GET, PUT |
| `/api/models` | GET |
| `/api/packs` | GET |
| `/api/pairs` | GET |
| `/api/paperclip/wake` | POST |
| `/api/profiles` | GET |
| `/api/profiles/active` | PUT |
| `/api/project-file` | GET, PUT |
| `/api/projects` | GET, POST |
| `/api/projects/{project_id}` | DELETE, PUT |
| `/api/providers/keys` | GET |
| `/api/providers/keys/{service_id}` | PUT |
| `/api/providers/xai/oauth` | DELETE, GET |
| `/api/providers/xai/oauth/login` | POST |
| `/api/search` | POST |
| `/api/sentinel` | GET |
| `/api/sessions` | GET, POST |
| `/api/sessions/{session_id}` | DELETE |
| `/api/sessions/{session_id}/checkpoints` | GET |
| `/api/sessions/{session_id}/checkpoints/{checkpoint_id}` | GET |
| `/api/sessions/{session_id}/checkpoints/{checkpoint_id}/restore` | POST |
| `/api/sessions/{session_id}/fork` | GET, POST |
| `/api/sessions/{session_id}/messages` | GET |
| `/api/sessions/{session_id}/model` | DELETE, GET, POST |
| `/api/sessions/{session_id}/pin` | PUT |
| `/api/sessions/{session_id}/profile` | GET, PUT |
| `/api/sessions/{session_id}/purge` | POST |
| `/api/sessions/{session_id}/title` | PUT |
| `/api/sessions/{session_id}/workspace` | DELETE, GET, PUT |
| `/api/skills` | GET |
| `/api/skills/list` | GET |
| `/api/skills/{name}` | GET |
| `/api/skills/{name}/pin` | DELETE, POST |
| `/api/status` | GET |
| `/api/stories` | GET, POST |
| `/api/stories/reorder` | POST |
| `/api/stories/{story_pk}` | DELETE, PUT |
| `/api/stories/{story_pk}/dispatch` | POST |
| `/api/stories/{story_pk}/undispatch` | POST |
| `/api/tasks` | GET |
| `/api/tasks/{task_id}` | GET |
| `/api/tasks/{task_id}/stop` | POST |
| `/api/telemetry` | GET |
| `/api/tools/deferred` | GET, PUT |
| `/api/tools/recent` | GET |
| `/api/usage` | GET |
| `/api/wiki/page` | GET |
| `/api/wiki/pages` | GET |
| `/api/wiki/search` | GET |
| `/api/wiki/stats` | GET |
| `/health` | GET |
| `/v1/chat/completions` | POST |
| `/v1/models` | GET |

## Setup app — 6 paths

A separate FastAPI app (`web/setup_server.py`), not the daemon app
behind a flag: the real route surface is deliberately never mounted
in setup mode.

| Path | Methods |
|---|---|
| `/api/setup/complete` | POST |
| `/api/setup/configure` | POST |
| `/api/setup/detect` | GET |
| `/api/setup/pair` | POST |
| `/api/setup/status` | GET |
| `/{path:path}` | DELETE, GET, PATCH, POST, PUT |

## FastAPI built-ins — 4 paths

Mounted by FastAPI itself. Listed because they are reachable:
`/openapi.json` publishes the entire surface.

| Path | Methods |
|---|---|
| `/docs` | GET |
| `/docs/oauth2-redirect` | GET |
| `/openapi.json` | GET |
| `/redoc` | GET |
