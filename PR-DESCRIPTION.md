# Model REST — Sprint B / Piece 1 (expose-only, no engine change)

**Branch:** `feat/model-rest` off origin/main (`4cef0c5`)
**Status:** PR-ready. **LIGHT** — exposes the EXISTING per-session `ModelRouter` override engine via bearer-authed REST. **No engine change.** Unblocks a later Beacon model-switcher UI.

## What it does
The model-switching engine already exists and is live-wired: `set_override(session_id, …)` is per-session, `route()` consults the override **first** (model_router.py:444-448), and `run_loop` re-routes every turn so a switch takes effect next turn. The only gap was no web-reachable API — the `/claude` slash path is Telegram-only / web-blocked. This adds REST routes over that engine. `model_router` / `run_loop` / `route()` are untouched.

## Routes (all bearer-authed via the existing `/api/*` middleware)
- **`GET /api/models`** — the switcher catalog: the configured primary as "Local" (key `local`) + each vetted preset (claude/gpt/gemini/xai). Each option: `key`, `label`, `provider`, `model`, `is_default`, `available` (api-key env **presence only** — never the value, never the env-var name). `default_key = "local"`.
- **`GET /api/sessions/{id}/model`** — the session's current effective option (active override, else local).
- **`POST /api/sessions/{id}/model`** `{key}` — `local` → `clear_override`; a preset key → `set_override` with the resolved config. Returns the new effective option.
- **`DELETE /api/sessions/{id}/model`** — `clear_override` (idempotent → clean 200).

## Key design points
- **Key-only, server-mapped.** Clients pick a KEY from the catalog; the daemon maps it to a vetted config. A client can **never** inject a raw provider/model/api_key payload — the switch surface is the 4 presets + local.
- **Faithful to the slash path.** The key→config mapping goes through the SAME `resolve_slash_command_target(key, config)` the `/claude` command uses, so REST `/claude` == Telegram `/claude`. **Live acceptance proved this matters:** this host's `slash_commands` config resolves **claude → claude-sonnet-4-5** and **gemini → gemini-2.5-pro** (not the hardcoded haiku/flash) — the catalog + stored override now reflect that.
- **No secrets.** Responses never include api-key values or env-var names — only `available` (presence).
- **No engine change.** Routes call `set_override`/`clear_override`/`get_override_for_session` only. The router is wired to the web layer via a new `create_app(model_router=…)` param (launcher passes `loop_context.model_router`) — plumbing, not engine.
- **Clean failure codes:** reserved ids (`None`/`"system"`) → **400** (not 500); unknown key → **400**; router unwired → **503**; no bearer → **401** (inherited middleware).

## Files
- `src/prometheus/web/server.py` — `model_router` param + `app.state.model_router`; the 4 routes + catalog / effective-model helpers.
- `src/prometheus/web/launcher.py` — pass `model_router=loop_context.model_router` to `create_app`.
- `tests/test_api_model_rest.py` *(new, 10 tests)*.

## Tests & acceptance (no `--no-verify`)
- `tests/test_api_model_rest.py` **10/10** — side-effect assertions on a REAL router: catalog (no secrets), set → stored override **+ `route()` picks it up** (override-first path returns the override target), local-key clears, DELETE idempotent, unknown→400, reserved→400, no-bearer→401, unwired→503, and resolve-through-user-`slash_commands`. Web API suite **120** + model_router/router **204** green.
- **Live (daemon restarted on the branch):** `GET /api/models` (claude→sonnet, gemini→2.5-pro per this host's config; gpt/xai default, availability reflects key presence) → set claude → GET confirms → DELETE → local; sent a real turn into a claude-overridden session (completed); reserved→400, unknown→400, no-bearer→401. All bearer-authed.

## Follow-ups
- **Beacon model-switcher UI** — a later composer-UI sprint, stacks on this + Piece 2.
- **Piece 2** (per-message options channel for the Agent|Chat toggle) — next daemon sprint, MODERATE (survey-confirmed clean `tool_schema` seam); **force-search deferred** (no clean force mechanism).

## Out of scope (honored)
- No engine change (`model_router`/`route()`/`run_loop` untouched). No per-message options. No raw client provider-configs. No Beacon work. No secrets exposed.
