# feat(api): provider-keys REST — /api/providers/keys catalog + secure per-service key writes (MODELS KEYS UI, daemon half)

**Branch:** `feat/api-provider-keys` off main (`06abc9d`)
**Status:** PR-ready — not merged, main untouched, live daemon untouched (E2E ran an ISOLATED daemon on :18055 with tmp dirs).
**Companion:** beacon-desktop `feat/model-keys-ui` (the "Models" Config tab that consumes this surface).

---

## What shipped

A small bearer-authed REST surface that lets a non-technical Beacon user paste provider API keys and have them live **immediately** — persisted to the daemon's 0600 env file AND exported into `os.environ`, no restart.

### 1. `GET /api/providers/keys` — the catalog, booleans only

```json
{"services": [{"id": "deepseek", "label": "DeepSeek", "kind": "llm",
               "env_vars": [{"name": "DEEPSEEK_API_KEY", "set": false}],
               "docs_url": "https://platform.deepseek.com/api_keys",
               "default_model": "deepseek-v4-flash"}, …]}
```

- 10 services: the 8 LLM presets (claude/gpt/gemini/xai/deepseek/kimi/glm/mimo) + `wan-image` (DASHSCOPE_API_KEY) + `kling-video` (**the one two-var service**: KLING_ACCESS_KEY + KLING_SECRET_KEY).
- **NEVER returns values, prefixes, or lengths** — `set` is a boolean computed from `os.environ` truthiness at request time.

### 2. ONE catalog, derived — new `providers/key_catalog.py`

- LLM entries derive from `router.OVERRIDE_PRESETS` via `resolve_slash_command_target()` — the SAME resolution path `/claude` and `GET /api/models` use, so a user's `slash_commands.<key>` config merges over the preset here too (test-pinned). **Adding a preset to OVERRIDE_PRESETS shows up in the endpoint with zero endpoint changes** (test-pinned via monkeypatched preset).
- Media entries derive from the image/video tool constants (`_DASHSCOPE_DEFAULT_KEY_ENV`, `_KLING_ACCESS_KEY_ENV`/`_KLING_SECRET_KEY_ENV`, default models) — imported, not re-typed.
- The only things declared in the catalog module are presentation metadata: human labels + console `docs_url`s (deepseek/kimi/z.ai/xiaomimimo/Alibaba Model Studio/Kling dev portal).
- **Label single-sourcing:** server.py's function-local `_PRESET_LABELS` dict is replaced by `key_catalog.PRESET_LABELS` — `/api/models` and `/api/providers/keys` can no longer disagree on names. (`test_cloud_expansion.py::test_model_rest_catalog_labels` updated from a source-grep to a direct import assert + a pin that server.py imports the shared dict.)

### 3. `PUT /api/providers/keys/{service}` — validate-all-then-write

Body `{"values": {"ENV_VAR": "key", …}}`:
- Env-var names must belong to that service → **400 otherwise, with NOTHING written** (all-or-nothing; a smuggled `OPENAI_API_KEY` in a deepseek PUT persists neither var — E2E step 7).
- Unknown service → 404. Non-string / >4096 chars / control characters → 400 (**the env file is line-based — a `\n` in a value could inject a second `VAR=` assignment**; refused outright, test-pinned).
- Values are `strip()`ed (pasted keys arrive with stray whitespace).
- Valid values → `set_env_value()` into the env file (config/env_file.py — 0600 on create, dedup, PROMETHEUS_ENV_FILE-overridable) **and** `os.environ[var] = value`.
- **Empty string = explicit clear** (documented choice, matching `api_token.py`'s deliberately-blank convention): the env file keeps a deliberate blank `VAR=` assignment, and the var is `pop`ped from `os.environ`. `load_env_file`'s setdefault semantics keep the blank harmless on next boot.
- Response = the same catalog + booleans as GET, so the client refreshes its whole view from the PUT response.

### 4. Immediate liveness — cited + verified, not assumed

The key is live in the SAME process because every consumer reads `os.environ` at use time:
- `providers/registry.py:93-107` — `ProviderRegistry.create()` reads `config["api_key_env"]` from `os.environ` at provider-CREATE time (overrides build providers lazily on first route).
- `gateway/commands.py::cmd_provider_override` — checks `_os.environ.get(api_key_env)` at command time (`/deepseek` stops saying "requires DEEPSEEK_API_KEY" instantly).
- `web/server.py::_model_catalog` — `available: bool(env and os.environ.get(env))` at request time → **`GET /api/models` flips `available: true` in-process** (E2E step 5, same PID).

### 5. Secrecy: grep-proof

- No log line ever carries a value — the audit log is `provider key <service>: set|cleared <ENV_VAR>` (names only).
- Tests assert the fake key is absent from **caplog at DEBUG** and from every response body; the E2E greps the whole daemon log for all three fake values → clean.

## Acceptance — isolated live E2E (transcript)

Booted a REAL `python -m prometheus.daemon` with tmp `PROMETHEUS_CONFIG_DIR`/`PROMETHEUS_DATA_DIR`/`PROMETHEUS_ENV_FILE`, stub model provider, `web.enabled: true`, token auth, **REST :18055 / WS :18056** — the live daemon on :8005/:8010 was never touched (verified 401-healthy before and after).

```
== 0. unauthed GET is 401 ==                    HTTP 401
== 1. GET catalog — all not-set ==              10 services, every var unset
                                                (claude/gpt/gemini/xai/deepseek/kimi/glm/mimo
                                                 + wan-image + kling-video w/ TWO vars)
== 2. GET /api/models — before ==               deepseek available = False
== 3. PUT fake DEEPSEEK_API_KEY ==              response: [{'name': 'DEEPSEEK_API_KEY', 'set': True}]
== 4. tmp env file ==                           mode=600, fake key present (1 line)
== 5. GET /api/models — SAME PROCESS ==         deepseek available = True   ← no restart
== 6. PUT kling pair ==                         both booleans True; 2 vars in env file
== 7. wrong var name → 400 ==                   HTTP 400; smuggled lines in env file: 0
== 8. PUT empty → explicit clear ==             set: False; deliberate blank DEEPSEEK_API_KEY= line;
                                                /api/models available = False again
== 9. grep-proof ==                             no key value anywhere in daemon.log;
                                                4 name-only audit lines
```

## Tests

- New `tests/test_api_provider_keys.py` — 16 tests: catalog shape/grouping/no-values, **derives-from-presets** (monkeypatched preset appears endpoint-untouched), user `slash_commands` merge, PUT persists + 0600 + `os.environ` + `/api/models` flip in-process, kling two-var + partial update, whitespace strip, empty-string clear, wrong-var-name 400 writes-nothing, malformed bodies 400, control-char 400, unknown service 404, bearer 401, catalog-module unit.
- `tests/test_cloud_expansion.py` label test upgraded (see §2).
- **Full suite: 3430 passed, 4 skipped, 0 failed, no deselects** (`uv sync` w/ web+anthropic+slack+discord extras + dev; `PYTHONPATH=$PWD/src uv run pytest`). That count includes the 16 new tests — pre-branch baseline ≈ 3414 passed.

## Files

- `src/prometheus/providers/key_catalog.py` — NEW: derived catalog + PRESET_LABELS + docs URLs
- `src/prometheus/web/server.py` — GET/PUT `/api/providers/keys*` + label import (model REST otherwise untouched)
- `tests/test_api_provider_keys.py` — NEW
- `tests/test_cloud_expansion.py` — label test direct-assert upgrade

## Notes / deviations

- Kling `docs_url` is `https://app.klingai.com/global/dev` (the actual dev-portal host; the sprint note's shorthand "kling.ai/dev" is not a live console URL).
- PUT allows a partial update on the two-var service (each var validated individually); the Beacon UI sends both fields for Kling, and `serviceKeyState` reports "partial" honestly if only one is set.
