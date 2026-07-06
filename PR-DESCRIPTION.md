# feat(cloud): CLOUD EXPANSION — four LLM providers + WAN 2.5 image backend + Kling 3.0 video tool

**Branch:** `feat/cloud-expansion` off main (`d0cbcb0`)
**Status:** PR-ready — not merged, main untouched, live daemon untouched.
**Fakes-tested, dormant-until-keyed** — NO API keys exist on this box; that is the explicitly accepted state. Every new service returns an actionable "here's the console, here's the env var" error until keyed. Live testing is tabled by design.

---

## What shipped

### 1. Four new LLM providers (DeepSeek · Kimi/Moonshot · GLM/Z.ai · MiMo/Xiaomi)

All four ride the existing OpenAI-compat wire path — no new provider classes.

| provider | base_url | default model | env var |
|---|---|---|---|
| `deepseek` | `https://api.deepseek.com` | `deepseek-v4-flash` | `DEEPSEEK_API_KEY` |
| `kimi` | `https://api.moonshot.ai/v1` | `kimi-k2.6` | `MOONSHOT_API_KEY` |
| `glm` | `https://api.z.ai/api/paas/v4` | `glm-5.2` | `ZAI_API_KEY` |
| `mimo` | `https://api.xiaomimimo.com/v1` | `mimo-v2.5-pro` | `MIMO_API_KEY` |

- `providers/registry.py`: `CLOUD_DEFAULTS` + `_OPENAI_COMPAT_PROVIDERS` + `create()` error text + `list_providers()` (7 → 11); `is_cloud()` covers all four.
- **DeepSeek V4 pin**: the legacy `deepseek-chat`/`deepseek-reasoner` aliases are deprecated 2026-07-24 — V4 names ship everywhere, and a test PINS that no default is a deprecated alias. `deepseek-v4-pro` (reasoning flagship) is documented in comments, the wizard menu, and the yaml default.
- **GLM base-with-path URL join fixed**: new `openai_compat._chat_completions_url()` — a base whose final path segment is a version tag (`/v1`, `/v4`, `/v1beta`) gets `/chat/completions` appended directly; everything else keeps the historical `/v1/chat/completions`. `https://api.z.ai/api/paas/v4` → `…/api/paas/v4/chat/completions` (the old `endswith("/v1")` logic would have produced a bogus `…/v4/v1/chat/completions`). Gemini's `/v1beta/openai` base keeps its historical byte-identical behavior (pinned by test).
- `router/model_router.py`: `OVERRIDE_PRESETS` + `SLASH_COMMAND_NAMES` gain `/deepseek /kimi /glm /mimo`; `resolve_slash_command_target` + startup wiring log work unchanged.
- `gateway/commands.py`: `PROVIDER_PRESET_DISPLAY_NAMES` + `/route` output list.
- **All three gateways** registered: Telegram (`CommandHandler` + `BotCommand` menu + `/help`), Slack (`/prometheus-deepseek` etc. + help), Discord (`/prometheus provider deepseek` etc. + help). Parity manifest extended honestly — four new `Family` rows, zero gaps (chart below).
- Keys/config: `ENV_OVERRIDES` (4 LLM vars + DASHSCOPE + the KLING pair), wizard cloud menus (rich `setup_wizard.py` + fast-path `cli/init.py`, appended so historical menu numbering stays stable), env-file template hint lines, `prometheus.yaml.default` `slash_commands` blocks, `model_registry.yaml` entries (function_calling + streaming true; contexts: deepseek 1M, kimi 256K, glm-5.2 1M, mimo ~1M), `telemetry/cost.py` PRICING (uncertain rows carry a "2026-07 research, verify at first live use" comment).

### 2. WAN 2.5 image backend (`dashscope`)

- `image_generate` gains a fourth backend: `BackendName` union + `_generate_via_dashscope()` — async-ONLY task API (POST `/services/aigc/image-generation/generation` with header `X-DashScope-Async: enable` → task_id → poll `GET /api/v1/tasks/{id}` → download bytes IMMEDIATELY since result URLs expire in 24h → existing `_save_image_bytes` cache path).
- **HARD RULE enforced + test-pinned**: `auto` backend resolution NEVER selects the paid API — auto stays comfyui→pollinations even with `DASHSCOPE_API_KEY` set. DashScope runs only via an explicit `backend=dashscope` argument or a deliberate `default_backend: dashscope` config.
- Missing key + explicit request → honest error naming the console, the env var, and the free alternatives (and makes zero HTTP calls — pinned).
- Config: `image_generation.dashscope: {api_key_env, model, base_url}` — base_url overridable for the newer workspace-scoped DashScope domains.

### 3. Kling 3.0 video tool (`video_generate`)

- New `src/prometheus/tools/builtin/video_generate.py` → `KlingVideoTool`, tool name `video_generate`. Input: `{prompt, image_path?→i2v, duration 5|10, model_name (default kling-v3), resolution 720p|1080p}` (resolution maps to Kling `mode` std/pro — flagged verify-at-first-live-use).
- **STDLIB-ONLY JWT** (hmac + hashlib + base64 + json — NO PyJWT, no new deps): HS256, payload `iss`=AccessKey, `exp`=now+1800, `nbf`=now−5, correct base64url-no-padding encoding, re-minted per request. Pinned by a known-answer test whose expected token was **cross-verified byte-for-byte against PyJWT 2.x** (throwaway venv) at authoring time.
- Flow: POST `/v1/videos/text2video` (or `image2video` with base64 image) → task_id → poll every ~10 s with budget `video_generation.kling.poll_budget_seconds` (default 600 — videos take MINUTES) → honest timeout message → download to `~/.prometheus/cache/videos/` (new `media_cache.cache_video_from_bytes` + `video_cache_dir`, mirroring the image cache).
- Missing keys → actionable error pointing at the Kling console; zero HTTP without keys (pinned).
- Registered in `create_tool_registry` (`__main__.py`) beside `image_generate` — the single registry BOTH entry points build (the daemon reuses `create_tool_registry` via `daemon.py:75-82`; no second registration site exists). Permission treatment mirrors image_generate: no model-chosen output path exists, so the tool is cache-confined (`is_read_only` → True, matching image_generate's `output_path is None` case).

### 1b. Drive-by catches (enumeration sweep after the main wiring)

A grep sweep for every hardcoded provider enumeration caught four sites the study's file list didn't name — all fixed + test-pinned:

- **`router/model_router.py::_build_adapter_for`** (the real one): the override path's adapter factory listed `("openai", "gemini", "xai")` — the new providers would have fallen through to the LOCAL full pipeline (QwenFormatter + text extraction) instead of tier=off. Now all four build `PassthroughFormatter` @ `TIER_OFF`.
- **`telemetry/tracker.py::_CLOUD_PROVIDERS`**: golden-trace capture's deliberate duplicate of `is_cloud()` — four names added.
- **`web/slash_router.py::WEB_NATIVE_ONLY`**: the web chat surface's boundary list mirrors Telegram registrations — `/deepseek` etc. now get the explicit boundary reply instead of silently running the agent.
- **`web/server.py::_PRESET_LABELS`**: GET `/api/models` catalog labels (the catalog itself iterates `OVERRIDE_PRESETS`, so the new presets appear automatically; labels now read DeepSeek/Kimi/GLM/MiMo instead of raw keys).
- **Tool execution timeout vs poll budgets**: the agent loop kills tools at `LoopContext.tool_timeout_seconds` (300s default) — Kling's poll budget alone is 600s and DashScope's is 300s + submit/download. Both tools now set the M5 `execution_timeout_seconds` class override (`video_generate` 900s, `image_generate` 480s) so a slow render surfaces the tool's own honest timeout message instead of being killed mid-poll. Test-pinned (override must exceed the poll budget).

### 4. Doctor

- New connectivity info line (`infra/doctor.py::_check_cloud_keys`): `Cloud keys: DeepSeek set/not set · Kimi … · GLM … · MiMo … · DashScope/WAN … · Kling AK+SK …` — set/not-set only, values never echoed (test-pinned: not even a prefix). Absence is `info`, never a warning — dormant-until-keyed is a non-event.

### 5. Docs

- README: provider counts/diagram/config comment, command table + slash-commands sample (all 8 override commands), new **Image & Video Generation** section stating the never-auto-paid rule. Env-file template (`cli/init.py::_ENV_TEMPLATE`) gains hint lines for all six services.

---

## Acceptance

### 1. Parity manifest (new families, three columns)

From `PYTHONPATH=$PWD/src uv run pytest -s tests/test_gateway_parity.py::TestParityReport`:

```
family        telegram                  slack                     discord                   shared
deepseek      deepseek                  prometheus-deepseek       deepseek                  cmd_provider_override
kimi          kimi                      prometheus-kimi           kimi                      cmd_provider_override
glm           glm                       prometheus-glm            glm                       cmd_provider_override
mimo          mimo                      prometheus-mimo           mimo                      cmd_provider_override
```

Zero platform gaps for the new families; both drift directions (manifest→registration and registration→manifest) remain CI-enforced.

### 2. Test counts

- **Full suite: 3415 passed, 4 skipped, 0 failed** (`PYTHONPATH=$PWD/src uv run pytest`), zero deselects; skips are pre-existing optional-dep skips. Pre-branch baseline: 3306 passed / 4 skipped.
- Sprint adds **109 new tests**: `tests/test_cloud_expansion.py` (81 — incl. adapter-tier, golden-trace-set, and web-surface pins for the drive-by catches), `tests/test_image_dashscope.py` (10), `tests/test_video_generate.py` (18); plus honest updates to existing pins: `test_cloud_providers.py` (list_providers 7→11, is_cloud), `test_cost.py` (+5 pricing-coverage models), `test_gateway_command_pins.py` (the `/route` list deliberately grew — pins updated with a note), `test_gateway_parity.py` (4 families).
- CI (GitHub Actions) green on the PR: `test (3.11)` pass, `test (3.12)` pass.

### 3. Fake-transport transcripts (mocked endpoints; fake keys; auth redacted)

<details><summary>WAN 2.5 image generate — backend=dashscope</summary>

```
>>> POST https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/image-generation/generation
    headers: {"Authorization": "Bearer sk-FAKE-T…<redacted>", "Content-Type": "application/json", "X-DashScope-Async": "enable"}
    body: {"model": "wan2.5-t2i-preview", "input": {"prompt": "a red lighthouse at dusk, crashing waves, photorealistic"}, "parameters": {"size": "1024*1024", "n": 1}}
<<< 200 {"output": {"task_id": "task-wan-0001", "task_status": "PENDING"}, "request_id": "req-fake-1"}

>>> GET https://dashscope-intl.aliyuncs.com/api/v1/tasks/task-wan-0001
<<< 200 {"output": {"task_id": "task-wan-0001", "task_status": "PENDING"}}
>>> GET https://dashscope-intl.aliyuncs.com/api/v1/tasks/task-wan-0001
<<< 200 {"output": {"task_id": "task-wan-0001", "task_status": "RUNNING"}}
>>> GET https://dashscope-intl.aliyuncs.com/api/v1/tasks/task-wan-0001
<<< 200 {"output": {"task_id": "task-wan-0001", "task_status": "SUCCEEDED", "results": [{"url": "https://dashscope-result.oss-accelerate.aliyuncs.com/wan/out-1.png"}], "task_metrics": {"TOTAL": 1, "SUCCEEDED": 1, "FAILED": 0}}}

>>> GET https://dashscope-result.oss-accelerate.aliyuncs.com/wan/out-1.png     # 24h-expiring URL → downloaded NOW
<<< 200 <binary, 240004 bytes, content-type=image/png>

--- ToolResult (is_error=False) ---
Saved image to ~/.prometheus/cache/images/img_9c2f1a.png
  backend: dashscope (WAN 2.5, paid)
  size: 1024x1024
  bytes: 234.4 KB
  model: wan2.5-t2i-preview
```
</details>

<details><summary>Kling 3.0 video generate — text2video</summary>

```
>>> POST https://api-singapore.klingai.com/v1/videos/text2video
    headers: {"Authorization": "Bearer eyJhbGciO…<redacted — self-signed HS256 JWT, re-minted per request>", "Content-Type": "application/json"}
    body: {"model_name": "kling-v3", "prompt": "a paper boat drifting down a rain gutter stream, low tracking shot, cinematic", "duration": "5", "mode": "pro"}
<<< 200 {"code": 0, "message": "SUCCEED", "data": {"task_id": "task-kling-0007", "task_status": "submitted"}}

>>> GET https://api-singapore.klingai.com/v1/videos/text2video/task-kling-0007
<<< 200 {"code": 0, "data": {"task_id": "task-kling-0007", "task_status": "submitted"}}
>>> GET https://api-singapore.klingai.com/v1/videos/text2video/task-kling-0007
<<< 200 {"code": 0, "data": {"task_id": "task-kling-0007", "task_status": "processing"}}
>>> GET https://api-singapore.klingai.com/v1/videos/text2video/task-kling-0007
<<< 200 {"code": 0, "data": {"task_id": "task-kling-0007", "task_status": "succeed", "task_result": {"videos": [{"id": "v-0007", "url": "https://kling-result.example.com/videos/v-0007.mp4", "duration": "5"}]}}}

>>> GET https://kling-result.example.com/videos/v-0007.mp4
<<< 200 <binary, 3400012 bytes, content-type=video/mp4>

--- ToolResult (is_error=False) ---
Saved video to ~/.prometheus/cache/videos/vid_5e7d2b.mp4
  backend: kling (text2video)
  model: kling-v3
  duration: 5s
  resolution: 1080p
  size: 3.2 MB
  task: task-kling-0007
```
</details>

(Reproducible via `scripts`-free scratch harness; the same shapes are pinned by `tests/test_image_dashscope.py::TestDashscopeHappyPath` and `tests/test_video_generate.py::TestText2Video`.)

### 4. First-live-use checklist (per service)

After adding any var to `~/.config/prometheus/env`: **set OAra maintenance mode, then `systemctl --user restart prometheus`** (the env file is loaded at startup).

| service | console | env var(s) | one-command smoke |
|---|---|---|---|
| DeepSeek | https://platform.deepseek.com/api_keys | `DEEPSEEK_API_KEY` | Telegram: `/deepseek what is 2+2?` |
| Kimi (Moonshot) | https://platform.moonshot.ai/console/api-keys — CN keys are separate: platform.moonshot.cn + swap `base_url` | `MOONSHOT_API_KEY` | Telegram: `/kimi what is 2+2?` |
| GLM (Z.ai) | https://z.ai → API console (CN mirror: open.bigmodel.cn, same shape) | `ZAI_API_KEY` | Telegram: `/glm what is 2+2?` |
| MiMo (Xiaomi) | https://api.xiaomimimo.com platform console | `MIMO_API_KEY` | Telegram: `/mimo what is 2+2?` |
| WAN 2.5 image | https://dashscope.console.aliyun.com (intl accounts: Alibaba Cloud Model Studio) | `DASHSCOPE_API_KEY` | ask the agent: `image_generate backend=dashscope prompt="a test pattern"` |
| Kling 3.0 video | https://app.klingai.com → API access (AccessKey + SecretKey pair) | `KLING_ACCESS_KEY` + `KLING_SECRET_KEY` | ask the agent: `video_generate prompt="a paper boat drifting down a stream" duration=5` (expect minutes; the tool polls) |

Also at first live use: verify the five PRICING rows marked "2026-07 research" against each provider's pricing page, and confirm the Kling `resolution→mode` mapping (720p=std, 1080p=pro) plus the WAN `size` parameter format.

### 5. Grep-proof — no secrets

- The full branch diff was scanned for key-shaped literals (`sk-…{24,}`, `xoxb-`, `xapp-`, `AKIA…`, long inline `api_key=` values): **zero hits**. All test keys are obvious fakes (`test-key`, `sk-FAKE-TRANSCRIPT-KEY`, `test-ak`/`test-sk`).
- Log paths: providers log model/URL only; `env_override.apply_env_overrides` masks anything with `key`/`token` in the name; doctor's new check prints set/not-set only (test-pinned to never echo a value or prefix); the minted Kling JWT appears in no log line.

---

## Deviations / notes

- **Wizard menu numbering**: "Where is your LLM?" grew 7 → 11 entries; the four cloud options were appended after xAI so local options keep positions 1–2, and "I don't have one running yet" moved 7 → 11 (`cloud_map` updated). No test pinned the old numbering; the fast-path and dead-end menus appended options so their existing scripted-input numbering is unchanged.
- **/route reply pins deliberately updated**: `test_gateway_command_pins.py` byte-pins the `/route` override list; the surface intentionally gained four lines, so the pins were updated in the same change with an explanatory note.
- **`resolution` → Kling `mode`**: the tool's `resolution` field (sprint-specified) maps to Kling's `mode` param (std/pro) — commented in-source as a 2026-07 research interpretation to verify live.
- **Doctor severity**: one `Cloud keys` info line (never warning/error) — a missing paid key is a non-event by design.
- **model_registry ordering**: the new `deepseek-v4` cloud family sits BEFORE the legacy local-GGUF `deepseek` family because `match_model` returns the first hit in file order.
- **Effective context limits**: wizard defaults for the four new providers are a conservative 64000 (not their advertised 256K–1M windows) — request cost stays sane; raising `context.effective_limit` is a deliberate user act.
- **Hard rules held**: live daemon untouched (no restarts, no :8005/:8010, no second daemon), no new pip dependencies (stdlib JWT; httpx for all HTTP), no real keys anywhere.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
