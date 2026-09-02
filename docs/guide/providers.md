# Models & providers

Prometheus is built for local inference first — llama.cpp or Ollama on your own GPU, with a Model Adapter Layer that makes open models reliable in a tool loop. But the same harness speaks to ten-plus providers, cloud included, and you can switch any single chat to a cloud model with one slash command and come home with `/local`. This page covers the local backends, the cloud providers and their keys, per-chat overrides, the SuperGrok subscription sign-in, key management, the router, force-search, and image/video generation.

[← README](../../README.md)

## Local backends

Prometheus talks to any inference server over HTTP; localhost or remote, it doesn't care.

- **llama.cpp** (`provider: llama_cpp`, default port 8080) — the first-class backend. This is the only backend that gets **GBNF grammar enforcement**: when the model is asked to emit a tool call, Prometheus sends a grammar that makes malformed JSON structurally impossible at the token level. Controlled by `model.grammar_enforcement: true` (the default). The loaded model is auto-detected from the server at startup — swap the GGUF, restart, done.
- **Ollama** (`provider: ollama`, default port 11434) — fully supported, and the default fallback (`model.fallback_provider`).
- **LM Studio, vLLM, LiteLLM, Together, …** — anything that serves the OpenAI-compatible chat API works: use `provider: openai` with a `base_url` pointing at it. `oara setup` probes the standard ports for all four local servers (llama.cpp :8080, Ollama :11434, LM Studio :1234, vLLM :8000) and writes the config for whichever it finds.

### Multi-machine setups

The daemon and the GPU don't have to share a box. Point `base_url` at the inference machine — over your LAN, Tailscale, WireGuard, whatever:

```yaml
model:
  provider: "llama_cpp"
  base_url: "http://gpu-machine:8080"    # or a Tailscale IP / MagicDNS name
  fallback:
    - provider: "ollama"
      base_url: "http://gpu-machine:11434"
    - provider: "anthropic"
      api_key_env: "ANTHROPIC_API_KEY"
      model: "claude-haiku-4-5-20251001"
```

## Adapter strictness

The Model Adapter Layer validates every tool call before execution and repairs what it can (fuzzy tool-name matching, JSON extraction from markdown fences, type coercion). How aggressively it intervenes is the **strictness tier**:

| Tier | For | Behavior |
|------|-----|----------|
| `STRICT` | small local models (7–14B) | full validation, aggressive repair, more retries |
| `MEDIUM` | capable local models (Qwen, Gemma) | validation + repair, standard retries |
| `NONE` | cloud APIs | passthrough — these providers already handle tool calling well |

You normally don't set this by hand: cloud providers get `NONE` automatically, local models get a sensible default. With `adapter.adaptive_strictness: true` (off by default), the adapter tunes per-tool strictness from its own telemetry — if a tool's success rate over the last `strictness_window` calls (default 100) drops below `strictness_threshold` (default 0.8), that tool gets stricter handling.

## Cloud providers

Every cloud provider except Anthropic rides the same OpenAI-compatible wire format; Anthropic has a native provider. Each service reads its key from one environment variable (in `~/.config/prometheus/env` — never in the yaml):

| Provider | Config name | Env var | Default model |
|----------|-------------|---------|---------------|
| OpenAI | `openai` | `OPENAI_API_KEY` | `gpt-4o` |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` | `claude-haiku-4-5-20251001` |
| Google Gemini | `gemini` | `GEMINI_API_KEY` | `gemini-2.5-flash` |
| xAI (Grok) | `xai` | `XAI_API_KEY` *or* SuperGrok OAuth | `grok-4.5` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` | `deepseek-v4-flash` |
| Kimi (Moonshot) | `kimi` | `MOONSHOT_API_KEY` | `kimi-k2.6` |
| GLM (Z.ai / Zhipu) | `glm` | `ZAI_API_KEY` | `glm-5.2` |
| MiMo (Xiaomi) | `mimo` | `MIMO_API_KEY` | `mimo-v2.5-pro` |
| Qwen (Alibaba Model Studio) | `qwen` | `QWEN_API_KEY` | `qwen3.8-max` |

The defaults live in `src/prometheus/providers/registry.py` (`CLOUD_DEFAULTS`) and carry some hard-won footnotes:

- **xAI** — `grok-4.5` is requested *explicitly* because the `grok-3` / `grok-4` / `grok-latest` aliases currently serve grok-4.3.
- **DeepSeek** — the legacy `deepseek-chat` / `deepseek-reasoner` aliases are deprecated as of 2026-07-24; the defaults use V4 names. The reasoning flagship is `deepseek-v4-pro` if you want to pin it.
- **Kimi** — the default base URL is the international endpoint (`api.moonshot.ai`). A separate CN endpoint (`api.moonshot.cn`) exists with *separate keys*; set `base_url` in config if your key is CN-issued.
- **Qwen** — the default is Alibaba's international **pay-as-you-go** endpoint (`dashscope-intl.aliyuncs.com/compatible-mode/v1`), which takes an ordinary `sk-` Model Studio key. Alibaba also sells subscription plans (Token Plan, Coding Plan) on *different hosts with different keys* — per their docs the key/base-URL pairs "are completely isolated and must be used in matching pairs", and plan keys are limited to **interactive use in programming tools and agents**, explicitly not automated scripts, application backends, or other non-interactive callers. Point `slash_commands.qwen.base_url` at a plan endpoint only if your usage fits that scope; unattended paths (cron, briefings, evals, autonomous loops) do not. `QWEN_API_KEY` is deliberately separate from `DASHSCOPE_API_KEY`, which is the WAN image backend on another host.

Any of these can also be your **primary** model — set `model.provider` to the config name and you get the full harness (memory, wiki, security, profiles) on a cloud backend, no GPU required.

## Per-chat overrides: the slash commands

You don't have to change your primary model to borrow a cloud one. In any chat (Telegram, Slack, Discord, Beacon, CLI):

| Command | Routes to |
|---------|-----------|
| `/claude` | Anthropic |
| `/gpt` | OpenAI |
| `/gemini` | Google Gemini |
| `/xai` (alias `/grok`) | xAI |
| `/deepseek` `/kimi` `/glm` `/mimo` `/qwen` | DeepSeek / Moonshot / Z.ai / Xiaomi / Alibaba |
| `/local` | clears the override — back to your primary model |
| `/route` | shows which provider + model this chat is currently using |

Overrides are **session-scoped** (one Telegram chat, one Slack channel, one Beacon session) and **sticky** — they persist until you `/local`. Other chats, system paths (evals, benchmarks, smoke tests), and SENTINEL are unaffected. Set `router.overrides.sticky: false` if you'd rather each override apply to a single turn:

```yaml
router:
  overrides:
    enabled: true          # per-chat commands are active (this is the default)
    sticky: true           # persist until /local (false = one-shot)
```

A command only works once its provider's key is in `~/.config/prometheus/env` (or pasted into Beacon's Models tab) — otherwise it tells you which variable is missing.

### Configuring which model each command uses

Each slash command maps to a configurable provider + key env + model in `prometheus.yaml`:

```yaml
slash_commands:
  claude:
    provider: anthropic
    api_key_env: ANTHROPIC_API_KEY
    model: claude-sonnet-4-5    # alias OR dated snapshot — Anthropic resolves either
  gpt:
    provider: openai
    api_key_env: OPENAI_API_KEY
    model: gpt-4o
  gemini:
    provider: gemini
    api_key_env: GEMINI_API_KEY
    model: gemini-2.5-pro
  xai:
    provider: xai
    api_key_env: XAI_API_KEY
    model: grok-4.5             # request explicitly — grok-3/grok-4 aliases serve grok-4.3
  deepseek:
    provider: deepseek
    api_key_env: DEEPSEEK_API_KEY
    model: deepseek-v4-flash    # reasoning flagship: deepseek-v4-pro
  kimi:
    provider: kimi
    api_key_env: MOONSHOT_API_KEY
    model: kimi-k2.6
  glm:
    provider: glm
    api_key_env: ZAI_API_KEY
    model: glm-5.2
  mimo:
    provider: mimo
    api_key_env: MIMO_API_KEY
    model: mimo-v2.5-pro
  qwen:
    provider: qwen
    api_key_env: QWEN_API_KEY
    model: qwen3.8-max
    # base_url is optional — omit it for Alibaba's international
    # pay-as-you-go endpoint. Set it to reach a different host, e.g. a
    # subscription plan (read the scope note above first) or a region:
    #   base_url: https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
    #   base_url: https://coding-intl.dashscope.aliyuncs.com/v1
    #   base_url: https://dashscope-us.aliyuncs.com/compatible-mode/v1
```

`base_url` is honored on any of these blocks, not just Qwen — useful for the CN-vs-international endpoint splits noted above.

### Endpoint overrides: `<PROVIDER>_BASE_URL`

Every cloud provider also reads an endpoint variable from the env file: `OPENAI_BASE_URL`, `GEMINI_BASE_URL`, `XAI_BASE_URL`, `ANTHROPIC_BASE_URL`, `DEEPSEEK_BASE_URL`, `MOONSHOT_BASE_URL`, `ZAI_BASE_URL`, `MIMO_BASE_URL`, `QWEN_BASE_URL`.

Resolution order is `base_url` in config → `<PROVIDER>_BASE_URL` → the built-in default. Unset simply falls through, so these cost nothing until you use one.

Prefer the env var over yaml when **the hostname identifies your account**. The motivating case: an Alibaba key issued against a workspace **403s on the shared regional host** and must use that workspace's own endpoint, which embeds the workspace id:

```bash
# ~/.config/prometheus/env  (0600)
QWEN_API_KEY=sk-...
QWEN_BASE_URL=https://ws-<workspace-id>.<region>.maas.aliyuncs.com/compatible-mode/v1
```

A 403 on a key you know is valid almost always means this. `oara doctor` reports whether `QWEN_BASE_URL` is set, so the endpoint override is visible when you are diagnosing one.

⚠ The repo's pre-commit hook does **not** recognise these hostnames — its host check matches `OAra-*` only. Keeping a workspace endpoint in the env file (gitignored, 0600) rather than yaml is what keeps it out of a commit; nothing will stop you if you paste it into tracked code.

**Anthropic-compatible third-party endpoints**: `ANTHROPIC_BASE_URL` now reaches the provider (the factory previously dropped it). Two caveats — the provider appends `/messages` unless the value already ends in it, so pass `.../apps/anthropic/v1` if the shim expects `/v1/messages`; and headers stay Anthropic's (`x-api-key`, `anthropic-version`, plus the prompt-caching beta when enabled). A shim that rejects those needs provider-side work, not just this passthrough.

### Switching models within a provider

`model:` sets the default a bare `/qwen` gives you. `models:` lists what else you can switch to:

```yaml
slash_commands:
  qwen:
    model: qwen3.8-max                  # the default — bare /qwen
    models:                             # everything selectable
      - qwen3.7-max
      - qwen3.7-plus
      - qwen3.6-flash
```

Then, in any chat:

```
/qwen                  → the default (qwen3.8-max)
/qwen qwen3.7-plus     → switch to that model
/qwen qwen3.6-flash summarize this  → switch, then ask, in one message
```

A leading token is read as a model **only** when it exactly matches one of the listed models, so `/qwen explain this` still just asks a question on the default. An unlisted model is rejected with the list rather than passed to the provider as a 404.

The same list flattens into `GET /api/models`, so Beacon's model picker shows one entry per model with no Beacon change. The default keeps the bare preset key (`qwen`); alternates get `qwen:<model>`.

Built-in lists ship for Claude, Gemini, DeepSeek, and Qwen, and are deliberately short — only names verified against provider docs, because a wrong name is a runtime 404 that looks like a Prometheus bug. **Your `models:` list replaces the built-in one entirely**, so a provider shipping something new is a config edit and a restart, never a release. Providers with no list offer just their default.

xAI is intentionally single-entry: `grok-3` / `grok-4` / `grok-4-latest` are all silently served as grok-4.3, so offering them would be offering models you do not actually get.

Discovery-by-API was considered and rejected: xAI returns 403 on `GET /v1/models` for standard SuperGrok subscribers even though chat completions work, so enumeration cannot be relied on across providers. The list is configured, not enumerated.

Restart the daemon and grep the journal to verify the wiring:

```bash
systemctl --user restart prometheus.service
journalctl --user -u prometheus.service | grep slash_commands
# INFO  slash_commands.claude  → anthropic / claude-sonnet-4-5
# INFO  slash_commands.gpt     → openai / gpt-4o
# ...
```

Omit a command (or the whole section) and it falls back to conservative built-in defaults (cheap/fast, listed in `src/prometheus/router/model_router.py` under `OVERRIDE_PRESETS`) — the daemon logs a one-time WARN noting the fallback, so you're never guessing which model answered.

## xAI SuperGrok OAuth — sign in with a subscription

If you have a SuperGrok subscription, you don't need an xAI API key at all. Prometheus supports xAI's OAuth device-code flow:

**From Beacon** — open the **Models** tab and click **Sign in with SuperGrok**. Beacon shows a user code and the approval URL; open [accounts.x.ai](https://accounts.x.ai) (Beacon links you straight there), enter the code, approve, and Beacon's poll flips to signed-in. The tab shows the token's expiry and offers **Sign out**.

**From the CLI** — same flow, no Beacon needed:

```bash
python scripts/xai_oauth_login.py
```

Tokens land in `~/.prometheus/xai_oauth.json` and auto-refresh before expiry — sign in once and forget it.

When both a subscription and an `XAI_API_KEY` exist, the provider **prefers the subscription** and falls back to the key if the OAuth token is missing or unrefreshable. It also reports which auth mode it's actually using, so `/route` and the Models tab never leave you guessing whether a request billed your key or rode the subscription.

For scripting, the daemon exposes the same flow over REST:

- `GET /api/providers/xai/oauth` — status: `logged_in`, `expires_at`, plus `pending` / `user_code` while a login is in flight
- `POST /api/providers/xai/oauth/login` — starts a device login; returns the `verification_uri` + `user_code` immediately and polls to completion in the background
- `DELETE /api/providers/xai/oauth` — sign out

## Key management

![Beacon's Models tab — per-provider keys, auth-mode badges, and SuperGrok subscription sign-in](../assets/shots/extra-models-tab.png)

Three ways to manage provider keys, pick whichever fits:

- **Beacon's Models tab** — one row per service, paste a key, save. Saved keys are **never displayed again** (only a set/not-set state), take effect **immediately with no daemon restart**, and can be removed from the same row. Each service has a **"Get a key ↗"** link straight to the right console page (console.anthropic.com, console.x.ai, DashScope, the Kling dev console, …), so you're never hunting for where a key comes from.
- **Hand-edit the env file** — all keys live in `~/.config/prometheus/env` as plain `NAME=value` lines; both `oara daemon` and the systemd unit load it. Editing by hand requires a daemon restart.
- **Check without exposing** — `oara doctor` prints a set/not-set line for every known key variable without ever echoing values.

Secrets never go in the yaml, and a pre-commit hook blocks them from ever landing in the repo.

## Model Router

The router (`model_router`) is the bigger, autonomous sibling of the slash commands: task-type classification (route coding tasks one way, chat another), fallback chains when a provider is down, and opt-in **cloud escalation** — when a tool call keeps failing validation past the adapter's retry budget, a subagent on a stronger provider attempts that one call and feeds the result back, while the main agent keeps running locally:

```yaml
model_router:
  enabled: false          # the default — see below

# router:
#   escalation:
#     enabled: false
#     provider:
#       provider: anthropic
#       api_key_env: ANTHROPIC_API_KEY
#       model: claude-sonnet-4-6
#     as_subagent: true
#     budget_usd: 1.00     # soft cost cap per escalation (future enforcement)
```

**Defaults, honestly:** the router itself ships **off** — most single-model setups don't need it. The per-chat overrides (`router.overrides`) are a separate, lightweight mechanism and ship **on**.

## Force-search / `tool_choice`

Sometimes you don't want the model to *decide* whether to search — you want it to search. Two surfaces for the same mechanism:

- **Beacon** — the 🔍 toggle next to the composer forces the next turn to start with a web search.
- **API** — `POST /api/chat/send` accepts a `tool_choice` field: `auto` (default), `none`, `required` (must start with *some* tool call), or `{"tool": "web_search"}` (must start with *that* tool).

The directive is threaded all the way down the stack — API → agent loop → adapter → provider — and on llama.cpp it lands as **GBNF grammar selection**, so the forced tool call isn't a suggestion in the prompt, it's the only thing the model can emit. One caveat from the trenches: this is a per-turn directive, and it works because it's enforced at decode time, not because the model was asked nicely.

## Image generation

The `image_generate` tool has three backends:

| Backend | Cost | Needs |
|---------|------|-------|
| `pollinations` | free | nothing — hosted endpoint |
| `comfyui` | free (local GPU) | a local ComfyUI server with a FLUX checkpoint |
| `dashscope` | **paid** | `DASHSCOPE_API_KEY` (Alibaba WAN 2.5, async task API) |

The default `auto` probes local ComfyUI and falls back to Pollinations. **`auto` never selects the paid backend** — DashScope runs only on an explicit `backend=dashscope` argument or a deliberate `image_generation.default_backend: dashscope` in `prometheus.yaml`:

```yaml
image_generation:
  default_backend: auto              # auto | pollinations | comfyui | dashscope
  comfyui:
    base_url: "http://127.0.0.1:8188"
    default_model: "flux1-schnell-fp8.safetensors"
  dashscope:
    api_key_env: DASHSCOPE_API_KEY
    model: "wan2.5-t2i-preview"
```

## Video generation

The `video_generate` tool drives the Kling 3.0 API (text-to-video, or image-to-video via `image_path`; 5s/10s durations). Paid and **dormant until keyed**: it needs `KLING_ACCESS_KEY` + `KLING_SECRET_KEY` from the [Kling console](https://app.klingai.com), and self-signs a short-lived HS256 JWT per request (stdlib only — no extra dependencies, and the keys never leave your machine). Renders take minutes; the tool polls (budget: `video_generation.kling.poll_budget_seconds`, default 600) and saves the finished `.mp4` to `~/.prometheus/cache/videos/`.

```yaml
video_generation:
  kling:
    access_key_env: KLING_ACCESS_KEY
    secret_key_env: KLING_SECRET_KEY
    base_url: "https://api-singapore.klingai.com"
    model_name: "kling-v3"
    poll_budget_seconds: 600
```

## Credential pools

For providers where you hold multiple keys, Prometheus can rotate them: a `CredentialPool` round-robins across the keys, marks a key dead on auth/rate-limit errors (a 429 rotates immediately), and revives dead keys after a cooldown (default 300 seconds). Key health is tracked per key; log lines show only the first/last few characters, never the full key.

## Honesty notes

- **Model IDs are configurable defaults, not guarantees.** The names in `CLOUD_DEFAULTS` and the `slash_commands` section were verified against the live APIs at the dates noted in the source comments, but providers rename, deprecate, and alias-shuffle constantly (see the DeepSeek deprecation and the Grok alias situation above). If a command starts erroring with an unknown-model message, the fix is one line of yaml.
- **Paid backends never bill without keys.** DashScope and Kling are inert until you set their env vars, `auto` image selection never reaches for the paid path, and every cloud slash command fails loudly rather than silently substituting a provider you didn't ask for.
- **Local stays the default.** Nothing in the cloud plumbing changes the primary model — every cloud path here is opt-in, per-chat, and reversible with `/local`.
