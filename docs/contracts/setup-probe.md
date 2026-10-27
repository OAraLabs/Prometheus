# Setup's look and probe: what AI app is there, and can Prometheus think with it

Setup mode can tell a client which local AI apps answer, what each one serves, and why nothing is usable when nothing
is. A client (Beacon's first run) uses it to offer "an AI you already run". One module decides all of it,
`src/prometheus/providers/local_probe.py`, and the three routes below share it.

No LAN scan. A **look** is the known places on the computer Prometheus runs on. A **probe** is the one computer
someone typed: its port, or the known places on it. Every place is asked in parallel.

## The known places

| App | Port | Identified by |
|---|---|---|
| llama.cpp | 8080 | `GET /props` answers `default_generation_settings` |
| Ollama | 11434 | `GET /api/version` answers `{"version"}` and `GET /api/tags` answers `{"models": [...]}` |
| LM Studio | 1234 | `GET /api/v0/models` answers `{"data": [...]}` with per-model `type` |
| vLLM | 8000 | `GET /version` answers `{"version"}` and `GET /v1/models` answers `{"data": [...]}` |
| any other | (any) | `GET /v1/models` answers `{"data": [...]}`: `openai_compatible` |

The app is identified by what it answers, never by its port. An Ollama also answers `/v1/models`; the old detector
labelled every typed address `llama_cpp`.

## What a probe may do

- GET to the fixed paths above, and Ollama's `POST /api/show {"model"}`. Nothing else.
- No redirect is followed.
- At most 256 KiB of one answer is read; a longer one is not read as an app.
- 1 s per request on this machine, 3 s elsewhere, and a deadline per place.
- Only parsed fields come back, never an answer's body. `detail` is Prometheus's own sentence.

So the route cannot be used to read pages it was not built for.

## Per model

| Field | Ollama | LM Studio | llama.cpp | vLLM | openai_compatible |
|---|---|---|---|---|---|
| `loaded` | in `/api/ps` | `state == "loaded"` | true | true | null |
| `chat` | `capabilities` has `completion` | `type` is `llm` or `vlm` | null | null | null |
| `window` | trained length (`/api/show`), `trained` | `loaded_context_length` when loaded, `loaded` | `/props` `n_ctx`, `served` | `max_model_len`, `served` | null |

A model is **usable** when `chat` is not false, it is loaded where the app needs loading (LM Studio), and `window` is
null or at least `min_window`. Otherwise `problem` says which: `no_chat`, `none_loaded` or `short_context`.

Ollama is budgeted at its trained length because Prometheus asks it to load at the window it budgets (`num_ctx`,
#815), capped by that length.

## The answer

```json
{
  "target": "studio-mac.local",
  "found": [
    {"app": "lm_studio", "provider": "lm_studio", "base_url": "http://studio-mac.local:1234", "version": null,
     "latency_ms": 12.0,
     "models": [{"id": "gemma-3-12b", "loaded": true, "chat": true, "window": 4096, "window_source": "loaded",
                 "usable": false, "problem": "short_context"}]}
  ],
  "reason": "short_context",
  "detail": "LM Studio at http://studio-mac.local:1234: 0 of 1 usable (gemma-3-12b with a window under 20,480; Prometheus needs 20,480)",
  "min_window": 20480
}
```

`reason` is null when at least one model is usable. Otherwise it is one of:

| `reason` | When |
|---|---|
| `bad_target` | The target is not a computer (probe only). |
| `name_not_found` | The name did not resolve within 3 s. |
| `needs_key` | A place answered 401 or 403. |
| `not_ai` | A place answered, but not like any app above. |
| `no_answer` | No place answered in time. |
| `refused` | Every place refused the connection (nothing is listening). |
| `short_context` | An app answered, and its chat models' windows are under `min_window`. |
| `none_loaded` | An app that needs loading has nothing loaded. |
| `no_chat` | An app serves only models that cannot chat (embeddings). |
| `no_models` | An app answered with no models at all. |

When no app answered, the first of `needs_key`, `not_ai`, `no_answer` and `refused` that any place gave is reported.
When an app answered, its models decide, in the order `short_context`, `none_loaded`, `no_chat`, `no_models`.

A client must read a `reason` it does not know as "could not check", never as found.

## The routes (setup mode; the setup credential)

- `GET /api/setup/detect` keeps `backends` (the old detector's answer, unchanged) and adds `local`: the look's
  answer, `target` null. With `?base_url=`, `local` is the probe of that address.
- `POST /api/setup/probe {"target"}` answers the probe.
  - A target is a name (`studio-mac`, `studio-mac.local`), an IPv4 address, a bracketed IPv6 address, any of them
    with `:port`, or an `http://` / `https://` URL with no path beyond `/`.
  - A bare name tries the known places on that computer; a URL with no port means its scheme's port.
  - Anything else is `400 bad_target`. A missing target is `400 bad_request`.
- `POST /api/setup/configure` with a local provider re-runs the probe for its `base_url`:
  - `400 model_not_served` when the app answers and does not serve the model;
  - `400 backend_unreachable` with `reason` when nothing usable answers;
  - otherwise it writes the config, with `context.effective_limit` set to the measured window when that is smaller
    than the default. LM Studio and vLLM report no window at boot.
  - Providers: `ollama` needs an Ollama. `llama_cpp`, `lm_studio`, `vllm` and `openai_compatible` are the same
    OpenAI-shaped client, and any app above answers it.
- `GET /api/setup/status` lists `"local_probe"` in `features`.

## `min_window`

`local_probe.MIN_WINDOW` is round 0 of a fresh install's first turn, plus the local reply cap (4,096), plus 4,096 of
conversation, rounded up to the next multiple of 4,096, plus 4,096 of headroom (`min_window_for`). It is set only from
`MIN_WINDOW_MEASURED`, which records the commit, the method and round 0 per server (`scripts/local_gate/measure_floor.py`).
Measured 2026-10-27 on this branch over main `1e36960`, through a scratch instance: round 0 is 8,178 tokens on llama.cpp (a 27B, 51 tools offered) and 6,417 on Ollama (`qwen2.5:7b-instruct`, 51 tools), so the need is 16,370 and the floor is **20,480**. That is 14 tokens under the step: a round 0 that grows past 8,192 raises the floor to 24,576, so measure again after a change that adds to the first prompt (a tool, a longer system prompt).
