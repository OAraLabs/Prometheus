# Voice settings: one API for the voice box, through Prometheus only

Design, 2026-11-26. The agreed stack (docs/design/speech-route.md): apps talk only to Prometheus; the voice box
(the OAra Voice plugin) does speech-to-text and text-to-speech behind it. This note adds the settings surface:
an operator reads the voice configuration and the options the box actually offers, and changes it, without any
client ever holding the box's address or key. It says what the routes answer, what a change is, what is applied
live and what waits for a restart, and what the box must expose for any of it to work.

## 0. In brief

- Two routes: `GET /api/speech/settings` (current config + live-discovered options + health per component) and
  `PUT /api/speech/settings` (a partial change). Both are the OPERATOR's (the master token or an owner device);
  a scoped device gets 403 `operator_only` on reading as well as on changing — the same class as `/api/model`.
- Options are discovered LIVE from the box: `GET /v1/models` (what it runs) plus `GET /v1/voices` and
  `GET /v1/settings` (the reads the plugin adds for this API). A box that predates those reads is still usable:
  the answer says `discovery: "unavailable"` and offers only what the config file already names.
- A change is saved to `prometheus.yaml` with the comment-preserving editor `/api/model` uses, verified before
  writing, and applied LIVE where possible: the speech route reads its config on every call, and voice/speed/
  engine/VAD/gate knobs are pushed to the box (`PUT /v1/settings`) in the same request. Only the box's ADDRESS
  waits for a restart, and the answer says so (`applied: "on_restart"`).
- Every accepted change and every refused attempt writes a security-audit row (`permissions/audit.py`), naming
  the caller and the fields — never a key.
- The box's key never appears in an answer, the yaml or a log line; the box's address appears only in this
  operator-only answer (a scoped device can never read it).

## 1. What exists, and what this adds

Prometheus already reaches a voice box over the OpenAI audio API (`infra/voice_remote.py`): `stt.url` / `tts.url`
in prometheus.yaml, the key from the env file under `api_key_env`, `GET /health` + `GET /v1/models` for checks.
The speech route (`web/speech_routes.py`, `speech/voice.py`) reads that config on every call through
`SpeechService.config`, so a config change is live for every new phrase without a restart.

What is missing is the settings surface: no route reports the voice configuration, no route changes it, and the
box offers no way to list its voices or change voice/speed/engine at runtime (the Kokoro server's voice is fixed
by `KOKORO_VOICE_NAME` at boot; `/v1/audio/speech` accepts `voice` for shape only). This note adds both sides.

## 2. Routes

Both under `/api/`, behind the bearer middleware, in `route_access.OPERATOR_ONLY` (classified on purpose, and
pinned in `tests/test_route_access.py` like every other route).

### 2.1 `GET /api/speech/settings`

```json
{
  "version": 1,
  "stt": {
    "configured": true, "model": "large-v3-turbo", "tier": "gpu",
    "health": {"state": "ok"},
    "key": {"env": "PROMETHEUS_STT_API_KEY", "present": true}
  },
  "tts": {
    "configured": true, "engine": "kokoro", "model": "kokoro-v1.0",
    "voice": "af_heart", "speed": 1.0,
    "health": {"state": "ok"},
    "key": {"env": "PROMETHEUS_TTS_API_KEY", "present": true}
  },
  "vad": {"sensitivity": 0.5, "health": {"state": "ok"}},
  "silence_gate": {"threshold_rms": 32.0},
  "box": {"address": "http://<tower-tailnet-ip>:8015", "discovery": "live"},
  "options": {
    "stt_models": [
      {"id": "base", "tier": "cpu"}, {"id": "small", "tier": "cpu"},
      {"id": "large-v3-turbo", "tier": "gpu", "active": true}
    ],
    "tts_engines": [
      {"id": "kokoro", "active": true}, {"id": "chatterbox", "available": false},
      {"id": "system", "available": false}
    ],
    "voices": ["af_heart", "af_bella", "am_michael", "..."],
    "speed": {"min": 0.5, "max": 2.0},
    "vad_sensitivity": {"min": 0.1, "max": 0.9},
    "silence_gate_rms": {"min": 0.0, "max": 1000.0}
  },
  "pending": false,
  "applied": "live",
  "warnings": []
}
```

- `health` per component is the speech route's own record (`Voice.status()`: `ok` / `down` / `unknown`, with
  `reason` when down — `unreachable` or `refused`). VAD health comes from the box's `GET /v1/settings` answer
  (the box probes its own VAD server); when discovery is unavailable it is `unknown`.
- `options` is what the box says it has, not a hand-written list: `stt_models` from `/v1/models` (the plugin
  lists every model its build can run, with `tier` and `active`), `voices` from `/v1/voices` (every voice in the
  loaded Kokoro voice pack — `af_heart` and the rest), engines and knob ranges from `/v1/settings`. An engine the
  box cannot run right now is listed with `available: false` and a `reason`, never hidden.
- `tier` is `cpu` or `gpu`: which compose profile the active STT model belongs to. Changing the model to a
  different tier is a deployment change (section 4), not a live one.
- `box.address` is the configured URL, shown to the operator only. `discovery` is `live` when the box answered
  the settings reads, `unavailable` when it did not (an older box, or one that is down); the answer then carries
  `options: null` and a warning, and PUT still accepts the fields the config file itself can validate.
- `key` reports the env variable NAME and whether a key is held — never the value, never in a warning either.
- When nothing is configured: `stt`/`tts` are `{"configured": false, "health": {"state": "unknown"}}`, `box` is
  null, `options` is null, and a warning says no voice box is set up.

### 2.2 `PUT /api/speech/settings`

A partial JSON object; every field optional, unknown fields refused (`400 invalid_request`). Nothing is written
until the whole request validates.

```json
{
  "stt_model": "large-v3-turbo",
  "tts_engine": "kokoro",
  "voice": "af_heart",
  "speed": 1.1,
  "vad_sensitivity": 0.5,
  "silence_gate_rms": 32.0,
  "box_address": "http://<tower-tailnet-ip>:8015"
}
```

| field | saves to | applied |
|---|---|---|
| `stt_model` | `stt.model` | live for the daemon's requests; the box switches its whisper server only when the model's tier is already running, else `on_restart` of the BOX (a warning says which) |
| `tts_engine` | `tts.provider` | pushed to the box (`PUT /v1/settings`); the box answers whether it switched live or needs its own restart |
| `voice` | `tts.voice` | live: pushed to the box, and sent on every speech call |
| `speed` | `tts.speed` | live: pushed to the box (0.5–2.0, the Kokoro graph's own range) |
| `vad_sensitivity` | `vad.sensitivity` | live: pushed to the box, which forwards it to the VAD server |
| `silence_gate_rms` | `stt.silence_gate_rms` | live: pushed to the box (0–1000 RMS of 32768 full scale) |
| `box_address` | `stt.url` and `tts.url` | **on_restart** of the daemon: the speech route's health record and the env-file key are bound to the configured box; the answer says so |

Rules:

- **Validate against the live options when discovery worked.** A voice the box does not have, a speed out of
  range, an engine the box cannot run: `400` with a stable code (`unknown_voice`, `bad_speed`,
  `unsupported_engine`, `unknown_model`) and nothing written. When discovery is unavailable, range/format checks
  still run; membership checks are skipped and the answer carries a warning that the box could not confirm.
- **The key is never a field.** `api_key`, `api_key_env` and any token-shaped field are refused
  (`400 invalid_request`); keys live in the env file only, as everywhere else.
- **The file is edited as text** (`web/network.py`'s editor), verified before writing: the document must read
  back as the old one plus exactly the intended keys, comments intact. A failed verification writes nothing.
- **The running config is updated in the same request** for every field except `box_address`, so the next
  phrase uses it; `box_address` is saved only, and `pending: true` / `applied: "on_restart"` reports it until
  the daemon restarts.
- **The push to the box is best-effort and announced.** If the box refuses or is unreachable, the daemon-side
  save still stands (the daemon sends `voice` on every call regardless), and the answer carries a warning
  naming what did not reach the box. A `box_address` change is never pushed anywhere — it is a restart change.
- **Audit.** Every PUT writes one `AuditLogger` row (`tool_name: "speech.settings"`): ALLOW with the changed
  field names and values for an accepted change, DENY with the refusal code for a rejected one, and the
  caller's device id (or `global-token`). Values are the settings themselves — no key can be among them,
  because no key is a field.

The answer is the GET shape (section 2.1) after the change, plus `changed: [field, ...]`.

### 2.3 Refusals

Every refusal is `{"error": <stable code>, "detail": <English>}` and writes nothing:
`403 operator_only` (a scoped device, on GET too), `400 invalid_request` (unknown field, wrong type, a key
field), `400 unknown_voice` / `unknown_model` / `unsupported_engine` / `bad_speed` / `bad_sensitivity` /
`bad_threshold` / `bad_address`, `409 no_config_file` (the daemon was not started from a file — nothing to save
into), `500 persist_failed` (the edit did not verify or the write failed; the file is as it was).

## 3. What the plugin exposes (the box side)

Additive reads and one write under `/v1`, bearer-authed like the audio routes; the legacy surface
(`/bridge/*`, `/integration/manifest`, `/health`, `/inference`, `/tts`) is untouched.

- `GET /v1/models` — extended: every STT model the build can run (`base`, `small`, `large-v3-turbo`) with
  `tier` (`cpu`/`gpu`) and `active`, plus the TTS models. Still OpenAI's list envelope; the daemon's existing
  probe reads only `data[].id`, so the extra fields cost it nothing.
- `GET /v1/voices` — `{"voices": [...], "active": "af_heart", "engine": "kokoro"}`: every voice in the loaded
  pack, read from the TTS server's new `GET /voices` (Kokoro's `get_voices()`), not a hand-written list.
- `GET /v1/settings` — the box's current runtime settings and what it can do:
  ```json
  {
    "stt": {"model": "large-v3-turbo", "tier": "gpu", "silence_gate_rms": 32.0},
    "tts": {"engine": "kokoro", "voice": "af_heart", "speed": 1.0,
            "engines": [{"id": "kokoro", "available": true},
                        {"id": "chatterbox", "available": false, "reason": "not installed"},
                        {"id": "system", "available": false, "reason": "macOS only"}]},
    "vad": {"sensitivity": 0.5, "health": "ok"},
    "ranges": {"speed": [0.5, 2.0], "vad_sensitivity": [0.1, 0.9], "silence_gate_rms": [0.0, 1000.0]}
  }
  ```
- `PUT /v1/settings` — the same shape, partial; sets voice/speed/engine, the silence gate and VAD sensitivity at
  runtime, answers what was applied and what needs a restart. An engine switch the box cannot do live (the
  other engine's server is not running) is refused with `409 engine_not_running` — never a silent no-op.
- The TTS servers grow `GET /voices` and accept `voice`/`speed` on `POST /tts` (additive; the old `{"text"}`
  body keeps working), and a runtime `PUT /settings` for the default voice/speed. The VAD server grows
  `GET /settings` / `PUT /settings` for its speech threshold, applied to new sessions at once.

`system` (macOS `say`) is listed as an engine the box knows but cannot run on a Linux tower
(`available: false, reason: "macOS only"`); it is a local-Mac option, not a tower one.

## 4. What a change is not

- **Not a tier switch.** `base`/`small` run in the CPU compose profile, `large-v3-turbo` in the GPU one. The
  settings API reports the tier and refuses a cross-tier model change with `409 tier_change` and a warning
  naming the compose profile to start — switching tiers is a deployment act (docker compose), not a PUT.
- **Not a second credential path.** The box keeps its own `OARA_API_TOKEN`; Prometheus keeps the key in the env
  file. Neither route reads or writes either token.
- **Not a client surface.** A scoped device never sees these routes; the client sessions build their settings
  UI against this API on an owner device, and against `GET /api/speech` (which stays scoped-allowed) for
  liveness. No client talks to the plugin directly.

## 5. Deployment and cutover

The 4090 tower deployment (compose, env, ports, the exact `stt:`/`tts:` yaml section) is prepared in the plugin
repo (`deploy/4090/`) with step-by-step instructions; nothing is deployed by this change. Retiring the Mini's
legacy oara-voice lab stack (Beacon's current `<mini-tailnet-ip>:8015`) is a separate cutover plan
(`docs/design/voice-cutover.md`), plan only.

## 6. Settled here, and why

- **D1 — settings live under `/api/speech/settings`, not `/api/voice`.** The speech route is the surface that
  already owns the box relationship; one prefix, one health record, one config source.
- **D2 — operator-only on GET too.** The answer carries the box's address; a chat-only device has no business
  knowing where the voice box lives. Same class as `/api/model`, same 403 code.
- **D3 — live apply where the architecture already allows it.** `SpeechService` reads config per call, so
  daemon-side changes are live by construction; only the address (bound into the health record and the key's
  meaning) waits for a restart, and the answer says so instead of pretending.
- **D4 — options discovered, never hardcoded.** A voice list written into Prometheus would drift from the
  pack the box actually loaded; the box reads its own loaded model, and Prometheus relays it.
- **D5 — a cross-tier model change is refused, not half-applied.** The whisper server runs one model per
  process; pretending a PUT switched CPU→GPU would leave the daemon naming a model the box is not running.