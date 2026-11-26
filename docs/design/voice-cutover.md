# Cutover: retiring the Mini's legacy oara-voice lab stack

**Plan only — nothing here is executed by the PR.** This is the sequence for retiring the legacy voice stack on
the Mini (`oara-mini`, `<mini-tailnet-ip>`) once OAra Voice runs on the 4090 tower and Prometheus reaches it through
the voice-settings API. Beacon currently connects to the legacy stack at `<mini-tailnet-ip>:8015`; that connection
moves to Prometheus, never to the new box directly.

The agreed end state (docs/design/speech-route.md, docs/design/voice-settings.md): **every client talks to
Prometheus only.** The voice box (wherever it runs) is behind Prometheus. No client holds the box's address or key.

## 0. What is on the Mini today (the thing being retired)

From the lab's SYSTEM_STATE / SYSTEM_AUDIT (oara-voice repo):

| unit / service | port | what | fate |
|---|---|---|---|
| `oara-middleware.service` | 8015 | archive + memory store + bridge adapter + integration contract | **the voice parts retire**; archive/memory is a separate decision (below) |
| `oara-tts.service` (Chatterbox Turbo) | 8003 | the mouth (3090 eGPU) | retire — Kokoro on the tower replaces it |
| whisper-server | 8081 | the ears (3090, large-v3-turbo) | retire — whisper on the tower replaces it |
| `oara-vad.service` | 8011 | hands-free VAD | retire — VAD on the tower (8012) replaces it |
| `oara-memory-extractor.service` | — | 30-min extraction (ollama) | **out of scope** — not voice; leave it |
| `prometheus.service` | 8005 / 8010 | the daemon | **stays** — it is the door every client uses |

**Beacon's current path:** Beacon → `<mini-tailnet-ip>:8015` (the middleware's `/bridge/*`, `/integration/*`).
**Beacon's end path:** Beacon → Prometheus (`<mini-tailnet-ip>:8005` REST / `:8010` WS) → the tower box. Beacon never
learns the tower's address.

## 1. Preconditions (all must be true before any retirement)

- [ ] The 4090 tower box is deployed and healthy (deploy/4090 in the plugin repo): `/health`, `/v1/models`,
      `/v1/voices`, `/v1/settings` answer over the tailnet with the bearer token.
- [ ] Prometheus's `stt:` / `tts:` point at the tower (`<tower-tailnet-ip>:8015`), the key is in the env file, and
      `GET /api/speech` reports both halves `ok`.
- [ ] `GET /api/speech/settings` (operator) discovers the tower's options live; a `PUT` of voice/speed applies
      without a restart and is heard on a spoken reply.
- [ ] A full spoken turn works end-to-end **through Prometheus**: a Beacon (or test client) sends a voice message
      to Prometheus, gets a transcribed turn and a spoken reply, with the tower doing STT/TTS.
- [ ] The client sessions (beacon-desktop, beacon-ios) have moved their voice path to Prometheus's speech route
      and voice-settings API — **not** to the tower. This is the load-bearing precondition: retiring the Mini
      stack before the clients move would break their voice.

## 2. Move the clients off the Mini stack (before retiring anything)

The client sessions own this; the contract they build against is posted to
`~/projects/.scratch/voice-settings-contract.md`.

- [ ] Beacon desktop: voice turns and dictation go through Prometheus (`POST /api/chat/send` with a transcribed
      voice message, `POST /api/speech/transcribe`, `GET /api/sessions/{id}/speech`, `POST /api/speech/speak`).
      The `oara` integration's bridge URL (`<mini-tailnet-ip>:8015`) is removed from the client config.
- [ ] Beacon iOS: same — the speech route under the device's own key, hands-free VAD on the device (the speech
      route keeps endpointing on the device that has the microphone).
- [ ] Settings UI (owner device): `GET`/`PUT /api/speech/settings`. A chat-only device never sees it (403).
- [ ] Verify on both clients: a spoken turn, dictation, and a voice/speed change all work against Prometheus with
      the Mini voice stack still running (so a rollback is trivial).

## 3. Cut over Prometheus's own voice surfaces

Prometheus's non-speech-route voice surfaces (the `tts` tool, Telegram voice notes, `voice_remote`) already read
`stt.url` / `tts.url`, so they follow the config to the tower automatically. Confirm:

- [ ] A Telegram voice note transcribes via the tower (or falls back to local with the named warning if the tower
      is down — the existing `voice_remote` behavior, unchanged).
- [ ] The `tts` tool speaks via the tower.
- [ ] `oara doctor` reports the tower box healthy.

## 4. Retire the Mini voice units (only after 1–3 are green)

Do these one at a time, verifying voice still works through Prometheus after each. **Stop, do not delete, first**
— a stopped unit is a trivial rollback; deletion is the last step after a soak period.

- [ ] Stop `oara-tts.service` (Chatterbox :8003). Verify a spoken reply still works (the tower's Kokoro answers).
- [ ] Stop whisper-server (:8081). Verify dictation still works (the tower's whisper answers).
- [ ] Stop `oara-vad.service` (:8011). Verify hands-free still works (device-side VAD + the tower's :8012).
- [ ] Stop the **voice parts** of `oara-middleware.service` (:8015). This is the delicate one — the middleware
      also serves archive/memory and the integration contract. Two options, decided with the owner:
      - **If the middleware is voice-only in practice:** stop the whole unit.
      - **If archive/memory still rides on it:** keep the unit but disable its bridge/voice routes, OR move
        archive/memory to their own service first. **Do not stop archive/memory as a side effect of retiring
        voice** — the memory boundary is inviolable (SYSTEM_STATE §1).
- [ ] Confirm Beacon no longer references `<mini-tailnet-ip>:8015` anywhere (grep the client configs).

## 5. Soak, then remove

- [ ] Run for a agreed soak period (suggest 7 days) with the Mini voice units stopped but present. Watch
      `oara_heartbeat_watcher` and the speech-route health; any voice regression is a rollback (restart the unit).
- [ ] After the soak: disable the units (`systemctl --user disable`), then remove their unit files and the
      Chatterbox venv (`~/oara-tts`). The 3090 eGPU is freed for other lab work.
- [ ] Update the lab's SYSTEM_STATE / SYSTEM_AUDIT to reflect the new topology (voice on the tower, behind
      Prometheus). Those docs live in the oara-voice repo — a separate PR there, not this one.

## 6. Rollback (at any step before removal)

Restart the stopped Mini unit(s), revert Prometheus's `stt:`/`tts:` to the Mini box (or local), restart the
daemon. Because the clients were moved to Prometheus (not to the tower) in step 2, **no client change is needed
to roll back** — Prometheus points back at the Mini and every client keeps working. This is the whole reason the
clients move first.

## 7. Out of scope (deliberately not touched here)

- The memory extractor, archive store, and the inviolable memory boundary.
- The 4090's llama-server (:8080) — the model backend, unrelated to voice.
- The Mini's Prometheus daemon itself — it stays; it is the door.
- Deploying the tower box (that is deploy/4090's runbook, executed separately).