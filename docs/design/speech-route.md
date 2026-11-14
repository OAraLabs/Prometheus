# Speech route: a reply read aloud, phrase by phrase

Design, 2026-11-13. The agreed stack: apps (Beacon for iOS and the desktop) talk only to Prometheus. OAra Voice, the
voice box, does speech-to-text and text-to-speech and nothing else, behind Prometheus. Prometheus hands an app the
reply as spoken audio, one phrase at a time, so the phone starts speaking before the reply has finished. Detecting
when someone starts and stops talking stays on the device that has the microphone.

This note says what the speech route is, who may call it, how phrases are cut, how the voice box is configured and
checked, and what happens when it is down. Section 10 lists the small questions settled here, and why.

## 0. In brief

- Four routes: `GET /api/speech` (what works right now), `POST /api/speech/transcribe` (audio in, text out),
  `GET /api/sessions/{session_id}/speech` (a session's reply as a stream of spoken phrases) and
  `POST /api/speech/speak` (any text as the same stream).
- The owner and chat-only devices may call all four, with chat's limits: a chat-only device hears only its own
  sessions.
- `speech` appears in the `subscribed` ack's `server_capabilities` when a voice box is set up. An app that does not
  see it behaves as today.
- Phrases end at sentence ends, line breaks and tool calls, never inside a code block, and are never longer than 200
  characters. Code blocks are not read; the listener hears "Code block skipped."
- The voice box is the one Prometheus already uses (`stt.url` / `tts.url`, or one the owner approved by its address).
  Nothing new to configure.
- When it is down, the route says so and why, and says where in the text it stopped. The app reads the rest with its
  own voice and tells the person plainly. Prometheus never quietly swaps in a worse voice.

## 1. What exists, and what this adds

Prometheus already reaches a voice box over the OpenAI audio API (`infra/voice_remote.py`): `POST
/v1/audio/transcriptions` and `POST /v1/audio/speech`, a bearer key from the env file, and `GET /health`. A box joins
by `stt.url` / `tts.url` in `prometheus.yaml`, or by its address with one approval (`docs/contracts/pieces.md`). The
WebSocket `chat_upload` already transcribes a voice message into a chat turn (`voice_input`).

What is missing is the other direction: the reply, spoken, under the device's own key. The speech route adds it, plus a
plain transcription route so an app can use the voice box as its speech-to-text source without making a chat turn.

## 2. Routes

All four are under `/api/`, behind the bearer middleware. None is public.

### 2.1 `GET /api/speech`: what works right now

```json
{
  "version": 1,
  "stt": {"configured": true, "state": "ok"},
  "tts": {"configured": true, "state": "down", "reason": "unreachable"},
  "audio": {"mime": "audio/wav"},
  "limits": {"upload_bytes": 2097152, "phrase_chars": 200, "spoken_chars": 5000,
             "speak_chars": 20000, "streams": 4}
}
```

- `state` is `ok`, `down` or `unknown` (configured, not yet checked). `reason` is present when `down`: `unreachable`,
  `refused` (the box refused its key), `rejected` (the box rejected the request: a bad model or voice name).
- Answering it may check the box (section 6.2). It never says the box's address, its key, or the key's variable name.

### 2.2 `POST /api/speech/transcribe`: audio in, text out

- **Body:** the audio bytes as they are. `Content-Type` names the type: one in `gateway.media`'s audio allowlist
  (shipped: ogg, mpeg, wav, mp4 for m4a, webm).
- **Query:** `language` (optional; 2 to 8 characters of letters and `-`, passed to the box as is).
- **Checks:** the same as a voice message on any other surface (`gateway/media_guard.py`): the declared type against
  the allowlist, the size against `gateway.media.max_file_size_mb` and the 2 MB `/api` body cap, whichever is smaller,
  and the bytes' own signature against the declared type.
- **200:** `{"text": "..."}`. An empty transcript is `""`, not an error. A leading `/` is removed (section 10, D4).
- **Refusals:** 400 `bad_audio` (signature does not match, or empty), 413 `too_large`, 415 `type_not_allowed`, 400
  `bad_language`, 503 `speech_unavailable` (section 7).

Prometheus does not send the transcript anywhere. The app sends it as the person's message with `POST /api/chat/send`,
so a spoken turn lands in the chat the person is in, by the same path as a typed one.

### 2.3 `GET /api/sessions/{session_id}/speech`: a reply, spoken

The response is a stream (section 2.5). Which reply it speaks:

1. **`message_id` given** (the `message_id` of a `chat_delta` frame): that reply, if it is still running or ended less
   than 120 s ago. Otherwise the stream ends at once with `done` / `no_reply`.
2. **A reply running in the session:** that one, from its first word.
3. **Otherwise:** the next reply to start in the session, waiting up to `wait` seconds (query, default 30, 1 to 120).
   None starts: `done` / `no_reply`.

The race-free order for an app is: open the stream, then send the message. An app that opens it later passes the
`message_id` from the first `chat_delta` it saw; nothing of the reply is lost either way, because Prometheus holds a
running reply's text from its start.

The reply's text is exactly what the session's `chat_delta` frames carry. When the turn stops (`chat_done` with
`interrupted`), the stream stops speaking at once and ends with `done` / `interrupted`. When the turn fails (an `error`
frame for the session), it ends with `done` / `failed`. A discarded round (`chat_discard`) drops what was not yet
spoken past `keep_chars` and is passed on as a `discard` line.

Closing the connection is how an app stops the speech without stopping the turn (the person started talking over it):
Prometheus stops synthesising for that stream.

### 2.4 `POST /api/speech/speak`: any text, spoken

- **Body:** `{"text": "..."}`, 1 to 20,000 characters. Over: 413 `too_long`. Empty: 400 `text_required`.
- **Response:** the same stream as 2.3, cut by the same rules, with no `reply`, `discard` or turn endings.

This is what lets an app use the voice box as its text-to-speech source for anything on screen, such as reading an
earlier reply aloud.

### 2.5 The stream

`200`, `Content-Type: application/x-ndjson`, one JSON object per line, flushed as each is ready:

| `type` | Fields | Meaning |
|---|---|---|
| `start` | `version`, `mime` | The stream is open. Always first. |
| `reply` | `message_id` | Which reply is being spoken (2.3 only). Comes before its first phrase. |
| `phrase` | `index`, `start`, `end`, `text`, `audio` | One phrase. `audio` is the base64 WAV. `text` is what was spoken. `start`/`end` are offsets into the reply's text (or the given text). |
| `discard` | `keep_chars` | A round was dropped: don't play queued phrases whose `start` is at or past `keep_chars`. |
| `keepalive` | | Nothing else was sent for 15 s. |
| `done` | `reason`, and see below | The last line. Exactly one per stream, unless the connection drops. |

`done.reason`:

- `complete`: everything was spoken. `truncated: true` when the 5,000-character cap stopped it (section 5).
- `interrupted`: the turn was stopped. `failed`: the turn failed.
- `no_reply`: there was no reply to speak (2.3).
- `replaced`: a newer stream for the same session (or, for `speak`, the same caller) took over.
- `timeout`: the reply sent no new text for 10 minutes.
- `unavailable`: the voice box failed. Also `voice` (`unreachable`, `refused`, `rejected`), `message` (a sentence
  for the person) and `resume_at` (where the unspoken text starts).

Offsets count Unicode code points, the same unit as `chat_discard`'s `keep_chars`. (In Swift: `unicodeScalars`.)

Before the stream starts, a request can still be refused with a normal JSON answer: 404 for a session that is not the
caller's, 503 `speech_unavailable` (section 7), 429 `speech_busy` (section 8), 400 for a bad query.

## 3. Who may call

The same callers as chat:

- **The owner**: the global token, or an owner device.
- **A chat-only (scoped) device**: all four routes are added to `SCOPED_ALLOWED` in `web/route_access.py`, and to
  the list pinned in `tests/test_route_access.py`.
- **A daemon run with auth off** (empty token): open, like every other route.

The limits are chat's:

- A scoped device can stream only a session it owns. The session path guard in `web/server.py` already answers 404
  for any other id ("not yours" is "not there"), before the handler runs; the stream route reads, so it never claims a
  new id.
- The reply text comes only from frames the bridge already sends for that session. The route reads no history and
  nothing a scoped device could not already see in its own chat.
- Transcription and `speak` belong to no session. Like sending a chat message, any paired device may use them, within
  the limits in section 8.

## 4. The `speech` capability

The `subscribed` ack's `server_capabilities` gains `speech` when a voice box is configured for speech-to-text or
text-to-speech: `stt.url` / `tts.url` (or `whisper.url`) in `prometheus.yaml`, or an approved piece. It says what is
configured, not that the box answers right now, the same as `voice_input`. Which half works, and whether it answers, is
`GET /api/speech`.

An older daemon sends no `speech`, and an app keeps its on-device voice and recognition, exactly as today. An app
should show OAra Voice as a voice source only when it sees `speech`.

## 5. Cutting phrases

Modelled on OAra Voice's own phrase chunker. Pure code: text in, phrases out, no clock and no I/O.

1. **Code blocks are never read.** A fenced block (```` ``` ```` or `~~~`) becomes the phrase "Code block skipped."
   Several in a row with only blank space between them are one. No cut lands inside a fence. An unclosed fence holds
   until it closes, or until the reply ends (then it is skipped like a closed one).
2. **A sentence end** cuts: `.` `!` `?` `…`, with any closing quote or bracket, then whitespace. A decimal (`3.5`) has no
   space after the dot, so it never cuts. It does not cut after a few common abbreviations (`e.g.`, `i.e.`, `Mr.`,
   `Mrs.`, `Ms.`, `Dr.`, `vs.`, `St.`) or after a list number at the start of a line (`1.`).
3. **A line break** after text cuts, so headings and list items are their own phrases.
4. **A tool call** cuts: whatever was said before the tool starts ("Let me check.") is spoken while the tool runs.
5. **A clause end** (`,` `;` `:` then whitespace) cuts once the phrase has enough words: 3 for the first phrase (the
   time to the first word is what the listener feels), 6 after that.
6. **The longest phrase is 200 characters.** Past that, the cut is at the last whitespace before 200, or at 200 when
   there is none.
7. **What is spoken** is the phrase with markdown taken out: heading `#`s, emphasis `*` `_` `**` `__`, inline code
   backticks, quote `>`, list bullets and numbers. A link `[text](url)` is its text; a bare URL is "link". A phrase that
   is empty after this is not spoken.
8. **At most 5,000 characters are spoken per reply** (about five minutes). Then one phrase, "The rest is too long to
   read aloud.", and the stream ends `complete` with `truncated: true`.
9. **When the reply ends**, what remains is the last phrase.

Phrases are synthesised one at a time, in order, while cutting goes on; each is sent the moment its audio exists. So
phrase 1 is being spoken on the phone while phrase 2 is on the voice box and the model is still writing phrase 3.

## 6. OAra Voice: configuration and health

### 6.1 Configuration

Nothing new. The speech route uses the voice box `infra/voice_remote.remote_settings` resolves for each half, in the
order already in the pieces contract (section 10 there):

1. `stt.url` (or `whisper.url`) and `tts.url` in `prometheus.yaml`, with the key in the env file under
   `api_key_env` (default `PROMETHEUS_STT_API_KEY` / `PROMETHEUS_TTS_API_KEY`);
2. else a voice box the owner approved by its address, with its key only on loopback or a mesh VPN address.

`tts.model`, `tts.voice` and `timeout_seconds` apply as they do today. The box serves WAV (16-bit mono), so WAV is
what the stream carries.

### 6.2 Health

- **Check:** `GET /health` at the box's root, then `GET /v1/models` to tell a refused key from a healthy box: the
  same check as `oara doctor`'s Remote STT/TTS row (`voice_remote.probe`), with a 3 s limit, off the event loop.
- **When:** only when asked. `GET /api/speech` checks a half whose last result is older than 30 s. Every real call
  also counts: a failure marks the half down at once, a success marks it up.
- **Kept apart from the chat notices.** The speech route keeps its own record of which half is down, so its failures
  never change the "isn't answering" lines Telegram and the `tts` tool send (`voice_remote._down`).
- Every failure leaves one log line: WARNING for unreachable, ERROR for refused or rejected, naming the box's
  redacted URL and never its key.

There is no background watcher in this version. When the pieces contract's 30 s watcher is built (its section 9), the
speech route reads its state instead.

## 7. When the voice box is down

Prometheus says so plainly and stops. It never falls back to this computer's own piper, espeak or Whisper for the
speech route: the phone's own voice and recognition are better and closer, and a quiet swap would hide the outage.

| When | What the app gets |
|---|---|
| No box is configured for the half | 503 `{"error": "speech_unavailable", "voice": "not_configured", "message": "No voice server is set up."}` |
| The half failed less than 30 s ago | 503 with `voice` `unreachable` / `refused` / `rejected` and its sentence |
| A phrase fails mid-stream | `done` / `unavailable`, with `voice`, `message` and `resume_at` |
| A transcription fails | 503 as above |

The sentences: "The voice server isn't answering.", "The voice server refused its key.", "The voice server rejected
the request." The app finishes the job on the device (it has the reply's text from its `chat_delta` frames, and the
recording it just sent) and tells the person once, in its own words, for example: "The voice server isn't answering,
so this phone's voice is reading the rest." One failed phrase ends synthesis for that stream; nothing is retried
quietly. The next stream tries the box again once its 30 s are up.

## 8. Limits

- Upload: the `/api` body cap (2 MB) and `gateway.media`, whichever is smaller. Compressed audio (m4a, ogg/opus) fits
  minutes of speech.
- `speak` text: 20,000 characters. Spoken per reply or text: 5,000 characters. Phrase: 200 characters.
- One reply stream per session and one `speak` stream per caller: a newer one replaces the older (`replaced`).
- At most 4 streams at once in all; the fifth is 429 `speech_busy`. The voice box is one GPU model, and four
  listeners is already more than one household.
- A reply that sends no new text for 10 minutes ends the stream (`timeout`). Long tool calls are fine; a lost turn is
  not.

## 9. How it is built

New module `src/prometheus/speech/`:

- `phrases.py`: the cutter (section 5).
- `voice.py`: the voice box client for the route (WAV bytes in and out, a `VoiceUnavailable` with its reason) and the
  health record (section 6.2). It reuses `voice_remote`'s settings, key handling and probe.
- `replies.py`: holds each session's running reply from the frames the bridge broadcasts, and the last finished one
  for 120 s; a stream follows one reply's events in order.
- `stream.py`: cutter plus one synthesis worker per stream, giving the NDJSON lines.

`src/prometheus/web/speech_routes.py` has the four routes. Outside the new module, the changes are small:

- `web/ws_server.py`, two one-line hooks (other work is going on in that file, so nothing more): `broadcast()` hands
  each frame to the observers the launcher registers, before fan-out; and `_server_capabilities()` adds `speech`.
- `web/launcher.py` registers the reply holder on the bridge; `web/server.py` registers the routes;
  `web/route_access.py` lists them.

PRs, red tests first, each with an independent review (two for the routes, which touch auth and networking):

1. This design note.
2. `phrases.py`.
3. `voice.py`, against a fake voice box built from the box's contract (wav out, `{"text"}` in, bearer key, `/health`
   without auth).
4. `replies.py` and `stream.py`, driven by frames recorded from a real `WebSocketBridge` turn.
5. The routes, the capability, the two hooks, the reference docs and the CHANGELOG.

## 10. Decisions settled here

Settled with the charter's principles (safer option, capability gating, smallest change):

- **D1. The app pulls the reply's speech over HTTP, per session.** Not pushed as WebSocket frames: that keeps
  `ws_server.py` to two hooks, and a scoped device's access rides the existing session guard instead of a new check.
- **D2. NDJSON with base64 WAV.** One request per reply, readable with a plain streaming HTTP client, no new
  dependency. Base64 costs a third more bytes on phrases that are a few seconds long.
- **D3. The voice box only, never the daemon's own engines.** Falling back is the app's job, said plainly (section 7).
- **D4. Transcription returns text; the app sends it.** A spoken turn then goes through `/api/chat/send` like a typed
  one. A leading `/` is removed, so spoken words are never read as a slash command (the WebSocket voice upload has
  the same rule, by its `[Voice message, transcribed]` marker).
- **D5. `speech` means configured, not answering**, like `voice_input`. Answering is `GET /api/speech`, and in the
  stream itself.
- **D6. No background health watcher.** A check on demand, kept 30 s, plus every real call.
- **D7. One failure ends synthesis for the stream.** No silent retries; `resume_at` lets the app carry on.
- **D8. WAV only**, because that is what the voice box serves.
- **D9. Code blocks are announced, not silently dropped**, so a listener knows something was left out.
- **D10. `speak` exists** so an app's swappable voice source can read anything aloud, not only a reply as it streams.
- **D11. The newest stream wins** per session or caller; four in all.
- **D12. Offsets in code points**, matching `chat_discard`.
- **D13. The reply's text comes only from frames the bridge already sends** for that session.

## 11. Not in this version

- Speech for other gateways (Telegram keeps its own voice replies).
- Reading a stored reply by its row id (an app can use `speak`).
- Audio formats other than WAV; a voice chosen per device.
- Start/stop detection on the daemon: it stays on the device with the microphone, by the agreed design.
- Moving the desktop app onto this route (its own step, after this ships).
- The pieces watcher and `piece_status` frames.
