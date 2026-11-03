# Pieces contract, v0.1

**Contract id:** `pieces/0.1`
**Status:** accepted 2026-10-27. Nothing in it is built yet; section 13 says which PR builds what.
**Scope:** how a *piece* joins a Prometheus setup with no config file edited. Prometheus notices it, the owner
approves it once, and it is used, with a pinned key and a fallback when it goes away. A piece is a service Prometheus
connects **out** to: a voice box, a model server, a tool server, another agent.
**Not in scope:** devices that connect *in* (Beacon, phones), which join through pairing (`docs/PAIRING-APPROVAL-API.md`,
`pairing/0.2`); chat-app gateways and other cloud accounts, which no discovery can find.
**Written for:** Prometheus PRs that build it, and Beacon, which shows it.

---

## 0. In brief

1. A piece is noticed in one of two ways: it advertises on the local network (mDNS/DNS-SD), or the owner types its
   address. Neither is trust; both are hints.
2. Prometheus reads the piece's **card** (what it offers) and, if the piece has a node key, asks it to sign a fresh
   challenge with the **same** identity challenge Prometheus answers for its own clients
   (`POST /api/identity/challenge`, message `prometheus-identity-challenge/v1`).
3. The owner sees one prompt, for example "Found a voice box: Kitchen voice. Use it?", in every operator's Beacon
   (and in the terminal, and in chat if they opted in). Only an operator can answer it. The agent cannot see, approve
   or edit pieces.
4. Approval pins the piece's node key (and its TLS certificate when it has one) in `devices.db` and the piece is used
   from the next call, with no restart. A secret it needs goes to the env file under a generated name.
5. On every connect Prometheus checks the pin **before** it sends a secret or any data. A changed key stops use and
   warns, as an SSH host-key change does.
6. When a piece stops answering, Prometheus falls back to what it used before (for voice, this computer's own engines)
   and says so, once, in the conversation that used it.
7. A piece that cannot prove who it is may still be used if the owner approves it, and is labelled **not verified**.
   Plain HTTP is allowed only to loopback or a mesh VPN, never to a LAN or public address.
8. A YAML entry for the same use still works, wins, and is shown read-only. Approving a piece never edits
   `prometheus.yaml`.

## 1. Words

| Word | Meaning |
|---|---|
| **piece** | One service, on one machine, that Prometheus may connect out to. Identified by its node key when it has one. |
| **offer** | One thing a piece serves, from its card: a `kind` (`stt`, `tts`, `chat`, `embeddings`), the wire `api`, and a `url`. A voice box usually has two offers, often on two ports. |
| **use** | What Prometheus uses an approved piece for: one of its offers' kinds. The owner may approve a piece for fewer uses than it offers. |
| **verified** | The piece answered a fresh identity challenge signed by the node key pinned for it (section 4). |
| **not verified** | The owner approved a piece that has no node key. It is used, and labelled. |
| **address class** | Where the piece's resolved address is: `loopback`, `mesh`, `lan` or `public` (section 5.1). |
| **operator** | `identity.is_operator`: the global token or an owner device, as in pairing. |

## 2. Trust rules

1. **Pieces have no authority.** A piece serves requests Prometheus sends it. It cannot approve, read sessions, call
   tools, or reach any route a device could. It is not an enrolled node: FOUNDATION 3.6 (`enrolled_nodes`) is
   unchanged and this contract does not use it.
2. **Discovery is a hint.** An mDNS record, a TXT field or a card can be forged by anything on the network. Only owner
   approval plus a pin is trust.
3. **Only an operator approves.** Every route in section 8 is operator-only (`web/route_access.py`, `OPERATOR_ONLY`,
   and the ratchet test classifies each one). A scoped device is refused them by default-deny. The agent gets no tool
   that lists, approves or edits pieces.
4. **Nothing before approval.** Before approval Prometheus fetches only a card, a challenge answer and the fixed probe
   paths (section 3.3). No secret, audio, prompt, file or other data goes to a piece the owner has not approved.
5. **Proof before secrets.** For a verified piece, every connect proves the pinned key before a secret or any data is
   sent (section 4.2).
6. **No quiet downgrade.** A piece pinned with a key that stops answering challenges is refused, not used unverified.
7. **The registry is trusted state.** It lives in `devices.db`, which is already under the gate's always-denied paths
   and the Seatbelt and AppArmor write floors, so a model's shell cannot plant an approved piece. Any new file this
   contract adds goes under all three floors in the PR that first writes it.

## 3. How a piece is noticed

### 3.1 mDNS/DNS-SD

Prometheus browses with the same optional `discovery` extra its advertiser uses, while `discovery.mdns` is on, in any
network mode but `open`. It uses the advertiser's address rules: private and link-local IPv4 only, tunnel, VM and
container interfaces left out.

| Service type | Who advertises it | TXT fields | Card path |
|---|---|---|---|
| `_oara-piece._tcp` | Non-agent pieces, e.g. one running `oara piece announce` | `v=1`; `card=<path>` (optional); `fp=<16 hex>`, the first 16 hex of SHA-256 over the raw 32-byte node public key (optional); `kinds=<comma list>` (optional) | `/.well-known/piece-card.json` unless `card` says otherwise |
| `_a2a._tcp` | A2A agents, following the individual Internet-Draft for A2A over DNS-SD | as the draft says | `/.well-known/agent-card.json` unless TXT says otherwise |

The A2A service type is one constant (`A2A_SERVICE_TYPE`) because the draft may change. Other Prometheus instances
(`_prometheus._tcp`) are not pieces under this version.

TXT values are display hints only. `fp` lets a prompt show a key before the card is read; the card and the challenge
decide.

### 3.2 A typed address

`POST /api/pieces/probe {"address": "<address>"}` or `{"address": "<address>:<port>"}` asks Prometheus to look at the
one machine the owner named. This is how a piece that does not advertise joins: anything behind a mesh VPN (mDNS does
not cross it) or on another subnet.

- With a port, only that port is asked.
- Without one, a short fixed list on that one host: the card port of `oara piece announce`, the local model server
  ports setup's probe already knows (`docs/contracts/setup-probe.md`), and the usual voice ports. Never a range, never
  another host. There is no LAN scan.

### 3.3 What a probe may do

The rules of setup's probe, unchanged: GET to fixed paths only (plus the identity challenge's POST); no redirect is
followed; at most 256 KiB of one answer is read; a deadline per request and per place; only parsed fields come back,
never an answer's body. In order:

1. the card, at its path;
2. `POST /api/identity/challenge`, when the card names a node key;
3. for a piece with no card: the model server answers setup's probe knows, then the OpenAI audio pair, recognised by
   `GET /health` or `GET /v1/models` answering, and by a well-formed 4xx from `POST /v1/audio/transcriptions` or
   `POST /v1/audio/speech` to an empty body.

A probe that finds nothing answers `found: []` with a `reason` (section 11).

### 3.4 Pending

A piece that was noticed and not yet answered is **pending**. Pending pieces live in memory only: at most 16 in all and
4 per source address, each for 10 minutes, then dropped. A piece seen again while pending refreshes it. "Ignore"
stores the piece as `ignored` (section 12), and it stays quiet until its key or address changes.

## 4. Identity

### 4.1 The same key, the same challenge

A verifiable piece holds an Ed25519 node key in the format Prometheus uses for its own (`config/node_identity.py`):
the id is the base64 raw public key. It answers `POST /api/identity/challenge` exactly as Prometheus does: a request
`{"nonce": <32 bytes, base64url>}`, a signed message of six lines starting `prometheus-identity-challenge/v1`, and the
`tls_spki_sha256` line set when the answer goes over TLS. Prometheus verifies with one shared verifier, tested against
the signer's vectors.

`instance_id` in that message is the piece's own id: a piece that has no vault sends its node key's `fp` there. Pieces
are not vault nodes.

`oara piece announce` (section 13, PR 5) is the supported way for a machine to become a verified piece: it makes and
keeps the node key, serves the card and the challenge, and advertises `_oara-piece._tcp`. On a `lan` or `public`
address it serves HTTPS with a self-signed certificate, and its challenge binds that certificate's SPKI.

### 4.2 On every connect

At start, on reconnect, after an outage, and at least every 10 minutes while in use, Prometheus sends a fresh nonce
before anything else:

| What the piece answers | What Prometheus does |
|---|---|
| Signed by the pinned key (and, over TLS, bound to the certificate it presented) | Uses it. |
| Signed by another key that no approved piece has | Sends nothing to this address. Health becomes `key_changed`, a `piece_key_changed` frame and a warning card go to operators, and the use falls back. Only "Trust the new key" (an operator) resumes it. |
| Signed by another key that *is* pinned to another approved piece | That piece at a new address. Its row's address is updated. No prompt. |
| Replayed, stale (`signed_at` more than 5 minutes off), bound to another TLS certificate, or a bad signature | Refused, as `key_changed`. |
| A 404 or 401 from a piece pinned with a key | Refused, as `key_changed`. No quiet downgrade. |
| An unverified piece pinned by TLS SPKI that presents another certificate | Refused, as `key_changed`. |
| No answer | `offline` (section 9), not a key change. |

## 5. Transport

### 5.1 Address classes

Decided from the address Prometheus will connect to. A name is resolved once; every address it resolves to must be
allowed, and Prometheus connects to the address it checked (so a later answer cannot move it).

| Class | Addresses |
|---|---|
| `loopback` | `127.0.0.0/8`, `::1` |
| `mesh` | IPv4 in the RFC 6598 shared address space, which mesh VPNs hand out. (IPv6 mesh addresses count as `lan` in this version.) |
| `lan` | RFC 1918 and link-local IPv4; IPv6 unique local and link-local |
| `public` | Anything else |

### 5.2 Rules

1. **Plain HTTP only to `loopback` or `mesh`.** A piece on a `lan` or `public` address must use HTTPS. A card or a
   typed address that asks for plain HTTP there is refused (`plaintext_not_allowed`).
2. **HTTPS is pinned.** Prometheus does not rely on public certificate authorities for a piece. It pins the SPKI
   SHA-256 the piece presented at approval, and for a verified piece the challenge binds it.
3. **No secret to an unverified piece over plain HTTP.** On `loopback` or `mesh` an unverified piece may be used over
   plain HTTP, but a piece that wants a key must be verified or use HTTPS there too.
4. **A card's URLs stay on its host.** Every offer's `url` must resolve to the address the card was fetched from (any
   port). A card cannot send Prometheus to another machine or to the internet.

## 6. Cards

### 6.1 Agents: the A2A Agent Card, as-is

An agent's card is an A2A 1.0 `AgentCard`, read and checked against the A2A schema. Prometheus adds no top-level field
to it. Anything Prometheus-specific would go in a declared `capabilities.extensions[]` entry; version 0.1 needs none.
Its `signatures[]` (JWS over the RFC 8785 canonical card) are checked with the agent's node key (`alg: EdDSA`, `kid` =
the base64 node key) when present. Agents can be listed under this version but not used until the A2A adapter ships
(section 13); approving one is `409 no_adapter` until then.

### 6.2 Everything else: the piece card

The smallest shape Prometheus needs, in the Agent Card's conventions (camelCase; `name`, `description`, `version`,
`provider`, `iconUrl`; the same JWS `signatures[]`) so one reader can read both. It is not an Agent Card and does not
claim to be one.

```json
{
  "pieceCardVersion": "1",
  "name": "Kitchen voice",
  "description": "Speech-to-text and text-to-speech",
  "version": "1.0.0",
  "identity": {"nodePub": "<base64 Ed25519 public key>", "challenge": "/api/identity/challenge"},
  "offers": [
    {"kind": "stt", "api": "openai.audio.transcriptions", "url": "http://<address>:<port>/v1/audio/transcriptions",
     "models": ["<model>"], "auth": "bearer"},
    {"kind": "tts", "api": "openai.audio.speech", "url": "http://<address>:<port>/v1/audio/speech",
     "models": ["<model>"], "voices": ["<voice>"], "formats": ["wav", "pcm"], "auth": "bearer"}
  ],
  "hardware": {"accelerator": "gpu", "memoryMiB": 16384},
  "signatures": [{"protected": "<base64url JWS header>", "signature": "<base64url>"}]
}
```

| Field | Rule |
|---|---|
| `pieceCardVersion` | Required, `"1"`. Another value: the card is not read (`card_invalid`). |
| `name` | Required, 1–63 UTF-8 bytes after trimming; shown as the piece's name until the owner renames it. |
| `description`, `version`, `provider`, `iconUrl` | Optional, display only. `iconUrl` is never fetched by Prometheus. |
| `identity` | Optional. Without it the piece is not verified. `challenge` must be a path on the card's own host. |
| `offers` | Required, 1–8 entries. |
| `offers[].kind` | `stt`, `tts`, `chat` or `embeddings`. An unknown kind is ignored, never guessed. |
| `offers[].api` | `openai.audio.transcriptions`, `openai.audio.speech`, `openai.chat.completions`, `openai.embeddings`. An unknown api is ignored. |
| `offers[].url` | Required; section 5.2 rules 1 and 4. |
| `offers[].models`, `voices`, `formats` | Optional lists of strings, display and defaults only. |
| `offers[].auth` | `none` (default) or `bearer`. |
| `hardware` | Optional, display only. There is no "GPU" kind: a GPU box is a piece that offers something. |
| `signatures` | Optional. When present, a valid signature by `identity.nodePub` is required, or the card is not read. A card is never trusted on its signature alone; the challenge still decides. |

A card larger than 64 KiB, or with unknown top-level fields that are not in this table, is still read for the fields
above; unknown fields are dropped.

### 6.3 No card

A piece with no card joins only by typed address (never from mDNS). Prometheus builds an unsigned card from what the
probe found, names it after its address until the owner renames it, and it is **not verified**.

## 7. Approval

### 7.1 The prompt

Prometheus sends the facts and a ready-made sentence; Beacon draws it.

> **Found a voice box: "Kitchen voice."** It offers speech-to-text and text-to-speech. Key ab12 cd34 ef56 7890.
> Your voice messages will be sent to it. **Use it?** [Use it] [Not now] [Ignore this box]

When the piece is not verified, the sentence adds: "It can't prove who it is, so something else on your network could
pretend to be it." When it wants a key, the prompt asks for it.

### 7.2 Channels

The pairing prompt's channels and rules (`pairing/0.2` section 5): operator-only WebSocket frames, sent with
`send_to_operators` and backfilled to an operator who connects later; the terminal (`oara pieces list|approve|deny`);
and a chat-app prompt, behind its own opt-in (`pieces.chat_prompts`, default off). No phone push, until the push
dispatcher can target operators only.

### 7.3 What approval stores

One row in `pieces` (section 12): the pinned `node_pub` and/or TLS SPKI, the card's digest, the address, the approved
uses, and the name of the env variable holding its secret. A secret is sent once, with the approval, and written to the
env file (`~/.config/prometheus/env`, 0600, already under the floors) as `PROMETHEUS_PIECE_<ID>_KEY`. It is never
echoed by any route, frame or log.

## 8. Routes and frames

All routes are operator-only. JSON in and out; errors are `{"error": <code>, "detail": <English>}` (section 11).

| Route | Does |
|---|---|
| `GET /api/pieces` | The fleet view (below). |
| `POST /api/pieces/probe` | `{"address"}` → probe (section 3.2); found pieces become pending and are in the answer. |
| `POST /api/pieces/{id}/approve` | `{"uses": [...], "secret"?: "<string>"}` → `approved`. `uses` defaults to every offered kind. |
| `POST /api/pieces/{id}/deny` | "Not now": a pending piece is dropped. |
| `POST /api/pieces/{id}/ignore` | Stored as `ignored`. |
| `POST /api/pieces/{id}/trust-new-key` | After `key_changed`: pins the key the piece now proves and resumes use. |
| `PATCH /api/pieces/{id}` | `{"name"?, "uses"?}` |
| `DELETE /api/pieces/{id}` | Forgets an approved or ignored piece; its env variable is removed. |

`GET /api/pieces` answers:

```json
{
  "v": 1,
  "pieces": [
    {"id": "pc_3f9a0c2e7b1d4a65", "name": "Kitchen voice", "kind": "piece", "state": "approved",
     "source": "manual", "verified": true, "editable": true,
     "fp": "ab12cd34ef567890", "tls_spki_sha256": null,
     "address": "http://<address>", "address_class": "mesh",
     "offers": [{"kind": "stt", "api": "openai.audio.transcriptions", "url": "http://<address>:<port>/v1/audio/transcriptions"}],
     "uses": ["stt"],
     "health": {"state": "online", "since": "2026-10-27T14:02:11Z"},
     "created_at": "2026-10-27T13:58:40Z", "updated_at": "2026-10-27T13:58:40Z",
     "last_seen": "2026-10-27T14:05:00Z"}
  ],
  "pending_limit": 16
}
```

- `id`: opaque, stable for the piece's life. Clients must not parse it.
- `kind`: `piece` or `agent`.
- `state`: `pending`, `approved` or `ignored`.
- `source`: `discovered`, `manual` or `config_file`. A `config_file` row is `editable: false`, and every write route
  answers it `409 config_file_entry`.
- `verified`: false shows as the "not verified" label.
- `health.state`: `unknown`, `online`, `offline`, `refused` (the piece rejected the key) or `key_changed`.
- `created_at`, `updated_at`: when the row was first stored and last changed (UTC, ISO 8601); `last_seen`: the last
  good answer from the piece, or null. A pending piece has no row: its `created_at` is when it was noticed.

WebSocket frames, to operators only: `piece_found` (a piece became pending; the same object as one `pieces[]` entry,
plus the prompt sentence), `piece_resolved` (`{id, state}` after approve, deny, ignore or expiry),
`piece_status` (`{id, health}` on every health change), `piece_key_changed` (`{id, fp_was, fp_now}`).

## 9. Fallback, and saying so

Each approved piece in use is checked every 30 s while healthy, backing off to every 5 minutes while down, and at once
after a call to it fails.

- **Order per use:** a YAML entry for the use (section 10), else the approved piece, else what Prometheus does without
  one. For voice that is this computer's own engines, as `stt.url` / `tts.url` already fall back.
- **Never to another remote piece** unless the owner chose it as the fallback.
- **Saying so:**
  - every health change is a `piece_status` frame and shows in the fleet view;
  - for voice, the conversation that used the piece gets one plain line per outage ("Your voice box isn't answering,
    so I used this computer's voice.") and one when it is back;
  - for models, the existing reply notice of model fallback.
  - A YAML-configured remote voice server (`stt.url`, `tts.url`) says so the same way.

## 10. Config files

- A YAML entry for a use (`stt.url`, `tts.url`, `backends.<name>`, `mcp_servers.<name>`) still works and **wins**.
- The fleet view lists it with `source: config_file` and `editable: false`.
- Approving, editing or forgetting a piece never edits `prometheus.yaml`. `oara pieces export` prints the YAML
  equivalent for anyone who wants files.

## 11. Error codes

| Code | Status | When |
|---|---|---|
| `operator_only` | 403 | A scoped device (the existing default-deny answer). |
| `unknown_piece` | 404 | No piece with that id, or it expired. |
| `config_file_entry` | 409 | A write to a `source: config_file` row. |
| `no_adapter` | 409 | Approving an agent before the A2A adapter ships. |
| `plaintext_not_allowed` | 422 | Plain HTTP to a `lan` or `public` address (section 5.2). |
| `secret_needs_proof` | 422 | A secret for an unverified piece over plain HTTP. |
| `address_not_allowed` | 422 | An address that does not parse, resolves to nothing, or a card URL off its host. |
| `card_invalid` | 422 | A card that is not version 1, or whose signature does not verify. |
| `pending_full` | 429 | The pending limit (section 3.4). |
| `not_a_piece` | 200, in `reason` | The probe got answers but nothing it recognises. |
| `no_answer` | 200, in `reason` | Nothing answered on the places asked. |

`detail` is English for logs. A client writes its own message from the code.

## 12. Storage

One table in `devices.db`, created on the first write (like `computer_devices`), so a setup that never sees a piece
keeps the schema it has and the parity fixtures do not change:

```
pieces (
  id               TEXT PRIMARY KEY,
  kind             TEXT NOT NULL,                -- 'piece' | 'agent'
  state            TEXT NOT NULL,                -- 'approved' | 'ignored'
  name             TEXT NOT NULL,
  address          TEXT NOT NULL,                -- scheme://host[:port]
  address_class    TEXT NOT NULL,                -- 'loopback' | 'mesh' | 'lan' | 'public' (section 5.1)
  node_pub         TEXT UNIQUE,                  -- NULL when not verified
  tls_spki_sha256  TEXT,
  card_sha256      TEXT,
  card_json        TEXT NOT NULL,                -- the card as last read (or as built, section 6.3)
  uses             TEXT NOT NULL,                -- JSON array of kinds
  secret_env       TEXT,                         -- env variable name, never the value
  source           TEXT NOT NULL,                -- 'discovered' | 'manual'
  created_at       REAL NOT NULL,
  updated_at       REAL NOT NULL
)
```

Pending pieces and health are in memory. Rows are written only by the daemon's own routes.

## 13. Build order

| PR | Builds |
|---|---|
| 1 | This contract. |
| 2 | Voice fallback says so (section 9) for the YAML-configured remote voice servers. |
| 3 | The `pieces` store and `GET /api/pieces` (`{"v": 1, "pieces": [], ...}` until something writes a row). |
| 4 | A typed address joins a voice box: probe, prompt, approve, pin, live use, health and fallback. The gateways resolve voice per call (through the pieces registry) instead of at boot. |
| 5 | `oara piece announce`. |
| 6 | The mDNS browser. |
| 7–8 | Model servers as pieces: named backends at runtime, then "make it the main model". |
| later | Remote tool servers (MCP over HTTP), the A2A adapter (client, then an optional server), others. Each extends this contract with a version bump. |

## 14. Limits of this version

- First contact with a piece is trust on first use. The prompt shows the key's fingerprint; nothing makes the owner
  compare it with the piece's own screen.
- An IPv6 mesh address counts as `lan`, so it needs HTTPS.
- A piece's card can claim offers it does not serve; use, not the card, finds that out, and health reports it.
- `oara piece announce` does not change the server it fronts. A server with its own login still needs its own key.
