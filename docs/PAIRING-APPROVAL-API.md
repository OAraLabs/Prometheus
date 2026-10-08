# Pairing and discovery API, v0.1 (draft for review)

**Status:** DRAFT. Nothing is built. Step 2 (implementation) starts only after Will OKs this document.
**Contract id:** `pairing/0.1`
**Scope:** how a new device finds a Prometheus on the home network and joins it without typing an address or a token: discovery (mDNS and `GET /api/hello`), approval requests, the operator prompt on Beacon and Telegram, token delivery, and the network setting that decides who can reach the daemon at all.
**Not in scope:** the first device (setup-mode pairing, installer P7), 6-digit codes (P4), per-device conversation scoping (P1), the approver role and per-device credentials (P2), `web.bind` itself (P6). Section 11 says who owns each.
**References:** names, not line numbers (they drift). Checked against `origin/main` at `411c63b` (2026-10-07). Beacon's side is `docs/ONBOARDING-v0.4.0-PROMETHEUS-PROMPTS.md` in the `beacon-desktop` `onboarding-spec` worktree (tasks P3, P4, P5, P6, P13); the installer's is `docs/design/macos-app-installer.md` in the `macos-app-installer` worktree.

---

## 0. In brief

1. A new device asks to join with `POST /api/pair/requests` (no credentials). It gets a request id, a poll secret and a 4-digit code, and shows the code on its own screen.
2. The operator sees "Jennifer's MacBook wants to connect. Code 4821" on Beacon and on Telegram with Approve and Deny buttons. They compare the code with the one on the new device and tap.
3. On approval the daemon mints an ordinary device token through the existing `DeviceStore.mint`. The token is **sealed to a public key the new device sent** and handed over on its next poll. It is never in plaintext on the wire and never stored in plaintext on the daemon.
4. A request lives 5 minutes. Approve, deny, cancel and expiry are one-way and atomic; the first decision wins.
5. Only four routes are reachable without a bearer token: `GET /api/hello` and the three requester routes under `/api/pair/requests`. They are an exact allowlist, tested.
6. Discovery is a hint. mDNS TXT records and `/api/hello` are unauthenticated and can be forged; the code, and later the TLS pin, are what establish trust.
7. Fresh installs listen on this Mac only. "Home network" mode listens on the LAN, and **I recommend it requires TLS with a pinned self-signed certificate** (section 7), because sealing the pairing token does nothing for the bearer token that every later request carries.

### 0.1 Decisions for Will

These are the places where I chose something your brief or Beacon's prompt left open or where the two disagree. Each has a default in this document; tell me which to flip.

| # | Question | My default | Alternative |
|---|---|---|---|
| D1 | Request lifetime. Your brief says 5 minutes; Beacon's P3 prompt says 2. | **5 minutes**, key `pairing.request_ttl_seconds`, allowed 60 to 900. Beacon adapts (its prompt says the first to land wins). | 2 minutes: safer, but too short to walk to another room. |
| D2 | Your brief has the device send "name and platform". Beacon's prompt adds a public key. | **Add `public_key` (required).** Without it the token has to cross Wi-Fi in clear at poll time. | Name and platform only, token in the poll response. Not recommended. |
| D3 | Who may approve, given P2's approver role does not exist yet. | **The global token only, plus Telegram chats in `allowed_chat_ids`.** This is the same rule as `POST /api/devices` today ("a stolen phone must not enrol a second attacker device"). When P2 lands, approver devices join. | Any valid device token may approve. Rejected: a stolen unprivileged phone could enrol an attacker. |
| D4 | Is pairing always open, or only while the operator has pressed "Add a device"? | **Always open** (your flow: the prompt just appears), bounded by limits. Key `pairing.require_window: true` flips to window-gated. | Window-gated by default: no unsolicited prompts on a café network, one extra step for the operator. |
| D5 | Does TLS ship with home-network mode or later (Beacon's P13 says later)? | **With it.** Home-network mode refuses to enable without TLS unless `network.allow_plaintext_lan: true`. | Ship sealed pairing and plain HTTP on the LAN first. Then the bearer token is readable by anyone on the same Wi-Fi from the first request on. |
| D6 | mDNS library. | **`zeroconf`** as an optional extra `discovery`, loud when enabled but missing. It is LGPL-2.1-or-later (verify before the app bundle ships it). | macOS-only `dns-sd -R` child process: no dependency, but an orphaned child keeps advertising after a daemon crash. |
| D7 | Who builds `web.bind`. The installer's notes claimed it; your brief gives me "a network setting". | **Settled with the installer session (2026-10-07): it owns the `web.bind` primitive** (config key `web.bind`, `--bind`, `PROMETHEUS_WEB_BIND`, precedence flag over env over config over `0.0.0.0`, REST plus the separately bound WebSocket bridge plus the setup server, fails closed on an unparseable value, Host check only when loopback-bound). Draft **PR #693** (`feat/web-bind`, head `e1e485f`): `web/bind.py` with `resolve_bind(config, flag=, env=) -> ResolvedBind(address, source)`, `web/loopback.py` with `is_loopback_address`, `is_loopback_host_header`, `is_loopback_peer` and an ASGI Host guard applied only on a loopback bind. Every listener (REST, the WebSocket bridge, the setup server) takes the one bind address. **I build only what sits on it:** the owner API, discovery, TLS, and the second listener set that TLS needs. PRs 4 and 5 are based on #693. | Nothing to decide unless you want it reassigned. |
| D8 | Who builds P2 (per-device credentials, approver role, owner device token). The installer's pairing step calls one function, `issue_owner_credential()`, which today returns the global token, and its public Mac app release is gated on P2. I was told only that device scoping (P1) is another session's. | **Not me.** This contract only consumes the approver role (D3). The installer's seam already exists: `api_token.issue_owner_credential(config)` (#694), today returning the global token; whoever builds P2 changes its body. I still need to know who that is. | I take P2 as a prerequisite PR before step 2. It touches the same device registry as P1, so it should not run in parallel with that session. |

| D9 | Who narrows the global `allow_origins=["*"]` CORS setting on the main app (Beacon's P6 item 5). The installer is only reporting on it. | **Unowned.** My routes do not depend on it (section 8). The installer's findings in #693 (reported, nothing changed): with a token set, `allow_origins=["*"]` is inert on `/api` because the bearer middleware wraps CORS (a preflight gets 401 with no `access-control-allow-origin`); `/` and `/health`, outside the gate, do get `*`; with **no** token a web page can read a loopback-bound daemon; and the WebSocket bridge has **no Origin check at all** (a handshake from `https://evil.example` returns 101; it still needs the token to do anything). | I add it to PR 1 as its own commit, or a separate small PR. Say which clients need CORS at all first (the bundled dashboard on the same origin does not). |

### 0.2 Where this differs from Beacon's P3 and P5 prompts

Beacon is building against a stub of its own prompt. These are the deltas; I will send them the final list once you OK this.

| Item | Beacon prompt | This contract | Why |
|---|---|---|---|
| Request lifetime | 2 minutes | 5 minutes (D1) | Your brief. |
| `tls` field | not present | added to `/api/hello` and TXT | A client cannot tell which scheme the advertised port speaks otherwise. `fp` is also the instance identity on a plain-HTTP install, so it cannot double as "TLS on". |
| Statuses | pending, denied, expired, approved | adds `canceled` and `delivered` | The requester can cancel, and a delivered token must not be fetchable twice. |
| Poll credential | "with poll_secret" | header `X-Pairing-Secret`, never in the URL | URLs land in logs and history. |
| Cancel and acknowledge | not present | `DELETE /api/pair/requests/{id}` | One verb: cancel while pending, acknowledge receipt once approved. |
| Approved response | token sealed to the device key | adds `endpoints` and `device_id` | A new device otherwise needs a second round trip to learn the WebSocket port. |
| Seal algorithm | unspecified | X25519 + HKDF-SHA256 + ChaCha20-Poly1305 (section 4.3) | Native in Node `crypto`, Apple CryptoKit and the `cryptography` package that is already a base dependency. |
| Match code | 4 digits from a hash of both keys and the id | same formula, written out (section 4.4), plus a stated limit | See the first-contact limit in 7.5. |

---

## 1. Actors and trust

| Actor | Who | Credential |
|---|---|---|
| **Operator** | The person who owns this Prometheus. | The global token, (after P2) an approver device token, or a Telegram chat in `allowed_chat_ids`. |
| **Requester** | A new phone or laptop. | None until approved. After approval, a device token. |
| **Anyone else** | Another device on the same network, a web page in a browser, a guest. | Nothing. Can reach only the four unauthenticated routes. |

What the unauthenticated surface can be used to do, and what stops it:

| Attack | Stopped by |
|---|---|
| Flood the operator with prompts | Limits (4.5): 1 pending per source, 3 pending overall, 10 per source per hour. Optional pairing window (D4). |
| Occupy all three pending slots so the real device cannot ask | Same limits. Known trade-off, written down in 4.5. The window (D4) removes it. |
| Pretend to be "Jennifer's MacBook" | The name is typed on the requester and is not verified. The prompt says so, shows the source address, and the operator compares the code against the screen in front of them. |
| Read the token off the Wi-Fi at pairing | Token sealed to the requester's key (4.3). |
| Read the bearer token off the Wi-Fi afterwards | TLS with a pinned certificate (section 7). **Sealing alone does not help here.** |
| Browser page posts to `localhost` or a LAN address | No CORS headers on these routes, any request carrying `Origin` is refused, and the loopback Host check from P6. |
| Guess a request id or poll secret | 128-bit id, 256-bit secret, constant-time compare, unknown id and wrong secret answer identically, wrong-secret attempts rate limited. |
| Active attacker relaying the first contact | Not fully stopped. Section 7.5. |

---

## 2. The network setting

Fresh installs listen on this Mac only. A "home network" mode listens on the LAN.

### 2.1 Modes

The primitive is `web.bind`, owned by the P6 work (D7). This contract names the three states a user or client can observe and maps them onto it.

| `mode` | `web.bind` | Plain HTTP listens on | TLS listens on | mDNS |
|---|---|---|---|---|
| `this_mac` | `127.0.0.1` | loopback | nothing | no |
| `home_network` | `0.0.0.0` with TLS configured | **loopback only** | all interfaces | yes |
| `open` | `0.0.0.0`, no TLS (legacy) | all interfaces | nothing | only if `discovery.mdns` and the operator turned it on |

`open` is what every existing install has today (an absent `web.bind` keeps `0.0.0.0`, so a Mac mini reached over Tailscale does not change). It is reported honestly as `open`, not hidden as "home network". The app installer's launcher passes `--bind 127.0.0.1`, so app installs start in `this_mac`.

### 2.2 Routes (owner only: the global token, or an approver device once P2 exists)

```
GET /api/network
200 {
  "mode": "this_mac" | "home_network" | "open",
  "bind": "127.0.0.1",
  "tls": {"enabled": false, "spki_sha256": null},
  "advertising": false,
  "advertising_reason": null,        // e.g. "loopback bind", "discovery.mdns is false", "zeroconf is not installed"
  "applied": "live" | "on_restart"   // whether the last PUT has taken effect
}

PUT /api/network   {"mode": "this_mac" | "home_network"}
200 {"mode": "...", "applied": "live" | "on_restart", "warnings": ["..."]}
409 {"error": "tls_unavailable"}      // home_network without TLS and without allow_plaintext_lan
403 {"error": "owner_only"}
```

`PUT` persists `web.bind` (and TLS material) the same way `POST /api/setup/configure` pins ports. Whether the change is live or needs a restart is P6's call; this contract only fixes the response shape so Beacon's "Let my other devices connect" toggle can show the truth. Switching back to `this_mac` stops the LAN listeners and mDNS and closes WebSocket connections whose peer is not loopback. It revokes no device.

`oara network show | this-mac | home` is the terminal route (calls the same code).

### 2.3 The laptop caveat

`home_network` means "whatever network this Mac is on right now". It does not know which Wi-Fi is home. A laptop switched to home mode and carried to a café is listening there. Mitigations that exist: TLS plus device tokens mean a stranger gets no API access; limits bound prompt spam; D4's window removes unsolicited prompts entirely. Mitigation I am not proposing: auto-narrowing to a "known networks" list (it needs a notion of home that no client has).

---

## 3. Discovery

### 3.1 mDNS

Advertise `_prometheus._tcp.local.` when the mode is `home_network`, or `open` with `discovery.mdns` true. Never in `this_mac`.

| Item | Value |
|---|---|
| Instance name | the display name (3.3), truncated to 63 UTF-8 bytes (DNS label limit). The library renames on a conflict ("Name (2)"); the TXT `name` stays as set. |
| Port | the REST port clients should use: the TLS port in `home_network`, the plain port in `open`. |
| Interfaces | only interfaces the daemon listens on **and** that carry a private IPv4 (RFC 1918) or link-local address. Never a public address, a tunnel, or a CGNAT (Tailscale) address. Re-evaluated every 60 s so Wi-Fi roaming re-registers. |

TXT records. No secrets, no paths, no usernames, no addresses:

| Key | Example | Meaning |
|---|---|---|
| `v` | `0.9.7` | package version |
| `name` | `Will's Mac mini` | display name |
| `agent` | `Prometheus` | the assistant's name (`system.name`) |
| `fp` | `3fa9c1e07b2d4a68` | first 16 hex chars of SHA-256 of the instance public key (SPKI DER). A display and change-detection hint, never a trust anchor. Empty if the instance has no key yet. |
| `pair` | `approve` | `approve` when approval requests are enabled and the daemon is configured; `code` when only a 6-digit code works (setup mode, or `pairing.requests_enabled: false`) |
| `tls` | `1` | `1` when the advertised port speaks TLS, else `0` |

Config opt-out: `discovery.mdns: false`. If it is enabled and the library is missing, the daemon logs a WARNING at boot, `GET /api/network` reports `advertising: false` with the reason, and `oara doctor` gets a row. A silent no-op would be the config-dark failure this repo already has rules against.

Crash behaviour: a SIGKILLed daemon cannot send a goodbye packet, so its record lingers until the TTL expires. Clients therefore **always confirm with `GET /api/hello`** before showing an instance to a user.

### 3.2 `GET /api/hello`

Unauthenticated. No CORS headers (the request is refused if it carries `Origin`). `Cache-Control: no-store`. Served by the normal app **and** by the setup-mode server, from one shared function, so a client never needs to know which one it hit.

```json
{"v": "0.9.7", "name": "Will's Mac mini", "agent": "Prometheus", "fp": "3fa9c1e07b2d4a68", "pair": "approve", "tls": false}
```

Exactly these six fields: the same six as the TXT record, `tls` as a boolean here and `"1"`/`"0"` there. An allowlist test pins the JSON key set and checks the TXT dictionary agrees with it, so a new field fails both. Not in it, ever: addresses, OS version, user names, model, uptime, session counts, device counts, whether anyone is paired.

It replaces the 401-body fingerprint that Beacon and the installer launcher use today to answer "is there a Prometheus here". They keep the old probe working (its wording is unchanged) and prefer hello when it answers.

Rate limit: 60 per minute per source, then `429`.

### 3.3 Display name

`pairing.display_name` if set, else macOS `scutil --get ComputerName`, else `platform.node()`. Control characters stripped, 64 characters maximum. One function, used by hello, TXT and the prompts.

---

## 4. Pairing requests

Approve-to-pair is for the **second and later** devices. A fresh install has no operator who could approve the first one; that is setup-mode pairing (P7 same-Mac, or the 6-digit code). In setup mode `POST /api/pair/requests` answers `403 pairing_unavailable`.

### 4.1 Flow

```
 New device                         Prometheus                         Operator (Beacon / Telegram)
     |  GET /api/hello  ----------------> |   (discovery confirms it is a Prometheus)
     |  POST /api/pair/requests --------> |   store request (pending, 5 min)
     | <-- {request_id, poll_secret,      |   -- pairing_pending frame / Telegram message ------->
     |      match_code, ...}              |      "Jennifer's MacBook wants to connect. Code 4821"
     |  shows "Code 4821, waiting..."     |
     |  GET .../{id}  (every 2 s) ------> |  <-- Approve tapped
     | <-- {status: approved, sealed}     |      mint device, seal token to the device key
     |  unseal, store token               |
     |  DELETE .../{id} (acknowledge) --> |      wipe the sealed blob; status: delivered
```

### 4.2 Requester routes (no bearer; exact allowlist, section 8)

**`POST /api/pair/requests`**

```json
{
  "device_name": "Jennifer's MacBook",
  "platform": "macos",
  "public_key": "<base64url, 32 raw bytes, X25519>"
}
```

* `device_name`: 1 to 64 characters after trimming; control characters and newlines are rejected, not stripped, so what the operator sees is what was sent.
* `platform`: one of `ios`, `macos`, `windows`, `linux`, `android`, `other`. Anything else is stored as `other`. (`DeviceStore.mint` today maps to `ios`, `macos`, `other`; the registry keeps that mapping, the request keeps the real value for the prompt.)
* Unknown keys are ignored (forward compatibility). Body at most 4 KiB, `Content-Type: application/json`.

`201`:

```json
{
  "request_id": "c3f1a9e07b2d4a6831ff09aa5e2b7c44",
  "poll_secret": "<43 chars, base64url, 256 bits>",
  "match_code": "4821",
  "instance_public_key": "<base64url SPKI DER of the instance key>",
  "expires_at": 1760000300,
  "ttl_seconds": 300,
  "poll_interval_seconds": 2,
  "notified": true
}
```

`notified` is true when at least one operator channel took the prompt (a connected Beacon, or a Telegram message that sent). When false the request is still valid; the new device can say "nobody is watching Will's Mac mini; open Beacon there". That reveals that no operator is online, which is information a LAN neighbour could already infer.

**`GET /api/pair/requests/{request_id}`** with header `X-Pairing-Secret: <poll_secret>`

```json
{"status": "pending", "expires_at": 1760000300}
{"status": "denied"}      {"status": "expired"}      {"status": "canceled"}      {"status": "delivered"}
{
  "status": "approved",
  "device_id": "9b2e...",
  "sealed": {
    "alg": "x25519-hkdf-sha256-chacha20poly1305",
    "ephemeral_public_key": "<base64url 32 bytes>",
    "nonce": "<base64url 12 bytes>",
    "ciphertext": "<base64url>"
  },
  "endpoints": {"rest": "https://192.168.1.20:8006", "ws": "wss://192.168.1.20:8011"},
  "tls": {"spki_sha256": "<hex, 64 chars>"},       // null on a plain-HTTP instance
  "approved_at": 1760000042
}
```

The sealed plaintext is JSON: `{"token": "...", "device_id": "...", "name": "..."}`. `endpoints` are built from the `Host` the requester used plus the configured ports and scheme, so they are reachable from the requester's side. `Cache-Control: no-store` on every response. Polling faster than once per second gets `429`.

A missing id, a wrong secret and a secret for a different request all return the same `404 {"error": "unknown_request"}`.

**`DELETE /api/pair/requests/{request_id}`** with `X-Pairing-Secret`

* `pending`: becomes `canceled`; the operator's prompts are retracted.
* `approved`: the requester confirms it holds the token; the sealed blob is wiped and the status becomes `delivered`.
* Anything else: `204`, no change (idempotent).

### 4.3 Sealing the token

Per request, in order:

1. The requester generates an X25519 key pair, sends the public half, keeps the private half in memory (or the Keychain) until it has unsealed the token.
2. At approval the daemon generates an ephemeral X25519 key pair, computes the shared secret with the requester's public key, and derives a 32-byte key with HKDF-SHA256: salt = the ASCII `request_id`, info = `prometheus-pair-v1/seal`.
3. It encrypts the JSON plaintext with ChaCha20-Poly1305: 12 random bytes of nonce, associated data = the ASCII `request_id`.
4. The sealed blob (no plaintext token) is stored with the request. **The plaintext token exists only in the daemon's memory for the duration of the approve call.** The device registry stores its SHA-256, as it does for every device.

Because the blob can only be opened with the requester's private key, a poll that is retried after a dropped response, or after a daemon restart, safely returns the same blob. It is served until the requester acknowledges (`DELETE`) or for 5 minutes after approval, whichever comes first.

A device that never collects its token is **revoked automatically** when that window closes and the audit log says so. A minted-but-never-delivered token does not stay live.

The implementation PR ships test vectors so Beacon desktop and iOS can check their side against fixed bytes.

### 4.4 The match code

```
digest = SHA-256( "prometheus-pair-v1" || 0x00 || device_public_key (32 bytes)
                  || instance_public_key (SPKI DER) || request_id (ASCII hex) )
match_code = (big-endian uint32 of digest[0:4]) mod 10000, zero-padded to 4 digits
```

The server returns it in the `201` and shows it to the operator. The requester **also computes it locally** from its own key and the values it received, and displays its own result. That makes the code catch a naive relay that swapped the key. Read 7.5 before relying on it for more than that.

`instance_public_key` is the instance identity key (7.3), generated on first need on any install, TLS or not.

### 4.5 Limits

Keyed on the TCP peer address. `X-Forwarded-For` is never read on these routes (an unauthenticated caller controls it).

| Limit | Default | Config key | On breach |
|---|---|---|---|
| Pending requests per source | 1 | `pairing.max_pending_per_source` | `429 rate_limited`, `reason: per_source_pending` |
| Pending requests overall | 3 | `pairing.max_pending` | `429`, `reason: pending_full` |
| Requests per source per hour | 10 | `pairing.max_requests_per_source_per_hour` | `429`, `reason: hourly` |
| Poll interval | 1 s minimum | (fixed) | `429`, `reason: poll_too_fast` |
| Wrong poll secret per source | 5 per minute | (fixed) | `429`, `reason: bad_secret` |
| Body size | 4 KiB | (fixed) | `413` |

Every `429` carries `Retry-After` and `{"error": "rate_limited", "reason": "...", "retry_after_seconds": n}`. Over loopback every caller shares one source, so the per-source and overall limits coincide; that path is the installer's `POST /api/pair/local`, not this one.

The overall cap is also a denial of service: a hostile neighbour can fill the three slots and block the real device for 5 minutes. D4's pairing window closes that hole at the cost of one operator step.

### 4.6 What approval grants

Approval calls **one function**, `mint_paired_device(name, platform)`, which calls `DeviceStore.mint(name, platform)` and nothing else. **It never calls `api_token.issue_owner_credential()`** (the installer's single source for the owner/first-device credential, #694). That function returns the owner credential, which today is the daemon's global token; an approved second device must get an ordinary non-approver token, never the owner's. When P2 lands there are two distinct mint paths on purpose: owner (installer, approver rights) and approved device (this contract, no approver rights). `mint_paired_device` passes no scope, no approver flag and no permissions, and the request has no field to ask for any. So the new device gets exactly what the registry gives an ordinary non-approver device at that moment: today that is full API access except the global-only routes (`POST /api/devices` and the pairing decision routes); when device scoping (P1) and the approver flag (P2) land it picks up their defaults with no change here. The device appears in `GET /api/devices` and is revoked with the existing `DELETE /api/devices/{id}`.

An operator may rename at approval (`{"name": "Jennifer's MacBook (kitchen)"}`), because the typed name is unverified.

### 4.7 State machine and storage

```
pending --approve--> approved --DELETE or 5 min--> delivered
   |                     \--5 min uncollected--> uncollected (device auto-revoked)
   |--deny--> denied
   |--ttl---> expired
   \--DELETE-> canceled
```

* Every transition is one SQLite statement guarded by the current state (`UPDATE ... WHERE state='pending' AND expires_at > now`, check the row count). Nothing here relies on a file delete succeeding as an exactly-once claim: the installer session measured concurrent `unlink()` of one path letting up to 7 of 8 callers "succeed" on APFS. Two operators tapping at once: one wins, the other gets `409 not_pending` with the winner's status. An approve that arrives after the TTL gets `410 expired`.
* Rows live in a `pair_requests` table in the existing `devices.db`, **created on first use** (the way `computer_devices` is), so a box that never pairs a device grows no table and the parity fixtures, which record every table in that file, do not change.
* Stored: id, SHA-256 of the poll secret (never the secret), name, platform, requester public key, source address, match code, state, timestamps, deciding channel and identity, device id, sealed blob until delivery. Never stored: the plaintext token, the poll secret.
* Terminal rows are deleted after 7 days.

### 4.8 Audit

One structured audit record per create, approve, deny, cancel, expiry, delivery and auto-revoke: request id, device name, platform, source address, deciding channel (`beacon:<device>`, `telegram:<chat>`, `cli`, `system`). Written through the existing permission audit trail if its row shape fits, else a logger line with a stable `pairing:` prefix. Never logged: the poll secret, the token, the sealed blob.

---

## 5. Operator prompts

### 5.1 Beacon (WebSocket)

Two frame types, sent **only to connections whose identity is the global token** (and approver devices after P2):

```json
{"type": "pairing_pending", "timestamp": 1760000000.0,
 "payload": {"request_id": "...", "device_name": "Jennifer's MacBook", "platform": "macos",
             "source_ip": "192.168.1.42", "match_code": "4821",
             "created_at": 1760000000, "expires_at": 1760000300, "ttl_seconds": 300}}

{"type": "pairing_resolved", "timestamp": 1760000042.0,
 "payload": {"request_id": "...", "resolution": "approved", "by": "telegram", "resolved_at": 1760000042}}
```

`resolution` is `approved`, `denied`, `expired` or `canceled`. `by` is `beacon`, `telegram`, `cli` or `system`.

* A connecting approver is **backfilled** with one `pairing_pending` per live request, so opening Beacon after the request arrived still shows it. The same list is `GET /api/pair/requests`.
* The frames go straight to those sockets through the bridge, **not through the SignalBus.** The SignalBus tail is durable and replayed to any authenticated client by the activity feed; a pairing frame carries a source address and a code and must not reach a non-approver device.
* Beacon renders `device_name` as text, never markup. Suggested copy: "**Jennifer's MacBook** wants to connect. Code **4821**" with platform and source address underneath, "Approve" and "Deny".
* Optionally the operator may type the code instead of just tapping: `approve` accepts `{"match_code": "4821"}` and answers `422 code_mismatch` when it differs. Clients may offer that stronger UX; the API does not require it.
* Phone push: if APNs push is configured, `pairing_pending` also goes through the existing push dispatcher to approver devices (alert text only, no code). If the dispatcher's seam does not take it cleanly, it becomes a follow-up. No OAra relay (Will's ruling Q3).

### 5.2 Telegram

Today the adapter has **no callback-query handler**: approvals are `/approve` text commands. This adds one.

```
Jennifer's MacBook wants to connect.
Code 4821

macOS · 192.168.1.42 · expires in 5 min
The name is typed on the new device and is not checked. Approve only if the code matches the one on its screen.

[ Approve ]  [ Deny ]
```

* Sent as plain text (`parse_mode=None`), never Markdown or HTML, because the name is attacker-controlled.
* Sent only to **private** chats in `allowed_chat_ids`. A group chat in that list gets nothing: group membership is not an identity, and any member could otherwise approve.
* `callback_data` is `pair:a:<request_id>` or `pair:d:<request_id>` (39 bytes, under Telegram's 64-byte limit). The handler matches `^pair:[ad]:[0-9a-f]{32}$`, runs after the existing `_authorize_update` check (group -1, which covers callback updates because they carry a chat), answers the query, then **edits the message** to its final state ("Approved from Telegram at 14:02", "Denied", "Expired", "Approved on Beacon") and removes the buttons. A tap on an old message answers "already approved" or "expired".
* One message per request, so at most three outstanding.

### 5.3 Terminal

`oara pair list | approve <id> | deny <id>` call the same routes with the global token read from the env file. This is the route for a headless box with no Beacon open and no Telegram.

### 5.4 Races and reachability

Beacon, Telegram and the CLI can all answer. The first decision wins; every other surface is told through `pairing_resolved` or a message edit. If no channel is reachable the request still lives out its 5 minutes and any channel opened within that window shows it.

---

## 6. Who can call what

| Route | Anonymous | Device token | Approver device (P2) | Global token |
|---|---|---|---|---|
| `GET /api/hello` | yes | yes | yes | yes |
| `POST /api/pair/requests` | yes (limited) | yes | yes | yes |
| `GET` / `DELETE /api/pair/requests/{id}` | with the poll secret | with the poll secret | with the poll secret | with the poll secret |
| `GET /api/pair/requests` (list) | no | no | yes | yes |
| `POST /api/pair/requests/{id}/approve` and `/deny` | no | no | yes | yes |
| `GET /api/network` | no | yes (read) | yes | yes |
| `PUT /api/network` | no | no | yes | yes |
| `pairing_*` WebSocket frames | n/a | not sent | sent | sent |

Until P2 lands, the "Approver device" column is empty and the decision routes are global-token only (D3). The Telegram button is authorised by `chat_allowed`, the same test every Telegram command uses. Errors: an authenticated non-owner gets `401`, matching how `POST /api/devices` answers today ("the wrong credential for this route, not a lesser one"); P2 may move that to `403` for all owner routes and this contract follows it.

---

## 7. Transport security: not sending tokens in plaintext over Wi-Fi

### 7.1 The problem

In `open` and plain-HTTP `home_network` operation, every REST call carries `Authorization: Bearer <token>`, the WebSocket handshake carries the token, and `/v1/*` does too. Anyone on the same Wi-Fi who can read that traffic gets a credential that runs tools on this Mac, plus every chat message. Sealing the pairing token (4.3) protects one message. It does nothing for the thousand that follow.

### 7.2 Options

| | Option | Protects against | Costs and breaks |
|---|---|---|---|
| **A (recommended)** | Self-signed TLS certificate, requester pins the public key (SPKI hash) at pairing | Passive sniffing of tokens and chat; impersonation of the instance after pairing | Certificate generation and a TLS listener; two more ports; the bundled dashboard over the LAN shows a browser warning (Beacon and iOS clients are unaffected); clients must implement a pinning check (Electron `setCertificateVerifyProc`, iOS `URLSessionDelegate`, both feasible); reinstalling that loses the key means re-pairing every device. |
| B | Sealed pairing only, plain HTTP afterwards | Passive sniffing **of the pairing message** | Everything after pairing is readable on shared Wi-Fi. Fine for loopback, wrong for a LAN. |
| C | Per-request signatures (HMAC or DPoP style) over plain HTTP | Token theft (the token is never sent) and replay | No confidentiality (chat is still readable); replaces the bearer scheme the OpenAI-compatible `/v1` clients and curl rely on; a large auth-middleware change. |
| D | A real certificate from a public CA, e.g. a wildcard on an OAra domain with DNS-01 issuance (the Plex model) | Same as A, with no browser warning | Needs OAra-run infrastructure and either escrow of each user's private key (they could impersonate any instance) or per-install issuance over the internet. Contradicts "no OAra relay". |
| E | No LAN mode; Tailscale or WireGuard for anything beyond this Mac | Everything | Already works for technical users. Not for the person this feature is for. |

### 7.3 Recommended design (A)

* **Instance identity key.** An ECDSA P-256 key generated on first need, mode 0600, in a data directory that survives app upgrades and is removed only by `--purge-data`. P-256 rather than Ed25519 because it is accepted by every TLS stack the clients use. The same key is `instance_public_key` in 4.2 and the `fp` in 3.
* **Certificate.** Self-signed X.509 over that key, SAN `<hostname>.local` and `localhost`, 5-year validity, re-issued over the same key 30 days before expiry. Clients ignore names and dates and pin the key, so a renewal changes nothing for them.
* **Pin.** SHA-256 of the SPKI DER, all 32 bytes. TXT and hello carry the first 16 hex characters for display and change detection only.
* **Learned at first contact.** The client accepts the certificate for the single `POST /api/pair/requests` call, records the SPKI it saw, and **checks it equals `instance_public_key` in the response.** The match code (4.4) is a hash that includes that key, so the code the operator compares is bound to the certificate the requester actually met.
* **Pinned afterwards.** The client stores `(token, spki_sha256)` and enforces the pin on every REST and WebSocket connection. A mismatch is a hard failure ("this is not the Prometheus you paired with"), with no click-through, like an SSH host-key change.
* **Listeners.** Under #693 REST and the WebSocket bridge each bind exactly one address, so `home_network` needs a second listener set, which is this PR's to add on top of `resolve_bind`. Loopback keeps plain HTTP on the existing ports (8005, 8010), so Beacon on the same Mac, the installer launcher, `oara doctor` and curl are unchanged. In `home_network` the LAN gets TLS on **separate ports** (defaults REST 8006, WebSocket 8011; keys `web.tls_api_port`, `web.tls_ws_port`), so the daemon never has to track interface addresses to run two kinds of listener on one port. mDNS advertises the TLS REST port with `tls=1`.
* **Opt-out.** `network.allow_plaintext_lan: true` is the only way to run `home_network` without TLS. It logs a WARNING at every boot and adds a `/api/network` warning. `open` installs (the Mac mini) are not touched.

### 7.4 Phasing

This stack ships in five PRs (section 12). PRs 1 to 3 (hello, requests, prompts) are useful on a loopback or `open` install before TLS exists. The `home_network` mode is not offered until PR 5 lands, so there is never a released state where the LAN mode is plain HTTP by default. That is the part that departs from Beacon's P13 ("later"); D5.

### 7.5 What this does not stop: first contact

The 4-digit code is a hash the attacker can grind. An attacker who is actively relaying the very first request can choose the request id and key it presents to the requester after seeing the code the real daemon issued, and try about 10,000 values until the two codes agree. So trust on first use here is as strong as SSH's: an attacker must be on the path at the moment of pairing, which is a much smaller population than "anyone on the Wi-Fi", but it is not nothing.

A non-grindable code needs a commit-reveal step: the requester sends a hash of a random nonce in the request, the prompt is shown only after the requester reveals the nonce in a second call, and the code mixes both nonces. It costs one extra call and a hold on the prompt. It is additive (unknown request fields are already ignored), so v0.1 omits it and this section reserves the idea. If Will wants it in v1, the change is local to 4.2 and 4.4.

---

## 8. The unauthenticated surface

`_check_bearer_token` in `web/server.py` today gates every path starting with `/api/` or `/v1/` (its own comment: "widen here, never elsewhere"). The exemption is an **exact allowlist of (method, path pattern)**, agreed with the installer session, in one shared module:

```python
# src/prometheus/web/public_routes.py
PUBLIC_ROUTES: frozenset[tuple[str, str]]      # (METHOD, "/api/path/{param}")
def is_public_route(method: str, path: str) -> bool
```

`{name}` matches exactly one non-empty path segment: no prefix match, no trailing-slash tolerance, so `/api/pair/requests/{request_id}` does not admit `.../approve`. `_check_bearer_token` calls it with the **routed** path, never `request.url` (the existing comment and `tests/test_auth_uses_routed_path.py` explain why). The module is created by the installer's draft **PR #694** (stacked on #693), starting as `{("POST", "/api/pair/local")}`; my PRs are stacked on it and add lines. This contract's lines, each added in the PR that adds its route, because the enumeration test fails an entry that matches no registered route (PR 1 adds the hello line, PR 2 the three `pair/requests` lines):

```
("GET",    "/api/hello")
("POST",   "/api/pair/requests")
("GET",    "/api/pair/requests/{request_id}")
("DELETE", "/api/pair/requests/{request_id}")
```

`tests/test_public_routes.py` (also #694) walks the real route table and requires "open routes == `{GET /, GET /health}` plus `PUBLIC_ROUTES`", so each line is an explicit review item. Today the only routes that answer without a bearer are `GET /` and `GET /health`, both outside the `/api` and `/v1` prefixes; nothing under those prefixes is public until #694 lands. A neighbouring path (`.../approve`, `/api/pair/codes`) stays gated because the match is exact.

**What the gate does, and what my routes must do themselves.** For a public route the gate (#694) refuses any request carrying an `Origin` header with `400 browser_not_allowed`, before `call_next` and from outside the CORS layer. It does nothing else: the bearer check is skipped. So each of my routes does its own checks: read the poll secret from the `X-Pairing-Secret` header only (never the URL); require `application/json` and a body of at most 4 KiB; compare secrets in constant time against their SHA-256; apply the section 4.5 limits; and keep `X-Pairing-Secret` out of logs (it is added to the log redaction list). The installer's route does loopback-peer and loopback-Host checks for its own purpose; mine deliberately accept LAN peers.

**No CORS headers, without touching the global CORS setting.** The Origin refusal runs in `_check_bearer_token`, which is registered after `CORSMiddleware` and so sits outside it; a refused request, preflight included, never reaches the CORS layer. I checked this on a toy app with the same middleware order: the public route answered 400 with no `access-control-*` header while an ordinary route kept `access-control-allow-origin: *`. The gate in #694 is that same check, and PR 1 pins it on the real app for hello. This does **not** narrow `allow_origins=["*"]` for the rest of the API; that is Beacon's P6 item 5, the installer is only reporting on it, and so it currently has no owner (raised with Will).

The 401 body keeps **both** "unauthorized" and "Bearer" in the error text; Beacon and the installer launcher match on both.

---

## 9. Configuration

All keys go in `config/prometheus.yaml.default` with the defaults below, are read by exactly one place each, and are documented in `docs/reference/config-keys.md`.

```yaml
web:
  bind: "127.0.0.1"        # owned by the P6 work; absent means the legacy 0.0.0.0
  tls_api_port: 8006
  tls_ws_port: 8011
network:
  allow_plaintext_lan: false
discovery:
  mdns: true               # advertise when listening beyond loopback
pairing:
  requests_enabled: true
  request_ttl_seconds: 300           # 60 to 900
  max_pending: 3
  max_pending_per_source: 1
  max_requests_per_source_per_hour: 10
  require_window: false              # D4
  display_name: ""                   # empty = computer name
  telegram_prompts: true
```

---

## 10. Error codes

All JSON bodies are `{"error": "<code>", ...}`.

| HTTP | `error` | When |
|---|---|---|
| 400 | `invalid_request` | bad JSON, missing or malformed field (`fields` lists them) |
| 400 | `browser_not_allowed` | request carried `Origin` |
| 401 | `unauthorized` | operator route without a valid owner credential (wording of the existing 401 unchanged) |
| 403 | `pairing_unavailable` | setup mode, or `pairing.requests_enabled` is false |
| 403 | `pairing_closed` | `pairing.require_window` is true and no window is open |
| 404 | `unknown_request` | no such id, wrong secret, or a secret for another request |
| 409 | `not_pending` | decision on a request that is no longer pending (`status` says which) |
| 409 | `tls_unavailable` | `PUT /api/network` to `home_network` with no TLS and no `allow_plaintext_lan` |
| 410 | `expired` | decision arrived after the TTL |
| 413 | `too_large` | body over 4 KiB |
| 422 | `code_mismatch` | optional `match_code` on approve differs |
| 429 | `rate_limited` | see 4.5 (`reason`, `retry_after_seconds`, `Retry-After`) |

---

## 11. Coordination

| Session | What this contract needs from it | What it can build against now |
|---|---|---|
| **Beacon desktop onboarding** (`b9b342`) | Confirm it calls hello and the pairing routes from the Electron **main process** (no `Origin`), not the renderer. Confirm 5 minutes. Accept `tls`, `endpoints`, the two extra statuses, the header name and the seal algorithm (4.3). | Sections 3, 4.2 to 4.4 and 5.1 are the wire shapes. Test vectors come with PR 2. |
| **macOS installer** (`339a70`) | Share the one exemption allowlist with its `POST /api/pair/local` (whichever PR lands first creates it, the other adds a line). Keep the 401 wording. Use its loopback helpers (`web/loopback.py`: `is_loopback_address`, `is_loopback_host_header`, `is_loopback_peer`) rather than a second implementation. | Hello gives its launcher a probe that does not depend on the 401 text. App installs start in `this_mac`. `/api/pair/local` is out of this contract's route set; mine are `/api/pair/requests` and (P4) `/api/pair/codes`. |
| **Device scoping** (`44100f`) | The registry's default scope for a non-approver device, and whether `DeviceStore.mint` gains parameters (it should keep today's call working). The approver flag (P2): who builds it (D8). | Pairing mints through one wrapper (4.6), so their change lands in one place. |

I have sent no contract text to any of them. The only exchange so far is D7 with the installer session, and its answer is recorded above. PR 4 and PR 5 are based on draft #693.

---

## 12. Implementation plan (step 2, after approval)

Every PR: branched from a freshly fetched `origin/main` in its own worktree, red test first with the failing output kept, Auto-fix off, **not merged, not deployed**. Each updates `docs/guide/api.md` and `docs/reference/routes.md`.

| PR | Contents | Red tests (fail on main today) |
|---|---|---|
| 1 | Stacked on #694: my hello line in `PUBLIC_ROUTES` (the module and the enumeration test already exist there); `GET /api/hello` in both servers; the instance identity key; the display-name function | hello is a 404; the key set is exactly the six fields; no route under `/api` or `/v1` is public today; hello and every other public route answer an `Origin` request with 400 and no `access-control-*` header |
| 2 | Pairing core: `pair_requests` table (lazy), state machine, sealing, requester and operator REST routes, limits, audit, `mint_paired_device` | the approver and the requester compute the same code; the token never appears in any response body, log line or database row in plaintext; unseal with the test vectors; expiry at the TTL (injected clock); two simultaneous approves, one wins; non-owner cannot approve; each limit returns 429; an uncollected device is auto-revoked |
| 3 | Operator channels: targeted WebSocket frames with backfill, Telegram buttons with edits, `oara pair` | frames reach the global token and not a plain device; the Telegram handler rejects a callback from an unlisted or group chat; a second tap reports "already approved"; edits remove the buttons |
| 4 | mDNS advertising, `discovery.mdns`, doctor row, `GET` / `PUT /api/network` on top of `web.bind` | no registration on loopback; none when opted out; enabled-but-missing library is loud; TXT equals hello. I will also run `dns-sd -B _prometheus._tcp` on this Mac and paste the output. Based on #693 (D7) |
| 5 | TLS for `home_network`: identity key to certificate, TLS listeners, `allow_plaintext_lan` | `home_network` without TLS refuses; the certificate's SPKI equals `instance_public_key`; plain HTTP is refused on a non-loopback address in `home_network`; loopback still answers plain |

Other tests every PR keeps green: the full suite on the project venv (system `python3` fails at collection), the parity goldens, and `ruff` on `src/`.

---

## 13. What I have not verified

* **A real cross-device mDNS test.** I can register and browse on this Mac. Seeing the service from an iPhone or a second machine on the same Wi-Fi needs one of those present, and Will's phone is the only one that matters for iOS multicast permissions.
* **Telegram callbacks.** No code path in this repo handles a callback query today; the claim that `_authorize_update` covers them is from the library's `effective_chat` behaviour and gets a test in PR 3.
* **`zeroconf` licence and bundling.** LGPL-2.1-or-later from memory; to be checked against the package metadata before the app zip includes it.
* **Client pinning on iOS WebSockets.** I believe `URLSessionWebSocketTask` honours the session delegate's trust challenge; Beacon iOS should confirm before PR 5.
* **Live rebind.** Whether `PUT /api/network` can apply without a restart is P6's call.
