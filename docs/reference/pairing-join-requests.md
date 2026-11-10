# Pairing join requests: reference

**What this is.** How a new device (a phone, a desktop Beacon) asks to join a Prometheus and how the owner lets it in,
exactly as the daemon does it. It is written from the code, for the clients that call it. Read it when you build the
requester side (iOS, desktop) or the owner's approval screen.

**As of:** v0.9.20; nothing in this area has changed on `main` since. The numbers in section 8 are checked against the
code by `tests/test_pairing_reference.py`, and so is every route and who may call it, so this file cannot drift quietly.
**If this file and `docs/PAIRING-APPROVAL-API.md` (the longer design contract, `pairing/0.2`) disagree, this file and the
code win.** That document is still where the sealing recipe (4.3), the match code (4.4) and the reasons behind the design
are written down.

---

## 0. In brief

1. A new device has no credential. It sends `POST /api/pair/requests` with a name, a platform and a one-time public key,
   and gets back a request id, a **poll secret** and a four-digit **match code**.
2. Every operator is told: each operator Beacon socket gets a `pairing_pending` frame. The owner compares the match code on
   the device's screen with the one on theirs, and approves or denies.
3. The device polls `GET /api/pair/requests/{id}` with the poll secret. On approval it receives its token **sealed** to the
   public key it sent, opens it, and confirms with `DELETE /api/pair/requests/{id}`.
4. **An approval always makes an ordinary, scoped device.** No field on any route can ask for more. A scoped device sees
   only its own conversations (`docs/contracts/device-scoping.md`).
5. **The owner tier is a separate, explicit grant**: an **owner code**. Someone at the machine Prometheus runs on mints one
   (`oara pair --owner`), reads the eight digits to the new device, and the device redeems it once. Section 6.
6. Who may approve: the global token, or an **owner device**. A scoped device is refused with `403 operator_only`. An owner
   device cannot make another owner device: only the global token, from this machine, can mint an owner code.

## 1. Who may call what

| Route | Who | Notes |
|---|---|---|
| `POST /api/pair/requests` | anyone (no bearer) | the new device asks to join |
| `GET /api/pair/requests/{request_id}` | anyone with the poll secret | the new device polls |
| `DELETE /api/pair/requests/{request_id}` | anyone with the poll secret | cancel while pending, confirm once approved |
| `POST /api/pair/owner` | anyone with a live owner code | redeem an owner code |
| `GET /api/pair/requests` | operator | who is waiting |
| `POST /api/pair/requests/{request_id}/approve` | operator | |
| `POST /api/pair/requests/{request_id}/deny` | operator | |
| `POST /api/pair/owner-codes` | global token, from this machine | mint an owner code |
| `GET /api/pair/owner-codes/{code_id}` | global token, from this machine | what became of it |
| `DELETE /api/pair/owner-codes/{code_id}` | global token, from this machine | cancel it |

*Operator* is the global API token or an owner device (`identity.is_operator`). A scoped device that calls an operator
route gets `403 operator_only`, never `401` (a `401` would tell a client its token is dead).
*From this machine* means a loopback peer, or the daemon's own address, and never a request carrying a relay header
(`X-Forwarded-For`, `X-Real-IP`, `Via`, ...).

The four public routes also refuse a request that has an `Origin` header (`400 browser_not_allowed`): they are for apps,
not for a web page. They take JSON only, at most 4096 bytes (`413 too_large`). When the daemon has no API token set (the
API is open), pairing is off: create and redeem answer `403 pairing_unavailable`.

---

## 2. The new device: ask, poll, finish

### 2.1 `POST /api/pair/requests`

`Content-Type: application/json` (otherwise `400`). Body, a JSON object:

| Field | Type | Rules |
|---|---|---|
| `device_name` | string, required | 1 to 64 characters after trimming. No control, format or line-separator characters: such a name is **refused**, not cleaned, so the owner sees exactly what you sent. |
| `public_key` | string, required | 43 characters of unpadded base64url: a 32-byte X25519 public key, made for this request only. The token is sealed to it. |
| `platform` | string, optional | `ios`, `macos`, `windows`, `linux`, `android` or `other` (case-insensitive). Anything else, or nothing, is `other`. |
| `match_code_version` | integer or null, optional | `1` or `2`. Absent or `null` means `1`. A client that asks for `2` must refuse an answer that says `1`. |

**`201`** (with `Cache-Control: no-store`):

| Field | Meaning |
|---|---|
| `request_id` | 32 hex characters. Used in the poll URL and as the sealing salt. |
| `poll_secret` | a secret for this request only (URL-safe, about 43 characters). Send it as the `X-Pairing-Secret` header, **never in a URL**. Shown once. |
| `match_code` | four digits. Compute it yourself (section 4) and show it; the owner compares. |
| `match_code_version` | the version the daemon used |
| `instance_public_key` | base64url of the daemon's instance key (DER), an input to the match code |
| `instance_id`, `node_pub` | the daemon's identity. Pin `node_pub` for the identity challenge afterwards. Either may be `null`. |
| `expires_at` | epoch seconds when the request lapses |
| `ttl_seconds` | the request's lifetime (section 8) |
| `poll_interval_seconds` | how often to poll (`2`) |
| `notified` | true when an operator channel took the prompt (a connected operator Beacon, or Telegram if the owner opted in). When false the request is still valid, and the device should tell its user to open Beacon on the owner's device. |

Errors: `400 invalid_request` with `fields` naming each bad field (`device_name`, `public_key`, `match_code_version`; an
empty `fields` means a bad body); `413 too_large`; `403 pairing_unavailable`; `429 rate_limited` (below);
`503 identity_unavailable` (the daemon has no instance key, or for version 2 no node key, yet: restart it).

`429` carries `{"error": "rate_limited", "reason", "retry_after_seconds"}` and a `Retry-After` header. Reasons:
`per_source_pending` (this source already has a request waiting), `pending_full` (too many waiting in all), `hourly` (this
source has asked too often in the last hour). The source is the TCP peer, never `X-Forwarded-For`.

### 2.2 `GET /api/pair/requests/{request_id}`

Header `X-Pairing-Secret: <poll_secret>`. Poll every `poll_interval_seconds`; a poll sooner than a second after the last
is `429` with reason `poll_too_fast`. Answers, all `200` with a `status`:

| `status` | Body also has | Meaning |
|---|---|---|
| `pending` | `expires_at`, `notified` | still waiting. `notified` only ever goes from false to true. |
| `approved` | `device_id`, `sealed`, `endpoints`, `tls`, `approved_at`, `instance_id`, `node_pub` | open `sealed` (section 4), connect to `endpoints`, then `DELETE`. |
| `denied` | | the owner said no |
| `canceled` | | withdrawn (by the device, section 2.3) |
| `expired` | | no one decided in time, **or** the token was approved but not collected in time and the device was revoked |
| `delivered` | | you confirmed with `DELETE`; the sealed token is gone |

`endpoints` has two fields, `rest` and `ws`, built from the `Host` you used so the address is reachable from your side. A
request that came in on the TLS listener is answered with `https://` and `wss://` (the TLS WebSocket port, default `8011`),
and `tls` is an object with one field, `spki_sha256` (hex), the pin to hold the connection to. Over plain HTTP, `tls` is `null` and `ws` is
`ws://<host>:<web.ws_port>` (default `8010`).

Errors: an unknown id, a wrong secret and another request's secret are **one identical** `404 {"error": "unknown_request"}`,
so a stranger learns nothing. Wrong secrets from one source are limited: after five a minute, further wrong guesses get
`429 rate_limited` with reason `bad_secret`. The right secret is never refused for that reason.

### 2.3 `DELETE /api/pair/requests/{request_id}`

Same header. Always `204` with no body for a valid secret. If the request is still `pending` it becomes `canceled` (and the
operators are told). If it is `approved` and uncollected, it becomes `delivered` and the sealed token is **wiped**: send this
once you hold the token. Any other state: `204`, nothing changes.

---

## 3. The owner: list, approve, deny

All three need the operator's bearer token.

### 3.1 `GET /api/pair/requests`

`200 {"requests": [ {...pending payload...} ]}`, newest first, live pending requests only. Payload:

| Field | Meaning |
|---|---|
| `request_id` | |
| `device_name`, `platform` | as sent (`platform` is one of the six names) |
| `source_ip` | the TCP peer the request came from |
| `match_code` | the four digits to compare with the device's screen |
| `created_at`, `expires_at` | epoch seconds |
| `ttl_seconds` | |

### 3.2 `POST /api/pair/requests/{request_id}/approve`

Body optional; if present, a JSON object with only these keys (any other key is `400 invalid_request` and `fields` names it):

| Field | Meaning |
|---|---|
| `name` | rename the device (same rules as `device_name`) |
| `match_code` | four digits the owner **typed** from the device's screen. If present and wrong: `422 code_mismatch`, and nothing is approved. Omit it to approve without typing. |

`200 {"request_id", "status": "approved", "device_id", "name", "platform"}`. The token is minted **in the same transaction**
as the state change and sealed to the device's key: it exists in clear only inside this call, and is never returned here.
Errors: `404 unknown_request`, `410 expired` (a decision after the lifetime is too late, never "someone else decided"),
`409 not_pending` with `status` (someone decided first), `403 operator_only`.

### 3.3 `POST /api/pair/requests/{request_id}/deny`

No body. `200 {"request_id", "status": "denied"}`. Errors as approve (without the match code).

### 3.4 Operator WebSocket frames

Sent to **operator sockets only** (the global token or an owner device), never broadcast, never to a scoped device, because
they carry a source address and a match code. A socket that connects later is backfilled with what is still waiting, oldest
first, so a frame can arrive twice: **key on `request_id`**.

```
{"type": "pairing_pending",  "timestamp": <float>, "payload": { ...the pending payload above... }}
{"type": "pairing_resolved", "timestamp": <float>, "payload": {"request_id", "resolution", "by", "resolved_at"}}
```

`resolution` is `approved`, `denied`, `canceled` or `expired`. `by` is `beacon` (a REST decision), `cli` (the terminal),
`requester` (the device withdrew) or `system` (it lapsed). Telegram prompts are opt-in and off by default
(`pairing.telegram_prompts`); a decision from any channel goes through the same code, so they cannot disagree.

---

## 4. Opening the token, and the match code

The recipe is fixed so a client in any language can implement it. Bytes from a separate implementation are in
`tests/vectors/pairing_seal_v1.json`; the long form is `docs/PAIRING-APPROVAL-API.md` 4.3 and 4.4.

`sealed` has four fields: `alg` (always `x25519-hkdf-sha256-chacha20poly1305`), `ephemeral_public_key`, `nonce` and
`ciphertext`, the last three unpadded base64url.

```
shared    = X25519(your_private_key, ephemeral_public_key)
key       = HKDF-SHA256(shared, salt = ASCII(request_id), info = "prometheus-pair-v1/seal", length 32)
plaintext = ChaCha20-Poly1305(key).decrypt(nonce, ciphertext, aad = ASCII(request_id))
```

`plaintext` is JSON with three fields: `token`, `device_id` and `name`. The request id is both salt and associated data, so a blob lifted
onto another request opens nothing.

The **match code** is four digits you compute locally: `SHA-256(material + ASCII(request_id))`, first four bytes big-endian,
modulo 10000, zero-padded. Version 1: `material = "prometheus-pair-v1" 0x00 device_public_key instance_public_key_der`.
Version 2 uses `"prometheus-pair-v2"` and appends the daemon's 32-byte `node_pub` (so confirming the digits also confirms the
key the device pins). Show it on the device's own screen. It catches a relay that swapped a key; it is four digits, so it is
trust on first use, as strong as SSH's and no stronger.

---

## 5. What an approval grants

An ordinary, scoped device, made by one function (`mint_paired_device`) that takes no scope, no approver flag and no
permission, because the request has no field to ask for any. It can be revoked on its own. If the device never collects the
token, it is revoked automatically (section 8). Approving one device does not make it an operator, so a stolen phone that
was approved cannot approve or enrol another.

The registry records the device's platform as `ios`, `macos` or `other` (a request that said `windows` is listed as
`other` by `GET /api/devices`), while the request and its frames keep the six names.

---

## 6. The explicit owner grant: the owner code

The owner tier (an owner device: it is an operator, so it can approve, and it sees the owner-only screens) is never the
result of an approval. It comes only from the person at the machine: same-Mac pairing, or an owner code.

**Mint.** `POST /api/pair/owner-codes`, global token, from this machine (`oara pair --owner` does exactly this and prints
the code). No body.
`201` with four fields: `code_id`, `code` (eight ASCII digits), `expires_at` (epoch seconds) and `ttl_seconds`. Minting replaces any live code: there is
only ever one. (The global token alone can also create an owner device directly, with `POST /api/devices` and `owner: true`.)
Refusals: `403 global_token_only` (a device, **even an owner device, cannot enrol another**), `403 not_this_machine`
(not loopback, or relayed), `400 browser_not_allowed`, `403 pairing_unavailable`.

**Status / cancel.** `GET` / `DELETE /api/pair/owner-codes/{code_id}`, same access.
Status has five fields: `code_id`, `status`, `expires_at`, `redeemed_by` and `wrong_attempts`. `status` is `pending`, `redeemed`,
`expired`, `replaced` or `canceled`. `redeemed_by` is `null` until it is redeemed, then an object with `id`, `name`, `platform`,
`source` and `this_host`.
`DELETE` on a code that is not pending is `409 not_pending` with the same fields; an unknown id is `404 unknown_code`.

**Redeem.** `POST /api/pair/owner`, no bearer. Body: `{"code": "<digits>", "device_name": "...", "platform": "..."}`.
Spaces and dashes in `code` are ignored. A body that is not valid JSON, or whose `code` is not a string or whose
`device_name` breaks the rules above, is `400 invalid_request` with `fields`, and **spends no try**.
`200` with eight fields: `token`, `device_id`, `name`, `owner` (always `true`), `endpoints` and `tls` (as in section 2.2),
`instance_id` and `node_pub`. The token is in
clear here (not sealed): on this path the code is the only credential, and the reply travels over the tailnet, TLS or
loopback. `GET /api/devices` lists the device with `"owner": true`, and its token reaches the operator routes (an approved,
scoped device gets `403` there). It is recorded as enrolled by an owner code, is revocable on its own, and is not revoked
by a later redemption or by same-Mac re-pairing.
Any failure is **one** answer, `401 invalid_code`: wrong, expired, already used, replaced, cancelled, a source that is shut
out, a minute already full of wrong guesses, or no code at all. A guesser learns nothing.

**The four properties.**

| | |
|---|---|
| Single use | one redemption mints at most one device. The check, the mint and the mark happen under one lock; two right answers at once mint one device. |
| Short expiry | five minutes, measured on a monotonic clock. |
| Rate-limited | five wrong tries from one source shut that source out for the code's life; at most ten wrong tries a minute are compared at all, from anyone. No guess burns the code, so strangers can delay a redemption but never cancel the owner's code. |
| Logged and shown | every issue, redemption, replacement, expiry, cancellation and wrong try is a `pairing:` log line (never the code or the token), and each issue and redemption (who, from where) is sent as an `owner_code` frame to **operator sockets only**. |

```
{"type": "owner_code", "timestamp": <float>,
 "payload": {"event": "issued" | "redeemed", "code_id", "expires_at", "redeemed_by": null | {...}}}
```

---

## 7. What this does not do

- It does not let an approval, or an owner device, create an owner. That is deliberate (a chain from one stolen phone) and
  is the first thing a client should not try to work around.
- The match code and the sealing protect against a relay on an open network; they are not a defence against someone who
  already holds the global token. Keep it off other computers and out of the agent's shell.
- Redeeming an owner code returns the token in clear. Use the TLS endpoints (`tls` in the answer) where you have them.
- Discovery (`GET /api/hello`, mDNS) is a separate contract: `docs/PAIRING-APPROVAL-API.md` section 3.

---

## 8. The numbers

Each row is checked against the code by `tests/test_pairing_reference.py`.

| Constant | Value | Meaning |
|---|---|---|
| `pairing.request_ttl_seconds` default | 300 | a request's lifetime |
| `pairing.request_ttl_seconds` minimum | 60 | configured values below this are raised |
| `pairing.request_ttl_seconds` maximum | 900 | configured values above this are lowered |
| `pairing.max_pending` | 3 | requests waiting in all |
| `pairing.max_pending_per_source` | 1 | requests waiting from one source |
| `pairing.max_requests_per_source_per_hour` | 10 | requests one source may make an hour |
| `DELIVERY_WINDOW_SECONDS` | 300 | an approved token not collected in this long is revoked |
| `PURGE_AFTER_SECONDS` | 604800 | finished request rows are kept this long (7 days) |
| `MAX_BODY_BYTES` | 4096 | the largest body on any public pairing route |
| `POLL_INTERVAL_SECONDS` | 2 | what the requester is told to poll at |
| `MIN_POLL_GAP_SECONDS` | 1 | a faster poll is `429 poll_too_fast` |
| `WRONG_SECRETS_PER_MINUTE` | 5 | wrong poll secrets per source before `429 bad_secret` |
| `OWNER_CODE_DIGITS` | 8 | digits in an owner code |
| `OWNER_CODE_TTL_SECONDS` | 300 | an owner code's lifetime |
| `OWNER_CODE_MAX_WRONG_PER_SOURCE` | 5 | wrong tries before a source is shut out |
| `OWNER_CODE_MAX_WRONG_PER_MINUTE` | 10 | wrong tries compared per minute in all |
| `DEVICE_NAME_MAX_CHARS` | 64 | longest device name |

## 9. Configuration

The `pairing:` section of `prometheus.yaml`: `requests_enabled` (default `true`; `false` makes
`POST /api/pair/requests` answer `403 pairing_unavailable`), `request_ttl_seconds`, `max_pending`, `max_pending_per_source`,
`max_requests_per_source_per_hour`, and `telegram_prompts` (default `false`). Owner codes have no settings: their numbers are
fixed. A value of the wrong type is logged and the default used.

## 10. Error codes

| Status | `error` | Where |
|---|---|---|
| 400 | `invalid_request` (with `fields`) | create, approve, redeem |
| 400 | `browser_not_allowed` | every public pairing route and the owner-code routes |
| 401 | `invalid_code` | redeem |
| 403 | `operator_only` | operator routes, from a scoped device |
| 403 | `global_token_only`, `not_this_machine` | owner-code mint, status, cancel |
| 403 | `pairing_unavailable` | create (off, or no API token), owner-code routes (no API token) |
| 404 | `unknown_request` | poll, cancel, approve, deny |
| 404 | `unknown_code` | owner-code status, cancel |
| 409 | `not_pending` (with `status`) | approve, deny, owner-code cancel |
| 410 | `expired` | approve, deny |
| 413 | `too_large` | create, approve, redeem |
| 422 | `code_mismatch` | approve |
| 429 | `rate_limited` (with `reason`, `retry_after_seconds`) | create, poll |
| 503 | `identity_unavailable` | create |
