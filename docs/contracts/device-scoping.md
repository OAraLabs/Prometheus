# Device scoping contract, v1

**Contract id:** `device-scoping/1` (section 6, the owner tier, is `device-scoping/1.1`; section 6.5, an owner device on another computer, is `device-scoping/1.3`)
**Scope:** What a *device token* may see and do with chat sessions, over REST and the WebSocket bridge. Code: `web/session_scope.py` (the rule), `config/device_store.py` (the record), `web/server.py` and `web/ws_server.py` (the enforcement points).
**Written for:** whoever builds on the device model next (approve-to-pair): section 2 is what changed in it, and section 2.3 is what to keep true.

---

## 1. The rule

A **device token** (`POST /api/devices`, one row in `devices.db`) sees and manages only the sessions that device **owns**. The operator's **global token** is unrestricted, and so is a daemon deliberately run with auth off (an empty token).

1. A session belongs to the device that brought it into existence: by `POST /api/sessions`, by its first send, upload, `switch_session`, or write to an id that exists **nowhere** (no live session, no durable row), or by forking into a fresh id.
2. A device cannot take an id in a namespace the daemon writes into itself (`telegram:`, `slack:`, `discord:`, `cli:`, `api:`, `coding:`, and the bare id `system`). Otherwise it could send to `telegram:<chat id>` before the operator's chat existed, and own it.
3. A session nobody claimed belongs to the operator: a Telegram chat, a cron run, anything that predates this contract. A device sees none of them.
4. To a device, **"not yours" is the same as "not there"**: a 404 `{"error": "unknown session"}` over REST, an `error` frame with `kind: "not_found"` over the socket. The answer must not tell it whether an id exists.
5. A device can **revoke only itself** (`DELETE /api/devices/{its own id}`). Another id is a 403, before any lookup, so an unknown id and someone else's get the same answer. The global token and an **owner device** (section 6) revoke any device.

Ownership is first-writer-wins, never reassigned and never deleted. Revoking a device leaves its sessions owned by it, readable by the operator and claimable by no one.

## 2. The device-model change

### 2.1 What was added

One table in `devices.db`, one record per owned session:

```
device_sessions (session_id TEXT PRIMARY KEY, device_id TEXT NOT NULL, claimed_at REAL NOT NULL)
```

It is created on the **first claim**, not when the database opens, like `computer_devices`: a daemon that never scopes a device keeps the exact schema it had, and the parity fixtures (which record every table in this file) are unchanged.

Three methods on `DeviceStore`:

| Method | Does |
|---|---|
| `claim_session(session_id, device_id) -> bool` | Take the session if nobody owns it. True if the device owns it afterwards. False for another owner, or a device that is unknown or revoked. It arbitrates between *devices* only; checking that the id exists nowhere else is `SessionAccess`'s job. |
| `session_owner(session_id) -> str \| None` | The owning device id; None = the operator's. Swallows "no such table" and nothing else. |
| `owned_session_ids(device_id) -> set[str]` | Everything one device owns. |

### 2.2 What did not change

Token hashing, verification, `last_seen_at`, the `api_devices` schema, push registration and the computer-use mark. The behavioural changes on the `/api/devices` routes are the revoke rule in section 1 and the owner tier in section 6 (`DeviceIdentity.owner`, `owner`/`owner_source` on the list, `owner: true` on enrolment).

### 2.3 What approve-to-pair must keep true

- **A device id is the owner key.** Mint through `DeviceStore.mint`. A newly paired device owns nothing; it starts from an empty list.
- **A pairing request is not a device.** Until it is approved there is no token and no identity, so there is nothing to scope. A pairing endpoint must not read, list or create sessions.
- **Handing a session to another device** (if pairing wants it) is one `UPDATE device_sessions` in `DeviceStore`, written there and nowhere else, and mind that a turn in flight keeps streaming: its remaining frames follow the new owner. No such method exists yet. Do not delete rows, and do not add a second table that also says who owns a session.
- **A new route keyed by a device id** needs the same self-or-operator check as revoke (`_scope(request)` in `server.py`), or it reintroduces the hole this contract closes.
- **A new route keyed by a session id** must name the path parameter `session_id`. The router-level guard keys on that name; `test_no_route_names_its_session_parameter_anything_but_session_id` fails the build otherwise. A session id in a body or query is the route's own job: `_access.admit` (may create) or `_access.owns` (may not).

## 3. Where it is enforced

| Surface | Enforcement |
|---|---|
| Every REST route with a `{session_id}` path parameter (messages, delete, purge, title, pin, fork, profile, workspace, checkpoints, `/api/lcm/{id}`, computer binding, model) | One router-level dependency (`_guard_session_path`), installed before the first route. A read needs ownership. A write may also claim a brand-new id. |
| `GET /api/sessions` | Filtered to the caller's sessions (durable rows and the live overlay alike). |
| `POST /api/sessions`, fork | The new id is the creator's from that moment. A caller-chosen fork target must be claimable. |
| `POST /api/chat/send`, `POST /api/chat` (as `web:<id>`), `POST /api/stories/{pk}/dispatch`, `POST /api/computer/tasks`, `GET /api/computer/tasks/{id}` | `_access.admit` / `owns` on the id in the body, before anything uses the session. |
| `POST /api/chat/interrupt`, WS `interrupt` | Someone else's session gets the answer a quiet one does (`stopped: false`), and nothing is stopped. |
| `POST /api/search` | A named session must be the caller's. With none named, a device searches each session it owns and merges by rank. |
| `GET /api/events/recent`, `GET /api/activity/recent` | Rows whose payload names a session are returned only for the caller's sessions; session-less (daemon-level) rows are returned to everyone. A device may get fewer than `limit` rows. |
| WS `switch_session`, `send_message`, `chat_upload` | Admitted like the REST equivalents; otherwise the `not_found` error frame and nothing happens. |
| WS `computer_step_thumbnail` (direct-only, never broadcast) | `_thumbnail_eligible`: a device token marked for computer use that declared `computer-thumbnails`, and that **owns** the task's session. With no session filter that is its own sessions; a filter naming another device's session gets nothing. An owner device is the operator and owns every session. |
| WS broadcast | `WebSocketBridge._wants` drops, for a device socket, every frame that names a session the device does not own, **whatever it subscribed to**. The session id is found flat (`payload.session_id`) or nested (`payload.payload.session_id`, the `sentinel_signal` wrapper). A frame naming no session reaches everyone. The operator's socket is the firehose it always was. |

A socket whose identity was never recorded, on a daemon with auth on, owns nothing (fail closed). A socket dies with its token: revoking a device (the route, or an owner mint replacing earlier ones) detaches its open sockets at once and closes them 4401, through a listener on the registry (`DeviceStore.add_revoke_listener`), so no revocation path can forget to.

## 4. What this does not cover

Stated so nobody reads "device scoping" as more than it is. None of these changed.

- **The agent.** A device's turn runs the operator's agent with the operator's tools; `lcm_grep` searches every session. This scopes the API, not the model.
- **Slash commands** (changed, section 7): a scoped device may run only `/help` and the commands that act on its own session; every other known command is refused in the chat. They used to run with the operator's authority, which let a phone type `/gate off`.
- **Approvals** (changed, section 7): a scoped device sees and answers only the tool calls raised in sessions it owns, with `once` or `until-restart`; grants are the operator's. (`until-restart` is process-wide: the grant a device makes for its own call applies to every matching call until the daemon restarts.)
- **Background and coding tasks** (`/api/tasks*`, `/api/code*`): task-id keyed and owned by no device, so a scoped device is denied them (section 7).
- **Stopping a desktop task** stays open to any valid token, by the door's design: a stop can only end a task.
- **`/api/tools/recent`** and **`/api/media?path=`**: denied to a scoped device (section 7); they were open.
- **`/api/media?path=`** serves any cached image to whoever holds the path (`img_<uuid>`); a capability, not a listing.
- **Operator-wide resources**: config, skills, memory, wiki, files, artifacts, cron, providers, MCP servers, stories, projects, profiles. Not sessions, and now denied to a scoped device (section 7); they were open to any valid token.
- **Push notifications** (`push/dispatcher.py`): `turn_completed`, `task_*` and approval pushes go to every registered device, whoever owns the session. A product decision, not a bug fix.
- **`GET /api/devices`**: an operator (the global token or an owner device) lists every enrolled device; a **scoped device lists only itself** (its own row, `is_self: true`), so it can still find its id and push state without learning who else is enrolled. This was open until the pairing work made approved devices common: an approved device could read every other device's name, platform, last-seen time, push status and owner flag. Changed by the pairing stack (#698).
- **An id-existence oracle remains** for ids outside the daemon namespaces: a device that sends to a guessed id learns whether it existed, and takes it if not. Real ids are `<gateway>:<uuid4>`.

## 5. Upgrading

Sessions that existed before this change are **unowned, so operator-only**. A device that used sessions before the upgrade no longer sees them over REST or the socket until the operator gets them back to it: there is no reassignment tool yet. New sessions are scoped from the first message. An **owner device** (section 6) sees them all, which is how the person's own cockpit keeps its history; a device already enrolled with an ordinary token stays scoped until it is paired again as an owner, or its owner promotes it with a one-time code (6.5).

## 6. The owner tier (`device-scoping/1.1`)

The person's **own** device is not a guest. Same-Mac pairing used to hand Beacon the daemon's API token (the master key); a device approved from Telegram or by another device is scoped by sections 1 to 5, and would leave the person's own Beacon with an empty session list. The owner tier is the third thing: a token of its own, revocable alone, that is operator-equivalent where the person needs it to be.

### 6.1 What an owner device is

- **Marker:** a row in `owner_devices (device_id PRIMARY KEY, marked_at, marked_by)`, created on the first owner mint (so `api_devices` and the parity goldens keep their schema), written in the same transaction as the device row. `marked_by` is the route that issued it. `DeviceIdentity.owner` is read from it at every authentication, never from anything the caller sends. `DeviceIdentity.is_operator` = global token or owner device; pairing-approval's "approver" test is exactly that predicate.
- **Two mint paths, and the tier is not a parameter on either.** `DeviceStore.mint(name, platform)` makes an ordinary scoped device (an approved device gets this). `DeviceStore.mint_owner(name, platform, by=, replaces=)` makes an owner device. `api_token.issue_owner_credential(config, *, devices, ...)` is the only caller on the pairing path; it needs the registry and has no fallback to the global token. `DeviceStore.mint_owner_in_transaction(name, platform, by=)` is the same owner mint inside a transaction its caller owns, for the one-time owner code (6.5), whose spending and the device it buys commit together.
- **Who can mint one:** **same-Mac pairing** (Will, 2026-10-08): `POST /api/pair/local` with the file secret, or — for a fresh install, which cannot mint during setup — the global token's `POST /api/devices` with `{"owner": true}` **from this Mac's loopback address**, as the second step of the same pairing. From any other address that request is a 403. **And, for another computer, a one-time code printed on the daemon's machine** (Will, 2026-10-11; 6.5). Approved devices (Telegram, or another device) always get `DeviceStore.mint`: another computer, Jennifer's Mac approved from yours, sees only its own conversations. Never a request that merely asks, never the API token alone, and never a device: an owner device cannot enrol another.

### 6.2 What "operator-equivalent" covers — and what it does not

| Operator-equivalent | Still the global token's alone |
|---|---|
| Session scoping: sees, reads, drives and searches every session, including the operator's Telegram and CLI chats, and receives every frame (`scope_for` returns the operator scope for it). | `POST /api/devices`: a stolen owner device must not be able to enrol an attacker device without the physical-code step. |
| Device management: lists and revokes any device. | Defining an MCP server (`POST/PATCH/DELETE /api/mcp/servers`): it spawns a process as the daemon user. |
| The approver tier for pairing decisions and `pairing_*` frames (`is_operator`). | Anything else that tests `is_global`. |

Approvals of tool calls are still open to every valid token (section 4): "approver" here is a tier pairing-approval can test, not a restriction this change adds.

### 6.3 Two setups, one rule for re-pairing

- **A running daemon** (an existing install, or after setup): `POST /api/pair/local` returns the owner device's own token. The response keeps `token`, `api_base_port` and `ws_port` and adds `revoked_previous`, an integer.
- **Setup mode** (a fresh install): setup mutations authenticate with the global token only, and setup mode creates no `~/.prometheus` state (`devices.db` lives there), so `POST /api/setup/pair` still returns the global token (`issue_setup_credential`, named for what it is). After `POST /api/setup/complete` the client trades it with `POST /api/devices` and `{"owner": true, "name": ...}` and keeps only the owner token.

Minting an owner device **for this Mac** replaces the earlier owner credentials of this Mac. "This Mac" is a source: `same-mac-pairing` (`/api/pair/local`) and `same-mac-mint` (`POST /api/devices` with `owner: true`, loopback peer only). The revocations and the new row are one transaction (a crash leaves neither zero owners nor two), go through the registry's normal revoke (so the old token answers 401 and its open socket is closed 4401), and never touch the new device, an ordinary device, or an owner device of any other source (`DeviceStore.mint_owner` replaces only the sources it is told to). A listener that fails cannot undo the mint. The loopback test is the peer address plus the absence of a forwarded header, so an HTTP proxy on this machine is caught, but a raw TCP relay (`tailscale serve --tcp`, `ssh -L`, socat) makes a remote mint look local: the remote holder of the API token then gets an owner device, and this Mac's own owner credential is revoked (`Prometheus --pair` restores it). A known gap (6.5).

### 6.4 What this does not fix

The global token stays valid and stays in `~/.config/prometheus/env`, where the bash tool can read it, and macOS has no shell floor. This change stops a paired **device** from holding the master key. It does not stop a **model** from reading it; that needs the restrictive default permission mode, which is separate work.

### 6.5 An owner device on another computer (`device-scoping/1.3`)

Since #744 only a device token approves a worker's request (a model can read the API token), and owner devices came only from this Mac, so Beacon on another computer, connected with the API token, could approve no worker. The way up for it:

- **`oara pair --owner`, run on the daemon's machine,** writes the SHA-256 of a one-time code into the daemon's `devices.db` (table `owner_codes`, created on first use) and prints the code: 12 characters from `23456789ABCDEFGHJKMNPQRSTVWXYZ` in three groups of four (about 2^59), good for 10 minutes and one use. There is one live code at a time; issuing replaces the one before. It uses no token and no network. **No route issues a code** (`tests/test_owner_code_promotion.py` scans the source: only the CLI calls `issue_owner_code`), so nothing that only reaches the daemon over the network can make one, whatever bearer it holds. Being able to write `devices.db` is already being able to mint any device, so this widens nothing. The command prints the code only to a terminal, so asking the agent to run it does not put the code in a transcript (an accident guard; a determined process is the shell floors' job).
- **`POST /api/pair/owner`** trades the code for an owner device (`marked_by = "owner-code"`). It is behind the bearer gate and takes any valid token, scoped devices included (it is in `SCOPED_ALLOWED`: a Beacon that joined by approve-to-pair holds a scoped token). The body is `{"code", "device_name", "platform", "public_key"}`; the token comes back **sealed** to `public_key`, with the new `device_id` as the request id of the pairing seal (`config/pair_seal.py`), because an owner token can approve a worker and the API token cannot, so it must not cross a plaintext link in clear.
- **Guessing.** Every problem with the body is refused (400) before the code is looked at, and spends nothing. A source, or a token, that has guessed wrong 3 times in a minute is told to wait (429) before the code is looked at, so one device rotating addresses cannot burn the owner's code. Every wrong guess, from anyone, counts against the live code, and the hundredth burns it. Spending the code and minting the device are one `BEGIN IMMEDIATE` transaction, so two redemptions have one winner (the CLI and the daemon are two processes on the file).
- **It replaces nothing**, and same-Mac re-pairing never revokes it (`owner-code` is not a same-Mac source). The person revokes it like any device. The client keeps the new token and drops the API token.
- **Open mode** (no API token): 403 `pairing_unavailable`, and the CLI refuses to issue.
- **Still true, and not changed here (found by the independent review, 2026-10-11).** No route trades the API token alone for an owner device, but the API token is still more than a route:
  - it can define a stdio MCP server (`POST /api/mcp/servers`, on while `mcp.rest_management` is), which runs any command as the daemon's user outside the shell floor, and such a command can write `devices.db` or run `oara pair --owner`;
  - through a raw TCP relay on the daemon's machine (`tailscale serve --tcp`, `ssh -L`, socat), which adds no forwarded header, a remote request looks local to `POST /api/devices` with `owner: true` (6.3).
  Both predate this and both are the master key's power; closing them changes contracts Beacon and the installer rely on. Likewise a model whose shell can write `devices.db`, or read the API token and reach loopback, could already mint an owner device; the shell floors are what stop that.

## 7. Route access: a scoped device is default-deny (`device-scoping/1.2`)

Sections 1 to 6 scoped what a device sees in the SESSION routes and left every other route open to any valid token. Once approve-to-pair made scoped tokens common (a phone on the home network asks, the owner taps Approve), that was a hole an independent review verified: an approved phone could approve any session's tool call with a persistent grant, create and run cron jobs (which run through `/bin/bash` with the daemon's environment), write provider keys, and type `/gate off`.

**The rule.** The bearer middleware answers `403 operator_only` to a scoped token on any route that is not in `web/route_access.py`'s allowlist. Default-deny: a route added tomorrow, an unknown path, a method nobody listed, and `/v1/*` are refused without anyone remembering to refuse them. The code is 403, not 401, because the token is valid.

**What a scoped device may use** (`SCOPED_ALLOWED`): hello; its own device (list itself, sign itself out, its push and live-activity registration, the live activity for a session it owns); chat (`/api/chat`, `/send`, `/interrupt`); its own sessions (list, create, delete, messages, title, pin, fork, `events/recent`, `activity/recent`, search); approvals, below; and trading a one-time owner code for an owner device (`POST /api/pair/owner`, 6.5). Everything else is the operator's: the global token or an owner device.

**Approvals.** A scoped caller sees and answers only the requests raised in a session it owns: the queue stamps each request with the run's session (`engine/tool_context.RUN_SESSION`). The id must be exactly a pending id (no `all`, no scope smuggled through the id), the scope is `once` or `until-restart` (`SCOPED_APPROVE_SCOPES`), and the same filter decides which `approval_pending` / `approval_resolved` frames its socket is sent. A request that is not its own is a 404, like a session that is not its own. The wire shapes did not change.

**Two things are wider on purpose**, both the computer-use door's: the desktop-task stop is open to every valid token (a stop can only end a task), and a device an operator MARKED for computer use is a person, so it additionally reaches the computer routes and may answer a desktop prompt (`MARKED_DEVICE_ALLOWED`). An unmarked device gets neither.

**Slash commands.** A scoped device's chat may run `SCOPED_SLASH_COMMANDS` (`/help` and the commands on its own session) and nothing else; any other known command is refused in the chat and does not run.

**The ratchet.** `tests/test_route_access.py` classifies every route the app registers exactly once (public, scoped-allowed, marked-device or operator-only), pins the allowlists literally, and runs the table through the real middleware. A new route fails it until someone says which.

**Still true.** A scoped device's turn runs the operator's agent with the operator's tools (section 4): the gate and its own `once` approvals are what stand between a phone and a command. That is the product's choice for an approved device.

