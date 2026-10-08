# Device scoping contract, v1

**Contract id:** `device-scoping/1` (section 6, the owner tier, is `device-scoping/1.1`)
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
| WS broadcast | `WebSocketBridge._wants` drops, for a device socket, every frame that names a session the device does not own, **whatever it subscribed to**. The session id is found flat (`payload.session_id`) or nested (`payload.payload.session_id`, the `sentinel_signal` wrapper). A frame naming no session reaches everyone. The operator's socket is the firehose it always was. |

A socket whose identity was never recorded, on a daemon with auth on, owns nothing (fail closed). A socket dies with its token: revoking a device (the route, or an owner mint replacing earlier ones) detaches its open sockets at once and closes them 4401, through a listener on the registry (`DeviceStore.add_revoke_listener`), so no revocation path can forget to.

## 4. What this does not cover

Stated so nobody reads "device scoping" as more than it is. None of these changed.

- **The agent.** A device's turn runs the operator's agent with the operator's tools; `lcm_grep` searches every session. This scopes the API, not the model.
- **Slash commands.** A command typed into a device's own session runs with the operator's authority: `/events` lists recent signals from every session, `/memory`, `/wiki`, `/grants`, `/revoke` and `/gate` read or change daemon-wide state. Their replies reach only the sockets of that session's owner, but what they read is not scoped.
- **Approvals** (`/api/approvals*`). Ordinary prompts carry no session id, so there is nothing to scope by. Approving another session's prompt, and `grants`, are open to every valid token. Needs the session id plumbed into the queue first.
- **Background and coding tasks** (`/api/tasks*`, `/api/code*`) are task-id keyed and owned by no device.
- **Stopping a desktop task** stays open to any valid token, by the door's design: a stop can only end a task.
- **`/api/tools/recent`** returns tool inputs from every session and its rows carry no session id.
- **`/api/media?path=`** serves any cached image to whoever holds the path (`img_<uuid>`); a capability, not a listing.
- **Operator-wide resources**: config, skills, memory, wiki, files, artifacts, cron, providers. Not sessions.
- **Push notifications** (`push/dispatcher.py`): `turn_completed`, `task_*` and approval pushes go to every registered device, whoever owns the session. A product decision, not a bug fix.
- **`GET /api/devices`** still lists every enrolled device (name, platform, last seen, push status).
- **An id-existence oracle remains** for ids outside the daemon namespaces: a device that sends to a guessed id learns whether it existed, and takes it if not. Real ids are `<gateway>:<uuid4>`.

## 5. Upgrading

Sessions that existed before this change are **unowned, so operator-only**. A device that used sessions before the upgrade no longer sees them over REST or the socket until the operator gets them back to it: there is no reassignment tool yet. New sessions are scoped from the first message. An **owner device** (section 6) sees them all, which is how the person's own cockpit keeps its history; a device already enrolled with an ordinary token stays scoped until it is paired again as an owner.

## 6. The owner tier (`device-scoping/1.1`)

The person's **own** device is not a guest. Same-Mac pairing used to hand Beacon the daemon's API token (the master key); a device approved from Telegram or by another device is scoped by sections 1 to 5, and would leave the person's own Beacon with an empty session list. The owner tier is the third thing: a token of its own, revocable alone, that is operator-equivalent where the person needs it to be.

### 6.1 What an owner device is

- **Marker:** a row in `owner_devices (device_id PRIMARY KEY, marked_at, marked_by)`, created on the first owner mint (so `api_devices` and the parity goldens keep their schema), written in the same transaction as the device row. `marked_by` is the route that issued it. `DeviceIdentity.owner` is read from it at every authentication, never from anything the caller sends. `DeviceIdentity.is_operator` = global token or owner device; pairing-approval's "approver" test is exactly that predicate.
- **Two mint paths, and the tier is not a parameter on either.** `DeviceStore.mint(name, platform)` makes an ordinary scoped device (an approved device gets this). `DeviceStore.mint_owner(name, platform, by=, replaces=)` makes an owner device. `api_token.issue_owner_credential(config, *, devices, ...)` is the only caller on the pairing path; it needs the registry and has no fallback to the global token.
- **Who can mint one:** **only same-Mac pairing** (Will, 2026-10-08): `POST /api/pair/local` with the file secret, or — for a fresh install, which cannot mint during setup — the global token's `POST /api/devices` with `{"owner": true}` **from this Mac's loopback address**, as the second step of the same pairing. From any other address that request is a 403. Approved devices (Telegram, or another device) always get `DeviceStore.mint`: another computer, Jennifer's Mac approved from yours, sees only its own conversations. Never a request that merely asks, and never a device: an owner device cannot enrol another.

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

Minting an owner device **for this Mac** replaces the earlier owner credentials of this Mac. "This Mac" is a source: `same-mac-pairing` (`/api/pair/local`) and `same-mac-mint` (`POST /api/devices` with `owner: true`, loopback peer only). The revocations and the new row are one transaction (a crash leaves neither zero owners nor two), go through the registry's normal revoke (so the old token answers 401 and its open socket is closed 4401), and never touch the new device, an ordinary device, or an owner device of any other source (`DeviceStore.mint_owner` replaces only the sources it is told to). A listener that fails cannot undo the mint. The loopback test is the peer address, so a reverse proxy on this machine makes a remote mint look local; the only cost is that it replaces this Mac's own owner credential, and `Prometheus --pair` restores it.

### 6.4 What this does not fix

The global token stays valid and stays in `~/.config/prometheus/env`, where the bash tool can read it, and macOS has no shell floor. This change stops a paired **device** from holding the master key. It does not stop a **model** from reading it; that needs the restrictive default permission mode, which is separate work.

